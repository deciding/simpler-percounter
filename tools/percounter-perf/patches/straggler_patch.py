#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The worst-held block of one task: how long, by which lane, and what that lane ran meanwhile.

Applies on top of `counter_patch.py count` and `window_patch.py`.

`blocktrace_patch.py` records every block, which costs four words each and so fits
eight blocks in the spare ticket lines. That is enough when the task is small, and
useless when it is not: on `prefill_attn_swa` the task whose span grows from 555 us
under the AICPU scheduler to 4,090 us has 64 blocks, and its first eight are all
normal -- held 60-107 us, one kernel each, which is their own.

This keeps one record instead, the block with the longest claim-to-start gap, and
packs it into a single word so a max-CAS carries the whole record consistently:

    hold_ns << 32 | block << 24 | kernels << 16 | lane

`kernels` is what the lane ran between claiming the block and running it, its own
kernel included. One means the lane sat idle on the position and the hold is a
dependency wait. Two means it ran one other kernel first -- the shape that costs
`prefill_attn_c2a_reuse` 450 us, where the other kernel is a 453 us MIX block.
Ten would mean a lane cycling through rendezvous while holding the position,
which is the only way a single hold reaches milliseconds.

`hold_ns` saturates at 2^32 ns, about 4.3 s; `lane` is
`cluster_index * 8 + cluster_lane + 1`, so zero means no record.

usage: straggler_patch.py <checkout> [task_id]
"""

import sys
from pathlib import Path

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
TASK = int(sys.argv[2]) if len(sys.argv) == 3 else 32


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

# sg[0] packed worst record, [1] claim stamp of the block this lane holds,
# [2] that block's index + 1, [3] cfg.kx[0] at claim.
sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *sg;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    "    uint64_t sg_slots[4] = {};\n    cfg.sg = sg_slots;\n",
)

CLAIM = (
    "                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {\n"
    "                        claimed = true;\n"
)
PROBE = (
    "                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {\n"
    "                        {\n"
    "                            const uint64_t now = get_sys_cnt_aicore();\n"
    "                            for (uint64_t q = range_first; q < range_end; ++q) {\n"
    "                                const uint64_t entry = scheduler_gm_query(\n"
    "                                    cfg.order[percounter_group_position(q, range_group, cfg.ticket_group_count)]\n"
    "                                );\n"
    f"                                if (percounter_order_task_id(entry) != {TASK}) continue;\n"
    "                                cfg.sg[1] = now;\n"
    "                                cfg.sg[2] = static_cast<uint64_t>(percounter_order_block_idx(entry)) + 1;\n"
    "                                cfg.sg[3] = cfg.kx[0];\n"
    "                            }\n"
    "                        }\n"
    "                        claimed = true;\n"
)
sub(E, CLAIM, PROBE)

KTIME = (
    "    {\n"
    "        const uint64_t e = get_sys_cnt_aicore();\n"
    "        cfg.kx[0] += 1;\n"
    "        cfg.kx[1] += e - *kernel_start;\n"
    "        if (cfg.kx[3] == 0) cfg.kx[3] = *kernel_start;\n"
    "        cfg.kx[4] = e;\n"
    "    }\n"
)
sub(
    E,
    KTIME,
    KTIME[:-6]
    + (
        f"        if (task.local == {TASK} && cfg.sg[2] != 0) {{\n"
        "            const uint64_t hold = *kernel_start - cfg.sg[1];\n"
        "            const uint64_t kernels = cfg.kx[0] - cfg.sg[3];\n"
        "            const uint64_t packed = ((hold & 0xffffffffULL) << 32) |\n"
        "                                    (((cfg.sg[2] - 1) & 0xffULL) << 24) | ((kernels & 0xffULL) << 16) |\n"
        "                                    ((static_cast<uint64_t>(cfg.cluster_index) * 8 +\n"
        "                                      static_cast<uint64_t>(cfg.cluster_lane) + 1) & 0xffffULL);\n"
        "            if (packed > cfg.sg[0]) cfg.sg[0] = packed;\n"
        "            cfg.sg[2] = 0;\n"
        "        }\n"
        "    }\n"
    ),
)

EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
SG_FLUSH = (
    "    if (sg_slots[0] != 0) {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    "        for (uint64_t old = scheduler_gm_query(g[1]); sg_slots[0] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(g[1], old, sg_slots[0]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + SG_FLUSH)

A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
ANCHOR = (
    "            for (int i = 0; i < 7; ++i)\n"
    "                k[i] = 0;\n"
    "            cache_flush_range(k - 1, 64);\n"
)
READ = (
    "            {\n"
    "                uint64_t *g = reinterpret_cast<uint64_t *>(\n"
    "                    reinterpret_cast<uint8_t *>(control) + control->tickets_offset +\n"
    "                    t * simpler::hbg::percounter::PERCOUNTER_TICKET_CAPACITY * 64 + 8 * 64\n"
    "                );\n"
    "                cache_invalidate_range(g, 64);\n"
    "                if (g[1] != 0) {\n"
    "                    LOG_WARN(\n"
    f"                        \"EXPX type=%d task={TASK} worst_hold=%\" PRIu64 \" block=%\" PRIu64\n"
    "                        \" kernels=%\" PRIu64 \" lane=%\" PRIu64,\n"
    "                        t, g[1] >> 32, (g[1] >> 24) & 0xff, (g[1] >> 16) & 0xff, g[1] & 0xffff\n"
    "                    );\n"
    "                }\n"
    "                g[1] = 0;\n"
    "                cache_flush_range(g, 64);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched straggler task=%d" % TASK)
