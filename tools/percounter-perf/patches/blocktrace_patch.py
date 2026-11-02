#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One task's blocks, each with the time it was claimed, the time it ran, and by which lane.

Applies on top of `counter_patch.py count` and `window_patch.py`.

Every probe so far aggregates per task, and per task the question is already
answered: `prefill_attn_c2a_reuse` task 15 has four blocks and nine microseconds
of work, three blocks run at 118,610 and the fourth at 471,779 -- the instant
task 7's 453 us MIX blocks finish. Per-task numbers cannot say which of two
things that is, and the two call for opposite fixes:

  claimed early, ran late   a lane took the position at t=3 us and did not come
                            back to it for 350 us. The lane is the problem --
                            find what held it.
  claimed late, ran at once the ticket never reached the position, so no lane
                            was committed to it and none was free to take it.
                            The list is the problem.

So this records, per block of one task, `claim` (the draw that handed the
position out), `start` and `end` (the kernel), and the lane that ran it. The gap
`start - claim` is the whole question.

`EXP_TASK` selects the task by its graph-local id and `EXP_BLOCKS` bounds how
many of its blocks fit; the store is the eight spare ticket lines of one core
type, four words per block, which no lane draws from. A lane accumulates its own
blocks and flushes at exit, so a block untouched by a lane stays zero and the
per-lane flushes merge. Blocks past `EXP_BLOCKS` are not recorded.

usage: blocktrace_patch.py <checkout> [task_id]
"""

import sys
from pathlib import Path

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
TASK = int(sys.argv[2]) if len(sys.argv) == 3 else 15
BLOCKS = 8
FIELDS = 8


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *pb;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    f"    uint64_t pb_slots[{BLOCKS} * {FIELDS}] = {{}};\n    cfg.pb = pb_slots;\n",
)

# --- the claim: stamp every position of the target task this range holds ----
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
    "                                const int32_t b = percounter_order_block_idx(entry);\n"
    f"                                if (b < 0 || b >= {BLOCKS}) continue;\n"
    f"                                uint64_t *c = &cfg.pb[static_cast<uint64_t>(b) * {FIELDS}];\n"
    "                                c[0] = now;\n"
    "                                c[4] = cfg.kx[0];\n"
    "                                c[5] = cfg.kx[1];\n"
    "                            }\n"
    "                        }\n"
    "                        claimed = true;\n"
)
sub(E, CLAIM, PROBE)

# --- the kernel: start, end and the lane that ran it ------------------------
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
        f"        if (task.local == {TASK} && block_idx >= 0 && block_idx < {BLOCKS}) {{\n"
        f"            uint64_t *s = &cfg.pb[static_cast<uint64_t>(block_idx) * {FIELDS}];\n"
        "            s[1] = *kernel_start;\n"
        "            s[2] = e;\n"
        "            s[3] = static_cast<uint64_t>(cfg.cluster_index) * 8 + static_cast<uint64_t>(cfg.cluster_lane) + 1;\n"
        "            s[6] = cfg.kx[0];\n"
        "            s[7] = cfg.kx[1];\n"
        "        }\n"
        "    }\n"
    ),
)

# --- flush, into ticket lines 8..15 of this lane's core type ----------------
EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
PB_FLUSH = (
    "    {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    f"        for (int i = 0; i < {BLOCKS} * {FIELDS}; ++i) {{\n"
    "            if (pb_slots[i] != 0) scheduler_gm_fetch_add(g[i], pb_slots[i]);\n"
    "        }\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + PB_FLUSH)

# --- readout ---------------------------------------------------------------
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
    f"                cache_invalidate_range(g, {BLOCKS} * {FIELDS} * 8);\n"
    f"                for (int b = 0; b < {BLOCKS}; ++b) {{\n"
    f"                    uint64_t *s = g + b * {FIELDS};\n"
    "                    if (s[0] == 0 && s[1] == 0) continue;\n"
    "                    LOG_WARN(\n"
    f"                        \"EXPB type=%d task={TASK} block=%d claim=%\" PRIu64 \" start=%\" PRIu64\n"
    "                        \" end=%\" PRIu64 \" lane=%\" PRIu64 \" kc=%\" PRIu64 \"/%\" PRIu64\n"
    "                        \" kb=%\" PRIu64 \"/%\" PRIu64,\n"
    "                        t, b, s[0], s[1], s[2], s[3], s[4], s[6], s[5], s[7]\n"
    "                    );\n"
    "                    for (int f = 0; f < 8; ++f)\n"
    "                        s[f] = 0;\n"
    "                }\n"
    f"                cache_flush_range(g, {BLOCKS} * {FIELDS} * 8);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched blocktrace task=%d" % TASK)
