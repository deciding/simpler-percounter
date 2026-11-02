#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""How often a percounter lane draws ahead, split by whether the task is wider than the machine.

Applies on top of `counter_patch.py count` and `window_patch.py`, to a pristine
percounter (no `percounter_draw_width`). It answers one question: at the moment a
lane claims a range, how many positions does it take, and does the task those
positions belong to have more blocks than there are lanes of that core type?

The lane loop's own comment names the failure this looks for -- a range withholds
its positions from every other lane, and a lane that drew ahead into a wide
fan-out left the wave running on 24 of 32 AIC lanes. The guard against it reads
the kernel the lane has already run, so it cannot fire on the first wide task a
lane meets, and a family whose kernels are bimodal defeats it: a lane arriving off
a 1.4 us kernel asks for the full batch no matter how wide the task ahead is.

Four counts and two sums per core type, written to ticket line 8, which no lane
draws from (a core type has lanes / PERCOUNTER_TICKET_GROUP_LANES live groups, at
most 8 of the 16 `PERCOUNTER_TICKET_CAPACITY` lines):

  w1 .. w4   draws on a task with more blocks than lanes, by granted width 1 / >1
  n1 .. n4   the same for a task no wider than the lane count
  wpos npos  positions granted in each case, so mean width is pos / draws

`blocks > lanes` is the line the span measurements put the break at: tasks at or
below the lane count packed to 94-103% of their ideal, tasks of 256 blocks to
23-34%. Reading the order entry costs one GM load per draw, which is a draw-path
cost this build adds and a clean one does not -- take no latency figures from it.

usage: drawahead_patch.py <checkout>
"""

import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

# --- the lane's own accumulator --------------------------------------------
sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *dw;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    "    uint64_t dw_counts[6] = {};\n    cfg.dw = dw_counts;\n",
)

# --- the probe, at the point a range becomes this lane's --------------------
# Pristine draws `batch` positions; the claim clamps that against the share, so
# `range_end - range_first` is what the lane actually got.
CLAIM = (
    "                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {\n"
    "                        claimed = true;\n"
)
PROBE = (
    "                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {\n"
    "                        {\n"
    "                            const uint64_t entry = scheduler_gm_query(\n"
    "                                cfg.order[percounter_group_position(range_first, range_group, cfg.ticket_group_count)]\n"
    "                            );\n"
    "                            const uint64_t blocks = static_cast<uint64_t>(percounter_order_block_num(entry));\n"
    "                            const uint64_t lanes =\n"
    "                                static_cast<uint64_t>(cfg.ticket_group_count) * PERCOUNTER_TICKET_GROUP_LANES;\n"
    "                            const uint64_t got = range_end - range_first;\n"
    "                            const uint64_t wide = blocks > lanes ? 1 : 0;\n"
    "                            cfg.dw[wide * 2 + (got > 1 ? 1 : 0)] += 1;\n"
    "                            cfg.dw[4 + wide] += got;\n"
    "                        }\n"
    "                        claimed = true;\n"
)
sub(E, CLAIM, PROBE)

# --- flush at lane exit, into ticket line 8 --------------------------------
EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
DW_FLUSH = (
    "    {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    "        for (int i = 0; i < 6; ++i) {\n"
    "            if (dw_counts[i] != 0) scheduler_gm_fetch_add(g[i + 1], dw_counts[i]);\n"
    "        }\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + DW_FLUSH)

# --- readout, beside the existing lane-counter one -------------------------
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
    "                if (g[1] != 0 || g[2] != 0 || g[3] != 0 || g[4] != 0) {\n"
    "                    LOG_WARN(\n"
    "                        \"EXPD type=%d n1=%\" PRIu64 \" n4=%\" PRIu64 \" w1=%\" PRIu64 \" w4=%\" PRIu64\n"
    "                        \" npos=%\" PRIu64 \" wpos=%\" PRIu64,\n"
    "                        t, g[1], g[2], g[3], g[4], g[5], g[6]\n"
    "                    );\n"
    "                }\n"
    "                for (int i = 1; i < 8; ++i)\n"
    "                    g[i] = 0;\n"
    "                cache_flush_range(g, 64);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched drawahead")
