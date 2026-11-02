#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""How long a cluster goes without claiming MIX work, because its claimer is in a kernel.

Applies on top of `counter_patch.py count` and `window_patch.py`.

`percounter_mix_tick` returns immediately unless the calling lane is its cluster's
`PERCOUNTER_MIX_CLAIMER_LANE`, so one lane per cluster both claims the cluster's
MIX work and runs ordinary blocks like any other. While that lane is inside a
kernel it is not in the loop, and the cluster claims nothing. A graph whose AIC
work is MIX and which also has a task of 453 us blocks therefore has a mechanism
that stalls MIX claiming for the length of that kernel -- and it is reached by
none of `PERCOUNTER_CLAIM_BATCH`, `percounter_draw_width` or
`PERCOUNTER_TICKET_GROUP_LANES`, which is the invariance the span measurements
keep running into.

Per core type, into ticket line 8 (no lane draws from it -- a core type has at
most 8 live groups of the 16 `PERCOUNTER_TICKET_CAPACITY` lines):

  ticks     calls that passed the claimer test
  maxgap    longest interval between two consecutive such calls, ns
  sumgap    their total, so the mean is sumgap / (ticks - 1)
  over100   intervals longer than 100 us

A maxgap near the module's longest kernel confirms the mechanism; a maxgap in the
single-digit microseconds rules it out.

usage: mixgap_patch.py <checkout>
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

sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *mg;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    "    uint64_t mg_counts[5] = {};\n    cfg.mg = mg_counts;\n",
)

# mg[0] ticks, [1] maxgap, [2] sumgap, [3] over100us, [4] last tick stamp.
TICK = (
    "inline __aicore__ void percounter_mix_tick(const LaneConfig &cfg, PercounterLaneState *state) {\n"
    "    if (cfg.mix_order_count == 0 || cfg.cluster_lane != PERCOUNTER_MIX_CLAIMER_LANE) return;\n"
)
sub(
    E,
    TICK,
    TICK
    + "    {\n"
    "        const uint64_t now = get_sys_cnt_aicore();\n"
    "        if (cfg.mg[4] != 0) {\n"
    "            const uint64_t gap = now - cfg.mg[4];\n"
    "            if (gap > cfg.mg[1]) cfg.mg[1] = gap;\n"
    "            cfg.mg[2] += gap;\n"
    "            if (gap > 100000) cfg.mg[3] += 1;\n"
    "        }\n"
    "        cfg.mg[4] = now;\n"
    "        cfg.mg[0] += 1;\n"
    "    }\n",
)

EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
MG_FLUSH = (
    "    if (mg_counts[0] != 0) {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    "        scheduler_gm_fetch_add(g[1], mg_counts[0]);\n"
    "        scheduler_gm_fetch_add(g[3], mg_counts[2]);\n"
    "        scheduler_gm_fetch_add(g[4], mg_counts[3]);\n"
    "        for (uint64_t old = scheduler_gm_query(g[2]); mg_counts[1] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(g[2], old, mg_counts[1]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + MG_FLUSH)

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
    "                        \"EXPM type=%d ticks=%\" PRIu64 \" maxgap=%\" PRIu64 \" sumgap=%\" PRIu64\n"
    "                        \" over100=%\" PRIu64,\n"
    "                        t, g[1], g[2], g[3], g[4]\n"
    "                    );\n"
    "                }\n"
    "                for (int i = 1; i < 8; ++i)\n"
    "                    g[i] = 0;\n"
    "                cache_flush_range(g, 64);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched mixgap")
