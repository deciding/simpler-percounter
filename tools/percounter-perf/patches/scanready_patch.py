#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""How far a lane would have to look forward to find a position whose task is ready.

Measurement only: the draw still takes what it takes, and nothing about the
schedule changes. Applies on top of `counter_patch.py count` and
`window_patch.py`.

AIV lanes spend 78-93% of their life blocked on a claimed position whose task is
not ready, against 43% on the decode module that wins by 38%. The repair that
addresses that directly is to claim a ready position instead of the next one --
scan forward a bounded window, take the first that is ready, and fall back to the
lowest position when none is, which keeps `percounter_claim_range`'s deadlock
argument intact because some lane is still committed to the position that
argument names.

Whether that is worth building depends on a number nobody has: at the moment a
lane draws, how far ahead is the nearest ready task? This patch answers it before
the implementation exists. At each draw it steps over up to `WINDOW` **task runs**
from the ticket and records how many it passed before one was ready, as a
histogram per core type:

    EXPR type=T n0 n1 n2 n3 n4_7 n8_15 n16_31 miss draws

**Task runs, not positions, and the first version of this probe got that wrong.**
Counting positions, the histogram came out bimodal -- 23-27% at offset zero,
63-75% finding nothing within 32, and almost nothing between. A task's blocks are
contiguous, so a 256-block task fills any 32-position window on its own: the probe
was measuring how wide the task at the draw point is, not how far away ready work
is. Stepping by run makes each GM read a different task, and the step is O(1)
because the entry carries both `block_idx` and `block_num`.

A distribution concentrated at `n0`/`n1` means ready work is nearly always a task
or two away and a small window buys most of the 93%. A distribution piled on
`miss` means a lane genuinely has nothing to reach, and the repair's ceiling is
low -- which is the cheap way to find out that this design is not worth writing.

The scan costs a resolve and a fanin read per task examined, on the draw path, so
**this build's timings are not comparable to anything**. Read the histogram, not
the wall clock. Readiness also moves while the scan runs, so an offset is what one
lane saw at one moment, not a property of the list.

usage: scanready_patch.py <checkout> [window]
"""

import sys
from pathlib import Path

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
WINDOW = int(sys.argv[2]) if len(sys.argv) == 3 else 32


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *sr;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    "    uint64_t sr_slots[9] = {};\n    cfg.sr = sr_slots;\n",
)

# The scan sits before the draw, reading the same ticket value
# `percounter_draw_width` reads, so it sees the draw point the lane is about to
# take. It claims nothing and changes no state.
DRAW = (
    "                if (size != 0) {\n"
    "                    ++cfg.exp[0];\n"
    "                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, batch);\n"
)
SCAN = (
    "                if (size != 0) {\n"
    "                    ++cfg.exp[0];\n"
    "                    {\n"
    f"                        const uint64_t window = {WINDOW};\n"
    "                        const uint64_t from = scheduler_gm_query(cfg.tickets[range_group].next);\n"
    "                        uint64_t at = window;\n"
    "                        uint64_t p = from;\n"
    "                        for (uint64_t k = 0; k < window && p < size; ++k) {\n"
    "                            const uint64_t entry = scheduler_gm_query(\n"
    "                                cfg.order[percounter_group_position(p, range_group, cfg.ticket_group_count)]\n"
    "                            );\n"
    "                            const uint64_t blocks = static_cast<uint64_t>(percounter_order_block_num(entry));\n"
    "                            if (blocks == 0) break;\n"
    "                            PercounterTask probe{};\n"
    "                            if (!percounter_resolve(cfg, percounter_order_task_id(entry), &probe)) break;\n"
    "                            if (percounter_fanin_ready(cfg, probe)) { at = k; break; }\n"
    "                            // A task's blocks are contiguous, so the rest of this one is\n"
    "                            // as unready as its head: step over the whole run. A\n"
    "                            // sync-start task holds one arming entry instead.\n"
    "                            const int32_t idx = percounter_order_block_idx(entry);\n"
    "                            const uint64_t step =\n"
    "                                percounter_order_is_arming(entry) || static_cast<uint64_t>(idx) >= blocks\n"
    "                                    ? uint64_t{1}\n"
    "                                    : blocks - static_cast<uint64_t>(idx);\n"
    "                            p += step;\n"
    "                        }\n"
    "                        const uint64_t bucket = at < 4 ? at : (at < 8 ? 4 : (at < 16 ? 5 : (at < 32 ? 6 : 7)));\n"
    "                        cfg.sr[bucket] += 1;\n"
    "                        cfg.sr[8] += 1;\n"
    "                    }\n"
    "                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, batch);\n"
)
sub(E, DRAW, SCAN)

EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
SR_FLUSH = (
    "    if (sr_slots[8] != 0) {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    "        for (int i = 0; i < 9; ++i) {\n"
    "            if (sr_slots[i] != 0) scheduler_gm_fetch_add(g[i + 1], sr_slots[i]);\n"
    "        }\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + SR_FLUSH)

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
    "                cache_invalidate_range(g, 128);\n"
    "                if (g[9] != 0) {\n"
    "                    LOG_WARN(\n"
    "                        \"EXPR type=%d n0=%\" PRIu64 \" n1=%\" PRIu64 \" n2=%\" PRIu64 \" n3=%\" PRIu64\n"
    "                        \" n4_7=%\" PRIu64 \" n8_15=%\" PRIu64 \" n16_31=%\" PRIu64 \" miss=%\" PRIu64\n"
    "                        \" draws=%\" PRIu64,\n"
    "                        t, g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]\n"
    "                    );\n"
    "                }\n"
    "                for (int i = 1; i < 16; ++i)\n"
    "                    g[i] = 0;\n"
    "                cache_flush_range(g, 128);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched scanready window=%d" % WINDOW)
