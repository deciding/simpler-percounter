#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bound a percounter draw by what the tasks at the draw point are worth.

Applies to an experiment checkout that already carries `counter_patch.py count`
and `window_patch.py`. Two forms, so the short-list guard can be measured rather
than assumed:

  guard   the rule as committed -- the per-task bound runs only once the group's
          remaining share drops below PERCOUNTER_TICKET_GROUP_LANES * want, which
          on a list of a few hundred positions per group is the last 32 of them.
          lm_head has 92 positions in all, so the guard never holds there and the
          bound always applies; a 2138-position graph spends 88% of its draws
          taking the early return, paying one ticket load for nothing.
  noguard the per-task bound on every draw.

The bound itself is each task's own per-lane share, `ceil(block_num / lanes)`: a
256-block task on 64 lanes is worth four positions to one lane and keeps the full
batch, while a 32-block task is worth one. That is the shape the span
measurements call for -- forcing every draw to one position collapses task 19 of
`prefill_attn_c2a_reuse` from 424,553 ns to 9,874 ns and task 31 from 38,496 to
4,589, while costing task 10 (512 blocks) 11% and task 21 (256 blocks) 31%.

usage: drawwidth_patch.py <checkout> guard|noguard
"""

import sys
from pathlib import Path

if len(sys.argv) != 3 or sys.argv[2] not in ("guard", "noguard"):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

GUARD = (
    "    const uint64_t index = scheduler_gm_query(cfg.tickets[group].next);\n"
    "    if (index < group_size && group_size - index >= PERCOUNTER_TICKET_GROUP_LANES * want) return want;\n"
)
NOGUARD = "    const uint64_t index = scheduler_gm_query(cfg.tickets[group].next);\n"

FN = (
    "inline __aicore__ uint64_t\n"
    "percounter_draw_width(const LaneConfig &cfg, uint32_t group, uint64_t group_size, uint64_t want) {\n"
    "    if (want <= 1) return want;\n"
    "    const uint64_t lanes = static_cast<uint64_t>(cfg.ticket_group_count) * PERCOUNTER_TICKET_GROUP_LANES;\n"
    + (GUARD if mode == "guard" else NOGUARD)
    + "    uint64_t taken = 0;\n"
    "    int64_t run_task = -1;\n"
    "    uint64_t run = 0;\n"
    "    uint64_t allowed = 1;\n"
    "    while (taken < want && index + taken < group_size) {\n"
    "        const uint64_t entry =\n"
    "            scheduler_gm_query(cfg.order[percounter_group_position(index + taken, group, cfg.ticket_group_count)]);\n"
    "        const uint64_t blocks = static_cast<uint64_t>(percounter_order_block_num(entry));\n"
    "        if (blocks == 0) break;\n"
    "        const int64_t id = percounter_order_task_id(entry);\n"
    "        if (id != run_task) {\n"
    "            run_task = id;\n"
    "            run = 0;\n"
    "            allowed = lanes == 0 ? blocks : (blocks + lanes - 1) / lanes;\n"
    "            if (allowed == 0) allowed = 1;\n"
    "        }\n"
    "        if (run == allowed) break;\n"
    "        ++run;\n"
    "        ++taken;\n"
    "    }\n"
    "    return taken == 0 ? 1 : taken;\n"
    "}\n"
    "\n"
)

# The function needs `percounter_order_block_num` and `percounter_group_position`,
# both declared by the time `percounter_run_task` is defined.
ANCHOR = "/**\n * Run `kernel_id` as subtask `subtask_slot` of one block of `task_id`."
sub(E, ANCHOR, FN + ANCHOR)

DRAW = (
    "                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, batch);\n"
    "                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {\n"
)
sub(
    E,
    DRAW,
    "                    const uint64_t width = percounter_draw_width(cfg, range_group, size, batch);\n"
    "                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, width);\n"
    "                    if (percounter_claim_range(base, width, size, &range_first, &range_end)) {\n",
)

print("patched drawwidth %s" % mode)
