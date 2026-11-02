#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Order the percounter lists by a supplied ranking instead of by b-level.

Host-side. `SIMPLER_EXP_ORDER_RANK=<id,id,id,...>` replaces the level sort with
that ranking; a task the ranking omits sorts after the ranked ones, among
themselves by b-level as before.

What it is for: b-level ranks a task by how much work hangs off it, which says
nothing about when it can start. A lane commits to a position before that
position's task is ready and cannot hand it back, so a list in b-level order
hands the pool to whichever task is most critical even when that task has to wait
-- and the ready task behind it cannot be reached. The AICPU scheduler has no
such problem because it dispatches on readiness, so **the order it dispatches in
is an empirical ready order**, and `tasktime3_patch.py` already records it per
task for a legacy run. Feeding that back is how to find out what ordering alone
is worth, before building a cost model that could derive it.

Safety: the ranking must keep every producer ahead of its consumer inside a list,
which is what `percounter_claim_range`'s deadlock argument needs. A legacy
dispatch order has that property by construction -- a consumer is not dispatched
until its producers complete -- but the ranking is an input here, so each list is
checked after sorting and a list that fails falls back to the b-level order with
an error logged. Read the log before reading a time.

usage: orderrank_patch.py <checkout>
"""

import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
S = "src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h"
p = root / S
s = p.read_text()

ANCHOR = """        std::stable_sort(eligible[l].begin(), eligible[l].end(), by_blevel);
"""
if s.count(ANCHOR) != 1:
    raise SystemExit(f"{S}: expected 1 sort site, found {s.count(ANCHOR)}")

RANKED = ANCHOR + """        if (const char *spec = std::getenv("SIMPLER_EXP_ORDER_RANK")) {
            std::vector<int32_t> rank(static_cast<size_t>(task_count), -1);
            int32_t next_rank = 0;
            for (const char *c = spec; *c != '\\0';) {
                while (*c == ',' || *c == ' ') ++c;
                if (*c == '\\0') break;
                const int64_t id = std::atoll(c);
                while (*c != '\\0' && *c != ',') ++c;
                if (id >= 0 && static_cast<uint64_t>(id) < task_count) rank[static_cast<size_t>(id)] = next_rank++;
            }
            const std::vector<int64_t> before = eligible[l];
            std::stable_sort(
                eligible[l].begin(), eligible[l].end(),
                [&rank, &by_blevel](int64_t a, int64_t b) {
                    const int32_t ra = rank[static_cast<size_t>(a)];
                    const int32_t rb = rank[static_cast<size_t>(b)];
                    if (ra >= 0 && rb >= 0) return ra < rb;
                    if (ra >= 0) return true;
                    if (rb >= 0) return false;
                    return by_blevel(a, b);
                }
            );
            std::vector<int32_t> seat(static_cast<size_t>(task_count), -1);
            for (size_t k = 0; k < eligible[l].size(); ++k)
                seat[static_cast<size_t>(eligible[l][k])] = static_cast<int32_t>(k);
            bool topological = true;
            for (int64_t id : eligible[l]) {
                const TaskInput &consumer = tasks[static_cast<size_t>(id)];
                for (int32_t e = 0; e < consumer.fanin_count; ++e) {
                    const size_t producer = static_cast<size_t>(consumer.fanin_ids[e]);
                    if (seat[producer] < 0) continue;
                    if (seat[producer] >= seat[static_cast<size_t>(id)]) {
                        LOG_ERROR(
                            "EXP rank list=%u rejected: producer %zu sits after consumer %lld", l, producer,
                            static_cast<long long>(id)
                        );
                        topological = false;
                    }
                }
            }
            if (!topological) {
                eligible[l] = before;
            } else {
                std::string shown;
                for (int64_t id : eligible[l]) {
                    if (!shown.empty()) shown += ",";
                    shown += std::to_string(static_cast<long long>(id));
                }
                LOG_WARN("EXP rank list=%u order=%s", l, shown.c_str());
            }
        }
"""
s = s.replace(ANCHOR, RANKED)

for header in ("#include <algorithm>", "#include <string>", "#include <cstdlib>", "#include <vector>"):
    if header not in s:
        first = s.index("#include ")
        s = s[:first] + header + "\n" + s[first:]

p.write_text(s)
print("patched orderrank")
