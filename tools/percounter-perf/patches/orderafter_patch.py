#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Move one task behind another in its list, to price an ordering change before building one.

Host-side. `SIMPLER_EXP_ORDER_AFTER=A:B` places task A immediately after task B
in whichever list holds them, applied after the level sort. Several pairs may be
given, comma separated.

**Only equal-b-level pairs are moved.** `A -> B` forces blevel(A) >= blevel(B) + 1,
so tasks at one level have no edge between them and any order among them is still
topological; a pair at different levels is refused rather than silently breaking
the deadlock argument that `percounter_claim_range` rests on.

What it is for: a lane commits to a position before that position's task is ready
and cannot hand it back, so a not-yet-ready task sitting just ahead of a ready one
takes the lane pool with it. On `prefill_attn_c2a_reuse` the AIV list puts a
64-block task waiting on a 256-block MIX task at positions 361..424 -- one per AIV
lane -- and a 256-block task whose producers are already done at 425.. behind it.
The second one's dependencies are met at 749,848 and it starts at 979,232.

The two orderings that cost nothing to compute both get that pair wrong: by edge
depth the waiting task has t-level 4 against 6, and by block count 271 against
329, because both producers have 256 blocks and only their per-block cost (27.9 us
against 5.5 us) tells them apart. So the question this patch answers is what a
correct order is worth, before paying for the cost model that would find it.

usage: orderafter_patch.py <checkout>
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

MOVE = ANCHOR + """        if (const char *spec = std::getenv("SIMPLER_EXP_ORDER_AFTER")) {
            std::string rest(spec);
            while (!rest.empty()) {
                const size_t comma = rest.find(',');
                std::string pair = rest.substr(0, comma);
                rest = comma == std::string::npos ? std::string() : rest.substr(comma + 1);
                const size_t colon = pair.find(':');
                if (colon == std::string::npos) continue;
                const int64_t mover = std::atoll(pair.substr(0, colon).c_str());
                const int64_t anchor = std::atoll(pair.substr(colon + 1).c_str());
                auto &ids = eligible[l];
                const auto mi = std::find(ids.begin(), ids.end(), mover);
                const auto ai = std::find(ids.begin(), ids.end(), anchor);
                if (mi == ids.end() || ai == ids.end()) continue;
                if (blevel[static_cast<size_t>(mover)] != blevel[static_cast<size_t>(anchor)]) {
                    LOG_ERROR(
                        "EXP order move %lld:%lld refused: levels %d and %d differ",
                        static_cast<long long>(mover), static_cast<long long>(anchor),
                        blevel[static_cast<size_t>(mover)], blevel[static_cast<size_t>(anchor)]
                    );
                    continue;
                }
                ids.erase(mi);
                const auto at = std::find(ids.begin(), ids.end(), anchor);
                ids.insert(at + 1, mover);
                LOG_WARN(
                    "EXP order list=%u moved task=%lld after task=%lld", l,
                    static_cast<long long>(mover), static_cast<long long>(anchor)
                );
            }
        }
"""
s = s.replace(ANCHOR, MOVE)

for header in ("#include <algorithm>", "#include <string>", "#include <cstdlib>"):
    if header not in s:
        first = s.index("#include ")
        s = s[:first] + header + "\n" + s[first:]

p.write_text(s)
print("patched orderafter")
