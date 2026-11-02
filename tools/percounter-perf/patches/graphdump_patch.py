#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Print the graph percounter is about to schedule: per task, its shape and producers.

Host-side only -- no device run, no instrumentation overhead, and the numbers are
the same `TaskPayload` fanin the AICPU scheduler reads, so the dump describes both
schedulers' view of the graph.

One line per task: block count, placement (MIX / AIC / AIV), whether it is
sync-start, and the producer ids. That settles by inspection what per-task device
timestamps can only hint at -- which tasks are independent of the long one, how
many clusters a MIX task occupies, and whether a task that runs late under
percounter had a producer that was still running.

usage: graphdump_patch.py <checkout>
"""

import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()

S = "src/a5/runtime/host_build_graph/host/percounter/percounter_state.h"
p = root / S
s = p.read_text()

OLD = """    for (size_t t = 0; t < task_count; ++t)
        inputs[t].fanin_ids = inputs[t].fanin_count > 0 ? edges.data() + edge_begin[t] : nullptr;
"""
if s.count(OLD) != 1:
    raise SystemExit(f"{S}: expected 1 fanin-wiring site, found {s.count(OLD)}")

NEW = OLD + """    for (size_t t = 0; t < task_count; ++t) {
        const TaskInput &in = inputs[t];
        std::string fin;
        for (int32_t e = 0; e < in.fanin_count; ++e) {
            if (e != 0) fin += ",";
            fin += std::to_string(static_cast<long long>(in.fanin_ids[e]));
        }
        LOG_WARN(
            "EXPG task=%zu blocks=%d mix=%d sync=%d type=%d subtasks=%d fanin=%d [%s]", t,
            static_cast<int>(in.block_num), in.is_mix ? 1 : 0, in.sync_start ? 1 : 0,
            static_cast<int>(in.core_type_index), static_cast<int>(in.subtask_count),
            static_cast<int>(in.fanin_count), fin.c_str()
        );
    }
"""
s = s.replace(OLD, NEW)

if "#include <string>" not in s:
    first = s.index("#include ")
    s = s[:first] + "#include <string>\n" + s[first:]

p.write_text(s)
print("patched graphdump")
