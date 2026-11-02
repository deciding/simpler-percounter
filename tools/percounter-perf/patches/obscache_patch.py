#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Price the per-block whole-cache invalidate on the percounter execution path.

Diagnostic only. `percounter_run_task` and `percounter_run_subtask` each open with
`scheduler_observe_data_cache(cfg.payload)`, which is `dcci(address,
ENTIRE_DATA_CACHE)` plus a `dsb` -- once per block executed. Its cost is a
function of how much dirty data the core holds, not of the block count or the
graph's shape, which is the one profile left that fits a deficit immune to every
draw, claim and ticket-grouping knob: prefill attention carries a far larger
working set than decode, and prefill is where percounter trails.

Two forms, neither of them a candidate fix:

  line  invalidate only the payload's own cache lines. Whether this is even
        correct depends on what the whole-cache form is there to publish -- if it
        covers the kernel's input tensors rather than the payload, the run gives
        wrong results, and `passed=False` is itself the answer to that question.
  none  drop the call. Strictly incorrect; it exists to put an upper bound on
        what the invalidate can be worth.

Read the timing from a run that also reports `passed=True`, or read it as an
upper bound and say so.

usage: obscache_patch.py <checkout> line|none
"""

import sys
from pathlib import Path

if len(sys.argv) != 3 or sys.argv[2] not in ("line", "none"):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]

E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
p = root / E
s = p.read_text()

OLD = "    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));\n"
n = s.count(OLD)
# Two per-block sites; the two setup sites pass `control`, not `cfg.payload`.
if n != 2:
    raise SystemExit(f"{E}: expected 2 per-block observe sites, found {n}")

if mode == "none":
    NEW = ""
else:
    NEW = (
        "    {\n"
        "        __gm__ uint8_t *pl = reinterpret_cast<__gm__ uint8_t *>(cfg.payload);\n"
        "        for (uint64_t off = 0; off < sizeof(DispatchPayload); off += 64) dcci(pl + off, SINGLE_CACHE_LINE);\n"
        "        dsb((mem_dsb_t)0);\n"
        "    }\n"
    )

p.write_text(s.replace(OLD, NEW))
print("patched obscache %s" % mode)
