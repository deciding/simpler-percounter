#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Set percounter's spin backoff duty cycle, to measure what it costs a chain.

`PERCOUNTER_SPIN_BACKOFF_CYCLES` is how long a lane spins on the cycle counter
between polls once it has missed `PERCOUNTER_SPIN_BACKOFF_AFTER` times at full
rate. It exists because a waiting lane's back-to-back loads otherwise contend
with the producer's publishing atomic for the same line.

The committed 256 was measured down from 1024 on `mixed_chain_1024`, whose waits
are short. A graph whose latency is dominated by long waits -- a deep serial
chain with comm-window notification tasks, where lanes poll hundreds of times
per kernel -- pays that duty cycle for the whole wait, so the figure that helps
one shape may be the wrong end for the other.

The value lives in AICore code, which cannot read an environment variable, so
this rewrites the constant and the build has to follow.

usage: spin_backoff_patch.py <checkout> <cycles>
"""

import pathlib
import re
import sys

if len(sys.argv) != 3:
    raise SystemExit(__doc__.strip())

root = pathlib.Path(sys.argv[1]).resolve()
cycles = int(sys.argv[2])
if cycles <= 0:
    raise SystemExit("cycles must be positive")

rel = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
path = root / rel
text = path.read_text()

pattern = re.compile(r"(inline constexpr uint64_t PERCOUNTER_SPIN_BACKOFF_CYCLES = )(\d+)(;)")
match = pattern.search(text)
if match is None:
    raise SystemExit(f"{rel}: PERCOUNTER_SPIN_BACKOFF_CYCLES declaration not found")

old = match.group(2)
path.write_text(pattern.sub(rf"\g<1>{cycles}\g<3>", text, count=1))
print(f"patched {rel}: PERCOUNTER_SPIN_BACKOFF_CYCLES {old} -> {cycles}")
