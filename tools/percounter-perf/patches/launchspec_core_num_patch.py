#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Let a host_build_graph LaunchSpec answer to `core_num` as well as `block_num`.

pypto's generated orchestration spells this span `core_num` for a5 and
`block_num` for a2a3, while one header serves both, so an a5 program built from
current pypto codegen fails to compile its orchestration with

    'class LaunchSpec' has no member named 'set_core_num'

Only a pypto-generated program hits it, which is why no in-repo a5 HBG example
does: none of them consumes pypto codegen.

usage: launchspec_core_num_patch.py <checkout>
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()


def sub(rel, old, new, count=1):
    path = root / rel
    text = path.read_text()
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{rel}: expected {count} occurrence(s) of the anchor, found {found}")
    path.write_text(text.replace(old, new))
    print(f"patched {rel}")


sub(
    "src/common/host_build_graph/submit_types.h",
    "    int16_t block_num() const { return block_num_; }\n"
    "    void set_block_num(int16_t n) { block_num_ = n; }\n",
    "    int16_t block_num() const { return block_num_; }\n"
    "    void set_block_num(int16_t n) { block_num_ = n; }\n"
    "\n"
    "    // pypto's generated orchestration spells this span `core_num` for a5 and\n"
    "    // `block_num` for a2a3, and one header serves both, so both names reach the\n"
    "    // same field. a5's tensormap_and_ringbuffer LaunchSpec carries only the\n"
    "    // `core_num` spelling.\n"
    "    int16_t core_num() const { return block_num_; }\n"
    "    void set_core_num(int16_t n) { block_num_ = n; }\n",
)
