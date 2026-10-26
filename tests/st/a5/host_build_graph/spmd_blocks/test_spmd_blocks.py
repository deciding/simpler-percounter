#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD with one slot per task and no sync-start: multiplicity on its own.

Every other SPMD workload in the tree is also MIX or sync-start, so this is the
one graph where `block_num` is the only thing under test.

Each block writes the cache line its own `block_idx` names, and the value
encodes the line, so a block that ran with the wrong index writes a value the
golden does not have rather than one that happens to match.

`join_24` is the separate shape, and the only falsifiable test of the N-way
join: a dependency edge is task to task, so a consumer must not run until every
block of its producer has finished. A runtime that released it on the first
block would still finish the graph with every marker correct -- the remaining
blocks go on to write their own lines -- so nothing but a consumer that counts
what it can see can tell the two apart.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
JOIN_CASE = 4


def _case(name, graph_case, group_count, block_num, manual=True):
    case = {
        "name": name,
        "platforms": ["a5sim", "a5"],
        "params": {"graph_case": graph_case, "group_count": group_count, "block_num": block_num},
    }
    if manual:
        case["manual"] = True
    return case


@scene_test(level=2, runtime="host_build_graph")
class TestSpmdBlocksHostBuildGraphA5(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_blocks_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_block_mark.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_block_mark.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_join_check.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    # `chain_32_b4` and `join_24` are the two the automatic suite runs: the
    # first is the cheapest shape that expands anything, the second is the only
    # one that can fail on a wrong join. The rest are the shapes to compare
    # schedulers on, and `*_b1` are the controls -- the identical graph with
    # nothing to expand, so the cost of expansion reads off one suite.
    CASES = [
        _case("chain_32_b4", 0, 32, 4, manual=False),
        _case("join_24", JOIN_CASE, 1, 24, manual=False),
        _case("chain_256_b1", 0, 256, 1),
        _case("chain_256_b4", 0, 256, 4),
        _case("fanin32_256_b2", 1, 256, 2),
        _case("multi_root_256_b1", 2, 256, 1),
        _case("multi_root_256_b8", 2, 256, 8),
        _case("multi_root_16_b24", 2, 16, 24),
        _case("random_256_b1", 3, 256, 1),
        _case("random_256_b4", 3, 256, 4),
        _case("join_96", JOIN_CASE, 1, 96),
    ]

    @staticmethod
    def _line_count(params):
        # The join shape has one producer of `block_num` lines plus the
        # consumer's count at the line after them.
        if params["graph_case"] == JOIN_CASE:
            return params["block_num"] + 1
        return params["group_count"] * params["block_num"]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg(
                "marks",
                torch.zeros(self._line_count(params) * FLOATS_PER_CACHE_LINE, dtype=torch.float32),
            ),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("group_count", ctypes.c_int64(params["group_count"])),
            Scalar("block_num", ctypes.c_int64(params["block_num"])),
        )

    def compute_golden(self, args, params):
        block_num = params["block_num"]
        if params["graph_case"] == JOIN_CASE:
            for block in range(block_num):
                args.marks[block * FLOATS_PER_CACHE_LINE] = float(block + 1)
            # Every block of the producer, which is the whole point.
            args.marks[block_num * FLOATS_PER_CACHE_LINE] = float(block_num)
            return
        # Slot `group * block_num + block_idx` holds `slot + 1`, so the lines
        # are a dense 1..N regardless of which core type ran which group.
        for slot in range(params["group_count"] * block_num):
            args.marks[slot * FLOATS_PER_CACHE_LINE] = float(slot + 1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
