#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MIX without sync-start, which one cluster can run on its own.

MIX tasks are interleaved with single-core tasks across dependency edges in
four DAG shapes, so a lane blocked on a producer of another kind is regularly
the lane a rendezvous is waiting for. The `_bN` cases add blocks on top: a MIX
task of N blocks is N rendezvous of the same task, running at once on
different clusters.

Each lane writes its own cache line and the value encodes which line it is, so
a lane that ran the wrong subtask of a task writes a value the golden does not
have rather than one that happens to match.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_GROUP = 3


def _case(name, graph_case, group_count, with_mix=True, block_num=1, manual=True):
    case = {
        "name": name,
        "platforms": ["a5sim", "a5"],
        "params": {
            "graph_case": graph_case,
            "group_count": group_count,
            "with_mix": with_mix,
            "block_num": block_num,
        },
    }
    if manual:
        case["manual"] = True
    return case


@scene_test(level=2, runtime="host_build_graph")
class TestMixRendezvousHostBuildGraphA5(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mix_rendezvous_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_mark.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_mark.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    # The same four shapes as multi_core_dag, which is the comparison these are
    # meant to be read against. `chain_24` and `mix_chain_24_b4` are the two the
    # automatic suite runs: the cheapest of each kind.
    #
    # Four kinds of row, and the prefix says which, because `spmd_blocks`
    # carries shapes of the same name that are not MIX at all:
    #
    #   (bare)      MIX at one block. Every rendezvous runs alone.
    #   mix_*_bN    MIX at N blocks -- N rendezvous of one task, on different
    #               clusters at once, with no sync-start.
    #   *_nomix     the control: the identical graph with every group
    #               single-core, so the cost of a rendezvous reads off one
    #               suite and one kernel rather than being inferred across two.
    #   spmdonly_*  the same control at N blocks, which is `spmd_blocks`'
    #               shape reached through this orchestration.
    CASES = [
        _case("chain_24", 0, 24, manual=False),
        _case("mix_chain_24_b4", 0, 24, block_num=4, manual=False),
        _case("chain_1024", 0, 1024),
        _case("fanin32_1024", 1, 1024),
        _case("multi_root_1024", 2, 1024),
        _case("multi_root_4096", 2, 4096),
        _case("random_1024", 3, 1024),
        _case("chain_1024_nomix", 0, 1024, with_mix=False),
        _case("fanin32_1024_nomix", 1, 1024, with_mix=False),
        _case("multi_root_1024_nomix", 2, 1024, with_mix=False),
        _case("multi_root_4096_nomix", 2, 4096, with_mix=False),
        _case("random_1024_nomix", 3, 1024, with_mix=False),
        _case("chain_256", 0, 256),
        _case("chain_256_nomix", 0, 256, with_mix=False),
        _case("mix_chain_256_b4", 0, 256, block_num=4),
        _case("mix_chain_1024_b4", 0, 1024, block_num=4),
        _case("mix_multi_root_256_b4", 2, 256, block_num=4),
        _case("mix_multi_root_32_b24", 2, 32, block_num=24),
        _case("mix_random_256_b4", 3, 256, block_num=4),
        _case("spmdonly_chain_256_b4", 0, 256, with_mix=False, block_num=4),
        _case("spmdonly_chain_1024_b4", 0, 1024, with_mix=False, block_num=4),
        _case("spmdonly_multi_root_256_b4", 2, 256, with_mix=False, block_num=4),
    ]

    @staticmethod
    def _unit_count(params):
        # One "unit" is one (group, block) pair, and each owns three slots.
        return params["group_count"] * params["block_num"]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg(
                "marks",
                torch.zeros(self._unit_count(params) * SLOTS_PER_GROUP * FLOATS_PER_CACHE_LINE, dtype=torch.float32),
            ),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("group_count", ctypes.c_int64(params["group_count"])),
            Scalar("with_mix", ctypes.c_int64(1 if params["with_mix"] else 0)),
            Scalar("block_num", ctypes.c_int64(params["block_num"])),
        )

    def compute_golden(self, args, params):
        # Which slots a group writes is decided by its kind, and the
        # orchestration's layout is mirrored here rather than inferred: a bug
        # that ran a MIX task as a single-core one would otherwise pass by
        # leaving the other two slots at whatever this function predicted.
        block_num = params["block_num"]
        for group in range(params["group_count"]):
            if group % SLOTS_PER_GROUP == 0:
                # MIX fills all three; the control turns the same group into a
                # lone cube task, which fills only the first.
                offsets = (0, 1, 2) if params["with_mix"] else (0,)
            elif group % SLOTS_PER_GROUP == 1:
                offsets = (0,)  # AIC alone
            else:
                offsets = (1,)  # AIV alone, which does not read sub_block_id
            for block in range(block_num):
                unit = group * block_num + block
                for offset in offsets:
                    slot = unit * SLOTS_PER_GROUP + offset
                    args.marks[slot * FLOATS_PER_CACHE_LINE] = float(slot + 1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
