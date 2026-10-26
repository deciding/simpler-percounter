#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""sync-start cohorts, with a kernel that really waits for its peers.

A marker kernel cannot test a cohort: a sync-start task whose blocks did not
all start still writes correct values from the blocks that did, and the defect
is that blocks never start. So the kernel here barriers across every lane of
the task, the way the workload that motivates the attribute does -- qwen3's
attention kernel and its `epochBarrierArriveAndWait`.

**A cohort one lane short therefore hangs, and the hang is the test.** The
failure signature is the 45 s op timeout (`HandleTaskTimeout` in the device
log), not a golden mismatch; a reader who sees one should read it as the kernel
doing exactly what it was written to do rather than as a deadlock in the
scheduler. The op timeout ends a hang and fails that case with 507018.

Every `_nosync` case builds the identical graph with the attribute and the
barrier both off. That pair is what gives the suite meaning: the control must
pass under every scheduler, and so must the cohort cases.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
INTS_PER_CACHE_LINE = 16


def _case(name, graph_case, group_count, block_num, with_sync=True, mixed=False, sync_every=2, manual=True):
    case = {
        "name": name,
        "platforms": ["a5sim", "a5"],
        "params": {
            "graph_case": graph_case,
            "group_count": group_count,
            "block_num": block_num,
            "with_sync": with_sync,
            "mixed": mixed,
            "sync_every": sync_every,
        },
    }
    if manual:
        case["manual"] = True
    return case


@scene_test(level=2, runtime="host_build_graph")
class TestSyncCohortHostBuildGraphA5(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/sync_cohort_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_cohort.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_cohort.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT],
            },
        ],
    }

    # Four kinds of row, and the prefix says which:
    #
    #   (bare)      single-slot cohort. The serving pool is the cube lanes, and
    #               the lane that takes a block runs it itself.
    #   mix_*       MIX cohort -- qwen3's shape, and the harder half. The lane
    #               that takes a block does not run it: it arms its own
    #               cluster's rendezvous with it, which is what keeps the
    #               commitment count equal to the block count.
    #   *_nosync    the control: the identical graph with no attribute and no
    #               barrier.
    #   solo_*      one group and nothing else, which is the minimal cohort.
    #
    # `solo_b4` and `mix_solo_b4` are the two the automatic suite runs: the
    # cheapest of each kind.
    CASES = [
        _case("solo_b4", 1, 1, 4, manual=False),
        _case("mix_solo_b4", 1, 1, 4, mixed=True, manual=False),
        _case("solo_b4_nosync", 1, 1, 4, with_sync=False),
        _case("mix_solo_b4_nosync", 1, 1, 4, with_sync=False, mixed=True),
        # **The pair that removes the shared term.** One cohort, nothing else in
        # the graph, so every lane is already idle when it arms: there is no
        # "wait for 24 cores to free" for either scheduler to pay, and the
        # delta against the control is each scheduler's own gather plus a
        # barrier they both run identically. On every other shape the two are
        # confounded.
        _case("solo_b24", 1, 1, 24),
        _case("solo_b24_nosync", 1, 1, 24, with_sync=False),
        _case("mix_solo_b24", 1, 1, 24, mixed=True),
        _case("mix_solo_b24_nosync", 1, 1, 24, with_sync=False, mixed=True),
        # A chain of cohorts: the generation cycle, armed and freed in turn.
        _case("chain_8_b24", 0, 8, 24),
        _case("chain_8_b24_nosync", 0, 8, 24, with_sync=False),
        _case("mix_chain_8_b24", 0, 8, 24, mixed=True),
        _case("mix_chain_8_b24_nosync", 0, 8, 24, with_sync=False, mixed=True),
        # No edges, so every arming entry is ready at once and the cohorts
        # queue behind the one open slot.
        _case("multi_root_8_b24", 1, 8, 24),
        _case("multi_root_8_b24_nosync", 1, 8, 24, with_sync=False),
        _case("mix_multi_root_8_b24", 1, 8, 24, mixed=True),
        _case("mix_multi_root_8_b24_nosync", 1, 8, 24, with_sync=False, mixed=True),
        _case("multi_root_32_b24", 1, 32, 24),
        _case("mix_multi_root_32_b24", 1, 32, 24, mixed=True),
        # A cohort arming into a device that is already busy. The 32 roots
        # release a wide front at once, so the pool a cohort must recruit from
        # is inside kernels rather than at a checkpoint -- which the two shapes
        # above cannot show, because there the device is idle when a cohort
        # arms.
        _case("fanin32_64_b24", 2, 64, 24),
        _case("fanin32_64_b24_nosync", 2, 64, 24, with_sync=False),
        _case("mix_fanin32_64_b24", 2, 64, 24, mixed=True),
        _case("mix_fanin32_64_b24_nosync", 2, 64, 24, with_sync=False, mixed=True),
        # Irregular readiness: cohorts become ready at times no lane can
        # predict, so the arming entries do not arrive in a batch and cohorts
        # queue at a rate neither chain nor multi_root produces.
        _case("random_64_b24", 3, 64, 24),
        _case("random_64_b24_nosync", 3, 64, 24, with_sync=False),
        _case("mix_random_64_b24", 3, 64, 24, mixed=True),
        _case("mix_random_64_b24_nosync", 3, 64, 24, with_sync=False, mixed=True),
        # **The shape every real workload has, and the only one that separates
        # the two costs.** A large ordinary graph carrying a handful of
        # cohorts: 256 groups at `sync_every=64` is four of them. The
        # per-cohort cost is then a rounding error and what remains is the tax
        # percounter's open-cohort check puts on every task of the graph, which the
        # dense cases cannot see.
        _case("sparse_256_b24", 1, 256, 24, sync_every=64),
        _case("sparse_256_b24_nosync", 1, 256, 24, with_sync=False, sync_every=64),
        _case("mix_sparse_256_b24", 1, 256, 24, mixed=True, sync_every=64),
        _case("mix_sparse_256_b24_nosync", 1, 256, 24, with_sync=False, mixed=True, sync_every=64),
    ]

    @staticmethod
    def _lanes(params):
        # A MIX group fills all three subtask slots of each of its blocks; a
        # single-slot one fills the cube alone.
        return 3 if params["mixed"] else 1

    @classmethod
    def _slot_count(cls, params):
        return params["group_count"] * params["block_num"] * cls._lanes(params)

    def generate_args(self, params):
        slots = self._slot_count(params)
        return TaskArgsBuilder(
            TensorArg("marks", torch.zeros(slots * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("barrier", torch.zeros(slots * INTS_PER_CACHE_LINE, dtype=torch.int32)),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("group_count", ctypes.c_int64(params["group_count"])),
            Scalar("block_num", ctypes.c_int64(params["block_num"])),
            Scalar("with_sync", ctypes.c_int64(1 if params["with_sync"] else 0)),
            Scalar("mixed", ctypes.c_int64(1 if params["mixed"] else 0)),
            Scalar("sync_every", ctypes.c_int64(params["sync_every"])),
        )

    def compute_golden(self, args, params):
        # The orchestration's layout is mirrored here rather than inferred: a
        # bug that ran a MIX group as a lone cube task would otherwise pass by
        # leaving the other two lanes at whatever this function predicted.
        lanes = self._lanes(params)
        per_group = params["block_num"] * lanes
        for group in range(params["group_count"]):
            # Only the even groups carry the attribute, so only they barrier --
            # and the barrier slot is the one place the two kinds differ in the
            # output, which is what makes a cohort that silently ran as an
            # ordinary SPMD task visible.
            cohort = params["with_sync"] and group % params["sync_every"] == 0
            for offset in range(per_group):
                slot = group * per_group + offset
                # The value encodes the position, so a lane that ran the wrong
                # block or the wrong subtask writes a value the golden does not
                # have rather than one that happens to match.
                args.marks[slot * FLOATS_PER_CACHE_LINE] = float(slot + 1)
                if cohort:
                    args.barrier[slot * INTS_PER_CACHE_LINE] = 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
