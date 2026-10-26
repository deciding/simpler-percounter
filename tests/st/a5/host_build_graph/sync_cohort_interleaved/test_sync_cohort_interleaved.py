#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A cube-only sync-start cohort and a MIX one, in flight at the same time.

`sync_cohort` next door is homogeneous by construction -- one `mixed` flag for
the whole graph -- so it never has two kinds of cohort open together. This
graph does, with ordinary filler tasks between them so that a lane holding a
block of one cohort is often still busy when the next cohort starts arming
clusters.

That window is where percounter can wedge if a lane joins the next cohort's MIX
rendezvous before serving the block it already owes: the rendezvous needs a
lane that is inside the first cohort's barrier, waiting for that very block.
percounter serves a released block before it looks at the MIX line, and these
cases are its regression test.

**A failure here is a hang, not a wrong value**: both kernels barrier across
every lane of their task. The scheduler watchdog does not see it, since the
wait is inside a kernel, so a hang lasts until the op timeout. Run each case in
a process of its own, and read it against its `_nosync` twin -- the identical
graph with neither attribute nor barrier, which must always pass.

Every case is manual and runs with `SIMPLER_HBG_SCHEDULER_TYPE=percounter`,
since the default selection would not take percounter. The variable is set for
this class's `test_run` only, so other modules in the same session keep their
own scheduler, and an explicit setting in the environment wins.
"""

import ctypes
import os

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

# The host reads this on every bind, in the process that holds the L2 worker,
# so it must be set before the cases run and only while they run: pytest
# imports every module it collects, and a value set at import would move every
# other a5 HBG case in the session onto percounter.
SCHEDULER_ENV, SCHEDULER = "SIMPLER_HBG_SCHEDULER_TYPE", "percounter"

FLOATS_PER_CACHE_LINE = 16
INTS_PER_CACHE_LINE = 16
# Mirrors kSlotsPerTask in the orchestration: every task owns a region of this
# size whatever its own shape, so both sides index them the same way.
SLOTS_PER_TASK = 32 * 3

# a5sim runs 8 clusters against the part's 32, and a cohort wider than the pool
# is refused at bring-up, so the sim-sized cases stay at 8.
SIM_CLUSTERS = 8


# Which tasks carry the sync-start attribute. The graph is identical under all
# three, so a difference between them is a difference in the cohorts alone.
KIND_BOTH, KIND_CUBE_ONLY, KIND_MIX_ONLY = 0, 1, 2


def _case(
    name,
    aic_blocks,
    mix_blocks,
    filler_count,
    filler_spin,
    pairs=1,
    with_sync=True,
    kind=KIND_BOTH,
    platforms=None,
    with_barrier=None,
):
    return {
        "name": name,
        "platforms": platforms or ["a5sim", "a5"],
        "params": {
            "aic_blocks": aic_blocks,
            "mix_blocks": mix_blocks,
            "filler_count": filler_count,
            "filler_spin": filler_spin,
            "cohort_pairs": pairs,
            "with_sync": with_sync,
            "cohort_kind": kind,
            # A cohort barriers unless the case says otherwise; without the
            # attribute there is nothing to barrier, and the orchestration
            # refuses the combination.
            "with_barrier": with_sync if with_barrier is None else with_barrier,
        },
        "manual": True,
    }


@scene_test(level=2, runtime="host_build_graph")
class TestSyncCohortInterleavedHostBuildGraphA5(SceneTestCase):
    RTOL = 0
    ATOL = 0

    def test_run(self, st_platform, st_worker, request, monkeypatch):
        if SCHEDULER_ENV not in os.environ:
            monkeypatch.setenv(SCHEDULER_ENV, SCHEDULER)
        super().test_run(st_platform, st_worker, request)

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/sync_cohort_interleaved_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_interleaved.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_interleaved.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        # Two cube blocks, two MIX blocks, a few fillers between them, on a5sim's
        # 8 clusters. `filler_count` is a total over the graph and one pair has
        # two gaps, so `f2` is the smallest that fills both.
        _case("min_f2_spin4k", 2, 2, 2, 4000),
        _case("min_f2_spin0", 2, 2, 2, 0),
        _case("min_f2_spin64k", 2, 2, 2, 64000),
        _case("min_f8_spin4k", 2, 2, 8, 4000),
        _case("min_f32_spin4k", 2, 2, 32, 4000),
        _case("min_f2_spin4k_nosync", 2, 2, 2, 4000, with_sync=False),
        _case("min_f32_spin4k_nosync", 2, 2, 32, 4000, with_sync=False),
        # Two full-width cohorts, and a full-width cube cohort beside a narrow
        # MIX one.
        _case("full_f32_spin4k", SIM_CLUSTERS, SIM_CLUSTERS, 32, 4000),
        _case("full_f32_spin16k", SIM_CLUSTERS, SIM_CLUSTERS, 32, 16000),
        _case("wide_narrowmix_spin4k", SIM_CLUSTERS, 2, 32, 4000),
        # The same shapes against the part's 32 clusters.
        _case("a5_min_f2_spin4k", 2, 2, 2, 4000, platforms=["a5"]),
        _case("a5_min_f32_spin4k", 2, 2, 32, 4000, platforms=["a5"]),
        _case("a5_min_f2_spin4k_nosync", 2, 2, 2, 4000, with_sync=False, platforms=["a5"]),
        _case("a5_full_f32_spin4k", 32, 32, 32, 4000, platforms=["a5"]),
        # Fillers well past one per lane. With few tasks the lane that took a
        # block has nothing else to run and serves at once, so the window never
        # opens; it opens only where every lane has work queued behind the one
        # it is running.
        _case("a5_busy_f256_spin16k", 2, 2, 256, 16000, platforms=["a5"]),
        _case("a5_busy_f256_spin64k", 2, 2, 256, 64000, platforms=["a5"]),
        _case("a5_busy_f256_full_spin16k", 32, 32, 256, 16000, platforms=["a5"]),
        _case("a5_busy_f256_spin16k_nosync", 2, 2, 256, 16000, with_sync=False, platforms=["a5"]),
        _case("busy_f256_spin16k", 2, 2, 256, 16000),
        _case("busy_f256_full_spin16k", SIM_CLUSTERS, SIM_CLUSTERS, 256, 16000),
        # Cohort pairs alternating down the graph, which gives the window many
        # chances to coincide with a MIX cohort arming. `a5_alt16_f256_spin16k`
        # is the regression case: it hung in 9 of 30 processes (5 rounds each)
        # before percounter served released blocks first.
        _case("alt16_f256_spin16k", 2, 2, 256, 16000, pairs=16),
        _case("alt16_f256_spin64k", 2, 2, 256, 64000, pairs=16),
        _case("alt16_f256_spin16k_nosync", 2, 2, 256, 16000, pairs=16, with_sync=False),
        _case("a5_alt16_f256_spin16k", 2, 2, 256, 16000, pairs=16, platforms=["a5"]),
        _case("a5_alt32_f256_spin16k", 2, 2, 256, 16000, pairs=32, platforms=["a5"]),
        _case("a5_alt32_f256_spin64k", 2, 2, 256, 64000, pairs=32, platforms=["a5"]),
        _case("a5_alt16_f256_wide_spin16k", 8, 8, 256, 16000, pairs=16, platforms=["a5"]),
        _case("a5_alt32_f256_spin16k_nosync", 2, 2, 256, 16000, pairs=32, with_sync=False, platforms=["a5"]),
        # The attribute on one kind of cohort only, and on both with no kernel
        # waiting on its peers: the same graph, separating the two kinds of
        # cohort from their barriers.
        _case("a5_alt16_cubeonly_spin16k", 2, 2, 256, 16000, pairs=16, kind=KIND_CUBE_ONLY, platforms=["a5"]),
        _case("a5_alt16_mixonly_spin16k", 2, 2, 256, 16000, pairs=16, kind=KIND_MIX_ONLY, platforms=["a5"]),
        _case("a5_alt16_nobarrier_spin16k", 2, 2, 256, 16000, pairs=16, platforms=["a5"], with_barrier=False),
        _case("alt16_nobarrier_spin16k", 2, 2, 256, 16000, pairs=16, with_barrier=False),
    ]

    @staticmethod
    def _regions(params):
        """(base, slots written, did it barrier) per task, in submission order.

        Mirrors the orchestration rather than inferring from the capture: a MIX
        cohort that silently ran as a lone cube task would otherwise pass by
        leaving the other two lanes at whatever this predicted.
        """
        sync, kind = params["with_sync"], params["cohort_kind"]
        barrier = params["with_barrier"]
        cube_cohort = sync and kind != KIND_MIX_ONLY
        mix_cohort = sync and kind != KIND_CUBE_ONLY
        pairs = params["cohort_pairs"]
        stride = params["filler_count"] // (2 * pairs) if pairs else params["filler_count"]
        regions, index = [], 0

        def emit(written, barriered):
            nonlocal index
            regions.append((index * SLOTS_PER_TASK, written, barriered))
            index += 1

        for _ in range(pairs):
            emit(params["aic_blocks"], cube_cohort and barrier)
            for _ in range(stride):
                emit(1, False)
            emit(params["mix_blocks"] * 3, mix_cohort and barrier)
            for _ in range(stride):
                emit(1, False)
        return regions

    @classmethod
    def _slot_count(cls, params):
        return len(cls._regions(params)) * SLOTS_PER_TASK

    def generate_args(self, params):
        slots = self._slot_count(params)
        return TaskArgsBuilder(
            TensorArg("marks", torch.zeros(slots * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("barrier", torch.zeros(slots * INTS_PER_CACHE_LINE, dtype=torch.int32)),
            Scalar("aic_blocks", ctypes.c_int64(params["aic_blocks"])),
            Scalar("mix_blocks", ctypes.c_int64(params["mix_blocks"])),
            Scalar("filler_count", ctypes.c_int64(params["filler_count"])),
            Scalar("filler_spin", ctypes.c_int64(params["filler_spin"])),
            Scalar("with_sync", ctypes.c_int64(1 if params["with_sync"] else 0)),
            Scalar("cohort_pairs", ctypes.c_int64(params["cohort_pairs"])),
            Scalar("cohort_kind", ctypes.c_int64(params["cohort_kind"])),
            Scalar("with_barrier", ctypes.c_int64(1 if params["with_barrier"] else 0)),
        )

    def compute_golden(self, args, params):
        for base, written, barriered in self._regions(params):
            for offset in range(written):
                slot = base + offset
                # The value encodes the position, so a lane that ran the wrong
                # block or the wrong subtask writes a value the golden does not
                # have rather than one that happens to match.
                args.marks[slot * FLOATS_PER_CACHE_LINE] = float(slot + 1)
                if barriered:
                    args.barrier[slot * INTS_PER_CACHE_LINE] = 1


if __name__ == "__main__":
    os.environ.setdefault(SCHEDULER_ENV, SCHEDULER)
    SceneTestCase.run_module(__name__)
