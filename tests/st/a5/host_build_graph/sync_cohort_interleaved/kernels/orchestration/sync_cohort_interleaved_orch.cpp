/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * Two kinds of sync-start cohort in flight at once.
 *
 * `sync_cohort` next door broadcasts one `mixed` flag to every group, so a run
 * there is homogeneous: all-MIX or all-cube. The shape neither it nor any
 * captured workload builds is a **cube-only cohort and a MIX cohort open
 * together**, and that is the one this graph exists to put on the device.
 *
 * The graph is three kinds of task, no edges, every one a root, emitted as
 * `cohort_pairs` repeats of `A, fillers, K, fillers`:
 *
 *   A       a cube-only cohort of `aic_blocks` blocks. Its serving pool is the
 *           cube lanes, and the lane that takes a block runs it itself.
 *   filler  ordinary single-block cube tasks, each spinning `filler_spin`
 *           times, `filler_count` of them spread evenly between the cohorts.
 *           They carry no attribute and no barrier.
 *   K       a MIX cohort of `mix_blocks` blocks. Its serving pool is the
 *           claimer lanes alone: the lane that takes a block arms its own
 *           cluster rather than running it.
 *
 * **The filler is the load-bearing part, not padding.** A cohort leaves the
 * open slot when it is fully *claimed*, not when its blocks have *started*, so
 * K opens while holders of A's blocks may still be inside fillers. A holder
 * whose cluster is armed for K in that window must serve the block it owes
 * before joining K's rendezvous, or K waits on a lane inside A's barrier that
 * waits on it. With no filler every holder reaches its checkpoint at once and
 * serves before K can arm, so the window never opens.
 *
 * Both kernels barrier across every lane of their own task, so a cohort one
 * lane short never finishes and the failure is a hang rather than a wrong
 * value. Read it against `_nosync`, which builds the identical graph with
 * neither attribute nor barrier and must always pass.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;
constexpr int64_t kMaxBlocks = 32;
constexpr int64_t kFillerMax = 256;
constexpr int64_t kPairsMax = 64;
constexpr int64_t kTaskMax = 1024;
// Every task owns one region of this many slots, whatever its own shape, so a
// cube-only task and a MIX task can be indexed the same way from both sides.
constexpr int64_t kSlotsPerTask = kMaxBlocks * 3;

struct TaskSpec {
    int64_t base;
    int64_t block_num;
    int64_t lanes;
    bool barrier;
    bool cohort;
    int64_t spin;
};

void submit_one(const simpler::hbg::Tensor &marks, const simpler::hbg::Tensor &barrier, const TaskSpec &spec) {
    CoreTaskArgs task;
    // `add_no_dep`, not `add_inout`: each task writes its own slots of these two
    // tensors, and letting the overlap map infer dependencies would chain every
    // task write-after-write and leave nothing concurrent to test.
    task.add_no_dep(marks);
    task.add_no_dep(barrier);
    task.add_scalar(spec.base);
    task.add_scalar(spec.block_num);
    task.add_scalar(spec.lanes);
    task.add_scalar(static_cast<int64_t>(spec.barrier ? 1 : 0));
    task.add_scalar(spec.spin);
    task.launch_spec.set_block_num(static_cast<int16_t>(spec.block_num));
    if (spec.cohort) task.launch_spec.set_require_sync_start(true);

    if (spec.lanes > 1) {
        MixedKernels kernels;
        kernels.aic_kernel_id = kAicFuncId;
        kernels.aiv0_kernel_id = kAivFuncId;
        kernels.aiv1_kernel_id = kAivFuncId;
        (void)rt_submit_task(kernels, task);
        return;
    }
    (void)rt_submit_aic_task(kAicFuncId, task);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 10,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &marks = args.tensor(0).ref();
    const simpler::hbg::Tensor &barrier = args.tensor(1).ref();
    const int64_t aic_blocks = args.scalar<int64_t>(0);
    const int64_t mix_blocks = args.scalar<int64_t>(1);
    const int64_t filler_count = args.scalar<int64_t>(2);
    const int64_t filler_spin = args.scalar<int64_t>(3);
    const bool with_sync = args.scalar<int64_t>(4) != 0;
    const int64_t cohort_pairs = args.scalar<int64_t>(5);
    // Which kind of task carries the attribute. Task count, block counts and
    // kernels are the same in all three settings; what moves with the attribute
    // is the barrier below, since only a task whose blocks are guaranteed to
    // start together can wait on its peers without hanging on its own.
    const int64_t cohort_kind = args.scalar<int64_t>(6);
    const bool cube_cohort = with_sync && cohort_kind != 2;
    const bool mix_cohort = with_sync && cohort_kind != 1;
    // Whether a cohort's kernel waits on its peers. Off gives the control the
    // attribution needs: both kinds of cohort in flight with no lane ever held
    // inside a barrier, which separates the scheduler's cohort bookkeeping from
    // the kernel-side wait. The reverse -- a barrier without the attribute --
    // is refused below: its blocks need not be co-resident, so it would hang
    // for a reason that has nothing to do with cohorts.
    const bool with_barrier = args.scalar<int64_t>(7) != 0;
    // Fillers are spread evenly between the cohorts rather than massed in
    // front of them: the window is one task wide and opens at a checkpoint, so
    // what matters is that a lane has work queued wherever a cohort arms.
    const int64_t stride = cohort_pairs > 0 ? filler_count / (2 * cohort_pairs) : filler_count;

    // Two blocks is the narrowest real cohort: one would be released by its own
    // arrival, and the orchestrator sets the attribute only above one block.
    if (aic_blocks < 2 || aic_blocks > kMaxBlocks) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "aic_blocks must be in [2, %ld], got %ld", static_cast<long>(kMaxBlocks),
            static_cast<long>(aic_blocks)
        );
        return;
    }
    if (mix_blocks < 2 || mix_blocks > kMaxBlocks) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "mix_blocks must be in [2, %ld], got %ld", static_cast<long>(kMaxBlocks),
            static_cast<long>(mix_blocks)
        );
        return;
    }
    if (filler_count < 0 || filler_count > kFillerMax) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "filler_count must be in [0, %ld], got %ld", static_cast<long>(kFillerMax),
            static_cast<long>(filler_count)
        );
        return;
    }
    if (filler_spin < 0) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "filler_spin must be >= 0, got %ld", static_cast<long>(filler_spin)
        );
        return;
    }

    if (with_barrier && !with_sync) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "%s",
            "with_barrier needs with_sync: a barrier whose blocks are not guaranteed to start together hangs on its own"
        );
        return;
    }

    if (cohort_pairs < 1 || cohort_pairs > kPairsMax) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "cohort_pairs must be in [1, %ld], got %ld", static_cast<long>(kPairsMax),
            static_cast<long>(cohort_pairs)
        );
        return;
    }
    // filler_count is a total over the graph, so a count below one per gap
    // divides to none at all. Refuse it: the window this test needs exists only
    // where a lane already has work queued, and a case that asked for fillers
    // and got none tests the opposite shape under a name that says otherwise.
    if (filler_count > 0 && stride == 0) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "filler_count %ld over %ld gaps leaves none per gap; use at least %ld",
            static_cast<long>(filler_count), static_cast<long>(2 * cohort_pairs), static_cast<long>(2 * cohort_pairs)
        );
        return;
    }
    const int64_t task_count = cohort_pairs * (2 + 2 * stride);
    if (task_count > kTaskMax) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "the shape asks for %ld tasks, over the %ld cap", static_cast<long>(task_count),
            static_cast<long>(kTaskMax)
        );
        return;
    }

    // **Cube and MIX cohorts alternate down the graph, fillers between them.**
    // One cohort is claimable device-wide, so with a single pair of them the
    // second only ever arms into a device that already finished the first. The
    // state under test needs a MIX cohort arming while a cube cohort is claimed
    // and not yet started, and that is a coincidence in time: give it many
    // chances rather than one.
    int64_t next = 0;
    // Only a cohort barriers, and only when asked: a filler is one block, so it
    // has no peers to wait for.
    const auto emit = [&](int64_t block_num, int64_t lanes, bool cohort, int64_t spin) {
        submit_one(
            marks, barrier, TaskSpec{next * kSlotsPerTask, block_num, lanes, cohort && with_barrier, cohort, spin}
        );
        ++next;
    };
    for (int64_t pair = 0; pair < cohort_pairs; ++pair) {
        emit(aic_blocks, 1, cube_cohort, 0);
        for (int64_t filler = 0; filler < stride; ++filler)
            emit(1, 1, false, filler_spin);
        emit(mix_blocks, 3, mix_cohort, 0);
        for (int64_t filler = 0; filler < stride; ++filler)
            emit(1, 1, false, filler_spin);
    }
}

}  // extern "C"
