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
 * sync-start cohorts outside a Graph body.
 *
 * `graph_sync_start_early_dispatch` submits its cohort from inside a Graph body;
 * this graph submits the same attribute directly.
 *
 * Every group has the same geometry, and `sync_every` decides which of them
 * carry the attribute:
 *
 *   g % sync_every == 0  sync-start: its `block_num` blocks must be
 *                        co-resident, and its kernel barriers across every lane
 *                        of the task, so a cohort one lane short never
 *                        finishes.
 *   otherwise            the same blocks with no attribute and no barrier. They
 *                        may run one after another on one lane, which is what
 *                        makes them the wrong shape for a barrier and the right
 *                        one beside it.
 *
 * **`sync_every` is the knob that separates the two costs a cohort has.** At 2
 * half the graph is cohorts and the per-cohort cost dominates. At a large value
 * the graph is almost all ordinary work carrying two or three cohorts, which is
 * the shape every real workload has and the only one that can show the other
 * cost: percounter looks at the open cohort at every lane checkpoint, so a graph
 * with any sync task at all pays one shared-line read per lane per task for the
 * whole run, however few cohorts it has.
 *
 * `with_sync` off turns every group into the second kind. That control is what
 * gives the suite meaning: it must pass under every scheduler, and so must the
 * cohort cases.
 *
 *   0  chain       `g - 1 -> g`. One task runnable at a time, so each cohort
 *                  arms and frees in turn -- the generation cycle, repeated.
 *   1  multi_root  no edges at all. Every arming entry is ready at once, so the
 *                  cohorts queue behind the one open slot while the lanes go
 *                  on with ordinary work.
 *   2  fanin32     32 roots, then every later group waits on all of them. The
 *                  roots release a wide front at once, so a cohort arms into a
 *                  device that is already busy rather than into an idle one --
 *                  which is where the pool it must recruit from is in kernels
 *                  rather than at a checkpoint.
 *   3  random      0 to 4 producers per group. Irregular readiness, so cohorts
 *                  become ready at times no lane can predict and the arming
 *                  entries do not arrive in a batch.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;
constexpr int64_t kGroupCapacity = 1024;
// A cohort cannot be wider than the pool it draws from, and the AICPU refuses
// the graph at bring-up when it is. 24 is qwen3's own `attention_core_num`,
// against 32 clusters.
constexpr int64_t kBlockNumMax = 32;
constexpr int32_t kMaxFanin = 4;
constexpr int32_t kFaninRoots = 32;

TaskId submit_group(
    const simpler::hbg::Tensor &marks, const simpler::hbg::Tensor &barrier, int64_t group, bool with_sync, bool mixed,
    int64_t block_num, int64_t sync_every, const TaskId *deps, int32_t dep_count
) {
    const bool cohort = with_sync && group % sync_every == 0;
    CoreTaskArgs task;
    // `add_no_dep`, not `add_inout`: every group writes its own cache lines of
    // these two tensors, so letting the overlap map infer dependencies would
    // chain every group write-after-write and collapse both shapes above into
    // the chain. The structure under test is the explicit one.
    task.add_no_dep(marks);
    task.add_no_dep(barrier);
    task.add_scalar(group);
    // The kernel places its slot from this rather than from get_block_num(), so
    // a block count that did not reach the launch spec shows up as a wrong slot
    // instead of as agreement between two copies of one mistake.
    task.add_scalar(block_num);
    task.add_scalar(static_cast<int64_t>(mixed ? 3 : 1));
    task.add_scalar(static_cast<int64_t>(cohort ? 1 : 0));
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    task.launch_spec.set_block_num(static_cast<int16_t>(block_num));
    if (cohort) task.launch_spec.set_require_sync_start(true);

    if (mixed) {
        MixedKernels kernels;
        kernels.aic_kernel_id = kAicFuncId;
        kernels.aiv0_kernel_id = kAivFuncId;
        kernels.aiv1_kernel_id = kAivFuncId;
        return rt_submit_task(kernels, task).task_id();
    }
    return rt_submit_aic_task(kAicFuncId, task).task_id();
}

struct BuildArgs {
    const simpler::hbg::Tensor &marks;
    const simpler::hbg::Tensor &barrier;
    int64_t group_count;
    bool with_sync;
    bool mixed;
    int64_t block_num;
    int64_t sync_every;
};

TaskId submit(const BuildArgs &a, int64_t group, const TaskId *deps, int32_t dep_count) {
    return submit_group(a.marks, a.barrier, group, a.with_sync, a.mixed, a.block_num, a.sync_every, deps, dep_count);
}

void build_chain(const BuildArgs &a) {
    TaskId previous = submit(a, 0, nullptr, 0);
    for (int64_t group = 1; group < a.group_count; ++group) {
        previous = submit(a, group, &previous, 1);
    }
}

void build_multi_root(const BuildArgs &a) {
    for (int64_t group = 0; group < a.group_count; ++group) {
        (void)submit(a, group, nullptr, 0);
    }
}

void build_fanin32(const BuildArgs &a) {
    TaskId roots[kFaninRoots];
    const int32_t root_count =
        static_cast<int32_t>(a.group_count < kFaninRoots ? a.group_count : static_cast<int64_t>(kFaninRoots));
    for (int32_t group = 0; group < root_count; ++group) {
        roots[group] = submit(a, group, nullptr, 0);
    }
    for (int64_t group = root_count; group < a.group_count; ++group) {
        (void)submit(a, group, roots, root_count);
    }
}

void build_random(const BuildArgs &a) {
    TaskId ids[kGroupCapacity];
    ids[0] = submit(a, 0, nullptr, 0);
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t group = 1; group < a.group_count; ++group) {
        random = random * UINT64_C(6364136223846793005) + 1;
        const int32_t requested = static_cast<int32_t>(random % 5);
        TaskId deps[kMaxFanin];
        int32_t count = 0;
        for (int32_t edge = 0; edge < requested; ++edge) {
            random = random * UINT64_C(6364136223846793005) + 1;
            const int64_t producer = static_cast<int64_t>(random % static_cast<uint64_t>(group));
            bool duplicate = false;
            for (int32_t i = 0; i < count; ++i)
                duplicate = duplicate || deps[i] == ids[producer];
            if (!duplicate) deps[count++] = ids[producer];
        }
        ids[group] = submit(a, group, deps, count);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 8,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &marks = args.tensor(0).ref();
    const simpler::hbg::Tensor &barrier = args.tensor(1).ref();
    const int64_t graph_case = args.scalar<int64_t>(0);
    const int64_t group_count = args.scalar<int64_t>(1);
    const int64_t block_num = args.scalar<int64_t>(2);
    const bool with_sync = args.scalar<int64_t>(3) != 0;
    const bool mixed = args.scalar<int64_t>(4) != 0;
    const int64_t sync_every = args.scalar<int64_t>(5);
    if (group_count < 1 || group_count > kGroupCapacity) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "group_count must be in [1, %ld], got %ld", static_cast<long>(kGroupCapacity),
            static_cast<long>(group_count)
        );
        return;
    }
    // Two blocks is the narrowest real cohort: one would be released by its own
    // arrival, and the orchestrator sets the attribute only above one block.
    if (block_num < 2 || block_num > kBlockNumMax) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "block_num must be in [2, %ld], got %ld", static_cast<long>(kBlockNumMax),
            static_cast<long>(block_num)
        );
        return;
    }
    if (sync_every < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "sync_every must be >= 1, got %ld", static_cast<long>(sync_every));
        return;
    }
    const BuildArgs build{marks, barrier, group_count, with_sync, mixed, block_num, sync_every};
    switch (graph_case) {
    case 0:
        build_chain(build);
        break;
    case 1:
        build_multi_root(build);
        break;
    case 2:
        build_fanin32(build);
        break;
    case 3:
        build_random(build);
        break;
    default:
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unknown graph_case %ld", static_cast<long>(graph_case));
        break;
    }
}

}  // extern "C"
