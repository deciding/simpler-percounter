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
 * MIX without sync-start, mixed into single-core DAGs: a shape one cluster can
 * run on its own, exercising the per-cluster rendezvous.
 *
 * Group `g` writes cache lines around `3g` and is laid out as:
 *
 *   g % 3 == 0  MIX      cube at 3g, the two vector lanes at 3g+1 and 3g+2
 *   g % 3 == 1  AIC only 3g
 *   g % 3 == 2  AIV only 3g+1
 *
 * so every shape below carries all three kinds and crosses the dependencies
 * between them. That is what the test is for: a lane blocked on a producer of
 * another kind while a rendezvous wants it is the case the second MIX
 * checkpoint exists to break, and it cannot arise in a graph of one kind.
 *
 * The four shapes mirror `multi_core_dag`'s, which is the comparison this is
 * meant to be read against:
 *
 *   0  chain       `g - 1 -> g` throughout. One task runnable at a time, so
 *                  every rendezvous runs against an otherwise idle device.
 *   1  fanin32     32 roots, then every later group waits on all of them. The
 *                  roots release a wide front at once.
 *   2  multi_root  no edges at all. Every MIX task is runnable immediately, so
 *                  the clusters contend for the MIX ticket at full width while
 *                  their own lists are full.
 *   3  random      0 to 4 producers per group, drawn from the groups before it.
 *                  Irregular depth and fanin, which is what makes a lane block
 *                  on a producer it cannot predict.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;
constexpr int64_t kGroupCapacity = 4096;
constexpr int32_t kMaxFanin = 4;
constexpr int32_t kFaninRoots = 32;

TaskId submit_group(
    const simpler::hbg::Tensor &marks, int64_t group, bool with_mix, int64_t block_num, const TaskId *deps,
    int32_t dep_count
) {
    const bool mixed = with_mix && group % 3 == 0;
    CoreTaskArgs task;
    // `add_no_dep`, not `add_inout`: every group writes its own cache lines of
    // this one tensor, so letting the overlap map infer dependencies from it
    // would chain all the groups write-after-write and collapse every shape
    // below into the chain. The structure under test is the explicit one.
    task.add_no_dep(marks);
    task.add_scalar(group);
    // `sub_block_id` is only meaningful inside a MIX task, so the vector kernel
    // is told which it is in rather than reading it unconditionally. The
    // schedulers resolve it differently for a lone AIV task -- legacy from the
    // physical lane it dispatched to, percounter from the slot it materialized
    // -- and both are within what intrinsic.h promises.
    task.add_scalar(static_cast<int64_t>(mixed ? 1 : 0));
    // Both halves of a MIX task place their line from this, so a block count
    // that did not reach the launch spec shows up as a wrong slot rather than
    // as agreement between two copies of one mistake.
    task.add_scalar(block_num);
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    task.launch_spec.set_block_num(static_cast<int16_t>(block_num));
    // No sync-start: a MIX task of several blocks is several per-cluster
    // rendezvous, which needs no cohort.
    if (mixed) {
        MixedKernels kernels;
        kernels.aic_kernel_id = kAicFuncId;
        kernels.aiv0_kernel_id = kAivFuncId;
        kernels.aiv1_kernel_id = kAivFuncId;
        return rt_submit_task(kernels, task).task_id();
    }
    // A group that would have been MIX becomes a lone cube task in the control,
    // which writes the same slot the cube subtask would have. The other two of
    // its three slots then stay zero, which is what the golden expects.
    if (group % 3 != 2) return rt_submit_aic_task(kAicFuncId, task).task_id();
    return rt_submit_aiv_task(kAivFuncId, task).task_id();
}

void build_chain(const simpler::hbg::Tensor &marks, int64_t group_count, bool with_mix, int64_t block_num) {
    TaskId previous = submit_group(marks, 0, with_mix, block_num, nullptr, 0);
    for (int64_t group = 1; group < group_count; ++group) {
        previous = submit_group(marks, group, with_mix, block_num, &previous, 1);
    }
}

void build_fanin32(const simpler::hbg::Tensor &marks, int64_t group_count, bool with_mix, int64_t block_num) {
    TaskId roots[kFaninRoots];
    const int32_t root_count =
        static_cast<int32_t>(group_count < kFaninRoots ? group_count : static_cast<int64_t>(kFaninRoots));
    for (int32_t group = 0; group < root_count; ++group) {
        roots[group] = submit_group(marks, group, with_mix, block_num, nullptr, 0);
    }
    for (int64_t group = root_count; group < group_count; ++group) {
        (void)submit_group(marks, group, with_mix, block_num, roots, root_count);
    }
}

void build_multi_root(const simpler::hbg::Tensor &marks, int64_t group_count, bool with_mix, int64_t block_num) {
    for (int64_t group = 0; group < group_count; ++group) {
        (void)submit_group(marks, group, with_mix, block_num, nullptr, 0);
    }
}

void build_random(const simpler::hbg::Tensor &marks, int64_t group_count, bool with_mix, int64_t block_num) {
    TaskId ids[kGroupCapacity];
    ids[0] = submit_group(marks, 0, with_mix, block_num, nullptr, 0);
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t group = 1; group < group_count; ++group) {
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
        ids[group] = submit_group(marks, group, with_mix, block_num, deps, count);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &marks = args.tensor(0).ref();
    const int64_t graph_case = args.scalar<int64_t>(0);
    const int64_t group_count = args.scalar<int64_t>(1);
    // The control: the same four shapes with no MIX group at all, so the cost
    // of a rendezvous can be read off one graph rather than inferred across
    // two suites with different kernels.
    const bool with_mix = args.scalar<int64_t>(2) != 0;
    // How many blocks each group runs. A MIX group at block_num > 1 is that
    // many per-cluster rendezvous of the same task, which is the shape only
    // this suite covers.
    const int64_t block_num = args.scalar<int64_t>(3);
    if (block_num < 1 || block_num > 32767) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "block_num must be in [1, 32767], got %ld", static_cast<long>(block_num)
        );
        return;
    }
    if (group_count < 1 || group_count > kGroupCapacity) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "group_count must be in [1, %ld], got %ld", static_cast<long>(kGroupCapacity),
            static_cast<long>(group_count)
        );
        return;
    }
    switch (graph_case) {
    case 0:
        build_chain(marks, group_count, with_mix, block_num);
        break;
    case 1:
        build_fanin32(marks, group_count, with_mix, block_num);
        break;
    case 2:
        build_multi_root(marks, group_count, with_mix, block_num);
        break;
    case 3:
        build_random(marks, group_count, with_mix, block_num);
        break;
    default:
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unknown graph_case %ld", static_cast<long>(graph_case));
        break;
    }
}

}  // extern "C"
