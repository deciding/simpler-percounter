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
 * Plain SPMD: one slot per task, `block_num` blocks, no sync-start.
 *
 * Every other SPMD workload in the tree is also MIX or sync-start. This is the
 * shape that is nothing but multiplicity -- the same subtask run N times with a
 * different `block_idx` each -- which makes it the one graph that isolates it.
 *
 * Group `g` writes the `block_num` cache lines at `g * block_num ..`, and
 * alternates core type by parity so both order lists carry SPMD entries:
 *
 *   g % 2 == 0  AIC
 *   g % 2 == 1  AIV
 *
 * The first four shapes mirror `multi_core_dag`'s, which
 * is the comparison this is meant to be read against:
 *
 *   0  chain       `g - 1 -> g` throughout. One task runnable at a time, so
 *                  every task's blocks have the whole device to spread over.
 *   1  fanin32     32 roots, then every later group waits on all of them.
 *   2  multi_root  no edges at all. Every block of every task is runnable at
 *                  once, which is where a claim batch that is too large leaves
 *                  lanes idle on a spent list.
 *   3  random      0 to 4 producers per group, drawn from the groups before it.
 *   4  join        one wide producer and one consumer that counts how many of
 *                  its blocks had published. The only shape that tells a
 *                  correct N-way join from a consumer released on block 0.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicMarkFuncId = 0;
constexpr int32_t kAivMarkFuncId = 1;
constexpr int32_t kAivJoinFuncId = 2;
constexpr int64_t kGroupCapacity = 4096;
constexpr int32_t kMaxFanin = 4;
constexpr int32_t kFaninRoots = 32;
constexpr int64_t kBlockNumMax = 32767;

TaskId submit_group(
    const simpler::hbg::Tensor &marks, int64_t group, int64_t block_num, const TaskId *deps, int32_t dep_count
) {
    CoreTaskArgs task;
    // `add_no_dep`, not `add_inout`: every block writes its own cache line of
    // this one tensor, so letting the overlap map infer dependencies from it
    // would chain every group write-after-write and collapse each shape below
    // into the chain. The structure under test is the explicit one.
    task.add_no_dep(marks);
    task.add_scalar(group);
    // The kernel places its line from this rather than from get_block_num(), so
    // a block count that did not reach the launch spec shows up as a wrong slot
    // instead of as agreement between two copies of one mistake.
    task.add_scalar(block_num);
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    task.launch_spec.set_block_num(static_cast<int16_t>(block_num));
    // No sync-start: this graph is multiplicity alone.

    if (group % 2 == 0) return rt_submit_aic_task(kAicMarkFuncId, task).task_id();
    return rt_submit_aiv_task(kAivMarkFuncId, task).task_id();
}

void build_chain(const simpler::hbg::Tensor &marks, int64_t group_count, int64_t block_num) {
    TaskId previous = submit_group(marks, 0, block_num, nullptr, 0);
    for (int64_t group = 1; group < group_count; ++group) {
        previous = submit_group(marks, group, block_num, &previous, 1);
    }
}

void build_fanin32(const simpler::hbg::Tensor &marks, int64_t group_count, int64_t block_num) {
    TaskId roots[kFaninRoots];
    const int32_t root_count =
        static_cast<int32_t>(group_count < kFaninRoots ? group_count : static_cast<int64_t>(kFaninRoots));
    for (int32_t group = 0; group < root_count; ++group) {
        roots[group] = submit_group(marks, group, block_num, nullptr, 0);
    }
    for (int64_t group = root_count; group < group_count; ++group) {
        (void)submit_group(marks, group, block_num, roots, root_count);
    }
}

void build_multi_root(const simpler::hbg::Tensor &marks, int64_t group_count, int64_t block_num) {
    for (int64_t group = 0; group < group_count; ++group) {
        (void)submit_group(marks, group, block_num, nullptr, 0);
    }
}

void build_random(const simpler::hbg::Tensor &marks, int64_t group_count, int64_t block_num) {
    TaskId ids[kGroupCapacity];
    ids[0] = submit_group(marks, 0, block_num, nullptr, 0);
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
        ids[group] = submit_group(marks, group, block_num, deps, count);
    }
}

// One producer of `block_num` blocks writing lines 0..block_num-1, and one
// single-block consumer that counts how many of them it can see.
void build_join(const simpler::hbg::Tensor &marks, int64_t block_num) {
    const TaskId producer = submit_group(marks, 0, block_num, nullptr, 0);

    CoreTaskArgs consumer;
    consumer.add_no_dep(marks);
    consumer.add_scalar(block_num);
    consumer.set_dependencies(&producer, 1);
    (void)rt_submit_aiv_task(kAivJoinFuncId, consumer);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &marks = args.tensor(0).ref();
    const int64_t graph_case = args.scalar<int64_t>(0);
    const int64_t group_count = args.scalar<int64_t>(1);
    const int64_t block_num = args.scalar<int64_t>(2);
    if (group_count < 1 || group_count > kGroupCapacity) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "group_count must be in [1, %ld], got %ld", static_cast<long>(kGroupCapacity),
            static_cast<long>(group_count)
        );
        return;
    }
    if (block_num < 1 || block_num > kBlockNumMax) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "block_num must be in [1, %ld], got %ld", static_cast<long>(kBlockNumMax),
            static_cast<long>(block_num)
        );
        return;
    }
    switch (graph_case) {
    case 0:
        build_chain(marks, group_count, block_num);
        break;
    case 1:
        build_fanin32(marks, group_count, block_num);
        break;
    case 2:
        build_multi_root(marks, group_count, block_num);
        break;
    case 3:
        build_random(marks, group_count, block_num);
        break;
    case 4:
        build_join(marks, block_num);
        break;
    default:
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unknown graph_case %ld", static_cast<long>(graph_case));
        break;
    }
}

}  // extern "C"
