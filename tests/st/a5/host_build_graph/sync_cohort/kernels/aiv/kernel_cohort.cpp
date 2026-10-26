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
 * Vector half of the cohort kernel: a barrier across every lane of the task, and
 * then a marker.
 *
 * **A marker alone cannot test a cohort.** A sync-start task whose blocks did
 * not all start still writes correct values from the blocks that did; the
 * defect is that blocks never start, which no amount of checking the values
 * written can see. So this kernel genuinely waits on its peers, built like the
 * workload that motivates the attribute -- qwen3's attention kernel and its
 * `epochBarrierArriveAndWait`: each lane bumps its own slot, then sums every
 * slot of the task until the total reaches one epoch per lane.
 *
 * A cohort one lane short therefore spins here forever. **The expected failure
 * is the 45 s op timeout (`HandleTaskTimeout`), not a golden mismatch**, and a
 * reader who sees one should read it as this kernel doing exactly what it was
 * written to do rather than as a deadlock in the scheduler.
 *
 * The per-slot invalidate before each load is not optional: the peers ran on
 * other cores, so this core's data cache can hold the line from before they
 * wrote it.
 *
 * Args:
 *   args[0] = marks Tensor* (INOUT, float32)
 *   args[1] = barrier Tensor* (INOUT, int32)
 *   args[2] = scalar: this task's group index
 *   args[3] = scalar: this task's block count
 *   args[4] = scalar: lanes per block -- 1 for a single-slot task, 3 for MIX
 *   args[5] = scalar: non-zero to run the barrier
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;
static constexpr int32_t INTS_PER_CACHE_LINE = 16;

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

#if defined(__CCE_AICORE__) && !defined(__CPU_SIM) && !defined(__COSTMODEL)
#define SYNC_COHORT_USE_DEV_INTRIN 1
#else
#define SYNC_COHORT_USE_DEV_INTRIN 0
#endif

static __aicore__ inline void cohort_store_i32(__gm__ int32_t *ptr, int32_t value) {
#if SYNC_COHORT_USE_DEV_INTRIN
    st_dev(static_cast<uint32_t>(value), reinterpret_cast<__gm__ uint32_t *>(ptr), 0);
#else
    *reinterpret_cast<volatile __gm__ int32_t *>(ptr) = value;
#endif
}

static __aicore__ inline int32_t cohort_load_i32(__gm__ int32_t *ptr) {
#if SYNC_COHORT_USE_DEV_INTRIN
    return static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(ptr), 0));
#else
    return *reinterpret_cast<volatile __gm__ int32_t *>(ptr);
#endif
}

static __aicore__ inline void cohort_ddr_fence() {
#if SYNC_COHORT_USE_DEV_INTRIN
    dsb(DSB_DDR);
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *marks_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *marks = reinterpret_cast<__gm__ float *>(marks_tensor->buffer.addr) + marks_tensor->start_offset;
    __gm__ Tensor *barrier_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *barrier =
        reinterpret_cast<__gm__ int32_t *>(barrier_tensor->buffer.addr) + barrier_tensor->start_offset;

    const int32_t group = static_cast<int32_t>(args[2]);
    const int32_t block_num = static_cast<int32_t>(args[3]);
    const int32_t lanes = static_cast<int32_t>(args[4]);
    const int32_t barrier_on = static_cast<int32_t>(args[5]);

    // `sub_block_id` is only meaningful inside a MIX task -- intrinsic.h says so
    // -- and the two schedulers resolve it differently for a lone AIV task, so
    // the lane count decides whether to read it at all. In a MIX task the two
    // vector subtasks are lanes 1 and 2 of their block, behind the cube.
    const int32_t physical_count = block_num * lanes;
    const int32_t base = group * physical_count;
    const int32_t lane_in_block = lanes > 1 ? 1 + get_sub_block_id(args) : 0;
    const int32_t slot = base + get_block_idx(args) * lanes + lane_in_block;

    if (barrier_on != 0) {
        __gm__ int32_t *mine = &barrier[slot * INTS_PER_CACHE_LINE];
        dcci(mine, SINGLE_CACHE_LINE);
        cohort_ddr_fence();
        const int32_t epoch = cohort_load_i32(mine) + 1;
        cohort_store_i32(mine, epoch);
        dcci(mine, SINGLE_CACHE_LINE, CACHELINE_OUT);
        cohort_ddr_fence();

        const int64_t expected = static_cast<int64_t>(physical_count) * epoch;
        while (true) {
            for (int32_t i = 0; i < physical_count; ++i) {
                dcci(&barrier[(base + i) * INTS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
            }
            cohort_ddr_fence();
            int64_t sum = 0;
            for (int32_t i = 0; i < physical_count; ++i) {
                sum += cohort_load_i32(&barrier[(base + i) * INTS_PER_CACHE_LINE]);
            }
            if (sum >= expected) break;
        }
    }

    // Reached only once every peer arrived, which is the whole point.
    marks[slot * FLOATS_PER_CACHE_LINE] = static_cast<float>(slot + 1);
    dcci(&marks[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
