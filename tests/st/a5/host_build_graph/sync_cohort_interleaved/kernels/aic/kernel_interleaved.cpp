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
 * Cube half of the interleaved-cohort kernel: an optional spin, an optional
 * barrier across every lane of the task, and then a marker.
 *
 * The barrier is `sync_cohort`'s, for the reason given there: a sync-start task
 * whose blocks did not all start still writes correct values from the blocks
 * that did, so only a kernel that genuinely waits on its peers can tell a
 * cohort from an ordinary SPMD task. It is built like qwen3's attention kernel
 * and its `epochBarrierArriveAndWait` -- each lane bumps its own slot, then
 * sums every slot of the task until the total reaches one epoch per lane.
 *
 * The slot base arrives as a scalar rather than being derived from a group
 * index: this graph mixes tasks of different shapes, so `group * block_num *
 * lanes` would place two of them on the same slots.
 *
 * The spin is what staggers the lanes. It only ever *reads* its own slot, which
 * starts at zero and no peer writes, so `sink` is zero on every path and the
 * guarded store never fires -- it is there to keep the loop from being folded
 * away, not to record anything.
 *
 * The per-slot invalidate before each load is not optional: the peers ran on
 * other cores, so this core's data cache can hold the line from before they
 * wrote it.
 *
 * Args:
 *   args[0] = marks Tensor* (INOUT, float32)
 *   args[1] = barrier Tensor* (INOUT, int32)
 *   args[2] = scalar: first slot of this task's region
 *   args[3] = scalar: this task's block count
 *   args[4] = scalar: lanes per block -- 1 for a single-slot task, 3 for MIX
 *   args[5] = scalar: non-zero to run the barrier
 *   args[6] = scalar: spin iterations before the barrier
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
#define INTERLEAVED_USE_DEV_INTRIN 1
#else
#define INTERLEAVED_USE_DEV_INTRIN 0
#endif

static __aicore__ inline void interleaved_store_i32(__gm__ int32_t *ptr, int32_t value) {
#if INTERLEAVED_USE_DEV_INTRIN
    st_dev(static_cast<uint32_t>(value), reinterpret_cast<__gm__ uint32_t *>(ptr), 0);
#else
    *reinterpret_cast<volatile __gm__ int32_t *>(ptr) = value;
#endif
}

static __aicore__ inline int32_t interleaved_load_i32(__gm__ int32_t *ptr) {
#if INTERLEAVED_USE_DEV_INTRIN
    return static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(ptr), 0));
#else
    return *reinterpret_cast<volatile __gm__ int32_t *>(ptr);
#endif
}

static __aicore__ inline void interleaved_ddr_fence() {
#if INTERLEAVED_USE_DEV_INTRIN
    dsb(DSB_DDR);
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *marks_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *marks = reinterpret_cast<__gm__ float *>(marks_tensor->buffer.addr) + marks_tensor->start_offset;
    __gm__ Tensor *barrier_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *barrier =
        reinterpret_cast<__gm__ int32_t *>(barrier_tensor->buffer.addr) + barrier_tensor->start_offset;

    const int32_t base = static_cast<int32_t>(args[2]);
    const int32_t block_num = static_cast<int32_t>(args[3]);
    const int32_t lanes = static_cast<int32_t>(args[4]);
    const int32_t barrier_on = static_cast<int32_t>(args[5]);
    const int32_t spin = static_cast<int32_t>(args[6]);

    // The cube subtask is lane 0 of its block in every shape this suite builds.
    const int32_t physical_count = block_num * lanes;
    const int32_t slot = base + get_block_idx(args) * lanes;
    __gm__ int32_t *mine = &barrier[slot * INTS_PER_CACHE_LINE];

    if (spin > 0) {
        int32_t sink = 0;
        for (int32_t i = 0; i < spin; ++i) {
            dcci(mine, SINGLE_CACHE_LINE);
            sink += interleaved_load_i32(mine);
        }
        if (sink != 0) interleaved_store_i32(mine, sink);
    }

    if (barrier_on != 0) {
        dcci(mine, SINGLE_CACHE_LINE);
        interleaved_ddr_fence();
        const int32_t epoch = interleaved_load_i32(mine) + 1;
        interleaved_store_i32(mine, epoch);
        dcci(mine, SINGLE_CACHE_LINE, CACHELINE_OUT);
        interleaved_ddr_fence();

        const int64_t expected = static_cast<int64_t>(physical_count) * epoch;
        while (true) {
            for (int32_t i = 0; i < physical_count; ++i) {
                dcci(&barrier[(base + i) * INTS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
            }
            interleaved_ddr_fence();
            int64_t sum = 0;
            for (int32_t i = 0; i < physical_count; ++i) {
                sum += interleaved_load_i32(&barrier[(base + i) * INTS_PER_CACHE_LINE]);
            }
            if (sum >= expected) break;
        }
    }

    // Reached only once every peer arrived, which is the whole point.
    marks[slot * FLOATS_PER_CACHE_LINE] = static_cast<float>(slot + 1);
    dcci(&marks[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
