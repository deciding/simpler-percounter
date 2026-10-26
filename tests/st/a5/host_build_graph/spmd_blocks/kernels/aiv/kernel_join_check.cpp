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
 * Counts how many of its producer's blocks had actually written when this task
 * started, and writes that count at cache line `producer_blocks`.
 *
 * This is the only falsifiable test of the N-way join. A dependency edge is
 * task to task, so a consumer must not run until **every** block of its
 * producer has finished; a runtime that released it on the first block would
 * still finish the graph, and every marker value would still be correct, since
 * the remaining blocks go on to write their own lines. The count is what
 * differs: 24 when the join is right, and whatever happened to be published
 * when it is not.
 *
 * The per-line invalidate before each load is not optional. The producer's
 * blocks ran on other cores, so this core's data cache can hold lines from
 * before they wrote.
 *
 * Args:
 *   args[0] = output Tensor* (INOUT)
 *   args[1] = scalar: how many blocks the producer had
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    const int32_t producer_blocks = static_cast<int32_t>(args[1]);

    int32_t written = 0;
    for (int32_t block = 0; block < producer_blocks; ++block) {
        __gm__ float *line = &out[block * FLOATS_PER_CACHE_LINE];
        dcci(line, SINGLE_CACHE_LINE);
        if (*line != 0.0f) ++written;
    }

    const int32_t slot = producer_blocks;
    out[slot * FLOATS_PER_CACHE_LINE] = static_cast<float>(written);
    dcci(&out[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
