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
 * Cube half of the rendezvous marker.
 *
 * Writes `slot + 1` at cache line `slot`, where `slot = group * 3 + 0`. The
 * value encodes the position, so a lane that ran the wrong subtask of a task --
 * the cube kernel on a vector lane, say -- writes a value the golden does not
 * have rather than one that happens to match.
 *
 * Args:
 *   args[0] = output Tensor* (INOUT)
 *   args[1] = scalar: this task's group index
 *   args[2] = scalar: non-zero when this is a MIX task (the vector half's flag)
 *   args[3] = scalar: this task's block count
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
static constexpr int32_t SLOTS_PER_GROUP = 3;

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

    const int32_t group = static_cast<int32_t>(args[1]);
    // A MIX task with several blocks is several rendezvous of the same task, so
    // the block index picks which group of three slots this one owns.
    // args[2] is the mixed flag the vector half reads; the cube half has no
    // use for it but shares the one argument list, so the block count is [3].
    const int32_t block_num = static_cast<int32_t>(args[3]);
    const int32_t unit = group * block_num + get_block_idx(args);
    const int32_t slot = unit * SLOTS_PER_GROUP + 0;

    out[slot * FLOATS_PER_CACHE_LINE] = static_cast<float>(slot + 1);

    dcci(&out[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
