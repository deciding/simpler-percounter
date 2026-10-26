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
 * Vector half of the block marker.
 *
 * Writes `slot + 1` at cache line `slot`, where `slot = group * block_num +
 * block_idx`. The value encodes the position, so a block that ran with the
 * wrong index writes a value the golden does not have rather than one that
 * happens to match -- which is the whole of what this suite checks, since
 * `block_idx` is the only thing that distinguishes two blocks of one task.
 *
 * Args:
 *   args[0] = output Tensor* (INOUT)
 *   args[1] = scalar: this task's group index
 *   args[2] = scalar: this task's block count
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

    const int32_t group = static_cast<int32_t>(args[1]);
    const int32_t block_num = static_cast<int32_t>(args[2]);
    const int32_t slot = group * block_num + get_block_idx(args);

    out[slot * FLOATS_PER_CACHE_LINE] = static_cast<float>(slot + 1);

    dcci(&out[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
