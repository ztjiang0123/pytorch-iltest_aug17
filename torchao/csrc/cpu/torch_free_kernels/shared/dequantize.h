// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cstdint>

namespace torchao::kernels::cpu::shared {

// Dequantize `count` int8 quantized values into `out` using an affine
// transform: out[i] = (qvals[i] - zero) * scale. Shared by the fallback
// embedding and linear kernels.
TORCHAO_ALWAYS_INLINE inline void dequantize_and_store_values(
    float* out,
    const int8_t* qvals,
    int count,
    float scale,
    float zero) {
  for (int i = 0; i < count; ++i) {
    out[i] = (static_cast<float>(qvals[i]) - zero) * scale;
  }
}

} // namespace torchao::kernels::cpu::shared
