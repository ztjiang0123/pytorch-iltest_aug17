// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/shared/dequantize.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include <stdexcept>
#include <string>

namespace torchao::kernels::cpu::fallback::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight {

namespace internal {

// The fallback (x86) backend implements only the shared embedding op below.
// The activation-packing and linear-kernel entry points exist solely to
// satisfy UKernelConfig validation; if any of them is actually invoked it
// means linear execution was attempted on a backend that requires ARM NEON.
//
// Every such entry point has the same body: ignore its arguments and throw an
// "unsupported on fallback" error. Folding that body into one variadic helper
// lets each public stub delegate in a single line, so the stubs no longer
// duplicate a per-parameter `(void)`-cast + throw block. `Args&&...` absorbs
// (and thereby marks as used) whatever arguments the caller passed, regardless
// of the individual stub's arity or parameter types.
template <typename... Args>
[[noreturn]] inline void unsupported_on_fallback(
    const char* fn_name,
    Args&&... /*args*/) {
  throw std::runtime_error(
      std::string(fn_name) +
      " not implemented for fallback (x86). "
      "Linear execution requires ARM NEON.");
}

} // namespace internal

// Stub functions for activation packing / linear execution - required for
// UKernelConfig validation but throw at runtime since the fallback (x86)
// backend does not support linear execution. Each delegates to the shared
// internal::unsupported_on_fallback helper, forwarding its arguments so none
// are flagged as unused.

inline size_t packed_activations_size(
    int m,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  internal::unsupported_on_fallback(
      "packed_activations_size", m, k, group_size, has_weight_zeros, mr, kr, sr);
}

inline size_t packed_activations_offset(
    int m_idx,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  internal::unsupported_on_fallback(
      "packed_activations_offset",
      m_idx,
      k,
      group_size,
      has_weight_zeros,
      mr,
      kr,
      sr);
}

template <int mr_, int kr_, int sr_>
void pack_activations(
    void* packed_activations,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  internal::unsupported_on_fallback(
      "pack_activations",
      packed_activations,
      m,
      k,
      group_size,
      activations,
      has_weight_zeros,
      mr,
      kr,
      sr);
}

template <int weight_nbit, bool has_weight_zeros, bool has_lut>
void kernel_1x8x16_f32_fallback(
    float* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros_runtime,
    bool has_bias,
    bool has_clamp) {
  internal::unsupported_on_fallback(
      "kernel_1x8x16_f32_fallback",
      output,
      output_m_stride,
      m,
      n,
      k,
      group_size,
      packed_weights,
      packed_activations,
      clamp_min,
      clamp_max,
      has_weight_zeros_runtime,
      has_bias,
      has_clamp);
}

// Shared embedding op that uses linear packed weights
template <int weight_nbit, int nr, int kr, int sr>
inline void shared_embedding(
    float* out,
    const void* packed_weights,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    int index) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  int groups_per_k = k / group_size;
  std::vector<int8_t> weight_qvals(k * nr);
  std::vector<float> weight_scales(groups_per_k * nr);
  std::vector<int8_t> weight_zeros(groups_per_k * nr);
  std::vector<float> bias(nr);

  int n_idx = index / nr;
  n_idx = n_idx * nr;
  int j = index - n_idx;

  torchao::weight_packing::unpack_weights_at_n_idx<weight_nbit, nr, kr, sr>(
      weight_qvals.data(),
      weight_scales.data(),
      has_weight_zeros ? weight_zeros.data() : nullptr,
      has_bias ? bias.data() : nullptr,
      n_idx,
      n,
      k,
      group_size,
      has_weight_zeros,
      has_bias,
      packed_weights);

  for (int i = 0; i < k; i += 16) {
    int chunk_size = std::min(16, k - i);
    float scale = weight_scales[j * groups_per_k + i / group_size];
    float zero = 0.0f;
    if (has_weight_zeros) {
      zero =
          static_cast<float>(weight_zeros[j * groups_per_k + i / group_size]);
    }
    torchao::kernels::cpu::shared::dequantize_and_store_values(
        out + i, weight_qvals.data() + j * k + i, chunk_size, scale, zero);
  }
}

} // namespace
  // torchao::kernels::cpu::fallback::linear::channelwise_8bit_activation_groupwise_lowbit_weight
