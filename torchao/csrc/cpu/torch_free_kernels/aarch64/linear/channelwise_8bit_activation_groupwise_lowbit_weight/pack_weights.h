// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>
#include <array>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing {

namespace internal {

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer_for_lut(
    void* packed_weights,
    const int8_t* buffer) {
  static_assert(weight_nbit >= 1);
  static_assert(weight_nbit <= 4);
  const uint8_t* buffer_u8 = reinterpret_cast<const uint8_t*>(buffer);
  if constexpr (kr * nr == 128) {
    bitpacking::vec_pack_128_uintx_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_u8(buffer_u8),
        vld1q_u8(buffer_u8 + 16),
        vld1q_u8(buffer_u8 + 32),
        vld1q_u8(buffer_u8 + 48),
        vld1q_u8(buffer_u8 + 64),
        vld1q_u8(buffer_u8 + 80),
        vld1q_u8(buffer_u8 + 96),
        vld1q_u8(buffer_u8 + 112));
    return;
  }
  assert(false);
}

TORCHAO_ALWAYS_INLINE inline void
map_values(int8_t* dst, int8_t* src, int8x16_t lut, int size) {
  // src will be in range 0 to 16, which fits in int8_t
  assert(size % 16 == 0);
  for (int i = 0; i < size; i += 16) {
    uint8x16_t idx = vreinterpretq_u8_s8(vld1q_s8(src + i));
    vst1q_s8(dst + i, vqtbl1q_s8(lut, idx));
  }
}

// Loop-invariant state for the LUT pack routine. Bundling it here lets the
// per-group/per-column helpers take a single context argument instead of a
// long parameter list, and keeps the main loop shallow.
template <int weight_nbit, int nr, int kr, int sr>
struct PackLutContext {
  // Inputs
  const int8_t* weight_qval_idxs;
  const float* weight_scales;
  const int8_t* weight_zeros; // nullptr if not packed
  const float* bias; // nullptr if not packed
  int n;
  int k;
  int groups_per_k;

  // Reused scratch buffers
  std::array<int8_t, nr * kr>& buffer;
  int8_t* packed_values;
  std::array<int8_t, kr>& mapped_val_buffer;
  std::array<int, nr>& qvals_sum;
  int packed_buffer_bytes;

  // Current LUT for the nr-column block being packed
  int8x16_t lut;

  // Output cursor
  char* out;
};

// Gather the kr values (per column) for the next nr columns at a given column
// offset, accumulate their LUT-mapped sums, and pack them into the output.
// Columns beyond n are zero-filled.
template <int weight_nbit, int nr, int kr, int sr>
TORCHAO_ALWAYS_INLINE inline void pack_next_nr_columns(
    PackLutContext<weight_nbit, nr, kr, sr>& ctx,
    int n_idx,
    int col_offset) {
  // Fill buffer with next kr values from the next nr columns
  // If there are fewer than nr columns, 0s are stored
  ctx.buffer.fill(0);
  for (int j = 0; j < nr; j++) {
    if (n_idx + j >= ctx.n) {
      continue;
    }
    std::memcpy(
        ctx.buffer.data() + kr * j,
        ctx.weight_qval_idxs + (n_idx + j) * ctx.k + col_offset,
        kr);
    internal::map_values(
        ctx.mapped_val_buffer.data(), ctx.buffer.data() + kr * j, ctx.lut, kr);
    ctx.qvals_sum[j] +=
        reduction::compute_sum(ctx.mapped_val_buffer.data(), kr);
  }

  // Pack buffer
  torchao::weight_packing::pack_values(
      ctx.packed_values, ctx.buffer.data(), nr, kr, sr);
  internal::pack_buffer_for_lut<weight_nbit, kr, nr>(ctx.out, ctx.packed_values);
  ctx.out += ctx.packed_buffer_bytes;
}

// Pack all kr-strided columns for one group (the inner idx_in_group loop).
template <int weight_nbit, int nr, int kr, int sr>
TORCHAO_ALWAYS_INLINE inline void pack_group_columns(
    PackLutContext<weight_nbit, nr, kr, sr>& ctx,
    int n_idx,
    int group_idx,
    int group_size) {
  int k_idx = group_idx * group_size;
  for (int idx_in_group = 0; idx_in_group < group_size; idx_in_group += kr) {
    pack_next_nr_columns<weight_nbit, nr, kr, sr>(
        ctx, n_idx, k_idx + idx_in_group);
  }
}

// Store the per-group attributes (scales, qval sums, and optional zeros) for
// the next nr columns. Columns beyond n are zero-filled.
template <int weight_nbit, int nr, int kr, int sr>
TORCHAO_ALWAYS_INLINE inline void store_group_attributes(
    PackLutContext<weight_nbit, nr, kr, sr>& ctx,
    int n_idx,
    int group_idx) {
  // Store weight scales
  for (int j = 0; j < nr; j++) {
    float32_t scale = 0.0;
    if (n_idx + j < ctx.n) {
      scale = ctx.weight_scales[(n_idx + j) * ctx.groups_per_k + group_idx];
    }
    *((float*)ctx.out) = scale;
    ctx.out += sizeof(float);
  }

  // Store weight qval sums
  for (int j = 0; j < nr; j++) {
    *((int*)ctx.out) = ctx.qvals_sum[j];
    ctx.out += sizeof(int);
  }

  // Store weight zeros
  if (ctx.weight_zeros == nullptr) {
    return;
  }
  for (int j = 0; j < nr; j++) {
    int32_t zero = 0;
    if (n_idx + j < ctx.n) {
      zero = ctx.weight_zeros[(n_idx + j) * ctx.groups_per_k + group_idx];
    }
    *((int32_t*)ctx.out) = zero;
    ctx.out += sizeof(int32_t);
  }
}

// Store the optional per-nr-column bias values. Columns beyond n are
// zero-filled.
template <int weight_nbit, int nr, int kr, int sr>
TORCHAO_ALWAYS_INLINE inline void store_bias(
    PackLutContext<weight_nbit, nr, kr, sr>& ctx,
    int n_idx) {
  if (ctx.bias == nullptr) {
    return;
  }
  for (int j = 0; j < nr; j++) {
    float bias_ = 0.0;
    if (n_idx + j < ctx.n) {
      bias_ = ctx.bias[n_idx + j];
    }
    *((float*)ctx.out) = bias_;
    ctx.out += sizeof(float);
  }
}

// LUT-specific pack_weights implementation (aarch64-only, uses NEON for LUT mapping)
template <int weight_nbit, int nr, int kr, int sr>
TORCHAO_ALWAYS_INLINE inline void pack_weights_with_lut_impl(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qval_idxs,
    // number of luts, must be nr group or coarser (per tensor)
    int n_luts,
    // luts (each 2**weight_nbit values)
    const int8_t* luts,
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  int groups_per_k = k / group_size;

  // LUT has size lut_size, which is <= 16
  // If lut_size < 16, we extend it with 0s in lut_buffer
  constexpr int lut_size = (1 << weight_nbit);
  std::array<int8_t, 16> lut_buffer;

  // Buffer to hold kr qvals, mapped with LUT from qval_idxs
  std::array<int8_t, kr> mapped_val_buffer;

  static_assert(weight_nbit <= 4);
  static_assert(lut_size <= 16);
  lut_buffer.fill(0);

  assert(n % n_luts == 0);
  int cols_per_lut = n / n_luts;
  assert((cols_per_lut == n) || (cols_per_lut % nr == 0));

  // Buffer to hold (kr * nr) values
  std::array<int8_t, nr * kr> buffer;

  // Buffer to hold (kr * nr) values after those values
  // are packed by params (nr, kr, sr)
  int8_t packed_values[buffer.size()];

  // Bytes of packed buffer of (nr * kr) values
  assert(nr * kr % 8 == 0);
  constexpr int packed_buffer_bytes = weight_nbit * nr * kr / 8;

  // Buffer to hold sum of weight_qvals in each column group
  std::array<int, nr> qvals_sum;

  // Bundle the loop-invariant state; ctx.out is the packed-weights cursor.
  PackLutContext<weight_nbit, nr, kr, sr> ctx{
      weight_qval_idxs,
      weight_scales,
      weight_zeros,
      bias,
      n,
      k,
      groups_per_k,
      buffer,
      packed_values,
      mapped_val_buffer,
      qvals_sum,
      packed_buffer_bytes,
      vdupq_n_s8(0),
      (char*)packed_weights};

  // Loop over n by nr
  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    // Look over groups along k
    for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
      // Populate lut and write it out to packed_weights
      if (group_idx == 0) {
        int lut_idx = n_idx / cols_per_lut;
        std::memcpy(lut_buffer.data(), luts + lut_idx * lut_size, lut_size);
        ctx.lut = vld1q_s8(lut_buffer.data());
        vst1q_s8((int8_t*)ctx.out, ctx.lut);
        ctx.out += 16;
      }

      // Initialize qvals_sum for each group to 0
      qvals_sum.fill(0);

      // Pack the weights for the next nr columns, group by group
      pack_group_columns<weight_nbit, nr, kr, sr>(
          ctx, n_idx, group_idx, group_size);

      // Store group attributes scale, qval_sums, and zeros for next nr columns.
      // If there are fewer than nr columns, 0s are stored.
      store_group_attributes<weight_nbit, nr, kr, sr>(ctx, n_idx, group_idx);
    } // loop over k (group_idx)

    // Store bias for next nr columns
    store_bias<weight_nbit, nr, kr, sr>(ctx, n_idx);
  } // n_idx
}

} // namespace internal

template <int weight_nbit, int nr, int kr, int sr>
void pack_weights_with_lut(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qval_idxs,
    int n_luts,
    const int8_t* luts,
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias) {
  internal::pack_weights_with_lut_impl<weight_nbit, nr, kr, sr>(
      packed_weights,
      n,
      k,
      group_size,
      weight_qval_idxs,
      n_luts,
      luts,
      weight_scales,
      weight_zeros,
      bias);
}

size_t inline packed_weights_with_lut_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr) {
  auto packed_weights_col_size =
      torchao::weight_packing::packed_weights_size_per_n(
          k, group_size, weight_nbit, has_weight_zeros, has_bias);

  // Replace n with next multiple of nr >= n
  n = ((n + nr - 1) / nr) * nr;

  // Per nr columns, we have one 16 byte lut
  auto lut_size = (n / nr) * 16;

  return packed_weights_col_size * n + lut_size;
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
