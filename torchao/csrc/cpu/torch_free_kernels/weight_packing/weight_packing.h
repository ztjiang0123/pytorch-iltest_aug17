// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Shared weight packing logic used by both aarch64 and fallback (x86).
// Architecture-specific pack_buffer/unpack_buffer/compute_sum live in
// aarch64/weight_packing_internals.h and fallback/weight_packing_internals.h.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/weight_packing_internals.h>
#else
#include <torchao/csrc/cpu/torch_free_kernels/fallback/weight_packing_internals.h>
#endif

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace torchao::weight_packing {

#if defined(__aarch64__) || defined(__ARM_NEON)
namespace impl = torchao::weight_packing::aarch64;
#else
namespace impl = torchao::weight_packing::fallback;
#endif

// ===== Value reordering for GEMM packing =====

// Packs nr * kr values for GEMM with packing params (nr, kr, sr)
// It takes (kr / sr) values from each of nr columns and writes to packed_values
// This is repeated sr times
template <typename T>
void pack_values(
    T* packed_values,
    const T* values,
    int nr,
    int kr,
    int sr) {
  assert(kr % sr == 0);
  int kr_per_sr = kr / sr;
  int dst_idx = 0;
  for (int sr_idx = 0; sr_idx < sr; sr_idx++) {
    for (int n_idx = 0; n_idx < nr; n_idx++) {
      std::memcpy(
          packed_values + dst_idx,
          values + n_idx * kr + sr_idx * kr_per_sr,
          sizeof(T) * kr_per_sr);
      dst_idx += kr_per_sr;
    }
  }
}

// Undoes pack_values
template <typename T>
void unpack_values(
    T* values,
    const T* packed_values,
    int nr,
    int kr,
    int sr) {
  assert(kr % sr == 0);
  int kr_per_sr = kr / sr;
  int dst_idx = 0;
  for (int sr_idx = 0; sr_idx < sr; sr_idx++) {
    for (int n_idx = 0; n_idx < nr; n_idx++) {
      std::memcpy(
          values + n_idx * kr + sr_idx * kr_per_sr,
          packed_values + dst_idx,
          sizeof(T) * kr_per_sr);
      dst_idx += kr_per_sr;
    }
  }
}

// ===== Sizing utilities =====

// Size in bytes of 1 packed weights column
inline size_t packed_weights_size_per_n(
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias) {
  assert(k % group_size == 0);
  int groups_per_col = k / group_size;
  int col_size = 0;

  // qvals
  col_size += (k / 8) * weight_nbit;

  // scales
  col_size += sizeof(float) * groups_per_col;

  // qvals_sum
  col_size += sizeof(int32_t) * groups_per_col;

  // zeros
  if (has_weight_zeros) {
    col_size += sizeof(int32_t) * groups_per_col;
  }

  // bias
  if (has_bias) {
    col_size += sizeof(float);
  }

  return col_size;
}

// Returns number of bytes required for weight_data
inline size_t packed_weights_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)kr;
  (void)sr;
  auto size_per_n = packed_weights_size_per_n(
      k, group_size, weight_nbit, has_weight_zeros, has_bias);

  // Replace n with next multiple of nr >= n
  n = ((n + nr - 1) / nr) * nr;
  return size_per_n * n;
}

// Returns offset into packed weights for a given n_idx
inline size_t packed_weights_offset(
    int n_idx,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)kr;
  (void)sr;
  auto size_per_n = packed_weights_size_per_n(
      k, group_size, weight_nbit, has_weight_zeros, has_bias);
  return (n_idx / nr) * nr * size_per_n;
}

// ===== Shared pack/unpack implementations =====

namespace detail {

// Append a single little-endian scalar to the packed buffer and advance the
// cursor. Keeps the store loops in pack_weights free of pointer arithmetic.
template <typename T>
inline void append_scalar(char*& cursor, T value) {
  *reinterpret_cast<T*>(cursor) = value;
  cursor += sizeof(T);
}

// Pack one group (group_size values) of the next nr columns starting at
// (n_idx, k_idx) into the packed buffer, accumulating per-column value sums.
// Columns past the end of n are zero-filled. Advances the cursor past the
// packed qvals.
template <int weight_nbit, int nr, int kr, int sr>
inline void pack_group_qvals(
    char*& cursor,
    int n_idx,
    int n,
    int k,
    int k_idx,
    int group_size,
    const int8_t* weight_qvals,
    std::array<int32_t, nr>& qvals_sum) {
  constexpr int packed_buffer_bytes = weight_nbit * nr * kr / 8;
  std::array<int8_t, nr * kr> buffer;
  int8_t packed_values[nr * kr];

  for (int idx_in_group = 0; idx_in_group < group_size; idx_in_group += kr) {
    // Fill buffer with the next kr values from the next nr columns.
    // If there are fewer than nr columns, 0s are stored.
    buffer.fill(0);
    for (int j = 0; j < nr; j++) {
      if (n_idx + j < n) {
        std::memcpy(
            buffer.data() + kr * j,
            weight_qvals + (n_idx + j) * k + (k_idx + idx_in_group),
            kr);
        qvals_sum[j] += impl::compute_sum(buffer.data() + kr * j, kr);
      }
    }

    torchao::weight_packing::pack_values(
        packed_values, buffer.data(), nr, kr, sr);
    impl::pack_buffer<weight_nbit, kr, nr>(cursor, packed_values);
    cursor += packed_buffer_bytes;
  }
}

// Store one float attribute per column (nr columns), zero-filling columns past
// the end of n. `value_for` maps a valid global column index to its value.
template <int nr, typename Fn>
inline void store_per_column_floats(char*& cursor, int n_idx, int n, Fn value_for) {
  for (int j = 0; j < nr; j++) {
    float value = 0.0f;
    if (n_idx + j < n) {
      value = value_for(n_idx + j);
    }
    append_scalar(cursor, value);
  }
}

// Store one int32 attribute per column (nr columns), zero-filling columns past
// the end of n. `value_for` maps a valid global column index to its value.
template <int nr, typename Fn>
inline void store_per_column_int32(char*& cursor, int n_idx, int n, Fn value_for) {
  for (int j = 0; j < nr; j++) {
    int32_t value = 0;
    if (n_idx + j < n) {
      value = value_for(n_idx + j);
    }
    append_scalar(cursor, value);
  }
}

} // namespace detail

// Pack weights into architecture-portable format
template <int weight_nbit, int nr, int kr, int sr>
inline void pack_weights(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias,
    int /*nr_rt*/,
    int /*kr_rt*/,
    int /*sr_rt*/) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  assert(nr * kr % 8 == 0);
  const bool has_weight_zeros = (weight_zeros != nullptr);
  const bool has_bias = (bias != nullptr);
  const int groups_per_k = k / group_size;

  // Buffer to hold the sum of weight_qvals in each column of the group.
  std::array<int32_t, nr> qvals_sum;

  // Cursor into the packed weights, advanced by each store below.
  char* cursor = reinterpret_cast<char*>(packed_weights);

  // Loop over n by nr.
  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    // Loop over groups along k.
    for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
      qvals_sum.fill(0);

      // Pack the group's qvals for the next nr columns and accumulate sums.
      detail::pack_group_qvals<weight_nbit, nr, kr, sr>(
          cursor,
          n_idx,
          n,
          k,
          group_idx * group_size,
          group_size,
          weight_qvals,
          qvals_sum);

      // Store the group's per-column attributes (scale, qval sums, zeros).
      // Columns past the end of n are zero-filled.
      detail::store_per_column_floats<nr>(
          cursor, n_idx, n, [&](int col) {
            return weight_scales[col * groups_per_k + group_idx];
          });
      detail::store_per_column_int32<nr>(
          cursor, n_idx, n, [&](int col) { return qvals_sum[col - n_idx]; });
      if (has_weight_zeros) {
        detail::store_per_column_int32<nr>(
            cursor, n_idx, n, [&](int col) {
              return static_cast<int32_t>(
                  weight_zeros[col * groups_per_k + group_idx]);
            });
      }
    } // loop over k (group_idx)

    // Store bias for the next nr columns, zero-filling past the end of n.
    if (has_bias) {
      detail::store_per_column_floats<nr>(
          cursor, n_idx, n, [&](int col) { return bias[col]; });
    }
  } // n_idx
}

// Unpack weights at n_idx to support shared embedding/unembedding
template <int weight_nbit, int nr, int kr, int sr>
void unpack_weights_at_n_idx(
    // Output
    int8_t* weight_qvals, // k * nr values at n_idx
    float* weight_scales, // groups_per_k * nr values at n_idx
    // weight_zeros is not extracted if has_weight_zeros is false
    int8_t* weight_zeros, // groups_per_k * nr values at n_idx
    // bias is not extracted if has_bias is false
    float* bias, // nr values at n_idx
    // Inputs
    int n_idx,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    const void* packed_weights) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  assert(n_idx % nr == 0);

  int groups_per_k = k / group_size;

  std::array<int8_t, nr * kr> buffer;
  int8_t packed_values[nr * kr];

  assert(nr * kr % 8 == 0);
  constexpr int packed_buffer_bytes = weight_nbit * nr * kr / 8;

  auto packed_weights_byte_ptr =
      reinterpret_cast<const char*>(packed_weights) +
      n_idx *
          torchao::weight_packing::packed_weights_size_per_n(
              k, group_size, weight_nbit, has_weight_zeros, has_bias);

  for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
    int k_idx = group_idx * group_size;
    for (int idx_in_group = 0; idx_in_group < group_size;
         idx_in_group += kr) {
      impl::unpack_buffer<weight_nbit, kr, nr>(
          packed_values, packed_weights_byte_ptr);
      packed_weights_byte_ptr += packed_buffer_bytes;
      torchao::weight_packing::unpack_values(
          buffer.data(), packed_values, nr, kr, sr);

      for (int j = 0; j < nr; j++) {
        if (n_idx + j < n) {
          std::memcpy(
              weight_qvals + j * k + (k_idx + idx_in_group),
              buffer.data() + kr * j,
              kr);
        }
      }
    }

    for (int j = 0; j < nr; j++) {
      float scale =
          *reinterpret_cast<const float*>(packed_weights_byte_ptr);
      packed_weights_byte_ptr += sizeof(float);
      if (n_idx + j < n) {
        weight_scales[j * groups_per_k + group_idx] = scale;
      }
    }

    // Skip over weight qval sums
    packed_weights_byte_ptr += nr * sizeof(int32_t);

    if (has_weight_zeros) {
      for (int j = 0; j < nr; j++) {
        int32_t zero =
            *reinterpret_cast<const int32_t*>(packed_weights_byte_ptr);
        packed_weights_byte_ptr += sizeof(int32_t);
        if (n_idx + j < n && weight_zeros != nullptr) {
          weight_zeros[j * groups_per_k + group_idx] =
              static_cast<int8_t>(zero);
        }
      }
    }
  }

  if (has_bias) {
    for (int j = 0; j < nr; j++) {
      float bias_ =
          *reinterpret_cast<const float*>(packed_weights_byte_ptr);
      packed_weights_byte_ptr += sizeof(float);
      if (n_idx + j < n && bias != nullptr) {
        bias[j] = bias_;
      }
    }
  }
}

template <int weight_nbit, int nr, int kr, int sr>
void unpack_weights(
    // Output
    int8_t* weight_qvals,
    float* weight_scales,
    // weight_zeros is not extracted if has_weight_zeros is false
    int8_t* weight_zeros,
    // bias is not extracted if has_bias is false
    float* bias,
    // Inputs
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    const void* packed_weights) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  int groups_per_k = k / group_size;

  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    torchao::weight_packing::unpack_weights_at_n_idx<
        weight_nbit,
        nr,
        kr,
        sr>(
        weight_qvals + n_idx * k,
        weight_scales + n_idx * groups_per_k,
        has_weight_zeros ? weight_zeros + n_idx * groups_per_k : nullptr,
        has_bias ? bias + n_idx : nullptr,
        n_idx,
        n,
        k,
        group_size,
        has_weight_zeros,
        has_bias,
        packed_weights);
  }
}

} // namespace torchao::weight_packing
