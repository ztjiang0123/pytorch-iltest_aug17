// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstdint>

// Shared, backend-agnostic index arithmetic for the per-row embedding entry
// points. Both the aarch64 (NEON) and fallback (scalar) backends expose an
// `embedding` / `pack_embedding_weight_qvals` wrapper whose only job is to
// advance the packed-weight / scale / zero-point pointers to the requested
// `index` and then delegate to a backend-specific single-row implementation.
// That pointer math was byte-for-byte identical across backends, so it lives
// here once and each backend supplies its own single-row implementation as a
// template argument.
namespace torchao::kernels::cpu::embedding_shared {

// Computes the per-row pointer offsets for `index` and forwards to
// `EmbeddingImpl`, the backend single-row dequantize-and-store routine with
// signature:
//   void(float* out, int embedding_dim, int group_size,
//        const uint8_t* packed_weight_qvals, const float* weight_scales,
//        const int8_t* weight_zeros)
template <int weight_nbit, typename EmbeddingImpl>
inline void embedding_at_index(
    EmbeddingImpl&& embedding_impl,
    float* out,
    int embedding_dim,
    int group_size,
    const void* packed_weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    int index) {
  assert(group_size % 32 == 0);
  assert(embedding_dim % group_size == 0);

  auto packed_weight_qvals_byte_ptr =
      reinterpret_cast<const uint8_t*>(packed_weight_qvals);

  int groups_per_embedding = embedding_dim / group_size;
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;

  packed_weight_qvals_byte_ptr += (index * packed_bytes_per_embedding);
  weight_scales += index * groups_per_embedding;
  if (weight_zeros != nullptr) {
    weight_zeros += index * groups_per_embedding;
  }

  embedding_impl(
      out,
      embedding_dim,
      group_size,
      packed_weight_qvals_byte_ptr,
      weight_scales,
      weight_zeros);
}

// Computes the per-row pointer offsets for `index` and forwards to
// `PackImpl`, the backend single-row packing routine with signature:
//   void(uint8_t* packed_qvals, int embedding_dim, const int8_t* qvals)
//
// `embedding_dim_multiple` is the backend's alignment requirement for
// `embedding_dim`; the aarch64 backend permits multiples of 8 while the
// fallback backend requires multiples of 32.
template <int weight_nbit, typename PackImpl>
inline void pack_embedding_weight_qvals_at_index(
    PackImpl&& pack_impl,
    int embedding_dim_multiple,
    void* packed_qvals,
    int embedding_dim,
    const int8_t* qvals,
    int index) {
  (void)embedding_dim_multiple; // only used by the assert below
  assert(embedding_dim % embedding_dim_multiple == 0);
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;
  auto packed_qvals_byte_ptr = reinterpret_cast<uint8_t*>(packed_qvals);

  pack_impl(
      packed_qvals_byte_ptr + index * packed_bytes_per_embedding,
      embedding_dim,
      qvals + index * embedding_dim);
}

} // namespace torchao::kernels::cpu::embedding_shared

// Emits the two per-row entry points (`embedding` and
// `pack_embedding_weight_qvals`) that every backend exposes. Their bodies were
// byte-for-byte identical across the aarch64 and fallback headers apart from the
// backend's `embedding_dim` alignment requirement, so they are generated here
// once. Each backend invokes this macro inside its own namespace; the
// `embedding_` / `pack_embedding_weight_qvals_` single-row implementations it
// forwards to are resolved from that enclosing namespace.
//
// `embedding_dim_multiple` is the backend's alignment requirement for
// `embedding_dim` (the aarch64 backend permits multiples of 8, the fallback
// backend requires multiples of 32).
#define TORCHAO_DEFINE_EMBEDDING_ENTRY_POINTS(embedding_dim_multiple)          \
  template <int weight_nbit>                                                   \
  inline void embedding(                                                       \
      /* Output */                                                             \
      float* out,                                                             \
      /* Inputs */                                                            \
      int embedding_dim,                                                       \
      int group_size,                                                          \
      const void* packed_weight_qvals,                                         \
      const float* weight_scales,                                              \
      const int8_t* weight_zeros,                                              \
      int index) {                                                             \
    torchao::kernels::cpu::embedding_shared::embedding_at_index<weight_nbit>(  \
        [](float* out,                                                         \
           int embedding_dim,                                                  \
           int group_size,                                                     \
           const uint8_t* packed_weight_qvals,                                 \
           const float* weight_scales,                                         \
           const int8_t* weight_zeros) {                                       \
          embedding_<weight_nbit>(                                             \
              out,                                                             \
              embedding_dim,                                                   \
              group_size,                                                      \
              packed_weight_qvals,                                             \
              weight_scales,                                                   \
              weight_zeros);                                                   \
        },                                                                     \
        out,                                                                   \
        embedding_dim,                                                         \
        group_size,                                                            \
        packed_weight_qvals,                                                   \
        weight_scales,                                                         \
        weight_zeros,                                                          \
        index);                                                               \
  }                                                                            \
                                                                              \
  template <int weight_nbit>                                                   \
  inline void pack_embedding_weight_qvals(                                     \
      /* Output */                                                             \
      void* packed_qvals,                                                      \
      /* Inputs */                                                             \
      int embedding_dim,                                                       \
      const int8_t* qvals,                                                     \
      int index) {                                                             \
    torchao::kernels::cpu::embedding_shared::                                  \
        pack_embedding_weight_qvals_at_index<weight_nbit>(                     \
            [](uint8_t* packed_qvals, int embedding_dim, const int8_t* qvals) {\
              pack_embedding_weight_qvals_<weight_nbit>(                       \
                  packed_qvals, embedding_dim, qvals);                         \
            },                                                                 \
            /*embedding_dim_multiple=*/(embedding_dim_multiple),               \
            packed_qvals,                                                      \
            embedding_dim,                                                     \
            qvals,                                                             \
            index);                                                            \
  }
