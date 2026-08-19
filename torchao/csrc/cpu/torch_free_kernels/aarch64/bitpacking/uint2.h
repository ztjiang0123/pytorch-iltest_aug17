// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint2.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

// This file contains bitpacking and unpacking methods for uint4.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

// The scalar (non-vectorized) pack/unpack routines are identical to the
// portable fallback implementations, so forward to that single source of truth
// instead of maintaining a byte-for-byte copy here.
TORCHAO_ALWAYS_INLINE inline void pack_4_uint2_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  torchao::kernels::cpu::fallback::bitpacking::internal::pack_4_uint2_values(
      packed, unpacked);
}

TORCHAO_ALWAYS_INLINE inline void unpack_4_uint2_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_4_uint2_values(
      unpacked, packed);
}

// Shared implementation of the vectorized uint2 pack, templated over the NEON
// vector width so the 64-bit (uint8x8_t) and 128-bit (uint8x16_t) variants stay
// a single source of truth. The width-specific intrinsics are provided by the
// small overload sets below. The shift amount is a template non-type parameter
// so the immediate operand of vshl_n_u8/vshlq_n_u8 is always a literal.
template <int n>
TORCHAO_ALWAYS_INLINE inline uint8x8_t vec_uint2_shl(const uint8x8_t& v) {
  return vshl_n_u8(v, n);
}
template <int n>
TORCHAO_ALWAYS_INLINE inline uint8x16_t vec_uint2_shl(const uint8x16_t& v) {
  return vshlq_n_u8(v, n);
}
TORCHAO_ALWAYS_INLINE inline uint8x8_t vec_uint2_orr(
    const uint8x8_t& a,
    const uint8x8_t& b) {
  return vorr_u8(a, b);
}
TORCHAO_ALWAYS_INLINE inline uint8x16_t vec_uint2_orr(
    const uint8x16_t& a,
    const uint8x16_t& b) {
  return vorrq_u8(a, b);
}
TORCHAO_ALWAYS_INLINE inline void vec_uint2_store(
    uint8_t* packed,
    const uint8x8_t& v) {
  vst1_u8(packed, v);
}
TORCHAO_ALWAYS_INLINE inline void vec_uint2_store(
    uint8_t* packed,
    const uint8x16_t& v) {
  vst1q_u8(packed, v);
}

template <typename vec_t>
TORCHAO_ALWAYS_INLINE inline void vec_pack_uint2_values(
    uint8_t* packed,
    const vec_t& unpacked0,
    const vec_t& unpacked1,
    const vec_t& unpacked2,
    const vec_t& unpacked3) {
  // Vectorize the following:
  // packed[0] = (unpacked[0] << 6) | (unpacked[1] << 4) | (unpacked[2] << 2) |
  // (unpacked[3]);
  vec_t vec_packed;
  vec_packed = vec_uint2_shl<6>(unpacked0);
  vec_packed = vec_uint2_orr(vec_packed, vec_uint2_shl<4>(unpacked1));
  vec_packed = vec_uint2_orr(vec_packed, vec_uint2_shl<2>(unpacked2));
  vec_packed = vec_uint2_orr(vec_packed, unpacked3);
  vec_uint2_store(packed, vec_packed);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint2_values(
    uint8_t* packed,
    const uint8x8_t& unpacked0,
    const uint8x8_t& unpacked1,
    const uint8x8_t& unpacked2,
    const uint8x8_t& unpacked3) {
  // Input is 32 bytes
  // Output is 8 bytes
  vec_pack_uint2_values(packed, unpacked0, unpacked1, unpacked2, unpacked3);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint2_values(
    uint8x8_t& unpacked0,
    uint8x8_t& unpacked1,
    uint8x8_t& unpacked2,
    uint8x8_t& unpacked3,
    const uint8_t* packed) {
  // Input is 8 bytes
  // Output is 32 bytes

  // Vectorize the following:
  // unpacked[0] = (packed[0] & 192) >> 6;
  // unpacked[1] = (packed[0] & 48) >> 4;
  // unpacked[2] = (packed[0] & 12) >> 2;
  // unpacked[3] = (packed[0] & 3);

  uint8x8_t vec_packed;

  vec_packed = vld1_u8(packed);
  unpacked0 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(192)), 6);
  unpacked1 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(48)), 4);
  unpacked2 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(12)), 2);
  unpacked3 = vand_u8(vec_packed, vdup_n_u8(3));
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint2_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // Input is 64 bytes
  // Output is 16 bytes
  vec_pack_uint2_values(packed, unpacked0, unpacked1, unpacked2, unpacked3);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint2_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Input is 16 bytes
  // Output is 64 bytes

  // Vectorize the following:
  // unpacked[0] = (packed[0] & 192) >> 6;
  // unpacked[1] = (packed[0] & 48) >> 4;
  // unpacked[2] = (packed[0] & 12) >> 2;
  // unpacked[3] = (packed[0] & 3);

  uint8x16_t vec_packed;

  vec_packed = vld1q_u8(packed);
  unpacked0 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(192)), 6);
  unpacked1 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(48)), 4);
  unpacked2 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(12)), 2);
  unpacked3 = vandq_u8(vec_packed, vdupq_n_u8(3));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
