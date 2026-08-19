// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cstdint>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {

/**
 * @brief Shared "transpose-and-pack" primitive for the 7-bit format.
 *
 * The count-64 and count-128 transpose loops are byte-for-byte identical apart
 * from the block size, so the loop lives here once, parameterized by `block`
 * (the number of columns, equal to the packed row stride). The unpacked data is
 * organized as eight rows of `block` values: rows 0..6 supply the low 7 bits of
 * each packed byte, and row 7 is bit-sliced across the most significant bit of
 * the seven packed bytes in each column.
 *
 * @tparam block Number of columns (== packed row stride).
 */
template <int block>
TORCHAO_ALWAYS_INLINE inline void pack_transpose_uint7_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int j = 0; j < block; ++j) { // Iterate through columns
    const uint8_t u7 = unpacked[7 * block + j] & 0x7F;
    for (int i = 0; i < 7; ++i) { // Iterate through rows
      uint8_t u7_bit = (u7 >> i) & 1;
      packed[i * block + j] = (unpacked[i * block + j] & 0x7F) | (u7_bit << 7);
    }
  }
}

/**
 * @brief Inverse of pack_transpose_uint7_values.
 *
 * @tparam block Number of columns (== packed row stride).
 */
template <int block>
TORCHAO_ALWAYS_INLINE inline void unpack_transpose_uint7_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int j = 0; j < block; ++j) { // Iterate through columns
    uint8_t u7 = 0;
    for (int i = 0; i < 7; ++i) { // Iterate through rows
      unpacked[i * block + j] = packed[i * block + j] & 0x7F;
      u7 |= ((packed[i * block + j] >> 7) & 1) << i;
    }
    unpacked[7 * block + j] = u7;
  }
}

/**
 * @brief Packs 8 bytes, each holding a 7-bit value (0-127), into 7 bytes.
 *
 * @param packed Pointer to the destination memory (7 bytes).
 * @param unpacked Pointer to the source memory (8 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_8_uint7_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // pack 8 uint7 values (u0..u7) into 7 bytes (p0..p6)
  // The 7 bits of u7 are distributed across the most significant bit (MSB)
  // of each of the 7 packed bytes.
  // p0 = u7_bit_0 | u0_all_7_bits
  // p1 = u7_bit_1 | u1_all_7_bits
  // ...
  // p6 = u7_bit_6 | u6_all_7_bits
  const uint8_t u7 = unpacked[7] & 0x7F;

  for (int i = 0; i < 7; ++i) {
    uint8_t u7_bit = (u7 >> i) & 1;
    packed[i] = (unpacked[i] & 0x7F) | (u7_bit << 7);
  }
}

/**
 * @brief Unpacks 7 bytes into 8 bytes, each containing a 7-bit value.
 *
 * @param unpacked Pointer to the destination memory (8 bytes).
 * @param packed Pointer to the source memory (7 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_8_uint7_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpacked[7] = 0;
  for (int i = 0; i < 7; ++i) {
    // The low 7 bits of the packed byte are the original value.
    unpacked[i] = packed[i] & 0x7F;
    // The high bit of the packed byte is the i-th bit of the 8th value.
    uint8_t u7_bit = packed[i] >> 7;
    unpacked[7] |= (u7_bit << i);
  }
}

/**
 * @brief Packs 64 bytes (each a 7-bit value) into 56 bytes.
 * @param packed Pointer to the destination memory (56 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint7_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint7_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  pack_transpose_uint7_values<8>(packed, unpacked);
}

/**
 * @brief Unpacks 56 bytes into 64 bytes (each a 7-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (56 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint7_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint7_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpack_transpose_uint7_values<8>(unpacked, packed);
}

/**
 * @brief Packs 128 bytes (each a 7-bit value) into 112 bytes.
 * @param packed Pointer to the destination memory (112 bytes).
 * @param unpacked Pointer to the source memory (128 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_128_uint7_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_128_uint7_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  pack_transpose_uint7_values<16>(packed, unpacked);
}

/**
 * @brief Unpacks 112 bytes into 128 bytes (each a 7-bit value).
 * @param unpacked Pointer to the destination memory (128 bytes).
 * @param packed Pointer to the source memory (112 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_128_uint7_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_128_uint7_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpack_transpose_uint7_values<16>(unpacked, packed);
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
