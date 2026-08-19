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
 * @brief Shared "pack" primitive for the wide 5-bit formats (count 64 and 128).
 *
 * The unpacked data is organized as `2 * groups` rows of `block` bytes. The rows
 * pair up: for each group `g`, the "low" row (2*g) supplies all 5 bits of a
 * value and the "high" row (2*g + 1) supplies its top 2 bits. Each group writes
 * one packed row `packed[g * block + i] = low | (high_low5 << 5)`; the top 2
 * bits of every high row are then bit-packed four-per-byte into a trailing
 * region of `groups * block / 4` bytes.
 *
 * The count-64 loop (groups = 2) and count-128 loop (groups = 4) are otherwise
 * identical, so the logic lives here once, parameterized by `groups` (with the
 * row width fixed at `block = 16`, matching the NEON layout).
 *
 * @tparam groups Number of value/high-bit row pairs (2 for count 64, 4 for 128).
 */
template <int groups>
TORCHAO_ALWAYS_INLINE inline void pack_wide_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  constexpr int block = 16;
  constexpr int tail_len = groups * block / 4;

  // Pack each group's low+high row pair into one packed row.
  for (int i = 0; i < block; ++i) {
    for (int g = 0; g < groups; ++g) {
      const uint8_t low = unpacked[2 * g * block + i];
      const uint8_t high = unpacked[(2 * g + 1) * block + i] & 0x1F;
      packed[g * block + i] = low | (high << 5);
    }
  }

  // Pack the top 2 bits of the `groups` high rows, four per trailing byte.
  for (int t = 0; t < tail_len; ++t) {
    uint8_t out = 0;
    for (int k = 0; k < 4; ++k) {
      const int hi_flat = k * tail_len + t;
      const int row = hi_flat / block;
      const int col = hi_flat % block;
      const uint8_t hi = unpacked[(2 * row + 1) * block + col] >> 3;
      out |= hi << (2 * k);
    }
    packed[groups * block + t] = out;
  }
}

/**
 * @brief Packs 8 bytes, each holding a 5-bit value (0-31), into 5 bytes.
 *
 * @param packed Pointer to the destination memory (5 bytes).
 * @param unpacked Pointer to the source memory (8 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_8_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // pack 8 uint5 values (u0..u7) into 5 bytes (p0..p4)
  // p0 = u0_all | u1_low_3_bits
  // p1 = u2_all | u3_low_3_bits
  // p2 = u4_all | u5_low_3_bits
  // p3 = u6_all | u7_low_3_bits
  // p4 = u1_high_2_bits | u3_high_2_bits | u5_high_2_bits | u7_high_2_bits
  packed[0] = unpacked[0] | ((unpacked[1] & 0x1F) << 5);
  packed[1] = unpacked[2] | ((unpacked[3] & 0x1F) << 5);
  packed[2] = unpacked[4] | ((unpacked[5] & 0x1F) << 5);
  packed[3] = unpacked[6] | ((unpacked[7] & 0x1F) << 5);
  packed[4] = ((unpacked[1] & 0x1F) >> 3) | (((unpacked[3] & 0x1F) >> 3) << 2) |
      (((unpacked[5] & 0x1F) >> 3) << 4) | (((unpacked[7] & 0x1F) >> 3) << 6);
}

/**
 * @brief Unpacks 5 bytes into 8 bytes, each containing a 5-bit value.
 *
 * @param unpacked Pointer to the destination memory (8 bytes).
 * @param packed Pointer to the source memory (5 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_8_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  const uint8_t p0 = packed[0];
  const uint8_t p1 = packed[1];
  const uint8_t p2 = packed[2];
  const uint8_t p3 = packed[3];
  const uint8_t p4 = packed[4];

  // This is compatible with the scalar NEON version.
  unpacked[0] = p0 & 0x1F;
  unpacked[1] = (p0 >> 5) | ((p4 & 0x03) << 3);
  unpacked[2] = p1 & 0x1F;
  unpacked[3] = (p1 >> 5) | ((p4 & 0x0C) << 1);
  unpacked[4] = p2 & 0x1F;
  unpacked[5] = (p2 >> 5) | ((p4 & 0x30) >> 1);
  unpacked[6] = p3 & 0x1F;
  unpacked[7] = (p3 >> 5) | ((p4 & 0xC0) >> 3);
}

/**
 * @brief Packs 64 bytes (each a 5-bit value) into 40 bytes.
 * @param packed Pointer to the destination memory (40 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  pack_wide_uint5_values<2>(packed, unpacked);
}

/**
 * @brief Unpacks 40 bytes into 64 bytes (each a 5-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (40 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 16];
    // p2 is only 8 bytes wide, so we use modulo to access it correctly.
    const uint8_t p2 = packed[32 + (i % 8)];

    unpacked[i] = p0 & 0x1F;
    unpacked[i + 32] = p1 & 0x1F;

    if (i < 8) {
      unpacked[i + 16] = (p0 >> 5) | ((p2 & 0x03) << 3);
      unpacked[i + 48] = (p1 >> 5) | ((p2 & 0x30) >> 1);
    } else {
      unpacked[i + 16] = (p0 >> 5) | ((p2 & 0x0C) << 1);
      unpacked[i + 48] = (p1 >> 5) | ((p2 & 0xC0) >> 3);
    }
  }
}

/**
 * @brief Packs 128 bytes (each a 5-bit value) into 80 bytes.
 * @param packed Pointer to the destination memory (80 bytes).
 * @param unpacked Pointer to the source memory (128 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_128_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_128_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  pack_wide_uint5_values<4>(packed, unpacked);
}

/**
 * @brief Unpacks 80 bytes into 128 bytes (each a 5-bit value).
 * @param unpacked Pointer to the destination memory (128 bytes).
 * @param packed Pointer to the source memory (80 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_128_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_128_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 16];
    const uint8_t p2 = packed[i + 32];
    const uint8_t p3 = packed[i + 48];
    const uint8_t p4 = packed[i + 64];

    unpacked[i + 16 * 0] = p0 & 0x1F;
    unpacked[i + 16 * 1] = (p0 >> 5) | ((p4 & 0x03) << 3);
    unpacked[i + 16 * 2] = p1 & 0x1F;
    unpacked[i + 16 * 3] = (p1 >> 5) | ((p4 & 0x0C) << 1);
    unpacked[i + 16 * 4] = p2 & 0x1F;
    unpacked[i + 16 * 5] = (p2 >> 5) | ((p4 & 0x30) >> 1);
    unpacked[i + 16 * 6] = p3 & 0x1F;
    unpacked[i + 16 * 7] = (p3 >> 5) | ((p4 & 0xC0) >> 3);
  }
}

}}
