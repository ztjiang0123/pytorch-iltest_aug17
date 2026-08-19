// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/shared/uint6_scalar.h>
#include <cstdint>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {

/**
 * @brief Shared "unpack-and-transpose" primitive for the 6-bit format.
 *
 * The count-32 and count-64 unpack loops are byte-for-byte identical apart from
 * the block size, so the loop lives here once, parameterized by `block` (the
 * packed row stride). The packed data is three rows of `block` bytes; each byte
 * carries a 6-bit value in its low bits plus a 2-bit slice of a fourth value in
 * its high bits, reassembled into the fourth output row.
 *
 * @tparam block Number of columns (== packed row stride).
 */
template <int block>
TORCHAO_ALWAYS_INLINE inline void unpack_transpose_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < block; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + block];
    const uint8_t p2 = packed[i + 2 * block];

    unpacked[i] = p0 & 0x3F;
    unpacked[i + block] = p1 & 0x3F;
    unpacked[i + 2 * block] = p2 & 0x3F;
    unpacked[i + 3 * block] =
        ((p0 & 0xC0) >> 6) | ((p1 & 0xC0) >> 4) | ((p2 & 0xC0) >> 2);
  }
}

/**
 * @brief Packs 4 bytes, each holding a 6-bit value (0-63), into 3 bytes.
 *
 * @param packed Pointer to the destination memory (3 bytes).
 * @param unpacked Pointer to the source memory (4 bytes).
 *
 * The scalar logic is backend-independent and shared with the NEON backend.
 */
using torchao::uint6_scalar::pack_4_uint6_values;

/**
 * @brief Unpacks 3 bytes into 4 bytes, each containing a 6-bit value.
 *
 * @param unpacked Pointer to the destination memory (4 bytes).
 * @param packed Pointer to the source memory (3 bytes).
 *
 * Compatible with the scalar NEON version; shared scalar logic.
 */
using torchao::uint6_scalar::unpack_4_uint6_values;

/**
 * @brief Packs 32 bytes (each a 6-bit value) into 24 bytes.
 * @param packed Pointer to the destination memory (24 bytes).
 * @param unpacked Pointer to the source memory (32 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_32_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_32_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t u0 = unpacked[i];
    const uint8_t u1 = unpacked[i + 8];
    const uint8_t u2 = unpacked[i + 16];
    const uint8_t u3 = unpacked[i + 24];

    packed[i] = (u0 & 0x3F) | ((u3 & 0x03) << 6);
    packed[i + 8] = (u1 & 0x3F) | ((u3 & 0x0C) << 4);
    packed[i + 16] = (u2 & 0x3F) | ((u3 & 0x30) << 2);
  }
}

/**
 * @brief Unpacks 24 bytes into 32 bytes (each a 6-bit value).
 * @param unpacked Pointer to the destination memory (32 bytes).
 * @param packed Pointer to the source memory (24 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_32_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_32_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpack_transpose_uint6_values<8>(unpacked, packed);
}

/**
 * @brief Packs 64 bytes (each a 6-bit value) into 48 bytes.
 * @param packed Pointer to the destination memory (48 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t u0 = unpacked[i];
    const uint8_t u1 = unpacked[i + 16];
    const uint8_t u2 = unpacked[i + 32];
    const uint8_t u3 = unpacked[i + 48];

    // Note: u0, u1, u2 are not masked - ARM stores raw value, unpacking extracts low 6 bits
    packed[i] = u0 | ((u3 & 0x03) << 6);
    packed[i + 16] = u1 | ((u3 & 0x0C) << 4);
    packed[i + 32] = u2 | ((u3 & 0x30) << 2);
  }
}

/**
 * @brief Unpacks 48 bytes into 64 bytes (each a 6-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (48 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpack_transpose_uint6_values<16>(unpacked, packed);
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
