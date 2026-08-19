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

// Bit-field ordering within a packed byte for the uniform transpose helpers.
// kMsbFirst places unpacked block 0 in the most-significant field (uint1/uint2
// convention); kLsbFirst places block 0 in the least-significant field (uint4
// convention). This is the only structural difference between the otherwise
// identical uint1/uint2 and uint4 transpose loops, so it is expressed as a
// template parameter rather than a second copy of the loop.
enum class TransposeOrder { kMsbFirst, kLsbFirst };

// Shared "transpose-and-pack" primitive for uniform bit widths (1, 2, and 4
// bits per value). These widths divide a byte evenly, so a packed byte holds
// exactly `8 / nbit` values, each taken from a separate `num_values`-sized
// block of the unpacked buffer. Byte `i` gathers unpacked[i + num_values * lane]
// for lane 0..num_lanes-1; `order` selects whether lane 0 lands in the most- or
// least-significant field.
//
// This is the single source of truth for the uint1/uint2/uint4 transpose loops
// that were previously copy-pasted per (count, nbit). The count-1 primitives
// (e.g. pack_8_uint1_values) keep their own explicit form because they have no
// strided block structure to share.
//
// @tparam nbit       Bits per value; must be 1, 2, or 4.
// @tparam num_values Number of packed output bytes (== unpacked block size).
// @tparam order      Field ordering within each packed byte.
template <int nbit, int num_values, TransposeOrder order>
TORCHAO_ALWAYS_INLINE inline void pack_transpose_uint_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  static_assert(nbit == 1 || nbit == 2 || nbit == 4, "nbit must be 1, 2, or 4");
  constexpr int num_lanes = 8 / nbit;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << nbit) - 1);
  for (int i = 0; i < num_values; ++i) {
    uint8_t byte = 0;
    for (int lane = 0; lane < num_lanes; ++lane) {
      const int field = (order == TransposeOrder::kMsbFirst)
          ? (num_lanes - 1 - lane)
          : lane;
      const int shift = field * nbit;
      byte |= static_cast<uint8_t>(
          (unpacked[i + num_values * lane] & mask) << shift);
    }
    packed[i] = byte;
  }
}

// Inverse of pack_transpose_uint_values: scatters the `8 / nbit` fields of each
// packed byte back into their respective `num_values`-sized blocks.
//
// @tparam nbit       Bits per value; must be 1, 2, or 4.
// @tparam num_values Number of packed input bytes (== unpacked block size).
// @tparam order      Field ordering within each packed byte.
template <int nbit, int num_values, TransposeOrder order>
TORCHAO_ALWAYS_INLINE inline void unpack_transpose_uint_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  static_assert(nbit == 1 || nbit == 2 || nbit == 4, "nbit must be 1, 2, or 4");
  constexpr int num_lanes = 8 / nbit;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << nbit) - 1);
  for (int i = 0; i < num_values; ++i) {
    const uint8_t packed_byte = packed[i];
    for (int lane = 0; lane < num_lanes; ++lane) {
      const int field = (order == TransposeOrder::kMsbFirst)
          ? (num_lanes - 1 - lane)
          : lane;
      const int shift = field * nbit;
      unpacked[i + num_values * lane] = (packed_byte >> shift) & mask;
    }
  }
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
