// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cstdint>

// Backend-agnostic scalar pack/unpack for 4 uint6 values <-> 3 bytes.
//
// The aarch64 (NEON) and fallback backends both need this scalar routine: the
// NEON backend uses it as the reference the vectorized paths mirror, and the
// fallback backend uses it directly. The two copies were bit-for-bit
// equivalent, so the logic lives here once and each backend's namespace
// forwards to it.
namespace torchao::uint6_scalar {

// Packs 4 uint6 values (u0..u3) into 3 bytes (p0..p2):
//   p0's low 6 bits = u0; p0's high 2 bits = u3's low 2 bits
//   p1's low 6 bits = u1; p1's high 2 bits = u3's mid 2 bits
//   p2's low 6 bits = u2; p2's high 2 bits = u3's high 2 bits
TORCHAO_ALWAYS_INLINE inline void pack_4_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  const uint8_t u3 = unpacked[3] & 0x3F;
  packed[0] = (unpacked[0] & 0x3F) | ((u3 & 0x03) << 6);
  packed[1] = (unpacked[1] & 0x3F) | ((u3 & 0x0C) << 4);
  packed[2] = (unpacked[2] & 0x3F) | ((u3 & 0x30) << 2);
}

// Inverse of pack_4_uint6_values: unpacks 3 bytes into 4 uint6 values.
TORCHAO_ALWAYS_INLINE inline void unpack_4_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpacked[0] = packed[0] & 0x3F;
  unpacked[1] = packed[1] & 0x3F;
  unpacked[2] = packed[2] & 0x3F;
  // Last value is packed in the upper 2 bits of the three bytes
  unpacked[3] = ((packed[0] & 0xC0) >> 6) | ((packed[1] & 0xC0) >> 4) |
      ((packed[2] & 0xC0) >> 2);
}

} // namespace torchao::uint6_scalar
