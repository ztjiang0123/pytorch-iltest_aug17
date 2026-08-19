// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint1.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint2.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint3.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint4.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint5.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint6.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint7.h>
#include <cassert>
#include <cstring>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {

/**
 * @brief Applies a fixed-size bitpacking primitive across a strided sequence.
 *
 * A count-N pack/unpack of `nbit` values decomposes into `num_blocks`
 * back-to-back invocations of a primitive that handles `unpacked_block` values
 * (consuming/producing `packed_block` packed bytes). This driver walks both the
 * packed and unpacked buffers in lockstep so the offset arithmetic lives in one
 * place instead of being spelled out per (count, nbit) case.
 *
 * @tparam num_blocks Number of primitive invocations.
 * @tparam unpacked_block Number of unpacked (uint8_t) values per invocation.
 * @tparam packed_block Number of packed bytes per invocation.
 * @param prim Primitive with signature void(uint8_t* dst, const uint8_t* src).
 * @param dst Destination base pointer.
 * @param src Source base pointer.
 * @param dst_is_packed Whether `dst` advances by `packed_block` (true, pack)
 *   or by `unpacked_block` (false, unpack); `src` advances by the other.
 */
template <int num_blocks, int unpacked_block, int packed_block, typename Prim>
inline void apply_bitpack_blocks(
    Prim&& prim,
    uint8_t* dst,
    const uint8_t* src,
    bool dst_is_packed) {
  const int dst_stride = dst_is_packed ? packed_block : unpacked_block;
  const int src_stride = dst_is_packed ? unpacked_block : packed_block;
  for (int b = 0; b < num_blocks; ++b) {
    prim(dst + b * dst_stride, src + b * src_stride);
  }
}

// Direction of a bitpacking transform. `kPack` maps `count` unpacked uint8
// values to packed nbit bytes; `kUnpack` is the inverse.
enum class BitpackDir { kPack, kUnpack };

/**
 * @brief Shared (count, nbit) dispatch skeleton for packing and unpacking.
 *
 * Packing and unpacking walk the exact same decision tree: dispatch on
 * (count, nbit) and either call a whole-count primitive directly or drive a
 * fixed-size primitive over `apply_bitpack_blocks`. Only two things differ
 * between the two directions: which primitive symbol is invoked (the `pack_*`
 * vs `unpack_*` family) and whether `dst` is the packed buffer. Both of those
 * are captured by the two callables and the `BitpackDir` tag, so the skeleton
 * lives here once instead of being spelled out twice.
 *
 * @tparam nbit  Bits per value (1-8).
 * @tparam count Number of unpacked values (128, 64, or 32).
 * @tparam Dir   BitpackDir::kPack or BitpackDir::kUnpack.
 * @param direct    Whole-count primitive: void(uint8_t* dst, const uint8_t*
 *   src) — e.g. pack_64_uint2_values / unpack_64_uint2_values.
 * @param block     Fixed-size primitive used with apply_bitpack_blocks, same
 *   signature as `direct`.
 * @param dst       Destination base pointer (packed for kPack, unpacked for
 *   kUnpack).
 * @param src       Source base pointer.
 *
 * The `direct`/`block` callables are the direction-specific primitive families;
 * within a single instantiation only the primitives named in the taken branch
 * are odr-used, so unused families need not exist for that (count, nbit).
 */
template <
    int nbit,
    int count,
    BitpackDir Dir,
    typename DirectFn,
    typename BlockFn>
inline void bitpack_uint_values_impl(
    DirectFn direct,
    BlockFn block,
    uint8_t* dst,
    const uint8_t* src) {
  static_assert(nbit >= 1 && nbit <= 8, "nbit must be between 1 and 8");
  static_assert(
      count == 128 || count == 64 || count == 32,
      "count must be 128, 64, or 32");
  constexpr bool dst_is_packed = (Dir == BitpackDir::kPack);
  (void)direct;
  (void)block;

  if constexpr (nbit == 8) {
    std::memcpy(dst, src, count);
  } else if constexpr (count == 128) {
    if constexpr (nbit == 1 || nbit == 3 || nbit == 5 || nbit == 7) {
      direct(dst, src);
    } else if constexpr (nbit == 2) {
      apply_bitpack_blocks<2, 64, 16>(block, dst, src, dst_is_packed);
    } else if constexpr (nbit == 4) {
      apply_bitpack_blocks<4, 32, 16>(block, dst, src, dst_is_packed);
    } else if constexpr (nbit == 6) {
      apply_bitpack_blocks<2, 64, 48>(block, dst, src, dst_is_packed);
    }
  } else if constexpr (count == 64) {
    if constexpr (nbit == 4) {
      apply_bitpack_blocks<2, 32, 16>(block, dst, src, dst_is_packed);
    } else {
      direct(dst, src);
    }
  } else { // count == 32
    if constexpr (nbit == 3) {
      apply_bitpack_blocks<4, 8, 3>(block, dst, src, dst_is_packed);
    } else if constexpr (nbit == 5) {
      apply_bitpack_blocks<4, 8, 5>(block, dst, src, dst_is_packed);
    } else if constexpr (nbit == 7) {
      apply_bitpack_blocks<4, 8, 7>(block, dst, src, dst_is_packed);
    } else {
      direct(dst, src);
    }
  }
}

/**
 * @brief Packs `count` unsigned 8-bit integers into a packed 'nbit' format.
 *
 * Shared by the 128/64/32 pack helpers below; selects the pack primitive
 * families and drives the common dispatch in `bitpack_uint_values_impl`.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @tparam count The number of values to pack (128, 64, or 32).
 */
template <int nbit, int count>
inline void pack_uint_values_impl(
    uint8_t* packed,
    const uint8_t* unpacked_values) {
  // `direct` is the whole-count primitive for this (count, nbit); `block` is
  // the fixed-size primitive used when the count decomposes into blocks. Only
  // the one selected by the taken constexpr branch is odr-used per
  // instantiation.
  auto direct = [](uint8_t* dst, const uint8_t* src) {
    if constexpr (count == 128) {
      if constexpr (nbit == 1) {
        pack_128_uint1_values(dst, src);
      } else if constexpr (nbit == 3) {
        pack_128_uint3_values(dst, src);
      } else if constexpr (nbit == 5) {
        pack_128_uint5_values(dst, src);
      } else if constexpr (nbit == 7) {
        pack_128_uint7_values(dst, src);
      }
    } else if constexpr (count == 64) {
      if constexpr (nbit == 1) {
        pack_64_uint1_values(dst, src);
      } else if constexpr (nbit == 2) {
        pack_64_uint2_values(dst, src);
      } else if constexpr (nbit == 3) {
        pack_64_uint3_values(dst, src);
      } else if constexpr (nbit == 5) {
        pack_64_uint5_values(dst, src);
      } else if constexpr (nbit == 6) {
        pack_64_uint6_values(dst, src);
      } else if constexpr (nbit == 7) {
        pack_64_uint7_values(dst, src);
      }
    } else { // count == 32
      if constexpr (nbit == 1) {
        pack_32_uint1_values(dst, src);
      } else if constexpr (nbit == 2) {
        pack_32_uint2_values(dst, src);
      } else if constexpr (nbit == 4) {
        pack_32_uint4_values(dst, src);
      } else if constexpr (nbit == 6) {
        pack_32_uint6_values(dst, src);
      }
    }
  };
  auto block = [](uint8_t* dst, const uint8_t* src) {
    if constexpr (nbit == 2) {
      pack_64_uint2_values(dst, src);
    } else if constexpr (nbit == 4) {
      pack_32_uint4_values(dst, src);
    } else if constexpr (nbit == 6) {
      pack_64_uint6_values(dst, src);
    } else if constexpr (nbit == 3) {
      pack_8_uint3_values(dst, src);
    } else if constexpr (nbit == 5) {
      pack_8_uint5_values(dst, src);
    } else if constexpr (nbit == 7) {
      pack_8_uint7_values(dst, src);
    }
  };
  bitpack_uint_values_impl<nbit, count, BitpackDir::kPack>(
      direct, block, packed, unpacked_values);
}

/**
 * @brief Unpacks 'nbit' data into `count` unsigned 8-bit integers.
 *
 * Shared by the 128/64/32 unpack helpers below; selects the unpack primitive
 * families and drives the common dispatch in `bitpack_uint_values_impl`.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @tparam count The number of values to unpack (128, 64, or 32).
 */
template <int nbit, int count>
inline void unpack_uint_values_impl(
    uint8_t* unpacked_values,
    const uint8_t* packed) {
  // `direct`/`block` mirror the pack side, selecting the unpack_* primitive
  // families. `dst` is the unpacked buffer, `src` the packed buffer.
  auto direct = [](uint8_t* dst, const uint8_t* src) {
    if constexpr (count == 128) {
      if constexpr (nbit == 1) {
        unpack_128_uint1_values(dst, src);
      } else if constexpr (nbit == 3) {
        unpack_128_uint3_values(dst, src);
      } else if constexpr (nbit == 5) {
        unpack_128_uint5_values(dst, src);
      } else if constexpr (nbit == 7) {
        unpack_128_uint7_values(dst, src);
      }
    } else if constexpr (count == 64) {
      if constexpr (nbit == 1) {
        unpack_64_uint1_values(dst, src);
      } else if constexpr (nbit == 2) {
        unpack_64_uint2_values(dst, src);
      } else if constexpr (nbit == 3) {
        unpack_64_uint3_values(dst, src);
      } else if constexpr (nbit == 5) {
        unpack_64_uint5_values(dst, src);
      } else if constexpr (nbit == 6) {
        unpack_64_uint6_values(dst, src);
      } else if constexpr (nbit == 7) {
        unpack_64_uint7_values(dst, src);
      }
    } else { // count == 32
      if constexpr (nbit == 1) {
        unpack_32_uint1_values(dst, src);
      } else if constexpr (nbit == 2) {
        unpack_32_uint2_values(dst, src);
      } else if constexpr (nbit == 4) {
        unpack_32_uint4_values(dst, src);
      } else if constexpr (nbit == 6) {
        unpack_32_uint6_values(dst, src);
      }
    }
  };
  auto block = [](uint8_t* dst, const uint8_t* src) {
    if constexpr (nbit == 2) {
      unpack_64_uint2_values(dst, src);
    } else if constexpr (nbit == 4) {
      unpack_32_uint4_values(dst, src);
    } else if constexpr (nbit == 6) {
      unpack_64_uint6_values(dst, src);
    } else if constexpr (nbit == 3) {
      unpack_8_uint3_values(dst, src);
    } else if constexpr (nbit == 5) {
      unpack_8_uint5_values(dst, src);
    } else if constexpr (nbit == 7) {
      unpack_8_uint7_values(dst, src);
    }
  };
  bitpack_uint_values_impl<nbit, count, BitpackDir::kUnpack>(
      direct, block, unpacked_values, packed);
}

/**
 * @brief Packs 128 unsigned 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory for the packed data.
 * @param unpacked_values Pointer to the source memory with 128 uint8_t values.
 */
template <int nbit>
inline void pack_128_uint_values(
    uint8_t* packed,
    const uint8_t* unpacked_values) {
  pack_uint_values_impl<nbit, 128>(packed, unpacked_values);
}

/**
 * @brief Packs 64 unsigned 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory for the packed data.
 * @param unpacked_values Pointer to the source memory with 64 uint8_t values.
 */
template <int nbit>
inline void pack_64_uint_values(
    uint8_t* packed,
    const uint8_t* unpacked_values) {
  pack_uint_values_impl<nbit, 64>(packed, unpacked_values);
}

/**
 * @brief Packs 32 unsigned 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory for the packed data.
 * @param unpacked_values Pointer to the source memory with 32 uint8_t values.
 */
template <int nbit>
inline void pack_32_uint_values(
    uint8_t* packed,
    const uint8_t* unpacked_values) {
  pack_uint_values_impl<nbit, 32>(packed, unpacked_values);
}

/**
 * @brief Unpacks 'nbit' data into 128 unsigned 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked_values Pointer to the destination memory (128 uint8_t
 * values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_128_uint_values(
    uint8_t* unpacked_values,
    const uint8_t* packed) {
  unpack_uint_values_impl<nbit, 128>(unpacked_values, packed);
}

/**
 * @brief Packs `count` signed 8-bit integers into a packed format of 'nbit'
 * bits.
 *
 * Converts the signed input to unsigned (offset by 2^(nbit-1) for nbit < 8),
 * then delegates to the generalized uint packing function. This single
 * implementation backs the 128/64/32 signed pack helpers below.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @tparam count The number of values to pack (128, 64, or 32).
 */
template <int nbit, int count>
inline void pack_lowbit_int_values(uint8_t* packed, const int8_t* unpacked) {
  // 1. Convert signed input to a temporary buffer of unsigned values.
  uint8_t temp_unpacked[count];
  if constexpr (nbit < 8) {
    const int8_t shift = 1 << (nbit - 1);
    for (int i = 0; i < count; ++i) {
      temp_unpacked[i] = static_cast<uint8_t>(unpacked[i] + shift);
    }
  } else { // nbit == 8
    for (int i = 0; i < count; ++i) {
      temp_unpacked[i] = static_cast<uint8_t>(unpacked[i]);
    }
  }

  // 2. Call the generalized uint packing function.
  pack_uint_values_impl<nbit, count>(packed, temp_unpacked);
}

/**
 * @brief Unpacks 'nbit' data into `count` signed 8-bit integers.
 *
 * Delegates to the generalized uint unpacking function, then applies the
 * signed conversion (offset by -2^(nbit-1) for nbit < 8). This single
 * implementation backs the 128/64/32 signed unpack helpers below.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @tparam count The number of values to unpack (128, 64, or 32).
 */
template <int nbit, int count>
inline void unpack_lowbit_int_values(int8_t* unpacked, const uint8_t* packed) {
  // 1. Get the raw unsigned values by calling the base function.
  uint8_t temp_unpacked[count];
  unpack_uint_values_impl<nbit, count>(temp_unpacked, packed);

  // 2. Perform the signed conversion.
  if constexpr (nbit < 8) {
    const int8_t unshift = -(1 << (nbit - 1));
    for (int i = 0; i < count; ++i) {
      unpacked[i] = static_cast<int8_t>(temp_unpacked[i]) + unshift;
    }
  } else { // nbit == 8
    for (int i = 0; i < count; ++i) {
      unpacked[i] = static_cast<int8_t>(temp_unpacked[i]);
    }
  }
}

/**
 * @brief Packs 128 signed 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory.
 * @param unpacked Pointer to the source memory containing 128 int8_t values.
 */
template <int nbit>
inline void pack_128_lowbit_int_values(
    uint8_t* packed,
    const int8_t* unpacked) {
  pack_lowbit_int_values<nbit, 128>(packed, unpacked);
}

template <int nbit>
inline void unpack_128_lowbit_int_values(
    int8_t* unpacked,
    const uint8_t* packed) {
  unpack_lowbit_int_values<nbit, 128>(unpacked, packed);
}

/**
 * @brief Packs 64 signed 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory.
 * @param unpacked Pointer to the source memory containing 64 int8_t values.
 */
template <int nbit>
inline void pack_64_lowbit_int_values(
    uint8_t* packed,
    const int8_t* unpacked) {
  pack_lowbit_int_values<nbit, 64>(packed, unpacked);
}

/**
 * @brief Packs 32 signed 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory.
 * @param unpacked Pointer to the source memory containing 32 int8_t values.
 */
template <int nbit>
inline void pack_32_lowbit_int_values(
    uint8_t* packed,
    const int8_t* unpacked) {
  pack_lowbit_int_values<nbit, 32>(packed, unpacked);
}

/**
 * @brief Unpacks 'nbit' data into 64 unsigned 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked_values Pointer to the destination memory (64 uint8_t values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_64_uint_values(
    uint8_t* unpacked_values,
    const uint8_t* packed) {
  unpack_uint_values_impl<nbit, 64>(unpacked_values, packed);
}

/**
 * @brief Unpacks 'nbit' data into 32 unsigned 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked_values Pointer to the destination memory (32 uint8_t values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_32_uint_values(
    uint8_t* unpacked_values,
    const uint8_t* packed) {
  unpack_uint_values_impl<nbit, 32>(unpacked_values, packed);
}

/**
 * @brief Unpacks 'nbit' data into 64 signed 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked Pointer to the destination memory (64 int8_t values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_64_lowbit_int_values(
    int8_t* unpacked,
    const uint8_t* packed) {
  unpack_lowbit_int_values<nbit, 64>(unpacked, packed);
}

/**
 * @brief Unpacks 'nbit' data into 32 signed 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked Pointer to the destination memory (32 int8_t values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_32_lowbit_int_values(
    int8_t* unpacked,
    const uint8_t* packed) {
  unpack_lowbit_int_values<nbit, 32>(unpacked, packed);
}

/**
 * @brief Unpacks 'nbit' data and de-quantizes it using a lookup table (LUT).
 *
 * @tparam nbit The number of bits per value in the packed format (1-4).
 * @param unpacked Pointer to the destination memory (128 int8_t values).
 * @param packed Pointer to the source packed data.
 * @param lut Pointer to the lookup table (must have 2^nbit entries).
 */
template <int nbit>
inline void unpack_128_lowbit_values_with_lut(
    int8_t* unpacked,
    const uint8_t* packed,
    const int8_t* lut) {
  static_assert(nbit >= 1 && nbit <= 4, "LUT version only supports nbit <= 4");

  // Create a temporary buffer on the stack for the indices.
  uint8_t indices[128];

  // 1. Call the utility function to handle all the unpacking logic.
  unpack_128_uint_values<nbit>(indices, packed);

  // 2. Apply the lookup table.
  for (int i = 0; i < 128; ++i) {
    unpacked[i] = lut[indices[i]];
  }
}
} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
