// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>

#define DEFINE_OP(weight_nbit)                                                                                                             \
  m.def(                                                                                                                                   \
      "_pack_8bit_act_" #weight_nbit                                                                                                       \
      "bit_weight(Tensor weight_qvals, Tensor weight_scales, Tensor? weight_zeros, int group_size, Tensor? bias, str? target) -> Tensor"); \
  m.def(                                                                                                                                   \
      "_linear_8bit_act_" #weight_nbit                                                                                                     \
      "bit_weight(Tensor activations, Tensor packed_weights, int group_size, int n, int k) -> Tensor");                                    \
  m.def(                                                                                                                                   \
      "_linear_8bit_act_" #weight_nbit                                                                                                     \
      "bit_weight.out(Tensor activations, Tensor packed_weights, int group_size, int n, int k, *, Tensor(a!) out) -> Tensor(a!)")

#define DEFINE_CPU_IMPL(weight_nbit)                     \
  m.impl(                                                \
      "_pack_8bit_act_" #weight_nbit "bit_weight",       \
      &pack_weights_cpu<weight_nbit>);                   \
  m.impl(                                                \
      "_linear_8bit_act_" #weight_nbit "bit_weight",     \
      &linear_cpu<weight_nbit>);                         \
  m.impl(                                                \
      "_linear_8bit_act_" #weight_nbit "bit_weight.out", \
      &linear_out_cpu<weight_nbit>)

#define DEFINE_META_IMPL(weight_nbit)                 \
  m.impl(                                             \
      "_pack_8bit_act_" #weight_nbit "bit_weight",    \
      &pack_weights_meta<weight_nbit>)

#define DEFINE_LUT_PACK_OP(weight_nbit)                                                                                                       \
  m.def(                                                                                                                                      \
      "_pack_8bit_act_" #weight_nbit                                                                                                          \
      "bit_weight_with_lut(Tensor weight_qval_ids, Tensor luts, Tensor weight_scales, int group_size, Tensor? bias, str? target) -> Tensor")

#define DEFINE_LUT_PACK_CPU_IMPL(weight_nbit)               \
  m.impl(                                                   \
      "_pack_8bit_act_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_cpu<weight_nbit>)

#define DEFINE_LUT_PACK_META_IMPL(weight_nbit)              \
  m.impl(                                                   \
      "_pack_8bit_act_" #weight_nbit "bit_weight_with_lut", \
      &pack_weights_with_lut_meta<weight_nbit>)

// Applies `macro(nbit)` for every weight bit-width supported by the linear op
// (1..8) followed by every bit-width supported by the LUT-based pack op (1..4).
// Centralizing the bit-width list here keeps the three TORCH_LIBRARY blocks
// below from repeating the same enumeration and drifting out of sync.
#define FOR_EACH_WEIGHT_NBIT(macro, lut_macro) \
  macro(1);                                    \
  macro(2);                                    \
  macro(3);                                    \
  macro(4);                                    \
  macro(5);                                    \
  macro(6);                                    \
  macro(7);                                    \
  macro(8);                                    \
  lut_macro(1);                                \
  lut_macro(2);                                \
  lut_macro(3);                                \
  lut_macro(4)

TORCH_LIBRARY_FRAGMENT(torchao, m) {
  FOR_EACH_WEIGHT_NBIT(DEFINE_OP, DEFINE_LUT_PACK_OP);
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  FOR_EACH_WEIGHT_NBIT(DEFINE_CPU_IMPL, DEFINE_LUT_PACK_CPU_IMPL);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  FOR_EACH_WEIGHT_NBIT(DEFINE_META_IMPL, DEFINE_LUT_PACK_META_IMPL);
}
