// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

// The rowwise_scaled_linear_sparse_cutlass entry points for the four
// float8 input/weight dtype pairs (e4m3/e5m2 x e4m3/e5m2) share an identical
// signature and differ only in the function-name suffix. This macro emits the
// declaration for one such pair so each per-dtype header (and any aggregate
// consumer) can generate them without duplicating the signature.
#define TORCHAO_DECLARE_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS(SUFFIX) \
  namespace torchao {                                                \
  torch::stable::Tensor rowwise_scaled_linear_sparse_cutlass_##SUFFIX( \
      const torch::stable::Tensor& Xq,                               \
      const torch::stable::Tensor& X_scale,                          \
      const torch::stable::Tensor& Wq,                               \
      const torch::stable::Tensor& W_meta,                           \
      const torch::stable::Tensor& W_scale,                          \
      const std::optional<torch::stable::Tensor>& bias_opt,          \
      const std::optional<torch::headeronly::ScalarType> out_dtype_opt); \
  } // namespace torchao
