# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

import torch

from torchao.float8.float8_training_tensor import GemmInputRole, LinearMMConfig
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
    hp_to_fp8_col_major,
    hp_to_fp8_col_major_t,
    hp_to_fp8_row_and_col_major,
    hp_to_fp8_row_major,
    hp_to_fp8_row_major_t,
    hp_to_fp8_row_major_t_and_non_t,
)


def _make_to_fp8_autograd_function(name, kernel, doc):
    """Build a ``torch.autograd.Function`` that converts a high-precision tensor
    to fp8 via ``kernel`` in the forward pass and passes the gradient through
    unchanged in the backward pass.

    Every ``ToFP8*`` conversion below shares this exact shape and differs only in
    the underlying ``hp_to_fp8_*`` kernel, so the shared forward/backward logic
    lives here in one place instead of being copied per class.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        return kernel(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None

    return type(
        name,
        (torch.autograd.Function,),
        {"__doc__": doc, "forward": forward, "backward": backward},
    )


ToFP8RowAndColumnMajor = _make_to_fp8_autograd_function(
    "ToFP8RowAndColumnMajor",
    hp_to_fp8_row_and_col_major,
    """
    A differentiable conversion to fp8.
    * forward: convert from high precision to float8 and produces both row-major and column-major outputs
    * backward: pass the gradient without changes
    """,
)


ToFP8RowMajor = _make_to_fp8_autograd_function(
    "ToFP8RowMajor",
    hp_to_fp8_row_major,
    """
    A differentiable conversion to fp8 in row-major layout.
    * forward: convert from high precision to float8 with row-major memory layout
    * backward: pass the gradient without changes
    """,
)


ToFP8RowMajorT = _make_to_fp8_autograd_function(
    "ToFP8RowMajorT",
    hp_to_fp8_row_major_t,
    """
    A differentiable conversion to fp8 with transposed dimensions in row-major layout.
    * forward: convert from high precision to float8 with transposed dimensions with row-major memory layout
    * backward: pass the gradient without changes
    """,
)


ToFP8ColumnMajor = _make_to_fp8_autograd_function(
    "ToFP8ColumnMajor",
    hp_to_fp8_col_major,
    """
    A differentiable conversion to fp8 in column-major layout.
    * forward: convert from high precision to float8 with column-major memory layout
    * backward: pass the gradient without changes
    """,
)


ToFP8ColumnMajorT = _make_to_fp8_autograd_function(
    "ToFP8ColumnMajorT",
    hp_to_fp8_col_major_t,
    """
    A differentiable conversion to fp8 with transposed dimensions in column-major layout.
    * forward: convert from high precision to float8 with transposed dimensions in column-major memory layout.
    * backward: pass the gradient without changes
    """,
)


ToFP8RowMajorTAndNonT = _make_to_fp8_autograd_function(
    "ToFP8RowMajorTAndNonT",
    hp_to_fp8_row_major_t_and_non_t,
    """
    A differentiable conversion to fp8.
    * forward: convert from high precision to float8 and produces both row-major (transposed) and row-major (non-transposed) outputs
    * backward: pass the gradient without changes
    """,
)
