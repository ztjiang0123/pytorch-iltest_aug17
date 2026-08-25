# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    can_use_deepgemm_grouped_training,
    deepgemm_blockwise_scaled_grouped_mm,
    deepgemm_blockwise_scaled_grouped_mm_wgrad,
    prepare_deepgemm_wgrad_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    DeepGemmGroupedOffsetPlan,
    build_deepgemm_grouped_offset_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
    triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm,
)
from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
    triton_fp8_blockwise_weight_quant_grouped_rhs,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.quantization.quantize_.common import KernelPreference


class _GroupedMMBackendKind(str, Enum):
    """Grouped GEMM backend selected for the FP8 MoE training op."""

    DEEPGEMM = "deepgemm"
    EMULATED = "emulated"


class _RhsQuantDirection(str, Enum):
    """Which grouped GEMM the RHS operand is being quantized for.

    ``FORWARD`` is the forward ``A @ B_t`` GEMM; ``DGRAD`` is the
    ``grad_output @ weight`` GEMM. Forward and dgrad expect transposed RHS
    layouts, so a backend picks the matching quantizer per direction.
    """

    FORWARD = "forward"
    DGRAD = "dgrad"


# Signature shared by every blockwise weight-quantizer used for an RHS operand.
_RhsQuantizer = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class _GroupedMMBackend:
    """Backend-specific quantization and grouped GEMM implementation."""

    kind: _GroupedMMBackendKind

    # Maps each grouped-GEMM direction to the backend's RHS weight quantizer.
    # Subclasses populate this; the shared ``_quantize_rhs`` dispatch below
    # picks the matching quantizer, so backends only declare *which* kernels to
    # use, not the (identical) dispatch/argument-forwarding logic.
    _rhs_quantizers: dict[_RhsQuantDirection, _RhsQuantizer] = {}

    def _quantize_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
        direction: _RhsQuantDirection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the RHS operand for ``direction``'s grouped GEMM."""
        try:
            quantizer = self._rhs_quantizers[direction]
        except KeyError:
            raise NotImplementedError(
                f"{type(self).__name__} has no RHS quantizer for {direction}"
            )
        return quantizer(B_t, block_size=block_size, dtype=dtype)

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._quantize_rhs(B_t, block_size, dtype, _RhsQuantDirection.FORWARD)

    def quantize_dgrad_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._quantize_rhs(B_t, block_size, dtype, _RhsQuantDirection.DGRAD)

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class _EmulatedGroupedMMBackend(_GroupedMMBackend):
    """TorchAO emulated backend that preserves the original grouped_mm path."""

    kind = _GroupedMMBackendKind.EMULATED

    # FORWARD: TorchAO's grouped RHS layout, (E, K, N) data with
    #          (E, K_blocks, N_blocks) scales.
    # DGRAD:   TorchAO's grouped RHS layout for grad_output @ weight, (E, N, K)
    #          data with (E, N_blocks, K_blocks) scales.
    _rhs_quantizers = {
        _RhsQuantDirection.FORWARD: (
            triton_fp8_blockwise_weight_quant_grouped_transposed_rhs
        ),
        _RhsQuantDirection.DGRAD: triton_fp8_blockwise_weight_quant_grouped_rhs,
    }

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        return emulated_blockwise_scaled_grouped_mm(
            a,
            b,
            a_s,
            scale_recipe_a,
            b_s,
            scale_recipe_b,
            offs,
            out_dtype,
            block_size,
        )

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grad_output_t_fp8, grad_output_t_scale = (
            triton_fp8_blockwise_act_quant_transposed_lhs(
                padded_grad_output.contiguous(),
                block_size=block_size,
                dtype=dtype,
            )
        )
        A_rhs_fp8, A_rhs_scale = triton_fp8_blockwise_act_quant_rhs(
            padded_a.contiguous(),
            block_size=block_size,
            dtype=dtype,
        )
        return emulated_blockwise_scaled_grouped_mm(
            grad_output_t_fp8,
            A_rhs_fp8,
            grad_output_t_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            A_rhs_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            group_end_offsets,
            out_dtype,
            block_size,
        )


@dataclass(frozen=True)
class _DeepGemmGroupedMMBackend(_GroupedMMBackend):
    """DeepGEMM backend plus the offset metadata shared by its kernels."""

    kind = _GroupedMMBackendKind.DEEPGEMM
    offset_plan: DeepGemmGroupedOffsetPlan

    # Both quantizers write DeepGEMM's expected layout directly, avoiding a
    # dispatch-time transpose/copy.
    # FORWARD: RHS as (E, N, K), K contiguous, scales (E, N_blocks, K_blocks).
    # DGRAD:   RHS as (E, K, N), N contiguous, scales (E, K_blocks, N_blocks).
    _rhs_quantizers = {
        _RhsQuantDirection.FORWARD: (
            triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm
        ),
        _RhsQuantDirection.DGRAD: (
            triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm
        ),
    }

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        return deepgemm_blockwise_scaled_grouped_mm(
            a,
            b,
            a_s,
            scale_recipe_a,
            b_s,
            scale_recipe_b,
            offs,
            out_dtype,
            block_size,
            offset_plan=self.offset_plan,
        )

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        wgrad_plan = prepare_deepgemm_wgrad_plan(
            padded_grad_output,
            padded_a,
            self.offset_plan,
            block_size,
            dtype,
        )
        assert wgrad_plan is not None, (
            "DeepGEMM backend requires block-aligned group sizes for wgrad"
        )
        return deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            self.offset_plan,
            out_dtype,
            block_size,
        )


_EMULATED_GROUPED_MM_BACKEND = _EmulatedGroupedMMBackend()


def _select_fp8_blockwise_grouped_mm_backend(
    kernel_preference: KernelPreference,
    A: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    group_end_offsets: torch.Tensor,
    *,
    original_group_end_offsets: Optional[torch.Tensor] = None,
    padded_group_start_offsets: Optional[torch.Tensor] = None,
    num_rows: Optional[int] = None,
) -> _GroupedMMBackend:
    """Select the grouped GEMM backend for one forward/backward pass.

    ``KernelPreference.EMULATED`` always selects the TorchAO emulated backend.
    ``KernelPreference.AUTO`` selects DeepGEMM only when the optional dependency
    exposes both M-grouped and K-grouped training kernels, the input is on
    CUDA SM90+, ``out_dtype`` is bf16, ``block_size`` is 128, and every expert
    group is block-aligned. Any unsupported AUTO case falls back to emulated.
    When DeepGEMM is selected, the returned backend owns the offset/layout plan
    reused by forward, dgrad, and wgrad.
    """

    if kernel_preference == KernelPreference.EMULATED:
        return _EMULATED_GROUPED_MM_BACKEND

    assert kernel_preference == KernelPreference.AUTO, (
        "kernel_preference must be AUTO or EMULATED"
    )
    if not can_use_deepgemm_grouped_training(A, out_dtype, block_size):
        return _EMULATED_GROUPED_MM_BACKEND

    groups_block_aligned_by_construction = original_group_end_offsets is not None
    offset_plan = build_deepgemm_grouped_offset_plan(
        group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=num_rows,
        groups_block_aligned_by_construction=groups_block_aligned_by_construction,
    )
    if not offset_plan.groups_are_block_aligned(block_size):
        return _EMULATED_GROUPED_MM_BACKEND

    return _DeepGemmGroupedMMBackend(offset_plan=offset_plan)
