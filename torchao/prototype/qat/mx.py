# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MX (Microscaling) Quantization-Aware Training (QAT) support.

This module provides QAT support for the OCP Microscaling MX formats (MXFP4, MXFP8).

Key differences between MX and NVFP4:
- Block size: MX uses 32 (default), NVFP4 uses 16 (fixed)
- Scale type: MX uses E8M0 (float8_e8m0fnu), NVFP4 uses float8_e4m3fn
- NVFP4 performs an extra per-tensor scaling, while MX does not
- Scale calculation: MX supports FLOOR, RCEIL, CEIL, EVEN modes
- MX supports multiple element dtypes:
  - MXFP4: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+)
  - MXFP8: torch.float8_e4m3fn, torch.float8_e5m2
"""

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.config import (
    ScaleCalculationMode,
    _validate_elem_dtype,
    _validate_kernel_preference,
)
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    _addmm_mx_dispatch,
)
from torchao.quantization.qat import FakeQuantizeConfigBase
from torchao.quantization.qat.utils import (
    _fake_quantized_linear_to_linear,
    _FakeQuantizedLinearBase,
    _linear_to_fake_quantized_linear,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

_DEFAULT_MX_DTYPE = torch.float4_e2m1fn_x2


@dataclass
class MXFakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to the OCP Microscaling MX format
    according to https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Fake quantization numerics follow `MXTensor` closely:
    https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py.

    Supported element dtypes:
    - MXFP4: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+)
    - MXFP8: torch.float8_e4m3fn, torch.float8_e5m2

    Key differences from NVFP4:
    - Block size: 32 (default) vs NVFP4's fixed 16
    - Scale type: E8M0 (float8_e8m0fnu) vs NVFP4's float8_e4m3fn
    - NVFP4 performs an extra per-tensor scaling, while MX does not
    - Supports multiple scale calculation modes (FLOOR, RCEIL, CEIL, EVEN)

    Args:
        dtype (torch.dtype): The element dtype for quantization.
            Supported values: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+),
            torch.float8_e4m3fn, torch.float8_e5m2.
            Default is float4_e2m1fn_x2 on PyTorch 2.8+, float8_e4m3fn otherwise.
        block_size (int): The block size for quantization (default 32, the OCP MX standard)
        scaling_mode (ScaleCalculationMode): How to calculate the block scales (default RCEIL)
        kernel_preference (KernelPreference): Which kernel to use for matmul (default EMULATED)
    """

    dtype: torch.dtype = _DEFAULT_MX_DTYPE
    block_size: int = 32
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL
    kernel_preference: KernelPreference = KernelPreference.EMULATED

    def __post_init__(self):
        _validate_elem_dtype(self.dtype)
        _validate_kernel_preference(self.kernel_preference, self.block_size, self.dtype)


class _MXQuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    """
    Autograd function for MX quantization + addmm in low precision during forward,
    and fake quantization in high precision during backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_config: MXFakeQuantizeConfig,
        weight_config: MXFakeQuantizeConfig,
    ) -> torch.Tensor:
        # Handle inputs of any rank by reshaping to 2D
        orig_shape = _input.shape
        _input_2d = _input.view(-1, orig_shape[-1])

        # quantize input activations
        _input_2d = MXTensor.to_mx(
            _input_2d,
            elem_dtype=activation_config.dtype,
            block_size=activation_config.block_size,
            scaling_mode=activation_config.scaling_mode,
            kernel_preference=activation_config.kernel_preference,
        )

        weight = MXTensor.to_mx(
            weight,
            elem_dtype=weight_config.dtype,
            block_size=weight_config.block_size,
            scaling_mode=weight_config.scaling_mode,
            kernel_preference=weight_config.kernel_preference,
        )

        ctx.save_for_backward(_input_2d, weight)
        ctx.orig_shape = orig_shape

        # Use addmm when bias is present, mm otherwise
        if bias is not None:
            aten_op = torch.ops.aten.addmm.default
        else:
            aten_op = torch.ops.aten.mm.default

        out = _addmm_mx_dispatch(
            _input_2d,
            weight.t(),
            aten_op,
            bias,
        )

        # Reshape output back to original shape (with last dim changed)
        out_shape = (*orig_shape[:-1], out.shape[-1])
        return out.view(*out_shape)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        _input_2d, weight = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        assert isinstance(_input_2d, MXTensor)
        assert isinstance(weight, MXTensor)
        _input_2d = _input_2d.dequantize(_input_2d.orig_dtype)
        weight = weight.dequantize(weight.orig_dtype)

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        grad_input_2d = torch.mm(grad_output_2d, weight)
        grad_weight = torch.mm(grad_output_2d.t(), _input_2d)

        grad_input = grad_input_2d.view(*orig_shape)
        return grad_input, grad_weight, None, None, None


class MXFakeQuantizedLinear(_FakeQuantizedLinearBase):
    """
    Linear module for fake quantized MX weights and/or activations.

    The forward pass follows quantization and addmm numerics in `MXTensor`
    in lower precision exactly, while the backward pass uses dequantized
    (fake quantized) values in high precision.


    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        from torchao.quantization.qat import QATConfig

        base_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
        )
        quantize_(model, QATConfig(base_config, step="prepare"))
        # Model contains `MXFakeQuantizedLinear` now

        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))
        # Model contains `nn.Linear` with `MXTensor` weights now
    """

    _activation_required_msg = "Weight only MX QAT not supported yet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fq = _MXQuantizedForwardFakeQuantizedBackward.apply(
            x, self.weight, self.bias, self.activation_config, self.weight_config
        )
        assert fq.dtype == x.dtype
        return fq

    def to_linear(self) -> torch.nn.Linear:
        return _fake_quantized_linear_to_linear(self)

    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: Optional[MXFakeQuantizeConfig] = None,
        weight_config: Optional[MXFakeQuantizeConfig] = None,
    ):
        return _linear_to_fake_quantized_linear(
            cls,
            mod,
            activation_config=activation_config,
            weight_config=weight_config,
        )
