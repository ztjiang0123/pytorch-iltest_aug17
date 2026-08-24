# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any

import torch

from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _fake_quantize_affine,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
)


class _FakeQuantizedLinearBase(torch.nn.Linear):
    """
    Shared base for fake quantized linear modules that require both an
    activation and a weight config (e.g. ``MXFakeQuantizedLinear``,
    ``NVFP4FakeQuantizedLinear``).

    Subclasses set ``_activation_required_msg`` to the format-specific error
    raised when only weight quantization is provided. The constructor
    initializes the underlying ``nn.Linear``, validates that both configs are
    present, and stores them on the instance.
    """

    # Format-specific error raised when only a weight config is provided.
    _activation_required_msg: str = "Weight only QAT not supported yet"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Any = None,
        weight_config: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            *args,
            **kwargs,
        )
        if weight_config is None:
            raise ValueError("Must specify `weight_config`")
        if activation_config is None:
            raise ValueError(self._activation_required_msg)
        self.activation_config = activation_config
        self.weight_config = weight_config


def _copy_fake_quantized_weights(
    src: torch.nn.Linear,
    dst: torch.nn.Linear,
) -> None:
    """
    Copy the weight and bias from ``src`` to ``dst`` in place.

    In distributed training, the model may be instantiated on the meta
    device, in which case there is no need to copy the weights, and doing
    so will result in an error.
    """
    if src.weight.device != torch.device("meta"):
        dst.weight = src.weight
        dst.bias = src.bias


def _fake_quantized_linear_to_linear(mod: torch.nn.Linear) -> torch.nn.Linear:
    """
    Build a plain ``torch.nn.Linear`` mirroring the shape, device, and dtype
    of a fake quantized linear module, copying over its weight and bias.

    Shared by the ``to_linear`` conversion of the various fake quantized
    linear modules (e.g. ``FakeQuantizedLinear``, ``MXFakeQuantizedLinear``,
    ``NVFP4FakeQuantizedLinear``).
    """
    new_linear = torch.nn.Linear(
        mod.in_features,
        mod.out_features,
        mod.bias is not None,
        device=mod.weight.device,
        dtype=mod.weight.dtype,
    )
    _copy_fake_quantized_weights(mod, new_linear)
    return new_linear


def _linear_to_fake_quantized_linear(
    cls: type,
    mod: torch.nn.Linear,
    activation_config: Any = None,
    weight_config: Any = None,
) -> torch.nn.Linear:
    """
    Build a fake quantized linear of type ``cls`` mirroring the shape, device,
    and dtype of ``mod``, copying over its weight and bias.

    Shared by the ``from_linear`` conversion of the various fake quantized
    linear modules (e.g. ``FakeQuantizedLinear``, ``MXFakeQuantizedLinear``,
    ``NVFP4FakeQuantizedLinear``).
    """
    new_linear = cls(
        mod.in_features,
        mod.out_features,
        mod.bias is not None,
        activation_config=activation_config,
        weight_config=weight_config,
        device=mod.weight.device,
        dtype=mod.weight.dtype,
    )
    _copy_fake_quantized_weights(mod, new_linear)
    return new_linear


def _fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    block_size = (1, group_size)
    return _fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
        zero_point_domain=zero_point_domain,
    )


def _fake_quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    from torch.ao.quantization.fx._decomposed import _per_token_quant_qparam_dim_check

    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    block_size = _get_per_token_block_size(input)
    fq = _fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
    )
    return fq.reshape_as(input).to(input.dtype)


def _get_qmin_qmax(n_bit: int, symmetric: bool = True):
    if symmetric:
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
    else:
        qmin = 0
        qmax = 2**n_bit - 1
    return (qmin, qmax)


def _log_deprecation_warning(old_api_object: Any):
    """
    Log a helpful deprecation message pointing users to the new QAT API,
    only once per deprecated class.
    """
    warnings.warn(
        """'%s' is deprecated and will be removed in a future release. Please use the following API instead:

    base_config = Int4WeightOnlyConfig(group_size=32)
    quantize_(model, QATConfig(base_config, step="prepare"))
    # train (not shown)
    quantize_(model, QATConfig(base_config, step="convert"))

Alternatively, if you prefer to pass in fake quantization configs:

    activation_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
    weight_config = IntxFakeQuantizeConfig(torch.int4, group_size=32)
    qat_config = QATConfig(
        activation_config=activation_config,
        weight_config=weight_config,
        step="prepare",
    )
    quantize_(model, qat_config)

Please see https://github.com/pytorch/ao/issues/2630 for more details.
        """
        % old_api_object.__class__.__name__
    )
