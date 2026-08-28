# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.utils import _is_device, find_multiple

from .quant_primitives import (
    MappingType,
    dequantize_affine,
)
from .utils import (
    group_quantize_tensor_symmetric,
    groupwise_affine_quantize_tensor,
    per_token_dynamic_quant,
)

aten = torch.ops.aten

__all__ = [
    "Int4LinearConfig",
    "Int4WeightOnlyQuantizerConfig",
    "Int8DynActInt4WeightConfig",
    "Int8DynActInt4WeightQuantizerConfig",
    "WeightOnlyInt4Linear",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
]


def _quantize_via_state_dict(quantizer, model: torch.nn.Module) -> torch.nn.Module:
    """Shared quantization flow: build a quantized state dict, convert the model
    for runtime, then load the quantized weights back in.

    Both ``Int4WeightOnlyQuantizer`` and ``Int8DynActInt4WeightQuantizer`` use
    this same sequence, so it lives here to avoid duplicated implementations.
    """
    state_dict = quantizer._create_quantized_state_dict(model)
    model = quantizer._convert_for_runtime(model)
    # TODO: make it strict
    model.load_state_dict(state_dict, strict=False)
    return model


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=None):
    k_divisible_by_groupsize = k % groupsize == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_groupsize


@dataclass
class Int4LinearWeights:
    """Packed int4 weights plus the metadata needed to run the matmul.

    These values are produced and stored together by
    :class:`WeightOnlyInt4Linear`, so grouping them keeps
    :func:`linear_forward_int4` down to its input tensor and the weights it
    operates on.
    """

    weight_int4pack: torch.Tensor
    scales_and_zeros: torch.Tensor
    out_features: int
    groupsize: int
    precision: torch.dtype = torch.bfloat16
    scales_precision: torch.dtype = torch.bfloat16


def linear_forward_int4(
    x: torch.Tensor,
    weights: Int4LinearWeights,
):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    if _is_device(x.device.type, "cpu"):
        c = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x.to(weights.precision),
            weights.weight_int4pack,
            weights.groupsize,
            weights.scales_and_zeros.to(weights.scales_precision),
        ).to(dtype=x.dtype)
    else:
        c = torch.ops.aten._weight_int4pack_mm(
            x.to(weights.precision),
            weights.weight_int4pack,
            weights.groupsize,
            weights.scales_and_zeros.to(weights.scales_precision),
        ).to(dtype=x.dtype)
    new_shape = origin_x_size[:-1] + (weights.out_features,)
    c = c.reshape(new_shape)
    return c


@dataclass
class Int4LinearConfig:
    """Construction options for the int4 weight-only linear layer.

    Grouping these related values (placement plus quantization settings) keeps
    ``WeightOnlyInt4Linear``'s constructor down to the required in/out shape and
    makes the settings easy to pass around together.
    """

    bias: bool = False
    device: Optional[torch.device] = None
    groupsize: int = 128
    inner_k_tiles: int = 8
    precision: torch.dtype = torch.bfloat16
    scales_precision: torch.dtype = torch.bfloat16


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[Int4LinearConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = Int4LinearConfig()
        groupsize = config.groupsize
        inner_k_tiles = config.inner_k_tiles
        device = config.device

        self.padding = not _check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not config.bias, "require bias=False"
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = config.precision
        self.scales_precision = config.scales_precision

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, (
            "require in_features % (innerKTiles * 16) == 0"
        )
        if _is_device(device.type, "cpu"):
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features,
                        in_features // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features // 8,
                        in_features // (inner_k_tiles * 16),
                        32,
                        inner_k_tiles // 2,
                    ),
                    dtype=torch.int32,
                    device=device,
                ),
            )
        self.register_buffer(
            "scales_and_zeros",
            torch.zeros(
                (in_features // groupsize, out_features, 2),
                dtype=self.scales_precision,
                device=device,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            Int4LinearWeights(
                weight_int4pack=self.weight,
                scales_and_zeros=self.scales_and_zeros,
                out_features=self.out_features,
                groupsize=self.groupsize,
                precision=self.precision,
                scales_precision=self.scales_precision,
            ),
        )


@dataclass
class _Linear4ReplaceSettings:
    """Quantization settings that travel together when replacing linear layers.

    Grouping these related values keeps :func:`_replace_linear_int4`'s signature
    down to the module being traversed and avoids positional-argument mistakes at
    call sites.
    """

    groupsize: int
    inner_k_tiles: Optional[int]
    padding_allowed: bool
    skip_layer_func: Optional[Callable] = None
    precision: torch.dtype = torch.bfloat16
    scales_precision: torch.dtype = torch.bfloat16
    linear_class: Type[torch.nn.Module] = WeightOnlyInt4Linear
    copy_weights: bool = False


def _replace_linear_int4(
    module: torch.nn.Module,
    settings: _Linear4ReplaceSettings,
):
    for name, child in module.named_children():
        # TODO: support linear bias
        if (
            isinstance(child, nn.Linear)
            and child.bias is None
            and (
                settings.skip_layer_func is None
                or not settings.skip_layer_func(child.weight)
            )
        ):
            if (
                _check_linear_int4_k(
                    child.in_features, settings.groupsize, settings.inner_k_tiles
                )
                or settings.padding_allowed
            ):
                new_linear = settings.linear_class(
                    child.in_features,
                    child.out_features,
                    config=Int4LinearConfig(
                        bias=False,
                        device=child.weight.device,
                        groupsize=settings.groupsize,
                        inner_k_tiles=settings.inner_k_tiles,
                        precision=settings.precision,
                        scales_precision=settings.scales_precision,
                    ),
                )
                # TODO: merge with 8da4w?
                # In distributed training, the model may be instantiated
                # on the meta device, in which case there is no need to
                # copy the weights, and doing so will result in an error
                if settings.copy_weights and child.weight.device != torch.device(
                    "meta"
                ):
                    new_linear.weight = child.weight
                setattr(module, name, new_linear)
        else:
            _replace_linear_int4(child, settings)


def replace_linear_int4(
    module, groupsize, inner_k_tiles, padding_allowed, skip_layer_func=None
):
    _replace_linear_int4(
        module,
        _Linear4ReplaceSettings(
            groupsize=groupsize,
            inner_k_tiles=inner_k_tiles,
            padding_allowed=padding_allowed,
            skip_layer_func=skip_layer_func,
            linear_class=WeightOnlyInt4Linear,
        ),
    )


@dataclass
class Int4WeightOnlyQuantizerConfig:
    """Options for :class:`Int4WeightOnlyQuantizer`.

    Grouping these related settings keeps the quantizer's constructor small.
    """

    groupsize: int = 256
    padding_allowed: bool = True
    inner_k_tiles: Optional[int] = 8
    device: torch.device = torch.device("cuda")
    precision: torch.dtype = torch.bfloat16


class Int4WeightOnlyQuantizer:
    def __init__(
        self,
        config: Optional[Int4WeightOnlyQuantizerConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = Int4WeightOnlyQuantizerConfig()
        assert config.inner_k_tiles in [2, 4, 8]
        assert config.groupsize in [32, 64, 128, 256]

        self.inner_k_tiles = config.inner_k_tiles
        self.groupsize: int = config.groupsize
        self.padding_allowed: bool = config.padding_allowed
        self.device: torch.device = config.device
        # precision and dtype are being used interchangeably here
        self.precision: torch.dtype = config.precision

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.bias is None:
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert in_features % self.groupsize == 0, (
                    f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"
                )

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (w_int4x8, scales_and_zeros) = groupwise_affine_quantize_tensor(
                    weight,
                    4,  # n_bit
                    self.groupsize,
                    self.precision,  # dtype for scales_and_zeros
                )
                # TODO: just get the device from mod.weight.device?
                if _is_device(w_int4x8.device.type, "cpu"):
                    weight_int4pack = (
                        torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                            w_int4x8.to(self.device), self.inner_k_tiles
                        )
                    )
                else:
                    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                        w_int4x8.to(self.device), self.inner_k_tiles
                    )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to(self.device)
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to(
                    self.device
                )
        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_linear_int4(
            model,
            _Linear4ReplaceSettings(
                groupsize=self.groupsize,
                inner_k_tiles=self.inner_k_tiles,
                padding_allowed=self.padding_allowed,
                skip_layer_func=None,
                precision=self.precision,
                scales_precision=self.precision,
            ),
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        return _quantize_via_state_dict(self, model)


@dataclass
class Int8DynActInt4Weights:
    """Unpacked int4 weights plus the metadata needed to run the matmul.

    These values are stored together by :class:`Int8DynActInt4WeightLinear`, so
    grouping them keeps :func:`linear_forward_8da4w` down to its input tensor and
    the weights it operates on.
    """

    weight_int8: torch.Tensor
    bias: Optional[torch.Tensor]
    scales: torch.Tensor
    zeros: torch.Tensor
    out_features: int
    groupsize: int
    output_precision: torch.dtype


def linear_forward_8da4w(
    x,
    weights: Int8DynActInt4Weights,
):
    weight_int8 = weights.weight_int8
    bias = weights.bias
    scales = weights.scales
    zeros = weights.zeros
    groupsize = weights.groupsize
    output_precision = weights.output_precision
    # uses fp32 to match the PTQ activation quantization scale dtype
    # and activation_scale_dtype in QAT configs
    # TODO: in future add ability to specify activation_scale_dtype to PTQ configs
    # and enable similar change here
    x = per_token_dynamic_quant(
        x,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
        eps=torch.finfo(torch.float32).eps,
    )

    # TODO: verify and remove following reshape code
    # origin_x_size = x.size()
    # x = x.reshape(-1, origin_x_size[-1])

    # TODO: better API
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    block_size = (1, groupsize)

    w_dq = dequantize_affine(
        weight_int8,
        block_size,
        scales,
        zeros,
        torch.int8,
        quant_min,
        quant_max,
        output_dtype=output_precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(x, w_dq, bias)

    # new_shape = origin_x_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c


@dataclass
class Int8DynActInt4WeightConfig:
    """Construction options for the int8-dynamic-act int4-weight linear layer.

    Grouping these related values (placement plus quantization settings) keeps
    ``Int8DynActInt4WeightLinear``'s constructor down to the required in/out
    shape and makes the settings easy to pass around together.
    """

    bias: bool = True
    device: Optional[torch.device] = None
    groupsize: int = 256
    precision: torch.dtype = torch.float32
    scales_precision: torch.dtype = torch.float32


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    groupsize: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[Int8DynActInt4WeightConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = Int8DynActInt4WeightConfig()
        bias = config.bias
        groupsize = config.groupsize
        precision = config.precision
        scales_precision = config.scales_precision
        # Note: config.device is accepted for API symmetry but, as in the
        # original implementation, buffers are created on the default device and
        # moved later by the caller (e.g. via copy_weights).
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert in_features % groupsize == 0, (
            f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        )
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, groupsize
        # )
        self.in_features = in_features
        self.out_features = out_features
        # TODO: align groupsize naming
        self.groupsize = groupsize
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=precision))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8da4w(
            input,
            Int8DynActInt4Weights(
                weight_int8=self.weight,
                bias=self.bias,
                scales=self.scales,
                zeros=self.zeros,
                out_features=self.out_features,
                groupsize=self.groupsize,
                output_precision=self.precision,
            ),
        )


@dataclass
class _Linear8da4wReplaceSettings:
    """Quantization settings that travel together when replacing linear layers.

    Grouping these related values keeps :func:`_replace_linear_8da4w`'s
    signature small and avoids positional-argument mistakes at call sites.
    """

    groupsize: int
    padding_allowed: bool
    precision: torch.dtype
    scales_precision: torch.dtype


def _replace_linear_8da4w(
    module: torch.nn.Module,
    settings: _Linear8da4wReplaceSettings,
    linear_class: Type[torch.nn.Module],
    copy_weights: bool = False,
):
    # import the util function here to avoid circular dependency
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        return isinstance(child, nn.Linear) and (
            _check_linear_int4_k(child.in_features, settings.groupsize)
            or settings.padding_allowed
        )

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = linear_class(
            child.in_features,
            child.out_features,
            config=Int8DynActInt4WeightConfig(
                bias=child.bias is not None,
                device=child.weight.device,
                groupsize=settings.groupsize,
                precision=settings.precision,
                scales_precision=settings.scales_precision,
            ),
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if copy_weights and child.weight.device != torch.device("meta"):
            new_linear.weight = child.weight
            new_linear.bias = child.bias
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    _replace_linear_8da4w(
        module,
        _Linear8da4wReplaceSettings(
            groupsize=groupsize,
            padding_allowed=padding_allowed,
            precision=precision,
            scales_precision=scales_precision,
        ),
        Int8DynActInt4WeightLinear,
    )


@dataclass
class Int8DynActInt4WeightQuantizerConfig:
    """Options for :class:`Int8DynActInt4WeightQuantizer`.

    Grouping these related settings keeps the quantizer's constructor small.
    """

    groupsize: int = 256
    padding_allowed: bool = False
    precision: torch.dtype = torch.float32
    scales_precision: torch.dtype = torch.float32
    device: torch.device = torch.device("cpu")
    mapping_type: MappingType = MappingType.SYMMETRIC


class Int8DynActInt4WeightQuantizer:
    def __init__(
        self,
        config: Optional[Int8DynActInt4WeightQuantizerConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = Int8DynActInt4WeightQuantizerConfig()
        self.groupsize: int = config.groupsize
        self.padding_allowed: bool = config.padding_allowed
        self.precision: torch.dtype = config.precision
        self.scales_precision: torch.dtype = config.scales_precision
        self.device: torch.device = config.device
        self.mapping_type: MappingType = config.mapping_type

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert in_features % self.groupsize == 0, (
                    f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"
                )

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (
                    weight_int8,
                    scales,
                    zeros,
                ) = group_quantize_tensor_symmetric(
                    weight.to(self.precision),
                    4,  # n_bit
                    self.groupsize,
                    self.scales_precision,
                    mapping_type=self.mapping_type,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int8.to(self.device)
                cur_state_dict[f"{fqn}.scales"] = scales.to(self.device)
                cur_state_dict[f"{fqn}.zeros"] = zeros.to(self.device)

        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            self.scales_precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        return _quantize_via_state_dict(self, model)
