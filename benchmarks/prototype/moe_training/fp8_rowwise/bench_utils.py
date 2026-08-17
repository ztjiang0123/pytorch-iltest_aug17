# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for the FP8 rowwise scale-and-cast benchmark scripts."""

import itertools
from dataclasses import dataclass
from typing import List, Tuple

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple
    round_scales_to_power_of_2: bool


def build_configs(
    input_shapes: List[Tuple[int, ...]],
    high_precision_dtypes: List[torch.dtype] = None,
    power_of_2_scales: List[bool] = None,
) -> List[ExperimentConfig]:
    """Build the cartesian product of shapes/dtypes/scale-rounding into configs.

    Only ``input_shapes`` differs between the 2D and 3D scale-and-cast
    benchmarks, so the enumeration loop lives here with one source of truth.
    """
    if high_precision_dtypes is None:
        high_precision_dtypes = [torch.bfloat16]
    if power_of_2_scales is None:
        power_of_2_scales = [True]
    configs = []
    for (
        input_shape,
        high_precision_dtype,
        round_scales_to_power_of_2,
    ) in itertools.product(input_shapes, high_precision_dtypes, power_of_2_scales):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                high_precision_dtype=high_precision_dtype,
                round_scales_to_power_of_2=round_scales_to_power_of_2,
            )
        )
    return configs


def reference_scale_and_cast(
    tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
    round_scales_to_power_of_2: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native 3-op scale-and-cast sequence (tensor_to_scale + multiply + cast).

    This is the unfused reference path that ``torch.compile`` is benchmarked
    against. ``axiswise_dim`` selects the reduction axis (-1 for the 2D rowwise
    case, -2 for the 3D colwise case).
    """
    scales = tensor_to_scale(
        tensor,
        float8_dtype,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=axiswise_dim,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    scaled = tensor.to(torch.float32) * scales
    data = to_fp8_saturated(scaled, float8_dtype)
    return data, scales
