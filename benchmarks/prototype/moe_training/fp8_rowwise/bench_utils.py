# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for the FP8 rowwise scale-and-cast benchmark scripts."""

import itertools
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch
from tabulate import tabulate

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


def build_per_group_configs(
    experiment_config_cls: Callable[..., object],
    input_shapes: List[Tuple[int, ...]],
    n_groups_list: List[int] = None,
    high_precision_dtypes: List[torch.dtype] = None,
) -> List[object]:
    """Build the cartesian product of shapes/groups/dtypes into per-group configs.

    Shared by the per-group colwise and rowwise scale benchmarks, which build an
    identical grid and differ only in their ``input_shapes``. Each benchmark
    passes its own ``ExperimentConfig`` dataclass as ``experiment_config_cls``;
    the dataclass fields (``input_shape``, ``n_groups``, ``high_precision_dtype``)
    are identical across those benchmarks.
    """
    if n_groups_list is None:
        n_groups_list = [1, 16, 64]
    if high_precision_dtypes is None:
        high_precision_dtypes = [torch.bfloat16]
    configs = []
    for input_shape, n_groups, high_precision_dtype in itertools.product(
        input_shapes, n_groups_list, high_precision_dtypes
    ):
        configs.append(
            experiment_config_cls(
                input_shape=input_shape,
                n_groups=n_groups,
                high_precision_dtype=high_precision_dtype,
            )
        )
    return configs


def print_experiment_table(
    experiments: Sequence[object],
    headers: List[str],
    row_fn: Callable[[object], List[object]],
) -> None:
    """Render a benchmark results table with ``tabulate``.

    Shared by the per-group scale benchmarks whose ``print_results`` functions
    only differ in their ``headers`` and the per-experiment row they build.
    ``row_fn`` maps a single experiment to the list of cell values for its row.
    """
    rows = [row_fn(experiment) for experiment in experiments]
    print(tabulate(rows, headers=headers))


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
