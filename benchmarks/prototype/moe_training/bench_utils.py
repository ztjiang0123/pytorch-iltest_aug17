# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for the MoE-training benchmark scripts."""

import itertools
import os
from typing import Any, Callable, List, Sequence, Type

import torch
import torch.distributed as dist
from tabulate import tabulate
from torch import nn


def make_moe_module_filter_fn(
    target_fqns: List[str],
) -> Callable[[nn.Module, str], bool]:
    """Build a ``quantize_`` filter that matches modules whose FQN contains any
    of ``target_fqns``.

    Returns a ``(module, cur_fqn) -> bool`` predicate suitable for the
    ``filter_fn`` argument of ``quantize_``. Shared by the MoE-layer benchmarks
    so the selection logic has a single source of truth.
    """

    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    return moe_module_filter_fn


def setup_distributed():
    """Initialize the NCCL process group from the standard torchrun env vars."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def generate_split_sizes(K: int, N: int, device: str = "cuda") -> torch.Tensor:
    """Generate a tensor of ``K`` random non-negative integers that sum to ``N``.

    Shared by the MoE EP benchmarks so the split-size generation logic has a
    single source of truth. Returns an ``int64`` tensor; callers that need a
    different integer width (e.g. ``int32``) can ``.to()`` the result.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if K == 1:
        return torch.tensor([N], dtype=torch.int64, device=device)

    # Generate K-1 random "dividers" in the range [0, N].
    dividers = torch.randint(0, N + 1, (K - 1,), device=device)

    # Add 0 and N to the set of dividers to form the boundaries.
    boundaries = torch.cat(
        [
            torch.tensor([0], device=device),
            dividers,
            torch.tensor([N], device=device),
        ]
    )

    # Sort the boundaries to ensure they are in order
    sorted_boundaries = torch.sort(boundaries).values

    # The K integers are the differences between consecutive boundaries (sum to N)
    result = sorted_boundaries[1:] - sorted_boundaries[:-1]

    return result.to(dtype=torch.int64)


def build_input_shape_configs(
    input_shapes: Sequence[Any],
    config_cls: Type,
) -> List:
    """Build a list of ``config_cls(input_shape=shape)`` from ``input_shapes``.

    Shared by the single-field ``ExperimentConfig`` benchmarks (which only vary
    the input shape) so the config-construction loop has a single source of
    truth.
    """
    return [config_cls(input_shape=shape) for shape in input_shapes]


def build_token_group_configs(config_cls: Type) -> List:
    """Build the standard token-group sweep of ``config_cls`` instances.

    Enumerates the Cartesian product of the token-group benchmark parameters
    (``num_tokens``, ``dim``, ``num_groups``, ``alignment_size``). Shared by the
    pad/unpad token-group benchmarks so the sweep has a single source of truth.
    """
    # Various token group sizes and dimensions
    num_tokens_list = [16384]
    dim_list = [1536, 2048, 5120, 7168]
    num_groups_list = [1, 4, 8, 16]
    alignment_size_list = [32]

    return [
        config_cls(
            num_tokens=num_tokens,
            dim=dim,
            num_groups=num_groups,
            alignment_size=alignment_size,
        )
        for num_tokens, dim, num_groups, alignment_size in itertools.product(
            num_tokens_list, dim_list, num_groups_list, alignment_size_list
        )
    ]


def print_token_group_results(experiments: List, second_impl_label: str = "cuda"):
    """Print token-group benchmark results as a formatted table.

    Shared by the pad/unpad token-group benchmarks, which report identical
    columns comparing a torch-eager implementation against a second (CUDA)
    implementation. ``second_impl_label`` names the second implementation in the
    ``<label>_us`` / ``<label>_mem_bw_gbps`` / ``<label>_vs_torch`` columns.
    """
    headers = [
        "num_tokens",
        "dim",
        "num_groups",
        "torch_us",
        f"{second_impl_label}_us",
        "torch_mem_bw_gbps",
        f"{second_impl_label}_mem_bw_gbps",
        f"{second_impl_label}_vs_torch",
    ]
    rows = []
    for experiment in experiments:
        cuda_time = experiment.result.cuda_time_us
        cuda_vs_torch = (
            f"{experiment.result.torch_eager_time_us / cuda_time:.2f}x"
            if cuda_time != float("inf") and cuda_time > 0
            else "N/A"
        )
        cuda_bw_str = (
            f"{experiment.result.cuda_mem_bw_gbps:.2f}"
            if experiment.result.cuda_mem_bw_gbps > 0
            else "N/A"
        )

        rows.append(
            [
                experiment.config.num_tokens,
                experiment.config.dim,
                experiment.config.num_groups,
                experiment.result.torch_eager_time_us,
                experiment.result.cuda_time_us,
                f"{experiment.result.torch_mem_bw_gbps:.2f}",
                cuda_bw_str,
                cuda_vs_torch,
            ]
        )
    print(tabulate(rows, headers=headers))


def print_cutedsl_rearrange_results(
    experiments: List,
    baseline_label: str,
    baseline_us_attr: str,
    baseline_gbps_attr: str,
):
    """Print CuTeDSL-vs-baseline benchmark results as a formatted table.

    Shared by the 2d 1x32 / 32x1 CuTeDSL quantize benchmarks, which compare a
    CuTeDSL blocked kernel against a "<baseline> + rearrange" pipeline. The
    baseline differs only in its label (``triton`` vs ``cuda``) and the
    corresponding ``ExperimentResult`` attribute names, passed in here.
    """
    headers = [
        "input_shape",
        "scaling_mode",
        "num_groups",
        "cutedsl_blocked_us",
        f"{baseline_label}+rearrange_us",
        "speedup",
        "cutedsl_gbps",
        f"{baseline_label}+rearrange_gbps",
    ]
    rows = []
    for experiment in experiments:
        baseline_us = getattr(experiment.result, baseline_us_attr)
        baseline_gbps = getattr(experiment.result, baseline_gbps_attr)
        speedup = baseline_us / experiment.result.cutedsl_blocked_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.scaling_mode,
                experiment.config.num_groups,
                f"{experiment.result.cutedsl_blocked_us:.2f}",
                f"{baseline_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.cutedsl_blocked_gbps:.1f}",
                f"{baseline_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))
