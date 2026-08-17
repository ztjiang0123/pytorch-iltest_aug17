# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for the MoE-training benchmark scripts."""

import os
from typing import Callable, List

import torch
import torch.distributed as dist
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


def generate_split_sizes(
    K: int,
    N: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Generate a tensor of ``K`` random non-negative integers that sum to ``N``.

    Shared by the MoE EP benchmarks so the split-size generation logic has a
    single source of truth. ``dtype`` selects the integer type of the returned
    tensor (e.g. ``torch.int64`` for all-to-all-v, ``torch.int32`` for the EP
    pipeline).
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if K == 1:
        return torch.tensor([N], dtype=dtype, device=device)

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

    return result.to(dtype=dtype)
