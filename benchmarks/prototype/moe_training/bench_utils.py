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
