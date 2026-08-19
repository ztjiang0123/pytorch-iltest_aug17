# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import argparse
from dataclasses import dataclass
from typing import List

import torch

from benchmarks.prototype.moe_training.bench_utils import (
    build_token_group_configs,
    print_token_group_results,
)
from benchmarks.utils import (
    benchmark_cuda_function_in_microseconds,
    profile_fn,
    run_experiments_and_print,
)
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mxfp8_cuda_kernels_available,
    fused_unpad_token_groups_cuda,
    torch_pad_token_groups,
    torch_unpad_token_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    dim: int
    num_groups: int
    alignment_size: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_eager_time_us: float
    cuda_time_us: float
    torch_mem_bw_gbps: float
    cuda_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return build_token_group_configs(ExperimentConfig)


def run_experiment(config: ExperimentConfig, profile=False) -> ExperimentResult:
    num_tokens, dim, num_groups, alignment_size = (
        config.num_tokens,
        config.dim,
        config.num_groups,
        config.alignment_size,
    )

    # Create inputs and pad them first
    inputs = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=device)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    )

    # Pad the inputs to get padded tensors for unpad benchmark
    padded_inputs, padded_group_start_offsets, padded_group_end_offsets = (
        torch_pad_token_groups(inputs, group_offsets, alignment_size)
    )

    def torch_eager_with_offsets():
        return torch_unpad_token_groups(
            padded_inputs,
            group_offsets,
            padded_group_start_offsets,
            num_tokens,
            alignment_size,
        )

    def warmup(fn):
        for _ in range(5):
            fn()

    # bench torch eager (includes buffer allocation overhead)
    warmup(torch_eager_with_offsets)
    torch_eager_time_us = benchmark_cuda_function_in_microseconds(
        torch_eager_with_offsets
    )
    if profile:
        profile_fn(
            torch_unpad_token_groups,
            padded_inputs,
            group_offsets,
            padded_group_start_offsets,
            alignment_size,
            profile_name="torch_unpad_token_groups_eager",
        )

    # bench CUDA kernel if available
    if _mxfp8_cuda_kernels_available:

        def cuda_with_offsets():
            return fused_unpad_token_groups_cuda(
                padded_inputs,
                group_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size,
            )

        warmup(cuda_with_offsets)
        cuda_time_us = benchmark_cuda_function_in_microseconds(cuda_with_offsets)
        if profile:
            profile_fn(
                fused_unpad_token_groups_cuda,
                padded_inputs,
                group_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size,
                profile_name="fused_unpad_token_groups_cuda",
            )
    else:
        cuda_time_us = float("inf")  # Not available

    # mem bw calculations
    bytes_per_el = torch.finfo(torch.bfloat16).bits / 8

    read_bytes = (
        padded_inputs.numel() * bytes_per_el  # Read padded input tokens
        + group_offsets.numel() * 4  # Read group offsets (int32)
        + padded_group_start_offsets.numel() * 4  # Read padded start offsets (int32)
    )

    write_bytes = (
        inputs.numel() * bytes_per_el  # Write unpadded data
    )

    total_bytes = read_bytes + write_bytes

    torch_mem_bw_gbps = (total_bytes / 1e9) / (torch_eager_time_us / 1e6)

    if _mxfp8_cuda_kernels_available and cuda_time_us != float("inf"):
        cuda_mem_bw_gbps = (total_bytes / 1e9) / (cuda_time_us / 1e6)
    else:
        cuda_mem_bw_gbps = 0.0

    return ExperimentResult(
        torch_eager_time_us=torch_eager_time_us,
        cuda_time_us=cuda_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        cuda_mem_bw_gbps=cuda_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    print_token_group_results(experiments)


def main(args: argparse.Namespace):
    run_experiments_and_print(
        get_configs,
        run_experiment,
        print_results,
        Experiment,
        run_experiment_kwargs={"profile": args.profile},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling with PyTorch profiler"
    )
    args = parser.parse_args()
    main(args)
