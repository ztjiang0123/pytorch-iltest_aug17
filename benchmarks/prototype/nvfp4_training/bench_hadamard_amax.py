# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List

import torch
import triton

from benchmarks.prototype.nvfp4_training.bench_common import (
    BenchmarkHarness,
    build_representative_model_configs,
    print_results,
    run_benchmark_main,
)
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    _hadamard_amax_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_rht_matrix,
)

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BENCH_OUTPUT_BUFFER_COUNT = 1_000_000

RHT_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int
    model: str = ""
    shape: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def get_representative_model_configs() -> List[ExperimentConfig]:
    return build_representative_model_configs(ExperimentConfig)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    x = torch.randn(config.m, config.n, dtype=torch.bfloat16, device=device)

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch
    # space. Set it outside the timed region so the benchmark measures the
    # kernel body instead of wrapper setup.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=x.device
            )
        )

    rht_matrix = get_rht_matrix(RHT_SIGN_VECTOR, x.device, torch.bfloat16, 16)
    global_rht_amaxes = torch.zeros(
        BENCH_OUTPUT_BUFFER_COUNT, dtype=torch.float32, device=x.device
    )
    global_a_amaxes = torch.zeros_like(global_rht_amaxes)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    next_output_idx = 0

    def run_kernel():
        nonlocal next_output_idx
        if next_output_idx >= BENCH_OUTPUT_BUFFER_COUNT:
            raise RuntimeError(
                "Exhausted pre-zeroed output buffers; increase "
                "BENCH_OUTPUT_BUFFER_COUNT."
            )
        output_idx = next_output_idx
        next_output_idx += 1
        _hadamard_amax_kernel[(num_sms,)](
            x,
            rht_matrix,
            global_rht_amaxes[output_idx],
            global_a_amaxes[output_idx],
            config.m,
            config.n,
            GROUP_SIZE_N=8,
            NUM_SMS=num_sms,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    read_bytes = x.numel() * (torch.finfo(torch.bfloat16).bits // 8)
    gbps = (read_bytes / 1e9) / (time_us / 1e6)

    return ExperimentResult(time_us=time_us, gbps=gbps)


def main():
    run_benchmark_main(
        BenchmarkHarness(
            get_configs=get_configs,
            get_representative_model_configs=get_representative_model_configs,
            run_experiment=run_experiment,
            make_experiment=lambda config, result: Experiment(
                config=config, result=result
            ),
            print_results=print_results,
        )
    )


if __name__ == "__main__":
    main()
