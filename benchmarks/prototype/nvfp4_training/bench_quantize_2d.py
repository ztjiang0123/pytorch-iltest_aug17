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
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
    triton_quantize_2d_weight,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
# N must be a multiple of BLOCK_N=256
N_SHAPES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


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


def run_experiment(config: ExperimentConfig) -> ExperimentResult | None:
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)

    if torch.cuda.is_available() and not is_sm_at_least_100():
        return None

    global_amax = x.float().abs().max()

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch
    # space. Set it outside the timed region so the benchmark measures the
    # kernel body instead of wrapper setup.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=x.device
            )
        )

    a_fp4 = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    a_sf = torch.empty(
        (m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    a_t_fp4 = torch.empty((n, m // 2), dtype=torch.uint8, device=x.device)
    a_t_sf = torch.empty(
        (n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count

    def run_kernel():
        triton_quantize_2d_weight[(num_sms,)](
            x,
            a_fp4,
            a_sf,
            a_t_fp4,
            a_t_sf,
            global_amax,
            m,
            n,
            GROUP_SIZE_N=8,
            NUM_SMS=num_sms,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    read_bytes = m * n * 2  # bf16 input
    write_fp4 = 2 * m * (n // 2)  # rowwise and colwise packed FP4 outputs
    write_scales = 2 * m * (n // 16)  # rowwise and colwise FP8 scale factors
    gbps = ((read_bytes + write_fp4 + write_scales) / 1e9) / (time_us / 1e6)

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
