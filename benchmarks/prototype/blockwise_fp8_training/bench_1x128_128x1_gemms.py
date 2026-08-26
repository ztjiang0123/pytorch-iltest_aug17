# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

from dataclasses import dataclass

import torch
from torch.nn.functional import ScalingType, scaled_mm
from triton.testing import do_bench

from benchmarks.prototype.blockwise_fp8_training.bench_gemm_utils import (
    ExperimentConfig,
    get_configs,
    print_gemm_results,
)
from benchmarks.utils import run_experiments_and_print
from torchao.prototype.blockwise_fp8_training.kernels import (
    Fp8Gemm1x128Operands,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_gemm_1x128_128x1,
)

device = torch.device("cuda")

# This benchmark requires CUDA 12.9+
assert torch.version.cuda is not None, "CUDA is not available"
cuda_major, cuda_minor = map(int, torch.version.cuda.split("."))
assert cuda_major >= 12 and cuda_minor >= 9, "CUDA 12.9+ is required"

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentResult:
    bf16_mm_us: float
    fp8_triton_us: float
    fp8_scaled_mm_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # Simulate `grad_weight = grad_output_t @ input`
    M, N, K = config.m, config.n, config.k
    A = torch.randn(M, N, dtype=config.out_dtype, device="cuda")
    B = torch.randn(M, K, dtype=config.out_dtype, device="cuda")
    A_t_q, A_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
        A, dtype=torch.float8_e4m3fn
    )
    B_q, B_s = triton_fp8_blockwise_act_quant_rhs(B, dtype=torch.float8_e4m3fn)

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    # Warmup then run bf16 torch.mm
    warmup(torch.mm, A.t(), B)

    bf16_mm_us = benchmark_cuda_function_in_microseconds(torch.mm, A.t(), B)

    # Warm up then run triton bench
    triton_operands = Fp8Gemm1x128Operands(
        a=A_t_q,
        b=B_q,
        a_s=1.0 / A_t_s,
        b_s=1.0 / B_s,
    )
    warmup(
        triton_fp8_gemm_1x128_128x1,
        triton_operands,
        out_dtype=config.out_dtype,
    )

    fp8_triton_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_gemm_1x128_128x1,
        triton_operands,
        out_dtype=config.out_dtype,
    )

    # Warm up then run torch bench
    scale_recipe_a = ScalingType.BlockWise1x128
    scale_recipe_b = ScalingType.BlockWise1x128
    B_s_t = B_s.t()

    warmup(
        scaled_mm,
        A_t_q,
        B_q,
        1.0 / A_t_s,
        scale_recipe_a,
        1.0 / B_s_t,
        scale_recipe_b,
        output_dtype=config.out_dtype,
    )

    fp8_scaled_mm_us = benchmark_cuda_function_in_microseconds(
        scaled_mm,
        A_t_q,
        B_q,
        1.0 / A_t_s,
        scale_recipe_a,
        1.0 / B_s_t,
        scale_recipe_b,
        output_dtype=config.out_dtype,
    )

    return ExperimentResult(
        bf16_mm_us=bf16_mm_us,
        fp8_triton_us=fp8_triton_us,
        fp8_scaled_mm_us=fp8_scaled_mm_us,
    )


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def main():
    run_experiments_and_print(
        get_configs, run_experiment, print_gemm_results, Experiment
    )


if __name__ == "__main__":
    main()
