# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared config and reporting helpers for the blockwise FP8 GEMM benchmarks.

``bench_1x128_128x128_gemms.py``, ``bench_1x128_128x1_gemms.py`` and
``bench_linear_fwd_bwd.py`` all sweep the same (M, N, K) shape list and produce
identically-shaped experiment configs. The two raw-GEMM scripts additionally
share the same TFLOP/s results table. Those shared pieces live here so each
script imports one implementation instead of duplicating it.
"""

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate


@dataclass(frozen=True)
class ExperimentConfig:
    out_dtype: torch.dtype
    m: int
    n: int
    k: int


def get_configs() -> List[ExperimentConfig]:
    mnk_list = [
        # Llama4 shapes
        (16640, 5120, 8192),
        (16640, 8192, 5120),
    ]
    out_dtypes = [torch.bfloat16]
    configs = []
    for mnk, out_dtype in itertools.product(mnk_list, out_dtypes):
        m, n, k = mnk
        configs.append(
            ExperimentConfig(
                out_dtype=out_dtype,
                m=m,
                n=n,
                k=k,
            )
        )
    return configs


def print_gemm_results(experiments: List) -> None:
    """Print the bf16/triton/scaled_mm TFLOP/s table for the raw GEMM benchmarks.

    Each experiment's ``result`` must expose ``bf16_mm_us``, ``fp8_triton_us`` and
    ``fp8_scaled_mm_us``.
    """
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "bf16_mm_us",
        "fp8_triton_us",
        "fp8_scaled_mm_us",
        "bf16 tflops/sec",
        "triton tflops/sec",
        "scaled_mm tflops/sec",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        flops = 2 * m * n * k
        bf16_mm_tflops_per_sec = (flops / 1e12) / (experiment.result.bf16_mm_us / 1e6)
        triton_tflops_per_sec = (flops / 1e12) / (experiment.result.fp8_triton_us / 1e6)
        scaled_mm_tflops_per_sec = (flops / 1e12) / (
            experiment.result.fp8_scaled_mm_us / 1e6
        )
        rows.append(
            [
                m,
                n,
                k,
                experiment.config.out_dtype,
                experiment.result.bf16_mm_us,
                experiment.result.fp8_triton_us,
                experiment.result.fp8_scaled_mm_us,
                bf16_mm_tflops_per_sec,
                triton_tflops_per_sec,
                scaled_mm_tflops_per_sec,
            ]
        )
    print(tabulate(rows, headers=headers))
