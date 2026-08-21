# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch

if torch.cuda.is_available():
    import pandas as pd
    from tqdm import tqdm
    from triton.testing import do_bench

    from torchao.float8.float8_utils import compute_error
    from torchao.kernel.blockwise_quantization import (
        blockwise_fp8_gemm,
        fp8_blockwise_act_quant,
        fp8_blockwise_weight_quant,
    )
    from torchao.utils import is_sm_at_least_89
else:
    raise RuntimeError("This benchmark is only avaible on CUDA hardware")


def benchmark_microseconds(f, *args, warmup=25, rep=100):
    return (
        do_bench(lambda: f(*args), warmup=warmup, rep=rep, return_mode="median") * 1e3
    )


@dataclass(frozen=True)
class BlockwiseProblemShape:
    """GEMM problem dimensions that travel together across the benchmarks."""

    m: int
    n: int
    k: int
    block_size: int


def get_blockwise_problem(shape: BlockwiseProblemShape, dtype: torch.dtype, device):
    m, n, k, block_size = shape.m, shape.n, shape.k, shape.block_size
    assert n % block_size == 0 and k % block_size == 0, (
        "N and K dims must be divisible by block_size"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    dtype_max = torch.finfo(dtype).max
    A = (dtype_max * (2 * torch.rand(m, k, device=device) - 1)).to(dtype)
    A_scale = torch.randn((m, k // block_size), dtype=torch.half, device=device)
    B = (dtype_max * (2 * torch.rand(n, k, device=device) - 1)).to(dtype)
    B_scale = torch.randn(
        (n // block_size, k // block_size), dtype=torch.half, device=device
    )

    return A, A_scale, B, B_scale


def benchmark_latency(shape: BlockwiseProblemShape, dtype: torch.dtype, device):
    m, n, k, block_size = shape.m, shape.n, shape.k, shape.block_size
    A_ref = torch.randn((m, k), dtype=torch.half, device=device)
    B_ref = torch.randn((n, k), dtype=torch.half, device=device)
    fp16_time = benchmark_microseconds(torch.nn.functional.linear, A_ref, B_ref)

    A, A_scale, B, B_scale = get_blockwise_problem(shape, dtype, device)
    blockwise_time = benchmark_microseconds(
        blockwise_fp8_gemm, A, A_scale, B, B_scale, block_size
    )

    return {
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "dtype": dtype,
        "fp16_latency (ms)": fp16_time,
        "blockwise_latency (ms)": blockwise_time,
        "blockwise_speedup": fp16_time / blockwise_time,
    }


def benchmark_precision(shape: BlockwiseProblemShape, dtype: torch.dtype, device):
    m, n, k, block_size = shape.m, shape.n, shape.k, shape.block_size
    lin = torch.nn.Linear(k, n, False, device, torch.half)
    A = torch.randn((m, k), dtype=torch.half, device=device)
    W = lin.weight
    output = A @ W.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size, dtype)
    W_q, W_s = fp8_blockwise_weight_quant(W, block_size, dtype)
    output_blockwise = blockwise_fp8_gemm(A_q, A_s, W_q, W_s, block_size)

    return {
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "dtype": dtype,
        "error_blockwise (dB)": compute_error(output, output_blockwise),
    }


if __name__ == "__main__" and torch.cuda.is_available():
    device = torch.device("cuda")
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)
    block_size_vals = (128, 128, 128, 128)

    latency_results = []
    precision_results = []

    available_dtypes = (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if is_sm_at_least_89()
        else [torch.float8_e5m2]
    )
    for m in tqdm([1 << i for i in range(14)]):
        for dtype in available_dtypes:
            for n, k, block_size in zip(n_vals, k_vals, block_size_vals):
                shape = BlockwiseProblemShape(m=m, n=n, k=k, block_size=block_size)
                latency_results.append(benchmark_latency(shape, dtype, device))
                precision_results.append(benchmark_precision(shape, dtype, device))

    df_latency = pd.DataFrame(latency_results)
    df_precision = pd.DataFrame(precision_results)

    df_latency.to_csv("blockwise_triton_latency_results.csv", index=False)
    df_precision.to_csv("blockwise_triton_precision_results.csv", index=False)

    print(df_latency.to_markdown(index=False))
    print(df_precision.to_markdown(index=False))
