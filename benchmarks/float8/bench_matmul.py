# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from dataclasses import dataclass
from typing import Optional

import fire
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType
from utils import (
    BenchmarkConfig,
    do_benchmarks,
    get_name_to_shapes_iter,
)

from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.testing.training.roofline_utils import get_specs
from torchao.utils import is_MI300


@dataclass(frozen=True)
class MatmulBenchConfig:
    """All options for a single ``bench_matmul`` run.

    The parameters of this benchmark all travel together as one benchmark
    configuration, so they are grouped into a single value object instead of a
    long parameter list.

    * ``shape_gen_name``/``M``/``K``/``N``: shape generation (``M``/``K``/``N``
      override the generated dimensions when the ``custom`` generator is used).
    * ``recipe``: quantization recipe to benchmark.
    * ``use_gpu_kernel_time``: measure GPU kernel time instead of wall time.
    * ``n_limit``: if set, only run this many shape iterations.
    * ``out_filename``: if set, write results to this CSV file.
    """

    shape_gen_name: str = "pow2_extended"
    M: Optional[int] = None
    K: Optional[int] = None
    N: Optional[int] = None
    recipe: str = "tensorwise"
    use_gpu_kernel_time: bool = True
    n_limit: Optional[int] = None
    out_filename: Optional[str] = None


# A frozen (immutable) instance is safe to share as a default argument.
_DEFAULT_CONFIG = MatmulBenchConfig()


_SUPPORTED_RECIPES = (
    "tensorwise",
    "rowwise",
    "mxfp4_cutlass",
    "nvfp4",
)


@dataclass(frozen=True)
class _MatmulInputs:
    """Per-iteration inputs the recipe helpers need to build the matmul.

    Bundling these together keeps the ``_prepare_*`` helpers to a single
    parameter instead of a long, easily-transposed argument list.
    """

    recipe: str
    A_hp: torch.Tensor
    B_hp_t: torch.Tensor
    dtype: torch.dtype
    M: int
    N: int
    device: str
    fp4_peak_tops: float
    fp8_peak_tops: float


@dataclass
class _MatmulOperands:
    """The quantized inputs, scales, and metadata for one recipe's matmul."""

    A: torch.Tensor
    B: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    peak_tops: float
    out_dtype: Optional[torch.dtype]  # fp8 output dtype; ``None`` for fp4 recipes


def _prepare_mxfp4_operands(inputs: _MatmulInputs):
    A_scales, A_data = to_mx(inputs.A_hp, torch.float4_e2m1fn_x2, 32)
    B_scales, Bt_data = to_mx(inputs.B_hp_t, torch.float4_e2m1fn_x2, 32)
    A = A_data.view(torch.float4_e2m1fn_x2)
    B = Bt_data.view(torch.float4_e2m1fn_x2).contiguous().T
    # Use the blockwise scales from to_mx
    return _MatmulOperands(
        A=A,
        B=B,
        scale_a=to_blocked(A_scales),
        scale_b=to_blocked(B_scales),
        peak_tops=inputs.fp4_peak_tops,
        out_dtype=None,
    )


def _prepare_nvfp4_operands(inputs: _MatmulInputs):
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

    A_scales, A_data = nvfp4_quantize(inputs.A_hp, block_size=16)
    B_scales, B_data = nvfp4_quantize(inputs.B_hp_t, block_size=16)
    A = A_data.view(torch.float4_e2m1fn_x2)
    B = B_data.view(torch.float4_e2m1fn_x2).T
    # Use the blockwise scales from nvfp4_quantize (pad if needed)
    return _MatmulOperands(
        A=A,
        B=B,
        scale_a=to_blocked(A_scales.view(torch.float8_e4m3fn)),
        scale_b=to_blocked(B_scales.view(torch.float8_e4m3fn)),
        peak_tops=inputs.fp4_peak_tops,
        out_dtype=None,
    )


def _prepare_fp8_operands(inputs: _MatmulInputs):
    # raw float8 matmul (upper bound for what we can achive in eager mode)
    # TODO(future): add e5m2
    e4m3_dtype = torch.float8_e4m3fn
    if torch.version.hip and torch.cuda.is_available() and is_MI300():
        e4m3_dtype = torch.float8_e4m3fnuz
    A = inputs.A_hp.to(e4m3_dtype)
    B = inputs.B_hp_t.to(e4m3_dtype).contiguous().T

    if inputs.recipe == "tensorwise":
        scale_a = torch.tensor([1.0], device=inputs.device)
        scale_b = torch.tensor([1.0], device=inputs.device)
    else:  # rowwise
        scale_a = torch.ones(inputs.M, 1, device=inputs.device)
        scale_b = torch.ones(1, inputs.N, device=inputs.device)

    return _MatmulOperands(
        A=A,
        B=B,
        scale_a=scale_a,
        scale_b=scale_b,
        peak_tops=inputs.fp8_peak_tops,
        out_dtype=inputs.dtype,
    )


def _prepare_operands(inputs: _MatmulInputs):
    """Quantize inputs and build scales for the recipe, returning ``_MatmulOperands``."""
    if inputs.recipe == "mxfp4_cutlass":
        return _prepare_mxfp4_operands(inputs)
    if inputs.recipe == "nvfp4":
        return _prepare_nvfp4_operands(inputs)
    return _prepare_fp8_operands(inputs)


def _select_matmul(recipe, operands, dtype, fast_accum):
    """Build the matmul closure for ``recipe`` over the prepared operands."""
    scale_a = operands.scale_a
    scale_b = operands.scale_b

    def do_matmul_fp8(A, B):
        return torch._scaled_mm(
            A,
            B,
            scale_a,
            scale_b,
            out_dtype=operands.out_dtype,
            use_fast_accum=fast_accum,
        )

    def do_matmul_mxfp4(A, B):
        return F.scaled_mm(
            A,
            B,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=dtype,
        )

    def do_matmul_nvfp4(A, B):
        return torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=dtype)

    if recipe == "mxfp4_cutlass":
        return do_matmul_mxfp4
    if recipe == "nvfp4":
        return do_matmul_nvfp4
    return do_matmul_fp8


@torch.inference_mode()
def run(config: MatmulBenchConfig = _DEFAULT_CONFIG):
    shape_gen_name = config.shape_gen_name
    M, K, N = config.M, config.K, config.N
    recipe = config.recipe
    use_gpu_kernel_time = config.use_gpu_kernel_time
    n_limit = config.n_limit
    out_filename = config.out_filename
    device = "cuda"
    # TODO(future PR): this is ugly
    assert recipe in _SUPPORTED_RECIPES, "unsupported"
    use_fp4 = recipe in ("mxfp4_cutlass", "nvfp4")

    specs = get_specs()
    bf16_peak_tops = specs["bf16_peak_tops"]
    fp8_peak_tops = specs["fp8_peak_tops"]
    fp4_peak_tops = specs.get("fp4_peak_tops", 0.0)  # only on sm120
    print(f"recipe: {recipe}")
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(
        f"peak tops: bf16 {bf16_peak_tops:.2e}, fp8 {fp8_peak_tops:.2e}, fp4 {fp4_peak_tops:.2e}"
    )
    headers = (
        "fast_accum",
        "name",
        "M",
        "K",
        "N",
        "ref_pct_top_peak",
        "pct_top_peak",
        "ref_time_s",
        "time_s",
        "fp8_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    fast_accum_vals = [False] if use_fp4 else [True, False]

    for idx, (fast_accum, (name, (M, K, N))) in enumerate(
        itertools.product(fast_accum_vals, name_to_shapes)
    ):
        if n_limit is not None and idx >= n_limit:
            break

        tops = 2 * M * N * K
        print("M, K, N:", M, K, N, f"tops: {tops:.2E}")

        # raw torch.mm
        A = torch.randn(M, K, device=device, dtype=dtype)
        m_ref = nn.Sequential(nn.Linear(K, N, dtype=dtype, device=device, bias=False))
        ref_time_sec, ref_tops_sec, ref_pct_top_peak = do_benchmarks(
            BenchmarkConfig(tops, bf16_peak_tops, use_gpu_kernel_time), m_ref, A
        )
        print(
            f"{dtype} time_sec {ref_time_sec:.2E}, tops/sec {ref_tops_sec:.2E}, pct_peak {ref_pct_top_peak:.3f}"
        )

        del A

        A_hp = torch.randn(M, K, device=device)
        B_hp_t = torch.randn(N, K, device=device)

        operands = _prepare_operands(
            _MatmulInputs(
                recipe=recipe,
                A_hp=A_hp,
                B_hp_t=B_hp_t,
                dtype=dtype,
                M=M,
                N=N,
                device=device,
                fp4_peak_tops=fp4_peak_tops,
                fp8_peak_tops=fp8_peak_tops,
            )
        )
        do_matmul = _select_matmul(recipe, operands, dtype, fast_accum)

        time_sec, tops_sec, pct_top_peak = do_benchmarks(
            BenchmarkConfig(tops, operands.peak_tops, use_gpu_kernel_time),
            do_matmul,
            operands.A,
            operands.B,
        )
        print(
            f"time_sec {time_sec:.2E}, tops/sec {tops_sec:.2E}, pct_peak {pct_top_peak:.3f}"
        )

        del operands

        results.append(
            [
                fast_accum,
                name,
                M,
                K,
                N,
                ref_pct_top_peak,
                pct_top_peak,
                ref_time_sec,
                time_sec,
                ref_time_sec / time_sec,
            ]
        )

    data_df = pd.DataFrame(results, columns=headers)
    print(data_df)

    if out_filename is not None:
        data_df.to_csv(out_filename)


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()  # pragma: no cover
