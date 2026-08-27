# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Per-kernel benchmark for the blockwise FP8 MoE grouped-GEMM path.

Mirrors the linear blockwise FP8 benchmarks: each kernel the MoE op dispatches
is timed in isolation. Quantization (cast) kernels are memory-bound and reported
against the memory-bandwidth roofline (GB/s, % of achievable). The DeepGEMM
grouped GEMMs are compute-bound and reported against the FP8 tensor-core
roofline (TFLOP/s, % of achievable); their modeled memory traffic is also shown
so memory-bound cases (e.g. the K-grouped wgrad, which reads+writes FP32
accumulators) are visible.

Byte accounting for the cast kernels counts the high-precision input read plus
the FP8 data and FP32 scale writes (the actual operand tensors produced).

Kernels timed, matching torchao.prototype.moe_training.blockwise_fp8.grouped_mm:

  forward
    - act_quant_lhs                 (activations -> FP8 1x128)
    - weight_quant_forward_rhs      (weights -> DeepGEMM (E,N,K) FP8 128x128)
    - deepgemm_grouped_mm           (out = A @ B_t)
  backward (dgrad)
    - act_quant_lhs(grad_out)
    - weight_quant_dgrad_rhs        (weights -> DeepGEMM (E,K,N) FP8 128x128)
    - deepgemm_grouped_mm_dgrad     (grad_A = grad_out @ B)
  backward (wgrad)
    - wgrad_quant_lhs               (K-grouped activation quant of grad_out)
    - wgrad_quant_rhs               (K-grouped activation quant of A)
    - deepgemm_grouped_mm_wgrad     (grad_B = grad_out^T @ A, K-grouped)
"""

import argparse
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from triton.testing import do_bench

from benchmarks.prototype.blockwise_fp8_training.roofline_utils import (
    lookup_roofline_specs,
    peak_mem_bw_from_device_properties,
)
from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    _quantize_wgrad_lhs,
    _quantize_wgrad_rhs,
    _should_quantize_k_grouped_directly,
    deepgemm_blockwise_scaled_grouped_mm,
    deepgemm_blockwise_scaled_grouped_mm_wgrad,
    is_deep_gemm_available,
    prepare_deepgemm_wgrad_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    build_deepgemm_grouped_offset_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
    triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs) -> float:
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


class Kind:
    MEM = "mem"  # memory-bound cast kernel -> GB/s
    GEMM = "gemm"  # compute-bound GEMM -> TFLOP/s (+ modeled mem traffic)


@dataclass
class KernelMeasurement:
    name: str
    kind: str
    us: float
    bytes_moved: int
    flops: float


@dataclass(frozen=True)
class Shape:
    M: int
    N: int
    K: int
    E: int


def _io_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _make_offsets(E: int, M: int, block_size: int, jagged: bool) -> torch.Tensor:
    if jagged:
        # Skewed per-expert token counts (real routing). Stresses load balance,
        # so the grouped GEMM rooflines reflect distribution, not just the kernel.
        return generate_jagged_offs(E, M, multiple_of=block_size, device=device)
    # Balanced: equal tokens per expert. Isolates kernel efficiency from skew.
    assert M % E == 0, "balanced offsets require M divisible by E"
    toks = M // E
    assert toks % block_size == 0, "balanced per-expert tokens must be block-aligned"
    return torch.arange(toks, M + 1, toks, dtype=torch.int32, device=device)


@dataclass
class _KernelTimers:
    """Records kernel timings into a shared measurement list.

    Bundles the per-kind timing helpers so the section benchmarks below read as
    a flat sequence of ``time_mem`` / ``time_gemm`` calls.
    """

    measurements: List[KernelMeasurement]

    def time_mem(self, name, fn, in_bytes, out_data, out_scale):
        us = benchmark_cuda_function_in_microseconds(fn)
        moved = in_bytes + _io_bytes(out_data, out_scale)
        self.measurements.append(KernelMeasurement(name, Kind.MEM, us, moved, 0.0))

    def time_gemm(self, name, fn, flops, bytes_moved):
        us = benchmark_cuda_function_in_microseconds(fn)
        self.measurements.append(
            KernelMeasurement(name, Kind.GEMM, us, bytes_moved, flops)
        )


def _bench_forward(timers, inputs, block_size):
    """Time the forward path: activation quant, weight quant, grouped GEMM."""
    M, N, K = inputs.M, inputs.N, inputs.K
    A, B_t = inputs.A, inputs.B_t
    fp8, out_dtype = inputs.fp8, inputs.out_dtype

    A_fp8, A_scale = triton_fp8_blockwise_act_quant_lhs(
        A.contiguous(), block_size=block_size, dtype=fp8
    )
    timers.time_mem(
        "fwd: act_quant_lhs",
        lambda: triton_fp8_blockwise_act_quant_lhs(
            A.contiguous(), block_size=block_size, dtype=fp8
        ),
        _io_bytes(A),
        A_fp8,
        A_scale,
    )

    B_fwd_fp8, B_fwd_scale = (
        triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm(
            B_t, block_size=block_size, dtype=fp8
        )
    )
    timers.time_mem(
        "fwd: weight_quant_forward_rhs",
        lambda: triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm(
            B_t, block_size=block_size, dtype=fp8
        ),
        _io_bytes(B_t),
        B_fwd_fp8,
        B_fwd_scale,
    )

    # GEMM mem traffic: read A_fp8 + scales + B_fp8 + scales, write bf16 out.
    fwd_gemm_bytes = (
        _io_bytes(A_fp8, A_scale, B_fwd_fp8, B_fwd_scale) + M * N * 2  # bf16 out
    )
    timers.time_gemm(
        "fwd: deepgemm_grouped_mm",
        lambda: deepgemm_blockwise_scaled_grouped_mm(
            A_fp8,
            B_fwd_fp8,
            A_scale,
            inputs.recipe_a,
            B_fwd_scale,
            inputs.recipe_b,
            inputs.offs,
            out_dtype,
            block_size,
            offset_plan=inputs.offset_plan,
        ),
        2.0 * M * N * K,
        fwd_gemm_bytes,
    )


def _bench_dgrad(timers, inputs, block_size):
    """Time the dgrad path: grad_out quant, weight quant, grouped GEMM."""
    M, N, K = inputs.M, inputs.N, inputs.K
    grad_out, B_t = inputs.grad_out, inputs.B_t
    fp8, out_dtype = inputs.fp8, inputs.out_dtype

    gout_fp8, gout_scale = triton_fp8_blockwise_act_quant_lhs(
        grad_out.contiguous(), block_size=block_size, dtype=fp8
    )
    timers.time_mem(
        "bwd: act_quant_lhs(grad_out)",
        lambda: triton_fp8_blockwise_act_quant_lhs(
            grad_out.contiguous(), block_size=block_size, dtype=fp8
        ),
        _io_bytes(grad_out),
        gout_fp8,
        gout_scale,
    )

    B_dgrad_fp8, B_dgrad_scale = triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm(
        B_t, block_size=block_size, dtype=fp8
    )
    timers.time_mem(
        "bwd: weight_quant_dgrad_rhs",
        lambda: triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm(
            B_t, block_size=block_size, dtype=fp8
        ),
        _io_bytes(B_t),
        B_dgrad_fp8,
        B_dgrad_scale,
    )

    dgrad_gemm_bytes = (
        _io_bytes(gout_fp8, gout_scale, B_dgrad_fp8, B_dgrad_scale) + M * K * 2
    )
    timers.time_gemm(
        "bwd: deepgemm_grouped_mm_dgrad",
        lambda: deepgemm_blockwise_scaled_grouped_mm(
            gout_fp8,
            B_dgrad_fp8,
            gout_scale,
            inputs.recipe_a,
            B_dgrad_scale,
            inputs.recipe_b,
            inputs.offs,
            out_dtype,
            block_size,
            offset_plan=inputs.offset_plan,
        ),
        2.0 * M * N * K,
        dgrad_gemm_bytes,
    )


def _bench_wgrad(timers, inputs, block_size):
    """Time the wgrad (K-grouped) path: operand quant then grouped GEMM."""
    M, N, K, E = inputs.M, inputs.N, inputs.K, inputs.E
    A, grad_out = inputs.A, inputs.grad_out
    fp8, out_dtype = inputs.fp8, inputs.out_dtype
    offset_plan, group_sizes = inputs.offset_plan, inputs.group_sizes

    # Build the per-block quant metadata once, outside the timed region (it is a
    # host-side python loop, not a kernel), so each quant row times only its
    # kernel. Each operand picks the direct K-grouped quant for wide dims and
    # TorchAO's transposed quant otherwise (see _DEEPGEMM_DIRECT..._MIN_DIM).
    lhs_md = (
        offset_plan.k_quant_metadata(block_size, N)
        if _should_quantize_k_grouped_directly(N)
        else None
    )
    rhs_md = (
        offset_plan.k_quant_metadata(block_size, K)
        if _should_quantize_k_grouped_directly(K)
        else None
    )
    lhs_op = _quantize_wgrad_lhs(
        grad_out, offset_plan.group_end_offsets, group_sizes, block_size, fp8, lhs_md
    )
    rhs_op = _quantize_wgrad_rhs(
        A, offset_plan.group_end_offsets, group_sizes, block_size, fp8, rhs_md
    )
    lhs_path = "direct" if _should_quantize_k_grouped_directly(N) else "transposed"
    rhs_path = "direct" if _should_quantize_k_grouped_directly(K) else "transposed"
    timers.time_mem(
        f"bwd: wgrad_quant_lhs(grad_out) [{lhs_path}]",
        lambda: _quantize_wgrad_lhs(
            grad_out,
            offset_plan.group_end_offsets,
            group_sizes,
            block_size,
            fp8,
            lhs_md,
        ),
        _io_bytes(grad_out),
        lhs_op.data,
        lhs_op.scale,
    )
    timers.time_mem(
        f"bwd: wgrad_quant_rhs(A) [{rhs_path}]",
        lambda: _quantize_wgrad_rhs(
            A, offset_plan.group_end_offsets, group_sizes, block_size, fp8, rhs_md
        ),
        _io_bytes(A),
        rhs_op.data,
        rhs_op.scale,
    )

    wgrad_plan = prepare_deepgemm_wgrad_plan(grad_out, A, offset_plan, block_size, fp8)
    assert wgrad_plan is not None, "wgrad plan requires block-aligned groups"
    # wgrad mem traffic: read lhs + rhs fp8 data + scales, read FP32 accum seed,
    # write FP32 (E,N,K) output. The two FP32 (E,N,K) buffers dominate.
    wgrad_gemm_bytes = (
        _io_bytes(
            wgrad_plan.lhs.data,
            wgrad_plan.lhs.scale,
            wgrad_plan.rhs.data,
            wgrad_plan.rhs.scale,
        )
        + 2 * E * N * K * 4  # FP32 accum read + FP32 out write
    )
    timers.time_gemm(
        "bwd: deepgemm_grouped_mm_wgrad",
        lambda: deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            offset_plan,
            out_dtype,
            block_size,
        ),
        2.0 * M * N * K,
        wgrad_gemm_bytes,
    )


@dataclass
class _ShapeInputs:
    """Shared inputs + plans reused across the fwd/dgrad/wgrad benchmarks."""

    M: int
    N: int
    K: int
    E: int
    A: torch.Tensor
    grad_out: torch.Tensor
    B_t: torch.Tensor
    offs: torch.Tensor
    offset_plan: object
    group_sizes: object
    recipe_a: object
    recipe_b: object
    fp8: object
    out_dtype: object


def _build_shape_inputs(shape: Shape, block_size: int, jagged: bool) -> _ShapeInputs:
    """Materialize the tensors, offsets, and plan a single shape needs once."""
    M, N, K, E = shape.M, shape.N, shape.K, shape.E
    out_dtype = torch.bfloat16
    fp8 = e4m3_dtype

    # Inputs in the layouts the MoE op uses.
    A = torch.randn(M, K, dtype=out_dtype, device=device)
    grad_out = torch.randn(M, N, dtype=out_dtype, device=device)
    # B_t logical (E, K, N) in per-expert column-major layout.
    weight = torch.randn(E, N, K, dtype=out_dtype, device=device)
    B_t = weight.contiguous().transpose(-2, -1)

    offs = _make_offsets(E, M, block_size, jagged)
    offset_plan = build_deepgemm_grouped_offset_plan(offs, num_rows=M)
    # Touch the cached host-side group sizes once, outside every timed region,
    # so the K-grouped quant/GEMM kernels are not charged for the D2H sync.
    group_sizes = offset_plan.group_sizes

    return _ShapeInputs(
        M=M,
        N=N,
        K=K,
        E=E,
        A=A,
        grad_out=grad_out,
        B_t=B_t,
        offs=offs,
        offset_plan=offset_plan,
        group_sizes=group_sizes,
        recipe_a=_scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        recipe_b=_scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
        fp8=fp8,
        out_dtype=out_dtype,
    )


def _bench_shape(
    shape: Shape, block_size: int, jagged: bool
) -> List[KernelMeasurement]:
    inputs = _build_shape_inputs(shape, block_size, jagged)
    timers = _KernelTimers(measurements=[])

    _bench_forward(timers, inputs, block_size)
    _bench_dgrad(timers, inputs, block_size)
    _bench_wgrad(timers, inputs, block_size)

    return timers.measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        # M, N, K, E. DeepSeek-V3 MoE FFN dims at a couple of token counts.
        default=[
            "16384,2048,7168,8",
            "32768,2048,7168,8",
            "16384,7168,2048,8",
            "32768,7168,2048,8",
        ],
        help="Comma-separated M,N,K,E groups.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--jagged",
        action="store_true",
        help="Use skewed per-expert token counts instead of balanced (default).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not is_deep_gemm_available():
        raise RuntimeError(
            "DeepGEMM is not importable; this benchmark targets the DeepGEMM backend."
        )

    gpu_name = torch.cuda.get_device_name(0)
    specs = lookup_roofline_specs(gpu_name)
    if specs is None:
        raise RuntimeError(f"No roofline specs for GPU: {gpu_name}")

    device_peak_bw = peak_mem_bw_from_device_properties()
    peak_bw = device_peak_bw or specs["peak_mem_bw_bytes_sec"]
    bw_source = "cuda_device_properties" if device_peak_bw else "roofline_utils"
    ach_bw = peak_bw * specs.get("pct_achievable_mem_bw", 1.0)
    peak_tops = specs["fp8_peak_tops"]
    ach_tops = peak_tops * specs.get("pct_achievable_gemm_tops", 1.0)

    print(f"GPU: {gpu_name}")
    print(
        f"Mem BW: peak {peak_bw / 1e9:.0f} GB/s (source: {bw_source}), "
        f"achievable {ach_bw / 1e9:.0f} GB/s "
        f"({specs.get('pct_achievable_mem_bw', 1.0) * 100:.0f}% of peak)"
    )
    print(
        f"FP8 compute: peak {peak_tops / 1e12:.0f} TFLOP/s, "
        f"achievable {ach_tops / 1e12:.0f} TFLOP/s "
        f"({specs.get('pct_achievable_gemm_tops', 1.0) * 100:.0f}% of peak)"
    )
    print(
        f"Tokens: {'jagged (skewed)' if args.jagged else 'balanced'}; "
        "128-aligned (no padding); DeepGEMM backend.\n"
    )

    torch.manual_seed(123)
    for shape_str in args.shapes:
        M, N, K, E = (int(x) for x in shape_str.split(","))
        shape = Shape(M, N, K, E)
        measurements = _bench_shape(shape, args.block_size, args.jagged)

        print(f"=== M={M} N={N} K={K} E={E} ===")
        rows = []
        for m in measurements:
            gbps = m.bytes_moved / 1e9 / (m.us * 1e-6)
            bw_pct = 100.0 * gbps * 1e9 / ach_bw
            if m.kind == Kind.MEM:
                rows.append(
                    [
                        m.name,
                        f"{m.us:.1f}",
                        "-",
                        "-",
                        f"{gbps:.0f}",
                        f"{bw_pct:.1f}",
                    ]
                )
            else:
                tflops = m.flops / 1e12 / (m.us * 1e-6)
                compute_pct = 100.0 * tflops * 1e12 / ach_tops
                rows.append(
                    [
                        m.name,
                        f"{m.us:.1f}",
                        f"{tflops:.0f}",
                        f"{compute_pct:.1f}",
                        f"{gbps:.0f}",
                        f"{bw_pct:.1f}",
                    ]
                )
        print(
            tabulate(
                rows,
                headers=[
                    "kernel",
                    "us",
                    "TFLOP/s",
                    "%ach_compute",
                    "GB/s",
                    "%ach_bw",
                ],
                tablefmt="github",
            )
        )
        print()


if __name__ == "__main__":
    main()
