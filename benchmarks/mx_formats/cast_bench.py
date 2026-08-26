# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import fire
import torch
import triton
from triton.testing import do_bench

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

torch.manual_seed(0)

bytes_per_el_bf16 = 2
bytes_per_el_fp8 = 1


def scale_dim0_reference(x_hp, block_size) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_hp.is_contiguous()
    x_hp_d0_block = x_hp.reshape(-1, block_size)
    x_hp_d0_block_abs = x_hp_d0_block.abs()
    amax_dim0 = torch.amax(x_hp_d0_block_abs, dim=1).unsqueeze(1)
    x_hp_d0_block_normalized = x_hp_d0_block / amax_dim0
    x_hp_d0_normalized = x_hp_d0_block_normalized.reshape(x_hp.shape)
    return x_hp_d0_normalized, amax_dim0


def scale_dim1_reference(x_hp, block_size) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_hp.is_contiguous()
    x_hp_d1 = x_hp.t().contiguous()
    x_hp_d1_block = x_hp_d1.reshape(-1, block_size)
    x_hp_d1_block_abs = x_hp_d1_block.abs()
    amax_dim1 = torch.amax(x_hp_d1_block_abs, dim=1).unsqueeze(1)
    x_hp_d1_block_normalized = x_hp_d1_block / amax_dim1
    x_hp_d1_normalized = x_hp_d1_block_normalized.reshape(x_hp_d1.shape)
    return x_hp_d1_normalized, amax_dim1


def scale_dim0_dim1_reference(
    x_hp: torch.Tensor, block_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalize across dim0
    x_hp_d0_normalized, amax_dim0 = scale_dim0_reference(x_hp, block_size)
    # normalize across dim1
    x_hp_d1_normalized, amax_dim1 = scale_dim1_reference(x_hp, block_size)
    return x_hp_d0_normalized, x_hp_d1_normalized.t(), amax_dim0, amax_dim1


def to_mx_dim0_reference(
    x_hp,
    block_size,
    scaling_mode=ScaleCalculationMode.FLOOR,
    target_dtype=torch.float8_e4m3fn,
):
    scale_d0, data_d0 = to_mx(x_hp, target_dtype, block_size, scaling_mode=scaling_mode)
    return data_d0, scale_d0


def to_mx_dim1_reference(
    x_hp,
    block_size,
    scaling_mode=ScaleCalculationMode.FLOOR,
    target_dtype=torch.float8_e4m3fn,
):
    x_hp = x_hp.t().contiguous()
    scale_d1, data_d1 = to_mx(x_hp, target_dtype, block_size, scaling_mode=scaling_mode)
    return data_d1.t(), scale_d1


def to_nvfp4_reference(x_hp):
    nvfp4_tensor = NVFP4Tensor.to_nvfp4(x_hp, use_triton_kernel=False)
    return nvfp4_tensor.qdata, nvfp4_tensor.scale


def to_nvfp4_reference_triton_swizzle(x_hp):
    per_tensor_scale = torch.tensor(1.0, dtype=torch.float32, device=x_hp.device)
    nvfp4_tensor = NVFP4Tensor.to_nvfp4(
        x_hp,
        per_tensor_scale=per_tensor_scale,
        use_triton_kernel=True,
        is_swizzled_scales=True,
    )
    return nvfp4_tensor.qdata, nvfp4_tensor.scale


def benchmark_cuda_function_in_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def _bytes_rw_bf16(*tensors):
    return sum(t.numel() for t in tensors) * bytes_per_el_bf16


def _bytes_rw_quant(x, y, s):
    # bf16 input read + fp8-packed output (data + scale) written
    bytes_r = x.numel() * bytes_per_el_bf16
    bytes_w = (y.numel() + s.numel()) * bytes_per_el_fp8
    return bytes_r + bytes_w


def _bench(fn, x, *bench_args):
    """Warm up ``fn`` twice, then benchmark it and return the median time in us.

    ``fn`` is invoked as ``fn(x, *bench_args)``. The warmup and the timed call
    use the same arguments.
    """
    out = fn(x, *bench_args)
    for _ in range(2):
        fn(x, *bench_args)
    time_us = benchmark_cuda_function_in_microseconds(fn, x, *bench_args)
    return out, time_us


def _bench_memcpy(x, BLOCK_SIZE):
    # Baseline memcpy benchmark to establish max achievable bandwidth
    y = torch.randn_like(x)

    # Warmup
    for _ in range(2):
        y.copy_(x)

    time_us = benchmark_cuda_function_in_microseconds(
        lambda src, dst: dst.copy_(src),
        x,
        y,
    )

    # bytes_read + bytes_written
    bytes_rw = 2 * x.numel() * bytes_per_el_bf16
    return time_us, bytes_rw


def _bench_scale_reference(x, BLOCK_SIZE, ref_fn):
    """Benchmark a (compiled) bf16 scaling reference that returns one or more
    tensors. Byte accounting reads/writes all tensors as bf16.
    """
    fn = torch.compile(ref_fn)
    outs, time_us = _bench(fn, x, BLOCK_SIZE)
    for t in outs:
        assert t.dtype == torch.bfloat16
    return time_us, _bytes_rw_bf16(x, *outs)


def _bench_quant(x, BLOCK_SIZE, make_fn, expected_dtypes):
    """Benchmark a quantization op producing (data, scale).

    ``make_fn`` builds the callable to benchmark (e.g. wrapping torch.compile);
    it is invoked as ``make_fn()`` and the result is called as
    ``fn(x, BLOCK_SIZE)``. ``expected_dtypes`` is (data_dtype, scale_dtype).
    """
    fn = make_fn()
    (y, s), time_us = _bench(fn, x, BLOCK_SIZE)
    assert y.dtype == expected_dtypes[0]
    assert s.dtype == expected_dtypes[1]
    return time_us, _bytes_rw_quant(x, y, s)


# --- scaling-reference (bf16) modes -----------------------------------------
# mode -> reference function whose (compiled) output is timed with bf16 accounting.
_SCALE_REFERENCE_MODES = {
    "dim0": scale_dim0_reference,
    "dim1": scale_dim1_reference,
    "dim0_dim1": scale_dim0_dim1_reference,
}


def _make_scale_reference_bench(ref_fn):
    def bench(x, BLOCK_SIZE):
        return _bench_scale_reference(x, BLOCK_SIZE, ref_fn)

    return bench


# --- quantization modes ------------------------------------------------------
# Each entry describes how to benchmark one mode. ``make_fn`` receives BLOCK_SIZE
# and returns the callable to time; ``bench_args`` are extra args passed after x;
# ``dtypes`` is the expected (data, scale) output dtype pair.

_FP8 = (torch.float8_e4m3fn, torch.float8_e8m0fnu)
_FP4 = (torch.uint8, torch.float8_e8m0fnu)
_NVFP4 = (torch.uint8, torch.float8_e4m3fn)


def _to_mx_dim0_fn(scaling_mode=ScaleCalculationMode.FLOOR, target_dtype=None):
    compiled = torch.compile(to_mx_dim0_reference)

    def fn(x, block_size):
        if target_dtype is not None:
            return compiled(x, block_size, scaling_mode, target_dtype=target_dtype)
        return compiled(x, block_size, scaling_mode)

    return fn


def _to_mx_dim1_fn(scaling_mode=ScaleCalculationMode.FLOOR):
    compiled = torch.compile(to_mx_dim1_reference)

    # NOTE: matching the original benchmark, the rceil variant is only used to
    # build the reference output; the timed calls use the default scaling mode.
    def fn(x, block_size):
        return compiled(x, block_size)

    return fn


def _triton_mxfp8_fn(triton_fn, scaling_mode):
    def fn(x, block_size):
        return triton_fn(x, inner_block_size=block_size, scaling_mode=scaling_mode)

    return fn


def _mxfp8_cuda_fn(scaling_mode):
    def fn(x, block_size):
        from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda

        _, y, _, s = mxfp8_quantize_cuda(
            x, rowwise=False, colwise=True, scaling_mode=scaling_mode
        )
        return y, s

    return fn


def _cutedsl_1x32_fn(scaling_mode):
    def fn(x, block_size):
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mxfp8_quantize_2d_1x32_cutedsl,
        )

        return mxfp8_quantize_2d_1x32_cutedsl(
            x, block_size=block_size, scaling_mode=scaling_mode
        )

    return fn


def _cutedsl_32x1_fn(scaling_mode):
    def fn(x, block_size):
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mxfp8_quantize_2d_32x1_cutedsl,
        )

        return mxfp8_quantize_2d_32x1_cutedsl(
            x,
            block_size=block_size,
            scaling_mode=scaling_mode,
            blocked_scale_output=True,
        )

    return fn


def _nvfp4_fn(x, block_size):
    compiled = torch.compile(to_nvfp4_reference)
    return compiled(x, use_triton_kernel=False)


def _nvfp4_triton_swizzle_fn(x, block_size):
    return to_nvfp4_reference_triton_swizzle(x)


# mode -> (make_fn() -> callable(x, BLOCK_SIZE), expected (data, scale) dtypes).
# make_fn is deferred so per-mode setup (torch.compile, imports) happens inside
# the benchmark rather than at import time.
_QUANT_MODES = {
    "dim0_mxfp8_floor": (_to_mx_dim0_fn, _FP8),
    "dim0_mxfp4_floor": (
        lambda: _to_mx_dim0_fn(target_dtype=torch.float4_e2m1fn_x2),
        _FP4,
    ),
    "dim0_mxfp8_rceil": (lambda: _to_mx_dim0_fn(ScaleCalculationMode.RCEIL), _FP8),
    "dim0_mxfp8_triton_floor": (
        lambda: _triton_mxfp8_fn(triton_to_mxfp8_dim0, "floor"),
        _FP8,
    ),
    "dim0_mxfp8_triton_rceil": (
        lambda: _triton_mxfp8_fn(triton_to_mxfp8_dim0, "rceil"),
        _FP8,
    ),
    "dim0_nvfp4": (lambda: _nvfp4_fn, _NVFP4),
    "dim0_nvfp4_triton_swizzle": (lambda: _nvfp4_triton_swizzle_fn, _NVFP4),
    "dim1_mxfp8_floor": (_to_mx_dim1_fn, _FP8),
    "dim1_mxfp8_rceil": (lambda: _to_mx_dim1_fn(ScaleCalculationMode.RCEIL), _FP8),
    "dim1_mxfp8_triton_floor": (
        lambda: _triton_mxfp8_fn(triton_to_mxfp8_dim1, "floor"),
        _FP8,
    ),
    "dim1_mxfp8_triton_rceil": (
        lambda: _triton_mxfp8_fn(triton_to_mxfp8_dim1, "rceil"),
        _FP8,
    ),
    "dim1_mxfp8_cuda_floor": (lambda: _mxfp8_cuda_fn("floor"), _FP8),
    "dim1_mxfp8_cuda_rceil": (lambda: _mxfp8_cuda_fn("rceil"), _FP8),
    "dim0_mxfp8_cutedsl_2d_floor": (lambda: _cutedsl_1x32_fn("floor"), _FP8),
    "dim0_mxfp8_cutedsl_2d_rceil": (lambda: _cutedsl_1x32_fn("rceil"), _FP8),
    "dim1_mxfp8_cutedsl_2d_floor": (lambda: _cutedsl_32x1_fn("floor"), _FP8),
    "dim1_mxfp8_cutedsl_2d_rceil": (lambda: _cutedsl_32x1_fn("rceil"), _FP8),
}


def _make_quant_bench(mode):
    make_fn, dtypes = _QUANT_MODES[mode]

    def bench(x, BLOCK_SIZE):
        return _bench_quant(x, BLOCK_SIZE, make_fn, dtypes)

    return bench


# Maps each supported mode to the function that runs its benchmark. Each handler
# takes (x, BLOCK_SIZE) and returns (time_us, bytes_read_plus_written).
_MODE_TO_BENCH = {
    "memcpy": _bench_memcpy,
    **{
        mode: _make_scale_reference_bench(ref_fn)
        for mode, ref_fn in _SCALE_REFERENCE_MODES.items()
    },
    **{mode: _make_quant_bench(mode) for mode in _QUANT_MODES},
}


def run(
    M: int = 16384,
    K: int = 16384,
    BLOCK_SIZE: int = 32,
    mode: str = "dim0",
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")
    print(f"mode: {mode}")
    assert mode in _MODE_TO_BENCH, f"unknown mode {mode}"

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 1000

    time_us, bytes_rw = _MODE_TO_BENCH[mode](x, BLOCK_SIZE)
    bps = bytes_rw / (time_us / 1e6)

    print("time_us", time_us)
    print("mem_bw_gbps", bps / 1e9)


if __name__ == "__main__":
    fire.Fire(run)
