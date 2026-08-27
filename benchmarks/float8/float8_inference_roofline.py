# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a script to estimate the benefit from converting a `torch.nn.Linear`
layer to float8 given a single saturated GPU, by estimating the difference
in e2e GPU kernel time between:
1. bf16 gemms in fwd and
2. float8 gemms in fwd and float8 overhead

The gemm times are estimated either from direct measurements via benchmarks,
or with a roofline estimation based on TOPS and peak compute bandwidth of an
NVIDIA H100 or B200.

The float8 overhead times are estimated by counting memory reads and writes
based on the specified float8 scaling, and estimating that we can achieve
a certain % of machine peak memory bandwidth when performing these reads and writes.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

import fire
import pandas as pd
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tabulate import tabulate
from torch.nn.functional import ScalingType, SwizzleType
from torch.profiler import ProfilerActivity, profile
from utils import (
    get_gpu_kernel_conv_time_s,
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
    profiler_output_to_filtered_time_by_kernel_name,
)

import torchao
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.training.roofline_utils import (
    get_inference_bf16_activation_mem_sympy,
    get_inference_float8_mem_sympy,
    get_inference_gemm_time_sympy,
)
from torchao.utils import _is_mslk_available, is_MI300, is_sm_at_least_100

# Import mslk.conv to register the fp8 conv operator
if _is_mslk_available():
    import mslk.conv  # noqa: F401


@dataclass(frozen=True)
class ConvGeometry:
    """Geometry describing a single conv (or its implicit-GEMM equivalent).

    These values always travel together through the conv helpers, so they are
    grouped into one object instead of being passed as a long parameter list.
    ``batch``/``in_channels``/``out_channels`` correspond to the GEMM
    ``M``/``K``/``N`` used elsewhere in this script.
    """

    op_name: str
    batch: int
    in_channels: int
    out_channels: int
    kernel_size: Optional[int]
    D: Optional[int] = None
    H: Optional[int] = None
    W: Optional[int] = None
    stride: int = 1
    padding: int = 0


@dataclass(frozen=True)
class Shape:
    """A concrete GEMM shape ``(M, K, N)`` for a single benchmark iteration.

    For conv these correspond to ``(batch, in_channels, out_channels)``.
    Grouped so the shape triple travels as one value instead of three params.
    """

    M: int
    K: int
    N: int

    def as_tuple(self) -> tuple:
        return (self.M, self.K, self.N)


@dataclass(frozen=True)
class ShapeConfig:
    """Selects how benchmark shapes are generated.

    ``shape_gen_name`` picks the generator and ``M``/``K``/``N`` override the
    generated dimensions when the ``custom`` generator is used.
    """

    shape_gen_name: str = "pow2"
    M: Optional[int] = None
    K: Optional[int] = None
    N: Optional[int] = None


@dataclass(frozen=True)
class ConvConfig:
    """Conv-specific op selection and geometry shared across a run.

    ``batch``/``in_channels``/``out_channels`` are supplied per shape from the
    shape iterator, so only the op name and spatial/kernel geometry live here.
    """

    op_name: str = "linear"
    D: Optional[int] = None
    H: Optional[int] = None
    W: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: int = 1
    padding: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Controls benchmarking behavior and result reporting for a run."""

    do_benchmarks: bool = True
    enable_fusion_modeling: bool = False
    n_limit: Optional[int] = None
    save_profile_traces: bool = False
    skip_printing_detailed_metrics: bool = False
    outfile: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    """All options for a single ``float8_inference_roofline`` run.

    These values form one benchmark configuration, so they are grouped into a
    single value object (with cohesive sub-configs) instead of a long
    parameter list.

    * ``recipe_name``: quantization recipe (tensorwise, rowwise, mxfp8*,
      mxfp4*, nvfp4*).
    * ``shapes``: shape generation config (`ShapeConfig`).
    * ``conv``: op selection and conv geometry (`ConvConfig`).
    * ``bench``: benchmarking and reporting config (`BenchmarkConfig`).
    """

    recipe_name: str = "tensorwise"
    shapes: ShapeConfig = field(default_factory=ShapeConfig)
    conv: ConvConfig = field(default_factory=ConvConfig)
    bench: BenchmarkConfig = field(default_factory=BenchmarkConfig)


# Frozen (immutable) instances are safe to share as default arguments.
_DEFAULT_RUN_CONFIG = RunConfig()


@torch.no_grad()
def get_gpu_kernel_time(m, x, trace_filename=None):
    # warm up
    for _ in range(2):
        __ = m(x)

    # capture a profiling run
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    n_iter = 5
    with profile(activities=activities) as prof:
        for _ in range(n_iter):
            __ = m(x)
            torch.cuda.synchronize()

    # save a trace, if requested
    if trace_filename is not None:
        print(f"exporting trace to {trace_filename}")
        prof.export_chrome_trace(trace_filename)

    # get the gpu kernel time and aggregate it
    num_leaf_tensors = 1 + len(list(m.parameters()))
    ref_times = profiler_output_to_filtered_time_by_kernel_name(
        prof, n_iter, num_leaf_tensors
    )
    total_time_s = sum(v for v in ref_times.values()) / 1e6 / n_iter
    return total_time_s


def get_gemm_times(
    M: int,
    K: int,
    N: int,
    fast_accum: bool,
    recipe_name: Optional[str],
):
    device = torch.device("cuda")

    # bf16 time
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    bf16_time_s = get_gpu_kernel_gemm_time_s(torch.mm, x_bf16, w_bf16)

    if recipe_name in (
        "mxfp4_cutlass",
        "nvfp4",
        "nvfp4_static",
        "nvfp4_no_global_scale",
    ):
        d1, d2, d3 = torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, torch.bfloat16
        A = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8).view(
            d1
        )
        B = (
            torch.randint(0, 255, (K // 2, N), device=device, dtype=torch.uint8)
            .t()
            .contiguous()
            .t()
            .view(d2)
        )
    else:
        e4m3_dtype = torch.float8_e4m3fn
        if torch.version.hip and torch.cuda.is_available() and is_MI300():
            e4m3_dtype = torch.float8_e4m3fnuz
        d1, d2, d3 = e4m3_dtype, e4m3_dtype, torch.bfloat16
        A = torch.randint(0, 255, (M, K), device=device, dtype=torch.uint8).view(d1)
        B = (
            torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
            .view(d2)
            .t()
            .contiguous()
            .t()
        )

    if recipe_name == "rowwise":
        scale_a = torch.ones(M, 1, device=device)
        scale_b = torch.ones(1, N, device=device)
    elif recipe_name == "mxfp8_cublas":
        scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)
    elif recipe_name == "mxfp4_cutlass":
        scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)
    elif recipe_name in ("nvfp4", "nvfp4_static", "nvfp4_no_global_scale"):
        scale_a = torch.ones(M, K // 16, device=device, dtype=torch.float8_e4m3fn)
        scale_b = torch.ones(N, K // 16, device=device, dtype=torch.float8_e4m3fn)
        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)

    else:
        assert False, "unsupported"

    def do_matmul(A, B):
        if recipe_name == "mxfp4_cutlass":
            return F.scaled_mm(
                A,
                B,
                scale_a=scale_a,
                scale_recipe_a=ScalingType.BlockWise1x32,
                scale_b=scale_b,
                scale_recipe_b=ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                output_dtype=d3,
            )
        if recipe_name in ("nvfp4", "nvfp4_static", "nvfp4_no_global_scale"):
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=False
            )
        else:
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
            )

    f8_time_s = get_gpu_kernel_gemm_time_s(do_matmul, A, B)

    return bf16_time_s, f8_time_s


def get_conv_times(
    geometry: ConvGeometry,
    fast_accum: bool = True,
    recipe_name: Optional[str] = None,
):
    """
    Get conv kernel times for bf16 and fp8 operations.
    Similar to get_gemm_times but for conv operations.

    This measures only the conv kernel time itself, without quantization overhead.
    """
    op_name = geometry.op_name
    batch = geometry.batch
    in_channels = geometry.in_channels
    out_channels = geometry.out_channels
    kernel_size = geometry.kernel_size
    D, H, W = geometry.D, geometry.H, geometry.W
    stride, padding = geometry.stride, geometry.padding

    device = torch.device("cuda")

    # Create input tensors
    if op_name == "conv2d":
        x_bf16 = torch.randn(
            batch, in_channels, H, W, dtype=torch.bfloat16, device=device
        )
        w_bf16 = torch.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_fn = torch.nn.functional.conv2d
        conv_kwargs = {"stride": stride, "padding": padding}
    elif op_name == "conv3d":
        x_bf16 = torch.randn(
            batch, in_channels, D, H, W, dtype=torch.bfloat16, device=device
        )
        w_bf16 = torch.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            kernel_size,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_fn = torch.nn.functional.conv3d
        conv_kwargs = {"stride": stride, "padding": padding}
    else:
        raise ValueError(f"Unsupported op_name: {op_name}")

    # Measure bf16 conv time
    bf16_time_s = get_gpu_kernel_conv_time_s(
        lambda x, w: conv_fn(x, w, **conv_kwargs), x_bf16, w_bf16
    )

    # Measure fp8 conv time using mslk operator
    # Note: Only tensorwise recipe is supported for conv (validated in run())

    # Validate recipe for conv operations (defense in depth)
    _SUPPORTED_CONV_RECIPES = ["tensorwise"]
    if recipe_name not in _SUPPORTED_CONV_RECIPES:
        raise ValueError(
            f"Recipe '{recipe_name}' is not supported for conv operations. "
            f"Supported recipes: {_SUPPORTED_CONV_RECIPES}"
        )

    # Check if mslk fp8 conv is available
    if not _is_mslk_available():
        raise RuntimeError(
            "mslk fp8 conv operator is not available. "
            "To skip fp8 conv benchmarking, set --do_benchmarks=False to run roofline model only."
        )

    # Check if op is supported
    if op_name == "conv2d":
        raise NotImplementedError(
            "mslk fp8 conv2d is not yet implemented. "
            "To skip fp8 conv benchmarking, set --do_benchmarks=False to run roofline model only."
        )
    elif op_name != "conv3d":
        raise ValueError(f"Unsupported op_name: {op_name}")

    # Check kernel_size constraint
    # Note: kernel_size=1 causes ambiguous memory layout where tensors are both
    # contiguous and channels_last_3d, which the mslk operator cannot handle correctly
    if kernel_size == 1:
        raise ValueError(
            "kernel_size=1 is not supported for fp8 conv3d benchmarking. "
            "The mslk operator requires kernel_size > 1 to correctly determine memory layout. "
            "To skip fp8 conv benchmarking for this configuration, set --do_benchmarks=False to run roofline model only."
        )

    # Measure fp8 conv3d time
    # Create fp8 tensors for conv3d
    e4m3_dtype = torch.float8_e4m3fn
    if torch.version.hip and torch.cuda.is_available() and is_MI300():
        e4m3_dtype = torch.float8_e4m3fnuz

    # Quantize tensors to fp8
    x_scale = x_bf16.abs().max() / torch.finfo(e4m3_dtype).max
    w_scale = w_bf16.abs().max() / torch.finfo(e4m3_dtype).max

    # Convert to channels_last_3d format for conv3d
    x_bf16_cl = x_bf16.contiguous(memory_format=torch.channels_last_3d)
    w_bf16_cl = w_bf16.contiguous(memory_format=torch.channels_last_3d)

    x_fp8 = (x_bf16_cl / x_scale).to(e4m3_dtype)
    w_fp8 = (w_bf16_cl / w_scale).to(e4m3_dtype)

    # mslk operator now supports channels_first shape with channels_last_3d memory format
    # No permute needed - tensors are already in correct format
    # Input: (N, C_in, D, H, W) in channels_last_3d memory format
    # Weight: (C_out, C_in, K1, K2, K3) in channels_last_3d memory format

    # mslk expects a combined scale tensor
    combined_scale = float(x_scale * w_scale)
    scale_tensor = torch.tensor([combined_scale], device=device, dtype=torch.float32)

    # Use mslk fp8 conv operator
    # Signature: f8f8bf16_conv(activation, filter, scale, padding, stride, dilation)
    f8_time_s = get_gpu_kernel_conv_time_s(
        lambda x, w: torch.ops.mslk.f8f8bf16_conv(
            x,
            w,
            scale_tensor,
            padding if isinstance(padding, (list, tuple)) else [padding] * 3,
            stride if isinstance(stride, (list, tuple)) else [stride] * 3,
            [1, 1, 1],  # dilation
        ),
        x_fp8,
        w_fp8,
    )

    return bf16_time_s, f8_time_s


def get_conv_equivalent_gemm_dims(geometry: ConvGeometry):
    """
    Get equivalent GEMM dimensions for a conv operation using analytical calculation.

    Conv operations can be expressed as implicit GEMM. This function computes
    the equivalent GEMM dimensions without creating any tensors.

    Args:
        geometry: Conv geometry (op name, batch, channels, kernel size, spatial
            dimensions, stride, and padding).

    Returns:
        Tuple[int, int, int]: (gemm_M, gemm_K, gemm_N)
            gemm_M: Number of output spatial positions (batch * spatial_output_size)
            gemm_K: Size of each filter (in_channels * kernel_volume)
            gemm_N: Number of filters (out_channels)
    """
    op_name = geometry.op_name
    batch = geometry.batch
    in_channels = geometry.in_channels
    out_channels = geometry.out_channels
    kernel_size = geometry.kernel_size
    D, H, W = geometry.D, geometry.H, geometry.W
    stride, padding = geometry.stride, geometry.padding

    if op_name == "conv2d":
        # Output spatial dimensions
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1

        gemm_M = batch * H_out * W_out
        gemm_K = in_channels * kernel_size * kernel_size
        gemm_N = out_channels

    elif op_name == "conv3d":
        # Output spatial dimensions
        D_out = (D + 2 * padding - kernel_size) // stride + 1
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1

        gemm_M = batch * D_out * H_out * W_out
        gemm_K = in_channels * kernel_size * kernel_size * kernel_size
        gemm_N = out_channels

    else:
        raise ValueError(f"Unsupported op_name: {op_name}")

    return gemm_M, gemm_K, gemm_N


def _create_model_and_input(
    geometry: ConvGeometry,
    enable_fusion_modeling: bool,
) -> tuple[nn.Module, torch.Tensor]:
    """
    Build the model and its corresponding input tensor for benchmarking.
    """
    op_name = geometry.op_name
    batch = geometry.batch
    in_channels = geometry.in_channels
    out_channels = geometry.out_channels
    kernel_size = geometry.kernel_size
    D, H, W = geometry.D, geometry.H, geometry.W
    stride, padding = geometry.stride, geometry.padding

    def _stack_layers_conv(
        core_layer: nn.Module, add_post_relu: bool = False
    ) -> nn.Sequential:
        layers = []
        if enable_fusion_modeling:
            layers.append(nn.ReLU())
        layers.append(core_layer)
        if enable_fusion_modeling:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    if op_name == "conv2d":
        core_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        memory_format = torch.channels_last
        x = torch.randn(
            batch, in_channels, H, W, dtype=torch.bfloat16, device="cuda"
        ).to(memory_format=memory_format)
        m_orig = _stack_layers_conv(core_layer)
    elif op_name == "conv3d":
        core_layer = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        memory_format = torch.channels_last_3d
        x = torch.randn(
            batch, in_channels, D, H, W, dtype=torch.bfloat16, device="cuda"
        ).to(memory_format=memory_format)
        m_orig = _stack_layers_conv(core_layer)
    else:
        if not enable_fusion_modeling:
            m_orig = nn.Sequential(nn.Linear(in_channels, out_channels, bias=False))
        else:
            m_orig = nn.Sequential(
                nn.ReLU(), nn.Linear(in_channels, out_channels, bias=False)
            )
        memory_format = None
        x = torch.randn(
            batch, in_channels, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_()

    if memory_format is not None:
        m_orig = m_orig.to(memory_format=memory_format)
    m_orig = m_orig.cuda().bfloat16()

    return m_orig, x


@dataclass(frozen=True)
class RooflineModel:
    """The roofline sympy model: its symbols and the four time expressions.

    These are built once per run and evaluated per shape, so they travel as one
    object (keeping the evaluator's parameter list short).
    """

    symbols: tuple  # (M, K, N) sympy symbols
    bf16_gemm: object
    bf16_ovhd: object
    fp8_gemm: object
    fp8_ovhd: object

    def eval_times(self, gemm_vals, bf16_ovhd_vals):
        """Evaluate the roofline expressions at concrete (M, K, N) values.

        ``gemm_vals`` feeds the gemm terms and the fp8 overhead term;
        ``bf16_ovhd_vals`` feeds the bf16 overhead term. For linear they are
        identical; for conv the gemm terms use the equivalent implicit-GEMM dims
        while the bf16 overhead uses the original (batch, in_channels,
        out_channels). Returns the six roofline scalars in results-column order.
        """
        M, K, N = self.symbols

        def _eval(expr, vals):
            # cast from sympy.core.numbers.Float to float for pandas formatting
            return float(expr.subs(M, vals[0]).subs(K, vals[1]).subs(N, vals[2]))

        r_bf16_gemm_time_s = _eval(self.bf16_gemm, gemm_vals)
        r_fp8_gemm_time_s = _eval(self.fp8_gemm, gemm_vals)
        r_bf16_ovhd_time_s = _eval(self.bf16_ovhd, bf16_ovhd_vals)
        r_fp8_ovhd_time_s = _eval(self.fp8_ovhd, gemm_vals)
        r_fp8_gemm_and_ovhd_s = r_fp8_gemm_time_s + r_fp8_ovhd_time_s
        r_speedup = (r_bf16_gemm_time_s + r_bf16_ovhd_time_s) / (
            r_fp8_gemm_time_s + r_fp8_ovhd_time_s
        )
        return (
            r_bf16_gemm_time_s,
            r_fp8_gemm_time_s,
            r_bf16_ovhd_time_s,
            r_fp8_ovhd_time_s,
            r_fp8_gemm_and_ovhd_s,
            r_speedup,
        )


def _conv_geometry_for_shape(conv: ConvConfig, shape: Shape) -> ConvGeometry:
    """Build a per-shape ``ConvGeometry`` from the run's conv config and shape."""
    return ConvGeometry(
        op_name=conv.op_name,
        batch=shape.M,
        in_channels=shape.K,
        out_channels=shape.N,
        kernel_size=conv.kernel_size,
        D=conv.D,
        H=conv.H,
        W=conv.W,
        stride=conv.stride,
        padding=conv.padding,
    )


def _validate_run_ops(recipe_name, conv: ConvConfig):
    """Validate that the requested op/recipe/spatial-dims combination is supported."""
    op_name = conv.op_name
    _SUPPORTED_OPS = ["linear", "conv2d", "conv3d"]
    assert op_name in _SUPPORTED_OPS, (
        f"Unsupported op: {op_name}, supported are: {_SUPPORTED_OPS}"
    )

    is_conv = op_name in ("conv2d", "conv3d")
    if is_conv:
        _SUPPORTED_CONV_RECIPES = ["tensorwise"]
        assert recipe_name in _SUPPORTED_CONV_RECIPES, (
            f"Recipe '{recipe_name}' is not supported for {op_name}. "
            f"Supported recipes for conv operations: {_SUPPORTED_CONV_RECIPES}. "
        )

    if op_name == "conv2d":
        assert conv.H is not None and conv.W is not None, (
            "Expected D, H, W to be specified for conv2d"
        )
        assert conv.kernel_size is not None, (
            "Expected kernel_size to be specified for conv2d"
        )
    elif op_name == "conv3d":
        assert conv.D is not None and conv.H is not None and conv.W is not None, (
            "Expected D, H, W to be specified for conv3d"
        )
        assert conv.kernel_size is not None, (
            "Expected kernel_size to be specified for conv3d"
        )


def _build_roofline_model(symbols, recipe_name, op_name, enable_fusion_modeling):
    """Build the ``RooflineModel`` (fp8/bf16 gemm and overhead time expressions)."""
    M, K, N = symbols
    fp8_ovhd_time_sympy = get_inference_float8_mem_sympy(
        M,
        K,
        N,
        recipe_name,
        # TODO(future): also enable fusion modeling here
    )
    bf16_gemm_time_sympy = get_inference_gemm_time_sympy(M, K, N, torch.bfloat16, None)
    if enable_fusion_modeling and op_name == "linear":
        bf16_ovhd_time_sympy = get_inference_bf16_activation_mem_sympy(M, K, N)
    else:
        # multiply by M to ensure we get a sympy symbol
        bf16_ovhd_time_sympy = M * 0

    if recipe_name and recipe_name.startswith(("nvfp4", "mxfp4")):
        fp8_gemm_time_sympy = get_inference_gemm_time_sympy(
            M, K, N, torch.float4_e2m1fn_x2, recipe_name
        )
    else:
        gemm_recipe_name = "mxfp8" if recipe_name.startswith("mxfp8") else None
        fp8_gemm_time_sympy = get_inference_gemm_time_sympy(
            M, K, N, torch.float8_e4m3fn, gemm_recipe_name
        )
    return RooflineModel(
        symbols=symbols,
        bf16_gemm=bf16_gemm_time_sympy,
        bf16_ovhd=bf16_ovhd_time_sympy,
        fp8_gemm=fp8_gemm_time_sympy,
        fp8_ovhd=fp8_ovhd_time_sympy,
    )


def _build_quant_config(recipe_name):
    """Map a recipe name to its quantization config, plus an optional calibration config.

    Returns ``(config, config_calib)`` where ``config_calib`` is None unless the
    recipe requires a calibration pass before conversion.
    """
    if recipe_name == "tensorwise":
        return Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()), None
    if recipe_name == "rowwise":
        return (
            Float8DynamicActivationFloat8WeightConfig(
                granularity=PerRow(),
                # for now, use TORCH. In the future might be interesting
                # to benchmark AUTO and MSLK.
                kernel_preference=KernelPreference.TORCH,
            ),
            None,
        )
    if recipe_name == "mxfp8_cublas":
        return (
            MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                kernel_preference=KernelPreference.AUTO,
            ),
            None,
        )
    if recipe_name == "mxfp4_cutlass":
        return (
            MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float4_e2m1fn_x2,
                weight_dtype=torch.float4_e2m1fn_x2,
                kernel_preference=KernelPreference.AUTO,
            ),
            None,
        )
    if recipe_name == "nvfp4":
        return (
            NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True),
            None,
        )
    if recipe_name == "nvfp4_no_global_scale":
        return (
            NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=False),
            None,
        )
    if recipe_name == "nvfp4_static":
        config_calib = NVFP4DynamicActivationNVFP4WeightConfig(step="prepare")
        config = NVFP4DynamicActivationNVFP4WeightConfig(step="convert")
        return config, config_calib
    assert False, "unsupported"


def _quantize_model_for_op(m_fp8_dyn, op_name, config):
    """Apply ``quantize_`` to the model, filtering to the layer type for the op."""
    if op_name == "linear":
        quantize_(m_fp8_dyn, config)
    elif op_name == "conv2d":
        _is_conv2d = lambda m, fqn: isinstance(m, torch.nn.Conv2d)
        quantize_(m_fp8_dyn, config, filter_fn=_is_conv2d)
    else:
        _is_conv3d = lambda m, fqn: isinstance(m, torch.nn.Conv3d)
        quantize_(m_fp8_dyn, config, filter_fn=_is_conv3d)


def _roofline_and_gemm_linear(
    roofline: RooflineModel, shape: Shape, recipe_name, do_benchmarks
):
    """Roofline scalars + optional measured gemm times/ratios for a linear shape."""
    vals = shape.as_tuple()
    times = roofline.eval_times(vals, vals)
    r_bf16_gemm_time_s, r_fp8_gemm_time_s = times[0], times[1]

    # if enabled, also measure observed gemm time
    b_bf16_gemm_time_s, b_fp8_gemm_time_s = 0, 0
    rb_bf16_gemm_ratio = -1
    rb_fp8_gemm_ratio = -1
    if do_benchmarks:
        # TODO(future): make the bf16 gemm times exactly match the e2e
        # benchmarks, there is a slight deviation, probably related to gemm
        # operand memory formats/transpositions below not exactly matching
        # what PyTorch core is doing for `torch.mm`
        # input @ weight_t = output
        b_bf16_gemm_time_s, b_fp8_gemm_time_s = get_gemm_times(
            shape.M,
            shape.K,
            shape.N,
            True,
            recipe_name,
        )
        rb_bf16_gemm_ratio = r_bf16_gemm_time_s / b_bf16_gemm_time_s
        rb_fp8_gemm_ratio = r_fp8_gemm_time_s / b_fp8_gemm_time_s

    return times, (
        b_bf16_gemm_time_s,
        b_fp8_gemm_time_s,
        rb_bf16_gemm_ratio,
        rb_fp8_gemm_ratio,
    )


def _roofline_and_gemm_conv(
    roofline: RooflineModel, shape: Shape, conv: ConvConfig, recipe_name, do_benchmarks
):
    """Roofline scalars + optional measured conv kernel times/ratios for a conv shape."""
    # For conv ops, compute equivalent GEMM dimensions
    # shape.M=batch, shape.K=in_channels, shape.N=out_channels
    conv_geometry = _conv_geometry_for_shape(conv, shape)
    gemm_vals = get_conv_equivalent_gemm_dims(conv_geometry)
    times = roofline.eval_times(gemm_vals, shape.as_tuple())
    r_bf16_gemm_time_s, r_fp8_gemm_time_s = times[0], times[1]

    # measure actual conv kernel times (without quant overhead)
    b_bf16_gemm_time_s, b_fp8_gemm_time_s = 0, 0
    rb_bf16_gemm_ratio = -1
    rb_fp8_gemm_ratio = -1
    if do_benchmarks:
        b_bf16_gemm_time_s, b_fp8_gemm_time_s = get_conv_times(
            conv_geometry,
            fast_accum=True,
            recipe_name=recipe_name,
        )
        if b_bf16_gemm_time_s > 0:
            rb_bf16_gemm_ratio = r_bf16_gemm_time_s / b_bf16_gemm_time_s
        if b_fp8_gemm_time_s > 0:
            rb_fp8_gemm_ratio = r_fp8_gemm_time_s / b_fp8_gemm_time_s

    return times, (
        b_bf16_gemm_time_s,
        b_fp8_gemm_time_s,
        rb_bf16_gemm_ratio,
        rb_fp8_gemm_ratio,
    )


def _benchmark_e2e(shape: Shape, conv: ConvConfig, bench: BenchmarkConfig, recipe_name):
    """Measure e2e bf16 and fp8 gpu kernel times for one shape.

    Returns ``(b_bf16_e2e_time_s, b_fp8_e2e_time_s)``; both are 0 when the op is
    an unsupported conv on the current GPU.
    """
    op_name = conv.op_name
    if op_name in ("conv2d", "conv3d") and not is_sm_at_least_100():
        print(
            f"WARNING: Skipping {op_name} benchmarks for shape ({shape.M}, {shape.K}, {shape.N}). "
            f"Float8 convolution requires SM 10.0+ (Blackwell/B100 GPUs). "
            f"Current GPU: {torch.cuda.get_device_name(0)} with SM {torch.cuda.get_device_capability()}. "
            f"Roofline model estimates are still valid."
        )
        return 0, 0

    model_geometry = _conv_geometry_for_shape(conv, shape)
    m_orig, x = _create_model_and_input(model_geometry, bench.enable_fusion_modeling)

    # get the bf16 gpu kernel time
    torch._dynamo.reset()
    m_bf16 = torch.compile(copy.deepcopy(m_orig))

    trace_prefix = f"{bench.outfile}_{shape.M}_{shape.K}_{shape.N}"
    bf16_trace_filename = None
    if bench.save_profile_traces:
        bf16_trace_filename = f"{trace_prefix}_bf16.json"
    b_bf16_e2e_time_s = get_gpu_kernel_time(m_bf16, x, bf16_trace_filename)

    # get the float8 dynamic scaling gpu kernel time
    torch._dynamo.reset()

    config, config_calib = _build_quant_config(recipe_name)

    m_fp8_dyn = copy.deepcopy(m_orig)

    if config_calib is not None:
        # calibrate with sample data
        # this benchmark is performance-only, so a toy datum is fine
        quantize_(m_fp8_dyn, config_calib)
        toy_datum = torch.randn(shape.M, shape.K, dtype=torch.bfloat16, device="cuda")
        m_fp8_dyn(toy_datum)

    _quantize_model_for_op(m_fp8_dyn, op_name, config)

    m_fp8_dyn = torch.compile(m_fp8_dyn)

    fp8_trace_filename = None
    if bench.save_profile_traces:
        fp8_trace_filename = f"{trace_prefix}_fp8.json"
    b_fp8_e2e_time_s = get_gpu_kernel_time(m_fp8_dyn, x, fp8_trace_filename)

    return b_bf16_e2e_time_s, b_fp8_e2e_time_s


def run(config: RunConfig = _DEFAULT_RUN_CONFIG):
    """
    Args:
    * `config`: run configuration (`RunConfig`), grouping the cohesive options:
      * `recipe_name`: quantization recipe (tensorwise, rowwise, mxfp8*, mxfp4*, nvfp4*)
      * `shapes` (`ShapeConfig`):
        * `shape_gen_name`: `llama`, `pow2`, `pow2_extended`, `sweep`, or `custom`
        * `M|K|N`: if shape_gen_name is `custom`, then these values are used for MKN
      * `conv` (`ConvConfig`):
        * `op_name`: linear, conv2d or conv3d, decides which op to benchmark
        * `D`, `H`, `W`: spatial dimensions for conv3d / conv2d
        * `kernel_size`: kernel_size for conv3d / conv2d
        * `stride`: stride for conv ops (default: 1)
        * `padding`: padding for conv ops (default: 0)
      * `bench` (`BenchmarkConfig`):
        * `do_benchmarks`: if True, gemm and e2e fwd+bwd of LNLinearSigmoid are benchmarked
        * `enable_fusion_modeling`: if True, models activation -> gemm instead of just gemm
        * `n_limit (optional)`: if specified, only runs `n_limit` iterations
        * `save_profile_traces (optional)`: if True, saves profiling traces
        * `skip_printing_detailed_metrics`: if True, prints e2e roofline
          and observed speedups only, skipping all other intermediate metrics
        * `outfile`: if specified, writes results to this CSV file
    """
    recipe_name = config.recipe_name
    shapes = config.shapes
    conv = config.conv
    bench = config.bench

    shape_gen_name = shapes.shape_gen_name
    M, K, N = shapes.M, shapes.K, shapes.N
    op_name = conv.op_name
    D, H, W = conv.D, conv.H, conv.W
    kernel_size = conv.kernel_size
    stride, padding = conv.stride, conv.padding
    outfile = bench.outfile
    do_benchmarks = bench.do_benchmarks
    n_limit = bench.n_limit
    enable_fusion_modeling = bench.enable_fusion_modeling
    skip_printing_detailed_metrics = bench.skip_printing_detailed_metrics

    _validate_run_ops(recipe_name, conv)

    config_table = [
        ["GPU", torch.cuda.get_device_name(0)],
        ["torch version", torch.__version__],
        ["torchao version", torchao.__version__],
        ["recipe_name", recipe_name],
        ["do_benchmarks", do_benchmarks],
        ["shape_gen_name", shape_gen_name],
        ["enable_fusion_modeling", enable_fusion_modeling],
        ["op_name", op_name],
        ["MKN", f"{M} {K} {N}"],
        ["DHW", f"{D} {H} {W}"],
        ["kernel_size", kernel_size],
        ["stride", stride],
        ["padding", padding],
    ]
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="simple"))

    # reassign user specified MKN, so we can use them for sympy
    user_M, user_K, user_N = M, K, N

    M, K, N = sympy.symbols("M K N")

    # Roofline model setup: linear uses M/K/N directly, conv uses equivalent
    # implicit GEMM dimensions (computed per-iteration in the loop below)
    roofline_model = _build_roofline_model(
        (M, K, N), recipe_name, op_name, enable_fusion_modeling
    )
    print("bf16_gemm_time_sympy", roofline_model.bf16_gemm)
    print("bf16_ovhd_time_sympy", roofline_model.bf16_ovhd)
    print("fp8_gemm_time_sympy", roofline_model.fp8_gemm)
    print("fp8_ovhd_time_sympy", roofline_model.fp8_ovhd)
    print()

    headers = [
        "fwd_M",  # for conv: batch size
        "fwd_K",  # for conv: in_channels
        "fwd_N",  # for conv: out_channels
        "D",
        "H",
        "W",
        "kernel_size",
        # roofline - gemm time (fwd + bwd, 3 gemms; for conv: using equivalent implicit gemm dims)
        "r_bf16_gemm_s",
        "r_fp8_gemm_s",
        # roofline - bf16 overhead time (read-write prev activation, only if fusion modeling is on)
        "r_bf16_ovhd_s",
        # roofline - fp8 overhead time (by counting reads/writes in the ideal case)
        "r_fp8_ovhd_s",
        # roofline - fp8 gemm + fp8 overhead time (does not include LN or sigmoid)
        "r_fp8_gemm_and_ovhd_s",
        "r_fp8_gemm_and_ovhd_spdp",
        # benchmarks - gemm time (fwd + bwd, 3 gemms)
        "b_bf16_gemm_s",
        "b_fp8_gemm_s",
        # benchmarks - e2e LNLinearSigmoid time fwd + bwd
        "b_bf16_e2e_s",
        "b_fp8_e2e_s",
        # note that e2e speedup is not the same as the roofline speedup:
        # 1. roofline speedup: (bf16_gemm_time) / (fp8_gemm_time + fp8_ovhd_time)
        # 2. e2e speedup: (ln + bf16_gemm_time + sigmoid) / (ln + fp8_gemm_time + fp8_ovhd_time + sigmoid)
        # the difference is the fwd+bwd ln and sigmoid terms, for now to keep things simple
        # we don't break them out and don't have a roofline for them.
        "b_fp8_e2e_spdp",
        # how well benchmarked gemms match roofline predicted gemms
        "rb_bf16_gemm_ratio",
        "rb_fp8_gemm_ratio",
    ]

    results = []

    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, user_M, user_K, user_N)

    for idx, (name, (M_val, K_val, N_val)) in enumerate(tqdm.tqdm(name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break

        shape = Shape(M_val, K_val, N_val)
        if op_name == "linear":
            times, gemm_bench = _roofline_and_gemm_linear(
                roofline_model, shape, recipe_name, do_benchmarks
            )
        else:
            times, gemm_bench = _roofline_and_gemm_conv(
                roofline_model, shape, conv, recipe_name, do_benchmarks
            )

        (
            r_bf16_gemm_time_s,
            r_fp8_gemm_time_s,
            r_bf16_ovhd_time_s,
            r_fp8_ovhd_time_s,
            r_fp8_gemm_and_ovhd_s,
            r_speedup,
        ) = times
        (
            b_bf16_gemm_time_s,
            b_fp8_gemm_time_s,
            rb_bf16_gemm_ratio,
            rb_fp8_gemm_ratio,
        ) = gemm_bench

        b_bf16_e2e_time_s, b_fp8_e2e_time_s = 0, 0
        if do_benchmarks:
            b_bf16_e2e_time_s, b_fp8_e2e_time_s = _benchmark_e2e(
                shape, conv, bench, recipe_name
            )

        # Calculate e2e speedup if benchmarks were run, otherwise -1
        if b_bf16_e2e_time_s > 0 and b_fp8_e2e_time_s > 0:
            b_fp8_e2e_speedup = b_bf16_e2e_time_s / b_fp8_e2e_time_s
        else:
            b_fp8_e2e_speedup = -1

        results.append(
            [
                M_val,
                K_val,
                N_val,
                D,
                H,
                W,
                kernel_size,
                # roofline - gemm
                r_bf16_gemm_time_s,
                r_fp8_gemm_time_s,
                # roofline - overhead
                r_bf16_ovhd_time_s,
                r_fp8_ovhd_time_s,
                # roofline - gemm + overhead, and speedup
                r_fp8_gemm_and_ovhd_s,
                r_speedup,
                # benchmarks - gemm
                b_bf16_gemm_time_s,
                b_fp8_gemm_time_s,
                # benchmarks - e2e, and speedup
                b_bf16_e2e_time_s,
                b_fp8_e2e_time_s,
                b_fp8_e2e_speedup,
                # gemm ratios
                rb_bf16_gemm_ratio,
                rb_fp8_gemm_ratio,
            ]
        )

    pd.set_option("display.precision", 2)
    df = pd.DataFrame(results, columns=headers)

    if outfile is not None:
        df.to_csv(outfile)

    if op_name == "linear":
        # drop conv-only columns to simplify linear results
        df = df.drop(columns=["D", "H", "W", "kernel_size"])

    if skip_printing_detailed_metrics:
        df = df[
            ["fwd_M", "fwd_K", "fwd_N", "r_fp8_gemm_and_ovhd_spdp", "b_fp8_e2e_spdp"]
        ]

    print(df)
    print("done")


if __name__ == "__main__":
    fire.Fire(run)
