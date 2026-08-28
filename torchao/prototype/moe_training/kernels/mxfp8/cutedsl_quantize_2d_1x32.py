# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    issue_tma_store_s2g,
    make_kernel_namespace,
    make_tile_2d_smem_layouts,
    resolve_input_cutlass_dtype,
    run_quantize_2d_kernel,
    select_cutedsl_config,
)


def _make_tile_smem_layouts(tile_m: int, tile_k: int):
    """Create shared memory layouts for input and output tiles.

    Both layouts use row-major format (K is fastest-changing dimension).
    """
    return make_tile_2d_smem_layouts(tile_m, tile_k, out_column_major=False)


# Config format:
# (compute_warps, tile_m, tile_k, k_tiles_per_cta)
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 128, 32, 4),
    "fallback": (6, 128, 32, 2),
}


def _select_cutedsl_config(
    input_dtype: torch.dtype,
    scaling_mode: str,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """Select kernel configuration based on input dtype.

    Args:
        input_dtype: Input dtype
        scaling_mode: Scaling mode ("floor" or "rceil")

    Returns:
        Tuple of (config_name, (compute_warps, tile_m, tile_k, k_tiles_per_cta))
    """
    return select_cutedsl_config(input_dtype, _CUTEDSL_CONFIGS)


@dataclass(frozen=True)
class _RawKernelArgs:
    """The raw kernel-shaping arguments passed to the compile driver.

    Bundling these keeps :func:`_compute_kernel_config`'s signature small; the
    values still hash individually through the cached
    :func:`_compile_mxfp8_quantize_2d_cutedsl` entry point.
    """

    input_dtype_name: str
    scaling_mode: str
    compute_warps: int
    tile_m: int
    tile_k: int
    requested_stage_count: int
    k_tiles_per_cta: int
    is_full_k_tiles: bool
    blocked_scale_output: bool


@dataclass(frozen=True)
class _CuteDSLKernelConfig:
    """Derived compile-time constants for the 1x32 MXFP8 CuTeDSL kernel.

    All values are computed once from the raw arguments to
    :func:`_compile_mxfp8_quantize_2d_cutedsl` and then bundled so the kernel
    builder and its nested methods can read them from a single object instead
    of a long list of closure variables.
    """

    input_dtype_name: str
    scaling_mode: str
    compute_warps: int
    tile_m: int
    tile_k: int
    k_tiles_per_cta: int
    is_full_k_tiles: bool
    blocked_scale_output: bool
    threads_per_block: int
    scale_dim_k: int
    k_blocks_per_tile: int
    stage_count: int
    tile_copy_bytes: int
    m_threads: int
    m_iters_per_lane: int


def _compute_kernel_config(raw: _RawKernelArgs) -> _CuteDSLKernelConfig:
    """Validate the raw kernel arguments and derive compile-time constants."""
    # Warp-specialized TMA kernel:
    # - warp 0: producer (issues TMA G2S and S2G)
    # - warps [1..compute_warps]: consumers (quantize)
    # Note: we intentionally keep store on warp 0 (no dedicated store
    # warp).  A split load-warp/store-warp design was tested and
    # mostly regressed throughput, so this layout is the tuned
    # default.
    assert raw.compute_warps >= 1
    assert raw.tile_m > 0 and raw.tile_k > 0
    assert raw.tile_k % 32 == 0

    scale_dim_k = 32
    k_blocks_per_tile = raw.tile_k // scale_dim_k
    assert k_blocks_per_tile > 0
    assert raw.requested_stage_count >= 1
    # B200 sweeps on our representative shapes showed no benefit
    # beyond 2 stages. We keep stage setup generic so future tuning can
    # revisit this, but the current tuned contract is 1 or 2 stages.
    assert raw.requested_stage_count <= 2
    assert raw.k_tiles_per_cta >= 1

    input_elem_bytes = 4 if raw.input_dtype_name == "torch.float32" else 2
    m_threads = raw.compute_warps * 32
    return _CuteDSLKernelConfig(
        input_dtype_name=raw.input_dtype_name,
        scaling_mode=raw.scaling_mode,
        compute_warps=raw.compute_warps,
        tile_m=raw.tile_m,
        tile_k=raw.tile_k,
        k_tiles_per_cta=raw.k_tiles_per_cta,
        is_full_k_tiles=raw.is_full_k_tiles,
        blocked_scale_output=raw.blocked_scale_output,
        threads_per_block=(1 + raw.compute_warps) * 32,
        scale_dim_k=scale_dim_k,
        k_blocks_per_tile=k_blocks_per_tile,
        stage_count=min(raw.requested_stage_count, raw.k_tiles_per_cta),
        tile_copy_bytes=raw.tile_m * raw.tile_k * input_elem_bytes,
        m_threads=m_threads,
        m_iters_per_lane=ceil_div(raw.tile_m, m_threads),
    )


def _build_fake_compile_inputs(cfg: _CuteDSLKernelConfig, has_offs: bool):
    """Build the fake tensors, stream, and compile options for AOT compilation."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    input_cutlass_dtype = resolve_input_cutlass_dtype(cfg.input_dtype_name, "quantize_2d")

    m = cute.sym_int(divisibility=32)
    k = cute.sym_int(divisibility=32)
    kb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()
    scale_stride0 = cute.sym_int()
    scale_stride1 = cute.sym_int()

    fake_inp = make_fake_tensor(
        input_cutlass_dtype,
        (m, k),
        stride=(inp_stride0, inp_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (m, k),
        stride=(out_stride0, out_stride1),
    )
    if cfg.blocked_scale_output:
        scale_flat = cute.sym_int()
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (scale_flat,),
            stride=(scale_stride0,),
        )
    else:
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (m, kb),
            stride=(scale_stride0, scale_stride1),
        )
    fake_stream = make_fake_stream()

    if has_offs:
        offs_stride = cute.sym_int()
        fake_offs = make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int(),),
            stride=(offs_stride,),
        )
    else:
        fake_offs = None

    compile_options = (
        "--enable-tvm-ffi"
        if fake_offs is None
        else "--enable-tvm-ffi --enable-assertions"
    )
    return fake_inp, fake_out, fake_scales, fake_stream, fake_offs, compile_options


def _make_device_helpers(scale_dim: int):
    """Build the device-side ``@cute.jit`` block load/store helpers.

    ``scale_dim`` (the per-block element count) is captured here so the load
    helpers don't carry it as a parameter. Returned as plain free functions
    (the established ``cute_utils`` pattern); the kernel class exposes them as
    thin methods so ``cute_utils`` can call them via ``kernel._<name>``. The
    S2G store lives in the shared ``issue_tma_store_s2g`` helper.
    """
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def load_block_full(sIN_tile, m_rel, k_base):
        # Load a full ``scale_dim``-element block from smem to registers (no bounds check).
        vals_block = cute.make_rmem_tensor((scale_dim,), cutlass.Float32)
        for i in range(scale_dim):
            vals_block[i] = cutlass.Float32(sIN_tile[m_rel, k_base + i])
        return vals_block

    @cute.jit
    def load_block_tail(sIN_tile, k0, m_rel, k_base, K):
        # Bounds-checked block load; out-of-bounds elements are set to 0.0.
        vals_block = cute.make_rmem_tensor((scale_dim,), cutlass.Float32)
        for i in range(scale_dim):
            k = k0 + k_base + i
            if k < K:
                vals_block[i] = cutlass.Float32(sIN_tile[m_rel, k_base + i])
            else:
                vals_block[i] = cutlass.Float32(0.0)
        return vals_block

    @cute.jit
    def store_q_fp8(q_fp8_vals4, sOUT_tile, lane_rel, chunk_base):
        # Vectorize 4 FP8 values (32 bits) into a single uint32 smem write.
        sOUT_tile_u32 = cute.recast_tensor(sOUT_tile, cutlass.Uint32)
        q_fp8_vals4_u32 = cute.recast_tensor(q_fp8_vals4, cutlass.Uint32)
        sOUT_tile_u32[lane_rel, chunk_base // cutlass.Int32(4)] = q_fp8_vals4_u32[0]

    return load_block_full, load_block_tail, store_q_fp8


def _run_1x32_kernel_body(kernel_self, cfg: _CuteDSLKernelConfig, ctx):
    """Build the kernel namespaces and dispatch the shared 2D quantize loop.

    Plain (non-``@cute.jit``) helper: its cute ops inline into the calling
    ``@cute.kernel``. ``ctx`` is the launch namespace built by the kernel
    method (``storage``, the ``tma`` quad, ``scales_out_u8``, ``M``/``K``/
    ``k_blocks``, ``blocked_scale_layout`` and ``offs``). Split out so the axis
    mapping and IO wiring read as one focused step.
    """
    import cutlass

    shape = make_kernel_namespace(
        tile_m=cfg.tile_m,
        tile_k=cfg.tile_k,
        stage_count=cfg.stage_count,
        tile_copy_bytes=cfg.tile_copy_bytes,
    )

    def _axis_1x32(bidx, bidy):
        # 1x32: M is the fixed CTA tile (bidy), K is grouped (bidx).
        return make_kernel_namespace(
            shape=shape,
            tiles_per_cta=cfg.k_tiles_per_cta,
            group_tile_idx=cutlass.Int64(bidx),
            fixed_tile=cutlass.Int64(bidy),
            group_is_first_coord=False,
            lane_tile_size=cfg.tile_m,
            lane_iters=cfg.m_iters_per_lane,
            lane_threads=cfg.m_threads,
            lane_axis_size=ctx.M,
            block_tile_size=cfg.tile_k,
            blocks_per_tile=cfg.k_blocks_per_tile,
            scale_dim=cfg.scale_dim_k,
            block_axis_size=ctx.K,
            num_blocks=ctx.k_blocks,
            is_full_tiles=cfg.is_full_k_tiles,
        )

    io = make_kernel_namespace(
        storage=ctx.storage,
        smem_layouts=_make_tile_smem_layouts(cfg.tile_m, cfg.tile_k),
        tma=ctx.tma,
        shape=shape,
        offs=ctx.offs,
        scales_out_u8=ctx.scales_out_u8,
        blocked_scale_output=cfg.blocked_scale_output,
        blocked_scale_layout=ctx.blocked_scale_layout,
        opts=make_kernel_namespace(
            use_rceil=(cfg.scaling_mode == "rceil"),
            blocked_scale_output=cfg.blocked_scale_output,
        ),
        compute_warps=cfg.compute_warps,
    )
    run_quantize_2d_kernel(kernel_self, io, _axis_1x32)


def _setup_tma_and_scale_layout(cfg: _CuteDSLKernelConfig, inp_mk, out_mk, M, k_blocks):
    """Create the G2S/S2G TMA atoms and the optional blocked scale layout.

    Plain (non-``@cute.jit``) helper: its cute ops inline into the calling
    ``@cute.jit`` launcher. Returns the TMA atoms/tensors plus the blocked
    scale layout used by the kernel.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import cpasync, tcgen05

    smem_layout_in, smem_layout_out = _make_tile_smem_layouts(cfg.tile_m, cfg.tile_k)
    # Use tcgen05.CtaGroup.ONE for the optimised single-CTA Blackwell (SM 10.x) TMA load path.
    g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
        g2s_op,
        inp_mk,
        smem_layout_in,
        (cfg.tile_m, cfg.tile_k),
    )
    tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        out_mk,
        smem_layout_out,
        (cfg.tile_m, cfg.tile_k),
    )

    blocked_scale_layout = cute.make_layout((1,))
    if cutlass.const_expr(cfg.blocked_scale_output):
        padded_scale_cols = cute.round_up(k_blocks, 4)
        m_block_tiles = cute.ceil_div(M, 128)
        k_block_tiles = padded_scale_cols // cutlass.Int64(4)
        blocked_scale_layout = cute.make_layout(
            ((32, 4, m_block_tiles), (4, k_block_tiles)),
            stride=(
                (16, 4, cutlass.Int64(128) * padded_scale_cols),
                (1, cutlass.Int64(512)),
            ),
        )
    return (
        tma_atom_in,
        tma_tensor_in,
        tma_atom_out,
        tma_tensor_out,
        blocked_scale_layout,
    )


def _build_quantize_2d_kernel_class(cfg: _CuteDSLKernelConfig) -> Any:
    """Construct the warp-specialized CuTeDSL kernel class for ``cfg``.

    The class and its nested ``@cute`` methods close over ``cfg`` for all the
    compile-time constants (tile sizes, stage count, scale dims, etc.), keeping
    the surrounding compile driver readable.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils

    input_cutlass_dtype = resolve_input_cutlass_dtype(cfg.input_dtype_name, "quantize_2d")

    # Aliases used by the config-bound ``SharedStorage`` struct and the thin
    # device-helper methods below. Runtime methods read the rest from ``cfg``.
    INPUT_CUTLASS_DTYPE = input_cutlass_dtype
    TILE_M = cfg.tile_m
    TILE_K = cfg.tile_k
    STAGE_COUNT_VALUE = cfg.stage_count

    (
        _load_block_full,
        _load_block_tail,
        _store_q_fp8,
    ) = _make_device_helpers(cfg.scale_dim_k)

    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, STAGE_COUNT_VALUE]
        in_smem: cute.struct.Align[
            cute.struct.MemRange[
                INPUT_CUTLASS_DTYPE, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]
        out_smem: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float8E4M3FN, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]

    class Mxfp8Quantize2dKernel:
        # Device-side block load/store helpers. These are exposed as methods so
        # ``cute_utils`` can invoke them via ``kernel._<name>``; the bodies live in
        # ``_make_device_helpers`` (see that function for docs). ``self`` is unused.
        @cute.jit
        def _load_block_full_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
        ):
            return _load_block_full(sIN_tile, m_rel, k_base)

        @cute.jit
        def _load_block_tail_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            k0: cutlass.Int64,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
            K: cutlass.Int64,
        ):
            return _load_block_tail(sIN_tile, k0, m_rel, k_base, K)

        @cute.jit
        def _store_q_fp8_reg_to_smem(
            self,
            q_fp8_vals4: cute.Tensor,
            sOUT_tile: cute.Tensor,
            lane_rel: cutlass.Int32,
            chunk_base: cutlass.Int32,
        ):
            _store_q_fp8(q_fp8_vals4, sOUT_tile, lane_rel, chunk_base)

        @cute.jit
        def _issue_tma_store(
            self,
            tma_atom_out: cute.CopyAtom,
            gOUT_tile: cute.Tensor,
            sOUT_tile: cute.Tensor,
            warp_idx: cutlass.Int32,
        ):
            # 2D tiles group a single leading mode for the TMA partition.
            issue_tma_store_s2g(tma_atom_out, gOUT_tile, sOUT_tile, warp_idx, 1)

        @cute.kernel
        def kernel(
            self,
            inp_mk: cute.Tensor,
            tma_atom_in: cute.CopyAtom,
            tma_tensor_in: cute.Tensor,
            out_mk: cute.Tensor,
            tma_atom_out: cute.CopyAtom,
            tma_tensor_out: cute.Tensor,
            scales_out_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            offs: Optional[cute.Tensor],
            SCALE_DIM_K: cutlass.Constexpr[int],
            USE_RCEIL: cutlass.Constexpr[bool],
            IS_FULL_K_TILES: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
        ):
            """MXFP8 1x32 quantization kernel (scales along K, K//32 per row).

            Warp-specialized TMA pipeline: warp 0 issues loads/stores, warps
            1..compute_warps quantize in registers. Here M indexes the fixed CTA
            tile axis and K is the grouped, per-32-element-block scaling axis.
            The shared body lives in ``run_quantize_2d_kernel``; this entry only
            supplies the 1x32 axis mapping via ``_axis_1x32``.
            """
            storage = utils.SmemAllocator().allocate(SharedStorage)
            ctx = make_kernel_namespace(
                storage=storage,
                tma=make_kernel_namespace(
                    atom_in=tma_atom_in,
                    tensor_in=tma_tensor_in,
                    atom_out=tma_atom_out,
                    tensor_out=tma_tensor_out,
                ),
                scales_out_u8=scales_out_u8,
                M=M,
                K=K,
                k_blocks=k_blocks,
                blocked_scale_layout=blocked_scale_layout,
                offs=offs,
            )
            _run_1x32_kernel_body(self, cfg, ctx)

        @cute.jit
        def __call__(
            self,
            inp_mk: cute.Tensor,
            out_mk: cute.Tensor,
            scales_out_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            stream: cuda.CUstream,
            offs: Optional[cute.Tensor],
        ):
            """Kernel launcher: set up TMA descriptors and blocked scale layout,
            then launch the warp-specialized quantize kernel.

            All tensors live in global memory; see :func:`mxfp8_quantize_cutedsl_2d_1x32`
            for the shapes of ``inp_mk``/``out_mk``/``scales_out_u8`` and the meaning
            of ``M``, ``K``, ``k_blocks``, the CTA tile counts, ``stream`` and ``offs``.
            """
            (
                tma_atom_in,
                tma_tensor_in,
                tma_atom_out,
                tma_tensor_out,
                blocked_scale_layout,
            ) = _setup_tma_and_scale_layout(cfg, inp_mk, out_mk, M, k_blocks)

            self.kernel(
                inp_mk,
                tma_atom_in,
                tma_tensor_in,
                out_mk,
                tma_atom_out,
                tma_tensor_out,
                scales_out_u8,
                M,
                K,
                k_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                offs,
                SCALE_DIM_K=cfg.scale_dim_k,
                USE_RCEIL=(cfg.scaling_mode == "rceil"),
                IS_FULL_K_TILES=cfg.is_full_k_tiles,
                STAGE_COUNT=cfg.stage_count,
            ).launch(
                grid=(k_cta_tiles, m_cta_tiles, 1),
                block=(cfg.threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    return Mxfp8Quantize2dKernel


@functools.cache
def _compile_mxfp8_quantize_2d_cutedsl(
    input_dtype_name: str,
    scaling_mode: str,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    k_tiles_per_cta: int,
    is_full_k_tiles: bool,
    blocked_scale_output: bool,
    has_offs: bool = False,
):
    """Compile the 2D MXFP8 quantization kernel using CuTeDSL.

    Uses warp-specialized TMA kernel with:
    - Warp 0: Producer (issues TMA global→shared and shared→global)
    - Warps 1..compute_warps: Consumers (quantize in registers)

    Args:
        input_dtype_name: Input dtype ("torch.float32" or "torch.bfloat16")
        scaling_mode: Scaling mode ("floor" or "rceil")
        compute_warps: Number of compute warps
        tile_m: Tile size in M dimension
        tile_k: Tile size in K dimension
        requested_stage_count: Requested pipeline stages (capped by k_tiles_per_cta)
        k_tiles_per_cta: Number of K tiles per CTA
        is_full_k_tiles: Whether K dimension is perfectly tiled
        blocked_scale_output: Whether to output scales in blocked layout for tcgen05

    Returns:
        Compiled CuTeDSL kernel callable
    """
    import cutlass.cute as cute

    # PTX lowering note:
    # - RCEIL uses inline PTX on Blackwell-family targets because
    #   CuTeDSL does not currently lower this conversion to
    #   `cvt.rp.satfinite.ue8m0x2.f32` on its own.
    # - FLOOR still uses a different lowered sequence than C++
    #   helper routines.
    cfg = _compute_kernel_config(
        _RawKernelArgs(
            input_dtype_name=input_dtype_name,
            scaling_mode=scaling_mode,
            compute_warps=compute_warps,
            tile_m=tile_m,
            tile_k=tile_k,
            requested_stage_count=requested_stage_count,
            k_tiles_per_cta=k_tiles_per_cta,
            is_full_k_tiles=is_full_k_tiles,
            blocked_scale_output=blocked_scale_output,
        )
    )

    kernel_cls = _build_quantize_2d_kernel_class(cfg)
    kernel = kernel_cls()

    (
        fake_inp,
        fake_out,
        fake_scales,
        fake_stream,
        fake_offs,
        compile_options,
    ) = _build_fake_compile_inputs(cfg, has_offs)

    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_out_u8=fake_scales,
        M=0,
        K=0,
        k_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        stream=fake_stream,
        offs=fake_offs,
        options=compile_options,
    )


def mxfp8_quantize_cutedsl_2d_1x32(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel.

    Quantizes along the K dimension - each row has K//32 scales, one per block of 32 K elements.

    Args:
        x: Input tensor of shape (M, K)
        block_size: Block size for quantization along K (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Number of pipeline stages (1 or 2)
        blocked_scale_output: Whether to output scales in blocked layout
        offs: Optional tensor of group end offsets for validation (must have group sizes as multiples of 128)

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K)
        scales: Scales tensor with shape (M, K//32) or blocked layout
    """
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    M, K = x.shape
    assert K % 128 == 0, "K must be divisible by 128"
    assert M % 128 == 0, "M must be divisible by 128"

    if offs is not None:
        assert offs.is_cuda, "offs tensor must be CUDA"
        assert offs.dtype == torch.int32, "offs must be int32 tensor"
        assert offs.dim() == 1, "offs must be 1D tensor"

    _, config = _select_cutedsl_config(x.dtype, scaling_mode)
    compute_warps, tile_m, tile_k, k_tiles_per_cta = config
    # B200 sweeps over representative shapes showed no
    # measurable benefit above 2 stages. We keep this configurable for
    # benchmarking, and the effective stage count remains capped by
    # k_tiles_per_cta below.
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"
    is_full_k_tiles = K % (tile_k * k_tiles_per_cta) == 0
    is_sm_10x = torch.cuda.get_device_capability()[0] == 10
    if blocked_scale_output and not is_sm_10x:
        raise NotImplementedError(
            "blocked_scale_output is only supported on SM 10.x GPUs "
            "because it produces the tcgen05 blocked scale layout"
        )

    # Output in row-major layout: stride (K, 1).
    q_data = torch.empty_strided(
        (M, K),
        (K, 1),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    k_blocks = K // block_size
    padded_scale_rows = ceil_div(M, 128) * 128
    padded_scale_cols = ceil_div(k_blocks, 4) * 4
    if blocked_scale_output:
        scales_u8 = torch.empty(
            (padded_scale_rows * padded_scale_cols,),
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        scales_u8 = torch.empty(
            (M, k_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    compiled = _compile_mxfp8_quantize_2d_cutedsl(
        str(x.dtype),
        scaling_mode,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        k_tiles_per_cta,
        is_full_k_tiles,
        blocked_scale_output,
        offs is not None,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    m_cta_tiles = ceil_div(M, tile_m)
    k_cta_tiles = ceil_div(K, tile_k * k_tiles_per_cta)

    compiled(
        x,
        q_data,
        scales_u8,
        int(M),
        int(K),
        int(k_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        stream,
        offs,
    )
    scales = scales_u8.view(torch.float8_e8m0fnu)
    scales = (
        scales.view(padded_scale_rows, padded_scale_cols)
        if blocked_scale_output
        else scales_u8.view(M, k_blocks)
    )
    return q_data, scales
