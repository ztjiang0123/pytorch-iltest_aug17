# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Optional, Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    issue_tma_store_s2g,
    make_kernel_namespace,
    make_tile_2d_smem_layouts,
    run_quantize_2d_kernel,
)


def _make_tile_smem_layouts(tile_m: int, tile_k: int):
    """Create shared memory layouts for input and output tiles.

    Input uses row-major format. Output uses column-major format.
    """
    return make_tile_2d_smem_layouts(tile_m, tile_k, out_column_major=True)


# Config format:
# (compute_warps, tile_m, tile_k, m_tiles_per_cta)
# NOTE: Swapped tile_m/tile_k and changed k_tiles_per_cta to m_tiles_per_cta for 32x1 scaling
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 32, 128, 4),
    "fallback": (6, 32, 128, 2),
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
        Tuple of (config_name, (compute_warps, tile_m, tile_k, m_tiles_per_cta))
    """
    if input_dtype == torch.bfloat16:
        config_name = "bf16_default"
    else:
        config_name = "fallback"
    return config_name, _CUTEDSL_CONFIGS[config_name]


@functools.cache
def _compile_mxfp8_quantize_2d_32x1_cutedsl(
    input_dtype_name: str,
    scaling_mode: str,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    m_tiles_per_cta: int,
    is_full_m_tiles: bool,
    blocked_scale_output: bool,
    has_offs: bool = False,
):
    """Compile the 2D MXFP8 quantization kernel using CuTeDSL for 32x1 scaling.

    Uses warp-specialized TMA kernel with:
    - Warp 0: Producer (issues TMA global→shared and shared→global)
    - Warps 1..compute_warps: Consumers (quantize in registers)

    Args:
        input_dtype_name: Input dtype ("torch.float32" or "torch.bfloat16")
        scaling_mode: Scaling mode ("floor" or "rceil")
        compute_warps: Number of compute warps
        tile_m: Tile size in M dimension (32 for 32x1 scaling)
        tile_k: Tile size in K dimension (128 for 32x1 scaling)
        requested_stage_count: Requested pipeline stages (capped by m_tiles_per_cta)
        m_tiles_per_cta: Number of M tiles per CTA
        is_full_m_tiles: Whether M dimension is perfectly tiled
        blocked_scale_output: Whether to output scales in blocked layout for tcgen05

    Returns:
        Compiled CuTeDSL kernel callable
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    if input_dtype_name == "torch.float32":
        INPUT_CUTLASS_DTYPE = cutlass.Float32
    elif input_dtype_name == "torch.bfloat16":
        INPUT_CUTLASS_DTYPE = cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported input dtype for CuTeDSL quantize_2d_32x1: {input_dtype_name}"
        )

    # Warp-specialized TMA kernel:
    # - warp 0: producer (issues TMA G2S and S2G)
    # - warps [1..compute_warps]: consumers (quantize)
    COMPUTE_WARPS = compute_warps
    TILE_M = tile_m
    TILE_K = tile_k
    M_TILES_PER_CTA = m_tiles_per_cta
    IS_FULL_M_TILES_VALUE = is_full_m_tiles
    BLOCKED_SCALE_OUTPUT_VALUE = blocked_scale_output

    THREADS_PER_BLOCK = (1 + COMPUTE_WARPS) * 32
    assert COMPUTE_WARPS >= 1
    assert TILE_M > 0 and TILE_K > 0
    assert TILE_M % 32 == 0  # Changed from TILE_K % 32 == 0 for 32x1 scaling

    SCALE_DIM_M_VALUE = 32  # Changed from SCALE_DIM_K_VALUE for 32x1 scaling
    M_BLOCKS_PER_TILE = TILE_M // SCALE_DIM_M_VALUE
    assert M_BLOCKS_PER_TILE > 0
    assert requested_stage_count >= 1
    assert requested_stage_count <= 2
    assert M_TILES_PER_CTA >= 1
    STAGE_COUNT_VALUE = min(requested_stage_count, M_TILES_PER_CTA)

    input_elem_bytes = 4 if input_dtype_name == "torch.float32" else 2
    TILE_COPY_BYTES = TILE_M * TILE_K * input_elem_bytes
    K_THREADS = COMPUTE_WARPS * 32  # Changed from M_THREADS for 32x1 scaling
    K_ITERS_PER_LANE = ceil_div(TILE_K, K_THREADS)

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

    class Mxfp8Quantize2dKernel32x1:
        @cute.jit
        def _load_block_full_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
        ):
            """Load a full 32-element quantization block from shared memory to registers.

            Loads all elements without bounds checking. For 32x1 scaling, we load 32 elements
            along the M dimension for a given K position.

            Args:
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile

            Returns:
                vals_block: 32 input elements in register memory
            """
            vals_block = cute.make_rmem_tensor((SCALE_DIM_M_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_M_VALUE):
                vals_block[i] = cutlass.Float32(sIN_tile[m_base + i, k_rel])
            return vals_block

        @cute.jit
        def _load_block_tail_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            m0: cutlass.Int64,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
            M: cutlass.Int64,
        ):
            """Load a 32-element quantization block from shared memory to registers with bounds checking.

            Out-of-bounds elements are set to 0.0. For 32x1 scaling, we check M dimension bounds.

            Args:
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                m0: Global M offset for this tile
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile
                M: Total M dimension size for bounds checking

            Returns:
                vals_block: 32 input elements in register memory (out-of-bounds set to 0.0)
            """
            vals_block = cute.make_rmem_tensor((SCALE_DIM_M_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_M_VALUE):
                m = m0 + m_base + i
                if m < M:
                    vals_block[i] = cutlass.Float32(sIN_tile[m_base + i, k_rel])
                else:
                    vals_block[i] = cutlass.Float32(0.0)
            return vals_block

        @cute.jit
        def _store_q_fp8_reg_to_smem(
            self,
            q_fp8_vals4: cute.Tensor,
            sOUT_tile: cute.Tensor,
            lane_rel: cutlass.Int32,
            chunk_base: cutlass.Int32,
        ):
            """Store 4 quantized FP8 values from registers to shared memory.

            For 32x1 scaling with column-major SMEM, the 4 values are stored along
            the M dimension. Called by the shared ``quantize_block_store_full``/
            ``_tail`` loops in ``cute_utils`` with the canonical
            ``(lane_rel, chunk_base)`` coordinate pair; for the 32x1 kernel the
            axes are transposed relative to 1x32, so the lane is the tile column
            (K) and ``chunk_base`` the starting M index of the chunk.

            Args:
                q_fp8_vals4: 4 quantized FP8 values in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K) with column-major layout
                lane_rel: Column index within tile (K)
                chunk_base: Starting M index for this chunk within tile

            Storage locations:
                Input: q_fp8_vals4 (registers)
                Output: sOUT_tile (shared memory, column-major)
            """
            for i in range(4):
                sOUT_tile[chunk_base + i, lane_rel] = q_fp8_vals4[i]

        @cute.jit
        def _issue_tma_store(
            self,
            tma_atom_out: cute.CopyAtom,
            gOUT_tile: cute.Tensor,
            sOUT_tile: cute.Tensor,
            warp_idx: cutlass.Int32,
        ):
            """Issue TMA store from shared memory to global memory (producer warp only).

            Synchronizes threads before store; only warp 0 executes the store.
            2D tiles group a single leading mode for the TMA partition.
            """
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
            m_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            offs: Optional[cute.Tensor],
            SCALE_DIM_M: cutlass.Constexpr[int],
            USE_RCEIL: cutlass.Constexpr[bool],
            IS_FULL_M_TILES: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
        ):
            """MXFP8 32x1 quantization kernel (scales along M, M//32 per column).

            This is the transpose of the 1x32 kernel: K indexes the fixed CTA
            tile axis while M is the grouped, per-32-element-block scaling axis,
            so each column carries M//32 scales. block_idx.x selects the grouped
            M tile and block_idx.y the fixed K tile. The warp-specialized TMA
            pipeline body is shared in ``run_quantize_2d_kernel``; ``_axis_32x1``
            supplies this kernel's transposed axis mapping.
            """
            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            shape = make_kernel_namespace(
                tile_m=TILE_M,
                tile_k=TILE_K,
                stage_count=STAGE_COUNT_VALUE,
                tile_copy_bytes=TILE_COPY_BYTES,
            )
            tma = make_kernel_namespace(
                atom_in=tma_atom_in,
                tensor_in=tma_tensor_in,
                atom_out=tma_atom_out,
                tensor_out=tma_tensor_out,
            )
            opts = make_kernel_namespace(
                use_rceil=USE_RCEIL,
                blocked_scale_output=BLOCKED_SCALE_OUTPUT_VALUE,
            )

            def _axis_32x1(bidx, bidy):
                # 32x1: K is the fixed CTA tile (bidy), M is grouped (bidx).
                return make_kernel_namespace(
                    shape=shape,
                    tiles_per_cta=M_TILES_PER_CTA,
                    group_tile_idx=cutlass.Int64(bidx),
                    fixed_tile=cutlass.Int64(bidy),
                    group_is_first_coord=True,
                    lane_tile_size=TILE_K,
                    lane_iters=K_ITERS_PER_LANE,
                    lane_threads=K_THREADS,
                    lane_axis_size=K,
                    block_tile_size=TILE_M,
                    blocks_per_tile=M_BLOCKS_PER_TILE,
                    scale_dim=SCALE_DIM_M_VALUE,
                    block_axis_size=M,
                    num_blocks=m_blocks,
                    is_full_tiles=IS_FULL_M_TILES,
                )

            io = make_kernel_namespace(
                storage=storage,
                smem_layouts=_make_tile_smem_layouts(TILE_M, TILE_K),
                tma=tma,
                shape=shape,
                offs=offs,
                scales_out_u8=scales_out_u8,
                blocked_scale_output=BLOCKED_SCALE_OUTPUT_VALUE,
                blocked_scale_layout=blocked_scale_layout,
                opts=opts,
                compute_warps=compute_warps,
            )
            run_quantize_2d_kernel(self, io, _axis_32x1)

        @cute.jit
        def __call__(
            self,
            inp_mk: cute.Tensor,
            out_mk: cute.Tensor,
            scales_out_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            m_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            stream: cuda.CUstream,
            offs: Optional[cute.Tensor],
        ):
            """Kernel launcher that sets up TMA descriptors and blocked scale layout.

            Args:
                inp_mk: Input tensor in global memory (M, K)
                out_mk: Output quantized data tensor in global memory (M, K)
                scales_out_u8: Output scales tensor in global memory (M//32, K) or blocked layout
                M: M dimension size
                K: K dimension size
                m_blocks: Number of 32-element blocks in M
                m_cta_tiles: Number of tiles in M dimension
                k_cta_tiles: Number of tile groups in K dimension
                stream: CUDA stream
                offs: Tensor of group end offsets for validation (group sizes must be multiples of 128)

            Storage locations:
                All tensors in global memory
            """
            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(TILE_M, TILE_K)
            # Use tcgen05.CtaGroup.ONE for the optimised single-CTA Blackwell (SM 10.x) TMA load path.
            g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                g2s_op,
                inp_mk,
                smem_layout_in,
                (TILE_M, TILE_K),
            )
            out_colmajor = cute.make_tensor(
                out_mk.iterator,
                cute.make_layout((M, K), stride=(1, M)),  # col major
            )
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_colmajor,
                smem_layout_out,
                (
                    TILE_M,
                    TILE_K,
                ),  # Keep original tile dimensions to match global memory expectation
            )

            blocked_scale_layout = cute.make_layout((1,))
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                # Use same blocked layout as regular 2D kernel but for (K, M//32) tensor
                # For scales shape (K, M//32): K is rows, M//32 is columns
                padded_scale_rows = cute.round_up(
                    K, 128
                )  # K rounded to multiple of 128 (first dim)
                padded_scale_cols = cute.round_up(
                    m_blocks, 4
                )  # M//32 rounded to multiple of 4 (second dim)
                m_block_tiles = padded_scale_rows // cutlass.Int64(
                    128
                )  # Number of 128-row tiles
                k_block_tiles = padded_scale_cols // cutlass.Int64(
                    4
                )  # Number of 4-col tiles
                blocked_scale_layout = cute.make_layout(
                    ((32, 4, m_block_tiles), (4, k_block_tiles)),
                    stride=(
                        (16, 4, cutlass.Int64(128) * padded_scale_cols),
                        (1, cutlass.Int64(512)),
                    ),
                )

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
                m_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                offs,
                SCALE_DIM_M=SCALE_DIM_M_VALUE,
                USE_RCEIL=(scaling_mode == "rceil"),
                IS_FULL_M_TILES=IS_FULL_M_TILES_VALUE,
                STAGE_COUNT=STAGE_COUNT_VALUE,
            ).launch(
                grid=(m_cta_tiles, k_cta_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = Mxfp8Quantize2dKernel32x1()

    m = cute.sym_int(divisibility=32)
    k = cute.sym_int(divisibility=128)
    mb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
    scale_stride0 = cute.sym_int()
    scale_stride1 = cute.sym_int()

    fake_inp = make_fake_tensor(
        INPUT_CUTLASS_DTYPE,
        (m, k),
        stride=(inp_stride0, inp_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (m, k),
        stride=(1, m),  # col major
    )
    if blocked_scale_output:
        scale_flat = cute.sym_int()
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (scale_flat,),
            stride=(scale_stride0,),
        )
    else:
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (k, mb),  # Changed to (K, M//32) to match torch._scaled_mm format
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
    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_out_u8=fake_scales,
        M=0,
        K=0,
        m_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        stream=fake_stream,
        offs=fake_offs,
        options=compile_options,
    )


def mxfp8_quantize_cutedsl_2d_32x1(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel with 32x1 scaling.

    Quantizes along the M dimension - each column has M//32 scales, one per block of 32 M elements.

    Args:
        x: Input tensor of shape (M, K)
        block_size: Block size for quantization along M (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Number of pipeline stages (1 or 2)
        blocked_scale_output: Whether to output scales in blocked layout
        offs: Optional tensor of group end offsets for validation (must have group sizes as multiples of 128)

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K) (no padding on data)
        scales: Scales tensor with shape (K, M//32) or blocked layout compatible with torch._scaled_mm
                Same format as other dim1 MXFP8 kernels for consistency
    """
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    M, K = x.shape
    assert M % 128 == 0, "M must be divisible by 128"
    assert K % 128 == 0, "K must be divisible by 128"

    if offs is not None:
        assert offs.is_cuda, "offs tensor must be CUDA"
        assert offs.dtype == torch.int32, "offs must be int32 tensor"
        assert offs.dim() == 1, "offs must be 1D tensor"

    _, config = _select_cutedsl_config(x.dtype, scaling_mode)
    compute_warps, tile_m, tile_k, m_tiles_per_cta = config
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"
    is_full_m_tiles = M % (tile_m * m_tiles_per_cta) == 0
    is_sm_10x = torch.cuda.get_device_capability()[0] == 10
    if blocked_scale_output and not is_sm_10x:
        raise NotImplementedError(
            "blocked_scale_output is only supported on SM 10.x GPUs "
            "because it produces the tcgen05 blocked scale layout"
        )

    # For 32x1 scaling, only the scales need padding for 4x128 tiles
    # The output data tensor has the same dimensions as input (no padding)

    # Output in column-major layout: stride (1, M) - M fastest-changing
    q_data = torch.empty_strided(
        (M, K),
        (1, M),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    m_blocks = M // block_size
    if blocked_scale_output:
        # Use same blocked layout as regular 2D kernel for (K, M//32) tensor
        padded_scale_rows = (
            ceil_div(K, 128) * 128
        )  # K rounded to multiple of 128 (first dim)
        padded_scale_cols = (
            ceil_div(m_blocks, 4) * 4
        )  # M//32 rounded to multiple of 4 (second dim)
        scales_u8 = (
            torch.empty(  # Initialize with zeros to match to_blocked() padding behavior
                (padded_scale_rows * padded_scale_cols,),
                device=x.device,
                dtype=torch.uint8,
            )
        )
    else:
        scales_u8 = torch.empty(
            (K, m_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    compiled = _compile_mxfp8_quantize_2d_32x1_cutedsl(
        str(x.dtype),
        scaling_mode,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        m_tiles_per_cta,
        is_full_m_tiles,
        blocked_scale_output,
        offs is not None,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    m_cta_tiles = ceil_div(M, tile_m * m_tiles_per_cta)
    k_cta_tiles = ceil_div(K, tile_k)

    compiled(
        x,
        q_data,
        scales_u8,
        int(M),
        int(K),
        int(m_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        stream,
        offs,
    )

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
