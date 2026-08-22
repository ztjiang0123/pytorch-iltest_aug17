# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for CuTeDSL quantization kernels."""

import importlib.util

# Runtime package detection
_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list[str]:
    """Check which CuTeDSL runtime packages are missing.

    Returns:
        List of missing package names
    """
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            # ModuleNotFoundError: parent module doesn't exist (e.g., 'cuda' on CPU)
            # ValueError: can occur with malformed module names
            spec = None

        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """Check if all CuTeDSL runtime packages are available.

    Returns:
        True if all required packages are installed
    """
    return len(_missing_cutedsl_runtime_packages()) == 0


if _cutedsl_runtime_available():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.base_dsl._mlir_helpers import arith as _dsl_arith
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cutlass_dsl import T, dsl_user_op

    def make_tile_smem_layout(tile_m: int, tile_k: int, column_major: bool):
        """Create a 2D shared-memory tile layout.

        Shared by the 2D CuTeDSL quantization kernels, whose tile layouts differ
        only in whether a given operand is row-major (K contiguous) or
        column-major (M contiguous).

        Args:
            tile_m: Tile size in the M dimension
            tile_k: Tile size in the K dimension
            column_major: If True, M is the fastest-changing dimension
                (stride ``(1, tile_m)``); otherwise K is (stride ``(tile_k, 1)``)

        Returns:
            A ``cute`` layout for the tile in shared memory
        """
        stride = (1, tile_m) if column_major else (tile_k, 1)
        return cute.make_layout((tile_m, tile_k), stride=stride)

    @cute.jit
    def _tma_partition_stage0(tma_atom: cute.CopyAtom, tiles: tuple):
        """Partition a (global, shared) tile pair for a single-CTA TMA copy.

        Args:
            tma_atom: TMA copy atom.
            tiles: ``(g_tile, s_tile, group_modes_end)`` — the global and shared
                tiles plus the ``cute.group_modes`` end index (1 for 2D tiles,
                2 for 3D tiles whose leading singleton mode is folded in).

        Returns:
            Tuple ``(t_shared_stage0, t_global_stage0)`` ready for ``cute.copy``.
        """
        g_tile, s_tile, group_modes_end = tiles
        cta_layout = cute.make_layout((1,))
        s_grouped = cute.group_modes(s_tile, 0, group_modes_end)
        g_grouped = cute.group_modes(g_tile, 0, group_modes_end)
        t_shared, t_global = cpasync.tma_partition(
            tma_atom,
            0,
            cta_layout,
            s_grouped,
            g_grouped,
        )
        return t_shared[(None, 0)], t_global[(None, 0)]

    @cute.jit
    def issue_tma_load(
        tma_atom_in: cute.CopyAtom,
        tiles: tuple,
        tma_mbar_ptr: cutlass.Int64,
        warp_idx: cutlass.Int32,
        tile_copy_bytes: cutlass.Constexpr[int],
    ):
        """Issue a TMA load from global to shared memory (producer warp only).

        Shared across the 2D and 3D CuTeDSL quantization kernels.

        Args:
            tma_atom_in: TMA copy atom for G2S
            tiles: ``(gIN_tile, sIN_tile, group_modes_end)`` — global source and
                shared dest tiles plus the ``cute.group_modes`` end index
            tma_mbar_ptr: TMA barrier pointer
            warp_idx: Warp index (only warp 0 issues the copy)
            tile_copy_bytes: Number of bytes copied by the TMA transaction
        """
        if warp_idx == 0:
            tINs_stage0, tINg_stage0 = _tma_partition_stage0(tma_atom_in, tiles)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_ptr, tile_copy_bytes)
            cute.copy(
                tma_atom_in,
                tINg_stage0,
                tINs_stage0,
                tma_bar_ptr=tma_mbar_ptr,
            )

    @cute.jit
    def issue_tma_store(
        tma_atom_out: cute.CopyAtom,
        tiles: tuple,
        warp_idx: cutlass.Int32,
    ):
        """Issue a TMA store from shared to global memory (producer warp only).

        Shared across the 2D and 3D CuTeDSL quantization kernels. Synchronizes
        threads before the store.

        Args:
            tma_atom_out: TMA copy atom for S2G
            tiles: ``(gOUT_tile, sOUT_tile, group_modes_end)`` — global dest and
                shared source tiles plus the ``cute.group_modes`` end index
            warp_idx: Warp index (only warp 0 issues the copy)
        """
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        cute.arch.sync_threads()
        if warp_idx == 0:
            tOUTs_stage0, tOUTg_stage0 = _tma_partition_stage0(tma_atom_out, tiles)
            cute.copy(
                tma_atom_out,
                tOUTs_stage0,
                tOUTg_stage0,
            )

    @cute.jit
    def _init_staged_pipeline(
        storage, smem_layouts, tile_shape, stage_count, tidx, tma
    ):
        """Allocate the staged smem tiles and initialize the TMA barriers.

        Returns ``(sIN_tiles, sOUT_tiles, mbar_ptr0)`` where the tile lists have
        one entry per stage (both entries alias stage 0 when ``stage_count`` is
        1, so the caller can index by stage unconditionally).
        """
        smem_layout_in, smem_layout_out = smem_layouts
        tile_m, tile_k = tile_shape
        stage_elems = tile_m * tile_k
        tma_atom_in, tma_atom_out = tma

        mbar_ptr0 = storage.tma_mbar_ptr.data_ptr()
        staged_layout = cute.make_layout(
            (stage_count, tile_m, tile_k),
            stride=(tile_m * tile_k, tile_k, 1),
        )
        sIN_staged = storage.in_smem.get_tensor(staged_layout)
        sOUT_staged = storage.out_smem.get_tensor(staged_layout)

        sIN_tiles = [
            cute.make_tensor(sIN_staged.iterator + 0 * stage_elems, smem_layout_in)
        ]
        sOUT_tiles = [
            cute.make_tensor(sOUT_staged.iterator + 0 * stage_elems, smem_layout_out)
        ]
        if cutlass.const_expr(stage_count > 1):
            sIN_tiles.append(
                cute.make_tensor(sIN_staged.iterator + 1 * stage_elems, smem_layout_in)
            )
            sOUT_tiles.append(
                cute.make_tensor(
                    sOUT_staged.iterator + 1 * stage_elems, smem_layout_out
                )
            )
        else:
            sIN_tiles.append(sIN_tiles[0])
            sOUT_tiles.append(sOUT_tiles[0])

        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_in)
            cpasync.prefetch_descriptor(tma_atom_out)
            cute.arch.mbarrier_init(mbar_ptr0, 1)
            if cutlass.const_expr(stage_count > 1):
                cute.arch.mbarrier_init(mbar_ptr0 + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        return sIN_tiles, sOUT_tiles, mbar_ptr0

    @cute.jit
    def _load_pipeline_tile(load_ctx, tile_idx, stage_idx):
        """Issue the TMA load for one pipeline tile (producer warp only)."""
        (
            tma_atom_in,
            tma_tensor_in,
            tile_shape,
            make_tile_coord,
            orthogonal_idx,
            sIN_tiles,
            mbar_ptr0,
            warp_idx,
            tile_copy_bytes,
            stage_count,
        ) = load_ctx
        mbar_ptr = mbar_ptr0
        if cutlass.const_expr(stage_count > 1):
            mbar_ptr = mbar_ptr0 + stage_idx
        gIN_tile = cute.local_tile(
            tma_tensor_in, tile_shape, make_tile_coord(tile_idx, orthogonal_idx)
        )
        issue_tma_load(
            tma_atom_in,
            (gIN_tile, sIN_tiles[stage_idx], 1),
            mbar_ptr,
            warp_idx,
            tile_copy_bytes,
        )

    @cute.jit
    def run_2d_tma_quant_pipeline(
        pipeline_ctx: tuple,
        tma_ctx: tuple,
        make_tile_coord: cutlass.Constexpr,
        consume: cutlass.Constexpr,
    ):
        """Drive the warp-specialized TMA quantization pipeline for a 2D tile.

        Shared by the ``1x32`` (scale along K) and ``32x1`` (scale along M) MXFP8
        kernels. The two differ only in which grid axis is pipelined and how each
        tile maps to global coordinates / is consumed; those are supplied as the
        ``make_tile_coord`` and ``consume`` callbacks. Staged-buffer setup lives
        in ``_init_staged_pipeline`` and the per-step TMA load in
        ``_load_pipeline_tile`` to keep this loop shallow.

        Args:
            pipeline_ctx: ``(storage, smem_layout_in, smem_layout_out, offs,
                tile_shape, tile_copy_bytes, tiles_per_cta, stage_count,
                compute_warps)`` where ``tile_shape`` is ``(TILE_M, TILE_K)``.
            tma_ctx: ``(tma_atom_in, tma_tensor_in, tma_atom_out,
                tma_tensor_out)``.
            make_tile_coord: ``fn(tile_eff, orthogonal_idx) -> coord``.
            consume: ``fn(sIN_tile, sOUT_tile, tile_eff, orthogonal_idx,
                warp_idx)``.
        """
        (
            storage,
            smem_layout_in,
            smem_layout_out,
            offs,
            tile_shape,
            tile_copy_bytes,
            tiles_per_cta,
            stage_count,
            compute_warps,
        ) = pipeline_ctx
        tma_atom_in, tma_tensor_in, tma_atom_out, tma_tensor_out = tma_ctx

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        # Validate group sizes are multiples of 128 if offs is provided
        if cutlass.const_expr(offs is not None):
            if tidx == 0:
                validate_group_sizes(offs)

        # The tuned contract keeps stage_count <= 2.
        sIN_tiles, sOUT_tiles, mbar_ptr0 = _init_staged_pipeline(
            storage,
            (smem_layout_in, smem_layout_out),
            tile_shape,
            stage_count,
            tidx,
            (tma_atom_in, tma_atom_out),
        )

        tile_group_idx = cutlass.Int64(bidx)
        orthogonal_idx = cutlass.Int64(bidy)
        pipelined_first = cutlass.const_expr(
            not (stage_count > 1 and tiles_per_cta > 1)
        )
        load_ctx = (
            tma_atom_in,
            tma_tensor_in,
            tile_shape,
            make_tile_coord,
            orthogonal_idx,
            sIN_tiles,
            mbar_ptr0,
            warp_idx,
            tile_copy_bytes,
            stage_count,
        )

        step_ctx = (
            load_ctx,
            sIN_tiles,
            sOUT_tiles,
            mbar_ptr0,
            tile_group_idx,
            orthogonal_idx,
            tiles_per_cta,
            stage_count,
            pipelined_first,
            warp_idx,
            compute_warps,
            tma_atom_out,
            tma_tensor_out,
            tile_shape,
            make_tile_coord,
            consume,
        )
        for tile_step in cutlass.range_constexpr(tiles_per_cta):
            _run_pipeline_step(step_ctx, tile_step)

    @cute.jit
    def _run_pipeline_step(step_ctx: tuple, tile_step):
        """Run one tile step of the TMA quantization pipeline."""
        (
            load_ctx,
            sIN_tiles,
            sOUT_tiles,
            mbar_ptr0,
            tile_group_idx,
            orthogonal_idx,
            tiles_per_cta,
            stage_count,
            pipelined_first,
            warp_idx,
            compute_warps,
            tma_atom_out,
            tma_tensor_out,
            tile_shape,
            make_tile_coord,
            consume,
        ) = step_ctx

        tile_eff = tile_group_idx * tiles_per_cta + tile_step
        stage_idx = tile_step % stage_count
        sIN_tile = sIN_tiles[stage_idx]
        sOUT_tile = sOUT_tiles[stage_idx]
        mbar_ptr = mbar_ptr0
        if cutlass.const_expr(stage_count > 1):
            mbar_ptr = mbar_ptr0 + stage_idx
        tma_phase = (tile_step // stage_count) % 2

        if cutlass.const_expr(tile_step == 0 or pipelined_first):
            _load_pipeline_tile(load_ctx, tile_eff, stage_idx)

        if cutlass.const_expr(not pipelined_first and tile_step + 1 < tiles_per_cta):
            _load_pipeline_tile(
                load_ctx,
                tile_group_idx * tiles_per_cta + tile_step + 1,
                (tile_step + 1) % stage_count,
            )

        if warp_idx >= 1 and warp_idx <= compute_warps:
            cute.arch.mbarrier_wait(mbar_ptr, tma_phase)
            consume(sIN_tile, sOUT_tile, tile_eff, orthogonal_idx, warp_idx)

        gOUT_tile = cute.local_tile(
            tma_tensor_out, tile_shape, make_tile_coord(tile_eff, orthogonal_idx)
        )
        issue_tma_store(tma_atom_out, (gOUT_tile, sOUT_tile, 1), warp_idx)

    @cute.jit
    def _quant_full_lane(sizes: tuple, ops: tuple, flags: tuple, outer, outer_rel):
        """Quantize every block of one fully-in-bounds orthogonal lane."""
        _, _, _, blocks_per_tile, scale_dim, _, _ = sizes
        kernel_self, sIN_tile, sOUT_tile = ops
        use_rceil, _, blocked_scale_output, scales_tensor = flags[0:4]
        tile_block_base = flags[7]

        scale_buffer = cute.make_rmem_tensor((blocks_per_tile,), cutlass.Uint8)
        for b in cutlass.range_constexpr(blocks_per_tile):
            block_base = b * scale_dim
            vals_block = kernel_self._load_block_full_smem_to_reg(
                sIN_tile, outer_rel, block_base
            )
            amax = compute_amax(vals_block)
            scale_biased, inv_scale = compute_scale_from_amax(amax, use_rceil)
            scale_buffer[b] = cutlass.Uint8(scale_biased)
            kernel_self._quantize_block_then_store_reg_to_smem_full(
                vals_block, inv_scale, sOUT_tile, outer_rel, block_base, use_rceil
            )

        kernel_self._store_scales_reg_to_gmem_vec(
            scales_tensor,
            outer,
            tile_block_base,
            scale_buffer,
            cutlass.Int32(blocks_per_tile),
            blocked_scale_output,
        )

    @cute.jit
    def _quant_tail_lane(sizes: tuple, ops: tuple, flags: tuple, outer, outer_rel):
        """Quantize the valid blocks of one partially-in-bounds orthogonal lane."""
        _, _, _, blocks_per_tile, scale_dim, _, _ = sizes
        kernel_self, sIN_tile, sOUT_tile = ops
        use_rceil, _, blocked_scale_output, scales_tensor = flags[0:4]
        _, orthogonal_size, pipelined_offset, tile_block_base, total_blocks = flags[4:9]

        scale_buffer = cute.make_rmem_tensor((blocks_per_tile,), cutlass.Uint8)
        num_valid_scales = cutlass.Int32(0)
        for b in cutlass.range_constexpr(blocks_per_tile):
            if tile_block_base + b < total_blocks:
                block_base = b * scale_dim
                vals_block = kernel_self._load_block_tail_smem_to_reg(
                    sIN_tile, pipelined_offset, outer_rel, block_base, orthogonal_size
                )
                amax = compute_amax(vals_block)
                scale_biased, inv_scale = compute_scale_from_amax(amax, use_rceil)
                scale_buffer[num_valid_scales] = cutlass.Uint8(scale_biased)
                num_valid_scales = num_valid_scales + 1
                kernel_self._quantize_block_then_store_reg_to_smem_tail(
                    vals_block,
                    inv_scale,
                    sOUT_tile,
                    pipelined_offset,
                    outer_rel,
                    block_base,
                    orthogonal_size,
                    use_rceil,
                )

        if num_valid_scales > 0:
            kernel_self._store_scales_reg_to_gmem_vec(
                scales_tensor,
                outer,
                tile_block_base,
                scale_buffer,
                num_valid_scales,
                blocked_scale_output,
            )

    @cute.jit
    def run_2d_quant_consumer(sizes: tuple, ops: tuple, flags: tuple):
        """Quantize one tile's worth of blocks for the consumer warps.

        Shared by the ``1x32`` (scale along K) and ``32x1`` (scale along M)
        kernels. For each lane row along the *orthogonal* axis, buffer one scale
        per block along the *pipelined* axis, quantize each block, then
        vector-store the scales. The full-tile and tail-tile lane bodies live in
        ``_quant_full_lane`` / ``_quant_tail_lane`` so the axis-specific sizes,
        offsets, and per-block ops (all in ``sizes`` / ``ops`` / ``flags``) are
        shared by both callers.

        Args:
            sizes: ``(iters_per_lane, threads, tile_dim, blocks_per_tile,
                scale_dim, warp_idx, tidx)``.
            ops: ``(kernel_self, sIN_tile, sOUT_tile)`` — the per-axis kernel
                instance plus the input/output shared-memory tiles.
            flags: ``(use_rceil, is_full_tiles, blocked_scale_output,
                scales_tensor, orthogonal_base, orthogonal_size,
                pipelined_offset, tile_block_base, total_blocks)``.
        """
        iters_per_lane, threads, tile_dim = sizes[0:3]
        warp_idx, tidx = sizes[5:7]
        is_full_tiles = flags[1]
        orthogonal_base, orthogonal_size = flags[4:6]

        lane = tidx % 32
        outer_lane = (warp_idx - 1) * 32 + lane

        for it in cutlass.range_constexpr(iters_per_lane):
            outer_rel = outer_lane + it * threads
            outer = orthogonal_base + outer_rel
            if cutlass.const_expr(is_full_tiles):
                if outer_rel < tile_dim:
                    _quant_full_lane(sizes, ops, flags, outer, outer_rel)
            elif outer_rel < tile_dim and outer < orthogonal_size:
                _quant_tail_lane(sizes, ops, flags, outer, outer_rel)

    @cute.jit
    def run_2d_quant_kernel(kernel_self, kernel_ctx: tuple, axis: cutlass.Constexpr):
        """Full body of the 2D MXFP8 quantization kernel, shared by both scalings.

        The ``1x32`` (scale along K) and ``32x1`` (scale along M) kernels are
        transpose-symmetric: one axis is *orthogonal* (its position selects the
        lane) and the other is *pipelined* (tiled across the grid). ``axis``
        carries which is which plus the per-axis compile constants, so a single
        body serves both — eliminating the near-duplicate ``kernel`` methods.

        Args:
            kernel_self: The kernel instance providing the register/smem helpers.
            kernel_ctx: ``(storage, smem_layouts, tma_tensors, scales_out_u8,
                blocked_scale_layout, offs, orthogonal_size, total_blocks)`` —
                the launch-time tensors and the two runtime axis sizes
                (``orthogonal_size`` and ``total_blocks``).
            axis: ``(orthogonal_is_m, tile_shape, tile_orth, tile_pipe,
                iters_per_lane, threads, blocks_per_tile, scale_dim,
                tile_copy_bytes, tiles_per_cta, stage_count, compute_warps,
                use_rceil, is_full_tiles, blocked_scale_output)``.
        """
        (
            storage,
            smem_layouts,
            tma_tensors,
            scales_out_u8,
            blocked_scale_layout,
            offs,
            orthogonal_size,
            total_blocks,
        ) = kernel_ctx
        (
            orthogonal_is_m,
            tile_shape,
            tile_orth,
            tile_pipe,
            iters_per_lane,
            threads,
            blocks_per_tile,
            scale_dim,
            tile_copy_bytes,
            tiles_per_cta,
            stage_count,
            compute_warps,
            use_rceil,
            is_full_tiles,
            blocked_scale_output,
        ) = axis
        tma_atom_in, tma_tensor_in, tma_atom_out, tma_tensor_out = tma_tensors
        smem_layout_in, smem_layout_out = smem_layouts

        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(blocked_scale_output):
            scales_tensor = cute.make_tensor(
                scales_out_u8.iterator, blocked_scale_layout
            )
        else:
            scales_tensor = scales_out_u8

        def make_tile_coord(pipe_idx, orthogonal_idx):
            if cutlass.const_expr(orthogonal_is_m):
                return (orthogonal_idx, pipe_idx)
            return (pipe_idx, orthogonal_idx)

        def consume(sIN_tile, sOUT_tile, pipe_idx, orthogonal_idx, warp_idx):
            run_2d_quant_consumer(
                (
                    iters_per_lane,
                    threads,
                    tile_orth,
                    blocks_per_tile,
                    scale_dim,
                    warp_idx,
                    tidx,
                ),
                (kernel_self, sIN_tile, sOUT_tile),
                (
                    use_rceil,
                    is_full_tiles,
                    blocked_scale_output,
                    scales_tensor,
                    orthogonal_idx * tile_orth,
                    orthogonal_size,
                    pipe_idx * tile_pipe,
                    pipe_idx * blocks_per_tile,
                    total_blocks,
                ),
            )

        run_2d_tma_quant_pipeline(
            (
                storage,
                smem_layout_in,
                smem_layout_out,
                offs,
                tile_shape,
                tile_copy_bytes,
                tiles_per_cta,
                stage_count,
                compute_warps,
            ),
            (tma_atom_in, tma_tensor_in, tma_atom_out, tma_tensor_out),
            make_tile_coord,
            consume,
        )

    # FP8 constants
    F8_MAX = cutlass.Float32(448.0)
    INV_F8_MAX = cutlass.Float32(1.0 / 448.0)

    # PTX inline assembly for RCEIL conversion on Blackwell
    @dsl_user_op
    def _cvt_rp_satfinite_ue8m0x2_f32(
        a: cutlass.Float32,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Uint16:
        """PTX inline assembly for RCEIL conversion.

        Uses inline PTX on Blackwell-family targets because CuTeDSL does not
        currently lower this conversion to `cvt.rp.satfinite.ue8m0x2.f32` on its own.
        """
        return cutlass.Uint16(
            llvm.inline_asm(
                T.i16(),
                [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
                "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
                "=h,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    # Shared scale computation methods
    @cute.jit
    def compute_amax(vals_block: cute.Tensor):
        """Compute absolute maximum of a block of values.

        Args:
            vals_block: Tensor of values to compute amax from

        Returns:
            The absolute maximum value as Float32
        """
        vals_vec = vals_block.load()
        abs_vec = cute.where(vals_vec < 0, -vals_vec, vals_vec)
        return cutlass.Float32(
            abs_vec.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        )

    @cute.jit
    def compute_scale_rceil(amax: cutlass.Float32):
        """Compute scale using RCEIL (round-up) mode with Blackwell PTX inline assembly.

        Uses inline PTX `cvt.rp.satfinite.ue8m0x2.f32` instruction for optimal performance
        on Blackwell (SM 10.x) and later architectures.

        Args:
            amax: Absolute maximum value

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        # referene: https://github.com/pytorch/ao/blob/ac0b820899b0a5d415310f798c9c96b5a5973f53/torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh#L538
        descale = amax * INV_F8_MAX
        scale_biased = cutlass.Int32(_cvt_rp_satfinite_ue8m0x2_f32(descale))
        inv_scale = cutlass.Float32(1.0)
        if scale_biased == 0xFF:
            inv_scale = cutlass.Float32(0.0)
        elif scale_biased == 0:
            inv_scale = cute.exp2(cutlass.Float32(126.0))
        else:
            inv_scale = cute.exp2(cutlass.Float32(127 - scale_biased))
        return scale_biased, inv_scale

    @cute.jit
    def compute_scale_floor(amax: cutlass.Float32):
        """Compute scale using FLOOR mode.

        Args:
            amax: Absolute maximum value

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        # reference: https://github.com/pytorch/ao/blob/ac0b820899b0a5d415310f798c9c96b5a5973f53/torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh#L520
        bits = _dsl_arith.bitcast(amax.ir_value(), _dsl_arith.T.i32())
        exp_i = ((bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)) - cutlass.Int32(127)
        scale_exp_unbiased = exp_i - cutlass.Int32(8)
        if scale_exp_unbiased < -127:
            scale_exp_unbiased = cutlass.Int32(-127)
        if scale_exp_unbiased > 128:
            scale_exp_unbiased = cutlass.Int32(128)
        inv_scale = cute.exp2(cutlass.Float32(-scale_exp_unbiased))
        scale_biased = scale_exp_unbiased + 127
        return scale_biased, inv_scale

    @cute.jit
    def compute_scale_from_amax(
        amax: cutlass.Float32,
        USE_RCEIL: cutlass.Constexpr[bool],
    ):
        """Compute scale from absolute maximum using specified mode.

        Args:
            amax: Absolute maximum value
            USE_RCEIL: Constexpr boolean for scaling mode (True for RCEIL, False for FLOOR)

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        scale_biased = cutlass.Int32(0)
        inv_scale = cutlass.Float32(1.0)
        if amax > 0:
            if cutlass.const_expr(USE_RCEIL):
                scale_biased, inv_scale = compute_scale_rceil(amax)
            else:
                scale_biased, inv_scale = compute_scale_floor(amax)
        return scale_biased, inv_scale

    @cute.jit
    def load_vals_chunk_full(
        vals_block: cute.Tensor,
        local_base: cutlass.Int32,
    ):
        """Load a full chunk of 4 values from a values block.

        This helper loads 4 consecutive float32 values from a register tensor
        starting at the given local base index.

        Args:
            vals_block: Register tensor containing values to load from
            local_base: Starting index within vals_block for the chunk

        Returns:
            Register tensor of shape (4,) containing the loaded float32 values
        """
        chunk_vec = 4
        vals_chunk = cute.make_rmem_tensor((chunk_vec,), cutlass.Float32)
        for j in range(chunk_vec):
            vals_chunk[j] = vals_block[local_base + j]
        return vals_chunk

    @cute.jit
    def load_vals_chunk_tail(
        vals_block: cute.Tensor,
        dim0: cutlass.Int64,
        sout_base: cutlass.Int32,
        local_base: cutlass.Int32,
        dim_size: cutlass.Int64,
    ):
        """Load a tail chunk of 4 values with bounds checking.

        This helper loads 4 values from a values block, checking if each position
        is within the dimension bounds. Out-of-bounds values are replaced with 0.0.

        Args:
            vals_block: Register tensor containing values to load from
            dim0: Starting index in the dimension (e.g., k0 or n0)
            sout_base: Base offset for output indexing
            local_base: Starting index within vals_block for the chunk
            dim_size: Total size of the dimension for bounds checking (e.g., K or N)

        Returns:
            Register tensor of shape (4,) containing the loaded float32 values,
            with out-of-bounds positions set to 0.0
        """
        chunk_vec = 4
        vals_chunk = cute.make_rmem_tensor((chunk_vec,), cutlass.Float32)
        for j in range(chunk_vec):
            idx = dim0 + sout_base + j
            if idx < dim_size:
                vals_chunk[j] = vals_block[local_base + j]
            else:
                vals_chunk[j] = cutlass.Float32(0.0)
        return vals_chunk

    @cute.jit
    def validate_group_sizes(offs: cute.Tensor):
        # Only first thread validates to avoid redundant work
        num_groups = offs.shape[0]

        # Validate first group (from 0 to offs[0])
        if num_groups > 0:
            first_group_size = offs[0]
            cute.testing.assert_(
                first_group_size % 128 == 0,
                "Group sizes must be multiples of 128",
            )

        # Validate subsequent groups
        for i in range(1, num_groups):
            prev_end = offs[i - 1]
            curr_end = offs[i]
            group_size = curr_end - prev_end
            cute.testing.assert_(
                group_size % 128 == 0,
                "Group sizes must be multiples of 128",
            )
