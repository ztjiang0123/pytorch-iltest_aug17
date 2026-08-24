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
    from cutlass.cute.nvgpu import cpasync as _cpasync
    from cutlass.cutlass_dsl import T, dsl_user_op

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

    def setup_kernel_smem_and_barriers(
        storage,
        smem_layout_in,
        smem_layout_out,
        tidx,
        tma_atom_in,
        tma_atom_out,
        stage_count: int,
        tile_m: int,
        tile_k: int,
    ):
        """Allocate staged SMEM tiles and initialize the TMA mbarriers.

        This is the warp-specialization prologue shared verbatim by the 1x32
        and 32x1 MXFP8 quantization kernels: it computes the (up to two) staged
        input/output SMEM tiles, sets up the mbarrier pointers, prefetches the
        TMA descriptors and issues the init fence + block sync. Only the SMEM
        output layout (row- vs column-major) differs between the two callers,
        so it is passed in as ``smem_layout_out``.

        This is a plain (non-``@cute.jit``) helper: the cute operations it
        performs are inlined into the caller's trace, matching the existing
        pattern used by ``_make_tile_smem_layouts``. ``stage_count``, ``tile_m``
        and ``tile_k`` are compile-time constants supplied by the caller.

        Returns:
            (sIN_tile0, sOUT_tile0, sIN_tile1, sOUT_tile1,
             tma_mbar_ptr0, tma_mbar_ptr1)
            The ``*_tile1``/``tma_mbar_ptr1`` values alias stage 0 when
            ``stage_count == 1``.
        """
        # The tuned contract keeps stage_count <= 2.
        tma_mbar_ptr0 = storage.tma_mbar_ptr.data_ptr()
        tma_mbar_ptr1 = tma_mbar_ptr0
        if cutlass.const_expr(stage_count > 1):
            tma_mbar_ptr1 = tma_mbar_ptr0 + 1

        staged_layout_in = cute.make_layout(
            (stage_count, tile_m, tile_k),
            stride=(tile_m * tile_k, tile_k, 1),
        )
        staged_layout_out = cute.make_layout(
            (stage_count, tile_m, tile_k),
            stride=(tile_m * tile_k, tile_k, 1),
        )
        sIN_staged = storage.in_smem.get_tensor(staged_layout_in)
        sOUT_staged = storage.out_smem.get_tensor(staged_layout_out)
        stage_elems = tile_m * tile_k
        sIN_tile0 = cute.make_tensor(
            sIN_staged.iterator + 0 * stage_elems, smem_layout_in
        )
        sOUT_tile0 = cute.make_tensor(
            sOUT_staged.iterator + 0 * stage_elems, smem_layout_out
        )
        sIN_tile1 = sIN_tile0
        sOUT_tile1 = sOUT_tile0
        if cutlass.const_expr(stage_count > 1):
            sIN_tile1 = cute.make_tensor(
                sIN_staged.iterator + 1 * stage_elems, smem_layout_in
            )
            sOUT_tile1 = cute.make_tensor(
                sOUT_staged.iterator + 1 * stage_elems, smem_layout_out
            )

        if tidx == 0:
            _cpasync.prefetch_descriptor(tma_atom_in)
            _cpasync.prefetch_descriptor(tma_atom_out)
            cute.arch.mbarrier_init(tma_mbar_ptr0, 1)
            if cutlass.const_expr(stage_count > 1):
                cute.arch.mbarrier_init(tma_mbar_ptr1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        return (
            sIN_tile0,
            sOUT_tile0,
            sIN_tile1,
            sOUT_tile1,
            tma_mbar_ptr0,
            tma_mbar_ptr1,
        )

    def run_quantize_tile_loop(
        kernel,
        *,
        warp_idx,
        tidx,
        compute_warps,
        tma_atom_in,
        tma_tensor_in,
        tma_atom_out,
        tma_tensor_out,
        scales_tensor,
        sIN_tile0,
        sOUT_tile0,
        sIN_tile1,
        sOUT_tile1,
        tma_mbar_ptr0,
        tile_m: int,
        tile_k: int,
        stage_count: int,
        tiles_per_cta: int,
        group_tile_idx,
        fixed_tile,
        group_is_first_coord: bool,
        lane_tile_size: int,
        lane_iters: int,
        lane_threads: int,
        lane_axis_size,
        block_tile_size: int,
        blocks_per_tile: int,
        scale_dim: int,
        block_axis_size,
        num_blocks,
        is_full_tiles: bool,
        use_rceil: bool,
        blocked_scale_output: bool,
    ):
        """Warp-specialized tile loop shared by the 1x32 and 32x1 MXFP8 kernels.

        The two kernels quantize along transposed axes (1x32 along K, 32x1 along
        M) but are otherwise structurally identical: the same TMA pipeline, the
        same per-lane block scan, and the same vectorized scale store. This
        helper expresses that shared body once, parametrized by axis roles:

        - ``group_tile_idx`` / ``tiles_per_cta`` drive the outer CTA tile loop;
          ``fixed_tile`` is the tile index on the non-grouped axis.
        - ``group_is_first_coord`` selects the ``cute.local_tile`` coordinate
          order (32x1 puts the grouped M tile first, 1x32 puts the fixed M tile
          first).
        - ``lane_*`` describe the axis each thread-lane strides over; ``block_*``
          and ``scale_dim`` describe the axis quantized into 32-element blocks.
        - ``lane_axis_size`` / ``block_axis_size`` / ``num_blocks`` supply the
          bounds used by the tail (partial-tile) path.

        The block-level primitives (``kernel._load_block_*``,
        ``kernel._quantize_block_then_store_reg_to_smem_*``,
        ``kernel._store_scales_reg_to_gmem_vec``) take (lane_rel, block_base)
        arguments with identical roles in both kernels, so they are called
        directly on the passed ``kernel`` instance.

        This is a plain (non-``@cute.jit``) helper; its cute operations are
        inlined into the caller's kernel trace.
        """

        def _tile_coord(group_coord, fixed_coord):
            if cutlass.const_expr(group_is_first_coord):
                return (group_coord, fixed_coord)
            return (fixed_coord, group_coord)

        lane_base = fixed_tile * lane_tile_size

        for tile_step in cutlass.range_constexpr(tiles_per_cta):
            tile_eff = group_tile_idx * tiles_per_cta + tile_step
            block_axis_tile_base = tile_eff * block_tile_size

            stage_idx = tile_step % stage_count

            sIN_tile = sIN_tile0
            sOUT_tile = sOUT_tile0
            tma_mbar_ptr = tma_mbar_ptr0
            if cutlass.const_expr(stage_count > 1):
                tma_mbar_ptr = tma_mbar_ptr0 + stage_idx
            if cutlass.const_expr(stage_count > 1):
                if stage_idx == 1:
                    sIN_tile = sIN_tile1
                    sOUT_tile = sOUT_tile1

            tma_phase = (tile_step // stage_count) % 2

            if cutlass.const_expr(
                tile_step == 0 or not (stage_count > 1 and tiles_per_cta > 1)
            ):
                gIN_tile = cute.local_tile(
                    tma_tensor_in,
                    (tile_m, tile_k),
                    _tile_coord(tile_eff, fixed_tile),
                )
                kernel._issue_tma_load(
                    tma_atom_in,
                    gIN_tile,
                    sIN_tile,
                    tma_mbar_ptr,
                    warp_idx,
                )

            if cutlass.const_expr(stage_count > 1 and tiles_per_cta > 1):
                if cutlass.const_expr(tile_step + 1 < tiles_per_cta):
                    tile_next = group_tile_idx * tiles_per_cta + tile_step + 1
                    next_stage_idx = (tile_step + 1) % stage_count
                    sIN_tile_next = sIN_tile0
                    tma_mbar_ptr_next = tma_mbar_ptr0
                    if cutlass.const_expr(stage_count > 1):
                        tma_mbar_ptr_next = tma_mbar_ptr0 + next_stage_idx
                    if cutlass.const_expr(stage_count > 1):
                        if next_stage_idx == 1:
                            sIN_tile_next = sIN_tile1

                    gIN_tile_next = cute.local_tile(
                        tma_tensor_in,
                        (tile_m, tile_k),
                        _tile_coord(tile_next, fixed_tile),
                    )
                    kernel._issue_tma_load(
                        tma_atom_in,
                        gIN_tile_next,
                        sIN_tile_next,
                        tma_mbar_ptr_next,
                        warp_idx,
                    )

            if warp_idx >= 1 and warp_idx <= compute_warps:
                cute.arch.mbarrier_wait(tma_mbar_ptr, tma_phase)
                lane = tidx % 32
                lane_idx = (warp_idx - 1) * 32 + lane

                for it in cutlass.range_constexpr(lane_iters):
                    lane_rel = lane_idx + it * lane_threads
                    lane_global = lane_base + lane_rel
                    if cutlass.const_expr(is_full_tiles):
                        if lane_rel < lane_tile_size:
                            # Buffer scales for vectorized store
                            scale_buffer = cute.make_rmem_tensor(
                                (blocks_per_tile,), cutlass.Uint8
                            )

                            for blk in cutlass.range_constexpr(blocks_per_tile):
                                block_base = blk * scale_dim
                                vals_block = kernel._load_block_full_smem_to_reg(
                                    sIN_tile,
                                    lane_rel,
                                    block_base,
                                )

                                amax = compute_amax(vals_block)

                                scale_biased, inv_scale = compute_scale_from_amax(
                                    amax, use_rceil
                                )
                                scale_buffer[blk] = cutlass.Uint8(scale_biased)

                                kernel._quantize_block_then_store_reg_to_smem_full(
                                    vals_block,
                                    inv_scale,
                                    sOUT_tile,
                                    lane_rel,
                                    block_base,
                                    use_rceil,
                                )

                            # Vectorized scale store
                            block_index_base = tile_eff * blocks_per_tile
                            kernel._store_scales_reg_to_gmem_vec(
                                scales_tensor,
                                lane_global,
                                block_index_base,
                                scale_buffer,
                                cutlass.Int32(blocks_per_tile),
                                blocked_scale_output,
                            )
                    else:
                        lane_in_bounds = lane_global < lane_axis_size
                        if lane_rel < lane_tile_size and lane_in_bounds:
                            # Buffer scales for vectorized store
                            scale_buffer = cute.make_rmem_tensor(
                                (blocks_per_tile,), cutlass.Uint8
                            )
                            num_valid_scales = cutlass.Int32(0)

                            for blk in cutlass.range_constexpr(blocks_per_tile):
                                block_global = tile_eff * blocks_per_tile + blk
                                if block_global < num_blocks:
                                    block_base = blk * scale_dim
                                    vals_block = kernel._load_block_tail_smem_to_reg(
                                        sIN_tile,
                                        block_axis_tile_base,
                                        lane_rel,
                                        block_base,
                                        block_axis_size,
                                    )

                                    amax = compute_amax(vals_block)

                                    scale_biased, inv_scale = compute_scale_from_amax(
                                        amax, use_rceil
                                    )
                                    scale_buffer[num_valid_scales] = cutlass.Uint8(
                                        scale_biased
                                    )
                                    num_valid_scales = num_valid_scales + 1

                                    kernel._quantize_block_then_store_reg_to_smem_tail(
                                        vals_block,
                                        inv_scale,
                                        sOUT_tile,
                                        block_axis_tile_base,
                                        lane_rel,
                                        block_base,
                                        block_axis_size,
                                        use_rceil,
                                    )

                            # Vectorized scale store
                            if num_valid_scales > 0:
                                block_index_base = tile_eff * blocks_per_tile
                                kernel._store_scales_reg_to_gmem_vec(
                                    scales_tensor,
                                    lane_global,
                                    block_index_base,
                                    scale_buffer,
                                    num_valid_scales,
                                    blocked_scale_output,
                                )

            gOUT_tile = cute.local_tile(
                tma_tensor_out,
                (tile_m, tile_k),
                _tile_coord(tile_eff, fixed_tile),
            )
            kernel._issue_tma_store(
                tma_atom_out,
                gOUT_tile,
                sOUT_tile,
                warp_idx,
            )
