# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for CuTeDSL quantization kernels."""

import importlib.util
from types import SimpleNamespace

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
    def store_scales_reg_to_gmem_vec(
        scales_tensor: cute.Tensor,
        lane_coord: cutlass.Int64,
        block_base: cutlass.Int64,
        scale_buffer: cute.Tensor,
        num_scales: cutlass.Int32,
        BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
    ):
        """Store scales from registers to global memory using vectorized writes.

        Shared by the 1x32 and 32x1 MXFP8 kernels. The two kernels quantize
        along transposed axes, but both index the scale tensor as
        ``scales_tensor[lane_coord, block]`` (1x32: lane=M, block=K-block;
        32x1: lane=K, block=M-block), so the store body is identical. Uses a
        uint32 vectorized write for 4 scales in blocked layout, falling back to
        scalar stores otherwise.

        Args:
            scales_tensor: Output scales in global memory
            lane_coord: Global coordinate of the lane axis (M for 1x32, K for 32x1)
            block_base: Starting block index along the quantized axis
            scale_buffer: Buffer of scales in register memory (uint8)
            num_scales: Number of scales to store
            BLOCKED_SCALE_OUTPUT: Whether using blocked layout (enables vectorization)

        Storage locations:
            Input: scale_buffer (registers)
            Output: scales_tensor (global memory)
        """
        if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
            # Blocked layout with 4 contiguous scales - write as uint32
            if num_scales == 4:
                # Pack 4 uint8 scales into uint32 and write
                scales_tensor_u32 = cute.recast_tensor(scales_tensor, cutlass.Uint32)
                scale_buffer_u32 = cute.recast_tensor(scale_buffer, cutlass.Uint32)
                scales_tensor_u32[lane_coord, block_base // cutlass.Int64(4)] = (
                    scale_buffer_u32[0]
                )
            else:
                # Fallback for non-4 cases (e.g., tail tiles)
                for i in range(num_scales):
                    scales_tensor[lane_coord, block_base + i] = scale_buffer[i]
        else:
            # Row-major layout - scalar stores
            for i in range(num_scales):
                scales_tensor[lane_coord, block_base + i] = scale_buffer[i]

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

    def make_tile_shape(*, tile_m, tile_k, stage_count):
        """Bundle compile-time tile geometry for the MXFP8 kernels."""
        return SimpleNamespace(tile_m=tile_m, tile_k=tile_k, stage_count=stage_count)

    def make_tma_handles(*, atom_in, tensor_in, atom_out, tensor_out):
        """Bundle the input/output TMA atoms and tensor views for a kernel."""
        return SimpleNamespace(
            atom_in=atom_in,
            tensor_in=tensor_in,
            atom_out=atom_out,
            tensor_out=tensor_out,
        )

    def make_axis_spec(**fields):
        """Bundle the axis-role parameters distinguishing 1x32 from 32x1.

        The two kernels are structurally identical up to a transposed axis
        (1x32 quantizes along K, 32x1 along M). ``fields`` captures which axis
        is the grouped CTA-tile axis (``tiles_per_cta``, ``group_tile_idx``,
        ``group_is_first_coord``), which axis each lane strides over (``lane_*``)
        and which axis is quantized into 32-element blocks (``block_*``,
        ``scale_dim``), plus the ``*_axis_size`` / ``num_blocks`` bounds for the
        tail (partial-tile) path and the ``is_full_tiles`` flag.
        """
        return SimpleNamespace(**fields)

    def make_quant_opts(*, use_rceil, blocked_scale_output):
        """Bundle per-launch quantization options for the tile-loop helpers."""
        return SimpleNamespace(
            use_rceil=use_rceil, blocked_scale_output=blocked_scale_output
        )

    def _axis_tile_coord(axis, group_coord, fixed_coord):
        """Order a (group, fixed) tile index pair into (M, K) coordinates."""
        if cutlass.const_expr(axis.group_is_first_coord):
            return (group_coord, fixed_coord)
        return (fixed_coord, group_coord)

    def setup_kernel_smem_and_barriers(storage, smem_layouts, tidx, tma, shape):
        """Allocate staged SMEM tiles and initialize the TMA mbarriers.

        Warp-specialization prologue shared verbatim by the 1x32 and 32x1 MXFP8
        kernels: computes the (up to two) staged input/output SMEM tiles, sets
        up mbarrier pointers, prefetches TMA descriptors, and issues the init
        fence + block sync. Only the SMEM output layout (row- vs column-major)
        differs between callers, so it arrives via ``smem_layouts``.

        Plain (non-``@cute.jit``) helper: its cute ops inline into the caller's
        trace, like ``_make_tile_smem_layouts``.

        Args:
            storage: Allocated ``SharedStorage`` instance.
            smem_layouts: ``(smem_layout_in, smem_layout_out)`` per-tile layouts.
            tidx: Thread index (only lane 0 initializes the barriers).
            tma: TMA handles (see ``make_tma_handles``).
            shape: Tile geometry (see ``make_tile_shape``).

        Returns:
            A staging namespace with ``in0/out0/in1/out1`` SMEM tiles and
            ``mbar0/mbar1`` barrier pointers (stage-1 fields alias stage 0 when
            ``stage_count == 1``).
        """
        smem_layout_in, smem_layout_out = smem_layouts
        stage_count = shape.stage_count
        tile_m = shape.tile_m
        tile_k = shape.tile_k

        # The tuned contract keeps stage_count <= 2.
        mbar0 = storage.tma_mbar_ptr.data_ptr()
        mbar1 = mbar0
        if cutlass.const_expr(stage_count > 1):
            mbar1 = mbar0 + 1

        staged_layout = cute.make_layout(
            (stage_count, tile_m, tile_k),
            stride=(tile_m * tile_k, tile_k, 1),
        )
        sIN_staged = storage.in_smem.get_tensor(staged_layout)
        sOUT_staged = storage.out_smem.get_tensor(staged_layout)
        stage_elems = tile_m * tile_k
        in0 = cute.make_tensor(sIN_staged.iterator + 0 * stage_elems, smem_layout_in)
        out0 = cute.make_tensor(sOUT_staged.iterator + 0 * stage_elems, smem_layout_out)
        in1 = in0
        out1 = out0
        if cutlass.const_expr(stage_count > 1):
            in1 = cute.make_tensor(
                sIN_staged.iterator + 1 * stage_elems, smem_layout_in
            )
            out1 = cute.make_tensor(
                sOUT_staged.iterator + 1 * stage_elems, smem_layout_out
            )

        if tidx == 0:
            _cpasync.prefetch_descriptor(tma.atom_in)
            _cpasync.prefetch_descriptor(tma.atom_out)
            cute.arch.mbarrier_init(mbar0, 1)
            if cutlass.const_expr(stage_count > 1):
                cute.arch.mbarrier_init(mbar1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        return SimpleNamespace(
            in0=in0, out0=out0, in1=in1, out1=out1, mbar0=mbar0, mbar1=mbar1
        )

    def _select_tile_buffers(staging, stage_count, tile_step):
        """Pick the SMEM tiles and mbarrier for ``tile_step``'s pipeline stage.

        Returns ``(sIN_tile, sOUT_tile, tma_mbar_ptr)``.
        """
        sIN_tile = staging.in0
        sOUT_tile = staging.out0
        tma_mbar_ptr = staging.mbar0
        if cutlass.const_expr(stage_count > 1):
            stage_idx = tile_step % stage_count
            tma_mbar_ptr = staging.mbar0 + stage_idx
            if stage_idx == 1:
                sIN_tile = staging.in1
                sOUT_tile = staging.out1
        return sIN_tile, sOUT_tile, tma_mbar_ptr

    def _issue_tile_loads(state, tile_step, tile_eff, sIN_tile, tma_mbar_ptr):
        """Issue the current-tile TMA load and prefetch the next tile's load."""
        kernel, tma, shape, axis = state.kernel, state.tma, state.shape, state.axis
        stage_count = shape.stage_count
        tiles_per_cta = axis.tiles_per_cta
        tile_shape = (shape.tile_m, shape.tile_k)
        warp_idx = state.warp_idx

        if cutlass.const_expr(
            tile_step == 0 or not (stage_count > 1 and tiles_per_cta > 1)
        ):
            gIN_tile = cute.local_tile(
                tma.tensor_in,
                tile_shape,
                _axis_tile_coord(axis, tile_eff, axis.fixed_tile),
            )
            kernel._issue_tma_load(
                tma.atom_in, gIN_tile, sIN_tile, tma_mbar_ptr, warp_idx
            )

        if cutlass.const_expr(stage_count > 1 and tiles_per_cta > 1):
            if cutlass.const_expr(tile_step + 1 < tiles_per_cta):
                tile_next = axis.group_tile_idx * tiles_per_cta + tile_step + 1
                next_stage_idx = (tile_step + 1) % stage_count
                sIN_tile_next = (
                    state.staging.in1 if next_stage_idx == 1 else state.staging.in0
                )
                mbar_next = state.staging.mbar0 + next_stage_idx
                gIN_tile_next = cute.local_tile(
                    tma.tensor_in,
                    tile_shape,
                    _axis_tile_coord(axis, tile_next, axis.fixed_tile),
                )
                kernel._issue_tma_load(
                    tma.atom_in, gIN_tile_next, sIN_tile_next, mbar_next, warp_idx
                )

    def _quantize_lane_full(state, tiles, lane, tile_eff):
        """Quantize one full (in-bounds) lane: every block contributes a scale.

        ``tiles`` is ``(sIN_tile, sOUT_tile)`` and ``lane`` is
        ``(lane_rel, lane_global)``.
        """
        kernel, axis, opts = state.kernel, state.axis, state.opts
        sIN_tile, sOUT_tile = tiles
        lane_rel, lane_global = lane
        scale_buffer = cute.make_rmem_tensor((axis.blocks_per_tile,), cutlass.Uint8)
        for blk in cutlass.range_constexpr(axis.blocks_per_tile):
            block_base = blk * axis.scale_dim
            vals_block = kernel._load_block_full_smem_to_reg(
                sIN_tile, lane_rel, block_base
            )
            amax = compute_amax(vals_block)
            scale_biased, inv_scale = compute_scale_from_amax(amax, opts.use_rceil)
            scale_buffer[blk] = cutlass.Uint8(scale_biased)
            kernel._quantize_block_then_store_reg_to_smem_full(
                vals_block, inv_scale, sOUT_tile, lane_rel, block_base, opts.use_rceil
            )
        store_scales_reg_to_gmem_vec(
            state.scales_tensor,
            lane_global,
            tile_eff * axis.blocks_per_tile,
            scale_buffer,
            cutlass.Int32(axis.blocks_per_tile),
            opts.blocked_scale_output,
        )

    def _quantize_lane_tail(state, tiles, lane, tile_eff, block_axis_tile_base):
        """Quantize one lane on a partial tile, skipping out-of-bounds blocks.

        ``tiles`` is ``(sIN_tile, sOUT_tile)`` and ``lane`` is
        ``(lane_rel, lane_global)``.
        """
        kernel, axis, opts = state.kernel, state.axis, state.opts
        sIN_tile, sOUT_tile = tiles
        lane_rel, lane_global = lane
        scale_buffer = cute.make_rmem_tensor((axis.blocks_per_tile,), cutlass.Uint8)
        num_valid_scales = cutlass.Int32(0)
        for blk in cutlass.range_constexpr(axis.blocks_per_tile):
            block_global = tile_eff * axis.blocks_per_tile + blk
            if block_global < axis.num_blocks:
                block_base = blk * axis.scale_dim
                vals_block = kernel._load_block_tail_smem_to_reg(
                    sIN_tile,
                    block_axis_tile_base,
                    lane_rel,
                    block_base,
                    axis.block_axis_size,
                )
                amax = compute_amax(vals_block)
                scale_biased, inv_scale = compute_scale_from_amax(amax, opts.use_rceil)
                scale_buffer[num_valid_scales] = cutlass.Uint8(scale_biased)
                num_valid_scales = num_valid_scales + 1
                kernel._quantize_block_then_store_reg_to_smem_tail(
                    vals_block,
                    inv_scale,
                    sOUT_tile,
                    block_axis_tile_base,
                    lane_rel,
                    block_base,
                    axis.block_axis_size,
                    opts.use_rceil,
                )
        if num_valid_scales > 0:
            store_scales_reg_to_gmem_vec(
                state.scales_tensor,
                lane_global,
                tile_eff * axis.blocks_per_tile,
                scale_buffer,
                num_valid_scales,
                opts.blocked_scale_output,
            )

    def _quantize_tile(state, tiles, tile_eff, block_axis_tile_base):
        """Run every compute lane over one loaded tile.

        ``tiles`` is ``(sIN_tile, sOUT_tile)``.
        """
        axis = state.axis
        lane_base = axis.fixed_tile * axis.lane_tile_size
        lane_idx = (state.warp_idx - 1) * 32 + (state.tidx % 32)
        for it in cutlass.range_constexpr(axis.lane_iters):
            lane_rel = lane_idx + it * axis.lane_threads
            lane_global = lane_base + lane_rel
            lane = (lane_rel, lane_global)
            if cutlass.const_expr(axis.is_full_tiles):
                if lane_rel < axis.lane_tile_size:
                    _quantize_lane_full(state, tiles, lane, tile_eff)
            else:
                if lane_rel < axis.lane_tile_size and lane_global < axis.lane_axis_size:
                    _quantize_lane_tail(
                        state, tiles, lane, tile_eff, block_axis_tile_base
                    )

    def run_quantize_tile_loop(kernel, tma, staging, axis, opts_ctx):
        """Warp-specialized tile loop shared by the 1x32 and 32x1 MXFP8 kernels.

        The two kernels quantize along transposed axes (1x32 along K, 32x1 along
        M) but otherwise share the same TMA pipeline, per-lane block scan, and
        vectorized scale store. This expresses that shared body once, driven by
        the ``axis`` role spec.

        Args:
            kernel: Kernel instance providing the block-level primitives.
            tma: TMA handles (see ``make_tma_handles``).
            staging: Staging namespace from ``setup_kernel_smem_and_barriers``.
            axis: Axis-role spec (see ``make_axis_spec``); also carries
                ``shape`` (tile geometry).
            opts_ctx: ``(opts, warp_idx, tidx, compute_warps, scales_tensor)``:
                quantization options plus the per-thread runtime context.

        Plain (non-``@cute.jit``) helper; its cute ops inline into the trace.
        """
        opts, warp_idx, tidx, compute_warps, scales_tensor = opts_ctx
        shape = axis.shape
        stage_count = shape.stage_count
        state = SimpleNamespace(
            kernel=kernel,
            tma=tma,
            shape=shape,
            staging=staging,
            axis=axis,
            opts=opts,
            warp_idx=warp_idx,
            tidx=tidx,
            scales_tensor=scales_tensor,
        )

        for tile_step in cutlass.range_constexpr(axis.tiles_per_cta):
            tile_eff = axis.group_tile_idx * axis.tiles_per_cta + tile_step
            block_axis_tile_base = tile_eff * axis.block_tile_size

            sIN_tile, sOUT_tile, tma_mbar_ptr = _select_tile_buffers(
                staging, stage_count, tile_step
            )
            tma_phase = (tile_step // stage_count) % 2

            _issue_tile_loads(state, tile_step, tile_eff, sIN_tile, tma_mbar_ptr)

            if warp_idx >= 1 and warp_idx <= compute_warps:
                cute.arch.mbarrier_wait(tma_mbar_ptr, tma_phase)
                _quantize_tile(
                    state, (sIN_tile, sOUT_tile), tile_eff, block_axis_tile_base
                )

            gOUT_tile = cute.local_tile(
                tma.tensor_out,
                (shape.tile_m, shape.tile_k),
                _axis_tile_coord(axis, tile_eff, axis.fixed_tile),
            )
            kernel._issue_tma_store(tma.atom_out, gOUT_tile, sOUT_tile, warp_idx)

    def make_kernel_io(**fields):
        """Bundle the per-launch I/O + geometry a 2D quantize kernel needs.

        Expected fields: ``storage``, ``smem_layouts``, ``tma``, ``shape``,
        ``offs``, ``scales_out_u8``, ``blocked_scale_output``,
        ``blocked_scale_layout``, ``opts``, ``compute_warps``.
        """
        return SimpleNamespace(**fields)

    def run_quantize_2d_kernel(kernel, io, build_axis):
        """Full body of the warp-specialized 2D MXFP8 quantization kernel.

        Owns the entire kernel body — thread/warp setup, optional group-size
        validation, SMEM staging + barrier init, blocked-scale tensor selection,
        and the TMA-pipelined tile loop — so the 1x32 and 32x1 ``@cute.kernel``
        entry points reduce to a single call. The only per-kernel difference is
        the axis role assignment, provided by ``build_axis(bidx, bidy)`` which
        returns an ``make_axis_spec`` namespace.

        Args:
            kernel: Kernel instance providing the block-level primitives.
            io: I/O + geometry bundle (see ``make_kernel_io``).
            build_axis: ``(bidx, bidy) -> axis`` factory for the axis spec.

        Plain (non-``@cute.jit``) helper; its cute ops inline into the trace.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        # Validate group sizes are multiples of 128 if offs is provided.
        if cutlass.const_expr(io.offs is not None):
            if tidx == 0:
                validate_group_sizes(io.offs)

        staging = setup_kernel_smem_and_barriers(
            io.storage, io.smem_layouts, tidx, io.tma, io.shape
        )

        if cutlass.const_expr(io.blocked_scale_output):
            scales_tensor = cute.make_tensor(
                io.scales_out_u8.iterator, io.blocked_scale_layout
            )
        else:
            scales_tensor = io.scales_out_u8

        axis = build_axis(bidx, bidy)
        run_quantize_tile_loop(
            kernel,
            io.tma,
            staging,
            axis,
            (io.opts, warp_idx, tidx, io.compute_warps, scales_tensor),
        )
