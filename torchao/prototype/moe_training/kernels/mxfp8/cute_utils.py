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
    def issue_tma_load(
        tma_atom_in: cute.CopyAtom,
        gIN_tile: cute.Tensor,
        sIN_tile: cute.Tensor,
        tma_mbar_ptr: cutlass.Int64,
        warp_idx: cutlass.Int32,
        tile_copy_bytes: cutlass.Constexpr[int],
        group_modes_end: cutlass.Constexpr[int],
    ):
        """Issue a TMA load from global to shared memory (producer warp only).

        Shared across the 2D and 3D CuTeDSL quantization kernels. The only
        per-kernel difference is ``group_modes_end`` (1 for 2D tiles, 2 for the
        leading singleton mode of 3D tiles) and the tile byte count.

        Args:
            tma_atom_in: TMA copy atom for G2S
            gIN_tile: Input tile in global memory
            sIN_tile: Input tile in shared memory
            tma_mbar_ptr: TMA barrier pointer
            warp_idx: Warp index (only warp 0 issues the copy)
            tile_copy_bytes: Number of bytes copied by the TMA transaction
            group_modes_end: End mode index passed to ``cute.group_modes``
        """
        if warp_idx == 0:
            cta_layout = cute.make_layout((1,))
            sIN_for_tma_partition = cute.group_modes(sIN_tile, 0, group_modes_end)
            gIN_for_tma_partition = cute.group_modes(gIN_tile, 0, group_modes_end)
            tINs, tINg = cpasync.tma_partition(
                tma_atom_in,
                0,
                cta_layout,
                sIN_for_tma_partition,
                gIN_for_tma_partition,
            )
            tINg_stage0 = tINg[(None, 0)]
            tINs_stage0 = tINs[(None, 0)]
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
        gOUT_tile: cute.Tensor,
        sOUT_tile: cute.Tensor,
        warp_idx: cutlass.Int32,
        group_modes_end: cutlass.Constexpr[int],
    ):
        """Issue a TMA store from shared to global memory (producer warp only).

        Shared across the 2D and 3D CuTeDSL quantization kernels. Synchronizes
        threads before the store; the only per-kernel difference is
        ``group_modes_end`` (1 for 2D tiles, 2 for the leading singleton mode of
        3D tiles).

        Args:
            tma_atom_out: TMA copy atom for S2G
            gOUT_tile: Output tile in global memory
            sOUT_tile: Output tile in shared memory
            warp_idx: Warp index (only warp 0 issues the copy)
            group_modes_end: End mode index passed to ``cute.group_modes``
        """
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        cute.arch.sync_threads()
        if warp_idx == 0:
            cta_layout = cute.make_layout((1,))
            sOUT_for_tma_partition = cute.group_modes(sOUT_tile, 0, group_modes_end)
            gOUT_for_tma_partition = cute.group_modes(gOUT_tile, 0, group_modes_end)
            tOUTs, tOUTg = cpasync.tma_partition(
                tma_atom_out,
                0,
                cta_layout,
                sOUT_for_tma_partition,
                gOUT_for_tma_partition,
            )
            tOUTs_stage0 = tOUTs[(None, 0)]
            tOUTg_stage0 = tOUTg[(None, 0)]
            cute.copy(
                tma_atom_out,
                tOUTs_stage0,
                tOUTg_stage0,
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
