"""Triton kernel for 2D (16×16) NVFP4 E2M1 weight quantization."""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.10.0") and has_triton():
    import itertools
    from typing import NamedTuple, Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _compute_pid,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
        prepare_for_cuda_graph,
    )
    from torchao.utils import is_sm_at_least_100

    # Kernel argument bundles. Triton @triton.jit kernels cannot take dataclasses, but
    # they do accept (named)tuples, which Triton flattens into their leaf pointers/ints
    # and exposes as compile-time-indexable fields inside the kernel. Grouping the
    # tensors and shape this way keeps the persistent quantization kernel's parameter
    # list small while every tensor/scalar still travels together to the launch.
    class Quantize2DTensors(NamedTuple):
        """The input and four output tensors of the 2D NVFP4 weight quantization."""

        a: torch.Tensor  # (M, N) bf16 input
        global_amax: torch.Tensor  # scalar f32 global amax
        row_codes: torch.Tensor  # (M, N//2) uint8 rowwise FP4 codes
        row_scales: torch.Tensor  # (M//128, N//64, 32, 16) f8e4m3 rowwise scales
        col_codes: torch.Tensor  # (N, M//2) uint8 colwise FP4 codes
        col_scales: torch.Tensor  # (N//128, M//64, 32, 16) f8e4m3 colwise scales

    class Quantize2DShape(NamedTuple):
        """The logical (M, N) shape of the input tensor A."""

        M: int
        N: int

    # L2-cache-friendly tile grouping along N for the persistent grid-stride loop.
    # Constant across all launches, so it lives here instead of as a kernel parameter.
    # A captured int global is treated as a compile-time constant inside @triton.jit.
    QUANTIZE_2D_GROUP_SIZE_N = 8

    # SM100+ autotune configs.
    QUANTIZE_2D_TILE_SHAPES: list[tuple[int, int]] = [
        (128, 128),
        (128, 256),
        (256, 128),
        (256, 256),
    ]

    QUANTIZE_2D_CONFIGS: list[triton.Config] = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_STAGES": ns},
            num_warps=nw,
            num_stages=ns,
        )
        for (bm, bn), ns, nw in itertools.product(
            QUANTIZE_2D_TILE_SHAPES,
            [1, 2, 3],  # NUM_STAGES
            [2, 4, 8],  # NUM_WARPS
        )
    ]

    @triton.jit
    def _nvfp4_2d_quantize(
        a, global_amax, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        """Compute per-16×16-block FP8 scale factors and scaled FP32 values for FP4 packing.

        Args:
            a: (BLOCK_M, BLOCK_N) bfloat16 tensor.
            global_amax: scalar float32 global amax.

        Returns:
            scale_inv: (BLOCK_M // 16, BLOCK_N // 16) float8e4nv per-block decode scales.
            scaled:    (BLOCK_M, BLOCK_N) float32 values scaled and clamped to FP4 range.
        """
        FP8_E4M3_EPS: tl.constexpr = torch.finfo(torch.float8_e4m3fn).tiny
        FP8_E4M3_MAX: tl.constexpr = 448.0
        FP4_E2M1_MAX: tl.constexpr = 6.0
        FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

        a_tile = tl.reshape(a, [BLOCK_M // 16, 16, BLOCK_N // 16, 16])
        abs_a_tile = tl.abs(a_tile)  # (BLOCK_M//16, 16, BLOCK_N//16, 16)
        tile_max = tl.max(
            abs_a_tile, axis=-1, keep_dims=True
        )  # (BLOCK_M//16, 16, BLOCK_N//16, 1)
        tile_max = tl.max(
            tile_max, axis=-3, keep_dims=True
        )  # (BLOCK_M//16, 1, BLOCK_N//16, 1)

        is_global_amax = global_amax == 0
        safe_global_amax = tl.where(is_global_amax, 1.0, global_amax)
        candidate = tl.minimum(FP8_E4M3_MAX * FP4_E2M1_MAX / safe_global_amax, FP32_MAX)
        candidate = tl.where(candidate == 0, 1.0, candidate)
        global_encode_scale = tl.where(is_global_amax, 1.0, candidate)
        global_decode_scale = 1.0 / global_encode_scale

        pvscale = (tile_max / FP4_E2M1_MAX) * global_encode_scale
        pvscale = tl.clamp(pvscale, FP8_E4M3_EPS, FP8_E4M3_MAX)
        pvscale_fp8 = pvscale.to(tl.float8e4nv)
        scale_inv = tl.reshape(pvscale_fp8, [BLOCK_M // 16, BLOCK_N // 16])

        encode_scale = tl.minimum(
            1.0 / (pvscale_fp8.to(tl.float32) * global_decode_scale), FP32_MAX
        )

        scaled = a_tile * encode_scale
        scaled = tl.clamp(scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_N])
        return scale_inv, scaled

    @triton.autotune(
        configs=QUANTIZE_2D_CONFIGS,
        # Tune per distinct (M, N); the shape tuple is hashable so autotune keys on it
        # exactly as it would on separate M and N arguments.
        key=["shape"],
    )
    @triton.jit
    def triton_quantize_2d_weight(
        tensors: Quantize2DTensors,
        shape: Quantize2DShape,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        """2D (16×16) NVFP4 E2M1 weight quantization — one tile per CTA.

        Args:
            tensors: the input plus four output tensors (see ``Quantize2DTensors``).
            shape: the logical ``(M, N)`` shape of the input (see ``Quantize2DShape``).

        The persistent grid is launched with one program per SM, so the grid-stride
        step is ``tl.num_programs(0)`` and the L2 grouping factor is the module-level
        ``QUANTIZE_2D_GROUP_SIZE_N`` constant — neither needs to be a kernel parameter.
        """
        M, N = shape.M, shape.N

        # Create TMA descriptors in-kernel from raw pointers, shape, and stride
        a_desc = tl.make_tensor_descriptor(
            tensors.a,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )
        out_desc = tl.make_tensor_descriptor(
            tensors.row_codes,
            shape=[M, N // 2],
            strides=[N // 2, 1],
            block_shape=[BLOCK_M, BLOCK_N // 2],
        )
        sf_desc = tl.make_tensor_descriptor(
            tensors.row_scales,
            shape=[M // 128, N // 64, 32, 16],
            strides=[(N // 64) * 32 * 16, 32 * 16, 16, 1],
            block_shape=[BLOCK_M // 128, BLOCK_N // 64, 32, 16],
        )
        a_t_fp4_desc = tl.make_tensor_descriptor(
            tensors.col_codes,
            shape=[N, M // 2],
            strides=[M // 2, 1],
            block_shape=[BLOCK_N, BLOCK_M // 2],
        )
        a_t_sf_desc = tl.make_tensor_descriptor(
            tensors.col_scales,
            shape=[N // 128, M // 64, 32, 16],
            strides=[(M // 64) * 32 * 16, 32 * 16, 16, 1],
            block_shape=[BLOCK_N // 128, BLOCK_M // 64, 32, 16],
        )

        # Persistent grid-stride loop. The grid is launched with one program per SM,
        # so the stride between tiles handled by a program is tl.num_programs(0).
        start_pid = tl.program_id(0)
        num_sms = tl.num_programs(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = QUANTIZE_2D_GROUP_SIZE_N * num_pid_m
        num_tiles = num_pid_m * num_pid_n

        # Load global amax scalar once
        global_amax = tl.load(tensors.global_amax)

        for tile_id in tl.range(
            start_pid,
            num_tiles,
            num_sms,
            flatten=False,
            num_stages=NUM_STAGES,
        ):
            pid_n, pid_m = _compute_pid(
                tile_id, num_pid_in_group, num_pid_n, QUANTIZE_2D_GROUP_SIZE_N
            )

            # Load A (BLOCK_M, BLOCK_N)
            a = a_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])

            # Compute per-16×16-block scales and scaled values
            scale_inv, scaled = _nvfp4_2d_quantize(a, global_amax, BLOCK_M, BLOCK_N)

            # Pack FP4 values into uint8 — non-transposed: (BLOCK_M, BLOCK_N//2, 2)
            scaled_pairs = scaled.reshape(BLOCK_M, BLOCK_N // 2, 2).split()
            scaled_fp4x2 = convert_8xfp32_to_4xfp4_packed(scaled_pairs)
            out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N // 2], scaled_fp4x2)

            # Expand scales: (BLOCK_M//16, BLOCK_N//16) → (BLOCK_M, BLOCK_N//16)
            expand_sf = (
                tl.expand_dims(scale_inv, axis=1)
                .broadcast_to([BLOCK_M // 16, 16, BLOCK_N // 16])
                .reshape(BLOCK_M, BLOCK_N // 16)
            )
            swizzle_expand_sf = _swizzle_scales(expand_sf, BLOCK_M, BLOCK_N)
            sf_desc.store(
                [pid_m * BLOCK_M // 128, pid_n * BLOCK_N // 64, 0, 0], swizzle_expand_sf
            )

            # Colwise path: quantize transposed tile (rowwise W.T) — always swizzled
            a_t = tl.trans(a)  # (BLOCK_N, BLOCK_M)
            t_scale_inv, t_scaled = _nvfp4_2d_quantize(
                a_t, global_amax, BLOCK_N, BLOCK_M
            )

            t_scaled_pairs = t_scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
            t_scaled_fp4x2 = convert_8xfp32_to_4xfp4_packed(t_scaled_pairs)
            a_t_fp4_desc.store([pid_n * BLOCK_N, pid_m * BLOCK_M // 2], t_scaled_fp4x2)

            # Expand and swizzle colwise scales: (BLOCK_N//16, BLOCK_M//16) → (N//128, M//64, 32, 16)
            t_expand_sf = (
                tl.expand_dims(t_scale_inv, axis=1)
                .broadcast_to([BLOCK_N // 16, 16, BLOCK_M // 16])
                .reshape(BLOCK_N, BLOCK_M // 16)
            )
            t_swizzle_sf = _swizzle_scales(t_expand_sf, BLOCK_N, BLOCK_M)
            a_t_sf_desc.store(
                [pid_n * BLOCK_N // 128, pid_m * BLOCK_M // 64, 0, 0], t_swizzle_sf
            )

    @torch.library.custom_op("torchao::triton_weight_quantize_2d", mutates_args=())
    def triton_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D (16×16) NVFP4 E2M1 weight quantization without RHT.

        Args:
            A:           (M, N) bfloat16, row-major. M and N divisible by 16.
            global_amax: scalar float32 global absolute maximum of A. Caller computes
                         ``A.float().abs().max()`` (and optionally all-reduces for TP)
                         before passing in.

        Returns:
            4-tuple of:
              - (M, N//2) uint8: rowwise FP4 codes.
              - (M//128, N//64, 32, 16) float8_e4m3fn: rowwise swizzled scale factors.
              - (N, M//2) uint8: colwise FP4 codes (rowwise W.T).
              - (N//128, M//64, 32, 16) float8_e4m3fn: colwise swizzled scale factors.
        """
        if not is_sm_at_least_100():
            raise NotImplementedError("triton_weight_quantize_2d requires SM100+")
        if A.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16, got {A.dtype}")
        if A.ndim != 2:
            raise ValueError("Tensor A must be 2-D")
        if not A.is_contiguous():
            raise ValueError("A must be row-major (contiguous)")
        M, N = A.shape
        if M % 16 != 0:
            raise ValueError(f"M must be divisible by 16, got M={M}")
        if N % 16 != 0:
            raise ValueError(f"N must be divisible by 16, got N={N}")
        if M % 128 != 0:
            raise ValueError(f"M must be divisible by 128 for swizzling, got M={M}")
        if N % 128 != 0:
            raise ValueError(f"N must be divisible by 128 for swizzling, got N={N}")

        if hasattr(triton, "set_allocator"):
            _ws = prepare_for_cuda_graph(A.device)
            triton.set_allocator(lambda size, align, stream: _ws[: max(size, 1)])

        a_fp4 = torch.zeros((M, N // 2), dtype=torch.uint8, device=A.device)
        a_sf = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        a_t_fp4 = torch.zeros((N, M // 2), dtype=torch.uint8, device=A.device)
        a_t_sf = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

        try:
            # Launch one program per SM; the kernel reads the SM count back via
            # tl.num_programs(0) and uses the module-level QUANTIZE_2D_GROUP_SIZE_N,
            # so neither is passed as a kernel argument. Tensors and shape are grouped
            # into namedtuples that Triton flattens back into the individual pointers
            # and ints at launch.
            triton_quantize_2d_weight[(NUM_SMS,)](
                Quantize2DTensors(
                    a=A,
                    global_amax=global_amax,
                    row_codes=a_fp4,
                    row_scales=a_sf,
                    col_codes=a_t_fp4,
                    col_scales=a_t_sf,
                ),
                Quantize2DShape(M=M, N=N),
            )
        finally:
            if hasattr(triton, "set_allocator"):
                triton.set_allocator(None)
        return a_fp4, a_sf, a_t_fp4, a_t_sf

    @triton_weight_quantize_2d.register_fake
    def _(A, global_amax):
        M, N = A.shape
        codes = A.new_empty((M, N // 2), dtype=torch.uint8)
        sf = A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
        t_codes = A.new_empty((N, M // 2), dtype=torch.uint8)
        t_sf = A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
        return codes, sf, t_codes, t_sf

else:

    def triton_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_weight_quantize_2d requires torch 2.10.0+ and triton installed"
        )
