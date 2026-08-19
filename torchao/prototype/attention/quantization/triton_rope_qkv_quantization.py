# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + FP8 quantization kernels for Q, K, V.

Input: [B, S, H, D], output: [B, H, S, D].
Supports GQA (different head counts for Q vs K/V).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    _compute_num_chunks,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["chunk_size", "D_HALF"],
)
@triton.jit
def rope_single_phase1_kernel(
    # Input tensor [B, S, H, D]
    x_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Intermediate output tensor [B, H, S, D] - stores RoPE'd result
    x_rope_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_in_b,
    stride_in_s,
    stride_in_h,
    stride_in_d,
    # Output strides (for [B, H, S, D] layout)
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    # Dimensions
    S,
    D,
    D_HALF,
    H,
    chunk_size,
    num_chunks,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    ROPE_INTERLEAVED: tl.constexpr,
):
    """
    Phase 1 for a single tensor (Q or K): Apply RoPE, store to intermediate,
    compute partial max.

    Grid: (B, H, num_chunks)

    Supports two RoPE pairing variants (selected by ROPE_INTERLEAVED constexpr):
    - NeoX half-split (ROPE_INTERLEAVED=False): pairs (j, j+D/2) for j in [0, D/2)
    - Interleaved (ROPE_INTERLEAVED=True): pairs (2i, 2i+1) for i in [0, D/2)

    Each pair shares the same rotation angle and the 2D rotation formula is:
      out[first]  = x[first]*cos - x[second]*sin
      out[second] = x[second]*cos + x[first]*sin
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    actual_chunk_size = s_end - s_start

    # Number of pairs to process in this chunk
    chunk_pairs = actual_chunk_size * D_HALF

    # Base pointers for input [B, S, H, D]
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h

    # Base pointers for output [B, H, S, D]
    out_base_b = pid_b * stride_out_b
    out_base_h = pid_h * stride_out_h

    # Initialize max accumulator
    x_max = 0.0

    # Linearized iteration over chunk_size * D_HALF pairs
    for block_start in range(0, chunk_pairs, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_pairs

        # Convert linear offset to (s, pair_idx) coordinates
        local_s = offs // D_HALF
        pair_idx = offs % D_HALF
        s_idx = s_start + local_s

        # Compute element indices for this pair based on RoPE variant
        if ROPE_INTERLEAVED:
            # FLUX/GPT-J interleaved: pair (2i, 2i+1)
            d_first = pair_idx * 2
            d_second = pair_idx * 2 + 1
        else:
            # NeoX/LLaMA half-split: pair (j, j+D/2)
            d_first = pair_idx
            d_second = pair_idx + D_HALF

        # Input offsets [B, S, H, D]
        in_offset_first = (
            in_base_b + s_idx * stride_in_s + in_base_h + d_first * stride_in_d
        )
        in_offset_second = (
            in_base_b + s_idx * stride_in_s + in_base_h + d_second * stride_in_d
        )

        # Output offsets [B, H, S, D]
        out_offset_first = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_first * stride_out_d
        )
        out_offset_second = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_second * stride_out_d
        )

        # Load input pairs (each element loaded exactly once)
        x_first = tl.load(x_ptr + in_offset_first, mask=mask, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + in_offset_second, mask=mask, other=0.0).to(
            tl.float32
        )

        # Load cos/sin — both elements in a NeoX pair share the same
        # rotation angle.  cos[j] == cos[j+D/2] in LLaMA's frequency
        # layout, so we only need one load per pair.
        cos_offset = s_idx * D + d_first
        cos_val = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        # Apply NeoX RoPE rotation:
        #   out[j]     = x[j]     * cos - x[j+D/2] * sin
        #   out[j+D/2] = x[j+D/2] * cos + x[j]     * sin
        x_rope_first = tl.math.fma(x_first, cos_val, -(x_second * sin_val))
        x_rope_second = tl.math.fma(x_second, cos_val, x_first * sin_val)

        # Store RoPE'd result to intermediate buffer [B, H, S, D]
        tl.store(
            x_rope_ptr + out_offset_first, x_rope_first.to(x_first.dtype), mask=mask
        )
        tl.store(
            x_rope_ptr + out_offset_second, x_rope_second.to(x_first.dtype), mask=mask
        )

        # Update max values (from RoPE'd values)
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_rope_first)))
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_rope_second)))

    # Store partial max
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def v_phase1_kernel(
    # Input tensor [B, S, H, D]
    v_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_in_b,
    stride_in_s,
    stride_in_h,
    stride_in_d,
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    num_chunks,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1 for V: Compute partial absmax (no RoPE applied).

    Grid: (B, H, num_chunks)

    Uses linearized iteration over chunk_size * D elements.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointers for input [B, S, H, D]
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h

    # Initialize max accumulator
    v_max = 0.0

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Input offset [B, S, H, D]
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d

        v_val = tl.load(v_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)
        v_max = tl.maximum(v_max, tl.max(tl.abs(v_val)))

    # Store partial max
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, v_max)


@triton.jit
def single_reduce_kernel(
    partial_max_ptr,  # [B * H * num_chunks]
    scale_ptr,
    descale_ptr,
    H,
    num_chunks,
):
    """
    Reduce partial maxes and compute scale/descale for a single tensor.

    Grid: (B, H)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Reduce across chunks for this (batch, head)
    x_max = 0.0

    base_idx = (pid_b * H + pid_h) * num_chunks
    for c in range(num_chunks):
        x_max = tl.maximum(x_max, tl.load(partial_max_ptr + base_idx + c))

    # Compute scale and descale
    # FP8 E4M3 max value is 448.0
    FP8_MAX = 448.0
    eps = 1e-12
    scale_idx = pid_b * H + pid_h

    tl.store(scale_ptr + scale_idx, tl.where(x_max > eps, FP8_MAX / x_max, 1.0))
    tl.store(descale_ptr + scale_idx, tl.where(x_max > eps, x_max / FP8_MAX, 1.0))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def rope_single_phase2_kernel(
    # Intermediate tensor [B, H, S, D] - already RoPE'd
    x_rope_ptr,
    # Output tensor [B, H, S, D] - FP8 quantized
    x_out_ptr,
    # Precomputed scale [B, H_scale]
    scale_ptr,
    # Strides (for [B, H, S, D] layout) - same for intermediate and output
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    # Scale indexing for GQA: scale has H_scale entries per batch,
    # and each group of `groups` heads shares one scale.
    # For non-GQA: H_scale = H, groups = 1.
    H_scale,
    groups,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2 for a single tensor (Q or K): Quantize pre-computed RoPE'd values to FP8.

    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Load scale for this head (or head group for GQA)
    scale = tl.load(scale_ptr + pid_b * H_scale + pid_h // groups)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointer
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        ptr_offset = base_offset + s_idx * stride_s + d_idx * stride_d

        # Load pre-computed RoPE'd value from intermediate buffer
        x_val = tl.load(x_rope_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        x_fp8 = (x_val * scale).to(tl.float8e4nv)

        # Store to output
        tl.store(x_out_ptr + ptr_offset, x_fp8, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def v_phase2_kernel(
    # Original V tensor [B, S, H, D] - needs transpose only
    v_ptr,
    # Output tensor [B, H, S, D] - FP8 quantized
    v_out_ptr,
    # Precomputed scale [B, H]
    scale_ptr,
    # V input strides (for [B, S, H, D] layout)
    stride_v_in_b,
    stride_v_in_s,
    stride_v_in_h,
    stride_v_in_d,
    # Output strides (for [B, H, S, D] layout)
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2 for V: Transpose from [B,S,H,D] to [B,H,S,D] and quantize to FP8.

    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Load scale for this head
    scale = tl.load(scale_ptr + pid_b * H + pid_h)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointers
    v_in_base = pid_b * stride_v_in_b + pid_h * stride_v_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # V input offset [B, S, H, D]
        v_in_offset = v_in_base + s_idx * stride_v_in_s + d_idx * stride_v_in_d

        # Output offset [B, H, S, D]
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d

        # Load V from original input (no RoPE, just transpose)
        v_val = tl.load(v_ptr + v_in_offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        v_fp8 = (v_val * scale).to(tl.float8e4nv)

        # Store to output
        tl.store(v_out_ptr + out_offset, v_fp8, mask=mask)


def _rope_quantize_one(
    x, cos, sin, B, H, S, D, D_HALF, H_kv, groups, num_chunks, rope_interleaved
):
    """Apply RoPE to a single [B, S, H, D] tensor and quantize it to FP8, emitting
    the result in [B, H, S, D] layout. ``groups`` is ``H // H_kv`` for Q and 1 for
    K. Returns ``(x_fp8, x_descale)`` with ``x_descale`` of shape [B, H_kv]."""
    chunk_size = (S + num_chunks - 1) // num_chunks
    grid = (B, H, num_chunks)

    x_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=x.device)
    rope_intermediate = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    partial_max = torch.empty(B * H * num_chunks, dtype=torch.float32, device=x.device)
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)

    rope_single_phase1_kernel[grid](
        x,
        cos,
        sin,
        rope_intermediate,
        partial_max,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        rope_intermediate.stride(0),
        rope_intermediate.stride(1),
        rope_intermediate.stride(2),
        rope_intermediate.stride(3),
        S,
        D,
        D_HALF,
        H,
        chunk_size,
        num_chunks,
        ROPE_INTERLEAVED=rope_interleaved,
    )
    single_reduce_kernel[(B, H_kv)](
        partial_max, scale, descale, H_kv, groups * num_chunks
    )
    rope_single_phase2_kernel[grid](
        rope_intermediate,
        x_fp8,
        scale,
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_fp8.stride(2),
        x_fp8.stride(3),
        S,
        D,
        H,
        chunk_size,
        H_kv,
        groups,
    )
    return x_fp8, descale


def _quantize_v_transpose(v, B, H_kv, S, D, num_chunks):
    """Quantize V (no RoPE) with a [B, S, H, D] -> [B, H, S, D] transpose. Returns
    ``(v_fp8, v_descale)`` with ``v_descale`` of shape [B, H_kv]."""
    chunk_size = (S + num_chunks - 1) // num_chunks
    grid = (B, H_kv, num_chunks)

    v_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=v.device)
    partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=v.device
    )
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=v.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=v.device)

    v_phase1_kernel[grid](
        v,
        partial_max,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        num_chunks,
    )
    single_reduce_kernel[(B, H_kv)](partial_max, scale, descale, H_kv, num_chunks)
    v_phase2_kernel[grid](
        v,
        v_fp8,
        scale,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
    )
    return v_fp8, descale


def triton_fp8_rope_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,
    rope_interleaved: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Separated RoPE + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then quantizes all tensors to FP8 with per-head scaling.
    Also performs layout transformation from [B, S, H, D] to [B, H, S, D].
    Each of Q, K, V is processed with independent kernel launches.

    Supports GQA where Q has more heads than K/V (H_q = groups * H_kv).
    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    Args:
        q: Query tensor of shape [B, S, H_q, D] in bf16/fp16
        k: Key tensor of shape [B, S, H_kv, D] in bf16/fp16
        v: Value tensor of shape [B, S, H_kv, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with RoPE, shape [B, H_q, S, D] in fp8
        k_fp8: Quantized key with RoPE, shape [B, H_kv, S, D] in fp8
        v_fp8: Quantized value, shape [B, H_kv, S, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {v.dim()}D"
    assert k.shape == v.shape, (
        f"K and V must have the same shape, got {k.shape} vs {v.shape}"
    )
    assert q.shape[0] == k.shape[0], (
        f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}"
    )
    assert q.shape[1] == k.shape[1], (
        f"Sequence length mismatch: {q.shape[1]} vs {k.shape[1]}"
    )
    assert q.shape[3] == k.shape[3], f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}"
    assert q.shape[2] % k.shape[2] == 0, (
        f"Q heads ({q.shape[2]}) must be a multiple of K heads ({k.shape[2]})"
    )
    assert cos.dim() == 2, f"Expected 2D cos tensor [S, D], got {cos.dim()}D"
    assert sin.dim() == 2, f"Expected 2D sin tensor [S, D], got {sin.dim()}D"

    B, S, H_q, D = q.shape
    H_kv = k.shape[2]
    groups = H_q // H_kv

    assert D % 2 == 0, f"Head dimension D must be even for RoPE, got D={D}"
    assert cos.shape == (S, D), f"Expected cos shape [{S}, {D}], got {cos.shape}"
    assert sin.shape == (S, D), f"Expected sin shape [{S}, {D}], got {sin.shape}"

    D_HALF = D // 2

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q.device, B, H_q, S)

    # Q/K: RoPE + quantize (Q uses per-KV-group scaling, K is per-head).
    q_fp8, q_descale = _rope_quantize_one(
        q, cos, sin, B, H_q, S, D, D_HALF, H_kv, groups, num_chunks, rope_interleaved
    )
    k_fp8, k_descale = _rope_quantize_one(
        k, cos, sin, B, H_kv, S, D, D_HALF, H_kv, 1, num_chunks, rope_interleaved
    )
    # V: no RoPE, transpose + quantize (per-head).
    v_fp8, v_descale = _quantize_v_transpose(v, B, H_kv, S, D, num_chunks)

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
