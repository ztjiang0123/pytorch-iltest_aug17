# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + Hadamard + FP8 quantization kernels for Q, K, V.

Input: [B, S, H, D], output: [B, H, S, D].
Supports GQA (different head counts for Q vs K/V).

Q and K receive RoPE + Hadamard; V receives Hadamard only (no RoPE).
Supports both NeoX/LLaMA half-split and FLUX/GPT-J interleaved RoPE.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    _apply_hadamard,
    _compute_num_chunks,
    _get_log2_d,
)
from torchao.prototype.attention.quantization.triton_rope_qkv_quantization import (
    rope_single_phase1_kernel,
    rope_single_phase2_kernel,
    single_reduce_kernel,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def hadamard_rope_single_phase1_kernel(
    # Input tensor [B, S, H, D]
    x_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Intermediate output tensor [B, H, S, D]
    x_out_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D]
    temp_ptr,
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
    # Temp buffer strides [B, H, num_chunks, D]
    stride_temp_b,
    stride_temp_h,
    stride_temp_c,
    stride_temp_d,
    # Dimensions
    S,
    H,
    D_HALF,
    chunk_size,
    num_chunks,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
    USE_BFLOAT16: tl.constexpr,
    ROPE_INTERLEAVED: tl.constexpr,
):
    """
    Phase 1 for Q or K: Apply RoPE + Hadamard, store to intermediate,
    compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.

    Supports NeoX half-split (pair j, j+D/2) and interleaved (pair 2i, 2i+1).
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    # RoPE partner index and sign
    if ROPE_INTERLEAVED:
        # FLUX/GPT-J interleaved: pair (2i, 2i+1)
        partner_d = d_idx ^ 1
        is_first = (d_idx % 2) == 0
    else:
        # NeoX/LLaMA half-split: pair (j, j+D/2)
        partner_d = d_idx ^ D_HALF
        is_first = d_idx < D_HALF
    # first element: out = x*cos - partner*sin  (sign = -1)
    # second element: out = x*cos + partner*sin (sign = +1)
    sign = tl.where(is_first, -1.0, 1.0)

    x_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        # Load x and its RoPE partner from input [B, S, H, D]
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        partner_in_offset = (
            in_base_b + s_idx * stride_in_s + in_base_h + partner_d * stride_in_d
        )

        x_val = tl.load(x_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        x_partner = tl.load(x_ptr + partner_in_offset, mask=s_mask, other=0.0).to(
            tl.float32
        )

        # Load cos/sin [S, D] — both elements of a pair share the same
        # rotation angle (values are duplicated in the cos/sin tensors).
        cos_offset = s_idx * D + d_idx
        cos_val = tl.load(cos_ptr + cos_offset, mask=s_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + cos_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply RoPE rotation
        x_rope = tl.math.fma(x_val, cos_val, sign * x_partner * sin_val)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        x_rope = _apply_hadamard(x_rope, temp_ptr, temp_base, d_idx, D, LOG2_D)

        # Store to intermediate buffer [B, H, S, D]
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d
        if USE_BFLOAT16:
            tl.store(x_out_ptr + out_offset, x_rope.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(x_out_ptr + out_offset, x_rope.to(tl.float16), mask=s_mask)

        x_max = tl.maximum(x_max, tl.abs(x_rope))

    x_max_scalar = tl.max(x_max)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max_scalar)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def hadamard_v_phase1_kernel(
    # Input tensor [B, S, H, D]
    v_ptr,
    # Intermediate output tensor [B, H, S, D] - Hadamard'd and transposed
    v_out_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D]
    temp_ptr,
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
    # Temp buffer strides [B, H, num_chunks, D]
    stride_temp_b,
    stride_temp_h,
    stride_temp_c,
    stride_temp_d,
    # Dimensions
    S,
    H,
    chunk_size,
    num_chunks,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
    USE_BFLOAT16: tl.constexpr,
):
    """
    Phase 1 for V: Apply Hadamard (no RoPE), transpose [B,S,H,D] -> [B,H,S,D],
    compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    v_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        # Load V from input [B, S, H, D]
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        v_val = tl.load(v_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        v_val = _apply_hadamard(v_val, temp_ptr, temp_base, d_idx, D, LOG2_D)

        # Store to intermediate buffer [B, H, S, D] (transposed)
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d
        if USE_BFLOAT16:
            tl.store(v_out_ptr + out_offset, v_val.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(v_out_ptr + out_offset, v_val.to(tl.float16), mask=s_mask)

        v_max = tl.maximum(v_max, tl.abs(v_val))

    v_max_scalar = tl.max(v_max)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, v_max_scalar)


def _rope_hadamard_quantize_one(
    x,
    cos,
    sin,
    B,
    H,
    S,
    D,
    D_HALF,
    H_kv,
    groups,
    num_chunks,
    LOG2_D,
    use_bfloat16,
    rope_interleaved,
    apply_hadamard,
):
    """RoPE (+ optional Hadamard) a single [B, S, H, D] tensor and quantize it to
    FP8, emitting [B, H, S, D]. When ``apply_hadamard`` is False only RoPE is
    applied (the `v_only` path). ``groups`` is ``H // H_kv`` for Q, 1 for K.
    Returns ``(x_fp8, x_descale)`` with descale of shape [B, H_kv]."""
    chunk_size = (S + num_chunks - 1) // num_chunks
    grid = (B, H, num_chunks)

    x_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=x.device)
    intermediate = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    partial_max = torch.empty(B * H * num_chunks, dtype=torch.float32, device=x.device)
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)

    if apply_hadamard:
        temp = torch.empty(B, H, num_chunks, D, dtype=torch.float32, device=x.device)
        hadamard_rope_single_phase1_kernel[grid](
            x,
            cos,
            sin,
            intermediate,
            temp,
            partial_max,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            intermediate.stride(0),
            intermediate.stride(1),
            intermediate.stride(2),
            intermediate.stride(3),
            temp.stride(0),
            temp.stride(1),
            temp.stride(2),
            temp.stride(3),
            S,
            H,
            D_HALF,
            chunk_size,
            num_chunks,
            D=D,
            LOG2_D=LOG2_D,
            USE_BFLOAT16=use_bfloat16,
            ROPE_INTERLEAVED=rope_interleaved,
        )
    else:
        rope_single_phase1_kernel[grid](
            x,
            cos,
            sin,
            intermediate,
            partial_max,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            intermediate.stride(0),
            intermediate.stride(1),
            intermediate.stride(2),
            intermediate.stride(3),
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
        intermediate,
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


def _hadamard_v_quantize(v, B, H_kv, S, D, num_chunks, LOG2_D, use_bfloat16):
    """Hadamard-transform V (no RoPE) with a [B, S, H, D] -> [B, H, S, D]
    transpose and quantize to FP8. Returns ``(v_fp8, v_descale)``."""
    chunk_size = (S + num_chunks - 1) // num_chunks
    grid = (B, H_kv, num_chunks)

    v_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=v.device)
    intermediate = torch.empty(B, H_kv, S, D, dtype=v.dtype, device=v.device)
    temp = torch.empty(B, H_kv, num_chunks, D, dtype=torch.float32, device=v.device)
    partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=v.device
    )
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=v.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=v.device)

    hadamard_v_phase1_kernel[grid](
        v,
        intermediate,
        temp,
        partial_max,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        intermediate.stride(3),
        temp.stride(0),
        temp.stride(1),
        temp.stride(2),
        temp.stride(3),
        S,
        H_kv,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )
    single_reduce_kernel[(B, H_kv)](partial_max, scale, descale, H_kv, num_chunks)
    rope_single_phase2_kernel[grid](
        intermediate,
        v_fp8,
        scale,
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        H_kv,
        1,
    )
    return v_fp8, descale


def triton_fp8_hadamard_rope_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,
    rope_interleaved: bool = False,
    v_only: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fused RoPE + Hadamard + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then Hadamard transform to Q, K, and V,
    then quantizes all tensors to FP8 with per-head scaling. Also performs
    layout transformation from [B, S, H, D] to [B, H, S, D].
    Each of Q, K, V is processed with independent kernel launches.

    Supports GQA where Q has more heads than K/V (H_q = groups * H_kv).
    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    The caller must apply inverse Hadamard to the attention output:
        output = inverse_hadamard_transform(attention_output)

    Args:
        q: Query tensor of shape [B, S, H_q, D] in bf16/fp16
        k: Key tensor of shape [B, S, H_kv, D] in bf16/fp16
        v: Value tensor of shape [B, S, H_kv, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.
        rope_interleaved: If True, use FLUX/GPT-J interleaved RoPE pairing
                          (2i, 2i+1). If False, use NeoX/LLaMA half-split
                          pairing (j, j+D/2).

    Returns:
        q_fp8: Quantized query with RoPE+Hadamard, shape [B, H_q, S, D] in fp8
        k_fp8: Quantized key with RoPE+Hadamard, shape [B, H_kv, S, D] in fp8
        v_fp8: Quantized value with Hadamard, shape [B, H_kv, S, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32

    Note:
        D must be a power of 2 and <= 256 for the Hadamard transform.
        Q, K, V must have the same sequence length (RoPE requires matching positions).
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

    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"

    D_HALF = D // 2

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    use_bfloat16 = q.dtype == torch.bfloat16

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q.device, B, H_q, S)

    # Q/K: RoPE (+ Hadamard unless v_only) + quantize. Q uses per-KV-group scaling.
    q_fp8, q_descale = _rope_hadamard_quantize_one(
        q,
        cos,
        sin,
        B,
        H_q,
        S,
        D,
        D_HALF,
        H_kv,
        groups,
        num_chunks,
        LOG2_D,
        use_bfloat16,
        rope_interleaved,
        apply_hadamard=not v_only,
    )
    k_fp8, k_descale = _rope_hadamard_quantize_one(
        k,
        cos,
        sin,
        B,
        H_kv,
        S,
        D,
        D_HALF,
        H_kv,
        1,
        num_chunks,
        LOG2_D,
        use_bfloat16,
        rope_interleaved,
        apply_hadamard=not v_only,
    )
    # V: always Hadamard + transpose, no RoPE.
    v_fp8, v_descale = _hadamard_v_quantize(
        v, B, H_kv, S, D, num_chunks, LOG2_D, use_bfloat16
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
