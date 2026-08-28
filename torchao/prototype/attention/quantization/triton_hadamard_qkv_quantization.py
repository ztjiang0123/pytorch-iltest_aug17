# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Hadamard + FP8 quantization kernels for Q, K, V.

Input/output format: [B, H, S, D].
Supports GQA (different head counts for Q vs K/V) and cross-attention
(different sequence lengths for Q vs K/V).

The Hadamard transform spreads outliers across the head dimension,
improving per-head FP8 quantization quality.

Phase 1 uses D threads per block (one per head-dim element) to apply
the butterfly-based Hadamard transform.  Phase 2 and reduce kernels
are reused from triton_qkv_quantization.py.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    QuantizeSpec,
    _apply_hadamard,
    _compute_num_chunks,
    _get_log2_d,
)
from torchao.prototype.attention.quantization.triton_qkv_quantization import (
    single_phase1_kernel,
    single_phase2_kernel,
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
def hadamard_single_phase1_kernel(
    # Buffer pointers, in order:
    #   x_ptr           input tensor            [B, H, S, D]
    #   x_had_ptr       Hadamard'd intermediate [B, H, S, D]
    #   temp_ptr        butterfly scratch       [B, H, num_chunks, D] (contiguous)
    #   partial_max_ptr partial max values      [B * H * num_chunks]
    ptrs,
    # Input strides for [B, H, S, D] layout: (stride_b, stride_h, stride_s, stride_d)
    strides,
    # Dimensions: (S, H, chunk_size, num_chunks)
    dims,
    # Head dimension (kept separate so it can key the autotuner)
    D: tl.constexpr,
    # Remaining compile-time constants: (LOG2_D, USE_BFLOAT16)
    META: tl.constexpr,
):
    """
    Phase 1 for a single tensor: Apply Hadamard transform, store to
    intermediate buffer, compute partial absmax.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.

    ``ptrs``, ``strides`` and ``dims`` bundle what would otherwise be long,
    same-typed parameter runs; the temp buffer is contiguous so its per-block
    region is derived from the dimensions instead of passed-in strides.
    """
    x_ptr, x_had_ptr, temp_ptr, partial_max_ptr = ptrs
    stride_b, stride_h, stride_s, stride_d = strides
    S, H, chunk_size, num_chunks = dims
    LOG2_D, USE_BFLOAT16 = META

    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    # temp is contiguous [B, H, num_chunks, D], so each block owns the D-wide
    # slice starting at (linear block index) * D.
    block_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    temp_base = block_idx * D
    s_start = pid_chunk * chunk_size
    base = pid_b * stride_b + pid_h * stride_h

    x_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        offset = base + s_idx * stride_s + d_idx * stride_d
        x = tl.load(x_ptr + offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        x = _apply_hadamard(x, (temp_ptr, temp_base, d_idx), D, LOG2_D)

        # Store to intermediate buffer in input dtype
        if USE_BFLOAT16:
            tl.store(x_had_ptr + offset, x.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(x_had_ptr + offset, x.to(tl.float16), mask=s_mask)

        x_max = tl.maximum(x_max, tl.abs(x))

    x_max_scalar = tl.max(x_max)
    tl.store(partial_max_ptr + block_idx, x_max_scalar)


def _hadamard_quantize_one(x, spec):
    """Quantize a single [B, H, S, D] tensor to FP8 with per-(KV)head scaling.

    When ``spec.apply_hadamard`` is True the Hadamard transform is applied before
    quantization (phase1 emits a Hadamard'd intermediate that phase2 consumes);
    otherwise the tensor is quantized directly. ``spec`` is a
    :class:`QuantizeSpec`. Returns ``(x_fp8, x_descale)`` with descale [B, H_kv].
    """
    B, H, S, D, H_kv = spec.B, spec.H, spec.S, spec.D, spec.H_kv
    groups, num_chunks = spec.groups, spec.num_chunks
    LOG2_D, use_bfloat16 = spec.LOG2_D, spec.use_bfloat16
    apply_hadamard = spec.apply_hadamard
    grid = spec.grid
    chunk_size = spec.chunk_size

    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    partial_max = torch.empty(B * H * num_chunks, dtype=torch.float32, device=x.device)
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)

    if apply_hadamard:
        had = torch.empty_like(x)
        temp = torch.empty(B, H, num_chunks, D, dtype=torch.float32, device=x.device)
        hadamard_single_phase1_kernel[grid](
            (x, had, temp, partial_max),
            (x.stride(0), x.stride(1), x.stride(2), x.stride(3)),
            (S, H, chunk_size, num_chunks),
            D=D,
            META=(LOG2_D, use_bfloat16),
        )
        phase2_input = had
    else:
        single_phase1_kernel[grid](
            x,
            partial_max,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            S,
            D,
            H,
            chunk_size,
            num_chunks,
        )
        phase2_input = x

    single_reduce_kernel[(B, H_kv)](
        partial_max, scale, descale, H_kv, groups * num_chunks
    )
    single_phase2_kernel[grid](
        phase2_input,
        x_fp8,
        scale,
        phase2_input.stride(0),
        phase2_input.stride(1),
        phase2_input.stride(2),
        phase2_input.stride(3),
        S,
        D,
        H,
        chunk_size,
        H_kv,
        groups,
    )
    return x_fp8, descale


def triton_fp8_hadamard_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: Optional[int] = None,
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
    Hadamard + FP8 quantization for Q, K, V tensors.

    Applies Hadamard transform then quantizes to FP8 with per-head scaling.
    Each of Q, K, V is processed with independent kernel launches,
    supporting GQA where Q has more heads than K/V (H_q = groups * H_kv)
    and cross-attention where Q and K/V have different sequence lengths.

    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    The caller must apply inverse Hadamard to the attention output:
        output = inverse_hadamard_transform(attention_output)

    Args:
        q: Query tensor of shape [B, H_q, S_q, D] in bf16/fp16
        k: Key tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        v: Value tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        num_chunks: Number of chunks to split the S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with Hadamard, shape [B, H_q, S_q, D] in fp8
        k_fp8: Quantized key with Hadamard, shape [B, H_kv, S_kv, D] in fp8
        v_fp8: Quantized value with Hadamard, shape [B, H_kv, S_kv, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32

    Note:
        D must be a power of 2 and <= 256 for the Hadamard transform.
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {v.dim()}D"
    assert k.shape == v.shape, (
        f"K and V must have the same shape, got {k.shape} vs {v.shape}"
    )
    assert q.shape[0] == k.shape[0], (
        f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}"
    )
    assert q.shape[3] == k.shape[3], f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}"
    assert q.shape[1] % k.shape[1] == 0, (
        f"Q heads ({q.shape[1]}) must be a multiple of K heads ({k.shape[1]})"
    )

    B, H_q, S_q, D = q.shape
    H_kv = k.shape[1]
    S_kv = k.shape[2]
    groups = H_q // H_kv

    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    use_bfloat16 = q.dtype == torch.bfloat16

    # Compute number of chunks independently for Q and KV
    if num_chunks is None:
        q_num_chunks = _compute_num_chunks(q.device, B, H_q, S_q)
        kv_num_chunks = _compute_num_chunks(k.device, B, H_kv, S_kv)
    else:
        q_num_chunks = num_chunks
        kv_num_chunks = num_chunks

    # Q/K apply the Hadamard transform unless `v_only`; V always applies it.
    # Q uses per-KV-group scaling (groups); K/V are per-head (groups=1).
    q_fp8, q_descale = _hadamard_quantize_one(
        q,
        QuantizeSpec(
            B,
            H_q,
            S_q,
            D,
            H_kv,
            groups,
            q_num_chunks,
            LOG2_D=LOG2_D,
            use_bfloat16=use_bfloat16,
            apply_hadamard=not v_only,
        ),
    )
    k_fp8, k_descale = _hadamard_quantize_one(
        k,
        QuantizeSpec(
            B,
            H_kv,
            S_kv,
            D,
            H_kv,
            1,
            kv_num_chunks,
            LOG2_D=LOG2_D,
            use_bfloat16=use_bfloat16,
            apply_hadamard=not v_only,
        ),
    )
    v_fp8, v_descale = _hadamard_quantize_one(
        v,
        QuantizeSpec(
            B,
            H_kv,
            S_kv,
            D,
            H_kv,
            1,
            kv_num_chunks,
            LOG2_D=LOG2_D,
            use_bfloat16=use_bfloat16,
            apply_hadamard=True,
        ),
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
