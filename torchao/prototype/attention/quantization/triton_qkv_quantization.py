# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 quantization kernels for Q, K, V.

Input/output format: [B, H, S, D].
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
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def single_phase1_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, H, S, D] layout)
    stride_b,
    stride_h,
    stride_s,
    stride_d,
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
    Phase 1 for a single tensor: Compute partial absmax.

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

    # Base pointer for input [B, H, S, D]
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Initialize max accumulator
    x_max = 0.0

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Input offset [B, H, S, D]
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        x_val = tl.load(x_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_val)))

    # Store partial max
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max)


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
def single_phase2_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Output tensor [B, H, S, D] - FP8 quantized
    x_out_ptr,
    # Precomputed scale [B, H_scale]
    scale_ptr,
    # Strides (for [B, H, S, D] layout)
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
    Phase 2 for a single tensor: Quantize to FP8 using precomputed scale.

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

        # Load input value
        x_val = tl.load(x_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        x_fp8 = (x_val * scale).to(tl.float8e4nv)

        # Store to output
        tl.store(x_out_ptr + ptr_offset, x_fp8, mask=mask)


def _quantize_one_tensor(x, B, H, S, D, H_kv, groups, num_chunks):
    """Quantize a single [B, H, S, D] tensor to FP8 with block-wise per-(KV)head
    scaling via the phase1(max) -> reduce -> phase2(quantize) kernel pipeline.

    ``groups`` is ``H // H_kv`` for the Q tensor (per-KV-group scaling) and 1 for
    K/V (per-head scaling). Returns ``(x_fp8, x_descale)`` where ``x_descale`` has
    shape ``[B, H_kv]``.
    """
    chunk_size = (S + num_chunks - 1) // num_chunks
    grid = (B, H, num_chunks)

    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    partial_max = torch.empty(B * H * num_chunks, dtype=torch.float32, device=x.device)
    scale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)
    descale = torch.empty(B, H_kv, dtype=torch.float32, device=x.device)

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
    # A group reduce over the [B, H, num_chunks] buffer is a single reduce over
    # `groups * num_chunks` contiguous entries per (batch, kv_head), since the
    # `groups` heads mapping to a KV head are contiguous.
    single_reduce_kernel[(B, H_kv)](
        partial_max, scale, descale, H_kv, groups * num_chunks
    )
    single_phase2_kernel[grid](
        x,
        x_fp8,
        scale,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        S,
        D,
        H,
        chunk_size,
        H_kv,
        groups,
    )
    return x_fp8, descale


def triton_fp8_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Separated FP8 quantization for Q, K, V tensors.

    Quantizes all tensors to FP8 with per-head scaling.
    Each of Q, K, V is processed with independent kernel launches,
    supporting GQA where Q has more heads than K/V (H_q = groups * H_kv)
    and cross-attention where Q and K/V have different sequence lengths.

    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    Args:
        q: Query tensor of shape [B, H_q, S_q, D] in bf16/fp16
        k: Key tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        v: Value tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        num_chunks: Number of chunks to split the S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query, shape [B, H_q, S_q, D] in fp8
        k_fp8: Quantized key, shape [B, H_kv, S_kv, D] in fp8
        v_fp8: Quantized value, shape [B, H_kv, S_kv, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32
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

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Compute number of chunks independently for Q and KV
    if num_chunks is None:
        q_num_chunks = _compute_num_chunks(q.device, B, H_q, S_q)
        kv_num_chunks = _compute_num_chunks(k.device, B, H_kv, S_kv)
    else:
        q_num_chunks = num_chunks
        kv_num_chunks = num_chunks

    # Q uses per-KV-group scaling (groups Q heads share a scale); K/V are per-head.
    q_fp8, q_descale = _quantize_one_tensor(
        q, B, H_q, S_q, D, H_kv, groups, q_num_chunks
    )
    k_fp8, k_descale = _quantize_one_tensor(k, B, H_kv, S_kv, D, H_kv, 1, kv_num_chunks)
    v_fp8, v_descale = _quantize_one_tensor(v, B, H_kv, S_kv, D, H_kv, 1, kv_num_chunks)

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
