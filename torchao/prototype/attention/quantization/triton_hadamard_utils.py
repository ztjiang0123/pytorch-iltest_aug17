# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared Hadamard transform utilities for FP8 quantization kernels.

Provides the Hadamard butterfly helper used by both the RoPE-fused and
plain Hadamard quantization kernels, plus the inverse Hadamard transform
applied to attention output.

The Hadamard transform H/sqrt(D) is orthogonal and self-inverse:
    H/sqrt(D) @ H/sqrt(D) = I
so the same butterfly + normalization is used for both forward and inverse.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass
class RopeQkvInputs:
    """The Q/K/V tensors plus their shared RoPE cos/sin frequency tables.

    These five tensors always travel together into the fused RoPE + Hadamard +
    FP8 quantize entry point, so bundling them into one value type keeps that
    entry point's signature short and prevents positional call-site mistakes
    (e.g. swapping ``cos``/``sin`` or ``k``/``v``, which share dtype and shape).
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass
class QuantizeSpec:
    """Per-tensor launch parameters for the FP8 quantize kernel pipeline.

    Bundles the scalars shared by the phase1/reduce/phase2 kernel launches so the
    per-tensor helper functions take a single spec argument instead of a long
    parameter list. One instance is built per Q/K/V tensor at the call site.

    Fields used by every helper: ``B``, ``H`` (this tensor's head count), ``S``,
    ``D``, ``H_kv``, ``groups`` (``H // H_kv`` for Q, 1 for K/V), ``num_chunks``.
    RoPE/Hadamard helpers additionally use ``D_HALF``, ``LOG2_D``,
    ``use_bfloat16``, ``rope_interleaved`` and ``apply_hadamard``.
    """

    B: int
    H: int
    S: int
    D: int
    H_kv: int
    groups: int
    num_chunks: int
    D_HALF: Optional[int] = None
    LOG2_D: Optional[int] = None
    use_bfloat16: bool = False
    rope_interleaved: bool = False
    apply_hadamard: bool = True

    @property
    def chunk_size(self) -> int:
        return (self.S + self.num_chunks - 1) // self.num_chunks

    @property
    def grid(self):
        return (self.B, self.H, self.num_chunks)


def _get_log2_d(D: int) -> int:
    """Get log2(D), asserting D is a power of 2."""
    assert D > 0 and (D & (D - 1)) == 0, f"D must be a power of 2, got {D}"
    log2_d = 0
    temp = D
    while temp > 1:
        temp >>= 1
        log2_d += 1
    return log2_d


def _compute_num_chunks(device: torch.device, B: int, H: int, S: int) -> int:
    """Compute optimal number of chunks for parallelizing over the S dimension.

    Layout-agnostic: callers extract B and H from whichever tensor layout
    they use ([B, H, S, D] or [B, S, H, D]) and pass the scalars directly.
    """
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    base_parallelism = B * H
    target_blocks = num_sms * 4
    num_chunks = max(1, target_blocks // base_parallelism)
    num_chunks = min(num_chunks, S // 32) if S >= 32 else 1
    num_chunks = min(num_chunks, 64)
    num_chunks = min(num_chunks, S)
    return num_chunks


@triton.jit
def _hadamard_butterfly_stage(
    x,
    scratch,
    stage: tl.constexpr,
    D: tl.constexpr,
):
    """One stage of the Hadamard butterfly transform.

    Uses global memory temp buffer as shuffle buffer with barriers.
    Each thread stores its value, barrier, loads its partner's value,
    barrier, then computes the butterfly sum/difference.

    Args:
        x: Current D-element vector (vectorized across threads)
        scratch: Butterfly shuffle context ``(temp_ptr, temp_base, d_idx)`` where
            ``temp_ptr`` points to the temp buffer base, ``temp_base`` is the
            offset to this block's region and ``d_idx`` is the vectorized index
            tensor (``tl.arange(0, D)``). These three always travel together.
        stage: Butterfly stage (0 to log2(D)-1), must be constexpr
        D: Head dimension (compile-time constant)
    """
    temp_ptr, temp_base, d_idx = scratch
    stride = 1 << stage
    partner_d = d_idx ^ stride
    is_left = (d_idx & stride) == 0

    tl.store(temp_ptr + temp_base + d_idx, x)
    tl.debug_barrier()
    x_partner = tl.load(temp_ptr + temp_base + partner_d)
    tl.debug_barrier()

    return tl.where(is_left, x + x_partner, x_partner - x)


@triton.jit
def _apply_hadamard(
    x,
    scratch,
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """Apply full Hadamard butterfly transform with 1/sqrt(D) normalization.

    Uses tl.static_range so each stage index is a compile-time constant.
    Supports D up to 256 (LOG2_D up to 8).

    Args:
        x: Current D-element vector (vectorized across threads)
        scratch: Butterfly shuffle context ``(temp_ptr, temp_base, d_idx)`` passed
            through to each :func:`_hadamard_butterfly_stage` call.
        D: Head dimension (compile-time constant)
        LOG2_D: log2(D) (compile-time constant)
    """
    for stage in tl.static_range(LOG2_D):
        x = _hadamard_butterfly_stage(x, scratch, stage, D)
    inv_sqrt_d = 1.0 / tl.sqrt(float(D))
    return x * inv_sqrt_d


# =============================================================================
# Inverse Hadamard transform kernel
# Applied to attention output to recover correct results after V was transformed
# =============================================================================


@triton.jit
def _inverse_hadamard_kernel(
    # Buffer pointers, in order:
    #   input_ptr   input tensor            [B, H, S, D] (may be non-contiguous)
    #   output_ptr  output tensor           [B, H, S, D] (contiguous)
    #   temp_ptr    butterfly scratch       [B, H, num_chunks, D]
    ptrs,
    # Strides as (in_strides, out_strides, temp_strides), each a 4-tuple:
    #   in_strides   for input  [B, H, S, D]: (stride_b, stride_h, stride_s, stride_d)
    #   out_strides  for output [B, H, S, D]: (stride_b, stride_h, stride_s, stride_d)
    #   temp_strides for temp   [B, H, num_chunks, D]: (stride_b, stride_h, stride_c, stride_d)
    strides,
    # Dimensions: (S, H, chunk_size, num_chunks)
    dims,
    # Head dimension (kept separate so it can key the launch)
    D: tl.constexpr,
    # Remaining compile-time constants: (LOG2_D, USE_BFLOAT16)
    META: tl.constexpr,
):
    """Apply inverse Hadamard transform along D dimension.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across S positions in chunk.

    ``ptrs``, ``strides`` and ``dims`` bundle what would otherwise be long,
    same-typed parameter runs, matching the convention used by the phase1
    kernels in this package.
    """
    input_ptr, output_ptr, temp_ptr = ptrs
    in_strides, out_strides, temp_strides = strides
    stride_in_b, stride_in_h, stride_in_s, stride_in_d = in_strides
    stride_out_b, stride_out_h, stride_out_s, stride_out_d = out_strides
    stride_temp_b, stride_temp_h, stride_temp_c, stride_temp_d = temp_strides
    S, H, chunk_size, num_chunks = dims
    LOG2_D, USE_BFLOAT16 = META

    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base = pid_b * stride_in_b + pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        in_offset = in_base + s_idx * stride_in_s + d_idx * stride_in_d
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d

        x = tl.load(input_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        x = _apply_hadamard(x, (temp_ptr, temp_base, d_idx), D, LOG2_D)

        if USE_BFLOAT16:
            tl.store(output_ptr + out_offset, x.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(output_ptr + out_offset, x.to(tl.float16), mask=s_mask)


def inverse_hadamard_transform(
    x: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> torch.Tensor:
    """Apply inverse Hadamard transform along the last dimension.

    Input shape: [B, H, S, D] where D must be a power of 2 and <= 256.
    Output: same shape and dtype, always contiguous.

    The Hadamard transform is self-inverse up to normalization, so this
    applies the same butterfly + 1/sqrt(D) as the forward transform.
    """
    assert x.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {x.dim()}D"

    B, H, S, D = x.shape
    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"Expected bf16 or fp16, got {x.dtype}"
    )

    if num_chunks is None:
        num_chunks = _compute_num_chunks(x.device, B, H, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    output = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    temp_buffer = torch.empty(B, H, num_chunks, D, dtype=torch.float32, device=x.device)

    grid = (B, H, num_chunks)
    use_bfloat16 = x.dtype == torch.bfloat16

    _inverse_hadamard_kernel[grid](
        (x, output, temp_buffer),
        (
            (x.stride(0), x.stride(1), x.stride(2), x.stride(3)),
            (output.stride(0), output.stride(1), output.stride(2), output.stride(3)),
            (
                temp_buffer.stride(0),
                temp_buffer.stride(1),
                temp_buffer.stride(2),
                temp_buffer.stride(3),
            ),
        ),
        (S, H, chunk_size, num_chunks),
        D=D,
        META=(LOG2_D, use_bfloat16),
        num_warps=4,
    )

    return output
