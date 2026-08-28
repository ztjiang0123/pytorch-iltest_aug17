# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from triton import Config

from torchao.prototype.common.triton.matmul import get_configs_io_bound

# TODO: add early config prune and estimate_matmul_time to reduce autotuning time
# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def get_configs_compute_bound():
    configs = [
        # basic configs for compute-bound matmuls
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ]
    return configs


MIXED_MM_HEURISTICS = {
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    "BLOCK_K": lambda args: (
        min(args["BLOCK_K"], args["QGROUP_SIZE"])
        if not args["TRANSPOSED"]
        else args["BLOCK_K"]
    ),
    "BLOCK_N": lambda args: (
        min(args["BLOCK_N"], args["QGROUP_SIZE"])
        if args["TRANSPOSED"]
        else args["BLOCK_N"]
    ),
    "SPLIT_K": lambda args: (
        1 if args["IS_BFLOAT16"] else args["SPLIT_K"]
    ),  # atomic add not supported for bfloat16
}


@triton.jit
def _mixed_mm_kernel(
    # Operands: A, B, scales, zeros, C
    A,
    B,
    scales_ptr,
    zeros_ptr,
    C,
    # Matrix dims
    M,
    N,
    K,
    # a / b / c / scale / zero strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_k,
    stride_scale_n,
    # Meta-params (constexpr): dtype flag, quant group size, block / split sizes
    IS_BFLOAT16: tl.constexpr,
    QGROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    TRANSPOSED: tl.constexpr = False,
    GROUP_M: tl.constexpr = 8,
    # tl.dot options and debug toggle
    acc_dtype: tl.constexpr = tl.float32,
    input_precision: tl.constexpr = "ieee",
    fp8_fast_accum: tl.constexpr = False,
    DEBUG: tl.constexpr = False,
):
    """Mixed matmul kernel.

    A is (M, K) float16 / bfloat16 / float32. B is i4 / s4 packed as uint8 / int8
    with shape (K // 2, N) (see ``packed_2xint4``). Scales and zeros are (NUM_GROUPS, N),
    same dtype as A, where NUM_GROUPS = K // QGROUP_SIZE; QGROUP_SIZE must be a multiple
    of BLOCK_K so one scale / zero vector broadcasts to the block per mainloop iteration.

    In the transposed case A is M x N and B is K x N and we reduce along "N": rows of A and
    blocks of B are loaded, each B block dequantized and transposed (BLK_N x BLK_K ->
    BLK_K x BLK_N) so the matmul is transposed without unpacking / repacking B. Scale / zero
    indexing flips accordingly, iterating the non-grouping dim within the mac loop and the
    grouping dim across blocks.

    NOTE: Assumes quantization grouping was done along the K dimension originally (QGROUP_SIZE
    consecutive K-dim elements grouped together when computing min / max scaling factors).
    """

    if not TRANSPOSED:
        tl.static_assert(QGROUP_SIZE % BLOCK_K == 0)
    else:
        tl.static_assert(QGROUP_SIZE % BLOCK_N == 0)

    # Threadblock swizzling
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    if not DEBUG:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm
    rak = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    # BLOCK_K for b is effectively BLOCK_K // 2
    if not TRANSPOSED:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            rbn = rn
        rbk = pid_z * BLOCK_K // 2 + tl.arange(0, BLOCK_K // 2)
    else:
        rn = (pid_n * BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N // 2), BLOCK_N // 2)
        else:
            rbn = rn
        rbk = rak

    A = A + (ram[:, None] * stride_am + rak[None, :] * stride_ak)

    if not TRANSPOSED:
        B = B + (rbk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    else:
        # Transposed: load BLK_N x BLK_K but transpose to BLK_K x BLK_N, so B strides
        # are swapped (stride_bk for rows of BLK_N, stride_bn for columns of BLK_K).
        B = B + (rbn[:, None] * stride_bk + rbk[None, :] * stride_bn)

    # Grouping is along K: the mainloop marches down K (group idx = K // QGROUP_SIZE)
    # while grouping varies along N, so each block loads a BLK_K x BLK_N row vector.
    if not TRANSPOSED:
        offsets_scale_n = (
            pid_n * stride_scale_n * BLOCK_N + tl.arange(0, BLOCK_N) * stride_scale_n
        )
    else:
        scale_offset_k = pid_n * BLOCK_N * stride_scale_k // QGROUP_SIZE
        offsets_scale_n = tl.arange(0, BLOCK_K) * stride_scale_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            qb = tl.load(B)
        else:
            k_remaining_a = K - k * (BLOCK_K * SPLIT_K)
            if not TRANSPOSED:
                k_remaining_b = (
                    K - k * (BLOCK_K * SPLIT_K) // 2
                )  # Note the division by 2
            else:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K)  # = k_remaining_a

            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rak[None, :] < k_remaining_a, other=_0)
            qb = tl.load(B, mask=rbk[:, None] < k_remaining_b, other=_0)

        if not TRANSPOSED:
            scale_offset_k = k * BLOCK_K * SPLIT_K * stride_scale_k // QGROUP_SIZE
        else:
            offsets_scale_n = (
                k * stride_scale_n * BLOCK_K + tl.arange(0, BLOCK_K) * stride_scale_n
            )

        scales = tl.load(scales_ptr + offsets_scale_n + scale_offset_k)
        zeros = tl.load(zeros_ptr + offsets_scale_n + scale_offset_k)

        # Unpack qweights -- h/t jlebar!
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        qb_lo = (qb << _4_i8) >> _4_i8
        qb_hi = qb >> _4_i8

        # Upcast to fp16. bfloat16 needs an intermediate fp16 hop (direct int8 ->
        # bfloat16 conversion triggers a compilation error).
        if IS_BFLOAT16:
            dq_b = tl.join(
                qb_lo.to(tl.float16).to(A.dtype.element_ty),
                qb_hi.to(tl.float16).to(A.dtype.element_ty),
            ).permute(0, 2, 1)
        else:
            dq_b = tl.join(
                qb_lo.to(A.dtype.element_ty),
                qb_hi.to(A.dtype.element_ty),
            ).permute(0, 2, 1)
        if not TRANSPOSED:
            dq_b = dq_b.reshape(BLOCK_K, BLOCK_N)
        else:
            dq_b = dq_b.reshape(BLOCK_N, BLOCK_K)

        # Scale upcasted weights, broadcasting scales / zeros across the block
        # (all scales fall within a single QGROUP -- statically checked above).
        dq_b = (dq_b - zeros[None, :]) * scales[None, :]

        if TRANSPOSED:
            dq_b = tl.trans(dq_b)

        if fp8_fast_accum:
            acc = tl.dot(
                a, dq_b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, dq_b, out_dtype=acc_dtype, input_precision=input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak

        # Advance by half the block size, since each block is unpacked and upcasted into two fp16 values
        if not TRANSPOSED:
            B += BLOCK_K * SPLIT_K * stride_bk // 2
        else:
            # we iterating across a row of B (non-packing dim, hence no need for div 2)
            B += BLOCK_K * SPLIT_K * stride_bn
    acc = acc.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        # AMD GPUs need relaxed semantics for better performance
        if tl.constexpr(torch.version.hip is not None):
            tl.atomic_add(C, acc, mask=mask, sem="relaxed")
        else:
            tl.atomic_add(C, acc, mask=mask)


_mixed_mm = triton.heuristics(MIXED_MM_HEURISTICS)(_mixed_mm_kernel)
mixed_mm_kernel_max_autotune = triton.autotune(
    configs=get_configs_compute_bound() + get_configs_io_bound(), key=["M", "N", "K"]
)(_mixed_mm)
mixed_mm_kernel_compute_bound = triton.autotune(
    configs=get_configs_compute_bound(), key=["M", "N", "K"]
)(_mixed_mm)
_mixed_mm_debug = _mixed_mm
