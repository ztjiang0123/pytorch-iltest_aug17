# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import torch
from triton.testing import do_bench

from torchao.ops import (
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
from torchao.sparsity.utils import create_semi_structured_tensor


def benchmark_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark(m, k):
    torch.manual_seed(123)
    W_ref = create_semi_structured_tensor(m, k, dtype=torch.float8_e4m3fn).cuda()

    # packed, meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)
    cutlass_reference_args = (W_ref,)
    cutlass_custom_args = (W_ref, "", True)

    cutlass_reference_compression_time = benchmark_microseconds(
        to_sparse_semi_structured_cutlass_sm9x_f8, *cutlass_reference_args
    )
    cutlass_custom_compression_time = benchmark_microseconds(
        torch.ops.torchao.sparse_semi_structured_tile.default, *cutlass_custom_args
    )

    return {
        "cutlass_reference (ms)": cutlass_reference_compression_time,
        "cutlass_custom (ms)": cutlass_custom_compression_time,
    }


def profile():
    torch.manual_seed(123)
    W_ref = create_semi_structured_tensor(8192, 8192, dtype=torch.float8_e4m3fn).cuda()

    # clear cache
    new_val = torch.empty(10000, 10000, device="cuda")
    new_val[:, :] = 0

    packed, meta = torch.ops.torchao.sparse_semi_structured_tile.default(
        W_ref, "", True
    )


if __name__ == "__main__":
    results = []
    for m in (2048, 4096, 8192):
        results.append(benchmark(m, 8192))

    df = pd.DataFrame(results)
    df.to_csv("rowwise_scaled_linear_sparse_cutlass_time_results.csv", index=False)
    print(df.to_markdown(index=False))

    # print("PROFILING")
    # profile()
