# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run this benchmark, use the following command:
#
# torchrun --nproc-per-node=2 --local-ranks-filter=0 benchmarks/prototype/moe_training/mxfp8/bench_ep_pipeline.py
#
#######################################################################
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from tabulate import tabulate
from torch import distributed as dist
from torch.distributed._functional_collectives import all_to_all_single
from torch.nn import functional as F
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from benchmarks.prototype.moe_training.bench_utils import setup_distributed
from benchmarks.utils import profile_fn
from torchao.prototype.moe_training.ep import (
    a2a_combine_hp_fwd_mxfp8_bwd,
    a2a_dispatch_mxfp8_fwd_hp_bwd,
    permute_mxfp8_fwd_hp_bwd,
    unpermute_hp_fwd_mxfp8_bwd,
)
from torchao.prototype.moe_training.ep.permute import permute_and_pad
from torchao.prototype.moe_training.ep.unpermute import _unpermute_bf16
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    dim: int
    hidden_dim: int
    num_experts: int


@dataclass(frozen=True)
class ExperimentResult:
    # Forward times
    fwd_bf16_ms: float
    fwd_mxfp8_ms: float
    # Backward times
    bwd_bf16_ms: float
    bwd_mxfp8_ms: float
    # Speedup metrics
    fwd_speedup: float
    bwd_speedup: float
    total_speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


@dataclass(frozen=True)
class RoutingContext:
    """Token distribution and expert-parallel topology shared by both pipelines.

    These values are computed together in :func:`run_experiment` and always
    travel as a unit, so they are bundled here rather than passed individually.
    """

    num_tokens_per_expert: torch.Tensor
    num_tokens_per_expert_group: torch.Tensor
    input_splits_list: List[int]
    output_splits_list: List[int]
    ep_degree: int
    num_experts: int
    group: object


def get_configs() -> List[ExperimentConfig]:
    """Generate experiment configurations."""
    configs = [
        ExperimentConfig(num_tokens=131072, dim=8192, hidden_dim=5120, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=7168, hidden_dim=2048, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=2048, hidden_dim=1408, num_experts=8),
    ]
    return configs


def standard_pipeline(
    input_tensor: torch.Tensor,
    expert_weights_t: torch.Tensor,
    routing: RoutingContext,
) -> torch.Tensor:
    """
    Standard BF16 pipeline:
    bf16 a2a -> bf16 permute -> _to_mxfp8_then_scaled_grouped_mm -> bf16 unpermute -> bf16 a2a combine
    """
    block_size = 32

    # Step 1: All-to-all dispatch (BF16)
    dispatched = all_to_all_single(
        input_tensor,
        routing.output_splits_list,
        routing.input_splits_list,
        group=routing.group,
    )
    dispatched = torch.ops._c10d_functional.wait_tensor(dispatched)

    # Step 2: Permute (BF16)
    input_shape, permuted, permuted_indices, num_tokens_per_expert_padded, offsets = (
        permute_and_pad(
            dispatched,
            routing.num_tokens_per_expert_group,
            routing.ep_degree,
            routing.num_experts,
            block_size,
        )
    )

    # Step 3: BF16 Grouped MM
    gemm_output = _to_mxfp8_then_scaled_grouped_mm(
        permuted,
        expert_weights_t,
        offs=offsets,
        out_dtype=torch.bfloat16,
        wgrad_with_hp=True,
    )

    # Step 4: Unpermute (BF16)
    # Create output shape with same number of rows as input_shape, but output dimension from gemm_output
    output_shape = (input_shape[0], gemm_output.shape[-1])
    unpermuted = _unpermute_bf16(gemm_output, permuted_indices, output_shape)

    # Step 5: All-to-all combine (BF16)
    final_output = all_to_all_single(
        unpermuted,
        routing.input_splits_list,
        routing.output_splits_list,
        group=routing.group,
    )
    final_output = torch.ops._c10d_functional.wait_tensor(final_output)

    return final_output


def mxfp8_pipeline(
    input_tensor: torch.Tensor,
    expert_weights_t: torch.Tensor,
    routing: RoutingContext,
) -> torch.Tensor:
    """
    MXFP8 optimized pipeline with chained autograd functions:
    bf16 -> a2a_dispatch (MXTensor) -> permute (MXTensor) ->
    mxfp8_grouped_mm -> unpermute -> a2a_combine -> bf16
    """
    block_size = 32

    # Step 1: A2A dispatch - outputs MXTensor
    mx_dispatched = a2a_dispatch_mxfp8_fwd_hp_bwd(
        input_tensor,
        routing.output_splits_list,
        routing.input_splits_list,
        group_name=routing.group.group_name,
    )

    # Step 2: Permute - maintains MXTensor
    (
        padded_mx_shape,
        mx_permuted,
        permuted_indices,
        num_tokens_per_expert_padded,
        mx_group_offsets,
    ) = permute_mxfp8_fwd_hp_bwd(
        mx_dispatched,
        routing.num_tokens_per_expert_group,
        routing.ep_degree,
        routing.num_experts,
        block_size,
        use_triton_for_bwd=True,
    )

    # Step 3: MXFP8 Grouped MM - outputs BF16
    gemm_output = _to_mxfp8_then_scaled_grouped_mm(
        mx_permuted,
        expert_weights_t,
        offs=mx_group_offsets,
        wgrad_with_hp=True,
    )

    # Step 4: Unpermute - maintains BF16
    # Update padded_shape to have output dimension instead of input dimension
    padded_output_shape = torch.Size([padded_mx_shape[0], gemm_output.shape[-1]])
    unpermuted = unpermute_hp_fwd_mxfp8_bwd(
        gemm_output,
        permuted_indices,
        padded_output_shape,
    )

    # Step 5: A2A combine - maintains BF16
    final_output = a2a_combine_hp_fwd_mxfp8_bwd(
        unpermuted,
        output_splits=routing.input_splits_list,
        input_splits=routing.output_splits_list,
        group_name=routing.group.group_name,
        mxfp8_bwd=True,
    )

    return final_output


def mse_loss_and_bwd(output: torch.Tensor, labels: torch.Tensor):
    """Compute MSE loss and run backward pass."""
    loss = F.mse_loss(output, labels)
    loss.backward()


def _make_input_tensors(config: ExperimentConfig):
    """Create the input activations and expert weights for an experiment.

    Returns two (tensor, ref_tensor) pairs; the ``ref_`` variants are
    independent clones used for the BF16 pipeline so its autograd graph does
    not interfere with the MXFP8 pipeline's.
    """
    input_tensor = torch.randn(
        config.num_tokens,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

    expert_weights = torch.randn(
        config.num_experts,
        config.hidden_dim,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    ref_expert_weights = expert_weights.detach().clone().requires_grad_(True)

    return input_tensor, ref_input_tensor, expert_weights, ref_expert_weights


def _compute_token_distribution(config: ExperimentConfig, ep_degree: int, group):
    """Build the uniform token distribution and all-to-all splits."""
    total_experts = ep_degree * config.num_experts
    assert config.num_tokens % total_experts == 0
    uniform_group_size = config.num_tokens // total_experts
    num_tokens_per_expert = torch.full(
        (total_experts,), uniform_group_size, dtype=torch.int32, device="cuda"
    )

    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=group,
        )
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        input_splits = (
            num_tokens_per_expert.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = (
            num_tokens_per_expert_group.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )

    return (
        num_tokens_per_expert,
        num_tokens_per_expert_group,
        input_splits.tolist(),
        output_splits.tolist(),
    )


def _warmup(func_no_args, n=2):
    for _ in range(n):
        func_no_args()


def _make_bwd_warmup(fwd_fn, input_t, weight_t):
    # Build a backward-warmup closure for a given pipeline: reset grads,
    # run the forward pass, then run loss + backward so kernels compile.
    def bwd_warmup():
        input_t.grad = None
        weight_t.grad = None
        output = fwd_fn(input_t, weight_t.transpose(-2, -1))
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    return bwd_warmup


def _time_forward(fwd_fn, input_t, weight_t) -> float:
    """Warm up then time a single forward pass, in milliseconds."""
    _warmup(lambda: fwd_fn(input_t, weight_t.transpose(-2, -1)))
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = fwd_fn(input_t, weight_t.transpose(-2, -1))
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    return (end_sec - start_sec) * 1e3


def _time_backward(fwd_fn, input_t, weight_t):
    """Warm up then time a single backward pass, in milliseconds.

    Returns the elapsed time and the labels tensor used, so callers can reuse
    the labels for profiling.
    """
    _warmup(_make_bwd_warmup(fwd_fn, input_t, weight_t))

    # Do a fresh forward pass right before timing backward
    input_t.grad = None
    weight_t.grad = None
    output_for_bwd = fwd_fn(input_t, weight_t.transpose(-2, -1))
    labels = torch.ones_like(output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(output_for_bwd, labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    return (end_sec - start_sec) * 1e3, labels


def _profile_pipeline(fwd_fn, config: ExperimentConfig, labels, profile_name: str):
    """Profile a full forward+backward pass on fresh tensors."""

    def fwd_bwd(input_t, weight_t, labels):
        output = fwd_fn(input_t, weight_t)
        mse_loss_and_bwd(output, labels)

    # Create fresh tensors for profiling to avoid autograd graph conflicts
    input_tensor_profile = torch.randn(
        config.num_tokens,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    expert_weights_profile = torch.randn(
        config.num_experts,
        config.hidden_dim,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    profile_fn(
        fwd_bwd,
        input_tensor_profile,
        expert_weights_profile.transpose(-2, -1),
        labels,
        distributed=True,
        profile_name=profile_name,
    )


@dataclass
class _PipelineInputs:
    """Bundles the tensors and config a pipeline benchmark needs."""

    experiment: ExperimentConfig
    input_t: torch.Tensor
    weight_t: torch.Tensor


def _benchmark_pipeline(
    fwd_fn,
    inputs: _PipelineInputs,
    args: argparse.Namespace,
    profile_name: str,
):
    """Benchmark forward and backward for a single pipeline.

    The random seed is reset first so both pipelines see the same random state.
    Returns (fwd_ms, bwd_ms).
    """
    torch.manual_seed(42)

    fwd_ms = _time_forward(fwd_fn, inputs.input_t, inputs.weight_t)
    bwd_ms, labels = _time_backward(fwd_fn, inputs.input_t, inputs.weight_t)

    if args.profile:
        _profile_pipeline(fwd_fn, inputs.experiment, labels, profile_name)

    return fwd_ms, bwd_ms


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    """Run a single experiment comparing both pipelines."""
    (
        input_tensor,
        ref_input_tensor,
        expert_weights,
        ref_expert_weights,
    ) = _make_input_tensors(config)

    ep_degree = dist.get_world_size()
    group = dist.group.WORLD
    (
        num_tokens_per_expert,
        num_tokens_per_expert_group,
        input_splits_list,
        output_splits_list,
    ) = _compute_token_distribution(config, ep_degree, group)

    routing = RoutingContext(
        num_tokens_per_expert=num_tokens_per_expert,
        num_tokens_per_expert_group=num_tokens_per_expert_group,
        input_splits_list=input_splits_list,
        output_splits_list=output_splits_list,
        ep_degree=ep_degree,
        num_experts=config.num_experts,
        group=group,
    )

    def bf16_fwd(input_t, weight_t):
        return standard_pipeline(input_t, weight_t, routing)

    def mxfp8_fwd(input_t, weight_t):
        return mxfp8_pipeline(input_t, weight_t, routing)

    # === Benchmark Standard BF16 Pipeline ===
    fwd_bf16_ms, bwd_bf16_ms = _benchmark_pipeline(
        bf16_fwd,
        _PipelineInputs(config, ref_input_tensor, ref_expert_weights),
        args,
        "bf16_pipeline",
    )

    # === Benchmark MXFP8 Pipeline ===
    fwd_mxfp8_ms, bwd_mxfp8_ms = _benchmark_pipeline(
        mxfp8_fwd,
        _PipelineInputs(config, input_tensor, expert_weights),
        args,
        "mxfp8_pipeline",
    )

    # Calculate speedups
    fwd_speedup = fwd_bf16_ms / fwd_mxfp8_ms
    bwd_speedup = bwd_bf16_ms / bwd_mxfp8_ms
    total_bf16_ms = fwd_bf16_ms + bwd_bf16_ms
    total_mxfp8_ms = fwd_mxfp8_ms + bwd_mxfp8_ms
    total_speedup = total_bf16_ms / total_mxfp8_ms

    return ExperimentResult(
        fwd_bf16_ms=fwd_bf16_ms,
        fwd_mxfp8_ms=fwd_mxfp8_ms,
        bwd_bf16_ms=bwd_bf16_ms,
        bwd_mxfp8_ms=bwd_mxfp8_ms,
        fwd_speedup=fwd_speedup,
        bwd_speedup=bwd_speedup,
        total_speedup=total_speedup,
    )


def print_results(experiments: List[Experiment]):
    """Print benchmark results in a formatted table."""
    headers = [
        "tokens",
        "dim",
        "hidden_dim",
        "num_experts",
        "fwd_bf16_ms",
        "fwd_mxfp8_ms",
        "fwd_speedup",
        "bwd_bf16_ms",
        "bwd_mxfp8_ms",
        "bwd_speedup",
        "total_speedup",
    ]
    rows = []
    for experiment in experiments:
        cfg = experiment.config
        res = experiment.result
        rows.append(
            [
                cfg.num_tokens,
                cfg.dim,
                cfg.hidden_dim,
                cfg.num_experts,
                f"{res.fwd_bf16_ms:.3f}",
                f"{res.fwd_mxfp8_ms:.3f}",
                f"{res.fwd_speedup:.2f}x",
                f"{res.bwd_bf16_ms:.3f}",
                f"{res.bwd_mxfp8_ms:.3f}",
                f"{res.bwd_speedup:.2f}x",
                f"{res.total_speedup:.2f}x",
            ]
        )
    print("\n" + "=" * 120)
    print("Expert Parallelism Pipeline Benchmark Results")
    print(f"World Size: {dist.get_world_size()}")
    print("=" * 120)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 120 + "\n")


def main(args: argparse.Namespace):
    """Main benchmark entry point."""
    torch.random.manual_seed(123)

    # Set up process group
    setup_distributed()

    # Generate experiment configs
    configs = get_configs()
    results = []
    for config in tqdm(
        configs, desc="Running experiments", disable=dist.get_rank() != 0
    ):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Print results (only on rank 0)
    if dist.get_rank() == 0:
        print_results(results)

    # Clean up process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MoE Expert Parallelism pipelines (BF16 vs MXFP8)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for detailed performance analysis",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    args = parser.parse_args()
    main(args)
