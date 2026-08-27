# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared benchmarking helpers for the nvfp4_training benchmark scripts."""

import argparse
from dataclasses import dataclass
from typing import Callable, List

import torch
from tabulate import tabulate
from tqdm import tqdm

LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048


def build_representative_model_configs(
    make_config: Callable[[int, int, str, str], object],
):
    """Build the shared set of Llama-derived shapes.

    Args:
        make_config: Factory called as ``make_config(m, n, model, shape)`` that
            returns the caller's ``ExperimentConfig`` instance.

    Returns:
        List of experiment configs for the representative model shapes.
    """
    llama_m = LLAMA_BATCH_SIZE * LLAMA_SEQ_LEN

    return [
        make_config(llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        make_config(llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        make_config(llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        make_config(llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]


@dataclass
class BenchmarkHarness:
    """The callables a benchmark script provides to :func:`run_benchmark_main`.

    Grouping these hooks keeps ``run_benchmark_main``'s signature small and
    documents that they are the single unit of behavior a script plugs in.
    """

    get_configs: Callable[[], List[object]]
    get_representative_model_configs: Callable[[], List[object]]
    run_experiment: Callable[[object], object]
    make_experiment: Callable[[object, object], object]
    print_results: Callable[[List[object]], None]


def run_benchmark_main(
    harness: BenchmarkHarness,
    seed: int = 123,
):
    """Parse args, run the selected shape set, and print the results table.

    ``harness.run_experiment`` may return ``None`` for configs that are skipped
    (e.g. unsupported hardware); such configs are omitted from the results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("sweep", "representative-models"),
        default="sweep",
        help="Benchmark the original sweep or selected model-derived shapes.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(seed)
    configs = (
        harness.get_representative_model_configs()
        if args.shape_set == "representative-models"
        else harness.get_configs()
    )
    results = []
    for config in tqdm(configs):
        result = harness.run_experiment(config)
        if result is not None:
            results.append(harness.make_experiment(config, result))
    harness.print_results(results)


def print_results(experiments: List[object]):
    """Print an M/N/time/gbps results table for the given experiments.

    Each experiment is expected to expose ``config`` (with ``m``, ``n``,
    optional ``model``/``shape`` fields) and ``result`` (with ``time_us`` and
    ``gbps`` fields). ``model``/``shape`` label columns are added only when at
    least one experiment provides them.
    """
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "time_us", "gbps"]
    rows = []
    for e in experiments:
        row = [
            e.config.m,
            e.config.n,
            round(e.result.time_us, 3),
            round(e.result.gbps, 3),
        ]
        if has_labels:
            row = [e.config.model, e.config.shape] + row
        rows.append(row)
    if has_labels:
        headers = ["model", "shape"] + headers
    print(tabulate(rows, headers=headers))
