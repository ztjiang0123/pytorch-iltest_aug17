# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared benchmarking helpers for the nvfp4_training benchmark scripts."""

import argparse
from typing import Callable, List

import torch
from tqdm import tqdm

LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048


def build_representative_model_configs(make_config: Callable[[int, int, str, str], object]):
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


def run_benchmark_main(
    get_configs: Callable[[], List[object]],
    get_representative_model_configs: Callable[[], List[object]],
    run_experiment: Callable[[object], object],
    make_experiment: Callable[[object, object], object],
    print_results: Callable[[List[object]], None],
    seed: int = 123,
):
    """Parse args, run the selected shape set, and print the results table.

    ``run_experiment`` may return ``None`` for configs that are skipped (e.g.
    unsupported hardware); such configs are omitted from the results.
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
        get_representative_model_configs()
        if args.shape_set == "representative-models"
        else get_configs()
    )
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(make_experiment(config, result))
    print_results(results)
