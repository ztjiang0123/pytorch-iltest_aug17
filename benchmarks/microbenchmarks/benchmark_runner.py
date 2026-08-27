# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Benchmark Runner

This is the main entry point for the benchmarking application. It reads the YAML configuration
file and orchestrates the entire benchmarking process by:
- Loading and validating benchmark configurations
- Executing benchmark scenarios
- Collecting and processing results
- Generating reports

Usage:
    python benchmark_runner.py [config.yaml]

The YAML file should contain all necessary configuration parameters for the benchmarks.
"""

import argparse
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    BenchmarkConfigParams,
    generate_results_csv,
    print_results,
)

# Static model weight-shape tables, keyed by the shape-config ``name``.
# Each entry is a list of (label, (M, K, N)) tuples materialized verbatim.
_LLAMA_M = 4 * 4096  # bsz=4, seq_len=4096, fused attn.wqkv and ffn.w13
_STATIC_MODEL_SHAPES: Dict[str, List[Tuple[str, Tuple[int, int, int]]]] = {
    "llama": [
        ("attn.wqkv", (_LLAMA_M, 8192, 1280)),
        ("attn.w0", (_LLAMA_M, 1024, 8192)),
        ("ffn.w13", (_LLAMA_M, 8192, 7168)),
        ("ffn.w2", (_LLAMA_M, 3584, 8192)),
    ],
    "llama4": [
        ("FFN", (16384, 8192, 5120)),
        ("QO_proj", (16384, 8192, 8192)),
        ("KV_proj", (16384, 8192, 1024)),
        ("FFN", (128000, 8192, 5120)),
        ("QO_proj", (128000, 8192, 8192)),
        ("KV_proj", (128000, 8192, 1024)),
    ],
    "deepseek_v3_236b": [
        ("FFN", (16384, 1536, 5120)),
        ("QKVO_proj", (16384, 7168, 7168)),
        ("FFN", (128000, 1536, 5120)),
        ("QKVO_proj", (128000, 7168, 7168)),
    ],
    "deepseek_v3_671b": [
        ("FFN", (16384, 2048, 7168)),
        ("QKVO_proj", (16384, 7168, 7168)),
        ("FFN", (128000, 2048, 7168)),
        ("QKVO_proj", (128000, 7168, 7168)),
    ],
    "qwen3_32b": [
        ("QO_proj", (16384, 5120, 5120)),
        ("KV_proj", (16384, 5120, 640)),
        ("QO_proj", (128000, 5120, 5120)),
        ("KV_proj", (128000, 5120, 640)),
    ],
    "gemma3_27b": [
        ("QO_proj", (16384, 4096, 4096)),
        ("KV_proj", (16384, 4096, 1024)),
        ("QO_proj", (128000, 4096, 4096)),
        ("KV_proj", (128000, 4096, 1024)),
    ],
}


def _custom_shapes(name, shape_config):
    return [(name, shape) for shape in shape_config["shapes"]]


def _static_model_shapes(name, shape_config):
    return [(f"{name}_{label}", shape) for label, shape in _STATIC_MODEL_SHAPES[name]]


def _pow2_shapes(name, shape_config):
    min_power_of_2 = shape_config.get("min_power", 10)  # 1024
    max_power_of_2 = shape_config.get("max_power", 14)  # 16,384
    shapes = []
    for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
        val = 2**power_of_2
        shapes.append((f"{name}_{idx}", [val, val, val]))
    return shapes


def _pow2_extended_shapes(name, shape_config):
    # Powers of 2 and powers of 2 + half.
    min_power_of_2 = shape_config.get("min_power", 10)  # 1024
    max_power_of_2 = shape_config.get("max_power", 14)  # 16,384
    shapes = []
    for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
        val1 = 2**power_of_2
        val2 = 2**power_of_2 + 2 ** (power_of_2 - 1)
        shapes.append((f"{name}_{idx * 2}", [val1, val1, val1]))
        shapes.append((f"{name}_{idx * 2 + 1}", [val2, val2, val2]))
    return shapes


def _sweep_shapes(name, shape_config, *, default_min, default_max, increasing_only):
    min_p2 = shape_config.get("min_power", default_min)
    max_p2 = shape_config.get("max_power", default_max)
    shapes = []
    counter = 0
    for M_p2, K_p2, N_p2 in product(range(min_p2, max_p2 + 1), repeat=3):
        M, K, N = 2**M_p2, 2**K_p2, 2**N_p2
        if increasing_only and not (M <= K <= N):
            continue
        shapes.append((f"{name}_{counter}", [M, K, N]))
        counter += 1
    return shapes


def _small_sweep_shapes(name, shape_config):
    return _sweep_shapes(
        name, shape_config, default_min=10, default_max=14, increasing_only=True
    )


def _full_sweep_shapes(name, shape_config):
    return _sweep_shapes(
        name, shape_config, default_min=8, default_max=15, increasing_only=False
    )


# Dispatch table: shape-config ``name`` -> handler(name, shape_config) -> shapes.
_SHAPE_HANDLERS = {
    "custom": _custom_shapes,
    "pow2": _pow2_shapes,
    "pow2_extended": _pow2_extended_shapes,
    "small_sweep": _small_sweep_shapes,
    "sweep": _full_sweep_shapes,
    **{name: _static_model_shapes for name in _STATIC_MODEL_SHAPES},
}


def get_shapes_for_config(
    shape_configs: List[Dict[str, Any]],
) -> List[Tuple[str, List[int]]]:
    """Get shapes for a given configuration.

    Args:
        shape_configs: List of shape configurations from YAML

    Returns:
        List of tuples containing (shape_name, shape)
    """
    shapes = []
    for shape_config in shape_configs:
        name = shape_config["name"]
        handler = _SHAPE_HANDLERS.get(name)
        if handler is None:
            supported = ", ".join(sorted(_SHAPE_HANDLERS))
            raise NotImplementedError(
                f"Shape config {name} not supported. Supported options: {supported}."
            )
        shapes.extend(handler(name, shape_config))
    return shapes


def get_param_combinations(model_param):
    """Extract all parameter combinations from a model config"""
    # Get all shapes
    shapes = get_shapes_for_config(model_param["matrix_shapes"])

    # Extract all other parameters (excluding matrix_shapes)
    base_params = {
        key: value for key, value in model_param.items() if key not in ["matrix_shapes"]
    }

    return shapes, base_params


def get_quantization_sparsity_recipes(
    quantization_recipes: List[str], sparsity_recipes: List[str]
) -> Set[Tuple[str, Optional[str]]]:
    """Generate valid quantization and sparsity recipes.

    Args:
        quantization_recipes: List of quantization recipes
        sparsity_recipes: List of sparsity recipes

    Returns:
        Set of tuples containing (quantization_recipe, sparsity_recipe)
        For block sparsity, quantization is always "baseline"
        All quantization techniques are also run without sparsity
    """
    config_recipes = set()

    # Add all quantization techniques without sparsity
    for quant_config in quantization_recipes:
        config_recipes.add((quant_config, None))

    # Process combinations of quantization and sparsity
    for sparse_config in sparsity_recipes:
        if sparse_config is None:
            # Skip None sparsity as we've already added all quantization techniques without sparsity
            continue
        elif "block" in sparse_config:
            # For block sparsity, only pair with baseline quantization
            config_recipes.add(("baseline", sparse_config))
        elif "semi" in sparse_config or "2:4" in sparse_config:
            # For semi-sparse, only pair with compatible quantization methods
            for quant_config in quantization_recipes:
                if (
                    "marlin" in quant_config
                    or "int8dq" in quant_config
                    or "float8dq" in quant_config
                    or quant_config == "baseline"
                ):
                    config_recipes.add((quant_config, sparse_config))
        else:
            raise ValueError(f"Invalid sparsity recipe: {sparse_config}")

    return config_recipes


def load_benchmark_configs(cli_args: argparse.Namespace) -> List[BenchmarkConfig]:
    """Load benchmark configurations from CLI arguments and YAML file."""
    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "benchmarks/microbenchmarks/results")
    benchmark_mode = config.get("benchmark_mode", "inference")

    # Create all possible combinations
    configs = []
    quantization_sparsity_recipes = get_quantization_sparsity_recipes(
        config.get("quantization_config_recipe_names", []),
        config.get("sparsity_config_recipe_names", []),
    )
    for model_param in config["model_params"]:
        shapes, params = get_param_combinations(model_param)

        # Create configs for all combinations
        for (quant_config, sparse_config), (shape_name, shape) in product(
            quantization_sparsity_recipes,
            shapes,
        ):
            configs.append(
                BenchmarkConfig(
                    BenchmarkConfigParams(
                        quantization=quant_config,
                        sparsity=sparse_config,
                        params=params,
                        shape_name=shape_name,
                        shape=shape,
                        output_dir=output_dir,
                        benchmark_mode=benchmark_mode,
                    )
                )
            )
    return configs


def run_inference_benchmarks_from_config(configs: List[BenchmarkConfig]) -> None:
    """Run benchmarks using configurations from YAML file"""
    from benchmarks.microbenchmarks.benchmark_inference import run as run_inference

    results = []
    print("----------------- RUNNING BENCHMARKS FOR INFERENCE -----------------------")
    for config in configs:
        print("----------------------------------------")
        try:
            print(
                f"Running: {config.name} for Quantization: {config.quantization} and Sparsity: {config.sparsity} for {config.shape_name}: {config.m, config.k, config.n}"
            )
            result = run_inference(config)  # Pass the config object directly
            if result is not None:  # Only add successful results
                results.append(result)
        except Exception as e:
            print(f"Error running benchmark {config.name} with error: {e}")
            continue

    # Add results to csv if there are any
    if results:
        generate_results_csv(results, configs[0].output_dir)
        # Print results
        print_results(results)
    else:
        print("No benchmark results were collected. All benchmarks failed.")

    # TODO: Process results: Speedups:
    # 1. For different shapes for same model and quantization
    # 2. For different quantizations for same model and shape
    # 3. For different models for same quantization


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmarks from config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration file",
    )
    # TODO: Add support for args to override config values and run smaller benchmarks
    args = parser.parse_args()

    configs = load_benchmark_configs(cli_args=args)
    # Run benchmarks
    if configs[0].benchmark_mode == "inference":
        run_inference_benchmarks_from_config(configs)
    elif configs[0].benchmark_mode == "training":
        print("Training mode not implemented yet")
    else:
        raise ValueError(
            f"Invalid benchmark mode: {configs[0].benchmark_mode}, choose from inference or training"
        )

    # TODO: Add support for args to override config values and run smaller benchmarks
