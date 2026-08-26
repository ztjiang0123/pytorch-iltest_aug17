# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, TypeVar

import diffusers
import fire
import lpips
import numpy as np
import torch
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image, ImageDraw, ImageFont
from utils import string_to_config

import torchao
from torchao.quantization import (
    FqnToConfig,
    quantize_,
)

# Type variables for better type hinting
T = TypeVar("T")

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "benchmarks/data/flux_eval"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_pipeline_architecture(pipe):
    """
    Print the PyTorch model architecture for each component of a diffusion pipeline.

    Args:
        pipe: The diffusion pipeline to inspect
    """
    print("\n" + "=" * 80)
    print("DIFFUSION PIPELINE COMPONENTS")
    print("=" * 80)

    # Iterate through components specified in the model config
    total_params = 0
    components = ["vae", "transformer", "text_encoder", "text_encoder_2"]
    for idx, component_name in enumerate(components, 1):
        component = getattr(pipe, component_name)
        print("\n" + "-" * 80)
        print(f"{idx}. {component_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(component)
        param_count = sum(p.numel() for p in component.parameters())
        print(f"\n{component_name} Parameter Count: {param_count:,}")
        total_params += param_count

    print("\n" + "-" * 80)
    print("Other Components (Non-Neural)")
    print("-" * 80)
    print(f"Tokenizer: {type(pipe.tokenizer).__name__}")
    print(f"Scheduler: {type(pipe.scheduler).__name__}")

    print("\n" + "=" * 80)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 80 + "\n")


def generate_image(
    pipe,
    prompt: str,
    seed: int,
    device: str,
    num_inference_steps: int,
    batch_size: int = 1,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)

    prompts = [prompt] * batch_size
    image = pipe(
        prompt=prompts,
        num_inference_steps=num_inference_steps,  # can tweak for speed vs quality
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize (if needed) to a fixed size so LPIPS sees consistent shapes
    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


def create_comparison_image(
    baseline_img: Image.Image,
    modified_img: Image.Image,
    lpips_score: float,
    prompt: str = None,
    margin_top: int = 80,
) -> Image.Image:
    """
    Create a comparison image by stacking two images horizontally with a top margin
    and overlaying the prompt text and LPIPS score.

    Args:
        baseline_img: The baseline image
        modified_img: The modified/quantized image
        lpips_score: The LPIPS score between the two images
        prompt: Optional prompt text to display at the top
        margin_top: Height of the top margin for text (default 80 to fit prompt + LPIPS)
    """
    # Get dimensions
    width1, height1 = baseline_img.size
    width2, height2 = modified_img.size

    # Create new image with top margin
    total_width = width1 + width2
    total_height = max(height1, height2) + margin_top

    # Create composite image with dark gray background for margin
    composite = Image.new("RGB", (total_width, total_height), color=(50, 50, 50))

    # Paste the two images side by side, offset by margin_top
    composite.paste(baseline_img, (0, margin_top))
    composite.paste(modified_img, (width1, margin_top))

    # Add text overlay with prompt and LPIPS score
    draw = ImageDraw.Draw(composite)

    # Try to use reasonable font sizes, fallback to default if truetype fails
    try:
        prompt_font = ImageFont.truetype("arial.ttf", 20)
        lpips_font = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        prompt_font = ImageFont.load_default()
        lpips_font = ImageFont.load_default()

    # Draw prompt text at the top if provided
    y_offset = 5
    if prompt:
        # Wrap prompt text if it's too long
        max_width = total_width - 20  # 10px padding on each side
        prompt_lines = []
        words = prompt.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=prompt_font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    prompt_lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            prompt_lines.append(" ".join(current_line))

        # Draw each line of the prompt
        for line in prompt_lines:
            bbox = draw.textbbox((0, 0), line, font=prompt_font)
            text_width = bbox[2] - bbox[0]
            text_x = (total_width - text_width) // 2
            draw.text((text_x, y_offset), line, fill=(200, 200, 200), font=prompt_font)
            y_offset += (bbox[3] - bbox[1]) + 2  # line height + small gap

    # Format the LPIPS text
    lpips_text = f"LPIPS: {lpips_score:.4f}"

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), lpips_text, font=lpips_font)
    text_width = bbox[2] - bbox[0]

    # Center the LPIPS text horizontally, place it below the prompt
    text_x = (total_width - text_width) // 2
    text_y = y_offset + 5  # small gap after prompt

    # Draw LPIPS text in white
    draw.text((text_x, text_y), lpips_text, fill=(255, 255, 255), font=lpips_font)

    return composite


def create_combined_comparison_image(
    comparison_images: list[Image.Image],
) -> Image.Image:
    """
    Stack multiple comparison images vertically into a single combined image.

    Args:
        comparison_images: List of comparison images to stack vertically

    Returns:
        Combined image with all comparisons stacked vertically
    """
    if not comparison_images:
        raise ValueError("comparison_images list cannot be empty")

    # Calculate dimensions
    total_height = sum(img.size[1] for img in comparison_images)
    max_width = max(img.size[0] for img in comparison_images)

    # Create combined image
    combined_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for comp_img in comparison_images:
        combined_img.paste(comp_img, (0, y_offset))
        y_offset += comp_img.size[1]

    return combined_img


def pil_to_lpips_tensor(img: Image.Image, device: str):
    """
    Convert a PIL Image to a tensor suitable for LPIPS computation.

    Args:
        img: PIL Image to convert
        device: Device to place the tensor on ('cuda' or 'cpu')

    Returns:
        Tensor in shape (1, 3, H, W) normalized to [-1, 1]
    """
    t = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(img.size[1], img.size[0], 3)
                .numpy()
            )
        ).float()
        / 255.0
    )  # [0, 1]
    # reshape to (1, 3, H, W) and scale to [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = t * 2.0 - 1.0
    return t.to(device)


from torch.utils._pytree import tree_map_only


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.

    This wrapper is useful when working with torch.compile to prevent errors
    related to in-place operations on tensors.

    Args:
        f: The function whose CUDA tensor outputs should be cloned

    Returns:
        A wrapped function that clones any CUDA tensor outputs
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped


def apply_torch_compile(pipe, torch_compile_mode: str = "default"):
    """Apply torch.compile to the transformer blocks in-place."""
    for block in pipe.transformer.transformer_blocks:
        block.forward = clone_output_wrapper(
            torch.compile(block.forward, mode=torch_compile_mode)
        )
    for block in pipe.transformer.single_transformer_blocks:
        block.forward = clone_output_wrapper(
            torch.compile(block.forward, mode=torch_compile_mode)
        )


@dataclass
class RunContext:
    """Shared run-scoped state, threaded through the mode helpers.

    Bundling these values keeps each helper's signature small instead of
    passing the same dozen arguments around individually.
    """

    mode: str
    model: str
    quant_config_str: str
    num_inference_steps: int
    prompts_dataset: str
    use_compile: bool
    torch_compile_mode: str
    use_deterministic_algorithms: bool
    batch_size: int
    cache_baseline_images: bool
    print_model: bool
    perf_n_iter: int
    num_prompts: int
    debug_prompt: str
    num_gpus_used: int
    local_rank: int
    world_size: int
    output_dir: str = ""
    cache_dir: str = ""
    device: str = ""
    pipe: object = None
    loss_fn: object = None
    prompts_to_use: list = field(default_factory=list)
    my_prompts: list = field(default_factory=list)

    @property
    def rank_prefix(self):
        return f"[Rank {self.local_rank}/{self.world_size}]"


def _read_rank_lpips(csv_path, rank, num_gpus_used):
    """Parse one rank's summary CSV into {global_prompt_idx: lpips_value}."""
    result = {}
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if len(row) != 2 or not row[0].startswith("lpips_prompt_"):
                continue
            local_idx = int(row[0].split("_")[-1])
            # Calculate global prompt index
            global_idx = rank + local_idx * num_gpus_used
            result[global_idx] = float(row[1])
    return result


def _collect_lpips_data(output_dir, quant_config_str, num_gpus_used):
    """Read every rank's CSV and merge into a single global lpips dict."""
    all_lpips_data = {}  # dict mapping global prompt idx to lpips value
    for rank in range(num_gpus_used):
        csv_path = os.path.join(
            output_dir,
            f"summary_stats_prompt_mode_accuracy_config_str_{quant_config_str}_rank_{rank}.csv",
        )
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found for rank {rank}: {csv_path}")
            continue
        print(f"Reading {csv_path}")
        all_lpips_data.update(_read_rank_lpips(csv_path, rank, num_gpus_used))
    return all_lpips_data


def _write_aggregated_csv(path, all_lpips_data, sorted_prompts, stats, num_gpus_used):
    """Write the combined LPIPS results across all ranks to ``path``."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mode", "aggregated"])
        writer.writerow(["num_gpus_used", num_gpus_used])
        writer.writerow(["total_prompts", stats["total"]])
        writer.writerow(["average_lpips", f"{stats['avg']:.4f}"])
        writer.writerow(["max_lpips", f"{stats['max']:.4f}"])
        writer.writerow(["min_lpips", f"{stats['min']:.4f}"])
        # Write individual LPIPS values in global prompt order
        for global_idx in sorted_prompts:
            writer.writerow(
                [f"lpips_prompt_{global_idx}", f"{all_lpips_data[global_idx]:.4f}"]
            )


def _aggregate_accuracy(ctx):
    """Aggregate per-rank LPIPS CSV files into a single summary CSV."""
    num_gpus_used = ctx.num_gpus_used
    if num_gpus_used is None:
        raise ValueError("num_gpus_used is required for aggregate_accuracy mode")

    # Only run on rank 0
    if ctx.local_rank != 0:
        print(f"{ctx.rank_prefix} Skipping aggregate_accuracy mode (only rank 0 runs)")
        return

    print(f"Aggregating LPIPS results from {num_gpus_used} GPU runs")

    output_dir = os.path.join(OUTPUT_DIR, ctx.model)
    all_lpips_data = _collect_lpips_data(
        output_dir, ctx.quant_config_str, num_gpus_used
    )
    if not all_lpips_data:
        print("Error: No LPIPS data found in CSV files")
        return

    # Sort by global prompt index
    sorted_prompts = sorted(all_lpips_data.keys())
    sorted_lpips_values = [all_lpips_data[idx] for idx in sorted_prompts]
    stats = {
        "total": len(sorted_lpips_values),
        "avg": sum(sorted_lpips_values) / len(sorted_lpips_values),
        "max": max(sorted_lpips_values),
        "min": min(sorted_lpips_values),
    }

    print("=" * 80)
    print("Aggregated LPIPS Results:")
    print(f"  Total prompts: {stats['total']}")
    print(f"  Average LPIPS: {stats['avg']:.4f}")
    print(f"  Max LPIPS: {stats['max']:.4f}")
    print(f"  Min LPIPS: {stats['min']:.4f}")
    print(f"  All values: {[f'{v:.4f}' for v in sorted_lpips_values]}")
    print("=" * 80)

    aggregated_csv_path = os.path.join(
        output_dir,
        f"summary_stats_prompt_mode_accuracy_config_str_{ctx.quant_config_str}_aggregated.csv",
    )
    _write_aggregated_csv(
        aggregated_csv_path, all_lpips_data, sorted_prompts, stats, num_gpus_used
    )
    print(f"Aggregated results saved to {aggregated_csv_path}")


def _print_run_config(ctx):
    """Print the resolved run configuration for a rank."""
    prefix = ctx.rank_prefix
    print(f"{prefix} {torch.__version__=}")
    print(f"{prefix} {torchao.__version__=}")
    print(f"{prefix} {diffusers.__version__=}")
    print(f"{prefix} mode={ctx.mode}")
    print(f"{prefix} Model: {ctx.model}")
    print(f"{prefix} Quant config: {ctx.quant_config_str}")
    print(f"{prefix} num_inference_steps: {ctx.num_inference_steps}")
    print(f"{prefix} prompts_dataset: {ctx.prompts_dataset}")
    print(f"{prefix} use_compile: {ctx.use_compile}")
    print(f"{prefix} torch_compile_mode: {ctx.torch_compile_mode}")
    print(f"{prefix} use_deterministic_algorithms={ctx.use_deterministic_algorithms}")
    print(f"{prefix} batch_size={ctx.batch_size}")
    print(f"{prefix} cache_baseline_images={ctx.cache_baseline_images}")


def _resolve_prompts(ctx):
    """Load prompts, then shard/limit them for this rank."""
    if ctx.debug_prompt is None:
        dataset = load_dataset(ctx.prompts_dataset, split="train")
        all_prompts = [item["Prompts"] for item in dataset]
    else:
        all_prompts = [ctx.debug_prompt]

    # Limit prompts for debugging if requested
    prompts_to_use = (
        all_prompts if ctx.num_prompts is None else all_prompts[: ctx.num_prompts]
    )

    # Shard the prompts across GPUs (each rank processes every world_size-th prompt)
    if ctx.mode == "accuracy":
        my_prompts = prompts_to_use[ctx.local_rank :: ctx.world_size]
        print(
            f"{ctx.rank_prefix} Processing {len(my_prompts)} prompts "
            f"out of {len(prompts_to_use)} total"
        )
    else:
        # For performance modes, don't shard - only rank 0 runs
        my_prompts = prompts_to_use if ctx.local_rank == 0 else []

    return prompts_to_use, my_prompts


def _generate_baseline_images(ctx):
    """Generate (or load from cache) baseline images for accuracy mode."""
    baseline_data = []  # List of (prompt_idx, prompt, baseline_img, baseline_t)
    baseline_times = []
    for local_idx, prompt in enumerate(ctx.my_prompts):
        # Calculate global prompt index
        global_idx = ctx.local_rank + local_idx * ctx.world_size
        prompt_idx = f"prompt_{global_idx}"
        img_path = os.path.join(ctx.cache_dir, f"{prompt_idx}.png")
        if ctx.cache_baseline_images and os.path.exists(img_path):
            print(
                f"{ctx.rank_prefix} Loading baseline image for prompt "
                f"{prompt_idx}: {prompt} from cache"
            )
            t0 = time.time()
            baseline_img = Image.open(img_path)
            t1 = time.time()
        else:
            print(
                f"{ctx.rank_prefix} Generating baseline image for "
                f"prompt {prompt_idx}: {prompt}"
            )
            t0 = time.time()
            baseline_img = generate_image(
                ctx.pipe, prompt, RANDOM_SEED, ctx.device, ctx.num_inference_steps
            )
            t1 = time.time()
            baseline_img.save(img_path)
        baseline_t = pil_to_lpips_tensor(baseline_img, ctx.device)
        baseline_data.append((prompt_idx, prompt, baseline_img, baseline_t))
        baseline_times.append(t1 - t0)
    return baseline_data, baseline_times


def _measure_generation_perf(ctx, prompt):
    """Warm up compile, then time ``perf_n_iter`` generations. Returns times list."""
    # warm up compile
    _ = generate_image(
        ctx.pipe,
        prompt,
        RANDOM_SEED,
        ctx.device,
        ctx.num_inference_steps,
        batch_size=ctx.batch_size,
    )
    times = []
    for _ in range(ctx.perf_n_iter):
        t0 = time.time()
        _ = generate_image(
            ctx.pipe,
            prompt,
            RANDOM_SEED,
            ctx.device,
            ctx.num_inference_steps,
            batch_size=ctx.batch_size,
        )
        t1 = time.time()
        times.append(t1 - t0)
    return times


def _quantize_transformer(ctx):
    """Build an FqnToConfig from a heuristic and quantize the transformer.

    Returns the dict of quantized fqns -> config that was applied.
    """
    # Inspect Linear layers in main component
    component_linear_fqns_and_weight_shapes = []
    for fqn, module in ctx.pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            if ctx.print_model:
                print(f"  {fqn}: {weight_shape}")
            component_linear_fqns_and_weight_shapes.append([fqn, weight_shape])

    config_obj = string_to_config(ctx.quant_config_str)

    # Create FqnToConfig mapping
    fqn_to_config_dict = {}
    for fqn, weight_shape in component_linear_fqns_and_weight_shapes:
        if _should_quantize_fqn(fqn, weight_shape):
            fqn_to_config_dict[fqn] = config_obj
    fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)

    # Quantize the main component using this config
    quantize_(ctx.pipe.transformer, fqn_to_config, filter_fn=None)
    return fqn_to_config_dict


def _should_quantize_fqn(fqn, weight_shape):
    """Hand-crafted heuristic: skip embeddings, the last two layers, small weights.

    Activations for ``norm.linear`` have shape [batch_size, 3072], too small to
    see speedups from activation quantization.
    """
    if "embed" in fqn:
        return False
    if fqn in ("norm_out.linear", "proj_out"):
        return False
    if "norm.linear" in fqn:
        return False
    if weight_shape[0] < 1024 or weight_shape[1] < 1024:
        return False
    return True


def _run_accuracy_generation(ctx, baseline_data):
    """Generate quantized images, compute LPIPS, and save comparison images.

    Returns (lpips_values, times).
    """
    print(f"{ctx.rank_prefix} Generating images with quantized model for all prompts")
    lpips_values = []
    comparison_images = []
    times = []
    for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
        print(f"{ctx.rank_prefix} Generating image for {prompt_idx}")
        t0 = time.time()
        modified_img = generate_image(
            ctx.pipe, prompt, RANDOM_SEED, ctx.device, ctx.num_inference_steps
        )
        t1 = time.time()
        times.append(t1 - t0)

        # Compute LPIPS for fully quantized model
        modified_t = pil_to_lpips_tensor(modified_img, ctx.device)
        with torch.no_grad():
            lpips_value = ctx.loss_fn(baseline_t, modified_t).item()
        lpips_values.append(lpips_value)
        print(
            f"{ctx.rank_prefix} LPIPS distance "
            f"(full quantization, {prompt_idx}): {lpips_value:.4f}"
        )

        # Create and save comparison image
        print(f"{ctx.rank_prefix} Creating comparison image")
        comparison_img = create_comparison_image(
            baseline_img, modified_img, lpips_value, prompt=prompt
        )
        comparison_images.append(comparison_img)
        comparison_path = os.path.join(
            ctx.output_dir,
            f"comparison_prompt_mode_full_quant_config_str_{ctx.quant_config_str}_{prompt_idx}_rank_{ctx.local_rank}.png",
        )
        comparison_img.save(comparison_path)
        print(f"{ctx.rank_prefix} Saved comparison image to: {comparison_path}")

    # Create combined image with all comparisons stacked vertically
    combined_img = create_combined_comparison_image(comparison_images)
    combined_path = os.path.join(
        ctx.output_dir,
        f"comparison_prompt_mode_full_quant_config_str_{ctx.quant_config_str}_combined_rank_{ctx.local_rank}.png",
    )
    combined_img.save(combined_path)
    print(f"{ctx.rank_prefix} Saved combined comparison image to: {combined_path}")
    return lpips_values, times


def _write_summary_rows(writer, rows):
    """Write an ordered list of summary rows.

    Each row is either a scalar ``("scalar", key, value)`` or an enumerated
    ``("series", prefix, values)`` that expands to ``{prefix}{idx}`` rows. This
    single writer serves every mode, so accuracy/performance layouts stay in
    their callers as data rather than as near-duplicate writer functions.
    """
    for row in rows:
        kind = row[0]
        if kind == "scalar":
            _, key, value = row
            writer.writerow([key, value])
        else:  # "series"
            _, prefix, values = row
            for idx, val in enumerate(values):
                writer.writerow([f"{prefix}{idx}", f"{val:.4f}"])


def _save_summary_csv(ctx, mode, rows):
    """Write the per-rank summary stats CSV for accuracy/performance modes.

    ``rows`` is the ordered, mode-specific body built by the caller; the common
    header rows are written here.
    """
    summary_csv_path = os.path.join(
        ctx.output_dir,
        f"summary_stats_prompt_mode_{mode}_config_str_{ctx.quant_config_str}_rank_{ctx.local_rank}.csv",
    )
    header = [
        ("scalar", "metric", "value"),
        ("scalar", "mode", mode),
        ("scalar", "local_rank", ctx.local_rank),
        ("scalar", "world_size", ctx.world_size),
    ]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        _write_summary_rows(writer, header + rows)
    print(f"{ctx.rank_prefix} Summary stats saved to {summary_csv_path}\n\n")


def _run_accuracy_mode(ctx):
    """Generate baseline + quantized images, report LPIPS, and save the summary."""
    # note: never compile for baseline images
    baseline_data, baseline_times = _generate_baseline_images(ctx)

    fqn_to_config_dict = _quantize_transformer(ctx)
    if ctx.use_compile:
        apply_torch_compile(ctx.pipe, ctx.torch_compile_mode)
    if ctx.print_model:
        print_pipeline_architecture(ctx.pipe)

    lpips_values, times = _run_accuracy_generation(ctx, baseline_data)

    avg_lpips = sum(lpips_values) / len(lpips_values)
    max_lpips = max(lpips_values)
    min_lpips = min(lpips_values)
    avg_baseline_time = sum(baseline_times) / len(baseline_times)
    avg_quant_time = sum(times) / len(times)

    print("=" * 80)
    print("Test Mode Summary:")
    print(f"  Total Linear layers quantized: {len(fqn_to_config_dict)}")
    print(f"  Prompts tested: {len(baseline_data)}")
    print("")
    print("LPIPS Results:")
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Max LPIPS: {max_lpips:.4f}")
    print(f"  Min LPIPS: {min_lpips:.4f}")
    print(f"  All values: {[f'{v:.4f}' for v in lpips_values]}")
    print("=" * 80)
    print(f"Baseline times: {baseline_times}")
    print(f"Quantized times: {times}")
    print(f"Average baseline time: {avg_baseline_time:.4f}s")
    print(f"Average quantized time: {avg_quant_time:.4f}s")

    _save_summary_csv(
        ctx,
        "accuracy",
        [
            ("scalar", "total_linear_layers_quantized", len(fqn_to_config_dict)),
            ("scalar", "prompts_tested", len(baseline_data)),
            ("scalar", "average_lpips", f"{avg_lpips:.4f}"),
            ("scalar", "max_lpips", f"{max_lpips:.4f}"),
            ("scalar", "min_lpips", f"{min_lpips:.4f}"),
            ("series", "lpips_prompt_", lpips_values),
            ("scalar", "average_baseline_time", f"{avg_baseline_time:.4f}"),
            ("scalar", "average_quantized_time", f"{avg_quant_time:.4f}"),
        ],
    )


def _run_performance_mode(ctx, quantized):
    """Measure generation performance, optionally after quantizing the transformer.

    ``quantized`` selects performance_quant (True) vs performance_hp (False).
    """
    mode = "performance_quant" if quantized else "performance_hp"

    num_quantized = 0
    if quantized:
        num_quantized = len(_quantize_transformer(ctx))
        if ctx.print_model:
            print_pipeline_architecture(ctx.pipe)
    if ctx.use_compile:
        apply_torch_compile(ctx.pipe, ctx.torch_compile_mode)

    times = []
    # Only rank 0 runs performance measurements.
    if ctx.local_rank == 0:
        times = _measure_generation_perf(ctx, ctx.prompts_to_use[0])

    print("=" * 80)
    print("Test Mode Summary:")
    if quantized:
        print(f"  Total Linear layers quantized: {num_quantized}")

    avg_time = sum(times) / len(times)
    label = "Quantized Model" if quantized else "High Precision (Baseline)"
    print(f"{label} Times: {times}")
    print(f"Average time: {avg_time:.4f}s")

    perf_rows = [
        ("scalar", "perf_n_iter", ctx.perf_n_iter),
        ("scalar", "batch_size", ctx.batch_size),
        ("scalar", "average_time", f"{avg_time:.4f}"),
        ("series", "time_", times),
    ]
    if quantized:
        # performance_quant records the layer count right after the header,
        # matching the accuracy layout; performance_hp omits it.
        perf_rows.insert(
            0, ("scalar", "total_linear_layers_quantized", num_quantized)
        )
    _save_summary_csv(ctx, mode, perf_rows)


@torch.inference_mode()
def run(
    mode: str = "accuracy",
    num_prompts: int = None,
    num_inference_steps: int = 4,
    quant_config_str: str = "float8_rowwise",
    use_compile: bool = False,
    torch_compile_mode: str = "default",
    debug_prompt: str | None = None,
    print_model: bool = False,
    cache_baseline_images: bool = False,
    perf_n_iter: int = 10,
    batch_size: int = 1,
    use_deterministic_algorithms: bool = False,
    num_gpus_used: int = None,
):
    """
    A performance and accuracy eval script for quantizing flux-1.schnell:

      1. load flux-1.schnell model
      2a. for mode == 'accuracy':
        2. run it on a prompts dataset and save the images
        3. quantize the model, run it on the same dataset and save the images
        4. report accuracy difference (using LPIPS) between 2 and 3
      2b. for mode == 'performance_hp':
        2. run it on a debug prompt and measure performance (high precision / baseline)
      2c. for mode == 'performance_quant':
        2. quantize the model, run it on a debug prompt and measure performance
      2d. for mode == 'aggregate_accuracy':
        2. load CSV files from multiple GPU runs and aggregate LPIPS results

    Args:
        mode: 'accuracy', 'performance_hp', 'performance_quant', or 'aggregate_accuracy'
        num_prompts: Optional limit on number of prompts to use (for debugging)
        num_inference_steps: Number of passes through the transformer,
          default 4 for flux-1.schnell. Can set to 1 for speeding up debugging.
        quant_config_str: Quantization config to use ('float8_rowwise'). Default: 'float8_rowwise'
        use_compile: if true, uses torch.compile
        torch_compile_mode: mode to use torch.compile with
        debug_prompt: if specified, use this prompt instead of the drawbench dataset
        print_model: if True, prints model architecture
        cache_baseline_images: if specified, baseline images are read from cache (disk)
          instead of regenerated, if available. This is useful to make eval runs faster
          if we know the baseline is not changing.
        perf_n_iter: number of measurements to take for measuring performance
        batch_size: batch size for performance_hp and performance_quant modes (default 1)
        use_deterministic_algorithms: if True, sets torch.use_deterministic_algorithms(True)
        num_gpus_used: For 'aggregate_accuracy' mode, the number of GPUs that were used
          to generate the data. Required for aggregate_accuracy mode.
    """
    # Distributed setup for torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # TODO(future): maybe support other models and datasets
    # model = "black-forest-labs/FLUX.1-dev"
    model = "black-forest-labs/FLUX.1-schnell"
    prompts_dataset = "sayakpaul/drawbench"
    if debug_prompt is not None:
        prompts_dataset = "debug"

    if use_deterministic_algorithms:
        # this is needed to make torch.compile be deterministic with flux-1.schnell
        torch.use_deterministic_algorithms(True)

    ctx = RunContext(
        mode=mode,
        model=model,
        quant_config_str=quant_config_str,
        num_inference_steps=num_inference_steps,
        prompts_dataset=prompts_dataset,
        use_compile=use_compile,
        torch_compile_mode=torch_compile_mode,
        use_deterministic_algorithms=use_deterministic_algorithms,
        batch_size=batch_size,
        cache_baseline_images=cache_baseline_images,
        print_model=print_model,
        perf_n_iter=perf_n_iter,
        num_prompts=num_prompts,
        debug_prompt=debug_prompt,
        num_gpus_used=num_gpus_used,
        local_rank=local_rank,
        world_size=world_size,
    )

    _print_run_config(ctx)

    assert mode in (
        "accuracy",
        "performance_hp",
        "performance_quant",
        "aggregate_accuracy",
    )
    assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
    if mode in ("accuracy", "aggregate_accuracy"):
        assert batch_size == 1, (
            f"batch_size must be 1 for {mode} mode, got {batch_size}"
        )

    # Handle aggregate_accuracy mode separately (reads existing CSVs, no model load)
    if mode == "aggregate_accuracy":
        _aggregate_accuracy(ctx)
        return

    # Create model-specific output directory
    ctx.output_dir = os.path.join(OUTPUT_DIR, model)
    os.makedirs(ctx.output_dir, exist_ok=True)
    ctx.cache_dir = os.path.join(ctx.output_dir, "baseline_cache")
    os.makedirs(ctx.cache_dir, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load model
    ctx.device = f"cuda:{local_rank}"  # Each process uses its assigned GPU
    # TODO(future): support FqnToConfig in diffusers, so we can use it here
    # and easily save a quantized checkpoint to disk
    pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=True)

    print(f"{ctx.rank_prefix} Moving model to device {ctx.device}")
    ctx.pipe = pipe.to(ctx.device)

    ctx.loss_fn = lpips.LPIPS(net="vgg").to(ctx.device)

    # -----------------------------
    # 2. Dispatch to the selected mode
    # -----------------------------
    ctx.prompts_to_use, ctx.my_prompts = _resolve_prompts(ctx)

    if mode == "accuracy":
        _run_accuracy_mode(ctx)
    elif mode == "performance_hp":
        _run_performance_mode(ctx, quantized=False)
    elif mode == "performance_quant":
        _run_performance_mode(ctx, quantized=True)


if __name__ == "__main__":
    fire.Fire(run)
