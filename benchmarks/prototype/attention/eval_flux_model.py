# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark attention backends on FLUX.1-schnell.

Compares backends using LPIPS (perceptual similarity) on DrawBench prompts.

Usage:
    python eval_flux_model.py --baseline fa3 --test fa3_fp8 --debug_prompt "A red car"
"""

import argparse
import gc
import random
from dataclasses import dataclass
from typing import Optional

import lpips
import numpy as np
import torch
import torch._dynamo
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image
from torch.nn.attention import (
    SDPBackend,
    activate_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
)

from torchao.prototype.attention import (
    AttentionBackend,
    HadamardMode,
    apply_low_precision_attention,
)

BACKENDS = {
    "fa2": {
        "flash_impl": None,
        "fp8": False,
        "sdpa_backend": SDPBackend.FLASH_ATTENTION,
    },
    "fa3": {
        "flash_impl": "FA3",
        "fp8": False,
        "sdpa_backend": SDPBackend.FLASH_ATTENTION,
    },
    "fa3_fp8": {
        "flash_impl": "FA3",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA3,
    },
    "fa3_fp8_hadamard": {
        "flash_impl": "FA3",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA3,
        "hadamard": HadamardMode.QKV,
    },
    "fa3_fp8_hadamard_v": {
        "flash_impl": "FA3",
        "fp8": True,
        "fp8_backend": AttentionBackend.FP8_FA3,
        "hadamard": HadamardMode.V_ONLY,
    },
}

IMAGE_SIZE = (512, 512)  # (width, height) - resize for consistent LPIPS
RANDOM_SEED = 42
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


@dataclass
class BenchmarkConfig:
    """Settings shared across every phase of a benchmark run."""

    device: str
    num_inference_steps: int
    height: int
    width: int
    warmup_iters: int
    compile: bool


@dataclass
class GenSettings:
    """Per-backend generation settings resolved by ``setup_backend``."""

    flash_impl: Optional[str]
    sdpa_backend: Optional[SDPBackend]


@dataclass
class Runner:
    """Loaded model state reused across the baseline and test phases."""

    pipe: object
    orig_transformer: object
    loss_fn: object
    config: BenchmarkConfig


def cleanup_gpu():
    """Free GPU memory between benchmark phases."""
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def setup_backend(
    pipe,
    backend_name,
    compile_flag,
    orig_transformer,
):
    """Set up a backend and return (flash_impl, sdpa_backend)."""
    cfg = BACKENDS[backend_name]
    pipe.transformer = orig_transformer

    if cfg["fp8"]:
        print(f"Applying low-precision FP8 attention ({backend_name})...")
        pipe.transformer = apply_low_precision_attention(
            pipe.transformer,
            backend=cfg["fp8_backend"],
            hadamard=cfg.get("hadamard", HadamardMode.NONE),
        )
        if compile_flag:
            print(f"Compiling transformer with torch.compile ({backend_name})...")
            pipe.transformer = torch.compile(pipe.transformer)
        return cfg["flash_impl"], None
    else:
        if compile_flag:
            print(f"Compiling transformer with torch.compile ({backend_name})...")
            pipe.transformer = torch.compile(pipe.transformer)
        return cfg["flash_impl"], cfg.get("sdpa_backend")


def pil_to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert a PIL Image to a tensor suitable for LPIPS computation."""
    t = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(img.size[1], img.size[0], 3)
                .numpy()
            )
        ).float()
        / 255.0
    )
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t.to(device)


def generate_image(
    pipe,
    prompt: str,
    seed: int,
    device: str,
    num_inference_steps: int,
    height: int = 2048,
    width: int = 2048,
    flash_impl: Optional[str] = None,
    sdpa_backend: Optional[SDPBackend] = None,
) -> Image.Image:
    """Generate an image from a prompt with deterministic seed."""
    generator = torch.Generator(device=device).manual_seed(seed)

    # For BF16 backends, force the correct SDPA backend on the transformer
    # only (not the VAE, whose head_dim=512 exceeds flash/cuDNN limits and
    # needs the math backend). FP8 backends call their ops directly and
    # don't need this.
    orig_forward = None
    if sdpa_backend is not None:
        orig_forward = pipe.transformer.forward

        def _forced_backend_forward(*args, **kwargs):
            with sdpa_kernel(sdpa_backend):
                return orig_forward(*args, **kwargs)

        pipe.transformer.forward = _forced_backend_forward

    if flash_impl:
        activate_flash_attention_impl(flash_impl)
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
    finally:
        if orig_forward is not None:
            pipe.transformer.forward = orig_forward
        if flash_impl:
            restore_flash_attention_impl()

    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


def _load_prompts(debug_prompt: Optional[str], num_prompts: int) -> list:
    """Return the list of prompts to benchmark."""
    if debug_prompt is not None:
        print(f"Using debug prompt: {debug_prompt}")
        return [debug_prompt]

    print("Loading DrawBench dataset...")
    dataset = load_dataset("sayakpaul/drawbench", split="train")
    all_prompts = [item["Prompts"] for item in dataset]
    prompts = all_prompts[:num_prompts]
    print(
        f"Using {len(prompts)} prompts from DrawBench "
        f"(total available: {len(all_prompts)})"
    )
    return prompts


def _load_runner(config: BenchmarkConfig) -> Runner:
    """Load the FLUX pipeline and the LPIPS loss model into a ``Runner``."""
    print(f"\nLoading FLUX.1-schnell from {MODEL_ID}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(config.device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading LPIPS model (VGG)...")
    loss_fn = lpips.LPIPS(net="vgg").to(config.device)

    orig_transformer = pipe.transformer

    if config.compile:
        pipe.vae.decode = torch.compile(pipe.vae.decode)

    return Runner(
        pipe=pipe,
        orig_transformer=orig_transformer,
        loss_fn=loss_fn,
        config=config,
    )


def _timed_generate(pipe, prompt: str, config: BenchmarkConfig, gen: GenSettings):
    """Generate an image and return ``(image, elapsed_ms)``."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    image = generate_image(
        pipe,
        prompt,
        RANDOM_SEED,
        config.device,
        config.num_inference_steps,
        height=config.height,
        width=config.width,
        flash_impl=gen.flash_impl,
        sdpa_backend=gen.sdpa_backend,
    )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return image, elapsed_ms


def _warmup(
    pipe,
    backend_name: str,
    warmup_prompt: str,
    config: BenchmarkConfig,
    gen: GenSettings,
) -> None:
    """Run warmup iterations for a backend."""
    print(f"Warming up {backend_name} with {config.warmup_iters} iterations...")
    for i in range(config.warmup_iters):
        _ = _timed_generate(pipe, warmup_prompt, config, gen)
        print(f"  Warmup {i + 1}/{config.warmup_iters} complete")


def _prepare_phase(runner: Runner, backend: str, phase_num: int) -> GenSettings:
    """Print the phase header and set up ``backend``, returning its settings."""
    print("\n" + "-" * 80)
    print(f"Phase {phase_num}: Generating images ({backend})")
    print("-" * 80)

    flash_impl, sdpa_backend = setup_backend(
        runner.pipe,
        backend,
        runner.config.compile,
        runner.orig_transformer,
    )
    return GenSettings(flash_impl=flash_impl, sdpa_backend=sdpa_backend)


def _run_baseline_phase(runner: Runner, baseline_backend: str, prompts: list):
    """Generate baseline images. Returns ``(baseline_data, avg_baseline_ms)``.

    ``baseline_data`` is a list of ``(prompt, cpu_tensor)`` tuples.
    """
    config = runner.config
    gen = _prepare_phase(runner, baseline_backend, 1)
    _warmup(runner.pipe, baseline_backend, prompts[0], config, gen)

    baseline_data = []
    baseline_times_ms = []

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{len(prompts)}] {baseline_backend}: {prompt[:50]}...")
        baseline_img, elapsed_ms = _timed_generate(runner.pipe, prompt, config, gen)

        baseline_tensor = pil_to_lpips_tensor(baseline_img, config.device)
        # Store tensors on CPU to free GPU memory for the test phase.
        baseline_data.append((prompt, baseline_tensor.cpu()))
        baseline_times_ms.append(elapsed_ms)

    avg_baseline_ms = sum(baseline_times_ms) / len(baseline_times_ms)
    print(
        f"\n{baseline_backend} complete. Avg time per image: {avg_baseline_ms:.1f} ms"
    )
    return baseline_data, avg_baseline_ms


def _run_test_phase(runner: Runner, test_backend: str, baseline_data: list):
    """Generate test images and compute LPIPS. Returns ``(lpips_values, avg_test_ms)``."""
    config = runner.config
    gen = _prepare_phase(runner, test_backend, 2)
    _warmup(runner.pipe, test_backend, baseline_data[0][0], config, gen)

    lpips_values = []
    test_times_ms = []

    for idx, (prompt, baseline_tensor_cpu) in enumerate(baseline_data):
        print(f"[{idx + 1}/{len(baseline_data)}] {test_backend}: {prompt[:50]}...")
        test_img, elapsed_ms = _timed_generate(runner.pipe, prompt, config, gen)
        test_times_ms.append(elapsed_ms)

        test_tensor = pil_to_lpips_tensor(test_img, config.device)
        lpips_value = runner.loss_fn(
            baseline_tensor_cpu.to(config.device), test_tensor
        ).item()
        lpips_values.append(lpips_value)

        print(f"    LPIPS: {lpips_value:.4f}, Time: {elapsed_ms:.1f} ms")

    avg_test_ms = sum(test_times_ms) / len(test_times_ms)
    return lpips_values, avg_test_ms


def _report_results(
    config: BenchmarkConfig,
    backends: tuple,
    timings: tuple,
    lpips_values: list,
    num_prompts: int,
) -> dict:
    """Print benchmark statistics and return the summary dict.

    ``backends`` is ``(baseline_backend, test_backend)`` and ``timings`` is
    ``(avg_baseline_ms, avg_test_ms)``.
    """
    baseline_backend, test_backend = backends
    avg_baseline_ms, avg_test_ms = timings

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    avg_lpips = sum(lpips_values) / len(lpips_values)
    max_lpips = max(lpips_values)
    min_lpips = min(lpips_values)
    std_lpips = np.std(lpips_values)

    print("\nLPIPS Statistics (lower is better, 0 = identical):")
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Std Dev:       {std_lpips:.4f}")
    print(f"  Min LPIPS:     {min_lpips:.4f}")
    print(f"  Max LPIPS:     {max_lpips:.4f}")

    print("\nTiming Statistics:")
    print(f"  Avg {baseline_backend} time:  {avg_baseline_ms:.1f} ms per image")
    print(f"  Avg {test_backend} time: {avg_test_ms:.1f} ms per image")
    print(f"  Speedup:                {avg_baseline_ms / avg_test_ms:.2f}x")

    print("\nBenchmark Configuration:")
    print(f"  Baseline backend:  {baseline_backend}")
    print(f"  Test backend:      {test_backend}")
    print(f"  torch.compile:     {config.compile}")
    print(f"  Model:             {MODEL_ID}")
    print(f"  Prompts tested:    {num_prompts}")
    print(f"  Inference steps:   {config.num_inference_steps}")
    print(f"  Generation size:   {config.width}x{config.height}")
    print(f"  LPIPS resize:      {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"  Random seed:       {RANDOM_SEED}")
    print("=" * 80)

    return {
        "avg_lpips": avg_lpips,
        "std_lpips": std_lpips,
        "min_lpips": min_lpips,
        "max_lpips": max_lpips,
        "speedup": avg_baseline_ms / avg_test_ms,
        "lpips_values": lpips_values,
    }


@torch.inference_mode()
def run_benchmark(
    baseline_backend: str = "fa3",
    test_backend: str = "fa3_fp8",
    num_prompts: int = 50,
    num_inference_steps: int = 20,
    height: int = 2048,
    width: int = 2048,
    debug_prompt: Optional[str] = None,
    warmup_iters: int = 2,
    compile: bool = False,
):
    """Run the attention backend benchmark on FLUX.1-schnell."""
    compile_str = " + torch.compile" if compile else ""
    print("=" * 80)
    print("Attention Backend Benchmark for FLUX.1-schnell")
    print(f"Baseline: {baseline_backend}  |  Test: {test_backend}{compile_str}")
    print("=" * 80)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = BenchmarkConfig(
        device="cuda",
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        warmup_iters=warmup_iters,
        compile=compile,
    )

    prompts = _load_prompts(debug_prompt, num_prompts)
    runner = _load_runner(config)

    baseline_data, avg_baseline_ms = _run_baseline_phase(
        runner, baseline_backend, prompts
    )

    # ----- Cleanup before test phase -----
    cleanup_gpu()

    lpips_values, avg_test_ms = _run_test_phase(runner, test_backend, baseline_data)

    return _report_results(
        config,
        (baseline_backend, test_backend),
        (avg_baseline_ms, avg_test_ms),
        lpips_values,
        len(prompts),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends on FLUX.1-schnell"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="fa3",
        choices=list(BACKENDS.keys()),
        help="Baseline attention backend",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="fa3_fp8",
        choices=list(BACKENDS.keys()),
        help="Test attention backend",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="Number of prompts to use (50 for quick, 200 for full benchmark)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--debug_prompt",
        type=str,
        default=None,
        help="Use a single debug prompt instead of DrawBench",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile for both backends",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2048,
        help="Generated image height in pixels (default: 2048)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2048,
        help="Generated image width in pixels (default: 2048)",
    )

    args = parser.parse_args()

    run_benchmark(
        baseline_backend=args.baseline,
        test_backend=args.test,
        num_prompts=args.num_prompts,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        debug_prompt=args.debug_prompt,
        warmup_iters=args.warmup_iters,
        compile=args.compile,
    )


if __name__ == "__main__":
    main()
