# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.prototype.awq import AWQConfig
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization.quant_api import quantize_

from ..utils import get_size_of_dir
from .utils import string_to_calibration_config


@dataclass
class CalibrationConfig:
    """Calibration inputs that travel together through the workflow.

    Groups the evaluation tasks and per-task example limit so callers pass a
    single calibration spec instead of parallel ``tasks``/``limit`` arguments.
    """

    tasks: list[str]
    limit: int


@dataclass
class QuantizationSpec:
    """The step-based config class and its base config, which travel together.

    ``config_class`` and ``base_config`` are only ever used in combination to
    build a per-step config via ``config_class(base_config, step=...)``, so they
    are grouped into a single spec instead of two parallel arguments.
    """

    config_class: type
    base_config: object

    def build(self, step: str):
        """Instantiate the config for a given calibration ``step``."""
        return self.config_class(self.base_config, step=step)


def _apply_calibration(model, spec, calibration, tokenizer, filter_fn=None):
    """Apply prepare->calibrate->convert workflow for AWQ/SmoothQuant."""
    # Prepare
    quantize_(model, spec.build("prepare"), filter_fn=filter_fn)
    print(f"Calibrating with tasks={calibration.tasks}, limit={calibration.limit}")

    # Calibrate
    evaluator.simple_evaluate(
        HFLM(pretrained=model, tokenizer=tokenizer),
        tasks=calibration.tasks,
        limit=calibration.limit,
        batch_size=1,
    )
    quantize_(model, spec.build("convert"), filter_fn=filter_fn)
    load_config = spec.build("prepare_for_loading")
    model.config.quantization_config = TorchAoConfig(load_config)


def quantize_model_and_save(
    model: str,
    recipe: str,
    base_config_cls: dict,
    output_dir: str,
    calibration: CalibrationConfig,
):
    """Quantize model with calibration and save."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model, device_map="cuda:0", dtype=torch.bfloat16
    )

    if base_config_cls is None:
        pass
    elif recipe == "awq_int4_weight_only":
        spec = QuantizationSpec(AWQConfig, base_config_cls)
        _apply_calibration(model, spec, calibration, tokenizer)
    elif recipe == "smoothquant_int8":
        spec = QuantizationSpec(SmoothQuantConfig, base_config_cls)
        _apply_calibration(model, spec, calibration, tokenizer)
    else:
        raise AssertionError(f"unsupported recipe: {recipe}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize model with calibration (AWQ/SmoothQuant)"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--recipe",
        required=True,
        help="awq_int4_weight_only, smoothquant_int8, or None (no quantization)",
    )
    parser.add_argument("--output_dir", default="benchmarks/data/quantized_model/test")
    parser.add_argument("--calibration_tasks", nargs="+", default=["wikitext"])
    parser.add_argument("--calibration_limit", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{args.model} with {args.recipe}\n")
    base_config = string_to_calibration_config(args.recipe)
    model, _ = quantize_model_and_save(
        args.model,
        args.recipe,
        base_config,
        args.output_dir,
        CalibrationConfig(
            tasks=args.calibration_tasks,
            limit=args.calibration_limit,
        ),
    )
    print(f"Saved to {args.output_dir}")
    print(f"Size: {get_size_of_dir(args.output_dir) / 1e9:.2f} GB")
