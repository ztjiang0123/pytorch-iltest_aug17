# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import csv
import json
import random

import torch
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from naive_intNwo import intN_weight_only
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.quantization import quantize_


def get_initial_samples(sampling_spec, num_BO_initial_samples=10):
    """Generate initial BO search points based on a sampling specification.

    The layer indices are grouped into four bands that share the same
    structure across the different BO objectives (model size, throughput, ...);
    only the concrete bitwidth/groupsize choices and their sampling weights
    differ. ``sampling_spec`` describes those per-band choices so this single
    implementation can be reused by every caller.

    Args:
        sampling_spec (dict): mapping with the keys below. Each ``*_choices``
            entry is a ``(choices, weights)`` tuple passed to
            ``random.choices``.

            - ``fixed_low_bitwidth``: bitwidth used for layers 0-2.
            - ``fixed_high_bitwidth``: bitwidth used for layers 30-31.
            - ``mid_sensitive``: choices for the sensitive layers in 3-17.
            - ``mid_default``: choices for the remaining layers in 3-17.
            - ``late_sensitive``: choices for the sensitive layers in 18-29.
            - ``late_default``: choices for the remaining layers in 18-29.

            ``fixed_groupsize`` (default 32) is the groupsize for the fixed
            bands.
        num_BO_initial_samples (int): number of initial samples to generate.
    """
    fixed_groupsize = sampling_spec.get("fixed_groupsize", 32)
    initial_points_set = []

    # auto sample the bit choices with random choice probability positive
    # correlated to the sensitivity (FIT) score
    for _ in range(num_BO_initial_samples):
        initial_points = {}
        for i in range(0, 3):
            initial_points["bitwidth." + str(i) + "."] = sampling_spec[
                "fixed_low_bitwidth"
            ]
            initial_points["groupsize." + str(i) + "."] = fixed_groupsize

        for i in range(3, 18):
            band = "mid_sensitive" if i in [5, 6, 7, 10, 11, 12, 16] else "mid_default"
            (bw_choices, bw_weights), (gs_choices, gs_weights) = sampling_spec[band]
            initial_points["bitwidth." + str(i) + "."] = random.choices(
                bw_choices, bw_weights
            )[0]
            initial_points["groupsize." + str(i) + "."] = random.choices(
                gs_choices, gs_weights
            )[0]

        for i in range(18, 30):
            band = "late_sensitive" if i in [22, 23, 24] else "late_default"
            (bw_choices, bw_weights), (gs_choices, gs_weights) = sampling_spec[band]
            initial_points["bitwidth." + str(i) + "."] = random.choices(
                bw_choices, bw_weights
            )[0]
            initial_points["groupsize." + str(i) + "."] = random.choices(
                gs_choices, gs_weights
            )[0]

        for i in range(30, 32):
            initial_points["bitwidth." + str(i) + "."] = sampling_spec[
                "fixed_high_bitwidth"
            ]
            initial_points["groupsize." + str(i) + "."] = fixed_groupsize

        initial_points_set.append(initial_points)
    return initial_points_set


def write_history_to_csv(history, output_file, keyword):
    # keyword example: ['cal_PPL', 'cal_throughput', 'config']

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(keyword)

        for eval_results, config in history:
            obj1 = eval_results[keyword[0]][0]
            obj2 = eval_results[keyword[1]][0]

            writer.writerow([obj1, obj2, config])


# quantize a model based on a given quantization configuration
def quantize_by_fqn_to_config(model, device, fqn_to_config):
    it = iter(fqn_to_config.items())
    while True:
        try:
            k1, v1 = next(it)
            k2, v2 = next(it)
            fqn = k1[8:]
            bit_width, groupsize = v1, v2

            def filter_fn_sen(child: torch.nn.Module, cur_fqn: str) -> bool:
                return isinstance(child, torch.nn.Linear) and (fqn in cur_fqn)

            quantize_(
                model.to(device=device),
                intN_weight_only(n=bit_width, group_size=groupsize),
                filter_fn_sen,
            )
        except StopIteration:
            break


# calculate perplexity on wikitext-document, need to support more tasks
def cal_wikitext_ppl(model, tokenizer, limit=62):
    with torch.no_grad():
        result = evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1),
            get_task_dict("wikitext"),
            limit=limit,
        )

    return result["results"]["wikitext"]["word_perplexity,none"]


# TODO: make it generalize to more models
def cal_model_size(model, fqn_to_config):
    _sum = 0
    fqn_cofg_dict = dict()

    it = iter(fqn_to_config.items())
    while True:
        try:
            k1, v1 = next(it)
            k2, v2 = next(it)
            bit_width, groupsize = v1, v2
            bit_zeropoint = 32
            bit_scale = 8
            fqn = k1[8:]
            fqn_cofg_dict[fqn] = (bit_width, groupsize, bit_zeropoint, bit_scale)
        except StopIteration:
            break

    for name, parameter in model.named_parameters():
        flag = 0
        for fqn in fqn_cofg_dict:
            if fqn in name:
                flag = 1
                if "self_attn" in name or "mlp" in name:
                    _sum += parameter.numel() * fqn_cofg_dict[fqn][
                        0
                    ] + parameter.numel() // fqn_cofg_dict[fqn][1] * (
                        fqn_cofg_dict[fqn][2] + fqn_cofg_dict[fqn][3]
                    )
        if flag == 0:
            _sum += parameter.numel() * 16

    _sum_in_byte = _sum / 8.0
    _sum_in_GB = _sum_in_byte / (1024**3) / 1.0
    return _sum_in_GB


def load_model(repo_id, device):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16).to(
        device=device
    )
    return model, tokenizer


def load_parameters_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)

    bitwidth_config = next(
        param for param in config["parameters"] if param["name"] == "bitwidth"
    )
    groupsize_config = next(
        param for param in config["parameters"] if param["name"] == "groupsize"
    )

    parameters_list = []

    # Ensure that we are interleaving bitwidth and groupsize for each layer
    for bw_layer, gs_layer in zip(
        bitwidth_config["layers"], groupsize_config["layers"]
    ):
        start, end = bw_layer["range"]
        for i in range(start, end):
            # Add bitwidth parameter
            bitwidth_param = {
                "name": bitwidth_config["name_format"].format(i=i),
                "type": bw_layer["type"],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": True,
            }
            if bw_layer["type"] == "fixed":
                bitwidth_param["value"] = bw_layer["value"]
            elif bw_layer["type"] == "choice":
                bitwidth_param["values"] = bw_layer["values"]
            parameters_list.append(bitwidth_param)

            # Add groupsize parameter
            groupsize_param = {
                "name": groupsize_config["name_format"].format(i=i),
                "type": gs_layer["type"],
                "value_type": "int",
                "is_ordered": True,
                "sort_values": True,
            }
            if gs_layer["type"] == "fixed":
                groupsize_param["value"] = gs_layer["value"]
            elif gs_layer["type"] == "choice":
                groupsize_param["values"] = gs_layer["values"]
            parameters_list.append(groupsize_param)

    return parameters_list


def load_initial_samples(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config["initial_samples"]
