# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from torchao.prototype.parq.optim import QuantOptimizer
from torchao.prototype.parq.quant import (
    Quantizer,
    StretchedUnifTorchaoQuantizer,
    UnifTorchaoQuantizer,
)


@dataclass(frozen=True, slots=True)
class QuantConfig:
    bitwidth: int
    group_size: Optional[int] = None
    quantizer: Optional[Quantizer] = None

    def __post_init__(self):
        if self.bitwidth < 2:
            raise ValueError("bitwidth must be >= 2")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive")

        if self.quantizer is None:
            if self.bitwidth in [2, 3]:
                q = StretchedUnifTorchaoQuantizer(b=self.bitwidth)
            else:
                q = UnifTorchaoQuantizer()
            object.__setattr__(self, "quantizer", q)


def _build_initial_param_groups(quant_configs_and_filter_fns):
    """Create one param group per quant config, plus a trailing no-quant group.

    The no-quant group is kept last so that a quantized config's index in
    ``quant_configs_and_filter_fns`` matches its index in ``param_groups``.
    """
    param_groups = []
    for config, _ in quant_configs_and_filter_fns:
        param_group = {"params": [], "quant_bits": config.bitwidth}
        if config.group_size is not None:
            param_group["quant_block_size"] = config.group_size
        param_group["_quantizer"] = config.quantizer
        param_groups.append(param_group)

    param_groups.append({"params": [], "weight_decay": 0.0})
    return param_groups


def _assign_param_to_group(
    param, param_name, owning_module, quant_configs_and_filter_fns, param_groups
):
    """Append ``param`` to the single group whose filter matches it.

    Falls back to the trailing no-quant group when nothing matches, and raises
    if more than one config matches the same parameter.
    """
    matching_config = None
    for idx, (config, filter_fn) in enumerate(quant_configs_and_filter_fns):
        if not filter_fn(owning_module, param_name):
            continue
        param_groups[idx]["params"].append(param)
        if matching_config is not None:
            raise ValueError(
                f"Found multiple matching configs for {param_name}. "
                f"Previous match={matching_config}, new match={config}."
            )
        matching_config = config
        print(f"{config.bitwidth},{config.group_size}")

    if matching_config is None:
        print("NONE")
        param_groups[-1]["params"].append(param)


def create_param_groups_and_group_quantizer_map(
    model: torch.nn.Module,
    quant_configs_and_filter_fns: List[
        Tuple[QuantConfig, Callable[[torch.nn.Module, str], bool]]
    ],
):
    param_groups = _build_initial_param_groups(quant_configs_and_filter_fns)

    seen_data_ptrs = {}
    for param_name, param in model.named_parameters():
        module_name, _, _ = param_name.rpartition(".")
        owning_module = model.get_submodule(module_name) if module_name else model

        data_ptr = param.data_ptr()
        if data_ptr in seen_data_ptrs:
            print(
                f"Not considering {param} because it shares a data_ptr with "
                f"{seen_data_ptrs[data_ptr]}, which was previously considered"
            )
            continue
        seen_data_ptrs[data_ptr] = param_name

        print(
            "param_name",
            param_name,
            "module_type",
            type(owning_module),
            "matching_config:",
            end="",
        )
        _assign_param_to_group(
            param,
            param_name,
            owning_module,
            quant_configs_and_filter_fns,
            param_groups,
        )

    # Filter out empty param groups
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]

    # After filter define group_quantizer_map
    # The index in group_quantizer_map must correspond to index in
    # quantized params
    group_quantizer_map = {}
    for idx, param_group in enumerate(param_groups):
        if "_quantizer" in param_group:
            group_quantizer_map[idx] = param_group.pop("_quantizer")

    expected_n_params = sum(1 for p in model.parameters())
    n_found_params = sum(len(pg["params"]) for pg in param_groups)
    assert n_found_params == expected_n_params, (
        f"{n_found_params} != {expected_n_params=}"
    )

    return param_groups, group_quantizer_map


from torchao.prototype.parq import ProxHardQuant


def create_optimizer(
    model: torch.nn.Module,
    quant_configs_and_filter_fns: List[
        Tuple[QuantConfig, Callable[[torch.nn.Module, str], bool]]
    ],
    base_optimizer_cls: Type[torch.optim.Optimizer],
    base_optimizer_kwargs: Dict[str, Any],
    *,
    warmup_steps: int = 0,
    quant_period: int = 1,
    quant_per_channel: bool = True,
):
    param_groups, group_quantizer_map = create_param_groups_and_group_quantizer_map(
        model, quant_configs_and_filter_fns
    )
    base_optimizer = base_optimizer_cls(param_groups, **base_optimizer_kwargs)
    optimizer = QuantOptimizer(
        base_optimizer,
        quantizer=UnifTorchaoQuantizer(),
        prox_map=ProxHardQuant(),
        warmup_steps=warmup_steps,
        quant_period=quant_period,
        quant_per_channel=quant_per_channel,
        group_quantizer_map=group_quantizer_map,
    )
    return optimizer
