# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.prototype.autoround.multi_tensor import MultiTensor, _multi_tensor_config


@ar_utils.singleton
@dataclasses.dataclass
class _AutoRoundConfig:
    bits: int = 4
    group_size: int = 128
    iters: int = 200
    use_optimized_layer_output: bool = False
    gradient_accumulate_steps: int = 1
    compile_optimization_process: bool = False


_auto_round_config = _AutoRoundConfig()


@ar_utils.singleton
@dataclasses.dataclass
class _OptimizationTracker:
    num_layers: int = 0
    optimized_layers: int = 0

    def reset(self):
        self.num_layers = 0
        self.optimized_layers = 0


_optimization_tracker = _OptimizationTracker()


def _replace_model_buffers_and_params(model, replacement_fn):
    model = replacement_fn(model)
    for name, child in model.named_children():
        new_child = _replace_model_buffers_and_params(child, replacement_fn)
        if new_child is not child:
            setattr(model, name, new_child)
    return model


def _tensor_to_multi_tensor(model):
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, torch.nn.Parameter(MultiTensor([param]), False))
    return model


def _multi_tensor_to_tensor(model):
    for name, buf in model.named_buffers(recurse=False):
        if isinstance(buf, MultiTensor):
            assert len(buf.values) == 1, (
                f"The buffer should only have one tensor, but got {buf.count}."
            )
            model.register_buffer(name, buf.values[0])
    for name, param in model.named_parameters(recurse=False):
        if isinstance(param, MultiTensor):
            assert len(param.values) == 1, (
                f"The parameter should only have one tensor, but got {param.count}."
            )
            setattr(
                model, name, torch.nn.Parameter(param.values[0], requires_grad=False)
            )
    return model


@torch.no_grad()
def prepare_model_for_applying_auto_round_(
    model: torch.nn.Module,
    is_target_module: Callable[[torch.nn.Module, str], bool],
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    use_optimized_layer_output: bool = False,
    gradient_accumulate_steps: Optional[int] = 1,
    compile_optimization_process: Optional[bool] = False,
    device: Optional[torch.types.Device] = None,
):
    """Prepares the model for applying auto round optimization.

    Args:
        model (torch.nn.Module): The floating-point model to be quantized.
        is_target_module (Callable[[torch.nn.Module, str], bool]): A function that determines
            whether a module is a target module.
        bits (int, optional): The number of bits for quantization. Defaults to 4, options are 1 to 8.
        group_size (int, optional): The group size for quantization. Defaults to 128.
        iters (int, optional): The number of iterations for optimization. Defaults to 200.
        use_optimized_layer_output (bool, optional): Whether to use optimized layer output. Defaults to False.
        gradient_accumulate_steps (Optional[int]): Number of steps for accumulating gradients before
            performing the backward pass when optimizing each target module. Defaults to 1.
        compile_optimization_process (Optional[bool]): Whether to compile the optimization process. Defaults to False.
        device (Optional[torch.types.Device]): The device to use for accelrating optimization and calibration.
            Defaults to None.
    """
    _multi_tensor_config.device = device
    _multi_tensor_config.offload = next(model.parameters()).device.type != device
    _optimization_tracker.reset()

    _auto_round_config.bits = bits
    _auto_round_config.group_size = group_size
    _auto_round_config.iters = iters
    _auto_round_config.use_optimized_layer_output = use_optimized_layer_output
    _auto_round_config.gradient_accumulate_steps = gradient_accumulate_steps
    _auto_round_config.compile_optimization_process = compile_optimization_process

    logging.warning(f"config {_auto_round_config}")

    # Wrap the model buffers and parameters with `MultiTensor`
    model = _replace_model_buffers_and_params(model, _tensor_to_multi_tensor)

    def _revert_buffers_and_params_fn(
        module,
        input: Tuple[MultiTensor],
        output: Tuple[MultiTensor],
    ):
        module._forward_hook_handle_for_revert_buffers_and_params.remove()
        _replace_model_buffers_and_params(module, _multi_tensor_to_tensor)
        return output

    # Register forward hook for reverting the replacement of buffers and parameters
    model._forward_hook_handle_for_revert_buffers_and_params = (
        model.register_forward_hook(_revert_buffers_and_params_fn)
    )

    # Register forward hook for applying auto-round optimization
    def auto_round_optimization_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        apply_auto_round_optimization(
            module, args, kwargs, output, config=_auto_round_config
        )
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            auto_round_optimization_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_auto_round = forward_hook_handle
        _optimization_tracker.num_layers += 1
        return module

    model.eval()
    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_target_module
    )


def apply_auto_round():
    """Create the quantized model from the model optimized by auto-round.

    More details about the auto-round can be found at https://arxiv.org/abs/2309.05516.
    """

    raise AssertionError(
        "Please migrate this function to direct configuration, see https://github.com/pytorch/ao/issues/1690"
        " and https://github.com/pytorch/ao/pull/4245 for details"
    )


@torch.no_grad()
def _apply_auto_round_optimization(
    block, block_inputs, block_outputs, config: _AutoRoundConfig
):
    # Call the auto-round to execute the optimization process.
    # https://github.com/intel/auto-round/tree/patch-for-ao-2
    # TODO(Yi), make the branch more stable
    if ar_utils.is_auto_round_available():
        import auto_round
    else:
        raise ImportError(
            (
                "This example requires the `auto-round` library."
                "Please install it with `pip install git+https://github.com/intel/auto-round.git@patch-for-ao-2`"
            )
        )
    orig_device = next(block.parameters()).device
    block = block.to(_multi_tensor_config.device)
    _optimization_tracker.optimized_layers += 1
    logging.warning(
        "Apply auto-round optimization on layer %d / %d.",
        _optimization_tracker.optimized_layers,
        _optimization_tracker.num_layers,
    )

    # Start the training process to update the v, alpha and beta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,
        bits=config.bits,
        iters=config.iters,
        group_size=config.group_size,
        gradient_accumulate_steps=config.gradient_accumulate_steps,
        amp=True,
        model_dtype=next(block.parameters()).dtype,
    )
    if config.compile_optimization_process:
        rounder.quant_block_v2_ = torch.compile(rounder.quant_block_v2_)

    with torch.enable_grad():
        rounder.quant_block_v2_(
            block,
            inputs=block_inputs,
            outputs=block_outputs,
            device=_multi_tensor_config.device,
        )
    block.to(orig_device)


@ar_utils.dump_elapsed_time(record=True)
@torch.no_grad()
def apply_auto_round_optimization(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, Any],
    output: Any,
    config: _AutoRoundConfig,
):
    # Remove the hook to avoid recursive calls
    module._forward_hook_handle_for_auto_round.remove()
    # Revert the model to the original state for applying auto-round optimization
    module = _replace_model_buffers_and_params(module, _multi_tensor_to_tensor)

    block_inputs = MultiTensor.revert_to_tensor_pairs(args, kwargs)
    block_outputs = MultiTensor.revert_to_tensor_pairs(output)

    _apply_auto_round_optimization(module, block_inputs, block_outputs, config)
    # Get the new output of the optimized model
    if config.use_optimized_layer_output:
        # Re-replace the model buffers and parameters with `MultiTensor`
        _replace_model_buffers_and_params(module, _tensor_to_multi_tensor)
        output = module(*args, **kwargs)
    return output
