# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Protocols for some functionalities in tensor subclasses"""

from typing import Optional, Protocol, runtime_checkable

import torch

from torchao.quantization.quantize_.common.quantize_tensor_kwargs import (
    QuantizeTensorKwargs,
)


@runtime_checkable
class SupportsActivationPreScaling(Protocol):
    """Protocol for activation scale that should be multiplied with activation before quantization,
    or before we use activation in matrix multiplications, used for algorithms like AWQ

    A class that have `act_pre_scale: Optional[torch.Tensor]` attribute implements the Protocol
    """

    act_pre_scale: Optional[torch.Tensor]


def quantization_type_with_act_pre_scale(tensor) -> str:
    """Shared ``_quantization_type`` string for tensors with ``act_pre_scale``.

    Used by tensor subclasses whose ``_quantization_type`` reports shape,
    block size, device and (optionally) the activation pre-scale shape, so the
    identical string-building logic lives in one place.
    """
    s = f"shape={tensor.shape}, block_size={tensor.block_size}, device={tensor.device}"
    if tensor.act_pre_scale is not None:
        s += f", act_pre_scale.shape={tensor.act_pre_scale.shape}"
    return s


@runtime_checkable
class IsStaticQuantizationConfig(Protocol):
    """Protocol for static quantization configuration.

    A class that has `act_quant_scale: Optional[torch.Tensor]` attribute and
    `get_act_quant_kwargs() -> QuantizeTensorKwargs` method implements the Protocol
    """

    act_quant_scale: Optional[torch.Tensor]

    def get_act_quant_kwargs(self) -> QuantizeTensorKwargs: ...
