# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch


def _is_cuda_capability_major(major: int) -> bool:
    """Return True if a CUDA device is available and its compute-capability
    major version matches ``major``."""
    if not torch.cuda.is_available():
        return False
    device_major, _ = torch.cuda.get_device_capability()
    return device_major == major


def _is_hopper() -> bool:
    return _is_cuda_capability_major(9)


def _is_blackwell() -> bool:
    return _is_cuda_capability_major(10)


def _is_fa3_available() -> bool:
    try:
        importlib.import_module("flash_attn_interface")
        return True
    except ModuleNotFoundError:
        return False


def _is_fa4_available() -> bool:
    try:
        importlib.import_module("flash_attn.cute.interface")
        return True
    except ModuleNotFoundError:
        return False
