# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Shared roofline helpers for the blockwise FP8 training benchmarks.

Both ``bench_moe_grouped_kernels.py`` and ``benchmark_quant_kernel_bandwidth.py``
need to (1) look up per-GPU roofline specs by device name and (2) derive the peak
HBM bandwidth from the CUDA device properties. These live here so the two scripts
share one implementation instead of maintaining near-identical copies.
"""

from typing import Optional

import torch

from torchao.testing.training.roofline_utils import gpu_name_to_specs


def lookup_roofline_specs(gpu_name: str) -> Optional[dict]:
    """Look up roofline specs for ``gpu_name``, tolerating name mismatches.

    Falls back to a substring match in either direction so slightly different
    device-name strings (e.g. vendor suffixes) still resolve to known specs.
    """
    specs = gpu_name_to_specs.get(gpu_name)
    if specs is not None:
        return specs
    for known, candidate in gpu_name_to_specs.items():
        if known in gpu_name or gpu_name in known:
            return candidate
    return None


def peak_mem_bw_from_device_properties() -> Optional[float]:
    """Peak HBM bandwidth (bytes/sec) from CUDA device properties.

    Preferred over the roofline_utils value (which is a Meta-specific H100
    variant) when the device exposes memory clock and bus width. Returns
    ``None`` when those properties are unavailable.
    """
    props = torch.cuda.get_device_properties(0)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    return (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0
