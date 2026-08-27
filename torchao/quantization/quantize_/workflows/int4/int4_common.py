# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


def _int4_quantization_type(tensor) -> str:
    """Shared ``_quantization_type`` string for int4 quantized tensors.

    Used by the int4 tensor implementations so the representation stays
    consistent and cannot drift between the different packing formats.
    """
    s = f"shape={tensor.shape}, block_size={tensor.block_size}, device={tensor.device}"
    if tensor.act_pre_scale is not None:
        s += f", act_pre_scale.shape={tensor.act_pre_scale.shape}"
    return s
