# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FpnConv:
    """Configuration for the lateral 1x1 convolutions in :class:`FpnNeck`."""

    kernel_size: int = 1
    stride: int = 1
    padding: int = 0


@dataclass
class FpnFusion:
    """Configuration for the top-down feature fusion in :class:`FpnNeck`."""

    interp_model: str = "bilinear"
    fuse_type: str = "sum"
    # Levels to have top-down features in their outputs. ``None`` means all
    # levels receive top-down propagation.
    top_down_levels: Optional[List[int]] = field(default=None)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        )

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        with torch.autograd.profiler.record_function("self.neck(self.trunk(sample))"):
            from torchao._models.sam2.map_tensor import MapTensor, to_map_tensor

            if isinstance(sample, MapTensor):
                features, pos = self.neck(self.trunk(sample.elems.flatten(0, 1)))
                features = [to_map_tensor(t.unsqueeze(1)) for t in features]
                pos = [to_map_tensor(t.unsqueeze(1)) for t in pos]
            else:
                features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        conv: Optional[FpnConv] = None,
        fusion: Optional[FpnFusion] = None,
    ):
        """Initialize the neck
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param backbone_channel_list: input channels of each backbone level
        :param conv: kernel size, stride, and padding of the lateral convs
        :param fusion: interpolation mode, fuse type, and top-down levels
        """
        super().__init__()
        if conv is None:
            conv = FpnConv()
        if fusion is None:
            fusion = FpnFusion()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fusion.interp_model
        assert fusion.fuse_type in ["sum", "avg"]
        self.fuse_type = fusion.fuse_type

        # levels to have top-down features in its outputs
        # e.g. if top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        fpn_top_down_levels = fusion.top_down_levels
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
