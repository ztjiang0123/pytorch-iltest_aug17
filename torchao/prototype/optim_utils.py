# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class BaseWrappedOptimizer(Optimizer):
    """Common functionality shared by optimizers that wrap a ``base_optimizer``.

    Both :class:`~torchao.prototype.parq.optim.QuantOptimizer` and
    :class:`~torchao.prototype.pat.optim.PruneOptimizer` delegate to an
    underlying optimizer while tracking a per-group ``num_steps`` counter and
    latent (pre-proximal) copies of the parameters. This base class holds the
    logic that is identical between them so it lives in a single place.

    Subclasses are expected to define ``base_optimizer`` and
    ``regularized_param_groups()``.
    """

    def __getattribute__(self, name: str):
        try:
            attr = super(Optimizer, self).__getattribute__(name)
        except AttributeError:
            attr = self.base_optimizer.__getattribute__(name)
        return attr

    @property
    def state(self) -> defaultdict[Tensor, Any]:  # pyre-ignore[3]
        return self._state if hasattr(self, "_state") else self.base_optimizer.state

    @property
    def num_steps(self) -> int:
        for group in self.regularized_param_groups():
            return group.setdefault("num_steps", 0)

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        for group in self.regularized_param_groups():
            group["num_steps"] = value
            return

    @num_steps.deleter
    def num_steps(self) -> None:
        for group in self.regularized_param_groups():
            group.pop("num_steps", None)
            return

    @torch._disable_dynamo
    @torch.no_grad()
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.copy_(self.state[p]["latent"])
