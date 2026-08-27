# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict

from ...optim_utils import BaseWrappedOptimizer
from ..distributed_utils import (
    _is_dtensor,
    _is_main_process,
    _maybe_async_aggregate,
    _sum_async_streams,
)
from ..utils import get_index_linspace, instantiate_module
from .group_lasso import ProxGroupLasso, ProxGroupLassoVectorized
from .iterative_reweight import IterativeReweight


@dataclass
class _StepContext:
    """Step-wide context shared by every param processed in one ``step`` call."""

    init_sigma_reweight: bool
    update_tau_reweight: bool
    dist_is_init: bool
    zeros_buf: list = field(default_factory=list)
    factored_size_buf: list = field(default_factory=list)


@dataclass
class _GroupContext:
    """Per-param-group context reused for every param in the group."""

    group: dict
    prox_map: Any
    grouper_cls: Any
    grouper_kwargs: dict
    prox_kwargs: dict


@dataclass
class _SizeTotals:
    """Running accumulators for regularized-size statistics over a step."""

    zeros: float = 0
    params: int = 0
    factored_size: int = 0
    unfactored_size: int = 0

    def add(self, contribution: "_SizeTotals") -> None:
        self.zeros += contribution.zeros
        self.params += contribution.params
        self.factored_size += contribution.factored_size
        self.unfactored_size += contribution.unfactored_size


class PruneOptimizer(BaseWrappedOptimizer):
    """Wraps a base optimizer to apply proximal updates that induce sparsity
    or low-rank structure during training.

    Arguments:
        base_optimizer: The underlying optimizer (e.g., SGD or AdamW) that
            updates the latent parameters.
        warmup_steps: Number of initial steps to run before applying proximal
            updates, during which the optimizer behaves like the base optimizer.
        healing_start_step: Step at which to start the "healing" phase, where
            pruned parameters are frozen. Must be greater than warmup_steps.
        reg_lambda: Regularization strength for the proximal updates. Can be
            overridden per parameter group.
        reweight_tau_freq: Frequency in steps to apply an iterative reweighting
            heuristic after each proximal update to adjust the regularization
            strength based on the current magnitude of the parameters.
        reweight_tau_end_step: Last step at which to apply iterative reweighting.
        reweight_eps: Small constant to prevent division by zero in iterative
            reweighting.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        warmup_steps: int = 0,
        healing_start_step: int = sys.maxsize,
        reg_lambda: float = 0.0,
        reweight_tau_freq: int = 0,
        reweight_tau_end_step: int = sys.maxsize,
        reweight_eps: float = 1e-3,
    ) -> None:
        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer

        # need to store these attributes in state_dict for checkpoint
        assert warmup_steps < healing_start_step, (
            f"Invalid {warmup_steps=} >= {healing_start_step=}"
        )
        self.num_steps = 0
        self.warmup_steps = warmup_steps
        self.healing_start_step = healing_start_step

        for group in self.regularized_param_groups():
            group.setdefault("gamma", 0.0)
            group.setdefault("reg_lambda", reg_lambda)
            if group.get("min_sparsity_schedule", False):
                assert self.healing_start_step != sys.maxsize, (
                    "min_sparsity_schedule requires a finite healing_start_step; "
                    "the ramp ends when the mask freezes."
                )

        self.iterative_reweight = (
            IterativeReweight(reweight_tau_freq, reweight_tau_end_step, reweight_eps)
            if reweight_tau_freq > 0
            else None
        )

        self.relative_sparsity = 0
        self.relative_factored_frac = 0

        # NOTE: Filling state dict here cause Adam(W) error, which assumes
        # empty state[p] at first step() where optimizer states are initialized

    def __repr__(self) -> str:
        base_optimizer = "\n    ".join(self.base_optimizer.__repr__().split("\n"))
        extra_repr = "\n  ".join(("(", base_optimizer))
        return f"{self.__class__.__name__} {extra_repr}\n)"

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.base_optimizer.__setstate__(state)
        for i, group in enumerate(self.regularized_param_groups()):
            group.setdefault("gamma", 0.0)
            group.setdefault("reg_lambda", 0.0)
            if i == 0:
                group.setdefault("num_steps", 0)

    @torch._disable_dynamo
    def state_dict(self) -> StateDict:
        return self.base_optimizer.state_dict()

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        self.base_optimizer.load_state_dict(state_dict)

    @torch._disable_dynamo
    def patch_state_dict(self, state_dict: StateDict) -> None:
        """Fix missing state after calling torch.distributed.checkpoint.load"""
        for i, group in enumerate(self.regularized_param_groups()):
            state_group = state_dict["param_groups"][i]
            for k in ("reg_lambda", "num_steps", "gamma"):
                if k in state_group:
                    group[k] = state_group[k]

    def regularized_param_groups(self):  # pyre-ignore[3]
        """Yield parameter groups that need to be pruned."""
        for group in self.param_groups:
            if group.get("prox_type"):
                yield group

    def _get_prox_kwargs(self, group: dict[str, Any]) -> dict[str, Any]:
        prox_kwargs = {}
        if group["prox_type"] == "NMSparseConstraint":
            assert "n_nonzero" in group, (
                "NMSparseConstraint requires 'n_nonzero' in prune config"
            )
            prox_kwargs["n_nonzero"] = group["n_nonzero"]
        elif group["prox_type"] == "MinSparsityConstraint":
            assert "min_sparsity" in group, (
                "MinSparsityConstraint requires 'min_sparsity' in prune config"
            )
            prox_kwargs["min_sparsity"] = self._effective_min_sparsity(group)
        return prox_kwargs

    def _effective_min_sparsity(self, group: dict[str, Any]) -> float:
        """Cubic ramp from 0 -> ``min_sparsity`` over (warmup, healing_start).

        When ``min_sparsity_schedule`` is unset (default), returns the static
        target. The ramp ends at ``healing_start_step`` because the mask
        freezes there — pushing the target up after that would be a no-op.
        """
        target = group["min_sparsity"]
        if not group.get("min_sparsity_schedule", False):
            return target
        n = self.num_steps
        if n <= self.warmup_steps:
            return 0.0
        # Unreachable in training (step() short-circuits at healing_start_step);
        # kept as a boundary guard for direct callers.
        if n >= self.healing_start_step:
            return target
        t = (n - self.warmup_steps) / (self.healing_start_step - self.warmup_steps)
        return target * (1 - (1 - t) ** 3)

    @staticmethod
    def _get_grouper_kwargs(group: dict[str, Any]) -> dict[str, Any]:
        grouper_kwargs = {}
        if group["group_type"].startswith("AttentionHeadGrouper"):
            grouper_kwargs["num_heads"] = group["num_heads"]
        elif group["group_type"] == "QKGrouper":
            if "qk_pack_dim" in group:
                grouper_kwargs["qk_pack_dim"] = group["qk_pack_dim"]
            if "qk_reg_index" in group:
                grouper_kwargs["qk_reg_index"] = group["qk_reg_index"]
        elif group["group_type"] == "KElementGrouper":
            grouper_kwargs["k"] = group["k"]
        elif group["group_type"] == "PackedSVDGrouper":
            grouper_kwargs["npack"] = group["npack"]
            if "pack_dim" in group:
                grouper_kwargs["pack_dim"] = group["pack_dim"]
        return grouper_kwargs

    @staticmethod
    def _apply_prox_dtensor(
        grouper, prox_map, p, gamma, gamma_in_dims, tau_reweight, tau_reweight_in_dims
    ):
        """Apply prox_map to a DTensor parameter via local_map.

        Returns:
            zero_elts: number of zero elements (int, globally summed)
            group_norm: group-level norm DTensor
        """
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=p.device)

        # Derive input placements from grouper.p
        p_in_placements = tuple(
            Shard(grouper.in_dims)
            if grouper.in_dims is not None and plc.is_shard()
            else plc
            for plc in grouper.p.placements
        )
        if grouper.in_dims is not None and gamma.dim() > 0:
            # Shard gamma according to grouper.in_dims
            gamma = distribute_tensor(
                gamma.unsqueeze(int(not grouper.in_dims)),
                device_mesh=p.device_mesh,
                placements=p_in_placements,
            )
            gamma_in_dims = grouper.in_dims
        else:
            gamma = distribute_tensor(gamma, device_mesh=p.device_mesh)

        # Use ProxGroupLassoVectorized for group lasso
        if isinstance(prox_map, ProxGroupLasso):
            prox_map_vec = ProxGroupLassoVectorized(
                prox_map.reg_lambda,
                reduce_dim=int(not grouper.in_dims),
            )
            local_fn = prox_map_vec.apply_
            if torch.is_tensor(tau_reweight) and tau_reweight.dim() < grouper.p.dim():
                tau_reweight = tau_reweight.unsqueeze(int(not grouper.in_dims))
        else:
            # Use vmap for other prox types
            local_fn = torch.vmap(
                prox_map.apply_,
                in_dims=(
                    grouper.in_dims,
                    gamma_in_dims,
                    tau_reweight_in_dims,
                ),
                out_dims=(0, 0),
            )

        # Redistribute explicitly so the in-place prox mutation lands on a
        # tensor we can copy back; local_map's own redistribute would discard it.
        needs_redistribute = tuple(grouper.p.placements) != p_in_placements
        p_for_prox = (
            grouper.p.redistribute(placements=p_in_placements)
            if needs_redistribute
            else grouper.p
        )

        zero_elts_per_group, group_norm = local_map(
            local_fn,
            out_placements=(
                (Partial(),) * p.device_mesh.ndim,
                (Shard(0),) * p.device_mesh.ndim,
            ),
            in_placements=(
                p_in_placements,
                gamma.placements if _is_dtensor(gamma) else None,
                tau_reweight.placements if _is_dtensor(tau_reweight) else None,
            ),
            redistribute_inputs=False,
        )(p_for_prox, gamma, tau_reweight)

        if needs_redistribute:
            # Write mutated values back to the original parameter.
            grouper.p.copy_(p_for_prox.redistribute(placements=grouper.p.placements))

        return zero_elts_per_group.full_tensor().sum().item(), group_norm

    @staticmethod
    def _apply_prox(
        grouper, prox_map, p, tau_reweight=1.0, sv_count=None, **prox_kwargs
    ) -> tuple[Tensor, Tensor, bool]:
        """
        Apply `prox_map` to the grouped parameter tensor `p` in place. Update
        `sv_count` if provided. Handles both torch.Tensor and DTensor inputs,
        mirroring `torch.vmap` semantics. Assumes prox_map.apply_ returns an
        integer per group.

        Returns:
            zero_elts: number of zero elements after applying prox map
            group_norm: per-group norm divided by the prox map's tau
            zeros_are_summed: whether zero_elts is already globally summed
        """
        gamma = prox_kwargs["gamma"]
        zeros_are_summed = False
        with grouper:
            gamma_in_dims = None
            tau_reweight_in_dims = None
            if torch.is_tensor(tau_reweight) and tau_reweight.dim() > 0:
                tau_reweight_in_dims = 0
            if prox_kwargs["gamma_index_slope"] > 0:
                # y = slope(2x - 1) + 1
                gamma = gamma * get_index_linspace(
                    prox_kwargs["gamma_index_slope"],
                    grouper.n_groups(),
                    device=p.device,
                )
                gamma_in_dims = 0

            if prox_kwargs["disable_vmap"] or prox_map.whole_tensor:
                # Element-, layer-, or whole-tensor pruning: bypass vmap and
                # call apply_ once on the full grouped view. whole_tensor prox
                # maps treat p.size(0) as n_groups, so transpose when the
                # grouper iterates dim 1 (e.g. Dim1Grouper).
                transpose = getattr(grouper, "in_dims", 0) == 1 and grouper.p.dim() == 2
                if _is_dtensor(grouper.p):
                    # Prox maps that mutate via index_put_ (e.g.
                    # MinSparsityConstraint) have no DTensor sharding rule and
                    # the whole-tensor variants need a global view to compute
                    # correct top-k. Gather, mutate, then scatter back.
                    full = grouper.p.full_tensor()
                    view = full.transpose(0, 1) if transpose else full
                    zero_elts, group_norm = prox_map.apply_(view, gamma, tau_reweight)
                    grouper.p.copy_(
                        distribute_tensor(
                            full,
                            device_mesh=grouper.p.device_mesh,
                            placements=grouper.p.placements,
                        )
                    )
                else:
                    view = grouper.p.transpose(0, 1) if transpose else grouper.p
                    zero_elts, group_norm = prox_map.apply_(view, gamma, tau_reweight)
                zeros_are_summed = zero_elts.dim() == 0
            else:
                if not prox_kwargs["is_svd_grouper"] and _is_dtensor(p):
                    zero_elts, group_norm = PruneOptimizer._apply_prox_dtensor(
                        grouper,
                        prox_map,
                        p,
                        gamma,
                        gamma_in_dims,
                        tau_reweight,
                        tau_reweight_in_dims,
                    )
                else:
                    # torch.Tensor branch - use standard vmap
                    zero_elts_per_group, group_norm = torch.vmap(
                        prox_map.apply_,
                        in_dims=(
                            grouper.in_dims,
                            gamma_in_dims,
                            tau_reweight_in_dims,
                        ),
                        out_dims=(0, 0),
                    )(grouper.p, gamma, tau_reweight)
                    zero_elts = zero_elts_per_group.sum().item()
                zeros_are_summed = True

                # Adjust for group-based pruning
                if not prox_kwargs["is_svd_grouper"] and not prox_kwargs.get(
                    "zero_elts_are_counts", False
                ):
                    zero_elts *= grouper.group_size()

            # Record for reconstruction and logging
            if prox_kwargs["is_svd_grouper"]:
                dim = 0 if sv_count.dim() > 1 else None
                sv_count.copy_(
                    (grouper.p != 0).to(torch.uint8).sum(dim=dim)
                    if _is_dtensor(p)
                    else torch.count_nonzero(grouper.p, dim=dim)
                )

            return zero_elts, group_norm, zeros_are_summed

    def _set_gamma(self, group):
        # AProx in practice: ensure shrinkage coefficient >= 1
        group["gamma"] += group["lr"]

    def _init_latent_state(self):
        for group in self.regularized_param_groups():
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None or "latent" in state:
                    continue
                state["latent"] = p.detach().clone()

    def _collect_healing_masks(self) -> dict:
        """Zero grads of pruned params and return masks to re-zero them later.

        During healing, momentum may push pruned params non-zero, so we save the
        pre-step masks to re-apply after the base optimizer step.
        """
        healing_masks = {}
        if self.num_steps < self.healing_start_step:
            return healing_masks
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.grad is None:
                    continue
                mask = p.ne(0)
                healing_masks[id(p)] = mask
                if _is_dtensor(p):
                    p.grad.mul_(mask)
                else:
                    p.grad.masked_fill_(~mask, 0)
        return healing_masks

    def _rezero_pruned_params(self, healing_masks: dict) -> None:
        """Re-apply saved healing masks to zero pruned params after a step."""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                mask = healing_masks.get(id(p))
                if mask is None:
                    continue
                if _is_dtensor(p):
                    p.mul_(mask)
                else:
                    p.masked_fill_(~mask, 0)

    def _in_base_optimizer_only_phase(self) -> bool:
        """Whether this step should run only the base optimizer (warmup/healing)."""
        return (
            self.num_steps < self.warmup_steps
            or self.num_steps >= self.healing_start_step
        )

    def _base_optimizer_only_step(
        self, closure: Callable[[], float] | None, healing_masks: dict
    ) -> float | None:
        """Run only the base optimizer, re-zeroing pruned params afterwards."""
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]
        self._rezero_pruned_params(healing_masks)
        self._init_latent_state()
        self.num_steps += 1
        return loss

    def _update_latent_params(self) -> None:
        """Save or restore latent params so the base optimizer updates them."""
        if self.num_steps == self.warmup_steps:
            # first PAT step: save latent params
            self.save_latent_params()
        else:
            # restore latent params for base optimizer update
            self.restore_latent_params()

    def _restore_temporary_state(self) -> None:
        """Restore the temporary latent state buffer into the base optimizer."""
        if not hasattr(self, "_state"):
            return
        assert self.warmup_steps == 0
        for p in self._state.keys():
            self.base_optimizer.state[p]["latent"] = self._state[p]["latent"]
        del self._state

    def _reweight_flags(self) -> tuple[bool, bool]:
        """Return (init_sigma_reweight, update_tau_reweight) for this step."""
        if self.iterative_reweight is None:
            return False, False
        init_sigma_reweight = self.num_steps == self.warmup_steps
        # offset by 1 since we update tau_reweight for the next step's prox map
        update_tau_reweight = self.iterative_reweight.should_update(self.num_steps + 1)
        return init_sigma_reweight, update_tau_reweight

    def _make_prox_kwargs(self, group) -> dict:
        return {
            "gamma": group["gamma"],
            "gamma_index_slope": group.get("gamma_index_slope", 0.0),
            "disable_vmap": group["group_type"].endswith(
                ("ElemGrouper", "LayerGrouper")
            ),
            "is_svd_grouper": group["group_type"].endswith("SVDGrouper"),
            "zero_elts_are_counts": group["prox_type"]
            in ("NMSparseConstraint", "MinSparsityConstraint"),
        }

    def _update_reweight_state(
        self, state, group_norm, init_sigma_reweight, update_tau_reweight
    ) -> None:
        if self.iterative_reweight is None:
            return
        if init_sigma_reweight:
            state["sigma"] = group_norm
        if "sigma" in state and update_tau_reweight:
            state["tau_reweight"] = self.iterative_reweight(group_norm, state["sigma"])

    def _svd_size_contribution(self, group, grouper, zero_elts) -> _SizeTotals:
        """Return the size contribution for an SVD grouper (summed variant)."""
        unfactored_size = grouper.U.size(0) * grouper.Vh.size(1)
        n_singular_vals = grouper.p.numel() - zero_elts
        factored_size = (grouper.U.size(0) + grouper.Vh.size(1)) * n_singular_vals
        group["factored_frac"] = factored_size / unfactored_size
        # Only factor matrices if it reduces params
        zeros = max(unfactored_size - factored_size, 0)
        return _SizeTotals(zeros, unfactored_size, factored_size, unfactored_size)

    def _make_group_context(self, group) -> _GroupContext:
        self._set_gamma(group)
        # apply shrinkage to latent parameters in place
        prox_map = instantiate_module(
            f"torchao.prototype.pat.optim.{group['prox_type']}"
        )(group["reg_lambda"], **self._get_prox_kwargs(group))
        # grouper is a context manager that reshapes p if needed
        grouper_cls = instantiate_module(
            f"torchao.prototype.pat.group.{group['group_type']}"
        )
        return _GroupContext(
            group=group,
            prox_map=prox_map,
            grouper_cls=grouper_cls,
            grouper_kwargs=self._get_grouper_kwargs(group),
            prox_kwargs=self._make_prox_kwargs(group),
        )

    def _record_relative_stats(self, totals: _SizeTotals, sctx: _StepContext) -> None:
        if sctx.dist_is_init and _is_main_process():
            totals.zeros += _sum_async_streams(sctx.zeros_buf)
            totals.factored_size += _sum_async_streams(sctx.factored_size_buf)

        if not _is_main_process():
            return
        self.relative_sparsity = (
            totals.zeros / totals.params if totals.params > 0 else 0.0
        )
        self.relative_factored_frac = (
            totals.factored_size / totals.unfactored_size
            if totals.unfactored_size > 0
            else 0.0
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        healing_masks = self._collect_healing_masks()

        if self._in_base_optimizer_only_phase():
            return self._base_optimizer_only_step(closure, healing_masks)

        self._update_latent_params()

        # call base optimizer step() method to update latent parameters
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]

        self._restore_temporary_state()

        init_sigma_reweight, update_tau_reweight = self._reweight_flags()
        sctx = _StepContext(
            init_sigma_reweight=init_sigma_reweight,
            update_tau_reweight=update_tau_reweight,
            dist_is_init=torch.distributed.is_initialized(),
        )

        totals = _SizeTotals()
        for group in self.regularized_param_groups():
            gctx = self._make_group_context(group)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                totals.add(self._process_param(p, gctx, sctx))

        self.num_steps += 1

        self._record_relative_stats(totals, sctx)

        return loss

    def _process_param(self, p, gctx: _GroupContext, sctx: _StepContext) -> _SizeTotals:
        """Apply the prox map to one param and return its size contribution.

        The contribution is all-zero when this rank does not run the grouper.
        """
        # save latent parameters
        state = self.state[p]
        state["latent"].copy_(p)

        # store the number of non-zero singular values
        if gctx.prox_kwargs["is_svd_grouper"]:
            npack = gctx.grouper_kwargs.get("npack", 1)
            state.setdefault(
                "sv_count", torch.zeros(npack, dtype=torch.int, device=p.device)
            )

        # update the full tensor if sharded
        sharded_p = None
        if _is_dtensor(p) and gctx.prox_kwargs["is_svd_grouper"]:
            sharded_p = p
            p = p.full_tensor()

        # only rank 0 of the device mesh should run the grouper
        sv_count = state.get("sv_count")
        totals = _SizeTotals()
        if sharded_p is None or sharded_p.device_mesh.get_rank() == 0:
            totals = self._run_grouper(p, state, gctx, sctx)

        # copy the updated full tensor to the sharded tensor
        if sharded_p is not None:
            torch.distributed.barrier()
            if isinstance(sv_count, Tensor):
                torch.distributed.broadcast(sv_count, src=0)
            sharded_p.copy_(
                distribute_tensor(
                    p,
                    device_mesh=sharded_p.device_mesh,
                    placements=sharded_p.placements,
                )
            )

        return totals

    def _run_grouper(
        self, p, state, gctx: _GroupContext, sctx: _StepContext
    ) -> _SizeTotals:
        """Run the grouper/prox map on rank 0 and return the size contribution."""
        grouper = gctx.grouper_cls(p, **gctx.grouper_kwargs)
        zero_elts, group_norm, zeros_are_summed = self._apply_prox(
            grouper,
            gctx.prox_map,
            p,
            tau_reweight=state.get("tau_reweight", 1.0),
            sv_count=state.get("sv_count"),
            **gctx.prox_kwargs,
        )

        if zeros_are_summed:
            state["sparsity_frac"] = zero_elts / grouper.p.numel()
        elif sctx.dist_is_init:
            _maybe_async_aggregate(sctx.zeros_buf, zero_elts)

        if torch.is_tensor(zero_elts):
            zero_elts = zero_elts.item()

        self._update_reweight_state(
            state, group_norm, sctx.init_sigma_reweight, sctx.update_tau_reweight
        )

        if not gctx.prox_kwargs["is_svd_grouper"]:
            return _SizeTotals(zeros=zero_elts, params=grouper.p.numel())

        totals = self._svd_size_contribution(gctx.group, grouper, zero_elts)
        # Only aggregate factored size directly if not already globally summed
        if zeros_are_summed:
            return totals
        _maybe_async_aggregate(
            sctx.factored_size_buf,
            torch.tensor(totals.factored_size, dtype=torch.int, device=p.device),
        )
        # factored_size is aggregated asynchronously instead of added directly
        totals.factored_size = 0
        return totals

    @torch._disable_dynamo
    def save_latent_params(self) -> None:
        """Save updated latent parameters before applying prox-map"""
        if self.warmup_steps == 0:
            assert len(self.state) == 0, "Expected empty state at first step()"
            # Maintain the invariant that `len(self.state) == 0` before first
            # self.base_optimizer.step() call by using a temporary state buffer
            self._state = defaultdict(dict)

        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    try:
                        self.state[p]["latent"].copy_(p)
                    except KeyError:
                        self.state[p]["latent"] = p.detach().clone()
