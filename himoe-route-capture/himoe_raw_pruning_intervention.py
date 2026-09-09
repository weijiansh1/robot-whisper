"""Request-selectable HB5/d0 pruning using runtime internal activations."""

from __future__ import annotations

import torch

from himoe_hb5_intervention import (
    HB5InterventionResult,
    categorical_drop_slots,
    stable_seed,
)


RAW_PRUNING_ARMS = ("baseline", "drop")
RAW_PRUNING_POLICIES = (
    "baseline",
    "min_internal_l2",
    "max_internal_l2",
    "min_weight",
    "max_weight",
    "min_raw_output",
    "max_raw_output",
    "uniform_random",
    "categorical_weighted",
)
POLICY_TO_CODE = {policy: index for index, policy in enumerate(RAW_PRUNING_POLICIES)}
CODE_TO_POLICY = {value: key for key, value in POLICY_TO_CODE.items()}
PREREGISTERED_POLICIES = (
    "baseline",
    "min_internal_l2",
    "min_weight",
    "min_raw_output",
    "uniform_random",
)


def select_raw_pruning_slots(
    weights: torch.Tensor,
    raw_output: torch.Tensor,
    internal_l2: torch.Tensor,
    *,
    pair_id: int,
    draw_id: int,
    policy: str,
) -> torch.Tensor:
    """Return one selected slot per action token under a frozen policy."""
    if policy not in RAW_PRUNING_POLICIES or policy == "baseline":
        raise ValueError("a drop row requires a non-baseline raw-pruning policy")
    if weights.ndim != 3 or raw_output.ndim != 4:
        raise ValueError("weights/raw output have incompatible ranks")
    if raw_output.shape[:-1] != weights.shape or internal_l2.shape != weights.shape:
        raise ValueError("weights, raw output, and internal L2 axes must match")
    if policy == "min_internal_l2":
        return internal_l2.argmin(dim=-1)
    if policy == "max_internal_l2":
        return internal_l2.argmax(dim=-1)
    if policy == "min_weight":
        return weights.argmin(dim=-1)
    if policy == "max_weight":
        return weights.argmax(dim=-1)
    output_l2 = raw_output.float().square().sum(dim=-1).sqrt()
    if policy == "min_raw_output":
        return output_l2.argmin(dim=-1)
    if policy == "max_raw_output":
        return output_l2.argmax(dim=-1)
    if policy == "categorical_weighted":
        return categorical_drop_slots(weights, pair_id, draw_id)

    batch, actions, top_k = weights.shape
    slots = torch.empty((batch, actions), dtype=torch.long, device=weights.device)
    for batch_index in range(batch):
        for action_index in range(actions):
            token_index = batch_index * actions + action_index
            slots[batch_index, action_index] = (
                stable_seed(pair_id, draw_id, token_index, "uniform") % top_k
            )
    return slots


class HB5D0RawPruningIntervention:
    """Apply a request-selected ordinary drop-one counterfactual."""

    arms = RAW_PRUNING_ARMS
    policies = RAW_PRUNING_POLICIES

    def __init__(
        self,
        layer: int = 5,
        denoise: int = 0,
        drop_policy: str = "baseline",
    ) -> None:
        self.layer = int(layer)
        self.denoise = int(denoise)
        self.drop_policy = str(drop_policy)
        if self.layer != 5 or self.denoise != 0:
            raise ValueError("the frozen raw-pruning cell is HB5/d0")
        if self.drop_policy not in RAW_PRUNING_POLICIES:
            raise ValueError("invalid default raw-pruning policy")

    def apply(
        self,
        routed_flat: torch.Tensor,
        raw_flat: torch.Tensor,
        ids_flat: torch.Tensor,
        weights_flat: torch.Tensor,
        *,
        n_action_steps: int,
        pair_id: int,
        draw_id: int,
        arm: str,
        drop_policy: str | None = None,
        internal_l2_flat: torch.Tensor | None = None,
    ) -> HB5InterventionResult:
        policy = self.drop_policy if drop_policy is None else str(drop_policy)
        if arm not in self.arms or policy not in self.policies:
            raise ValueError("invalid raw-pruning arm or policy")
        if (arm == "baseline") != (policy == "baseline"):
            raise ValueError("baseline arm and baseline policy must occur together")
        suffix = int(n_action_steps) + 1
        if routed_flat.ndim != 2 or routed_flat.shape[0] % suffix:
            raise ValueError("routed tensor is incompatible with state+action suffix")
        batch = routed_flat.shape[0] // suffix
        hidden = routed_flat.shape[-1]
        if raw_flat.ndim != 3 or raw_flat.shape[0] != batch * suffix:
            raise ValueError("raw expert vectors are incompatible with routed output")
        top_k = raw_flat.shape[1]
        if raw_flat.shape[-1] != hidden:
            raise ValueError("raw expert output hidden width does not match routed")
        if ids_flat.shape != (batch * suffix, top_k):
            raise ValueError("selected expert IDs have an incompatible shape")
        if weights_flat.shape != (batch * suffix, top_k):
            raise ValueError("selected expert weights have an incompatible shape")
        if internal_l2_flat is None or internal_l2_flat.shape != (
            batch * suffix,
            top_k,
        ):
            raise ValueError("runtime internal L2 must cover every selected slot")
        if not torch.isfinite(internal_l2_flat).all():
            raise ValueError("runtime internal L2 contains unfilled slots")

        original_all = routed_flat.float()
        original = original_all.reshape(batch, suffix, hidden)[:, 1:].clone()
        raw = raw_flat.float().reshape(batch, suffix, top_k, hidden)[:, 1:]
        ids = ids_flat.reshape(batch, suffix, top_k)[:, 1:]
        weights = weights_flat.float().reshape(batch, suffix, top_k)[:, 1:]
        internal_l2 = internal_l2_flat.float().reshape(batch, suffix, top_k)[:, 1:]

        if policy == "baseline":
            dropped_slot = torch.full(
                (batch, n_action_steps), -1, dtype=torch.long, device=routed_flat.device
            )
            dropped_expert_id = torch.full(
                (batch, n_action_steps), 255, dtype=torch.long, device=routed_flat.device
            )
            delta = torch.zeros_like(original)
            executed_flat = routed_flat
        else:
            dropped_slot = select_raw_pruning_slots(
                weights,
                raw,
                internal_l2,
                pair_id=pair_id,
                draw_id=draw_id,
                policy=policy,
            )
            gather_vector = dropped_slot[..., None, None].expand(
                -1, -1, 1, hidden
            )
            selected_raw = raw.gather(2, gather_vector).squeeze(2)
            dropped_weight = weights.gather(
                -1, dropped_slot[..., None]
            ).squeeze(-1)
            if bool((dropped_weight >= 1.0 - 1e-7).any()):
                raise RuntimeError("drop-one renormalization is singular")
            dropped_expert_id = ids.gather(
                -1, dropped_slot[..., None]
            ).squeeze(-1)
            drop_routed = (
                original - dropped_weight[..., None] * selected_raw
            ) / (1.0 - dropped_weight[..., None])
            delta = drop_routed - original
            executed_all = original_all.clone().reshape(batch, suffix, hidden)
            executed_all[:, 1:] = original + delta
            if not torch.equal(
                executed_all[:, 0], original_all.reshape(batch, suffix, hidden)[:, 0]
            ):
                raise RuntimeError("raw-pruning intervention changed the state token")
            executed_flat = executed_all.reshape_as(original_all)

        executed = executed_flat.float().reshape(batch, suffix, hidden)[:, 1:].clone()
        applied = executed - original
        return HB5InterventionResult(
            original_routed=original.detach(),
            executed_routed=executed.detach(),
            executed_routed_flat=executed_flat.detach(),
            intervention_delta=applied.detach(),
            dropped_slot=dropped_slot.detach(),
            dropped_expert_id=dropped_expert_id.detach(),
        )
