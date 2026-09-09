"""Deterministic paired interventions on HB5/d0 action-token routing.

The intervention is deliberately applied after the deployed top-k experts have
run.  It therefore preserves the original hidden input, router decision, gate
weights, and pre-gate expert vectors for a paired causal comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np
import torch


ARMS = ("baseline", "drop", "random")
ARM_TO_CODE = {arm: index for index, arm in enumerate(ARMS)}
CODE_TO_ARM = {value: key for key, value in ARM_TO_CODE.items()}
DROP_POLICIES = (
    "categorical_weighted",
    "min_weight",
    "max_weight",
    "min_raw_output_rms",
    "max_raw_output_rms",
)


@dataclass
class HB5InterventionResult:
    """Runtime tensors for one intervention at the probe cell."""

    original_routed: torch.Tensor
    executed_routed: torch.Tensor
    executed_routed_flat: torch.Tensor
    intervention_delta: torch.Tensor
    dropped_slot: torch.Tensor
    dropped_expert_id: torch.Tensor


def stable_seed(
    pair_id: int,
    draw_id: int,
    token_index: int,
    purpose: str,
) -> int:
    """Return a process-independent 63-bit seed for one paired token."""
    if min(pair_id, draw_id, token_index) < 0:
        raise ValueError("pair_id, draw_id, and token_index must be non-negative")
    payload = struct.pack("<qqq", int(pair_id), int(draw_id), int(token_index))
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=("hb5-" + purpose).encode("ascii")[:16],
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _uniform_from_seed(seed: int) -> float:
    # Use the midpoint of a 53-bit cell, so the draw is strictly inside (0, 1).
    mantissa = (int(seed) >> 10) & ((1 << 53) - 1)
    return (mantissa + 0.5) / float(1 << 53)


def categorical_drop_slots(
    weights: torch.Tensor,
    pair_id: int,
    draw_id: int,
) -> torch.Tensor:
    """Draw one top-k slot per action token from normalized gate weights."""
    if weights.ndim != 3:
        raise ValueError("weights must be [batch, action, top_k]")
    values = weights.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("expert weights must be finite and non-negative")
    normalizer = values.sum(axis=-1, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise ValueError("expert weights must have positive row sums")
    values = values / normalizer
    batch, actions, _top_k = values.shape
    slots = np.empty((batch, actions), dtype=np.int64)
    for batch_index in range(batch):
        for action_index in range(actions):
            token_index = batch_index * actions + action_index
            seed = stable_seed(pair_id, draw_id, token_index, "drop")
            draw = _uniform_from_seed(seed)
            slot = int(
                np.searchsorted(
                    np.cumsum(values[batch_index, action_index]),
                    draw,
                    side="right",
                )
            )
            slots[batch_index, action_index] = min(slot, values.shape[-1] - 1)
    return torch.as_tensor(slots, dtype=torch.long, device=weights.device)


def select_drop_slots(
    weights: torch.Tensor,
    raw: torch.Tensor,
    pair_id: int,
    draw_id: int,
    policy: str,
) -> torch.Tensor:
    """Select one expert slot under a named, auditable policy."""
    if policy not in DROP_POLICIES:
        raise ValueError("drop policy must be one of %s" % (DROP_POLICIES,))
    if weights.ndim != 3 or raw.ndim != 4 or raw.shape[:-1] != weights.shape:
        raise ValueError("weights/raw must be [batch,action,top_k,(hidden)]")
    if policy == "categorical_weighted":
        return categorical_drop_slots(weights, pair_id, draw_id)
    if policy == "min_weight":
        return weights.argmin(dim=-1)
    if policy == "max_weight":
        return weights.argmax(dim=-1)
    raw_rms = raw.float().square().mean(dim=-1).sqrt()
    if policy == "min_raw_output_rms":
        return raw_rms.argmin(dim=-1)
    return raw_rms.argmax(dim=-1)


def _deterministic_normal(
    width: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.randn(
            width, generator=generator, device=device, dtype=torch.float32
        )
    except (RuntimeError, TypeError):
        # Older torch releases do not accept every device type in Generator.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(width, generator=generator, dtype=torch.float32).to(device)


def matched_random_delta(
    routed: torch.Tensor,
    drop_delta: torch.Tensor,
    pair_id: int,
    draw_id: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Randomize direction while matching drop norm and radial component.

    For nonzero ``r``, the result has the same ``||delta||`` and
    ``<delta, r>`` as ``drop_delta``.  Consequently ``||r + delta||`` also
    matches.  When ``r`` is numerically zero only the delta norm is matched.
    """
    if routed.shape != drop_delta.shape or routed.ndim != 3:
        raise ValueError("routed and drop_delta must match [batch, action, hidden]")
    r = routed.float()
    target = drop_delta.float()
    output = torch.empty_like(target)
    batch, actions, width = r.shape
    for batch_index in range(batch):
        for action_index in range(actions):
            token_index = batch_index * actions + action_index
            seed = stable_seed(pair_id, draw_id, token_index, "random")
            random = _deterministic_normal(width, seed, r.device)
            current = r[batch_index, action_index]
            delta = target[batch_index, action_index]
            r_norm = torch.linalg.vector_norm(current)
            delta_norm = torch.linalg.vector_norm(delta)
            if float(r_norm) <= eps:
                q_norm = torch.linalg.vector_norm(random)
                if float(q_norm) <= eps:
                    random = torch.ones_like(random)
                    q_norm = torch.linalg.vector_norm(random)
                output[batch_index, action_index] = random * (delta_norm / q_norm)
                continue

            r_unit = current / r_norm
            radial = torch.dot(delta, r_unit)
            tangent = random - torch.dot(random, r_unit) * r_unit
            tangent_norm = torch.linalg.vector_norm(tangent)
            if float(tangent_norm) <= eps:
                # Choose the coordinate axis least aligned with r as a stable
                # fallback, then project it into the orthogonal complement.
                axis = torch.zeros_like(current)
                axis[torch.argmin(r_unit.abs())] = 1.0
                tangent = axis - torch.dot(axis, r_unit) * r_unit
                tangent_norm = torch.linalg.vector_norm(tangent)
            tangent_size = torch.sqrt(
                torch.clamp(delta_norm.square() - radial.square(), min=0.0)
            )
            output[batch_index, action_index] = (
                radial * r_unit + tangent_size * tangent / tangent_norm
            )
    return output


class HB5D0Intervention:
    """Apply one frozen arm only to HB5/d0 action-token routed outputs."""

    arms = ARMS
    policies = DROP_POLICIES

    def __init__(
        self,
        layer: int = 5,
        denoise: int = 0,
        drop_policy: str = "categorical_weighted",
    ) -> None:
        self.layer = int(layer)
        self.denoise = int(denoise)
        if self.layer != 5 or self.denoise != 0:
            raise ValueError("the frozen intervention cell is HB5/d0")
        if drop_policy not in DROP_POLICIES:
            raise ValueError("drop policy must be one of %s" % (DROP_POLICIES,))
        self.drop_policy = drop_policy

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
        if arm not in ARMS:
            raise ValueError("arm must be one of %s" % (ARMS,))
        suffix = int(n_action_steps) + 1
        if routed_flat.ndim != 2 or routed_flat.shape[0] % suffix:
            raise ValueError("routed tensor is incompatible with state+action suffix")
        batch = routed_flat.shape[0] // suffix
        hidden = routed_flat.shape[-1]
        if raw_flat.shape[:1] != routed_flat.shape[:1] or raw_flat.shape[-1] != hidden:
            raise ValueError("raw expert vectors are incompatible with routed output")
        top_k = raw_flat.shape[1]
        if ids_flat.shape != (batch * suffix, top_k):
            raise ValueError("selected expert IDs have an incompatible shape")
        if weights_flat.shape != (batch * suffix, top_k):
            raise ValueError("selected expert weights have an incompatible shape")

        original_all = routed_flat.float()
        original = original_all.reshape(batch, suffix, hidden)[:, 1:].clone()
        raw = raw_flat.float().reshape(batch, suffix, top_k, hidden)[:, 1:]
        ids = ids_flat.reshape(batch, suffix, top_k)[:, 1:]
        weights = weights_flat.float().reshape(batch, suffix, top_k)[:, 1:]
        policy = self.drop_policy if drop_policy is None else str(drop_policy)
        dropped_slot = select_drop_slots(
            weights,
            raw,
            pair_id=pair_id,
            draw_id=draw_id,
            policy=policy,
        )
        gather_vector = dropped_slot[..., None, None].expand(-1, -1, 1, hidden)
        selected_raw = raw.gather(2, gather_vector).squeeze(2)
        dropped_weight = weights.gather(-1, dropped_slot[..., None]).squeeze(-1)
        if bool((dropped_weight >= 1.0 - 1e-7).any()):
            raise RuntimeError(
                "drop-one renormalization is singular for weight near one"
            )
        dropped_expert_id = ids.gather(-1, dropped_slot[..., None]).squeeze(-1)
        drop_routed = (original - dropped_weight[..., None] * selected_raw) / (
            1.0 - dropped_weight[..., None]
        )
        drop_delta = drop_routed - original

        if arm == "baseline":
            delta = torch.zeros_like(drop_delta)
        elif arm == "drop":
            delta = drop_delta
        else:
            delta = matched_random_delta(
                original, drop_delta, pair_id=pair_id, draw_id=draw_id
            )

        if arm == "baseline":
            # Return the original object for a strict no-op audit.
            executed_flat = routed_flat
        else:
            executed_all = original_all.clone().reshape(batch, suffix, hidden)
            executed_all[:, 1:] = original + delta
            if not torch.equal(
                executed_all[:, 0], original_all.reshape(batch, suffix, hidden)[:, 0]
            ):
                raise RuntimeError("the intervention changed the state token")
            executed_flat = executed_all.reshape_as(original_all)

        executed = executed_flat.float().reshape(batch, suffix, hidden)[:, 1:].clone()
        applied = executed - original
        if arm == "random":
            norm_error = (
                torch.linalg.vector_norm(applied, dim=-1)
                - torch.linalg.vector_norm(drop_delta, dim=-1)
            ).abs()
            radial_error = ((applied - drop_delta) * original).sum(dim=-1).abs()
            post_norm_error = (
                torch.linalg.vector_norm(original + applied, dim=-1)
                - torch.linalg.vector_norm(original + drop_delta, dim=-1)
            ).abs()
            scale = torch.linalg.vector_norm(drop_delta, dim=-1).clamp_min(1.0)
            if bool((norm_error > 2e-5 * scale).any()):
                raise RuntimeError("random control failed delta-norm matching")
            radial_scale = original.square().sum(dim=-1).sqrt().clamp_min(1.0) * scale
            if bool((radial_error > 3e-5 * radial_scale).any()):
                raise RuntimeError("random control failed radial matching")
            if bool((post_norm_error > 3e-5 * scale).any()):
                raise RuntimeError("random control failed post-routed norm matching")

        return HB5InterventionResult(
            original_routed=original.detach(),
            executed_routed=executed.detach(),
            executed_routed_flat=executed_flat.detach(),
            intervention_delta=applied.detach(),
            dropped_slot=dropped_slot.detach(),
            dropped_expert_id=dropped_expert_id.detach(),
        )
