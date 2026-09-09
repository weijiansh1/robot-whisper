"""Dueling chunk critic, losses, and snapshot-level evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "state_action",
    "state_route",
    "state_action_route",
    "state_action_router_hidden",
    "state_as_route",
)


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, output_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class HiddenEncoder(nn.Module):
    def __init__(
        self, groups: int, layers: int, hidden_dim: int, projection: int, output_dim: int
    ) -> None:
        super().__init__()
        self.groups = int(groups)
        self.layers = int(layers)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.project = nn.Linear(hidden_dim, projection)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(groups * layers * projection, output_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-3:]) != (self.groups, self.layers, self.hidden_dim):
            raise ValueError(
                "router-hidden tensor has shape %s, expected tail %s"
                % (tuple(value.shape), (self.groups, self.layers, self.hidden_dim))
            )
        projected = self.project(self.norm(value))
        return self.output(projected.flatten(start_dim=-3))


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


class DuelingChunkCritic(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        vlm_dim: int,
        action_dim: int,
        route_candidate_dim: int,
        route_state_dim: int,
        hidden_candidate_shape: Sequence[int],
        hidden_state_shape: Sequence[int],
        width: int = 256,
        latent: int = 128,
        hidden_projection: int = 16,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown critic mode: {mode}")
        self.mode = mode
        self.latent = int(latent)
        self.use_action = mode in {
            "state_action",
            "state_action_route",
            "state_action_router_hidden",
        }
        self.use_route = mode in {"state_route", "state_action_route", "state_as_route"}
        self.use_hidden = mode == "state_action_router_hidden"
        self.vlm_encoder = VectorEncoder(vlm_dim, width, latent)
        self.proprio_encoder = VectorEncoder(8, max(32, width // 4), latent)
        self.external_state = nn.Sequential(
            nn.Linear(2 * latent, latent), nn.GELU()
        )
        self.action_encoder = VectorEncoder(2 * action_dim, width, latent)
        self.route_candidate_encoder = VectorEncoder(
            2 * route_candidate_dim, width, latent
        )
        self.route_state_encoder = VectorEncoder(route_state_dim, width, latent)
        candidate_groups, candidate_layers, candidate_width = map(
            int, hidden_candidate_shape
        )
        state_groups, state_layers, state_width = map(int, hidden_state_shape)
        if candidate_layers != state_layers or candidate_width != state_width:
            raise ValueError("candidate/state router-hidden axes disagree")
        self.hidden_candidate_encoder = HiddenEncoder(
            2 * candidate_groups,
            candidate_layers,
            candidate_width,
            hidden_projection,
            latent,
        )
        self.hidden_state_encoder = HiddenEncoder(
            state_groups,
            state_layers,
            state_width,
            hidden_projection,
            latent,
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(3 * latent, width), nn.GELU(), nn.Linear(width, latent), nn.GELU()
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(3 * latent, width), nn.GELU(), nn.Linear(width, latent), nn.GELU()
        )
        self.value = nn.Sequential(
            nn.Linear(latent, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(2 * latent, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def _zeros(self, reference: torch.Tensor, *shape: int) -> torch.Tensor:
        return reference.new_zeros(*shape, self.latent)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"].bool()
        batch_size, candidates = mask.shape
        external = self.external_state(
            torch.cat(
                [
                    self.vlm_encoder(batch["state_vlm"]),
                    self.proprio_encoder(batch["state_proprio"]),
                ],
                dim=-1,
            )
        )
        state_route = self._zeros(external, batch_size)
        state_hidden = self._zeros(external, batch_size)
        if self.use_route:
            state_route = self.route_state_encoder(batch["route_state"])
        if self.use_hidden:
            state_hidden = self.hidden_state_encoder(batch["hidden_state"])
        state = self.state_fusion(
            torch.cat([external, state_route, state_hidden], dim=-1)
        )

        action_slot = self._zeros(external, batch_size, candidates)
        route_slot = self._zeros(external, batch_size, candidates)
        hidden_slot = self._zeros(external, batch_size, candidates)
        if self.use_action:
            action = batch["action"]
            action_slot = self.action_encoder(
                torch.cat([action, action - _masked_mean(action, mask)], dim=-1)
            )
        if self.use_route:
            route = batch["route_candidate"]
            route_slot = self.route_candidate_encoder(
                torch.cat([route, route - _masked_mean(route, mask)], dim=-1)
            )
        if self.use_hidden:
            hidden = batch["hidden_candidate"]
            hidden_slot = self.hidden_candidate_encoder(
                torch.cat([hidden, hidden - _masked_mean(hidden, mask)], dim=-3)
            )
        candidate = self.candidate_fusion(
            torch.cat([action_slot, route_slot, hidden_slot], dim=-1)
        )
        advantage = self.advantage(
            torch.cat([state[:, None, :].expand(-1, candidates, -1), candidate], dim=-1)
        ).squeeze(-1)
        centered = advantage - _masked_mean(advantage[..., None], mask).squeeze(-1)
        logits = self.value(state) + centered
        return logits.masked_fill(~mask, 0.0)


def critic_loss(
    logits: torch.Tensor,
    success_count: torch.Tensor,
    repeat_count: torch.Tensor,
    mask: torch.Tensor,
    *,
    rank_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = mask.bool()
    q = success_count.float() / repeat_count.float().clamp_min(1.0)
    value_raw = F.binary_cross_entropy_with_logits(logits, q, reduction="none")
    value_loss = (
        (value_raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    ).mean()
    q_difference = q[:, :, None] - q[:, None, :]
    logit_difference = logits[:, :, None] - logits[:, None, :]
    pair_mask = (
        mask[:, :, None]
        & mask[:, None, :]
        & torch.triu(torch.ones_like(q_difference, dtype=torch.bool), diagonal=1)
        & (q_difference != 0)
    )
    pair_weights = q_difference.abs() * torch.sqrt(
        torch.minimum(repeat_count[:, :, None], repeat_count[:, None, :]).float()
    )
    rank_raw = F.softplus(-torch.sign(q_difference) * logit_difference)
    numerator = (rank_raw * pair_weights * pair_mask).sum(dim=(1, 2))
    denominator = (pair_weights * pair_mask).sum(dim=(1, 2))
    informative = denominator > 0
    rank_loss = (
        (numerator[informative] / denominator[informative]).mean()
        if bool(informative.any())
        else logits.sum() * 0.0
    )
    total = value_loss + float(rank_weight) * rank_loss
    return total, {
        "value_loss": float(value_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "pairs": int(pair_mask.sum().detach()),
    }


def snapshot_metrics(
    logits: np.ndarray,
    success_count: np.ndarray,
    repeat_count: np.ndarray,
    offsets: np.ndarray,
    task_ids: np.ndarray,
    state_indices: Sequence[int],
) -> list[dict[str, float | int]]:
    prediction = np.asarray(logits, dtype=np.float64)
    success = np.asarray(success_count, dtype=np.float64)
    repeats = np.asarray(repeat_count, dtype=np.float64)
    rows = []
    for state_index in state_indices:
        start, stop = int(offsets[state_index]), int(offsets[state_index + 1])
        q = success[start:stop] / repeats[start:stop]
        score = prediction[start:stop]
        left, right = np.triu_indices(len(q), k=1)
        different = q[left] != q[right]
        correct = (
            np.sign(score[left] - score[right]) == np.sign(q[left] - q[right])
        )[different]
        decisive = np.abs(q[left] - q[right])[different] >= 0.5
        selected = int(np.argmax(score))
        random_q, oracle_q, selected_q = float(q.mean()), float(q.max()), float(q[selected])
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
        epsilon = 1e-8
        nll = -np.mean(q * np.log(probability + epsilon) + (1 - q) * np.log(1 - probability + epsilon))
        rows.append(
            {
                "state_index": int(state_index),
                "task_id": int(task_ids[state_index]),
                "pair_correct": int(correct.sum()),
                "pair_total": int(len(correct)),
                "pairwise_accuracy": float(correct.mean()) if len(correct) else float("nan"),
                "decisive_correct": int(correct[decisive].sum()),
                "decisive_total": int(decisive.sum()),
                "decisive_pairwise_accuracy": (
                    float(correct[decisive].mean()) if decisive.any() else float("nan")
                ),
                "top1_regret": oracle_q - selected_q,
                "selected_q": selected_q,
                "random_q": random_q,
                "oracle_q": oracle_q,
                "brier": float(np.mean((probability - q) ** 2)),
                "nll": float(nll),
            }
        )
    return rows


def aggregate_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    if not rows:
        raise ValueError("metrics require at least one snapshot")
    by_task: dict[int, list[Mapping[str, float | int]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)

    def task_macro(name: str) -> float:
        return float(
            np.mean(
                [np.mean([float(row[name]) for row in values]) for values in by_task.values()]
            )
        )

    pair_task = []
    decisive_task = []
    recovery_task = []
    for values in by_task.values():
        correct = sum(int(row["pair_correct"]) for row in values)
        total = sum(int(row["pair_total"]) for row in values)
        decisive_correct = sum(int(row["decisive_correct"]) for row in values)
        decisive_total = sum(int(row["decisive_total"]) for row in values)
        pair_task.append(correct / total if total else np.nan)
        decisive_task.append(decisive_correct / decisive_total if decisive_total else np.nan)
        selected = np.mean([float(row["selected_q"]) for row in values])
        random = np.mean([float(row["random_q"]) for row in values])
        oracle = np.mean([float(row["oracle_q"]) for row in values])
        recovery_task.append((selected - random) / (oracle - random) if oracle > random else np.nan)
    pair_finite = [value for value in pair_task if np.isfinite(value)]
    decisive_finite = [value for value in decisive_task if np.isfinite(value)]
    recovery_finite = [value for value in recovery_task if np.isfinite(value)]
    return {
        "snapshots": float(len(rows)),
        "tasks": float(len(by_task)),
        "pairwise_accuracy": float(np.mean(pair_finite)) if pair_finite else 0.5,
        "decisive_pairwise_accuracy": (
            float(np.mean(decisive_finite)) if decisive_finite else 0.5
        ),
        "top1_regret": task_macro("top1_regret"),
        "selected_q": task_macro("selected_q"),
        "random_q": task_macro("random_q"),
        "oracle_q": task_macro("oracle_q"),
        "oracle_recovery_ratio": float(np.mean(recovery_finite)) if recovery_finite else 0.0,
        "brier": task_macro("brier"),
        "binomial_proportion_nll": task_macro("nll"),
    }


def hierarchical_delta_ci(
    left: Sequence[Mapping[str, float | int]],
    right: Sequence[Mapping[str, float | int]],
    *,
    field: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    left_map = {int(row["state_index"]): row for row in left}
    right_map = {int(row["state_index"]): row for row in right}
    if set(left_map) != set(right_map):
        raise ValueError("paired metric rows do not cover the same snapshots")
    by_task: dict[int, list[float]] = defaultdict(list)
    for index in sorted(left_map):
        task = int(left_map[index]["task_id"])
        difference = float(left_map[index][field]) - float(right_map[index][field])
        if np.isfinite(difference):
            by_task[task].append(difference)
    tasks = sorted(by_task)
    if not tasks or any(not by_task[task] for task in tasks):
        raise ValueError("paired metric has no finite snapshot values")
    point = float(np.mean([np.mean(by_task[task]) for task in tasks]))
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        values = []
        for task_slot in rng.integers(0, len(tasks), size=len(tasks)):
            candidates = np.asarray(by_task[tasks[int(task_slot)]])
            values.append(
                float(candidates[rng.integers(0, len(candidates), size=len(candidates))].mean())
            )
        distribution[draw] = np.mean(values)
    return point, [float(value) for value in np.percentile(distribution, [2.5, 97.5])]
