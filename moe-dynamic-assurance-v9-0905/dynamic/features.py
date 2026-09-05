"""Full-router, train-free MoE dynamics for HiMoE-VLA."""

from __future__ import annotations

import math

import numpy as np
import torch


LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
TOKEN_NAMES = ("state", "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10")
TOKEN_GROUPS = ("state", "action")
GRAPH_STATS = (
    "commitment",
    "top1_mass",
    "top4_mass",
    "token_affinity",
    "token_expert_mi",
    "state_action_affinity",
)
LAYER_GROUPS = {
    "all": slice(0, 8),
    "early": slice(0, 4),
    "late": slice(4, 8),
}
EPSILON = 1e-8
SCHEMA = "himoe.dynamic_moe_profile.v1"


def within_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for family, steps in (
        ("flow_speed", 9),
        ("flow_accel", 8),
        ("flow_turn", 8),
        ("flow_flux", 9),
    ):
        for layer in LAYER_NAMES:
            for group in TOKEN_GROUPS:
                names.extend(f"{family}|{layer}|{group}|t{step}" for step in range(steps))
    for layer in LAYER_NAMES:
        for token in TOKEN_NAMES:
            for metric in ("token_path", "token_accel", "token_endpoint", "token_efficiency"):
                names.append(f"{metric}|{layer}|{token}")
    for layer in LAYER_NAMES:
        for flow in range(10):
            for statistic in GRAPH_STATS:
                names.append(f"graph_level|{layer}|f{flow}|{statistic}")
    for layer in LAYER_NAMES:
        for statistic in GRAPH_STATS:
            for metric in ("graph_endpoint", "graph_tv", "graph_curvature"):
                names.append(f"{metric}|{layer}|{statistic}")
    for layer in LAYER_NAMES:
        for expert in range(32):
            for metric in ("expert_endpoint", "expert_tv"):
                names.append(f"{metric}|{layer}|e{expert:02d}")
    for layer_group in LAYER_GROUPS:
        for token_group in TOKEN_GROUPS:
            for metric in (
                "flow_speed_mean",
                "flow_accel_mean",
                "flow_turn_instability",
                "flow_flux_mean",
            ):
                names.append(f"aggregate|{layer_group}|{token_group}|{metric}")
        names.extend(
            (
                f"aggregate|{layer_group}|flow_graph_tv_mean",
                f"aggregate|{layer_group}|flow_graph_curvature_mean",
                f"aggregate|{layer_group}|expert_load_tv_mean",
            )
        )
    return tuple(names)


def query_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for layer in LAYER_NAMES:
        for group in TOKEN_GROUPS:
            names.extend(f"query_recurrence|{layer}|{group}|lag{lag}" for lag in range(1, 9))
    for layer in LAYER_NAMES:
        for group in TOKEN_GROUPS:
            for metric in ("query_accel", "query_jerk", "query_turn"):
                names.append(f"{metric}|{layer}|{group}")
    for layer in LAYER_NAMES:
        for statistic in GRAPH_STATS:
            for derivative in ("query_graph_d1", "query_graph_d2", "query_graph_d3"):
                names.append(f"{derivative}|{layer}|{statistic}")
    for layer_group in LAYER_GROUPS:
        for token_group in TOKEN_GROUPS:
            names.extend(
                f"query_aggregate|{layer_group}|{token_group}|recurrence_lag{lag}"
                for lag in range(1, 9)
            )
            names.extend(
                (
                    f"query_aggregate|{layer_group}|{token_group}|query_accel",
                    f"query_aggregate|{layer_group}|{token_group}|query_jerk",
                    f"query_aggregate|{layer_group}|{token_group}|query_turn_instability",
                )
            )
        names.extend(
            (
                f"query_aggregate|{layer_group}|graph_d1_abs",
                f"query_aggregate|{layer_group}|graph_d2_abs",
                f"query_aggregate|{layer_group}|graph_d3_abs",
            )
        )
    return tuple(names)


WITHIN_FEATURE_NAMES = within_feature_names()
QUERY_FEATURE_NAMES = query_feature_names()


def _group_tokens(values: torch.Tensor) -> torch.Tensor:
    """Convert [..., 11 tokens] to [..., state/action] before time axis."""

    return torch.stack((values[..., 0], values[..., 1:].mean(dim=-1)), dim=2)


def _safe_turn(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dot = (left * right).sum(dim=-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return torch.where(
        denominator > EPSILON,
        (dot / denominator.clamp_min(EPSILON)).clamp(-1.0, 1.0),
        torch.ones_like(dot),
    )


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)


def compute_within_flow(router_probability: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute dynamics from `[batch,8,10,11,32]` soft routing."""

    if router_probability.ndim != 5 or tuple(router_probability.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected router shape: {tuple(router_probability.shape)}")
    probability = router_probability.float().clamp_min(0.0)
    probability /= probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    root = probability.sqrt()

    speed_token = torch.linalg.vector_norm(root[:, :, 1:] - root[:, :, :-1], dim=-1) / math.sqrt(2.0)
    second = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
    accel_token = torch.linalg.vector_norm(second, dim=-1) / math.sqrt(2.0)
    velocity = root[:, :, 1:] - root[:, :, :-1]
    turn_token = _safe_turn(velocity[:, :, :-1], velocity[:, :, 1:])
    flux_token = 0.5 * torch.abs(probability[:, :, 1:] - probability[:, :, :-1]).sum(dim=-1)

    speed = _group_tokens(speed_token)
    acceleration = _group_tokens(accel_token)
    turn = _group_tokens(turn_token)
    flux = _group_tokens(flux_token)

    endpoint_token = torch.linalg.vector_norm(root[:, :, -1] - root[:, :, 0], dim=-1) / math.sqrt(2.0)
    token_path = speed_token.sum(dim=2)
    token_summary = torch.stack(
        (
            token_path,
            accel_token.mean(dim=2),
            endpoint_token,
            endpoint_token / token_path.clamp_min(EPSILON),
        ),
        dim=-1,
    )

    action_probability = probability[:, :, :, 1:, :]
    action_root = root[:, :, :, 1:, :]
    row_entropy = _entropy(action_probability)
    commitment = 1.0 - row_entropy.mean(dim=3) / math.log(32.0)
    top4 = torch.topk(action_probability, k=4, dim=-1).values
    top1_mass = top4[..., 0].mean(dim=3)
    top4_mass = top4.sum(dim=-1).mean(dim=3)
    root_sum = action_root.sum(dim=3)
    token_affinity = ((root_sum.square().sum(dim=-1) - 10.0) / 90.0).clamp(0.0, 1.0)
    mean_probability = action_probability.mean(dim=3)
    token_expert_mi = (_entropy(mean_probability) - row_entropy.mean(dim=3)) / math.log(32.0)
    state_action_affinity = (
        root[:, :, :, :1, :] * action_root
    ).sum(dim=-1).mean(dim=3)
    graph = torch.stack(
        (
            commitment,
            top1_mass,
            top4_mass,
            token_affinity,
            token_expert_mi,
            state_action_affinity,
        ),
        dim=-1,
    )
    graph_delta = graph[:, :, 1:] - graph[:, :, :-1]
    graph_second = graph[:, :, 2:] - 2.0 * graph[:, :, 1:-1] + graph[:, :, :-2]
    graph_dynamic = torch.stack(
        (
            graph[:, :, -1] - graph[:, :, 0],
            graph_delta.abs().sum(dim=2),
            graph_second.abs().mean(dim=2),
        ),
        dim=-1,
    )

    expert_load = action_probability.mean(dim=3)
    expert_dynamic = torch.stack(
        (
            expert_load[:, :, -1] - expert_load[:, :, 0],
            torch.abs(expert_load[:, :, 1:] - expert_load[:, :, :-1]).sum(dim=2),
        ),
        dim=-1,
    )

    aggregates = []
    for layer_slice in LAYER_GROUPS.values():
        for group in range(2):
            aggregates.extend(
                (
                    speed[:, layer_slice, group].mean(dim=(1, 2)),
                    acceleration[:, layer_slice, group].mean(dim=(1, 2)),
                    (1.0 - turn[:, layer_slice, group]).mean(dim=(1, 2)),
                    flux[:, layer_slice, group].mean(dim=(1, 2)),
                )
            )
        aggregates.extend(
            (
                graph_dynamic[:, layer_slice, :, 1].mean(dim=(1, 2)),
                graph_dynamic[:, layer_slice, :, 2].mean(dim=(1, 2)),
                expert_dynamic[:, layer_slice, :, 1].mean(dim=(1, 2)),
            )
        )
    aggregate = torch.stack(aggregates, dim=-1)

    blocks = (speed, acceleration, turn, flux, token_summary, graph, graph_dynamic, expert_dynamic, aggregate)
    features = torch.cat([block.reshape(len(probability), -1) for block in blocks], dim=1)
    if features.shape[1] != len(WITHIN_FEATURE_NAMES):
        raise RuntimeError(f"within-flow schema mismatch: {features.shape[1]} != {len(WITHIN_FEATURE_NAMES)}")
    return {
        "features": features.detach().cpu().numpy(),
        "final_root": root[:, :, -1].detach().cpu().numpy(),
        "final_graph": graph[:, :, -1].detach().cpu().numpy(),
    }


def _numpy_group(values: np.ndarray) -> np.ndarray:
    return np.stack((values[..., 0], values[..., 1:].mean(axis=-1)), axis=2)


def compute_query_dynamics(
    final_root: np.ndarray,
    final_graph: np.ndarray,
    episode_id: np.ndarray,
    control_step: np.ndarray,
) -> np.ndarray:
    """Compute prefix-causal inter-query dynamics for one task."""

    root = np.asarray(final_root, dtype=np.float32)
    graph = np.asarray(final_graph, dtype=np.float32)
    count = len(root)
    recurrence = np.full((count, 8, 2, 8), np.nan, dtype=np.float32)
    kinematics = np.full((count, 8, 2, 3), np.nan, dtype=np.float32)
    graph_derivative = np.full((count, 8, len(GRAPH_STATS), 3), np.nan, dtype=np.float32)

    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        route = root[rows]
        graph_values = graph[rows]
        for lag in range(1, 9):
            if len(rows) <= lag:
                continue
            distance = np.linalg.norm(route[lag:] - route[:-lag], axis=-1) / np.sqrt(2.0)
            recurrence[rows[lag:], :, :, lag - 1] = _numpy_group(distance)
        if len(rows) >= 3:
            second = route[2:] - 2.0 * route[1:-1] + route[:-2]
            acceleration = np.linalg.norm(second, axis=-1) / np.sqrt(2.0)
            velocity_left = route[1:-1] - route[:-2]
            velocity_right = route[2:] - route[1:-1]
            numerator = (velocity_left * velocity_right).sum(axis=-1)
            denominator = np.linalg.norm(velocity_left, axis=-1) * np.linalg.norm(velocity_right, axis=-1)
            turn = np.ones_like(numerator)
            np.divide(numerator, denominator, out=turn, where=denominator > EPSILON)
            kinematics[rows[2:], :, :, 0] = _numpy_group(acceleration)
            kinematics[rows[2:], :, :, 2] = _numpy_group(np.clip(turn, -1.0, 1.0))
        if len(rows) >= 4:
            third = route[3:] - 3.0 * route[2:-1] + 3.0 * route[1:-2] - route[:-3]
            jerk = np.linalg.norm(third, axis=-1) / np.sqrt(2.0)
            kinematics[rows[3:], :, :, 1] = _numpy_group(jerk)

        graph_derivative[rows[1:], :, :, 0] = graph_values[1:] - graph_values[:-1]
        if len(rows) >= 3:
            graph_derivative[rows[2:], :, :, 1] = (
                graph_values[2:] - 2.0 * graph_values[1:-1] + graph_values[:-2]
            )
        if len(rows) >= 4:
            graph_derivative[rows[3:], :, :, 2] = (
                graph_values[3:]
                - 3.0 * graph_values[2:-1]
                + 3.0 * graph_values[1:-2]
                - graph_values[:-3]
            )

    aggregates = []
    for layer_slice in LAYER_GROUPS.values():
        for group in range(2):
            aggregates.extend(
                recurrence[:, layer_slice, group, lag].mean(axis=1)
                for lag in range(8)
            )
            aggregates.extend(
                (
                    kinematics[:, layer_slice, group, 0].mean(axis=1),
                    kinematics[:, layer_slice, group, 1].mean(axis=1),
                    (1.0 - kinematics[:, layer_slice, group, 2]).mean(axis=1),
                )
            )
        aggregates.extend(
            np.abs(graph_derivative[:, layer_slice, :, derivative]).mean(axis=(1, 2))
            for derivative in range(3)
        )
    aggregate = np.column_stack(aggregates).astype(np.float32)
    blocks = (
        recurrence.reshape(count, -1),
        kinematics.reshape(count, -1),
        graph_derivative.reshape(count, -1),
        aggregate,
    )
    features = np.column_stack(blocks).astype(np.float32)
    if features.shape[1] != len(QUERY_FEATURE_NAMES):
        raise RuntimeError(f"query schema mismatch: {features.shape[1]} != {len(QUERY_FEATURE_NAMES)}")
    return features


def episode_local_query(episode_id: np.ndarray, control_step: np.ndarray) -> np.ndarray:
    query = np.empty(len(episode_id), dtype=np.int16)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        query[rows] = np.arange(len(rows), dtype=np.int16)
    return query


def compute_dynamic_history(
    router_history: torch.Tensor,
    control_step: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build prefix-causal features for one online episode history."""

    count = len(router_history)
    if count == 0:
        raise ValueError("router history must contain at least one query")
    step = (
        np.arange(count, dtype=np.int32)
        if control_step is None
        else np.asarray(control_step, dtype=np.int32)
    )
    if step.shape != (count,):
        raise ValueError(f"control_step must have shape ({count},)")
    within = compute_within_flow(router_history)
    query = compute_query_dynamics(
        within["final_root"],
        within["final_graph"],
        np.zeros(count, dtype=np.int32),
        step,
    )
    return {
        "within_features": within["features"],
        "query_features": query,
        "features": np.column_stack((within["features"], query)),
    }
