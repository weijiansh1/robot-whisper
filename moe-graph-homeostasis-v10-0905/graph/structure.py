"""Weighted token-expert graph views and causal structural dynamics."""

from __future__ import annotations

import math

import numpy as np
import torch


LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
LAYER_GROUPS = {
    "all": np.arange(8),
    "front": np.arange(4),
    "back": np.arange(4, 8),
}
VIEW_NAMES = ("edge_action", "edge_state", "token_gram", "expert_load", "spectrum")
FLOW_METRICS = (
    "early_speed",
    "late_speed",
    "path",
    "endpoint",
    "efficiency",
    "curvature",
    "settling_log_ratio",
)
TERMINAL_METRICS = (
    "effective_rank_fraction",
    "token_affinity",
    "state_action_affinity",
    "expert_load_entropy",
)
WITHIN_STATE_NAMES = (
    "state|flow_nonsettling",
    "state|flow_curvature",
    "state|flow_directness",
    "state|flow_late_motion",
    "state|diversity",
    "state|token_coherence",
)
QUERY_METRICS = (
    "d1",
    "acceleration",
    "jerk",
    "lag4",
    "return_advantage",
    "turn_instability",
)
QUERY_STATE_NAMES = (
    "state|query_response",
    "state|query_acceleration",
    "state|query_jerk",
    "state|query_return",
    "state|query_turn",
)
EPSILON = 1e-8
SCHEMA = "himoe.graph_homeostasis_profile.v1"


def within_feature_names() -> tuple[str, ...]:
    names = []
    for group in LAYER_GROUPS:
        for view in VIEW_NAMES:
            names.extend(f"flow|{group}|{view}|{metric}" for metric in FLOW_METRICS)
    for group in LAYER_GROUPS:
        names.extend(f"terminal|{group}|{metric}" for metric in TERMINAL_METRICS)
    names.extend(WITHIN_STATE_NAMES)
    return tuple(names)


def query_feature_names() -> tuple[str, ...]:
    names = []
    for group in LAYER_GROUPS:
        for view in VIEW_NAMES:
            names.extend(f"query|{group}|{view}|{metric}" for metric in QUERY_METRICS)
    names.extend(QUERY_STATE_NAMES)
    return tuple(names)


WITHIN_FEATURE_NAMES = within_feature_names()
QUERY_FEATURE_NAMES = query_feature_names()


def normalize_probability(router_probability: torch.Tensor) -> torch.Tensor:
    probability = router_probability.float().clamp_min(0.0)
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_graph_views(
    router_probability: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode `[B,8,10,11,32]` graphs into normalized structural views."""

    if router_probability.ndim != 5 or tuple(router_probability.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected router shape: {tuple(router_probability.shape)}")
    probability = normalize_probability(router_probability)
    root = probability.sqrt()
    action_root = root[..., 1:, :]
    edge_action = action_root.flatten(-2) / math.sqrt(20.0)
    edge_state = root[..., 0, :] / math.sqrt(2.0)

    gram = torch.matmul(root, root.transpose(-1, -2))
    token_gram = gram.flatten(-2) / 11.0
    load_probability = probability[..., 1:, :].mean(dim=-2)
    expert_load = load_probability.sqrt() / math.sqrt(2.0)
    eigenvalue = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigenvalue /= eigenvalue.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    spectrum = eigenvalue.sqrt() / math.sqrt(2.0)

    terminal_gram = gram[:, :, -1]
    terminal_eigenvalue = eigenvalue[:, :, -1]
    terminal_load = load_probability[:, :, -1]
    action_gram = terminal_gram[..., 1:, 1:]
    diagonal = torch.diagonal(action_gram, dim1=-2, dim2=-1).sum(dim=-1)
    token_affinity = (action_gram.sum(dim=(-2, -1)) - diagonal) / 90.0
    state_action_affinity = terminal_gram[..., 0, 1:].mean(dim=-1)
    effective_rank = torch.exp(
        -(terminal_eigenvalue * terminal_eigenvalue.clamp_min(1e-12).log()).sum(dim=-1)
    ) / 11.0
    load_entropy = -(
        terminal_load * terminal_load.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(32.0)
    terminal = {
        "effective_rank_fraction": effective_rank,
        "token_affinity": token_affinity,
        "state_action_affinity": state_action_affinity,
        "expert_load_entropy": load_entropy,
    }
    views = {
        "edge_action": edge_action,
        "edge_state": edge_state,
        "token_gram": token_gram,
        "expert_load": expert_load,
        "spectrum": spectrum,
    }
    return views, terminal


def aggregate_layers(values: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    selected = values[:, indices]
    return selected.permute(0, 2, 1, 3).flatten(-2) / math.sqrt(len(indices))


def _turn_instability(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.ones_like(numerator)
    np.divide(numerator, denominator, out=cosine, where=denominator > EPSILON)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def compute_within_structure(router_probability: torch.Tensor) -> dict[str, object]:
    """Compute graph-flow settling summaries without using outcomes."""

    views, terminal = encode_graph_views(router_probability)
    flow_cube = []
    terminal_cube = []
    final_views: dict[str, np.ndarray] = {}
    for name, view in views.items():
        final_views[name] = view[:, :, -1].detach().cpu().numpy()

    for indices in LAYER_GROUPS.values():
        group_views = []
        for view in views.values():
            embedding = aggregate_layers(view, indices)
            delta = torch.linalg.vector_norm(embedding[:, 1:] - embedding[:, :-1], dim=-1)
            second = embedding[:, 2:] - 2.0 * embedding[:, 1:-1] + embedding[:, :-2]
            early = delta[:, :3].mean(dim=1)
            late = delta[:, -3:].mean(dim=1)
            path = delta.sum(dim=1)
            endpoint = torch.linalg.vector_norm(embedding[:, -1] - embedding[:, 0], dim=-1)
            group_views.append(
                torch.stack(
                    (
                        early,
                        late,
                        path,
                        endpoint,
                        endpoint / path.clamp_min(EPSILON),
                        torch.linalg.vector_norm(second, dim=-1).mean(dim=1),
                        torch.log((late + EPSILON) / (early + EPSILON)),
                    ),
                    dim=-1,
                )
            )
        flow_cube.append(torch.stack(group_views, dim=1))
        terminal_cube.append(
            torch.stack(
                [values[:, indices].mean(dim=1) for values in terminal.values()], dim=-1
            )
        )
    flow_tensor = torch.stack(flow_cube, dim=1)
    terminal_tensor = torch.stack(terminal_cube, dim=1)
    all_flow = flow_tensor[:, 0]
    all_terminal = terminal_tensor[:, 0]
    state = torch.stack(
        (
            all_flow[..., FLOW_METRICS.index("settling_log_ratio")].mean(dim=1),
            all_flow[..., FLOW_METRICS.index("curvature")].mean(dim=1),
            all_flow[..., FLOW_METRICS.index("efficiency")].mean(dim=1),
            all_flow[..., FLOW_METRICS.index("late_speed")].mean(dim=1),
            all_terminal[..., TERMINAL_METRICS.index("effective_rank_fraction")],
            all_terminal[..., TERMINAL_METRICS.index("token_affinity")],
        ),
        dim=-1,
    )
    feature = torch.cat(
        (
            flow_tensor.flatten(1),
            terminal_tensor.flatten(1),
            state,
        ),
        dim=1,
    )
    if feature.shape[1] != len(WITHIN_FEATURE_NAMES):
        raise RuntimeError(f"within schema mismatch: {feature.shape[1]}")
    return {
        "features": feature.detach().cpu().numpy(),
        "final_views": final_views,
    }


def compute_query_structure(
    final_views: dict[str, np.ndarray],
    episode_id: np.ndarray,
    control_step: np.ndarray,
) -> np.ndarray:
    """Compute prefix-causal inter-query dynamics of graph embeddings."""

    count = len(episode_id)
    cube = np.full(
        (count, len(LAYER_GROUPS), len(VIEW_NAMES), len(QUERY_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    for group_index, indices in enumerate(LAYER_GROUPS.values()):
        for view_index, name in enumerate(VIEW_NAMES):
            raw = np.asarray(final_views[name], dtype=np.float32)[:, indices]
            embedding = raw.reshape(count, -1) / np.sqrt(len(indices))
            for episode in np.unique(episode_id):
                rows = np.flatnonzero(episode_id == episode)
                rows = rows[np.argsort(control_step[rows], kind="stable")]
                value = embedding[rows]
                if len(rows) >= 2:
                    velocity = value[1:] - value[:-1]
                    cube[rows[1:], group_index, view_index, 0] = np.linalg.norm(
                        velocity, axis=-1
                    )
                if len(rows) >= 3:
                    second = value[2:] - 2.0 * value[1:-1] + value[:-2]
                    cube[rows[2:], group_index, view_index, 1] = np.linalg.norm(
                        second, axis=-1
                    )
                    cube[rows[2:], group_index, view_index, 5] = _turn_instability(
                        velocity[:-1], velocity[1:]
                    )
                if len(rows) >= 4:
                    third = (
                        value[3:]
                        - 3.0 * value[2:-1]
                        + 3.0 * value[1:-2]
                        - value[:-3]
                    )
                    cube[rows[3:], group_index, view_index, 2] = np.linalg.norm(
                        third, axis=-1
                    )
                if len(rows) >= 5:
                    lag = [
                        np.linalg.norm(value[offset:] - value[:-offset], axis=-1)
                        for offset in range(1, 5)
                    ]
                    cube[rows[4:], group_index, view_index, 3] = lag[3]
                    old_min = np.minimum.reduce((lag[1][2:], lag[2][1:], lag[3]))
                    cube[rows[4:], group_index, view_index, 4] = lag[0][3:] - old_min

    all_group = cube[:, 0]
    state = np.column_stack(
        (
            all_group[..., QUERY_METRICS.index("d1")].mean(axis=1),
            all_group[..., QUERY_METRICS.index("acceleration")].mean(axis=1),
            all_group[..., QUERY_METRICS.index("jerk")].mean(axis=1),
            all_group[..., QUERY_METRICS.index("return_advantage")].mean(axis=1),
            all_group[..., QUERY_METRICS.index("turn_instability")].mean(axis=1),
        )
    ).astype(np.float32)
    feature = np.column_stack((cube.reshape(count, -1), state)).astype(np.float32)
    if feature.shape[1] != len(QUERY_FEATURE_NAMES):
        raise RuntimeError(f"query schema mismatch: {feature.shape[1]}")
    return feature


def episode_local_query(episode_id: np.ndarray, control_step: np.ndarray) -> np.ndarray:
    query = np.empty(len(episode_id), dtype=np.int16)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        query[rows] = np.arange(len(rows), dtype=np.int16)
    return query
