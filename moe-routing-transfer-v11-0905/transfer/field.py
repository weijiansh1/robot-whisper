"""Train-free front-to-back routing transfer fields for HiMoE-VLA."""

from __future__ import annotations

import math

import numpy as np
import torch


EPSILON = 1e-8
SCHEMA = "himoe.routing_transfer_profile.v2"
RELATION_VIEWS = (
    "action_relation",
    "action_shape",
    "action_conditional",
    "action_partial",
    "state_relation",
    "state_shape",
)
FLOW_METRICS = (
    "gap_initial",
    "gap_terminal",
    "gap_log_ratio",
    "gap_path",
    "gap_endpoint",
    "gap_efficiency",
    "gap_curvature",
    "gap_settling_log_ratio",
    "front_path",
    "back_path",
    "back_front_path_log_ratio",
    "back_front_early_log_ratio",
    "back_front_late_log_ratio",
    "velocity_alignment_mean",
    "velocity_alignment_early",
    "velocity_alignment_late",
    "velocity_opposition_fraction",
    "mixed_velocity",
    "front_time_center",
    "back_time_center",
    "handoff_delay",
    "initial_alignment",
    "terminal_alignment",
    "alignment_change",
    "front_terminal_norm",
    "back_terminal_norm",
    "terminal_norm_log_ratio",
    "front_settling_log_ratio",
    "back_settling_log_ratio",
    "settling_difference",
    "lag_alignment_m2",
    "lag_alignment_m1",
    "lag_alignment_p1",
    "lag_alignment_p2",
)
CHANNEL_NAMES = (
    "front_edge_to_back_action_shape",
    "front_edge_to_back_state_shape",
    "front_edge_to_back_action_conditional",
    "front_state_to_back_action_conditional",
)
CHANNEL_FLOW_METRICS = (
    "source_path",
    "target_path",
    "target_source_path_log_ratio",
    "target_source_early_log_ratio",
    "target_source_late_log_ratio",
    "source_settling_log_ratio",
    "target_settling_log_ratio",
    "settling_difference",
    "source_time_center",
    "target_time_center",
    "handoff_delay",
    "speed_correlation",
    "speed_lag_correlation_m2",
    "speed_lag_correlation_m1",
    "speed_lag_correlation_p1",
    "speed_lag_correlation_p2",
    "source_roughness",
    "target_roughness",
    "roughness_difference",
)
QUERY_METRICS = (
    "front_d1",
    "back_d1",
    "gap_d1",
    "response_log_ratio",
    "velocity_alignment",
    "parallel_gain",
    "front_acceleration",
    "back_acceleration",
    "gap_acceleration",
    "acceleration_log_ratio",
    "gap_jerk",
    "front_lag4",
    "back_lag4",
    "gap_lag4",
    "lag4_log_ratio",
    "front_return_advantage",
    "back_return_advantage",
    "gap_return_advantage",
    "return_difference",
    "front_turn_instability",
    "back_turn_instability",
    "gap_turn_instability",
    "turn_difference",
)
CHANNEL_QUERY_METRICS = (
    "source_d1",
    "target_d1",
    "target_source_d1_log_ratio",
    "source_acceleration",
    "target_acceleration",
    "target_source_acceleration_log_ratio",
    "source_lag4",
    "target_lag4",
    "target_source_lag4_log_ratio",
    "recent_speed_correlation",
)
SCALAR_QUERY_METRICS = (
    "d1",
    "acceleration",
    "jerk",
    "lag4",
    "return_advantage",
    "turn_instability",
)
SCALAR_QUERY_SOURCES = (
    "flow|action_relation|gap_terminal",
    "flow|action_relation|gap_curvature",
    "flow|action_relation|handoff_delay",
    "flow|action_shape|gap_terminal",
    "flow|action_shape|gap_curvature",
    "flow|action_shape|handoff_delay",
    "flow|action_shape|terminal_alignment",
    "flow|action_shape|back_front_path_log_ratio",
    "flow|action_conditional|gap_curvature",
    "flow|action_conditional|handoff_delay",
    "flow|action_conditional|terminal_alignment",
    "flow|action_conditional|back_front_path_log_ratio",
    "flow|action_partial|gap_curvature",
    "flow|action_partial|terminal_alignment",
    "flow|state_shape|gap_curvature",
    "channel|front_edge_to_back_action_shape|target_source_path_log_ratio",
    "channel|front_edge_to_back_action_shape|handoff_delay",
    "channel|front_edge_to_back_action_shape|speed_correlation",
    "channel|front_state_to_back_action_conditional|target_source_path_log_ratio",
    "channel|front_state_to_back_action_conditional|handoff_delay",
    "channel|front_state_to_back_action_conditional|speed_correlation",
)


def within_feature_names() -> tuple[str, ...]:
    names = [
        f"flow|{view}|{metric}"
        for view in RELATION_VIEWS
        for metric in FLOW_METRICS
    ]
    names.extend(
        f"channel|{channel}|{metric}"
        for channel in CHANNEL_NAMES
        for metric in CHANNEL_FLOW_METRICS
    )
    return tuple(names)


def query_feature_names() -> tuple[str, ...]:
    names = [
        f"query|{view}|{metric}"
        for view in RELATION_VIEWS
        for metric in QUERY_METRICS
    ]
    names.extend(
        f"query_channel|{channel}|{metric}"
        for channel in CHANNEL_NAMES
        for metric in CHANNEL_QUERY_METRICS
    )
    names.extend(
        f"query_scalar|{source}|{metric}"
        for source in SCALAR_QUERY_SOURCES
        for metric in SCALAR_QUERY_METRICS
    )
    return tuple(names)


WITHIN_FEATURE_NAMES = within_feature_names()
QUERY_FEATURE_NAMES = query_feature_names()


def _normalize_probability(router_probability: torch.Tensor) -> torch.Tensor:
    probability = router_probability.float().clamp_min(0.0)
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _paired_groups(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned four-block front/back embeddings as `[B,F,4D]`."""

    front = value[:, :4].permute(0, 2, 1, 3).flatten(-2) / 2.0
    back = value[:, 4:].permute(0, 2, 1, 3).flatten(-2) / 2.0
    return front, back


def conditional_action_graph(
    action_gram: torch.Tensor, state_action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state-orthogonal action Gram and its normalized partial affinities."""

    conditional = action_gram - state_action.unsqueeze(-1) * state_action.unsqueeze(-2)
    diagonal = torch.diagonal(conditional, dim1=-2, dim2=-1).clamp_min(0.0)
    denominator = torch.sqrt(
        diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)
    ).clamp_min(EPSILON)
    partial = conditional / denominator
    partial = partial.clamp(-1.0, 1.0)
    return conditional, partial


def encode_transfer_views(
    router_probability: torch.Tensor,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    """Encode token relations and the front-edge to back-structure channels."""

    if router_probability.ndim != 5 or tuple(router_probability.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected router shape: {tuple(router_probability.shape)}")
    probability = _normalize_probability(router_probability)
    root = probability.sqrt()
    gram = torch.matmul(root, root.transpose(-1, -2))

    triangle = torch.triu_indices(10, 10, offset=1, device=gram.device)
    triangle_diagonal = torch.triu_indices(10, 10, offset=0, device=gram.device)
    action_gram = gram[..., 1:, 1:]
    state_action = gram[..., 0, 1:]
    action = action_gram[..., triangle[0], triangle[1]] / math.sqrt(45.0)
    state = state_action / math.sqrt(10.0)
    conditional, partial = conditional_action_graph(action_gram, state_action)
    conditional_vector = (
        conditional[..., triangle_diagonal[0], triangle_diagonal[1]] / math.sqrt(55.0)
    )
    partial_vector = partial[..., triangle[0], triangle[1]] / math.sqrt(45.0)
    relation = {
        "action_relation": action,
        "action_shape": action - action.mean(dim=-1, keepdim=True),
        "action_conditional": conditional_vector,
        "action_partial": partial_vector,
        "state_relation": state,
        "state_shape": state - state.mean(dim=-1, keepdim=True),
    }
    paired = {name: _paired_groups(value) for name, value in relation.items()}

    front_edge = (
        root[:, :4, :, 1:, :]
        .flatten(-2)
        .permute(0, 2, 1, 3)
        .flatten(-2)
        / math.sqrt(80.0)
    )
    channels = {
        "front_edge_to_back_action_shape": (front_edge, paired["action_shape"][1]),
        "front_edge_to_back_state_shape": (front_edge, paired["state_shape"][1]),
        "front_edge_to_back_action_conditional": (
            front_edge,
            paired["action_conditional"][1],
        ),
        "front_state_to_back_action_conditional": (
            paired["state_relation"][0],
            paired["action_conditional"][1],
        ),
    }
    return paired, channels


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value, dim=-1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = _norm(left) * _norm(right)
    result = (left * right).sum(dim=-1) / denominator.clamp_min(EPSILON)
    return torch.where(denominator > EPSILON, result.clamp(-1.0, 1.0), torch.zeros_like(result))


def _settling(speed: torch.Tensor) -> torch.Tensor:
    return torch.log((speed[:, -3:].mean(dim=1) + EPSILON) / (speed[:, :3].mean(dim=1) + EPSILON))


def _time_center(speed: torch.Tensor) -> torch.Tensor:
    location = torch.linspace(
        0.5 / speed.shape[1],
        1.0 - 0.5 / speed.shape[1],
        speed.shape[1],
        dtype=speed.dtype,
        device=speed.device,
    )
    return (speed * location).sum(dim=1) / speed.sum(dim=1).clamp_min(EPSILON)


def _lagged_vector_alignment(
    source: torch.Tensor, target: torch.Tensor, lag: int
) -> torch.Tensor:
    if lag > 0:
        source = source[:, :-lag]
        target = target[:, lag:]
    elif lag < 0:
        source = source[:, -lag:]
        target = target[:, :lag]
    return _cosine(source, target).mean(dim=1)


def _scalar_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return _cosine(left.unsqueeze(-2), right.unsqueeze(-2)).squeeze(-1)


def _lagged_scalar_correlation(
    source: torch.Tensor, target: torch.Tensor, lag: int
) -> torch.Tensor:
    if lag > 0:
        source = source[:, :-lag]
        target = target[:, lag:]
    elif lag < 0:
        source = source[:, -lag:]
        target = target[:, :lag]
    return _scalar_correlation(source, target)


def flow_transfer_features(front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    """Describe how a same-coordinate token graph changes from front to back."""

    if front.shape != back.shape or front.ndim != 3:
        raise ValueError("front/back transfer fields must share shape [B,F,D]")
    front_velocity = front[:, 1:] - front[:, :-1]
    back_velocity = back[:, 1:] - back[:, :-1]
    gap = back - front
    gap_velocity = gap[:, 1:] - gap[:, :-1]
    front_speed = _norm(front_velocity)
    back_speed = _norm(back_velocity)
    gap_speed = _norm(gap_velocity)

    gap_initial = _norm(gap[:, 0])
    gap_terminal = _norm(gap[:, -1])
    gap_path = gap_speed.sum(dim=1)
    gap_endpoint = _norm(gap[:, -1] - gap[:, 0])
    gap_second = gap[:, 2:] - 2.0 * gap[:, 1:-1] + gap[:, :-2]
    alignment = _cosine(front, back)
    velocity_alignment = _cosine(front_velocity, back_velocity)
    front_path = front_speed.sum(dim=1)
    back_path = back_speed.sum(dim=1)

    columns = (
        gap_initial,
        gap_terminal,
        torch.log((gap_terminal + EPSILON) / (gap_initial + EPSILON)),
        gap_path,
        gap_endpoint,
        gap_endpoint / gap_path.clamp_min(EPSILON),
        _norm(gap_second).mean(dim=1),
        _settling(gap_speed),
        front_path,
        back_path,
        torch.log((back_path + EPSILON) / (front_path + EPSILON)),
        torch.log(
            (back_speed[:, :3].mean(dim=1) + EPSILON)
            / (front_speed[:, :3].mean(dim=1) + EPSILON)
        ),
        torch.log(
            (back_speed[:, -3:].mean(dim=1) + EPSILON)
            / (front_speed[:, -3:].mean(dim=1) + EPSILON)
        ),
        velocity_alignment.mean(dim=1),
        velocity_alignment[:, :3].mean(dim=1),
        velocity_alignment[:, -3:].mean(dim=1),
        (velocity_alignment < 0.0).float().mean(dim=1),
        _norm(back_velocity - front_velocity).mean(dim=1),
        _time_center(front_speed),
        _time_center(back_speed),
        _time_center(back_speed) - _time_center(front_speed),
        alignment[:, 0],
        alignment[:, -1],
        alignment[:, -1] - alignment[:, 0],
        _norm(front[:, -1]),
        _norm(back[:, -1]),
        torch.log((_norm(back[:, -1]) + EPSILON) / (_norm(front[:, -1]) + EPSILON)),
        _settling(front_speed),
        _settling(back_speed),
        _settling(back_speed) - _settling(front_speed),
        _lagged_vector_alignment(front_velocity, back_velocity, -2),
        _lagged_vector_alignment(front_velocity, back_velocity, -1),
        _lagged_vector_alignment(front_velocity, back_velocity, 1),
        _lagged_vector_alignment(front_velocity, back_velocity, 2),
    )
    result = torch.stack(columns, dim=-1)
    if result.shape[-1] != len(FLOW_METRICS):
        raise RuntimeError("flow transfer schema mismatch")
    return result


def channel_flow_features(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relate front expert-edge motion to back token-structure motion."""

    source_velocity = source[:, 1:] - source[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    source_speed = _norm(source_velocity)
    target_speed = _norm(target_velocity)
    source_path = source_speed.sum(dim=1)
    target_path = target_speed.sum(dim=1)
    source_second = source[:, 2:] - 2.0 * source[:, 1:-1] + source[:, :-2]
    target_second = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    source_roughness = _norm(source_second).sum(dim=1) / source_path.clamp_min(EPSILON)
    target_roughness = _norm(target_second).sum(dim=1) / target_path.clamp_min(EPSILON)
    source_settling = _settling(source_speed)
    target_settling = _settling(target_speed)
    source_time = _time_center(source_speed)
    target_time = _time_center(target_speed)
    columns = (
        source_path,
        target_path,
        torch.log((target_path + EPSILON) / (source_path + EPSILON)),
        torch.log(
            (target_speed[:, :3].mean(dim=1) + EPSILON)
            / (source_speed[:, :3].mean(dim=1) + EPSILON)
        ),
        torch.log(
            (target_speed[:, -3:].mean(dim=1) + EPSILON)
            / (source_speed[:, -3:].mean(dim=1) + EPSILON)
        ),
        source_settling,
        target_settling,
        target_settling - source_settling,
        source_time,
        target_time,
        target_time - source_time,
        _scalar_correlation(source_speed, target_speed),
        _lagged_scalar_correlation(source_speed, target_speed, -2),
        _lagged_scalar_correlation(source_speed, target_speed, -1),
        _lagged_scalar_correlation(source_speed, target_speed, 1),
        _lagged_scalar_correlation(source_speed, target_speed, 2),
        source_roughness,
        target_roughness,
        target_roughness - source_roughness,
    )
    result = torch.stack(columns, dim=-1)
    if result.shape[-1] != len(CHANNEL_FLOW_METRICS):
        raise RuntimeError("channel flow schema mismatch")
    return result


def compute_within_transfer(router_probability: torch.Tensor) -> dict[str, object]:
    paired, channels = encode_transfer_views(router_probability)
    flow = torch.stack(
        [flow_transfer_features(*paired[name]) for name in RELATION_VIEWS], dim=1
    )
    channel = torch.stack(
        [channel_flow_features(*channels[name]) for name in CHANNEL_NAMES], dim=1
    )
    feature = torch.cat((flow.flatten(1), channel.flatten(1)), dim=1)
    if feature.shape[1] != len(WITHIN_FEATURE_NAMES):
        raise RuntimeError("within transfer schema mismatch")
    terminal_pairs = {
        name: (
            front[:, -1].detach().cpu().numpy(),
            back[:, -1].detach().cpu().numpy(),
        )
        for name, (front, back) in paired.items()
    }
    terminal_channels = {
        name: (
            source[:, -1].detach().cpu().numpy(),
            target[:, -1].detach().cpu().numpy(),
        )
        for name, (source, target) in channels.items()
    }
    return {
        "features": feature.detach().cpu().numpy(),
        "terminal_pairs": terminal_pairs,
        "terminal_channels": terminal_channels,
    }


def _numpy_norm(value: np.ndarray) -> np.ndarray:
    return np.linalg.norm(value, axis=-1)


def _numpy_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = _numpy_norm(left) * _numpy_norm(right)
    result = np.zeros_like(denominator, dtype=np.float32)
    np.divide(np.sum(left * right, axis=-1), denominator, out=result, where=denominator > EPSILON)
    return np.clip(result, -1.0, 1.0)


def _turn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 1.0 - _numpy_cosine(left, right)


def _return_advantage(value: np.ndarray) -> np.ndarray:
    lag = [
        _numpy_norm(value[offset:] - value[:-offset])
        for offset in range(1, 5)
    ]
    old_min = np.minimum.reduce((lag[1][2:], lag[2][1:], lag[3]))
    return lag[0][3:] - old_min


def _fill_query_pair(
    cube: np.ndarray,
    rows: np.ndarray,
    front: np.ndarray,
    back: np.ndarray,
) -> None:
    gap = back - front
    if len(rows) >= 2:
        front_velocity = front[1:] - front[:-1]
        back_velocity = back[1:] - back[:-1]
        gap_velocity = gap[1:] - gap[:-1]
        front_d1 = _numpy_norm(front_velocity)
        back_d1 = _numpy_norm(back_velocity)
        cube[rows[1:], 0] = front_d1
        cube[rows[1:], 1] = back_d1
        cube[rows[1:], 2] = _numpy_norm(gap_velocity)
        cube[rows[1:], 3] = np.log((back_d1 + EPSILON) / (front_d1 + EPSILON))
        cube[rows[1:], 4] = _numpy_cosine(front_velocity, back_velocity)
        cube[rows[1:], 5] = np.sum(front_velocity * back_velocity, axis=-1) / (
            np.sum(front_velocity * front_velocity, axis=-1) + EPSILON
        )
    if len(rows) >= 3:
        front_second = front[2:] - 2.0 * front[1:-1] + front[:-2]
        back_second = back[2:] - 2.0 * back[1:-1] + back[:-2]
        gap_second = gap[2:] - 2.0 * gap[1:-1] + gap[:-2]
        front_acceleration = _numpy_norm(front_second)
        back_acceleration = _numpy_norm(back_second)
        cube[rows[2:], 6] = front_acceleration
        cube[rows[2:], 7] = back_acceleration
        cube[rows[2:], 8] = _numpy_norm(gap_second)
        cube[rows[2:], 9] = np.log(
            (back_acceleration + EPSILON) / (front_acceleration + EPSILON)
        )
        cube[rows[2:], 19] = _turn(front_velocity[:-1], front_velocity[1:])
        cube[rows[2:], 20] = _turn(back_velocity[:-1], back_velocity[1:])
        cube[rows[2:], 21] = _turn(gap_velocity[:-1], gap_velocity[1:])
        cube[rows[2:], 22] = cube[rows[2:], 20] - cube[rows[2:], 19]
    if len(rows) >= 4:
        gap_third = gap[3:] - 3.0 * gap[2:-1] + 3.0 * gap[1:-2] - gap[:-3]
        cube[rows[3:], 10] = _numpy_norm(gap_third)
    if len(rows) >= 5:
        front_lag4 = _numpy_norm(front[4:] - front[:-4])
        back_lag4 = _numpy_norm(back[4:] - back[:-4])
        cube[rows[4:], 11] = front_lag4
        cube[rows[4:], 12] = back_lag4
        cube[rows[4:], 13] = _numpy_norm(gap[4:] - gap[:-4])
        cube[rows[4:], 14] = np.log((back_lag4 + EPSILON) / (front_lag4 + EPSILON))
        cube[rows[4:], 15] = _return_advantage(front)
        cube[rows[4:], 16] = _return_advantage(back)
        cube[rows[4:], 17] = _return_advantage(gap)
        cube[rows[4:], 18] = cube[rows[4:], 16] - cube[rows[4:], 15]


def _recent_correlation(left: np.ndarray, right: np.ndarray, window: int = 4) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=np.float32)
    for end in range(window - 1, len(left)):
        lhs = left[end - window + 1 : end + 1]
        rhs = right[end - window + 1 : end + 1]
        lhs = lhs - lhs.mean()
        rhs = rhs - rhs.mean()
        denominator = np.linalg.norm(lhs) * np.linalg.norm(rhs)
        result[end] = float(np.dot(lhs, rhs) / denominator) if denominator > EPSILON else 0.0
    return result


def _fill_query_channel(
    cube: np.ndarray,
    rows: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    if len(rows) >= 2:
        source_d1 = _numpy_norm(source[1:] - source[:-1])
        target_d1 = _numpy_norm(target[1:] - target[:-1])
        cube[rows[1:], 0] = source_d1
        cube[rows[1:], 1] = target_d1
        cube[rows[1:], 2] = np.log((target_d1 + EPSILON) / (source_d1 + EPSILON))
    if len(rows) >= 3:
        source_acceleration = _numpy_norm(source[2:] - 2.0 * source[1:-1] + source[:-2])
        target_acceleration = _numpy_norm(target[2:] - 2.0 * target[1:-1] + target[:-2])
        cube[rows[2:], 3] = source_acceleration
        cube[rows[2:], 4] = target_acceleration
        cube[rows[2:], 5] = np.log(
            (target_acceleration + EPSILON) / (source_acceleration + EPSILON)
        )
    if len(rows) >= 5:
        source_lag4 = _numpy_norm(source[4:] - source[:-4])
        target_lag4 = _numpy_norm(target[4:] - target[:-4])
        cube[rows[4:], 6] = source_lag4
        cube[rows[4:], 7] = target_lag4
        cube[rows[4:], 8] = np.log((target_lag4 + EPSILON) / (source_lag4 + EPSILON))
        recent = _recent_correlation(source_d1, target_d1)
        cube[rows[1:], 9] = recent


def _fill_scalar_query(cube: np.ndarray, rows: np.ndarray, value: np.ndarray) -> None:
    if len(rows) >= 2:
        velocity = value[1:] - value[:-1]
        cube[rows[1:], 0] = np.abs(velocity)
    if len(rows) >= 3:
        second = value[2:] - 2.0 * value[1:-1] + value[:-2]
        cube[rows[2:], 1] = np.abs(second)
        product = velocity[:-1] * velocity[1:]
        denominator = np.abs(velocity[:-1] * velocity[1:])
        cosine = np.ones_like(product)
        np.divide(product, denominator, out=cosine, where=denominator > EPSILON)
        cube[rows[2:], 5] = 1.0 - np.clip(cosine, -1.0, 1.0)
    if len(rows) >= 4:
        third = value[3:] - 3.0 * value[2:-1] + 3.0 * value[1:-2] - value[:-3]
        cube[rows[3:], 2] = np.abs(third)
    if len(rows) >= 5:
        cube[rows[4:], 3] = np.abs(value[4:] - value[:-4])
        lag = [np.abs(value[offset:] - value[:-offset]) for offset in range(1, 5)]
        old_min = np.minimum.reduce((lag[1][2:], lag[2][1:], lag[3]))
        cube[rows[4:], 4] = lag[0][3:] - old_min


def compute_query_transfer(
    terminal_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    terminal_channels: dict[str, tuple[np.ndarray, np.ndarray]],
    within_features: np.ndarray,
    episode_id: np.ndarray,
    control_step: np.ndarray,
) -> np.ndarray:
    """Compute prefix-causal dynamics of the front-to-back transfer field."""

    count = len(episode_id)
    pair_cube = np.full(
        (count, len(RELATION_VIEWS), len(QUERY_METRICS)), np.nan, dtype=np.float32
    )
    channel_cube = np.full(
        (count, len(CHANNEL_NAMES), len(CHANNEL_QUERY_METRICS)), np.nan, dtype=np.float32
    )
    scalar_cube = np.full(
        (count, len(SCALAR_QUERY_SOURCES), len(SCALAR_QUERY_METRICS)),
        np.nan,
        dtype=np.float32,
    )
    within_indices = {name: index for index, name in enumerate(WITHIN_FEATURE_NAMES)}

    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        for view_index, name in enumerate(RELATION_VIEWS):
            front, back = terminal_pairs[name]
            _fill_query_pair(pair_cube[:, view_index], rows, front[rows], back[rows])
        for channel_index, name in enumerate(CHANNEL_NAMES):
            source, target = terminal_channels[name]
            _fill_query_channel(
                channel_cube[:, channel_index], rows, source[rows], target[rows]
            )
        for scalar_index, name in enumerate(SCALAR_QUERY_SOURCES):
            _fill_scalar_query(
                scalar_cube[:, scalar_index], rows, within_features[rows, within_indices[name]]
            )

    feature = np.column_stack(
        (pair_cube.reshape(count, -1), channel_cube.reshape(count, -1), scalar_cube.reshape(count, -1))
    ).astype(np.float32)
    if feature.shape[1] != len(QUERY_FEATURE_NAMES):
        raise RuntimeError("query transfer schema mismatch")
    return feature


def episode_local_query(episode_id: np.ndarray, control_step: np.ndarray) -> np.ndarray:
    query = np.empty(len(episode_id), dtype=np.int16)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        query[rows] = np.arange(len(rows), dtype=np.int16)
    return query
