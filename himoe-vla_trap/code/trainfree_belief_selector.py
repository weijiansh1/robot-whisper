#!/usr/bin/env python3
"""Train-free stale-belief trigger and same-observation candidate ranker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


N_EXPERTS = 32


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[-1] != N_EXPERTS or not np.all(np.isfinite(values)):
        raise ValueError("router probabilities must be finite and end in 32 experts")
    if np.any(values < -1e-7):
        raise ValueError("router probabilities contain negative values")
    values = np.maximum(values, 0.0)
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every routing site must have positive mass")
    return values / mass


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = normalize(left)
    q = normalize(right)
    return np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum(axis=-1))


def layer_state_action_gap(route: np.ndarray, layer_axis: int = 3) -> float:
    """Mean full-flow state/action gate gap for one ``[L,F,U,E]`` route."""

    p = normalize(route[layer_axis])
    state = p[:, 0]
    action = normalize(p[:, 1:11].mean(axis=1))
    return float(hellinger(state, action).mean())


def group_chunk_jump(
    current: np.ndarray,
    previous: np.ndarray,
    layer_axes: Iterable[int] = (4, 5, 6, 7),
) -> float:
    """Mean action-token route jump across adjacent replans."""

    indices = tuple(layer_axes)
    return float(hellinger(current[indices, :, 1:11], previous[indices, :, 1:11]).mean())


def group_action_distance(
    route: np.ndarray,
    reference: np.ndarray,
    layer_axes: Iterable[int] = (0, 1, 2, 3),
) -> float:
    """Distance between action routes and a phase-matched healthy reference."""

    indices = tuple(layer_axes)
    return float(hellinger(route[indices, :, 1:11], reference[indices, :, 1:11]).mean())


def leave_one_out_action_distances(
    references: np.ndarray,
    layer_axes: Iterable[int] = (0, 1, 2, 3),
) -> np.ndarray:
    if references.ndim != 5 or len(references) < 3:
        raise ValueError("references must be [N,L,F,U,E] with N >= 3")
    return np.asarray([
        group_action_distance(
            references[index],
            np.delete(references, index, axis=0).mean(axis=0),
            layer_axes,
        )
        for index in range(len(references))
    ])


@dataclass(frozen=True)
class HealthyThresholds:
    layer5_gap_upper: float
    back_chunk_jump_upper: float
    front_action_distance_upper: float


@dataclass(frozen=True)
class SelectorDecision:
    reject_stale_chunk: bool
    layer5_gap: float
    back_chunk_jump: float
    front_action_distance: float
    layer5_gap_ratio: float
    back_chunk_jump_ratio: float
    front_action_distance_ratio: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def calibrate_healthy(
    reference_current: np.ndarray,
    reference_previous: np.ndarray,
) -> HealthyThresholds:
    """Set empirical upper bounds from phase-matched healthy routes only."""

    if reference_current.shape != reference_previous.shape:
        raise ValueError("current and previous healthy reference banks must align")
    if reference_current.ndim != 5 or len(reference_current) < 3:
        raise ValueError("healthy banks must be [N,L,F,U,E] with N >= 3")
    gaps = np.asarray([layer_state_action_gap(route) for route in reference_current])
    jumps = np.asarray([
        group_chunk_jump(current, previous)
        for current, previous in zip(reference_current, reference_previous)
    ])
    action_loo = leave_one_out_action_distances(reference_current)
    return HealthyThresholds(
        layer5_gap_upper=float(gaps.max()),
        back_chunk_jump_upper=float(jumps.max()),
        front_action_distance_upper=float(action_loo.max()),
    )


def evaluate_stale_belief(
    current: np.ndarray,
    previous: np.ndarray,
    reference_current: np.ndarray,
    thresholds: HealthyThresholds,
) -> SelectorDecision:
    """Reject only a changed-state / nominal-action conjunction."""

    gap = layer_state_action_gap(current)
    jump = group_chunk_jump(current, previous)
    action_distance = group_action_distance(current, reference_current.mean(axis=0))
    gap_ratio = gap / max(thresholds.layer5_gap_upper, 1e-12)
    jump_ratio = jump / max(thresholds.back_chunk_jump_upper, 1e-12)
    action_ratio = action_distance / max(
        thresholds.front_action_distance_upper, 1e-12
    )
    reject = bool(gap_ratio > 1.0 and jump_ratio > 1.0 and action_ratio <= 1.0)
    return SelectorDecision(
        reject_stale_chunk=reject,
        layer5_gap=gap,
        back_chunk_jump=jump,
        front_action_distance=action_distance,
        layer5_gap_ratio=gap_ratio,
        back_chunk_jump_ratio=jump_ratio,
        front_action_distance_ratio=action_ratio,
    )


def candidate_gap_scores(
    candidate_routes: np.ndarray,
    layer_axis: int = 3,
) -> np.ndarray:
    """Score ``[K,L,F,U,E]`` candidates; lower is more state/action consistent."""

    values = np.asarray(candidate_routes)
    if values.ndim != 5 or len(values) < 2:
        raise ValueError("candidate routes must have shape [K,L,F,U,E], K >= 2")
    return np.asarray([
        layer_state_action_gap(route, layer_axis) for route in values
    ])


def select_min_gap(
    candidate_routes: np.ndarray,
    candidate_ids: Iterable[int] | None = None,
    layer_axis: int = 3,
) -> int:
    """Return the candidate ID with minimum gap, with deterministic ID tie-break."""

    scores = candidate_gap_scores(candidate_routes, layer_axis)
    ids = (
        np.arange(len(scores), dtype=np.int64)
        if candidate_ids is None
        else np.asarray(tuple(candidate_ids), dtype=np.int64)
    )
    if ids.shape != scores.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and aligned")
    minimum = scores.min()
    return int(ids[np.isclose(scores, minimum, atol=1e-12, rtol=0.0)].min())
