#!/usr/bin/env python3
"""Causal train-free recovery evidence for a deadline-risk rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


MIN_REFERENCE = 32
HISTORY = 4
COMPONENT_NAMES = (
    "eef_motion_w4",
    "latest_eef_motion",
    "command_efficiency_w4",
    "route_mobility",
    "front_feedback_split",
    "inverse_lag_recurrence",
)
ROUTE_FEATURES = (
    "route_mobility",
    "front_feedback_split",
    "lag_recurrence",
)


def empirical_midrank(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) < MIN_REFERENCE or not np.isfinite(value):
        return float("nan")
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return float((left + right) / (2.0 * len(reference)))


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_REFERENCE:
        raise ValueError(f"need at least {MIN_REFERENCE} finite calibration scores")
    return float(np.quantile(finite, quantile, method="higher"))


def terminal_components(
    state: np.ndarray,
    actions: np.ndarray,
    route_features: Mapping[str, float],
    query: int | None = None,
) -> np.ndarray:
    """Measure activity using only states and actions available before `query`."""
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 8:
        raise ValueError(f"expected state [query,8], got {state.shape}")
    if actions.ndim != 3 or actions.shape[1:] != (10, 7):
        raise ValueError(f"expected actions [query,10,7], got {actions.shape}")
    if query is None:
        query = len(state) - 1
    query = int(query)
    if query < HISTORY or query >= len(state) or query > len(actions):
        raise ValueError("query lacks four realized transitions")
    if not np.isfinite(state[: query + 1]).all() or not np.isfinite(
        actions[:query]
    ).all():
        raise ValueError("non-finite causal state/action prefix")
    missing = set(ROUTE_FEATURES) - set(route_features)
    if missing:
        raise ValueError(f"missing route features: {sorted(missing)}")

    displacement = np.linalg.norm(
        np.diff(state[query - HISTORY : query + 1, :3], axis=0), axis=1
    )
    prior_command = np.linalg.norm(
        actions[query - HISTORY : query, :, :3], axis=-1
    ).mean(axis=1)
    command_efficiency = float(
        displacement.mean() / max(float(prior_command.mean()), 1e-6)
    )
    route_mobility = float(route_features["route_mobility"])
    feedback_split = float(route_features["front_feedback_split"])
    recurrence = float(route_features["lag_recurrence"])
    output = np.asarray(
        (
            displacement.mean(),
            displacement[-1],
            command_efficiency,
            route_mobility,
            feedback_split,
            -recurrence,
        ),
        dtype=np.float32,
    )
    if not np.isfinite(output).all():
        raise ValueError("non-finite recovery component")
    return output


@dataclass(frozen=True)
class RecoveryProfile:
    """Task-specific empirical terminal profile with no fitted weights."""

    completed_components: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.completed_components, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(COMPONENT_NAMES):
            raise ValueError("completed components have the wrong shape")
        if len(values) < MIN_REFERENCE or not np.isfinite(values).all():
            raise ValueError("completed profile is too small or non-finite")
        object.__setattr__(self, "completed_components", values)

    def component_ranks(self, components: np.ndarray) -> np.ndarray:
        values = np.asarray(components, dtype=np.float32)
        if values.shape != (len(COMPONENT_NAMES),):
            raise ValueError("current recovery components have the wrong shape")
        return np.asarray(
            [
                empirical_midrank(self.completed_components[:, column], value)
                for column, value in enumerate(values)
            ],
            dtype=np.float32,
        )

    def score(self, components: np.ndarray) -> float:
        values = np.asarray(components, dtype=np.float32)
        if values.shape != (len(COMPONENT_NAMES),):
            raise ValueError("current recovery components have the wrong shape")
        return float(self.scores(values[None, :])[0])

    def scores(self, components: np.ndarray) -> np.ndarray:
        values = np.asarray(components, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(COMPONENT_NAMES):
            raise ValueError("recovery component batch has the wrong shape")
        ranks = np.full(values.shape, np.nan, dtype=np.float32)
        for column in range(values.shape[1]):
            reference = np.sort(self.completed_components[:, column])
            finite = np.isfinite(values[:, column])
            left = np.searchsorted(reference, values[finite, column], side="left")
            right = np.searchsorted(reference, values[finite, column], side="right")
            ranks[finite, column] = (left + right) / (2.0 * len(reference))
        output = np.full(len(values), np.nan, dtype=np.float32)
        good = np.isfinite(ranks).all(axis=1)
        output[good] = ranks[good].mean(axis=1)
        return output


def allocation_state(score: float, q75: float, q90: float) -> str:
    if not all(np.isfinite(value) for value in (score, q75, q90)):
        raise ValueError("allocation inputs must be finite")
    if q90 < q75:
        raise ValueError("q90 threshold cannot be below q75")
    if score >= q90:
        return "strong_extend"
    if score >= q75:
        return "extend_watch"
    return "intervene_first"
