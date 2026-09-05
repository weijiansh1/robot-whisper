#!/usr/bin/env python3
"""Train-free, task-free MoE alarm using only an episode's own route prefix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "himoe.self_reference_coupling_collapse.config.v1"
SELECTOR_VERSION = "self_reference_coupling_collapse_v3"
EXPECTED_ROUTE_SHAPE = (8, 10, 11, 32)
FRONT_LAYERS = slice(0, 4)
BACK_LAYERS = slice(4, 8)
FINAL_FLOW = 9
ACTION_TOKENS = slice(1, 11)


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def route_observables(
    route: np.ndarray,
    previous_front_state: np.ndarray | None,
) -> tuple[float, float, float, np.ndarray]:
    """Return planning churn, state/action gap, and cross-query state response."""

    probability = normalize_probability(route)
    if tuple(probability.shape) != EXPECTED_ROUTE_SHAPE:
        raise ValueError(f"route must have shape {EXPECTED_ROUTE_SHAPE}")

    back_action_flow = probability[BACK_LAYERS, :, ACTION_TOKENS]
    root = np.sqrt(back_action_flow)
    second_difference = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
    planning_churn = float(
        np.linalg.norm(second_difference, axis=-1).mean() / np.sqrt(2.0)
    )

    front_state = probability[FRONT_LAYERS, FINAL_FLOW, 0]
    front_action = normalize_probability(
        probability[FRONT_LAYERS, FINAL_FLOW, ACTION_TOKENS].mean(axis=1)
    )
    state_action_gap = float(hellinger(front_state, front_action).mean())
    state_response = (
        float("nan")
        if previous_front_state is None
        else float(hellinger(front_state, previous_front_state).mean())
    )
    return planning_churn, state_action_gap, state_response, front_state


@dataclass(frozen=True)
class SelfReferenceConfig:
    warmup_transitions: int
    window: int
    state_response_ratio_max: float
    planning_churn_ratio_min: float
    state_action_gap_ratio_min: float

    @classmethod
    def load(cls, path: Path) -> "SelfReferenceConfig":
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise RuntimeError("unsupported self-reference selector config schema")
        runtime_guards = {
            "training": False,
            "learned_parameters": False,
            "runtime_normal_trajectory_reference_used": False,
            "runtime_outcome_labels_used": False,
            "runtime_physical_state_used": False,
            "runtime_action_values_used": False,
            "runtime_task_identity_used": False,
        }
        for name, expected in runtime_guards.items():
            if payload.get(name) is not expected:
                raise RuntimeError(f"config must declare {name}={expected}")
        config = cls(
            warmup_transitions=int(payload["warmup_transitions"]),
            window=int(payload["window"]),
            state_response_ratio_max=float(payload["state_response_ratio_max"]),
            planning_churn_ratio_min=float(payload["planning_churn_ratio_min"]),
            state_action_gap_ratio_min=float(payload["state_action_gap_ratio_min"]),
        )
        if config.warmup_transitions < 2 or config.window < 2:
            raise RuntimeError("warmup_transitions and window must both be at least two")
        if not 0.0 < config.state_response_ratio_max < 1.0:
            raise RuntimeError("state_response_ratio_max must be in (0, 1)")
        if config.planning_churn_ratio_min <= 1.0:
            raise RuntimeError("planning_churn_ratio_min must be greater than one")
        if config.state_action_gap_ratio_min <= 1.0:
            raise RuntimeError("state_action_gap_ratio_min must be greater than one")
        return config


@dataclass(frozen=True)
class SelfReferenceDecision:
    query: int
    warmup_complete: bool
    raw_reject: bool
    alarm: bool
    consecutive_raw_rejects: int
    back_route_acceleration: float
    front_state_action_gap: float
    front_state_route_jump: float
    baseline_back_route_acceleration: float
    baseline_front_state_action_gap: float
    baseline_front_state_route_jump: float
    planning_churn_ratio: float
    state_action_gap_ratio: float
    state_response_ratio: float
    planning_churn_ratio_min: float
    state_action_gap_ratio_min: float
    state_response_ratio_max: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SelfReferenceCouplingCollapseAlarm:
    """Causal selector with a frozen within-episode prefix as its only reference."""

    metric_names = (
        "back_route_acceleration",
        "front_state_action_gap",
        "front_state_route_jump",
        "baseline_back_route_acceleration",
        "baseline_front_state_action_gap",
        "baseline_front_state_route_jump",
        "planning_churn_ratio",
        "state_action_gap_ratio",
        "state_response_ratio",
        "planning_churn_ratio_min",
        "state_action_gap_ratio_min",
        "state_response_ratio_max",
    )

    def __init__(self, config: SelfReferenceConfig) -> None:
        self.config = config
        self.query = -1
        self.previous_front_state: np.ndarray | None = None
        self.observables: list[tuple[float, float, float]] = []
        self.consecutive = 0

    def update(self, route: np.ndarray) -> SelfReferenceDecision:
        self.query += 1
        planning, gap, response, front_state = route_observables(
            route, self.previous_front_state
        )
        self.previous_front_state = front_state
        self.observables.append((planning, gap, response))

        start = 1
        stop = 1 + self.config.warmup_transitions
        warmup_complete = self.query >= stop - 1
        evaluation_ready = self.query >= stop + self.config.window - 1
        baseline = np.full(3, np.nan, dtype=np.float64)
        ratios = np.full(3, np.nan, dtype=np.float64)
        raw_reject = False
        if warmup_complete:
            baseline = np.nanmedian(
                np.asarray(self.observables[start:stop], dtype=np.float64), axis=0
            )
        if evaluation_ready:
            recent = np.nanmedian(
                np.asarray(
                    self.observables[
                        self.query - self.config.window + 1 : self.query + 1
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )
            ratios = recent / np.maximum(baseline, 1e-12)
            raw_reject = bool(
                ratios[2] <= self.config.state_response_ratio_max
                and ratios[0] >= self.config.planning_churn_ratio_min
                and ratios[1] >= self.config.state_action_gap_ratio_min
            )
        self.consecutive = self.consecutive + 1 if raw_reject else 0
        return SelfReferenceDecision(
            query=self.query,
            warmup_complete=warmup_complete,
            raw_reject=raw_reject,
            alarm=raw_reject,
            consecutive_raw_rejects=self.consecutive,
            back_route_acceleration=planning,
            front_state_action_gap=gap,
            front_state_route_jump=response,
            baseline_back_route_acceleration=float(baseline[0]),
            baseline_front_state_action_gap=float(baseline[1]),
            baseline_front_state_route_jump=float(baseline[2]),
            planning_churn_ratio=float(ratios[0]),
            state_action_gap_ratio=float(ratios[1]),
            state_response_ratio=float(ratios[2]),
            planning_churn_ratio_min=self.config.planning_churn_ratio_min,
            state_action_gap_ratio_min=self.config.state_action_gap_ratio_min,
            state_response_ratio_max=self.config.state_response_ratio_max,
        )
