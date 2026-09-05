#!/usr/bin/env python3
"""Task-agnostic, train-free causal monitor for HiMoE router dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "himoe.intrinsic_guard_v7.profile.v1"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
BACK = slice(4, 8)
FINAL_FLOW = 9
ACTION = slice(1, 11)
N_EXPERTS = 32
EPSILON = 1e-6

FREEZE_BASELINE_START = 1
FREEZE_BASELINE_COUNT = 4
FREEZE_WIDTH = 6
FREEZE_CONFIRMATIONS = 1

ACCELERATION_BASELINE_START = 2
ACCELERATION_BASELINE_COUNT = 6
ACCELERATION_WIDTH = 3
ACCELERATION_CONFIRMATIONS = 8

PERIODICITY_BASELINE_START = 2
PERIODICITY_BASELINE_COUNT = 5
PERIODICITY_WIDTH = 6
PERIODICITY_CONFIRMATIONS = 4
LAGS = (1, 2, 3, 4)


def normalize_probability(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    numerator = np.minimum(left, right).sum(axis=-1)
    denominator = np.maximum(left, right).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def causal_trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"trailing mean expects [episode, query], got {values.shape}")
    if width < 1:
        raise ValueError("width must be positive")
    output = np.full_like(values, np.nan)
    for query in range(width - 1, values.shape[1]):
        window = values[:, query - width + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].mean(axis=1, dtype=np.float32)
    return output


def persistent_score(values: np.ndarray, confirmations: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"persistent score expects [episode, query], got {values.shape}")
    if confirmations < 1:
        raise ValueError("confirmations must be positive")
    if confirmations == 1:
        return values.copy()
    output = np.full_like(values, np.nan)
    for query in range(confirmations - 1, values.shape[1]):
        window = values[:, query - confirmations + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].min(axis=1)
    return output


def row_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmax(values[good], axis=1)
    return output


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.sort(finite[np.isfinite(finite)])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if len(finite) < 32:
        raise ValueError(f"at least 32 finite references are required, got {len(finite)}")
    position = int(np.ceil(quantile * (len(finite) - 1)))
    return float(finite[position])


def _baseline(
    values: np.ndarray, start: int, count: int, reducer: str
) -> np.ndarray:
    window = np.asarray(values[:, start : start + count], dtype=np.float32)
    if window.shape[1] != count:
        return np.full((len(values), *values.shape[2:]), np.nan, dtype=np.float32)
    if reducer == "mean":
        good = np.isfinite(window).all(axis=1)
        result = np.full((len(values), *values.shape[2:]), np.nan, dtype=np.float32)
        reduced = window.mean(axis=1, dtype=np.float32)
        result[good] = reduced[good]
        return result
    if reducer == "median":
        good = np.isfinite(window).all(axis=1)
        result = np.full((len(values), *values.shape[2:]), np.nan, dtype=np.float32)
        reduced = np.median(window, axis=1).astype(np.float32)
        result[good] = reduced[good]
        return result
    raise ValueError(f"unsupported reducer: {reducer}")


def intrinsic_score_arrays(
    layer_mobility: np.ndarray,
    route_acceleration: np.ndarray,
    lag_periodicity: np.ndarray,
    periodicity_scale: float,
) -> dict[str, np.ndarray]:
    """Compute all prefix-causal v7 score streams from cached raw features."""
    mobility = np.asarray(layer_mobility, dtype=np.float32)
    acceleration = np.asarray(route_acceleration, dtype=np.float32)
    periodicity = np.asarray(lag_periodicity, dtype=np.float32)
    if mobility.ndim != 3 or mobility.shape[2] != len(LAYER_NAMES):
        raise ValueError(f"layer mobility must be [episode, query, 8], got {mobility.shape}")
    if acceleration.shape != mobility.shape[:2] or periodicity.shape != mobility.shape[:2]:
        raise ValueError("scalar route features do not align with layer mobility")
    if not np.isfinite(periodicity_scale) or periodicity_scale <= 0.0:
        raise ValueError("periodicity scale must be finite and positive")

    freeze_base = _baseline(
        mobility, FREEZE_BASELINE_START, FREEZE_BASELINE_COUNT, "mean"
    )
    freeze_relative = -np.log(
        np.maximum(mobility, EPSILON) / np.maximum(freeze_base[:, None, :], EPSILON)
    )
    freeze_raw = np.full(mobility.shape[:2], np.nan, dtype=np.float32)
    freeze_good = np.isfinite(freeze_relative[:, :, BACK]).all(axis=2)
    freeze_raw[freeze_good] = np.median(
        freeze_relative[:, :, BACK][freeze_good], axis=1
    ).astype(np.float32)
    freeze = causal_trailing_mean(freeze_raw, FREEZE_WIDTH)

    acceleration_base = _baseline(
        acceleration[:, :, None],
        ACCELERATION_BASELINE_START,
        ACCELERATION_BASELINE_COUNT,
        "median",
    )[:, 0]
    acceleration_raw = np.log(
        np.maximum(acceleration, EPSILON)
        / np.maximum(acceleration_base[:, None], EPSILON)
    ).astype(np.float32)
    acceleration_raw[~np.isfinite(acceleration)] = np.nan
    acceleration_smooth = causal_trailing_mean(
        acceleration_raw, ACCELERATION_WIDTH
    )
    acceleration_persistent = persistent_score(
        acceleration_smooth, ACCELERATION_CONFIRMATIONS
    )

    periodicity_base = _baseline(
        periodicity[:, :, None],
        PERIODICITY_BASELINE_START,
        PERIODICITY_BASELINE_COUNT,
        "median",
    )[:, 0]
    periodicity_raw = -(
        periodicity - periodicity_base[:, None]
    ) / np.float32(periodicity_scale)
    periodicity_raw = periodicity_raw.astype(np.float32)
    periodicity_raw[~np.isfinite(periodicity)] = np.nan
    periodicity_smooth = causal_trailing_mean(periodicity_raw, PERIODICITY_WIDTH)
    periodicity_persistent = persistent_score(
        periodicity_smooth, PERIODICITY_CONFIRMATIONS
    )

    return {
        "freeze_raw": freeze_raw,
        "freeze": freeze,
        "acceleration_raw": acceleration_raw,
        "acceleration": acceleration_smooth,
        "acceleration_persistent": acceleration_persistent,
        "periodicity_raw": periodicity_raw,
        "periodicity": periodicity_smooth,
        "periodicity_persistent": periodicity_persistent,
    }


def first_from_score(
    values: np.ndarray, threshold: float, valid: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("score and validity masks do not align")
    trigger = np.isfinite(values) & (values > threshold) & valid
    any_trigger = trigger.any(axis=1)
    first = np.full(len(values), -1, dtype=np.int16)
    first[any_trigger] = trigger[any_trigger].argmax(axis=1).astype(np.int16)
    return first


def first_and(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int16)
    right = np.asarray(right, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("first-alarm arrays do not align")
    return np.where((left >= 0) & (right >= 0), np.maximum(left, right), -1).astype(
        np.int16
    )


def first_or(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int16)
    right = np.asarray(right, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("first-alarm arrays do not align")
    return np.where(left < 0, right, np.where(right < 0, left, np.minimum(left, right))).astype(
        np.int16
    )


@dataclass(frozen=True)
class GlobalIntrinsicProfile:
    freeze_threshold: float
    acceleration_threshold: float
    periodicity_threshold: float
    periodicity_scale: float

    def __post_init__(self) -> None:
        values = (
            self.freeze_threshold,
            self.acceleration_threshold,
            self.periodicity_threshold,
            self.periodicity_scale,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("profile values must be finite")
        if self.periodicity_scale <= 0.0:
            raise ValueError("periodicity scale must be positive")

    @classmethod
    def load(cls, path: Path) -> "GlobalIntrinsicProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                raise ValueError(f"unknown intrinsic profile schema in {path}")
            if "task_names" in archive.files or "task_index" in archive.files:
                raise ValueError("a global profile must not contain task metadata")
            return cls(
                freeze_threshold=float(archive["freeze_threshold"]),
                acceleration_threshold=float(archive["acceleration_threshold"]),
                periodicity_threshold=float(archive["periodicity_threshold"]),
                periodicity_scale=float(archive["periodicity_scale"]),
            )


class IntrinsicGuardMonitor:
    """Stateful one-rollout monitor using no task identity or learned weights."""

    def __init__(self, profile: GlobalIntrinsicProfile) -> None:
        self.profile = profile
        self.query = -1
        self._previous_action_route: np.ndarray | None = None
        self._back_action_history: list[np.ndarray] = []
        self._mobility_history: list[np.ndarray] = []
        self._acceleration_history: list[float] = []
        self._periodicity_history: list[float] = []
        self.freeze_alarm = False
        self.acceleration_alarm = False
        self.periodicity_alarm = False
        self.turbulence_alarm = False
        self.alarm = False
        self.first_freeze_query = -1
        self.first_acceleration_query = -1
        self.first_periodicity_query = -1
        self.first_turbulence_query = -1
        self.first_alarm_query = -1

    @staticmethod
    def _query_features(
        router_prob: np.ndarray,
        previous_action_route: np.ndarray | None,
        back_history: list[np.ndarray],
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        probability = normalize_probability(router_prob)
        expected = (len(LAYER_NAMES), 10, 11, N_EXPERTS)
        if probability.shape != expected:
            raise ValueError(f"router probability must have shape {expected}, got {probability.shape}")
        final_action = probability[:, FINAL_FLOW, ACTION, :]
        if previous_action_route is None:
            layer_mobility = np.full(len(LAYER_NAMES), np.nan, dtype=np.float32)
        else:
            layer_mobility = hellinger(final_action, previous_action_route).mean(
                axis=1, dtype=np.float32
            )

        action_flow = probability[BACK, :, ACTION, :]
        root = np.sqrt(action_flow)
        second = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
        acceleration = float(
            np.linalg.norm(second, axis=-1).mean(dtype=np.float32) / np.sqrt(2.0)
        )

        current_back = final_action[BACK].reshape(40, N_EXPERTS)
        similarities: list[float] = []
        for lag in LAGS:
            if len(back_history) >= lag:
                similarities.append(
                    float(weighted_jaccard(current_back, back_history[-lag]).mean())
                )
            else:
                similarities.append(float("nan"))
        if np.isfinite(similarities[0]) and np.isfinite(similarities[1:]).any():
            periodicity = float(np.nanmax(similarities[1:]) - similarities[0])
        else:
            periodicity = float("nan")
        return layer_mobility, acceleration, periodicity, final_action

    def _score_prefix(self) -> dict[str, float]:
        mobility = np.asarray(self._mobility_history, dtype=np.float32)[None, :, :]
        acceleration = np.asarray(self._acceleration_history, dtype=np.float32)[None, :]
        periodicity = np.asarray(self._periodicity_history, dtype=np.float32)[None, :]
        streams = intrinsic_score_arrays(
            mobility, acceleration, periodicity, self.profile.periodicity_scale
        )
        return {
            name: float(values[0, -1])
            for name, values in streams.items()
            if name in {"freeze", "acceleration_persistent", "periodicity_persistent"}
        }

    def update(self, hb_router_probs: np.ndarray) -> dict[str, Any]:
        self.query += 1
        mobility, acceleration, periodicity, final_action = self._query_features(
            hb_router_probs, self._previous_action_route, self._back_action_history
        )
        current_back = final_action[BACK].reshape(40, N_EXPERTS)
        self._mobility_history.append(mobility)
        self._acceleration_history.append(acceleration)
        self._periodicity_history.append(periodicity)
        self._back_action_history.append(current_back)
        self._previous_action_route = final_action

        scores = self._score_prefix()
        freeze_now = (
            self.query >= FREEZE_BASELINE_START + FREEZE_BASELINE_COUNT
            and np.isfinite(scores["freeze"])
            and scores["freeze"] > self.profile.freeze_threshold
        )
        acceleration_now = (
            self.query
            >= ACCELERATION_BASELINE_START + ACCELERATION_BASELINE_COUNT - 1
            and np.isfinite(scores["acceleration_persistent"])
            and scores["acceleration_persistent"]
            > self.profile.acceleration_threshold
        )
        periodicity_now = (
            self.query >= PERIODICITY_BASELINE_START + PERIODICITY_BASELINE_COUNT - 1
            and np.isfinite(scores["periodicity_persistent"])
            and scores["periodicity_persistent"] > self.profile.periodicity_threshold
        )

        if freeze_now and not self.freeze_alarm:
            self.first_freeze_query = self.query
        if acceleration_now and not self.acceleration_alarm:
            self.first_acceleration_query = self.query
        if periodicity_now and not self.periodicity_alarm:
            self.first_periodicity_query = self.query
        self.freeze_alarm |= freeze_now
        self.acceleration_alarm |= acceleration_now
        self.periodicity_alarm |= periodicity_now

        turbulence_now = self.acceleration_alarm and self.periodicity_alarm
        if turbulence_now and not self.turbulence_alarm:
            self.first_turbulence_query = self.query
        self.turbulence_alarm |= turbulence_now
        alarm_now = self.freeze_alarm or self.turbulence_alarm
        if alarm_now and not self.alarm:
            self.first_alarm_query = self.query
        self.alarm |= alarm_now

        if self.freeze_alarm and self.turbulence_alarm:
            mechanism = "freeze+turbulence"
        elif self.freeze_alarm:
            mechanism = "freeze"
        elif self.turbulence_alarm:
            mechanism = "confirmed_turbulence"
        else:
            mechanism = "none"
        return {
            "query": self.query,
            "alarm": self.alarm,
            "first_alarm_query": self.first_alarm_query,
            "mechanism": mechanism,
            "freeze_alarm": self.freeze_alarm,
            "acceleration_alarm": self.acceleration_alarm,
            "periodicity_alarm": self.periodicity_alarm,
            "turbulence_alarm": self.turbulence_alarm,
            "layer_mobility": mobility.copy(),
            "route_acceleration": acceleration,
            "lag_periodicity": periodicity,
            "freeze_score": scores["freeze"],
            "acceleration_score": scores["acceleration_persistent"],
            "periodicity_score": scores["periodicity_persistent"],
        }
