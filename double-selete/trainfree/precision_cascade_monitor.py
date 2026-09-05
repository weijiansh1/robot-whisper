#!/usr/bin/env python3
"""Single-rollout train-free high-precision routing monitor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import online_multihead_alarm as v1
from route_derivative_monitor import SealedRoutePrefixExtractor
from single_rollout_monitor import scalar_midrank


BASE_DETECTORS = (
    "mobility_w4_k1",
    "mobility_w4_k2",
    "mobility_w4_k4",
    "mobility_w8_k3",
)
GATED_DETECTORS = (
    "loop_confirmed",
    "static_confirmed",
    "feedback_confirmed",
    "typed_confirmed",
)
DETECTORS = (*BASE_DETECTORS, *GATED_DETECTORS)
ROUTE_HEADS = tuple(v1.HEADS)
QUANTILES = (0.95, 0.975, 0.99, 0.995)
PRIMARY_DETECTOR = "mobility_w4_k4"
PRIMARY_QUANTILE = 0.99
HEAD_GATE = 0.80
LOOP_LOOKBACK = 6
HEAD_PERSISTENCE = 2
MOBILITY_WIDTH = 4
MOBILITY_CONFIRMATIONS = 4


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Causal mean requiring a complete finite trailing window."""
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, np.nan, dtype=np.float32)
    for query in range(width - 1, values.shape[1]):
        window = values[:, query - width + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].mean(axis=1)
    return output


def consecutive_low_score(smoothed_mobility: np.ndarray, count: int) -> np.ndarray:
    """Higher is more abnormal: all recent mobility values must be low."""
    values = np.asarray(smoothed_mobility, dtype=np.float32)
    output = np.full_like(values, np.nan, dtype=np.float32)
    for query in range(count - 1, values.shape[1]):
        window = values[:, query - count + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = -window[good].max(axis=1)
    return output


def mobility_detector_scores(
    route_mobility: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    route_mobility = np.asarray(route_mobility, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if route_mobility.shape != valid.shape:
        raise ValueError("route mobility and valid mask must have the same shape")
    w4 = trailing_mean(route_mobility, 4)
    w8 = trailing_mean(route_mobility, 8)
    scores = {
        "mobility_w4_k1": consecutive_low_score(w4, 1),
        "mobility_w4_k2": consecutive_low_score(w4, 2),
        "mobility_w4_k4": consecutive_low_score(w4, 4),
        "mobility_w8_k3": consecutive_low_score(w8, 3),
    }
    for values in scores.values():
        values[~valid] = np.nan
    return scores


def past_any(values: np.ndarray, cutoff: float, lookback: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros(values.shape, dtype=bool)
    for query in range(1, values.shape[1]):
        window = values[:, max(0, query - lookback) : query]
        output[:, query] = np.any(np.isfinite(window) & (window > cutoff), axis=1)
    return output


def consecutive_above(values: np.ndarray, cutoff: float, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros(values.shape, dtype=bool)
    for query in range(count - 1, values.shape[1]):
        window = values[:, query - count + 1 : query + 1]
        output[:, query] = np.all(np.isfinite(window) & (window > cutoff), axis=1)
    return output


def typed_gates(route_heads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    shape = route_heads["instability"].shape
    if any(np.asarray(values).shape != shape for values in route_heads.values()):
        raise ValueError("route-head arrays must have matching shapes")
    static_route = np.maximum(
        route_heads["lock_in"], route_heads["flat_narrow_support"]
    )
    loop = past_any(route_heads["instability"], HEAD_GATE, LOOP_LOOKBACK)
    static = consecutive_above(static_route, HEAD_GATE, HEAD_PERSISTENCE)
    feedback = consecutive_above(
        route_heads["feedback_decoupling"], HEAD_GATE, HEAD_PERSISTENCE
    )
    return {
        "loop_confirmed": loop,
        "static_confirmed": static,
        "feedback_confirmed": feedback,
        "typed_confirmed": loop | static | feedback,
    }


class StreamingRouteHeads:
    """Normalize current features against a frozen same-query task profile."""

    def __init__(self, profile: "TaskProfile") -> None:
        self.profile = profile
        self.instant_history: dict[str, list[float]] = {
            name: [] for name in ROUTE_HEADS
        }

    def update(self, query: int, features: np.ndarray) -> dict[str, float]:
        feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
        rank: dict[tuple[str, int], float] = {}
        members = {
            member for head_members in v1.HEADS.values() for member in head_members
        }
        for name, direction in members:
            reference = self.profile.route_reference[
                self.profile.reference_valid[:, query], query, feature_index[name]
            ]
            value = direction * float(features[feature_index[name]])
            rank[(name, direction)] = scalar_midrank(direction * reference, value)

        output: dict[str, float] = {}
        for head, head_members in v1.HEADS.items():
            components = np.asarray([rank[member] for member in head_members])
            instant = (
                float(components.mean())
                if np.isfinite(components).all()
                else float("nan")
            )
            self.instant_history[head].append(instant)
            recent = self.instant_history[head][-v1.SMOOTHING_WIDTH :]
            output[head] = (
                float(np.mean(recent))
                if len(recent) == v1.SMOOTHING_WIDTH and np.isfinite(recent).all()
                else float("nan")
            )
        return output


@dataclass(frozen=True)
class TaskProfile:
    task: str
    route_reference: np.ndarray
    reference_valid: np.ndarray
    detector_names: tuple[str, ...]
    quantiles: np.ndarray
    thresholds: np.ndarray

    @classmethod
    def load(cls, path: Path, task: str) -> "TaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.precision_cascade.profile.v1":
                raise ValueError("unknown precision-cascade profile schema")
            task_names = archive["task_names"].astype(str).tolist()
            if task not in task_names:
                raise KeyError(f"task not present in profile: {task}")
            task_position = task_names.index(task)
            reference_task = archive["reference_task_index"].astype(int)
            take = reference_task == task_position
            return cls(
                task=task,
                route_reference=np.asarray(archive["route_reference"][take]),
                reference_valid=np.asarray(
                    archive["reference_valid"][take], dtype=bool
                ),
                detector_names=tuple(archive["detector_names"].astype(str)),
                quantiles=np.asarray(archive["quantiles"], dtype=np.float32),
                thresholds=np.asarray(
                    archive["thresholds"][task_position], dtype=np.float32
                ),
            )

    def threshold(self, detector: str, quantile: float) -> float:
        detector_index = self.detector_names.index(detector)
        quantile_index = int(np.argmin(np.abs(self.quantiles - quantile)))
        if not np.isclose(float(self.quantiles[quantile_index]), quantile):
            raise KeyError(f"quantile not present in profile: {quantile}")
        return float(self.thresholds[detector_index, quantile_index])


class PrecisionCascadeMonitor:
    """Stateful monitor whose update contract is one MoE snapshot at a time."""

    def __init__(
        self,
        profile: TaskProfile,
        quantile: float = PRIMARY_QUANTILE,
        alarm_detector: str = PRIMARY_DETECTOR,
    ) -> None:
        if alarm_detector not in DETECTORS:
            raise KeyError(f"unknown alarm detector: {alarm_detector}")
        self.profile = profile
        self.quantile = quantile
        self.alarm_detector = alarm_detector
        threshold_detector = (
            PRIMARY_DETECTOR
            if alarm_detector in GATED_DETECTORS
            else alarm_detector
        )
        self.threshold = profile.threshold(threshold_detector, quantile)
        self.primary_threshold = profile.threshold(PRIMARY_DETECTOR, quantile)
        self.extractor = SealedRoutePrefixExtractor()
        self.head_normalizer = StreamingRouteHeads(profile)
        self.mobility: list[float] = []
        self.w4: list[float] = []
        self.w8: list[float] = []
        self.route_head_history: dict[str, list[float]] = {
            name: [] for name in ROUTE_HEADS
        }
        self.latched = False
        self.first_alarm_query = -1
        self.query = -1

    @staticmethod
    def _finite_mean(values: list[float], width: int) -> float:
        recent = np.asarray(values[-width:], dtype=np.float64)
        return (
            float(recent.mean())
            if len(recent) == width and np.isfinite(recent).all()
            else float("nan")
        )

    @staticmethod
    def _low_score(values: list[float], count: int) -> float:
        recent = np.asarray(values[-count:], dtype=np.float64)
        return (
            -float(recent.max())
            if len(recent) == count and np.isfinite(recent).all()
            else float("nan")
        )

    def _current_gates(self) -> dict[str, bool]:
        instability = np.asarray(
            self.route_head_history["instability"], dtype=np.float64
        )
        prior = instability[max(0, len(instability) - 1 - LOOP_LOOKBACK) : -1]
        loop = bool(np.any(np.isfinite(prior) & (prior > HEAD_GATE)))
        static_values = np.maximum(
            np.asarray(self.route_head_history["lock_in"], dtype=np.float64),
            np.asarray(
                self.route_head_history["flat_narrow_support"], dtype=np.float64
            ),
        )[-HEAD_PERSISTENCE:]
        feedback_values = np.asarray(
            self.route_head_history["feedback_decoupling"], dtype=np.float64
        )[-HEAD_PERSISTENCE:]
        static = bool(
            len(static_values) == HEAD_PERSISTENCE
            and np.all(np.isfinite(static_values) & (static_values > HEAD_GATE))
        )
        feedback = bool(
            len(feedback_values) == HEAD_PERSISTENCE
            and np.all(np.isfinite(feedback_values) & (feedback_values > HEAD_GATE))
        )
        return {
            "loop_confirmed": loop,
            "static_confirmed": static,
            "feedback_confirmed": feedback,
            "typed_confirmed": loop or static or feedback,
        }

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray
    ) -> dict[str, Any]:
        self.query += 1
        if self.query >= self.profile.route_reference.shape[1]:
            raise IndexError("query exceeds frozen task profile")
        features = self.extractor.update(router_prob, expert_ids)
        feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
        self.mobility.append(float(features[feature_index["route_mobility"]]))
        self.w4.append(self._finite_mean(self.mobility, 4))
        self.w8.append(self._finite_mean(self.mobility, 8))

        heads = self.head_normalizer.update(self.query, features)
        for name, value in heads.items():
            self.route_head_history[name].append(value)
        gates = self._current_gates()

        scores = {
            "mobility_w4_k1": self._low_score(self.w4, 1),
            "mobility_w4_k2": self._low_score(self.w4, 2),
            "mobility_w4_k4": self._low_score(self.w4, 4),
            "mobility_w8_k3": self._low_score(self.w8, 3),
        }
        primary_score = scores[PRIMARY_DETECTOR]
        for name in GATED_DETECTORS:
            scores[name] = primary_score if gates[name] else float("nan")

        score = scores[self.alarm_detector]
        trigger = bool(np.isfinite(score) and score > self.threshold)
        if trigger and not self.latched:
            self.latched = True
            self.first_alarm_query = self.query

        current_low = bool(
            np.isfinite(self.w4[-1]) and -self.w4[-1] > self.primary_threshold
        )
        low_run = 0
        for value in reversed(self.w4):
            if np.isfinite(value) and -value > self.primary_threshold:
                low_run += 1
            else:
                break
        state = "alarm" if self.latched else ("suspect" if current_low else "normal")
        phenotype = "none"
        if gates["static_confirmed"]:
            phenotype = "static"
        elif gates["loop_confirmed"]:
            phenotype = "loop"
        elif gates["feedback_confirmed"]:
            phenotype = "feedback"
        return {
            "query": self.query,
            "state": state,
            "suspect": state == "suspect",
            "alarm": self.latched,
            "first_alarm_query": self.first_alarm_query,
            "score": score,
            "threshold": self.threshold,
            "low_mobility_run": low_run,
            "phenotype": phenotype,
            "route_heads": heads,
            "gates": gates,
        }
