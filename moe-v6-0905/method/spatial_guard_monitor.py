#!/usr/bin/env python3
"""Stateful runtime monitor for the task-conditional spatial guard v6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FINAL_FLOW = 9
ACTION = slice(1, 11)
EXPECTED_LAYERS = 8
EXPECTED_EXPERTS = 32
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")


def normalize(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class SpatialGuardTaskProfile:
    task: str
    lock_threshold: float
    instability_threshold: float
    lock_representation: str
    lock_direction: str
    lock_width: int
    lock_confirmations: int
    instability_layer: str
    instability_width: int
    instability_confirmations: int

    @classmethod
    def load(cls, path: Path, task: str) -> "SpatialGuardTaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.spatial_guard_v6.profile.v1":
                raise ValueError("unknown spatial-guard profile schema")
            names = archive["task_names"].astype(str).tolist()
            if task not in names:
                raise KeyError(f"task not present in profile: {task}")
            position = names.index(task)
            return cls(
                task=task,
                lock_threshold=float(archive["lock_thresholds"][position]),
                instability_threshold=float(
                    archive["instability_thresholds"][position]
                ),
                lock_representation=str(archive["lock_representations"][position]),
                lock_direction=str(archive["lock_directions"][position]),
                lock_width=int(archive["lock_widths"][position]),
                lock_confirmations=int(archive["lock_confirmations"][position]),
                instability_layer=str(archive["instability_layers"][position]),
                instability_width=int(archive["instability_widths"][position]),
                instability_confirmations=int(
                    archive["instability_confirmations"][position]
                ),
            )


class CausalFloat32Mean:
    """Match the float32 cumulative-sum path used by the sealed evaluator."""

    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError("mean width must be positive")
        self.width = width
        self.prefix = [np.float32(0.0)]
        self.finite_prefix = [0]

    def update(self, value: float) -> float:
        finite = bool(np.isfinite(value))
        filled = np.float32(value if finite else 0.0)
        self.prefix.append(np.float32(self.prefix[-1] + filled))
        self.finite_prefix.append(self.finite_prefix[-1] + int(finite))
        if len(self.prefix) <= self.width:
            return float("nan")
        finite_count = self.finite_prefix[-1] - self.finite_prefix[-1 - self.width]
        if finite_count != self.width:
            return float("nan")
        total = np.float32(self.prefix[-1] - self.prefix[-1 - self.width])
        return float(np.float32(total / self.width))


def _base_representation(layer_mobility: np.ndarray, name: str) -> float:
    if name in LAYER_NAMES:
        return float(layer_mobility[LAYER_NAMES.index(name)])
    if name == "front_back_mean_gap":
        return float(np.abs(layer_mobility[:4].mean() - layer_mobility[4:].mean()))
    try:
        group, operation = name.split("_", 1)
        selected = layer_mobility[
            {"front": slice(0, 4), "back": slice(4, 8), "all": slice(0, 8)}[group]
        ]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported lock representation: {name}") from error
    operations = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "median": np.median,
        "std": np.std,
    }
    if operation not in operations:
        raise ValueError(f"unsupported lock representation: {name}")
    return float(operations[operation](selected))


class SpatialGuardMonitor:
    """Consume one HB router snapshot at a time and latch the first alarm."""

    def __init__(self, profile: SpatialGuardTaskProfile) -> None:
        if profile.lock_direction not in {"low", "high"}:
            raise ValueError(f"unsupported lock direction: {profile.lock_direction}")
        if profile.instability_layer not in LAYER_NAMES:
            raise ValueError(
                f"unsupported instability layer: {profile.instability_layer}"
            )
        if min(profile.lock_confirmations, profile.instability_confirmations) < 1:
            raise ValueError("confirmation counts must be positive")
        base_name = profile.lock_representation.removeprefix("delta_")
        _base_representation(np.ones(EXPECTED_LAYERS, np.float32), base_name)
        self.profile = profile
        self.previous_action_route: np.ndarray | None = None
        self.previous_lock_base = float("nan")
        self.lock_mean = CausalFloat32Mean(profile.lock_width)
        self.instability_mean = CausalFloat32Mean(profile.instability_width)
        self.lock_run = 0
        self.instability_run = 0
        self.query = -1
        self.alarm_latched = False
        self.first_alarm_query = -1
        self.first_alarm_branch = "none"
        self.first_lock_query = -1
        self.first_instability_query = -1

    @staticmethod
    def _action_route(router_prob: np.ndarray) -> np.ndarray:
        probability = np.asarray(router_prob)
        expected = (EXPECTED_LAYERS, 10, 11, EXPECTED_EXPERTS)
        if probability.shape != expected:
            raise ValueError(
                f"expected HB router shape {expected}, got {probability.shape}"
            )
        return normalize(probability[:, FINAL_FLOW, ACTION, :])

    def _lock_value(self, layer_mobility: np.ndarray) -> float:
        representation = self.profile.lock_representation
        base_name = representation.removeprefix("delta_")
        current = _base_representation(layer_mobility, base_name)
        if representation.startswith("delta_"):
            value = float(np.abs(current - self.previous_lock_base))
        else:
            value = current
        self.previous_lock_base = current
        return value

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray | None = None
    ) -> dict[str, Any]:
        del expert_ids
        self.query += 1
        current = self._action_route(router_prob)
        layer_mobility = np.full(EXPECTED_LAYERS, np.nan, dtype=np.float32)
        if self.previous_action_route is not None:
            affinity = np.sqrt(current * self.previous_action_route).sum(axis=-1)
            token_mobility = np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))
            layer_mobility = token_mobility.mean(axis=-1)
        self.previous_action_route = current

        lock_value = self._lock_value(layer_mobility)
        instability_value = float(
            layer_mobility[LAYER_NAMES.index(self.profile.instability_layer)]
        )
        lock_smoothed = self.lock_mean.update(lock_value)
        instability_smoothed = self.instability_mean.update(instability_value)
        oriented_lock = (
            -lock_smoothed if self.profile.lock_direction == "low" else lock_smoothed
        )
        lock_crossing = bool(
            np.isfinite(oriented_lock) and oriented_lock > self.profile.lock_threshold
        )
        instability_crossing = bool(
            np.isfinite(instability_smoothed)
            and instability_smoothed > self.profile.instability_threshold
        )
        self.lock_run = self.lock_run + 1 if lock_crossing else 0
        self.instability_run = self.instability_run + 1 if instability_crossing else 0

        new_lock = (
            self.lock_run >= self.profile.lock_confirmations
            and self.first_lock_query < 0
        )
        new_instability = (
            self.instability_run >= self.profile.instability_confirmations
            and self.first_instability_query < 0
        )
        if new_lock:
            self.first_lock_query = self.query
        if new_instability:
            self.first_instability_query = self.query
        if (new_lock or new_instability) and not self.alarm_latched:
            self.alarm_latched = True
            self.first_alarm_query = self.query
            if new_lock and new_instability:
                self.first_alarm_branch = "lock+instability"
            elif new_lock:
                self.first_alarm_branch = "lock"
            else:
                self.first_alarm_branch = "instability"

        return {
            "query": self.query,
            "state": (
                "alarm"
                if self.alarm_latched
                else "suspect"
                if lock_crossing or instability_crossing
                else "normal"
            ),
            "alarm": self.alarm_latched,
            "first_alarm_query": self.first_alarm_query,
            "first_alarm_branch": self.first_alarm_branch,
            "first_lock_query": self.first_lock_query,
            "first_instability_query": self.first_instability_query,
            "lock_crossing": lock_crossing,
            "instability_crossing": instability_crossing,
            "lock_run": self.lock_run,
            "instability_run": self.instability_run,
            "lock_value": lock_value,
            "lock_smoothed": lock_smoothed,
            "lock_representation": self.profile.lock_representation,
            "lock_direction": self.profile.lock_direction,
            "instability_value": instability_value,
            "instability_smoothed": instability_smoothed,
            "lock_threshold": self.profile.lock_threshold,
            "instability_threshold": self.profile.instability_threshold,
        }
