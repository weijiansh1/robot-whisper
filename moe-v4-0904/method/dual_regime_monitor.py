#!/usr/bin/env python3
"""Stateful single-rollout runtime for the frozen dual-regime MoE alarm."""

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
INSTABILITY_LAYER_POSITION = 3  # L5 in (L2, L3, L4, L5, L12, L13, L14, L15)
WIDTH = 4
LOCK_CONFIRMATIONS = 4
INSTABILITY_CONFIRMATIONS = 8


def normalize(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class DualRegimeTaskProfile:
    task: str
    lock_threshold: float
    instability_threshold: float

    @classmethod
    def load(cls, path: Path, task: str) -> "DualRegimeTaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.dual_regime_v4.profile.v1":
                raise ValueError("unknown dual-regime profile schema")
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
            )


class CausalFloat32Mean:
    """Match the float32 cumulative-sum path used by the sealed replay."""

    def __init__(self, width: int) -> None:
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


class ConfigurableDualRegimeMonitor:
    """Consume router snapshots with a frozen, task-specific head configuration."""

    def __init__(
        self,
        profile: DualRegimeTaskProfile,
        *,
        lock_representation: str,
        lock_width: int,
        lock_confirmations: int,
        instability_layer: str,
        instability_width: int,
        instability_confirmations: int,
    ) -> None:
        if lock_representation != "all_median" and lock_representation not in LAYER_NAMES:
            raise ValueError(f"unsupported lock representation: {lock_representation}")
        if instability_layer not in LAYER_NAMES:
            raise ValueError(f"unsupported instability layer: {instability_layer}")
        sizes = (
            lock_width,
            lock_confirmations,
            instability_width,
            instability_confirmations,
        )
        if min(sizes) < 1:
            raise ValueError("window widths and confirmation counts must be positive")
        self.profile = profile
        self.lock_representation = lock_representation
        self.lock_confirmations = lock_confirmations
        self.instability_layer_position = LAYER_NAMES.index(instability_layer)
        self.instability_confirmations = instability_confirmations
        self.previous_action_route: np.ndarray | None = None
        self.lock_mean = CausalFloat32Mean(lock_width)
        self.instability_mean = CausalFloat32Mean(instability_width)
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

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray | None = None
    ) -> dict[str, Any]:
        del expert_ids  # Dense probabilities fully determine this detector.
        self.query += 1
        current = self._action_route(router_prob)
        layer_mobility = np.full(EXPECTED_LAYERS, np.nan, dtype=np.float32)
        if self.previous_action_route is not None:
            affinity = np.sqrt(current * self.previous_action_route).sum(axis=-1)
            token_mobility = np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))
            layer_mobility = token_mobility.mean(axis=-1)
        self.previous_action_route = current

        if self.lock_representation == "all_median":
            lock_mobility = float(np.median(layer_mobility))
        else:
            lock_position = LAYER_NAMES.index(self.lock_representation)
            lock_mobility = float(layer_mobility[lock_position])
        instability_mobility = float(layer_mobility[self.instability_layer_position])
        lock_mean = self.lock_mean.update(lock_mobility)
        instability_mean = self.instability_mean.update(instability_mobility)
        lock_crossing = bool(
            np.isfinite(lock_mean) and -lock_mean > self.profile.lock_threshold
        )
        instability_crossing = bool(
            np.isfinite(instability_mean)
            and instability_mean > self.profile.instability_threshold
        )
        self.lock_run = self.lock_run + 1 if lock_crossing else 0
        self.instability_run = self.instability_run + 1 if instability_crossing else 0

        new_lock = (
            self.lock_run >= self.lock_confirmations and self.first_lock_query < 0
        )
        new_instability = (
            self.instability_run >= self.instability_confirmations
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

        if self.alarm_latched:
            state = "alarm"
        elif lock_crossing or instability_crossing:
            state = "suspect"
        else:
            state = "normal"
        return {
            "query": self.query,
            "state": state,
            "alarm": self.alarm_latched,
            "first_alarm_query": self.first_alarm_query,
            "first_alarm_branch": self.first_alarm_branch,
            "lock_crossing": lock_crossing,
            "instability_crossing": instability_crossing,
            "lock_run": self.lock_run,
            "instability_run": self.instability_run,
            "lock_mobility": lock_mobility,
            "lock_mobility_mean": lock_mean,
            "lock_representation": self.lock_representation,
            # Backward-compatible names for the original all-layer-median head.
            "combined_mobility": lock_mobility,
            "instability_mobility": instability_mobility,
            "combined_mobility_w4": lock_mean,
            "instability_mobility_w4": instability_mean,
            "lock_threshold": self.profile.lock_threshold,
            "instability_threshold": self.profile.instability_threshold,
        }


class DualRegimeMonitor(ConfigurableDualRegimeMonitor):
    """Consume one HB router snapshot for the frozen v4 configuration."""

    def __init__(self, profile: DualRegimeTaskProfile) -> None:
        super().__init__(
            profile,
            lock_representation="all_median",
            lock_width=WIDTH,
            lock_confirmations=LOCK_CONFIRMATIONS,
            instability_layer=LAYER_NAMES[INSTABILITY_LAYER_POSITION],
            instability_width=WIDTH,
            instability_confirmations=INSTABILITY_CONFIRMATIONS,
        )
