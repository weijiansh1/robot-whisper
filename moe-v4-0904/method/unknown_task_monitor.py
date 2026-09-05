#!/usr/bin/env python3
"""Single-rollout cold-start alarm calibrated from its first four MoE routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dual_regime_monitor import (
    ACTION,
    EXPECTED_EXPERTS,
    EXPECTED_LAYERS,
    FINAL_FLOW,
    INSTABILITY_CONFIRMATIONS,
    INSTABILITY_LAYER_POSITION,
    LOCK_CONFIRMATIONS,
    WIDTH,
    CausalFloat32Mean,
    normalize,
)


@dataclass(frozen=True)
class ColdStartReference:
    task_names: np.ndarray
    action_centroids: np.ndarray
    lock_ratios: np.ndarray
    instability_cutoffs: np.ndarray
    prefix: int
    lock_ratio_quantile: float
    instability_neighbors: int
    instability_neighbor_quantile: float

    @classmethod
    def load(cls, path: Path) -> "ColdStartReference":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.unknown_task_reference.v1":
                raise ValueError("unknown cold-start reference schema")
            return cls(
                task_names=archive["task_names"].astype(str),
                action_centroids=archive["action_centroids"].astype(np.float32),
                lock_ratios=archive["lock_ratios"].astype(np.float32),
                instability_cutoffs=archive["instability_cutoffs"].astype(
                    np.float32
                ),
                prefix=int(archive["prefix"]),
                lock_ratio_quantile=float(archive["lock_ratio_quantile"]),
                instability_neighbors=int(archive["instability_neighbors"]),
                instability_neighbor_quantile=float(
                    archive["instability_neighbor_quantile"]
                ),
            )


class UnknownTaskMonitor:
    """Consume one route snapshot per query without receiving a task ID."""

    def __init__(self, reference: ColdStartReference) -> None:
        if reference.prefix != 4:
            raise ValueError("this monitor expects a four-query prefix")
        if len(reference.task_names) < reference.instability_neighbors:
            raise ValueError("insufficient reference tasks")
        self.reference = reference
        self.previous_action_route: np.ndarray | None = None
        self.fingerprint_sum = np.zeros(
            (EXPECTED_LAYERS, EXPECTED_EXPERTS), dtype=np.float32
        )
        self.early_mobility: list[float] = []
        self.lock_mean = CausalFloat32Mean(WIDTH)
        self.instability_mean = CausalFloat32Mean(WIDTH)
        self.lock_run = 0
        self.instability_run = 0
        self.query = -1
        self.lock_cutoff = float("nan")
        self.instability_cutoff = float("nan")
        self.nearest_reference_tasks: tuple[str, ...] = ()
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
            raise ValueError(f"expected HB router shape {expected}, got {probability.shape}")
        return normalize(probability[:, FINAL_FLOW, ACTION, :])

    def _calibrate(self) -> None:
        scale = float(np.mean(self.early_mobility))
        ratio = float(
            np.quantile(
                self.reference.lock_ratios,
                self.reference.lock_ratio_quantile,
                method="higher",
            )
        )
        self.lock_cutoff = ratio * scale

        probability = self.fingerprint_sum / self.reference.prefix
        embedding = np.sqrt(np.maximum(probability, 0.0)).reshape(-1)
        distance = np.square(
            self.reference.action_centroids - embedding[None, :]
        ).sum(axis=1)
        nearest = np.argsort(distance)[: self.reference.instability_neighbors]
        self.nearest_reference_tasks = tuple(
            self.reference.task_names[nearest].astype(str).tolist()
        )
        self.instability_cutoff = float(
            np.quantile(
                self.reference.instability_cutoffs[nearest],
                self.reference.instability_neighbor_quantile,
                method="higher",
            )
        )

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray | None = None
    ) -> dict[str, Any]:
        del expert_ids
        self.query += 1
        current = self._action_route(router_prob)
        if self.query < self.reference.prefix:
            self.fingerprint_sum += current.mean(axis=1)

        layer_mobility = np.full(EXPECTED_LAYERS, np.nan, dtype=np.float32)
        if self.previous_action_route is not None:
            affinity = np.sqrt(current * self.previous_action_route).sum(axis=-1)
            token_mobility = np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))
            layer_mobility = token_mobility.mean(axis=-1)
            if self.query < self.reference.prefix:
                self.early_mobility.append(float(np.median(layer_mobility)))
        self.previous_action_route = current
        if self.query == self.reference.prefix - 1:
            self._calibrate()

        combined_mobility = float(np.median(layer_mobility))
        instability_mobility = float(layer_mobility[INSTABILITY_LAYER_POSITION])
        combined_w4 = self.lock_mean.update(combined_mobility)
        instability_w4 = self.instability_mean.update(instability_mobility)
        calibrated = bool(
            np.isfinite(self.lock_cutoff)
            and np.isfinite(self.instability_cutoff)
        )
        lock_crossing = bool(
            calibrated
            and np.isfinite(combined_w4)
            and combined_w4 < self.lock_cutoff
        )
        instability_crossing = bool(
            calibrated
            and np.isfinite(instability_w4)
            and instability_w4 > self.instability_cutoff
        )
        self.lock_run = self.lock_run + 1 if lock_crossing else 0
        self.instability_run = self.instability_run + 1 if instability_crossing else 0

        new_lock = self.lock_run >= LOCK_CONFIRMATIONS and self.first_lock_query < 0
        new_instability = (
            self.instability_run >= INSTABILITY_CONFIRMATIONS
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
        elif not calibrated:
            state = "calibrating"
        else:
            state = "normal"
        return {
            "query": self.query,
            "state": state,
            "alarm": self.alarm_latched,
            "first_alarm_query": self.first_alarm_query,
            "first_alarm_branch": self.first_alarm_branch,
            "calibrated": calibrated,
            "lock_crossing": lock_crossing,
            "instability_crossing": instability_crossing,
            "lock_cutoff": self.lock_cutoff,
            "instability_cutoff": self.instability_cutoff,
            "nearest_reference_tasks": self.nearest_reference_tasks,
        }
