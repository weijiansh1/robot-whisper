#!/usr/bin/env python3
"""Task-agnostic, train-free online progress-ratio monitor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from progress_ratio import (
    LAGS,
    LAYER_GROUPS,
    SCHEMA,
    action_route,
    group_ratio,
    progress_ratio,
    route_distance,
)


@dataclass(frozen=True)
class GlobalProgressProfile:
    window: int
    confirmations: int
    ratio_threshold: float
    eps_length: float
    layer_group: str

    def __post_init__(self) -> None:
        if self.window < 1 or self.confirmations < 1:
            raise ValueError("window and confirmations must be positive")
        if self.window not in LAGS:
            raise ValueError(f"window {self.window} is not a cached lag")
        if not np.isfinite(self.ratio_threshold) or not 0.0 <= self.ratio_threshold <= 1.0:
            raise ValueError("ratio threshold must lie in [0, 1]")
        if not np.isfinite(self.eps_length) or self.eps_length <= 0.0:
            raise ValueError("eps_length must be finite and positive")
        if self.layer_group not in LAYER_GROUPS:
            raise ValueError(f"unknown layer group: {self.layer_group}")

    @classmethod
    def load(cls, path: Path) -> "GlobalProgressProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                raise ValueError(f"unknown progress profile schema in {path}")
            if "task_names" in archive.files or "task_index" in archive.files:
                raise ValueError("a global profile must not contain task metadata")
            return cls(
                window=int(archive["window"]),
                confirmations=int(archive["confirmations"]),
                ratio_threshold=float(archive["ratio_threshold"]),
                eps_length=float(archive["eps_length"]),
                layer_group=str(archive["layer_group"]),
            )


class ProgressGuardMonitor:
    """Stateful one-rollout monitor using no task identity or learned weights."""

    def __init__(self, profile: GlobalProgressProfile) -> None:
        self.profile = profile
        self.query = -1
        self._routes: deque[np.ndarray] = deque(maxlen=profile.window + 1)
        self._adjacent: deque[np.ndarray] = deque(maxlen=profile.window)
        self._below_run = 0
        self.alarm = False
        self.first_alarm_query = -1

    def _current_ratio(self) -> float:
        if self.query < self.profile.window:
            return float("nan")
        displacement = route_distance(self._routes[-1], self._routes[0])
        length = np.sum(np.stack(tuple(self._adjacent)), axis=0, dtype=np.float32)
        ratio = progress_ratio(
            displacement[None, None, :], length[None, None, :], self.profile.eps_length
        )
        return float(group_ratio(ratio, self.profile.layer_group)[0, 0])

    def update(self, hb_router_probs: np.ndarray) -> dict[str, Any]:
        self.query += 1
        routes = action_route(hb_router_probs)
        if self._routes:
            self._adjacent.append(route_distance(routes, self._routes[-1]))
        self._routes.append(routes)

        ratio = self._current_ratio()
        below = np.isfinite(ratio) and ratio < self.profile.ratio_threshold
        self._below_run = self._below_run + 1 if below else 0
        alarm_now = self._below_run >= self.profile.confirmations
        if alarm_now and not self.alarm:
            self.first_alarm_query = self.query
        self.alarm |= alarm_now
        return {
            "query": self.query,
            "ratio": ratio,
            "below_run": self._below_run,
            "alarm": self.alarm,
            "first_alarm_query": self.first_alarm_query,
        }
