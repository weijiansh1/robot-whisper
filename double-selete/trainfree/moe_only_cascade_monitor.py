#!/usr/bin/env python3
"""Single-rollout pure-MoE persistent-phenotype alarm cascade."""

from __future__ import annotations

from typing import Any

import numpy as np

from precision_cascade_monitor import TaskProfile
from shared_threshold_cascade_monitor import SharedThresholdCascadeMonitor


HEAD_GATE = 0.80
PHENOTYPE_CONFIRMATIONS = 4
LOCK_HEADS = ("lock_in", "flat_narrow_support")


class MoeOnlyCascadeMonitor:
    """Combine low mobility with a persistent MoE lock phenotype."""

    def __init__(self, profile: TaskProfile, quantile: float = 0.95) -> None:
        self.mobility = SharedThresholdCascadeMonitor(profile, quantile=quantile)
        self.head_runs = {name: 0 for name in LOCK_HEADS}
        self.static_latched = False
        self.first_static_query = -1
        self.first_static_head = "none"
        self.warning_latched = False
        self.hard_latched = False
        self.first_warning_query = -1
        self.first_hard_query = -1
        self.warning_branch = "none"
        self.hard_branch = "none"

    @staticmethod
    def _merge_branch(mobility: bool, phenotype: bool, head: str) -> str:
        if mobility and phenotype:
            return f"mobility+{head}"
        if mobility:
            return "mobility"
        if phenotype:
            return head
        return "none"

    def update(self, router_prob: np.ndarray, expert_ids: np.ndarray) -> dict[str, Any]:
        base = self.mobility.update(router_prob, expert_ids)
        heads = base["route_heads"]
        newly_static: list[str] = []
        for name in LOCK_HEADS:
            value = float(heads[name])
            self.head_runs[name] = (
                self.head_runs[name] + 1
                if np.isfinite(value) and value > HEAD_GATE
                else 0
            )
            if self.head_runs[name] == PHENOTYPE_CONFIRMATIONS:
                newly_static.append(name)

        if newly_static and not self.static_latched:
            self.static_latched = True
            self.first_static_query = int(base["query"])
            self.first_static_head = "+".join(newly_static)

        mobility_warning_now = bool(
            base["warning"] and int(base["first_warning_query"]) == int(base["query"])
        )
        mobility_hard_now = bool(
            base["alarm"] and int(base["first_alarm_query"]) == int(base["query"])
        )
        static_now = bool(
            self.static_latched and self.first_static_query == int(base["query"])
        )

        warning = bool(base["warning"] or self.static_latched)
        hard = bool(base["alarm"] or self.static_latched)
        if warning and not self.warning_latched:
            self.warning_latched = True
            self.first_warning_query = int(base["query"])
            self.warning_branch = self._merge_branch(
                mobility_warning_now, static_now, self.first_static_head
            )
        if hard and not self.hard_latched:
            self.hard_latched = True
            self.first_hard_query = int(base["query"])
            self.hard_branch = self._merge_branch(
                mobility_hard_now, static_now, self.first_static_head
            )

        current_static_suspect = any(value > 0 for value in self.head_runs.values())
        if self.hard_latched:
            state = "hard_alarm"
        elif self.warning_latched:
            state = "warning"
        elif bool(base["suspect"]) or current_static_suspect:
            state = "suspect"
        else:
            state = "normal"

        return {
            "query": int(base["query"]),
            "state": state,
            "warning": self.warning_latched,
            "hard_alarm": self.hard_latched,
            "first_warning_query": self.first_warning_query,
            "first_hard_alarm_query": self.first_hard_query,
            "warning_branch": self.warning_branch,
            "hard_alarm_branch": self.hard_branch,
            "static_latched": self.static_latched,
            "first_static_query": self.first_static_query,
            "static_head": self.first_static_head,
            "head_runs": dict(self.head_runs),
            "route_heads": heads,
            "mobility_w4": float(base["mobility_w4"]),
            "mobility_threshold": float(base["threshold"]),
            "low_mobility_run": int(base["low_mobility_run"]),
        }
