#!/usr/bin/env python3
"""Post-hoc shared-threshold online routing cascade."""

from __future__ import annotations

from typing import Any

import numpy as np

import online_multihead_alarm as v1
from precision_cascade_monitor import (
    HEAD_GATE,
    HEAD_PERSISTENCE,
    LOOP_LOOKBACK,
    StreamingRouteHeads,
    TaskProfile,
)
from route_derivative_monitor import SealedRoutePrefixExtractor


CALIBRATION_DETECTOR = "mobility_w4_k1"
DEFAULT_QUANTILE = 0.95
WARNING_COUNT = 2
ALARM_COUNT = 4


class SharedThresholdCascadeMonitor:
    """Use one frozen suspect threshold for every persistence level."""

    def __init__(
        self, profile: TaskProfile, quantile: float = DEFAULT_QUANTILE
    ) -> None:
        self.profile = profile
        self.quantile = quantile
        self.threshold = profile.threshold(CALIBRATION_DETECTOR, quantile)
        self.extractor = SealedRoutePrefixExtractor()
        self.head_normalizer = StreamingRouteHeads(profile)
        self.mobility: list[float] = []
        self.route_head_history: dict[str, list[float]] = {
            name: [] for name in v1.HEADS
        }
        self.low_run = 0
        self.warning_latched = False
        self.alarm_latched = False
        self.first_warning_query = -1
        self.first_alarm_query = -1
        self.query = -1

    @staticmethod
    def _mean_last(values: list[float], width: int) -> float:
        recent = np.asarray(values[-width:], dtype=np.float64)
        return (
            float(recent.mean())
            if len(recent) == width and np.isfinite(recent).all()
            else float("nan")
        )

    def _phenotype(self) -> tuple[str, dict[str, bool]]:
        instability = np.asarray(
            self.route_head_history["instability"], dtype=np.float64
        )
        prior = instability[max(0, len(instability) - 1 - LOOP_LOOKBACK) : -1]
        loop = bool(np.any(np.isfinite(prior) & (prior > HEAD_GATE)))
        static = np.maximum(
            np.asarray(self.route_head_history["lock_in"], dtype=np.float64),
            np.asarray(
                self.route_head_history["flat_narrow_support"], dtype=np.float64
            ),
        )[-HEAD_PERSISTENCE:]
        feedback = np.asarray(
            self.route_head_history["feedback_decoupling"], dtype=np.float64
        )[-HEAD_PERSISTENCE:]
        static_gate = bool(
            len(static) == HEAD_PERSISTENCE
            and np.all(np.isfinite(static) & (static > HEAD_GATE))
        )
        feedback_gate = bool(
            len(feedback) == HEAD_PERSISTENCE
            and np.all(np.isfinite(feedback) & (feedback > HEAD_GATE))
        )
        gates = {"loop": loop, "static": static_gate, "feedback": feedback_gate}
        if static_gate:
            return "static", gates
        if loop:
            return "loop", gates
        if feedback_gate:
            return "feedback", gates
        return "untyped", gates

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray
    ) -> dict[str, Any]:
        self.query += 1
        if self.query >= self.profile.route_reference.shape[1]:
            raise IndexError("query exceeds frozen task profile")
        features = self.extractor.update(router_prob, expert_ids)
        feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
        self.mobility.append(float(features[feature_index["route_mobility"]]))
        mobility_w4 = self._mean_last(self.mobility, 4)
        crossing = bool(
            np.isfinite(mobility_w4) and -mobility_w4 > self.threshold
        )
        self.low_run = self.low_run + 1 if crossing else 0

        heads = self.head_normalizer.update(self.query, features)
        for name, value in heads.items():
            self.route_head_history[name].append(value)
        phenotype, gates = self._phenotype()

        if self.low_run >= WARNING_COUNT and not self.warning_latched:
            self.warning_latched = True
            self.first_warning_query = self.query
        if self.low_run >= ALARM_COUNT and not self.alarm_latched:
            self.alarm_latched = True
            self.first_alarm_query = self.query

        if self.alarm_latched:
            state = "alarm"
        elif self.warning_latched:
            state = "warning"
        elif crossing:
            state = "suspect"
        else:
            state = "normal"
        return {
            "query": self.query,
            "state": state,
            "suspect": crossing,
            "warning": self.warning_latched,
            "alarm": self.alarm_latched,
            "first_warning_query": self.first_warning_query,
            "first_alarm_query": self.first_alarm_query,
            "mobility_w4": mobility_w4,
            "score": -mobility_w4 if np.isfinite(mobility_w4) else float("nan"),
            "threshold": self.threshold,
            "low_mobility_run": self.low_run,
            "phenotype": phenotype,
            "route_heads": heads,
            "gates": gates,
        }
