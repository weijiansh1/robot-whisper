#!/usr/bin/env python3
"""Strict single-rollout monitor combining route level and query derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import online_multihead_alarm as v1
from single_rollout_monitor import (
    RoutePrefixExtractor,
    finite_mean,
    scalar_midrank,
)


DETECTORS = (
    "instant_level",
    "persistent_level",
    "level_velocity",
    "level_acceleration",
    "derivative_mean",
    "derivative_fusion",
)
BRANCHES = ("persistent", "velocity", "acceleration")
PRIMARY_DETECTOR = "derivative_fusion"
WARMUP_QUERY = 7
DERIVATIVE_DECIMALS = 5


class SealedRoutePrefixExtractor:
    """Match the numeric path used by the frozen route-feature cache."""

    def __init__(self) -> None:
        self.base = RoutePrefixExtractor()
        self.action_route: list[np.ndarray] = []
        self.front_state: list[np.ndarray] = []
        self.front_action: list[np.ndarray] = []

    def update(self, router_prob: np.ndarray, expert_ids: np.ndarray) -> np.ndarray:
        values = self.base.update(router_prob, expert_ids)
        probability = v1.normalize_probability(
            np.asarray(router_prob, dtype=np.float32)
        )
        final_action = probability[:, v1.FINAL_FLOW, v1.ACTION, :]
        final_state = probability[:, v1.FINAL_FLOW, 0, :]
        front_action_mean = v1.normalize_probability(
            final_action[v1.FRONT].mean(axis=1)
        )

        # The sealed cache stored these temporal tensors as float16 before
        # computing cross-query distances. Reproduce that path at deployment.
        current_action = v1.normalize_probability(
            final_action[v1.BACK].astype(np.float16)
        ).reshape(40, v1.N_EXPERTS)
        current_state = v1.normalize_probability(
            final_state[v1.FRONT].astype(np.float16)
        )
        current_front = v1.normalize_probability(
            front_action_mean.astype(np.float16)
        )
        index = {name: position for position, name in enumerate(v1.FEATURES)}
        if self.action_route:
            values[index["route_mobility"]] = float(
                v1.hellinger(current_action, self.action_route[-1]).mean()
            )
            state_jump = float(
                v1.hellinger(current_state, self.front_state[-1]).mean()
            )
            action_jump = float(
                v1.hellinger(current_front, self.front_action[-1]).mean()
            )
            values[index["front_feedback_split"]] = state_jump - action_jump

        similarity: list[float] = []
        for lag in v1.LAGS:
            if len(self.action_route) < lag:
                similarity.append(float("nan"))
            else:
                similarity.append(
                    float(
                        v1.weighted_jaccard(
                            current_action, self.action_route[-lag]
                        ).mean()
                    )
                )
        finite_similarity = np.asarray(similarity)[np.isfinite(similarity)]
        if len(finite_similarity):
            values[index["lag_recurrence"]] = float(finite_similarity.max())
        if np.isfinite(similarity[0]) and np.isfinite(similarity[1:]).any():
            values[index["lag_periodicity"]] = float(
                np.nanmax(similarity[1:]) - similarity[0]
            )

        self.action_route.append(current_action)
        self.front_state.append(current_state)
        self.front_action.append(current_front)
        if len(self.action_route) > max(v1.LAGS):
            self.action_route.pop(0)
            self.front_state.pop(0)
            self.front_action.pop(0)
        return values


@dataclass(frozen=True)
class DerivativeTaskProfile:
    task: str
    route_reference: np.ndarray
    valid: np.ndarray
    velocity_reference: np.ndarray
    acceleration_reference: np.ndarray
    detector_names: tuple[str, ...]
    quantiles: np.ndarray
    thresholds: np.ndarray

    @classmethod
    def load(cls, path: Path, task: str) -> "DerivativeTaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.route_derivative.profile.v1":
                raise ValueError("unknown route-derivative profile schema")
            task_names = archive["task_names"].astype(str).tolist()
            if task not in task_names:
                raise KeyError(f"task not present in profile: {task}")
            task_position = task_names.index(task)
            take = archive["reference_task_index"].astype(int) == task_position
            return cls(
                task=task,
                route_reference=np.asarray(archive["route_reference"][take]),
                valid=np.asarray(archive["reference_valid"][take], dtype=bool),
                velocity_reference=np.asarray(archive["velocity_reference"][take]),
                acceleration_reference=np.asarray(
                    archive["acceleration_reference"][take]
                ),
                detector_names=tuple(archive["detector_names"].astype(str).tolist()),
                quantiles=np.asarray(archive["quantiles"], dtype=float),
                thresholds=np.asarray(archive["thresholds"][task_position], dtype=float),
            )


class RouteDerivativeMonitor:
    """One causal update per VLA forward for one live rollout."""

    def __init__(
        self,
        profile: DerivativeTaskProfile,
        quantile: float = 0.95,
        alarm_detector: str = PRIMARY_DETECTOR,
    ) -> None:
        if alarm_detector not in profile.detector_names:
            raise ValueError(f"detector not present in profile: {alarm_detector}")
        self.profile = profile
        self.alarm_detector = alarm_detector
        self.quantile_index = int(np.argmin(np.abs(profile.quantiles - quantile)))
        if not np.isclose(profile.quantiles[self.quantile_index], quantile, atol=1e-5):
            raise ValueError(f"quantile not present in profile: {quantile}")
        self.extractor = SealedRoutePrefixExtractor()
        self.instant_head_history: list[np.ndarray] = []
        self.head_history: list[np.ndarray] = []
        self.query = 0
        self.alarm_latched = False
        self.alarm_query = -1
        self.alarm_branch = "none"
        self.alarm_phenotype = "none"

    def _route_heads(self, feature: np.ndarray) -> np.ndarray:
        query = self.query
        if query >= self.profile.valid.shape[1]:
            raise IndexError("rollout exceeded the calibration profile query range")
        reached = self.profile.valid[:, query]
        feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
        instant: list[float] = []
        for members in v1.HEADS.values():
            component: list[float] = []
            for name, direction in members:
                position = feature_index[name]
                reference = direction * self.profile.route_reference[
                    reached, query, position
                ]
                component.append(
                    scalar_midrank(
                        reference, direction * float(feature[position])
                    )
                )
            component_array = np.asarray(component, dtype=np.float32)
            instant.append(
                float(component_array.mean(dtype=np.float32))
                if np.isfinite(component_array).all()
                else float("nan")
            )
        self.instant_head_history.append(np.asarray(instant, dtype=np.float32))
        if len(self.instant_head_history) < 3:
            return np.full(len(v1.HEADS), np.nan, dtype=np.float64)
        recent = np.stack(self.instant_head_history[-3:])
        if not np.isfinite(recent).all():
            return np.full(len(v1.HEADS), np.nan, dtype=np.float64)
        return recent.mean(axis=0).astype(np.float32)

    def update(
        self, router_prob: np.ndarray, expert_ids: np.ndarray
    ) -> dict[str, Any]:
        feature = self.extractor.update(router_prob, expert_ids)
        head = self._route_heads(feature)
        level = float(np.max(head)) if np.isfinite(head).all() else float("nan")
        persistent = float("nan")
        velocity = float("nan")
        acceleration = float("nan")
        if len(self.head_history) >= 1 and np.isfinite(head).all():
            previous = self.head_history[-1]
            if np.isfinite(previous).all():
                persistent = min(level, float(np.max(previous)))
                velocity_raw = float(
                    np.round(
                        np.linalg.norm(head - previous) / 2.0,
                        DERIVATIVE_DECIMALS,
                    )
                )
                reference = self.profile.velocity_reference[
                    self.profile.valid[:, self.query], self.query
                ]
                velocity = float(
                    np.float32(scalar_midrank(reference, velocity_raw))
                )
        if len(self.head_history) >= 2 and np.isfinite(head).all():
            previous = self.head_history[-1]
            previous_two = self.head_history[-2]
            if np.isfinite(previous).all() and np.isfinite(previous_two).all():
                acceleration_raw = float(
                    np.round(
                        np.linalg.norm(
                            head - 2.0 * previous + previous_two
                        )
                        / 4.0,
                        DERIVATIVE_DECIMALS,
                    )
                )
                reference = self.profile.acceleration_reference[
                    self.profile.valid[:, self.query], self.query
                ]
                acceleration = float(
                    np.float32(scalar_midrank(reference, acceleration_raw))
                )

        level_velocity = (
            float(np.float32(np.sqrt(level * velocity)))
            if np.isfinite(level) and np.isfinite(velocity)
            else float("nan")
        )
        level_acceleration = (
            float(np.float32(np.sqrt(level * acceleration)))
            if np.isfinite(level) and np.isfinite(acceleration)
            else float("nan")
        )
        derivative_mean = float(
            np.float32(finite_mean([level, velocity, acceleration]))
        )
        branches = np.asarray(
            [persistent, level_velocity, level_acceleration], dtype=np.float64
        )
        fusion = (
            float(np.float32(np.max(branches)))
            if np.isfinite(branches).all()
            else float("nan")
        )
        scores = {
            "instant_level": level,
            "persistent_level": persistent,
            "level_velocity": level_velocity,
            "level_acceleration": level_acceleration,
            "derivative_mean": derivative_mean,
            "derivative_fusion": fusion,
        }
        branch = (
            BRANCHES[int(np.argmax(branches))]
            if np.isfinite(branches).all()
            else "none"
        )
        phenotype = (
            tuple(v1.HEADS)[int(np.argmax(head))]
            if np.isfinite(head).all()
            else "none"
        )
        if self.query < WARMUP_QUERY:
            scores = {name: float("nan") for name in scores}
            branch = "none"
            phenotype = "none"

        thresholds = {
            name: float(self.profile.thresholds[position, self.quantile_index])
            for position, name in enumerate(self.profile.detector_names)
        }
        value = scores[self.alarm_detector]
        if (
            not self.alarm_latched
            and np.isfinite(value)
            and value > thresholds[self.alarm_detector]
        ):
            self.alarm_latched = True
            self.alarm_query = self.query
            self.alarm_branch = branch
            self.alarm_phenotype = phenotype
        suspect = np.isfinite(scores["instant_level"]) and (
            scores["instant_level"] > thresholds["instant_level"]
        )
        status = "alarm" if self.alarm_latched else ("suspect" if suspect else "normal")
        output = {
            "query": self.query,
            "status": status,
            "alarm": self.alarm_latched,
            "alarm_query": self.alarm_query,
            "alarm_detector": self.alarm_detector,
            "branch": self.alarm_branch if self.alarm_latched else branch,
            "phenotype": self.alarm_phenotype if self.alarm_latched else phenotype,
            "scores": scores,
            "thresholds": thresholds,
            "route_heads": {
                name: float(head[position])
                for position, name in enumerate(v1.HEADS)
            },
        }
        self.head_history.append(head)
        self.query += 1
        return output
