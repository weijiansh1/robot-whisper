#!/usr/bin/env python3
"""Stateful train-free monitor for one live HiMoE-VLA rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import online_closed_loop_alarm_v2 as v2
import online_multihead_alarm as v1


TYPED_STATES = ("loop_state", "static_state", "feedback_state")
DETECTORS = (
    *TYPED_STATES,
    "instant_route",
    "persistent_route",
    "multi_state",
    "single_rollout_multi",
)
PRIMARY_DETECTOR = "single_rollout_multi"
WARMUP_QUERY = 7
MIN_REFERENCE = 32


def scalar_midrank(reference: np.ndarray, value: float) -> float:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    reference = reference[np.isfinite(reference)]
    if len(reference) < MIN_REFERENCE or not np.isfinite(value):
        return float("nan")
    left = np.searchsorted(reference, value, side="left")
    right = np.searchsorted(reference, value, side="right")
    return float((left + right) / (2.0 * len(reference)))


def finite_mean(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else float("nan")


def finite_max(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else float("nan")


def geometric(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(np.sqrt(np.clip(left * right, 0.0, 1.0)))


class RoutePrefixExtractor:
    """Convert one raw routing snapshot at a time to the twelve v1 features."""

    def __init__(self) -> None:
        self.action_route: list[np.ndarray] = []
        self.front_state: list[np.ndarray] = []
        self.front_action: list[np.ndarray] = []

    def update(self, router_prob: np.ndarray, expert_ids: np.ndarray) -> np.ndarray:
        probability = v1.normalize_probability(np.asarray(router_prob, dtype=np.float32))
        selected = np.asarray(expert_ids)
        expected_router = (8, 10, 11, v1.N_EXPERTS)
        expected_ids = (8, 10, 11, 4)
        if probability.shape != expected_router:
            raise ValueError(f"expected router snapshot {expected_router}, got {probability.shape}")
        if selected.shape != expected_ids:
            raise ValueError(f"expected expert IDs {expected_ids}, got {selected.shape}")

        final_action = probability[:, v1.FINAL_FLOW, v1.ACTION, :]
        final_state = probability[:, v1.FINAL_FLOW, 0, :]
        front_action_mean = v1.normalize_probability(final_action[v1.FRONT].mean(axis=1))
        back_action_mean = v1.normalize_probability(final_action[v1.BACK].mean(axis=1))

        values = np.full(len(v1.FEATURES), np.nan, dtype=np.float32)
        index = {name: position for position, name in enumerate(v1.FEATURES)}
        back_tokens = final_action[v1.BACK]
        entropy = -(back_tokens * np.log(np.maximum(back_tokens, 1e-12))).sum(axis=-1)
        values[index["gate_entropy"]] = float(entropy.mean() / np.log(v1.N_EXPERTS))
        ordered = np.partition(back_tokens, -2, axis=-1)
        values[index["top12_margin"]] = float(
            (ordered[..., -1] - ordered[..., -2]).mean()
        )
        values[index["token_disagreement"]] = float(
            v1.hellinger(back_tokens, back_action_mean[:, None, :]).mean()
        )
        values[index["front_state_action_gap"]] = float(
            v1.hellinger(final_state[v1.FRONT], front_action_mean).mean()
        )
        values[index["layer5_state_action_gap"]] = float(
            v1.hellinger(final_state[3], front_action_mean[3])
        )

        ids = selected[v1.BACK, v1.FINAL_FLOW, v1.ACTION, :]
        values[index["top4_union"]] = float(v1.top4_union_fraction(ids[None])[0])
        action_flow = probability[v1.BACK, :, v1.ACTION, :]
        late = action_flow[:, v1.LATE_FLOW_START :]
        values[index["late_flow_volatility"]] = float(
            (1.0 - v1.weighted_jaccard(late[:, 1:], late[:, :-1])).mean()
        )
        root = np.sqrt(action_flow)
        acceleration = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
        values[index["route_acceleration"]] = float(
            np.linalg.norm(acceleration, axis=-1).mean() / np.sqrt(2.0)
        )

        current_action = v1.normalize_probability(back_tokens).reshape(40, v1.N_EXPERTS)
        current_state = v1.normalize_probability(final_state[v1.FRONT])
        if self.action_route:
            values[index["route_mobility"]] = float(
                v1.hellinger(current_action, self.action_route[-1]).mean()
            )
            state_jump = float(v1.hellinger(current_state, self.front_state[-1]).mean())
            action_jump = float(
                v1.hellinger(front_action_mean, self.front_action[-1]).mean()
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
        self.front_action.append(front_action_mean)
        if len(self.action_route) > max(v1.LAGS):
            self.action_route.pop(0)
            self.front_state.pop(0)
            self.front_action.pop(0)
        return values


class PhysicalPrefixExtractor:
    """Extract current physical signals without reading future state or sim_state."""

    def __init__(self) -> None:
        self.state: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def update(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"expected state [8], got {state.shape}")
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"expected action chunk [chunk,7], got {actions.shape}")
        self.state.append(state)
        self.actions.append(actions)
        feature = v2.extract_physical_features(
            np.stack(self.state, axis=0), np.stack(self.actions, axis=0)
        )
        return feature[-1]


class StreamingStateMachine:
    """Finite-history phenotype state machine shared by replay and deployment."""

    def __init__(self) -> None:
        self.route_history: list[dict[str, float]] = []
        self.loop_physical: list[float] = []
        self.static_physical: list[float] = []
        self.feedback_physical: list[float] = []
        self.phenotype_history: list[float] = []

    def update(
        self, route_heads: dict[str, float], physical_heads: dict[str, float]
    ) -> tuple[dict[str, float], str]:
        static_route = finite_max(
            [route_heads["lock_in"], route_heads["flat_narrow_support"]]
        )
        loop_physical = geometric(
            route_heads["instability"], physical_heads["physical_cycle"]
        )
        static_physical = geometric(static_route, physical_heads["motion_stall"])
        feedback_physical = geometric(
            route_heads["feedback_decoupling"], physical_heads["motion_stall"]
        )
        self.loop_physical.append(loop_physical)
        self.static_physical.append(static_physical)
        self.feedback_physical.append(feedback_physical)

        loop_switch = float("nan")
        if len(self.route_history) >= 3 and np.isfinite(static_route):
            past = [row["instability"] for row in self.route_history[-3:]]
            if np.isfinite(past).all():
                loop_switch = geometric(max(past), static_route)
        loop_hold = finite_mean(self.loop_physical[-3:]) if len(self.loop_physical) >= 3 else float("nan")
        loop_state = finite_max([loop_switch, loop_hold])
        static_state = (
            finite_mean(self.static_physical[-3:])
            if len(self.static_physical) >= 3
            else float("nan")
        )
        feedback_state = (
            finite_mean(self.feedback_physical[-3:])
            if len(self.feedback_physical) >= 3
            else float("nan")
        )
        typed = {
            "loop_state": loop_state,
            "static_state": static_state,
            "feedback_state": feedback_state,
        }
        phenotype_state = finite_max(list(typed.values()))
        instant_route = finite_max(list(route_heads.values()))
        previous_route = (
            finite_max(list(self.route_history[-1].values()))
            if self.route_history
            else float("nan")
        )
        persistent_route = (
            min(previous_route, instant_route)
            if np.isfinite(previous_route) and np.isfinite(instant_route)
            else float("nan")
        )
        previous_phenotype = (
            self.phenotype_history[-1] if self.phenotype_history else float("nan")
        )
        single_rollout = (
            min(previous_phenotype, phenotype_state)
            if np.isfinite(previous_phenotype) and np.isfinite(phenotype_state)
            else float("nan")
        )
        scores = {
            **typed,
            "instant_route": instant_route,
            "persistent_route": persistent_route,
            "multi_state": phenotype_state,
            "single_rollout_multi": single_rollout,
        }
        typed_values = np.asarray([typed[name] for name in TYPED_STATES])
        winner = (
            TYPED_STATES[int(np.nanargmax(typed_values))]
            if np.isfinite(typed_values).any()
            else "none"
        )
        self.route_history.append(dict(route_heads))
        self.phenotype_history.append(phenotype_state)
        return scores, winner


@dataclass(frozen=True)
class TaskProfile:
    task: str
    route_reference: np.ndarray
    physical_reference: np.ndarray
    valid: np.ndarray
    detector_names: tuple[str, ...]
    quantiles: np.ndarray
    thresholds: np.ndarray

    @classmethod
    def load(cls, path: Path, task: str) -> "TaskProfile":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.single_rollout.profile.v1":
                raise ValueError("unknown streaming profile schema")
            task_names = archive["task_names"].astype(str).tolist()
            if task not in task_names:
                raise KeyError(f"task not present in profile: {task}")
            task_position = task_names.index(task)
            reference_task = archive["reference_task_index"].astype(int)
            take = reference_task == task_position
            return cls(
                task=task,
                route_reference=np.asarray(archive["route_reference"][take]),
                physical_reference=np.asarray(archive["physical_reference"][take]),
                valid=np.asarray(archive["reference_valid"][take], dtype=bool),
                detector_names=tuple(archive["detector_names"].astype(str).tolist()),
                quantiles=np.asarray(archive["quantiles"], dtype=float),
                thresholds=np.asarray(archive["thresholds"][task_position], dtype=float),
            )


class SingleRolloutMonitor:
    """Public streaming API: one update per VLA query, one rollout per instance."""

    def __init__(
        self,
        profile: TaskProfile,
        quantile: float = 0.95,
        alarm_detector: str = PRIMARY_DETECTOR,
    ) -> None:
        self.profile = profile
        if alarm_detector not in profile.detector_names:
            raise ValueError(f"detector not present in profile: {alarm_detector}")
        self.alarm_detector = alarm_detector
        self.quantile_index = int(np.argmin(np.abs(profile.quantiles - quantile)))
        if not np.isclose(profile.quantiles[self.quantile_index], quantile, atol=1e-5):
            raise ValueError(f"quantile not present in profile: {quantile}")
        self.route_extractor = RoutePrefixExtractor()
        self.physical_extractor = PhysicalPrefixExtractor()
        self.state_machine = StreamingStateMachine()
        self.instant_route_history: list[dict[str, float]] = []
        self.query = 0
        self.alarm_latched = False
        self.alarm_query = -1
        self.alarm_phenotype = "none"

    def _rank(self, feature: np.ndarray, physical: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        query = self.query
        if query >= self.profile.valid.shape[1]:
            raise IndexError("rollout exceeded the calibration profile query range")
        reached = self.profile.valid[:, query]
        route_index = {name: i for i, name in enumerate(v1.FEATURES)}
        instant: dict[str, float] = {}
        for head, members in v1.HEADS.items():
            components: list[float] = []
            for name, direction in members:
                reference = direction * self.profile.route_reference[
                    reached, query, route_index[name]
                ]
                components.append(
                    scalar_midrank(reference, direction * float(feature[route_index[name]]))
                )
            instant[head] = finite_mean(components)
        self.instant_route_history.append(instant)
        route_heads = {
            head: (
                finite_mean([row[head] for row in self.instant_route_history[-3:]])
                if len(self.instant_route_history) >= 3
                else float("nan")
            )
            for head in v1.HEADS
        }

        physical_index = {name: i for i, name in enumerate(v2.PHYSICAL_FEATURES)}
        directed: dict[str, float] = {}
        specifications = {
            "low_eef_motion": ("eef_motion_w3", -1),
            "high_prior_command": ("prior_command_w3", 1),
            "high_eef_reversal": ("eef_reversal_w3", 1),
            "high_gripper_flip": ("gripper_flip_w4", 1),
            "high_state_return": ("state_return_w4", 1),
        }
        for label, (name, direction) in specifications.items():
            position = physical_index[name]
            reference = direction * self.profile.physical_reference[
                reached, query, position
            ]
            directed[label] = scalar_midrank(
                reference, direction * float(physical[position])
            )
        physical_heads = {
            "motion_stall": finite_mean(
                [directed["low_eef_motion"], directed["high_prior_command"]]
            ),
            "physical_cycle": finite_mean(
                [
                    directed["high_eef_reversal"],
                    directed["high_gripper_flip"],
                    directed["high_state_return"],
                ]
            ),
        }
        return route_heads, physical_heads

    def update(
        self,
        router_prob: np.ndarray,
        expert_ids: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, Any]:
        route_feature = self.route_extractor.update(router_prob, expert_ids)
        physical_feature = self.physical_extractor.update(state, actions)
        route_heads, physical_heads = self._rank(route_feature, physical_feature)
        scores, winner = self.state_machine.update(route_heads, physical_heads)
        if self.query < WARMUP_QUERY:
            scores = {name: float("nan") for name in scores}
            winner = "none"
        threshold = {
            name: float(self.profile.thresholds[position, self.quantile_index])
            for position, name in enumerate(self.profile.detector_names)
        }
        primary = scores[self.alarm_detector]
        if (
            not self.alarm_latched
            and np.isfinite(primary)
            and primary > threshold[self.alarm_detector]
        ):
            self.alarm_latched = True
            self.alarm_query = self.query
            if self.alarm_detector == "persistent_route":
                values = np.asarray([route_heads[name] for name in v1.HEADS])
                self.alarm_phenotype = (
                    tuple(v1.HEADS)[int(np.nanargmax(values))]
                    if np.isfinite(values).any()
                    else "none"
                )
            else:
                self.alarm_phenotype = winner
        suspect_detector = (
            "instant_route"
            if self.alarm_detector == "persistent_route"
            else "multi_state"
        )
        suspect = np.isfinite(scores[suspect_detector]) and (
            scores[suspect_detector] > threshold[suspect_detector]
        )
        status = "alarm" if self.alarm_latched else ("suspect" if suspect else "normal")
        result = {
            "query": self.query,
            "status": status,
            "alarm": self.alarm_latched,
            "alarm_query": self.alarm_query,
            "alarm_detector": self.alarm_detector,
            "phenotype": self.alarm_phenotype if self.alarm_latched else winner,
            "scores": scores,
            "thresholds": threshold,
            "route_heads": route_heads,
            "physical_heads": physical_heads,
        }
        self.query += 1
        return result
