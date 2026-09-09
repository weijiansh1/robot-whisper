#!/usr/bin/env python3
"""Fixed causal boundary schedules calibrated from unlabeled historical MoE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from intrinsic_guard_monitor import (
    ACCELERATION_CONFIRMATIONS, EPSILON, PERIODICITY_CONFIRMATIONS,
    GlobalIntrinsicProfile, IntrinsicGuardMonitor, intrinsic_score_arrays,
    persistent_score, row_max,
)
from unlabeled_budget_calibration import (
    DEFAULT_ALARM_BUDGET, alarm_capacity, alarms_from_scores, calibrate_peaks,
    calibrate_reference,
)


SCHEMA = "himoe.temporal_budget_guard.v1"
SCHEDULES = ("constant", "early_strict", "early_loose")
HEADS = ("freeze", "acceleration", "periodicity")
REFERENCE_TIME_SEED = 2026090601
EXTERNAL_TIME_SEED = 2026090602


@dataclass(frozen=True)
class BoundarySchedule:
    name: str
    tau_queries: float
    freeze_amplitude: float
    acceleration_amplitude: float
    periodicity_amplitude: float

    def __post_init__(self) -> None:
        if self.name not in SCHEDULES:
            raise ValueError(f"unknown boundary schedule: {self.name}")
        for name, value in asdict(self).items():
            if name != "name" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

    def offsets(self, queries: int) -> dict[str, np.ndarray]:
        if queries < 0:
            raise ValueError("query count cannot be negative")
        sign = {"constant": 0, "early_strict": 1, "early_loose": -1}[self.name]
        q = np.arange(queries, dtype=np.float32)
        g = np.float32(sign) * np.float32(self.tau_queries) / (np.float32(self.tau_queries) + q)
        return {head: np.float32(getattr(self, f"{head}_amplitude")) * g for head in HEADS}


@dataclass(frozen=True)
class TemporalProfile:
    base: GlobalIntrinsicProfile
    schedule: BoundarySchedule

    def save(self, path: Path) -> None:
        np.savez_compressed(path, schema=np.asarray(SCHEMA),
                            **{key: np.asarray(value) for key, value in
                               {**asdict(self.base), **asdict(self.schedule)}.items()})

    @classmethod
    def load(cls, path: Path) -> "TemporalProfile":
        with np.load(path, allow_pickle=False) as archive:
            expected = {"schema", *GlobalIntrinsicProfile.__dataclass_fields__,
                        *BoundarySchedule.__dataclass_fields__}
            if set(archive.files) != expected or str(archive["schema"]) != SCHEMA:
                raise ValueError("invalid temporal profile schema or unexpected metadata")
            if any(archive[name].ndim != 0 for name in expected):
                raise ValueError("temporal profile fields must be scalars")
            base = GlobalIntrinsicProfile(**{name: float(archive[name])
                                             for name in GlobalIntrinsicProfile.__dataclass_fields__})
            schedule = BoundarySchedule(**{name: str(archive[name]) if name == "name" else float(archive[name])
                                           for name in BoundarySchedule.__dataclass_fields__})
            return cls(base, schedule)


def adjusted_score_arrays(scores: dict[str, np.ndarray], schedule: BoundarySchedule) -> dict[str, np.ndarray]:
    """Time-adjust each observation before persistence, not the window minimum."""
    shape = np.asarray(scores["freeze"]).shape
    if len(shape) != 2 or any(np.asarray(scores[head]).shape != shape for head in HEADS):
        raise ValueError("smoothed head scores must align as [episode, query]")
    if schedule.name == "constant":
        return dict(scores)
    offsets = schedule.offsets(shape[1])
    adjusted = {**scores, **{head: np.asarray(scores[head], dtype=np.float32) - offsets[head][None]
                            for head in HEADS}}
    adjusted["acceleration_persistent"] = persistent_score(adjusted["acceleration"], ACCELERATION_CONFIRMATIONS)
    adjusted["periodicity_persistent"] = persistent_score(adjusted["periodicity"], PERIODICITY_CONFIRMATIONS)
    return adjusted


def calibrate_temporal_reference(
    layer_mobility: np.ndarray,
    route_acceleration: np.ndarray,
    lag_periodicity: np.ndarray,
    valid: np.ndarray,
    *,
    alarm_budget: float = DEFAULT_ALARM_BUDGET,
) -> tuple[dict[str, TemporalProfile], dict[str, Any]]:
    # Reuse the original validation and scale rule; outcomes are not accepted.
    constant = calibrate_reference(layer_mobility, route_acceleration, lag_periodicity, valid,
                                   alarm_budget=alarm_budget)
    valid = np.asarray(valid)
    mobility = np.where(valid[..., None], layer_mobility, np.nan)
    acceleration = np.where(valid, route_acceleration, np.nan)
    periodicity = np.where(valid, lag_periodicity, np.nan)
    scale = constant.profile.periodicity_scale
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, scale)
    tau = max(float(np.median(valid.sum(axis=1))) / 2, 1.0)
    amplitudes = {}
    for head in HEADS:
        peaks = row_max(np.where(valid, scores[head], np.nan))
        quartiles = np.quantile(peaks[np.isfinite(peaks)], [0.25, 0.75], method="linear")
        amplitudes[f"{head}_amplitude"] = max(float(quartiles[1] - quartiles[0]), EPSILON)
    profiles, audits = {}, {}
    for name in SCHEDULES:
        schedule = BoundarySchedule(name, tau, **amplitudes)
        adjusted = adjusted_score_arrays(scores, schedule)
        peaks = [row_max(np.where(valid, adjusted[key], np.nan)) for key in
                 (*HEADS, "acceleration_persistent", "periodicity_persistent")]
        result = calibrate_peaks(*peaks, scale, alarm_budget=alarm_budget)
        if name == "constant" and result.profile != constant.profile:
            raise RuntimeError("constant schedule changed the original automatic calibration")
        first = alarms_from_scores(adjusted, valid, result.profile)
        count = int((first["guard"] >= 0).sum())
        if count != result.audit["union_reference_alarms"]:
            raise RuntimeError(f"temporal peak calibration/replay mismatch: {name}")
        profiles[name] = TemporalProfile(result.profile, schedule)
        audits[name] = {**result.audit, "reference_replay_verified": True,
                        "reference_replay_alarms": count, "schedule": asdict(schedule),
                        "intercepts": asdict(result.profile)}
    return profiles, {"schema": SCHEMA, "alarm_budget": alarm_budget,
                      "reference_episodes": len(valid), "variants": audits,
                      "constant_matches_static_calibration": True,
                      "schedule_selection_using_outcomes": False}


class TemporalGuardMonitor(IntrinsicGuardMonitor):
    """The existing raw-feature extraction and alarm latches, with scheduled scores."""

    def __init__(self, profile: TemporalProfile) -> None:
        self.temporal_profile = profile
        super().__init__(profile.base)

    def _score_prefix(self) -> dict[str, float]:
        streams = intrinsic_score_arrays(
            np.asarray(self._mobility_history, dtype=np.float32)[None],
            np.asarray(self._acceleration_history, dtype=np.float32)[None],
            np.asarray(self._periodicity_history, dtype=np.float32)[None],
            self.profile.periodicity_scale,
        )
        adjusted = adjusted_score_arrays(streams, self.temporal_profile.schedule)
        return {name: float(adjusted[name][0, -1])
                for name in ("freeze", "acceleration_persistent", "periodicity_persistent")}


def _lengths(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid)
    if (valid.ndim != 2 or valid.dtype != np.dtype(bool)
            or np.any(valid[:, 1:] & ~valid[:, :-1]) or np.any(valid.sum(axis=1) == 0)):
        raise ValueError("validity must describe nonempty Boolean contiguous prefixes")
    return valid.sum(axis=1)


def time_priorities(episodes: int, seed: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).random(episodes)


@dataclass(frozen=True)
class TimeOnlyProfile:
    boundary_query: int
    tie_priority: float

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_query, (int, np.integer)) or self.boundary_query < 0:
            raise ValueError("time boundary must be a nonnegative integer")
        if not np.isfinite(self.tie_priority) or not 0 <= self.tie_priority < 1:
            raise ValueError("tie priority must be in [0, 1)")

    def first_alarms(self, valid: np.ndarray, priorities: np.ndarray) -> np.ndarray:
        lengths = _lengths(valid)
        priorities = np.asarray(priorities)
        if (priorities.shape != lengths.shape or not np.isfinite(priorities).all()
                or np.any((priorities < 0) | (priorities >= 1))):
            raise ValueError("priorities must be aligned finite values in [0, 1)")
        query = self.boundary_query + (priorities <= self.tie_priority).astype(np.int64)
        return np.where(query < lengths, query, -1).astype(np.int32)


def calibrate_time_only(valid: np.ndarray, priorities: np.ndarray, *,
                        alarm_budget: float = DEFAULT_ALARM_BUDGET) -> tuple[TimeOnlyProfile, dict[str, Any]]:
    lengths = _lengths(valid)
    capacity = alarm_capacity(len(lengths), alarm_budget)
    # Validate priorities without accepting any task, route or outcome information.
    TimeOnlyProfile(0, 0.5).first_alarms(valid, priorities)
    priorities = np.asarray(priorities)
    order = np.lexsort((priorities, lengths - 1))
    boundary = int(order[len(lengths) - capacity - 1])
    profile = TimeOnlyProfile(int(lengths[boundary] - 1), float(priorities[boundary]))
    count = int((profile.first_alarms(valid, priorities) >= 0).sum())
    if count > capacity:
        raise RuntimeError("time-only reference alarm budget exceeded")
    return profile, {"profile": asdict(profile), "reference_episodes": len(lengths),
                     "reference_alarm_capacity": capacity, "reference_replay_alarms": count,
                     "reference_alarm_rate": count / len(lengths),
                     "unique_priorities": len(np.unique(priorities)) == len(priorities),
                     "uses_moe": False, "uses_outcomes_or_task_identity": False,
                     "current_episode_final_length_used_for_boundary": False}
