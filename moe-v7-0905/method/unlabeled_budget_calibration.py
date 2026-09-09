#!/usr/bin/env python3
"""Calibrate unchanged v7 thresholds from unlabeled MoE and an alarm workload cap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any

import numpy as np

from intrinsic_guard_monitor import (
    GlobalIntrinsicProfile,
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    row_max,
)


SCHEMA = "himoe.unlabeled_budget_calibration.v1"
DEFAULT_ALARM_BUDGET = 0.03
MIN_REFERENCE = 32
PERIODICITY_SCALE_QUANTILE = 0.75


@dataclass(frozen=True)
class CalibrationResult:
    profile: GlobalIntrinsicProfile
    audit: dict[str, Any]


def alarm_capacity(episodes: int, alarm_budget: float) -> int:
    if not math.isfinite(alarm_budget) or not 0 < alarm_budget < 1:
        raise ValueError("alarm budget must be finite and strictly between zero and one")
    if episodes < MIN_REFERENCE:
        raise ValueError(f"at least {MIN_REFERENCE} reference trajectories are required")
    return int((Decimal(str(alarm_budget)) * episodes).to_integral_value(rounding=ROUND_FLOOR))


def _sorted_finite(values: np.ndarray, name: str) -> np.ndarray:
    finite = np.sort(values[np.isfinite(values)])
    if len(finite) < MIN_REFERENCE:
        raise ValueError(f"{name} needs at least {MIN_REFERENCE} finite reference peaks")
    return finite


def _quantile_schedule(ordered: np.ndarray, episodes: int) -> np.ndarray:
    # ceil(j * (n - 1) / N), without floating-point rank-boundary errors.
    indices = (np.arange(episodes + 1, dtype=np.int64) * (len(ordered) - 1)
               + episodes - 1) // episodes
    return ordered[indices]


def calibrate_peaks(
    freeze_peak: np.ndarray,
    acceleration_peak: np.ndarray,
    periodicity_peak: np.ndarray,
    acceleration_persistent_peak: np.ndarray,
    periodicity_persistent_peak: np.ndarray,
    periodicity_scale: float,
    *,
    alarm_budget: float = DEFAULT_ALARM_BUDGET,
) -> CalibrationResult:
    """No labels or identities are accepted. NaN denotes an unavailable score stream."""
    arrays = [np.asarray(values, dtype=np.float32) for values in
              (freeze_peak, acceleration_peak, periodicity_peak,
               acceleration_persistent_peak, periodicity_persistent_peak)]
    if arrays[0].ndim != 1 or any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("all peak arrays must be aligned one-dimensional arrays")
    if any(np.isinf(values).any() for values in arrays):
        raise ValueError("reference peaks may be finite or NaN, not infinite")
    if not math.isfinite(periodicity_scale) or periodicity_scale <= 0:
        raise ValueError("periodicity scale must be finite and positive")
    freeze, acceleration, periodicity, acceleration_persistent, periodicity_persistent = arrays
    episodes = len(freeze)
    capacity = alarm_capacity(episodes, alarm_budget)
    for smooth, persistent in ((acceleration, acceleration_persistent),
                               (periodicity, periodicity_persistent)):
        present = np.isfinite(persistent)
        if np.any(present & (~np.isfinite(smooth) | (persistent > smooth))):
            raise ValueError("a persistent peak cannot exceed its smoothed peak")
    ordered_freeze = _sorted_finite(freeze, "freeze")
    acceleration_schedule = _quantile_schedule(_sorted_finite(acceleration, "acceleration"), episodes)
    periodicity_schedule = _quantile_schedule(_sorted_finite(periodicity, "periodicity"), episodes)

    # Episode e crosses both thresholds at rank j exactly when j < clearance[e].
    clearance = np.minimum(
        np.searchsorted(acceleration_schedule, acceleration_persistent, side="left"),
        np.searchsorted(periodicity_schedule, periodicity_persistent, side="left"),
    )
    clearance[~(np.isfinite(acceleration_persistent) & np.isfinite(periodicity_persistent))] = 0
    ordered_clearance = np.sort(clearance)

    def at_allowance(allowance: int) -> tuple[GlobalIntrinsicProfile, dict[str, int]]:
        freeze_index = max(len(ordered_freeze) - allowance - 1, 0)
        rank = int(ordered_clearance[episodes - allowance - 1])
        if rank > episodes:
            raise ValueError("reference persistent peaks exceed the calibration schedule")
        profile = GlobalIntrinsicProfile(
            freeze_threshold=float(ordered_freeze[freeze_index]),
            acceleration_threshold=float(acceleration_schedule[rank]),
            periodicity_threshold=float(periodicity_schedule[rank]),
            periodicity_scale=float(np.float32(periodicity_scale)),
        )
        frozen = freeze > profile.freeze_threshold
        turbulent = clearance > rank
        counts = {
            "branch_allowance": allowance,
            "freeze_reference_alarms": int(frozen.sum()),
            "turbulence_reference_alarms": int(turbulent.sum()),
            "overlap_reference_alarms": int((frozen & turbulent).sum()),
            "union_reference_alarms": int((frozen | turbulent).sum()),
            "freeze_order_index": freeze_index,
            "turbulence_quantile_numerator": rank,
        }
        return profile, counts

    low, high = 0, capacity
    probes: list[dict[str, int]] = []
    while low < high:
        middle = (low + high + 1) // 2
        _, counts = at_allowance(middle)
        probes.append(counts)
        if counts["union_reference_alarms"] <= capacity:
            low = middle
        else:
            high = middle - 1
    profile, counts = at_allowance(low)
    if counts["union_reference_alarms"] > capacity:
        raise RuntimeError("calibration failed to respect the reference alarm budget")
    return CalibrationResult(profile, {
        "schema": SCHEMA,
        "reference_episodes": episodes,
        "alarm_budget": float(alarm_budget),
        "reference_alarm_capacity": capacity,
        "reference_alarm_rate": counts["union_reference_alarms"] / episodes,
        "unused_reference_capacity": capacity - counts["union_reference_alarms"],
        "turbulence_quantile_denominator": episodes,
        "turbulence_quantile": counts["turbulence_quantile_numerator"] / episodes,
        "freeze_empirical_quantile": counts["freeze_order_index"] / (len(ordered_freeze) - 1),
        "outcome_labels_used": False,
        "task_or_suite_conditioning": False,
        "reference_success_fpr_guarantee": False,
        "out_of_sample_alarm_budget_guarantee": False,
        "objective": "largest common branch allowance satisfying the observed union cap",
        "monotone_search_probes": probes,
        **counts,
    })


def calibrate_reference(
    layer_mobility: np.ndarray,
    route_acceleration: np.ndarray,
    lag_periodicity: np.ndarray,
    valid: np.ndarray,
    *,
    alarm_budget: float = DEFAULT_ALARM_BUDGET,
) -> CalibrationResult:
    """Extract the original score peaks from all unlabeled historical trajectories."""
    mobility = np.asarray(layer_mobility, dtype=np.float32)
    acceleration = np.asarray(route_acceleration, dtype=np.float32)
    periodicity = np.asarray(lag_periodicity, dtype=np.float32)
    valid = np.asarray(valid)
    if mobility.ndim != 3 or mobility.shape[-1] != 8:
        raise ValueError("mobility must have shape [episode, query, 8]")
    shape = mobility.shape[:2]
    if acceleration.shape != shape or periodicity.shape != shape or valid.shape != shape:
        raise ValueError("feature arrays and validity must align")
    if valid.dtype != np.dtype(bool) or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must be a Boolean contiguous-prefix mask")
    alarm_capacity(len(mobility), alarm_budget)
    if (not np.isfinite(mobility[:, 1:][valid[:, 1:]]).all()
            or not np.isfinite(acceleration[valid]).all()
            or not np.isfinite(periodicity[:, 2:][valid[:, 2:]]).all()):
        raise ValueError("reference features must be finite after their required warm-up")
    if np.any(mobility[valid] < 0) or np.any(acceleration[valid] < 0):
        raise ValueError("mobility and acceleration must be nonnegative")
    mobility = np.where(valid[..., None], mobility, np.nan)
    acceleration = np.where(valid, acceleration, np.nan)
    periodicity = np.where(valid, periodicity, np.nan)
    finite_periodicity = np.abs(periodicity[np.isfinite(periodicity)])
    if not len(finite_periodicity):
        raise ValueError("reference periodicity is unavailable")
    scale = float(np.quantile(finite_periodicity, PERIODICITY_SCALE_QUANTILE, method="linear"))
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, scale)
    peaks = [row_max(np.where(valid, scores[name], np.nan)) for name in
             ("freeze", "acceleration", "periodicity", "acceleration_persistent", "periodicity_persistent")]
    result = calibrate_peaks(*peaks, scale, alarm_budget=alarm_budget)
    alarms = alarms_from_scores(scores, valid, result.profile)
    replay_count = int((alarms["guard"] >= 0).sum())
    if replay_count != result.audit["union_reference_alarms"]:
        raise RuntimeError("peak calibration and actual v7 reference replay disagree")
    result.audit["reference_replay_alarms"] = replay_count
    result.audit["reference_replay_verified"] = True
    return result


def alarms_from_scores(scores: dict[str, np.ndarray], valid: np.ndarray,
                       profile: GlobalIntrinsicProfile) -> dict[str, np.ndarray]:
    """Use the original strict comparisons and latched conjunction unchanged."""
    freeze = first_from_score(scores["freeze"], profile.freeze_threshold, valid)
    acceleration = first_from_score(scores["acceleration_persistent"], profile.acceleration_threshold, valid)
    periodicity = first_from_score(scores["periodicity_persistent"], profile.periodicity_threshold, valid)
    turbulence = first_and(acceleration, periodicity)
    return {"freeze": freeze, "acceleration": acceleration, "periodicity": periodicity,
            "turbulence": turbulence, "guard": first_or(freeze, turbulence)}
