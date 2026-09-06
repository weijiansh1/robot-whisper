#!/usr/bin/env python3
"""Leave-one-suite-out folds and calibration for the v7 intrinsic routing guard."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

from evaluate_intrinsic_guard_v7 import PERIODICITY_SCALE_QUANTILE  # noqa: E402
from intrinsic_guard_monitor import (  # noqa: E402
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)
from select_operating_point import (  # noqa: E402
    ACCELERATION_QUANTILES,
    FREEZE_QUANTILES,
    PERIODICITY_QUANTILES,
    choose,
    metrics,
)


SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
PUBLISHED_QUANTILES = (0.975, 0.70, 0.65)
PEAK_STREAMS = ("freeze", "acceleration", "periodicity")


@dataclass(frozen=True)
class FoldConstants:
    """The four corpus constants plus the quantile levels that produced them."""

    freeze_threshold: float
    acceleration_threshold: float
    periodicity_threshold: float
    periodicity_scale: float
    freeze_quantile: float
    acceleration_quantile: float
    periodicity_quantile: float


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Per-trajectory `<suite>/<task>` name."""
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def suite_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Per-trajectory suite name."""
    return np.asarray([name.split("/", 1)[0] for name in task_of(cache)])


def periodicity_scale_of(periodicity: np.ndarray) -> float:
    """Robust scale that divides the recurrence score.

    Kept in float64 on purpose: the sealed v7 run passes a Python float into
    intrinsic_score_arrays, so rounding to the float32 stored in
    global_profile.npz would silently fail to reproduce the sealed alarms.
    """
    finite = np.abs(periodicity[np.isfinite(periodicity)])
    if len(finite) == 0:
        raise ValueError("no finite periodicity values in the calibration slice")
    return float(np.quantile(finite, PERIODICITY_SCALE_QUANTILE, method="linear"))


def cohort_scores(
    mobility: np.ndarray,
    acceleration: np.ndarray,
    periodicity: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    """All prefix-causal score streams for one cohort slice under one fold scale."""
    return intrinsic_score_arrays(mobility, acceleration, periodicity, scale)


def peaks_from_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-trajectory peaks of the three smoothed streams, as v7 calibrates them."""
    return {name: row_max(scores[name]) for name in PEAK_STREAMS}


def reference_peaks(
    mobility: np.ndarray,
    acceleration: np.ndarray,
    periodicity: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    return peaks_from_scores(cohort_scores(mobility, acceleration, periodicity, scale))


def constants_from_peaks(
    peaks: dict[str, np.ndarray],
    scale: float,
    freeze_quantile: float,
    acceleration_quantile: float,
    periodicity_quantile: float,
) -> FoldConstants:
    return FoldConstants(
        freeze_threshold=quantile_higher(peaks["freeze"], freeze_quantile),
        acceleration_threshold=quantile_higher(
            peaks["acceleration"], acceleration_quantile
        ),
        periodicity_threshold=quantile_higher(
            peaks["periodicity"], periodicity_quantile
        ),
        periodicity_scale=scale,
        freeze_quantile=freeze_quantile,
        acceleration_quantile=acceleration_quantile,
        periodicity_quantile=periodicity_quantile,
    )


def guard_alarms(
    scores: dict[str, np.ndarray],
    valid: np.ndarray,
    constants: FoldConstants,
) -> dict[str, np.ndarray]:
    """First-alarm query per trajectory, -1 when the rollout never alarms.

    Thresholds are calibrated on the smoothed streams but applied to the
    persistent streams, exactly as evaluate_intrinsic_guard_v7 does.
    """
    first_freeze = first_from_score(scores["freeze"], constants.freeze_threshold, valid)
    first_acceleration = first_from_score(
        scores["acceleration_persistent"], constants.acceleration_threshold, valid
    )
    first_periodicity = first_from_score(
        scores["periodicity_persistent"], constants.periodicity_threshold, valid
    )
    first_turbulence = first_and(first_acceleration, first_periodicity)
    return {
        "freeze": first_freeze,
        "acceleration": first_acceleration,
        "periodicity": first_periodicity,
        "turbulence": first_turbulence,
        "guard": first_or(first_freeze, first_turbulence),
    }


def select_fold_operating_point(
    peaks: dict[str, np.ndarray],
    development: dict[str, np.ndarray],
    valid: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Audit the 640-point quantile grid on the calibration suites only.

    Returns the full candidate table and the primary pick, or None when no
    candidate satisfies the predeclared constraints. The constraints are never
    relaxed: an infeasible fold is a result, not a failure to retry.
    """
    first_freeze = {
        quantile: first_from_score(
            development["freeze"], quantile_higher(peaks["freeze"], quantile), valid
        )
        for quantile in FREEZE_QUANTILES
    }
    first_acceleration = {
        quantile: first_from_score(
            development["acceleration_persistent"],
            quantile_higher(peaks["acceleration"], quantile),
            valid,
        )
        for quantile in ACCELERATION_QUANTILES
    }
    first_periodicity = {
        quantile: first_from_score(
            development["periodicity_persistent"],
            quantile_higher(peaks["periodicity"], quantile),
            valid,
        )
        for quantile in PERIODICITY_QUANTILES
    }
    rows: list[dict[str, float | int]] = []
    for freeze_q, freeze_alarm in first_freeze.items():
        for acceleration_q, acceleration_alarm in first_acceleration.items():
            for periodicity_q, periodicity_alarm in first_periodicity.items():
                guard = first_or(
                    freeze_alarm, first_and(acceleration_alarm, periodicity_alarm)
                )
                rows.append(
                    {
                        "freeze_quantile": freeze_q,
                        "acceleration_quantile": acceleration_q,
                        "periodicity_quantile": periodicity_q,
                        **metrics(guard, labels),
                    }
                )
    candidates = pd.DataFrame(rows).sort_values(
        ["freeze_quantile", "acceleration_quantile", "periodicity_quantile"],
        kind="stable",
    )
    try:
        primary = choose(candidates, conservative=False)
    except RuntimeError:
        primary = None
    return candidates, primary
