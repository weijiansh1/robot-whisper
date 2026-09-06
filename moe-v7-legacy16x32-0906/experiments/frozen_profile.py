#!/usr/bin/env python3
"""Reload the published v7 operating point without recalibrating anything.

The sealed evaluator derived ``periodicity_scale`` as

    float(np.quantile(np.abs(finite reference periodicity), 0.75, method="linear"))

and passed that Python float into ``intrinsic_score_arrays``; it then stored a
float32 copy in ``global_profile.npz``.  Reading the float32 back is not
guaranteed to give the value the sealed run actually used, so this module
recomputes the scale from the same pooled 16,000-trajectory reference corpus in
float64 and *asserts* that it agrees with the stored float32 to the bit.  The
three thresholds are handled the same way.  Nothing here is re-selected: the
quantile levels, the reference corpus and the Boolean mechanism all come from
the frozen v7 bundle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
V7 = WORKSPACE / "moe-v7-0905"
sys.path.insert(0, str(V7 / "method"))
sys.path.insert(0, str(V7 / "experiments"))

from intrinsic_guard_monitor import (  # noqa: E402
    GlobalIntrinsicProfile,
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)
import evaluate_intrinsic_guard_v7 as sealed  # noqa: E402


PROFILE_PATH = V7 / "results/intrinsic_guard_v7/global_profile.npz"
SEALED_ALARMS = V7 / "results/intrinsic_guard_v7/sealed_first_alarms.npz"


@dataclass(frozen=True)
class FrozenPoint:
    freeze_threshold: float
    acceleration_threshold: float
    periodicity_threshold: float
    periodicity_scale: float
    stored: GlobalIntrinsicProfile

    def as_dict(self) -> dict[str, float]:
        return {
            "freeze_threshold": self.freeze_threshold,
            "acceleration_threshold": self.acceleration_threshold,
            "periodicity_threshold": self.periodicity_threshold,
            "periodicity_scale": self.periodicity_scale,
        }


def reference_corpus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_layer = sealed.load_npz(sealed.MAIN_LAYER)
    extra_layer = sealed.load_npz(sealed.EXTRA_LAYER)
    main_feature = sealed.load_npz(sealed.MAIN_FEATURE)
    extra_feature = sealed.load_npz(sealed.EXTRA_FEATURE)
    sealed.assert_aligned(main_layer, main_feature, "main")
    sealed.assert_aligned(extra_layer, extra_feature, "extra")
    return sealed.combine_reference(main_layer, extra_layer, main_feature, extra_feature)


def load_frozen_point() -> FrozenPoint:
    """Recompute the sealed scalars and check them against the published file."""
    mobility, acceleration, periodicity = reference_corpus()
    finite = np.abs(periodicity[np.isfinite(periodicity)])
    scale = float(np.quantile(finite, sealed.PERIODICITY_SCALE_QUANTILE, method="linear"))
    scale64 = float(
        np.quantile(
            finite.astype(np.float64), sealed.PERIODICITY_SCALE_QUANTILE, method="linear"
        )
    )
    if scale != scale64:
        raise AssertionError(
            "the float32 and float64 reference periodicity quantiles disagree: "
            f"{scale!r} != {scale64!r}"
        )
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, scale)
    point = FrozenPoint(
        freeze_threshold=quantile_higher(row_max(scores["freeze"]), sealed.FREEZE_QUANTILE),
        acceleration_threshold=quantile_higher(
            row_max(scores["acceleration"]), sealed.ACCELERATION_QUANTILE
        ),
        periodicity_threshold=quantile_higher(
            row_max(scores["periodicity"]), sealed.PERIODICITY_QUANTILE
        ),
        periodicity_scale=scale,
        stored=GlobalIntrinsicProfile.load(PROFILE_PATH),
    )
    stored = point.stored
    mismatch = {
        name: (recomputed, published)
        for name, recomputed, published in (
            ("freeze_threshold", point.freeze_threshold, stored.freeze_threshold),
            (
                "acceleration_threshold",
                point.acceleration_threshold,
                stored.acceleration_threshold,
            ),
            (
                "periodicity_threshold",
                point.periodicity_threshold,
                stored.periodicity_threshold,
            ),
            ("periodicity_scale", point.periodicity_scale, stored.periodicity_scale),
        )
        if recomputed != published
    }
    if mismatch:
        raise AssertionError(f"recomputed profile differs from the sealed file: {mismatch}")
    return point


def guard_alarms(
    point: FrozenPoint,
    mobility: np.ndarray,
    acceleration: np.ndarray,
    periodicity: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the frozen rule.  ``freeze OR (acceleration AND periodicity)``."""
    scores = intrinsic_score_arrays(
        mobility, acceleration, periodicity, point.periodicity_scale
    )
    valid = np.asarray(valid, dtype=bool)
    freeze = first_from_score(scores["freeze"], point.freeze_threshold, valid)
    accel = first_from_score(
        scores["acceleration_persistent"], point.acceleration_threshold, valid
    )
    period = first_from_score(
        scores["periodicity_persistent"], point.periodicity_threshold, valid
    )
    turbulence = first_and(accel, period)
    return {
        "freeze": freeze,
        "acceleration": accel,
        "periodicity": period,
        "turbulence": turbulence,
        "guard": first_or(freeze, turbulence),
        "_scores": scores,
    }
