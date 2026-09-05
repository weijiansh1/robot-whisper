from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_persistent_calibration_v5 import (  # noqa: E402
    higher_quantiles,
    trajectory_peak,
)


RESULT = ROOT / "results/persistent_calibration_v5"


def test_exact_persistent_calibration_primitives() -> None:
    score = np.asarray(
        [[np.nan, 0.2, 0.4], [np.nan, np.nan, np.nan], [0.3, 0.1, np.nan]],
        dtype=np.float32,
    )
    peak = trajectory_peak(score)
    assert np.allclose(peak[[0, 2]], [0.4, 0.3])
    assert np.isneginf(peak[1])
    quantiles = higher_quantiles(np.arange(20, dtype=np.float32))
    assert quantiles[-1] == 19


def test_candidate_is_explicitly_rejected() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    selection = summary["selection"]
    assert selection["candidate_count"] == 225
    assert selection["eligible_count"] == 0
    assert selection["selected"] is None
    assert selection["status"] == "rejected_no_eligible_candidate"
    assert selection["constraint_pass_counts"]["precision"] == 0


def test_fixed_max_reference_control_metrics() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    expected = {
        ("development_main", "persistent_max_reference_dual"): (64, 35),
        ("external_8b", "persistent_max_reference_dual"): (37, 42),
    }
    for (cohort, detector), (tp, fp) in expected.items():
        row = metrics[
            (metrics["cohort"] == cohort) & (metrics["detector"] == detector)
        ].iloc[0]
        assert (int(row["tp"]), int(row["fp"])) == (tp, fp)
