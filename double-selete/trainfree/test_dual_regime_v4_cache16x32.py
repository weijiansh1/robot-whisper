from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULT = HERE / "results/online_dual_regime_v4_cache16x32"


def test_frozen_cache16x32_metrics_are_reproducible() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    expected = {
        "legacy_back_mean_q95_k2": (70, 2),
        "lock_layer_median_q75_k4": (229, 13),
        "instability_L5_q80_k8": (0, 0),
        "dual_regime_or": (229, 13),
    }
    for detector, (tp, fp) in expected.items():
        row = metrics[metrics["detector"] == detector].iloc[0]
        assert int(row["tp"]) == tp
        assert int(row["fp"]) == fp


def test_transfer_did_not_recalibrate_or_relabel_timeouts() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    assert summary["episodes"] == 2560
    assert summary["tasks"] == 5
    assert summary["v4_parameters_retuned"] is False
    assert summary["v4_thresholds_recalibrated_on_cache16x32"] is False
    assert summary["label_target"] == "raw_original_horizon_failure"
    assert summary["plus10_timeout_cleanup_available"] is False
