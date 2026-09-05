from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precision_cascade_monitor as calibration  # noqa: E402
import shared_threshold_cascade_monitor as cascade  # noqa: E402


SEALED = HERE / "results/online_precision_cascade_external"
SHARED = HERE / "results/online_precision_cascade_shared"
CACHE_ROOT = HERE.parents[1] / "VLA_MUI_HUB/cache_new/HiMoE-VLA"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_runtime_replay_matches_shared_threshold_first_alarms() -> None:
    sealed = load_npz(SEALED / "sealed_online_scores.npz")
    shared = load_npz(SHARED / "first_alarms.npz")
    levels = shared["levels"].astype(str).tolist()
    quantiles = shared["quantiles"].astype(float)
    warning_position = levels.index("mobility_w4_k2")
    alarm_position = levels.index("mobility_w4_k4")
    q95_position = int(np.flatnonzero(np.isclose(quantiles, 0.95))[0])

    expected_alarm = shared["external_first"][:, alarm_position, q95_position]
    row = int(np.flatnonzero(expected_alarm >= 0)[0])
    task_index = int(sealed["task_index"][row])
    task = str(sealed["task_names"][task_index])
    episode = int(sealed["episode"][row])
    expected_warning = int(
        shared["external_first"][row, warning_position, q95_position]
    )

    profile = calibration.TaskProfile.load(
        SEALED / "deployment_profiles.npz", task
    )
    runtime = cascade.SharedThresholdCascadeMonitor(profile, quantile=0.95)
    assert runtime.threshold == profile.threshold("mobility_w4_k1", 0.95)

    run = CACHE_ROOT / task / "right-50x8b-20260903"
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    take = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
    observed_warning = -1
    observed_alarm = -1
    for index in take:
        result = runtime.update(
            np.asarray(group["hb_router_probs"][index]),
            np.asarray(group["hb_expert_ids"][index]),
        )
        if result["warning"] and observed_warning < 0:
            observed_warning = int(result["query"])
        if result["alarm"] and observed_alarm < 0:
            observed_alarm = int(result["query"])

    assert observed_warning == expected_warning
    assert observed_alarm == int(expected_alarm[row])
    assert observed_alarm >= observed_warning


def test_shared_threshold_metrics_are_reproducible() -> None:
    metrics = pd.read_csv(SHARED / "outcome_metrics.csv")
    expected = {
        ("development_main", "mobility_w4_k2", 0.95): (234, 88),
        ("development_main", "mobility_w4_k4", 0.95): (177, 21),
        ("external_8b_posthoc", "mobility_w4_k2", 0.95): (252, 104),
        ("external_8b_posthoc", "mobility_w4_k4", 0.95): (189, 29),
        ("external_8b_posthoc", "mobility_w4_k4", 0.975): (125, 10),
    }
    for (cohort, level, quantile), (tp, fp) in expected.items():
        row = metrics[
            (metrics["cohort"] == cohort)
            & (metrics["level"] == level)
            & np.isclose(metrics["quantile"], quantile)
        ].iloc[0]
        assert int(row["tp"]) == tp
        assert int(row["fp"]) == fp


def test_reported_validity_scope_is_explicit() -> None:
    summary = json.loads((SHARED / "evaluation_summary.json").read_text())
    assert summary["runtime_outcome_free"] is True
    assert summary["threshold_calibration_outcome_free"] is True
    assert summary["method_selection_outcome_free"] is False
    assert "post-hoc" in summary["status"]
