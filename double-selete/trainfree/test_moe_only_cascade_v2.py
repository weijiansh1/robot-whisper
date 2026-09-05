from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_moe_only_cascade_v2 as evaluation  # noqa: E402
from moe_only_cascade_monitor import MoeOnlyCascadeMonitor  # noqa: E402
from precision_cascade_monitor import TaskProfile  # noqa: E402


RESULT = HERE / "results/online_moe_only_cascade_v2"
EXTERNAL = HERE / "results/online_precision_cascade_external"
CACHE_ROOT = HERE.parents[1] / "VLA_MUI_HUB/cache_new/HiMoE-VLA"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_first_consecutive_above_is_strict_and_nan_resets() -> None:
    values = np.asarray(
        [
            [0.81, 0.82, np.nan, 0.83, 0.84, 0.85, 0.86],
            [0.80, 0.81, 0.82, 0.83, 0.84, 0.10, 0.90],
            [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96],
        ],
        dtype=np.float32,
    )
    observed = evaluation.first_consecutive_above(values, 0.80, 4)
    np.testing.assert_array_equal(observed, [6, 4, 3])


def test_first_or_ignores_missing_alarms() -> None:
    left = np.asarray([-1, 8, 4, -1], dtype=np.int16)
    right = np.asarray([5, -1, 6, -1], dtype=np.int16)
    np.testing.assert_array_equal(evaluation.first_or(left, right), [5, 8, 4, -1])


def test_reported_headline_metrics_are_reproducible() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    expected = {
        ("development_main", "multihead_warning_k2_or_static_k4"): (276, 108),
        ("development_main", "multihead_hard_k4_or_static_k4"): (228, 46),
        ("external_8b", "multihead_warning_k2_or_static_k4"): (289, 120),
        ("external_8b", "multihead_hard_k4_or_static_k4"): (239, 51),
    }
    for (cohort, detector), (tp, fp) in expected.items():
        row = metrics[
            (metrics["cohort"] == cohort) & (metrics["detector"] == detector)
        ].iloc[0]
        assert int(row["tp"]) == tp
        assert int(row["fp"]) == fp


def test_runtime_replay_matches_external_static_alarm() -> None:
    first = load_npz(RESULT / "first_alarms.npz")
    sealed = load_npz(EXTERNAL / "sealed_online_scores.npz")
    target = np.flatnonzero(
        (first["external_static"] >= 0) & (first["external_mobility_hard_k4"] < 0)
    )
    assert len(target) > 0
    row = int(target[0])
    task = str(sealed["task_names"][int(sealed["task_index"][row])])
    episode = int(sealed["episode"][row])
    expected_static = int(first["external_static"][row])
    expected_warning = int(first["external_multihead_warning_k2_or_static_k4"][row])
    expected_hard = int(first["external_multihead_hard_k4_or_static_k4"][row])

    profile = TaskProfile.load(EXTERNAL / "deployment_profiles.npz", task)
    monitor = MoeOnlyCascadeMonitor(profile, quantile=0.95)
    run = CACHE_ROOT / task / "right-50x8b-20260903"
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    take = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
    for index in take:
        monitor.update(
            np.asarray(group["hb_router_probs"][index]),
            np.asarray(group["hb_expert_ids"][index]),
        )

    assert monitor.first_static_query == expected_static
    assert monitor.first_warning_query == expected_warning
    assert monitor.first_hard_query == expected_hard
    assert monitor.first_static_head in {"lock_in", "flat_narrow_support"}


def test_scope_and_decision_are_explicit() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    assert summary["runtime_only_moe"] is True
    assert summary["runtime_outcome_access"] is False
    assert summary["threshold_calibration_outcome_free"] is True
    assert summary["method_selection_outcome_free"] is False
    assert summary["criteria"] == {"hard_useful": True, "warning_useful": True}
    assert "post-hoc" in summary["status"]
