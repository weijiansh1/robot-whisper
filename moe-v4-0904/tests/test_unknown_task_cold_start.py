from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

from dual_regime_monitor import ACTION, FINAL_FLOW, normalize  # noqa: E402
from unknown_task_monitor import ColdStartReference, UnknownTaskMonitor  # noqa: E402


RESULT = BUNDLE / "results/unknown_task/cold_start_v1"


def test_fixed_unknown_task_metrics() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    rows = {
        (row["cohort"], row["detector"]): row for row in summary["metrics"]
    }
    development = rows[("development_main", "dual")]
    external = rows[("external_8b", "dual")]
    assert (development["tp"], development["fp"]) == (285, 46)
    assert (external["tp"], external["fp"]) == (324, 87)
    assert summary["method"]["runtime_task_id_used"] is False
    assert summary["method"]["runtime_inputs"] == ["hb_router_probs"]


def test_monitor_calibrates_from_four_queries_without_task_id() -> None:
    reference = ColdStartReference.load(RESULT / "cold_start_reference.npz")
    monitor = UnknownTaskMonitor(reference)
    rng = np.random.default_rng(20260904)
    snapshots = [rng.random((8, 10, 11, 32), dtype=np.float32) for _ in range(4)]

    states = [monitor.update(snapshot) for snapshot in snapshots]
    assert [state["state"] for state in states[:3]] == ["calibrating"] * 3
    assert states[3]["calibrated"] is True
    assert np.isfinite(states[3]["lock_cutoff"])
    assert np.isfinite(states[3]["instability_cutoff"])
    assert len(states[3]["nearest_reference_tasks"]) == 12

    routes = [normalize(snapshot[:, FINAL_FLOW, ACTION, :]) for snapshot in snapshots]
    mobility = []
    for previous, current in zip(routes, routes[1:]):
        affinity = np.sqrt(previous * current).sum(axis=-1)
        mobility.append(np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0)).mean(axis=-1))
    early_scale = np.mean([np.median(value) for value in mobility])
    ratio = np.quantile(reference.lock_ratios, 0.10, method="higher")
    assert np.isclose(monitor.lock_cutoff, early_scale * ratio)
