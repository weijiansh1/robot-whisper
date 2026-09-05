from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dual_regime_monitor import (  # noqa: E402
    CausalFloat32Mean,
    DualRegimeMonitor,
    DualRegimeTaskProfile,
)


RESULT = HERE / "results/online_dual_regime_v4"
LAYER_CACHE = HERE / "results/layerwise_mobility/external_8b.npz"
CACHE_ROOT = HERE.parents[1] / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
RUN_ID = "right-50x8b-20260903"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_causal_mean_requires_a_complete_window() -> None:
    mean = CausalFloat32Mean(4)
    observed = [mean.update(value) for value in (np.nan, 1.0, 2.0, 3.0, 4.0)]
    assert np.isnan(observed[:4]).all()
    assert observed[4] == 2.5


def replay_row(row: int) -> DualRegimeMonitor:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    cache = load_npz(LAYER_CACHE)
    task = str(cache["task_names"][int(cache["task_index"][row])])
    episode = int(cache["episode"][row])
    profile = DualRegimeTaskProfile.load(RESULT / "deployment_profiles.npz", task)
    monitor = DualRegimeMonitor(profile)
    group = zarr.open_group(
        str(CACHE_ROOT / task / RUN_ID / "server/routes.zarr"), mode="r"
    )
    take = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
    for index in take:
        monitor.update(
            np.asarray(group["hb_router_probs"][index]),
            np.asarray(group["hb_expert_ids"][index]),
        )
    assert monitor.first_lock_query == int(sealed["external_lock"][row])
    assert monitor.first_instability_query == int(sealed["external_instability"][row])
    assert monitor.first_alarm_query == int(sealed["external_dual"][row])
    return monitor


def test_runtime_replay_matches_lock_first_alarm() -> None:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    lock = sealed["external_lock"]
    instability = sealed["external_instability"]
    candidates = np.flatnonzero(
        (lock >= 0) & ((instability < 0) | (lock < instability))
    )
    assert len(candidates) > 0
    monitor = replay_row(int(candidates[0]))
    assert monitor.first_alarm_branch == "lock"


def test_runtime_replay_matches_instability_first_alarm() -> None:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    lock = sealed["external_lock"]
    instability = sealed["external_instability"]
    candidates = np.flatnonzero(
        (instability >= 0) & ((lock < 0) | (instability < lock))
    )
    assert len(candidates) > 0
    monitor = replay_row(int(candidates[0]))
    assert monitor.first_alarm_branch == "instability"


def test_headline_metrics_are_reproducible() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    expected = {
        ("development_main", "dual_regime_or"): (372, 67),
        ("external_8b", "dual_regime_or"): (410, 81),
        ("external_8b", "lock_layer_median_q75_k4"): (382, 76),
    }
    for (cohort, detector), (tp, fp) in expected.items():
        row = metrics[
            (metrics["cohort"] == cohort) & (metrics["detector"] == detector)
        ].iloc[0]
        assert int(row["tp"]) == tp
        assert int(row["fp"]) == fp


def test_validity_scope_and_threshold_provenance_are_explicit() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    assert summary["runtime_moe_only"] is True
    assert summary["runtime_outcome_access"] is False
    assert summary["threshold_calibration_outcome_free"] is True
    assert summary["method_selection_used_development_outcomes"] is True
    thresholds = pd.read_csv(RESULT / "external_outcome_blind_thresholds.csv")
    assert len(thresholds) == 78
    assert not thresholds["outcomes_used"].any()
