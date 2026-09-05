from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "method"))

from long_guard_monitor import LongGuardMonitor, LongGuardTaskProfile  # noqa: E402


RESULT = ROOT / "results/long_guard_v5"
V4_RESULT = ROOT / "results/cache_new_v4"
LAYER_CACHE = ROOT / "results/layerwise_mobility/external_8b.npz"
CACHE_ROOT = HERE.parents[1] / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
RUN_ID = "right-50x8b-20260903"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def replay_row(row: int) -> LongGuardMonitor:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    v4_sealed = load_npz(V4_RESULT / "sealed_first_alarms.npz")
    cache = load_npz(LAYER_CACHE)
    task = str(cache["task_names"][int(cache["task_index"][row])])
    episode = int(cache["episode"][row])
    profile = LongGuardTaskProfile.load(RESULT / "deployment_profiles.npz", task)
    monitor = LongGuardMonitor(profile)
    group = zarr.open_group(
        str(CACHE_ROOT / task / RUN_ID / "server/routes.zarr"), mode="r"
    )
    take = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
    for index in take:
        monitor.update(
            np.asarray(group["hb_router_probs"][index]),
            np.asarray(group["hb_expert_ids"][index]),
        )
    assert monitor.first_lock_query == int(sealed["external_deployed_lock"][row])
    assert monitor.first_instability_query == int(
        v4_sealed["external_instability"][row]
    )
    assert monitor.first_alarm_query == int(sealed["external_hybrid"][row])
    return monitor


def test_profiles_change_only_the_long_suite_lock() -> None:
    profile_path = RESULT / "deployment_profiles.npz"
    with np.load(profile_path, allow_pickle=False) as archive:
        tasks = archive["task_names"].astype(str)
        representations = archive["lock_representations"].astype(str)
        widths = archive["lock_widths"].astype(int)
        confirmations = archive["lock_confirmations"].astype(int)
    long = np.char.startswith(tasks, "libero_long/")
    assert long.sum() == 10
    assert set(representations[long]) == {"L12"}
    assert set(widths[long]) == {4}
    assert set(confirmations[long]) == {8}
    assert set(representations[~long]) == {"all_median"}
    assert set(widths[~long]) == {4}
    assert set(confirmations[~long]) == {4}


def test_runtime_replay_matches_long_and_standard_profiles() -> None:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    cache = load_npz(LAYER_CACHE)
    episode_tasks = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    long = np.char.startswith(episode_tasks, "libero_long/")
    alarmed = sealed["external_deployed_lock"] >= 0
    long_row = int(np.flatnonzero(long & alarmed)[0])
    standard_row = int(np.flatnonzero(~long & alarmed)[0])
    assert replay_row(long_row).lock_representation == "L12"
    assert replay_row(standard_row).lock_representation == "all_median"


def test_fixed_long_guard_metrics_and_provenance() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    external = metrics[
        (metrics["cohort"] == "external_8b")
        & (metrics["detector"] == "long_guard_v5")
    ].iloc[0]
    assert (int(external["tp"]), int(external["fp"])) == (391, 34)
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    assert summary["threshold_calibration_outcome_free"] is True
    assert summary["external_rule_frozen_before_outcome_join"] is True
    assert "deployment_profiles_sha256" in summary["artifacts"]
