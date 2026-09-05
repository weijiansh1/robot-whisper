from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
BASE = WORKSPACE / "moe-v4-0904"
sys.path.insert(0, str(ROOT / "method"))

from spatial_guard_monitor import (  # noqa: E402
    SpatialGuardMonitor,
    SpatialGuardTaskProfile,
)


RESULT = ROOT / "results/spatial_guard_v6"
V4_RESULT = BASE / "results/cache_new_v4"
LAYER_CACHE = BASE / "results/layerwise_mobility/external_8b.npz"
CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
RUN_ID = "right-50x8b-20260903"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_row(row: int) -> SpatialGuardMonitor:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    v4_sealed = load_npz(V4_RESULT / "sealed_first_alarms.npz")
    cache = load_npz(LAYER_CACHE)
    task = str(cache["task_names"][int(cache["task_index"][row])])
    episode = int(cache["episode"][row])
    profile = SpatialGuardTaskProfile.load(RESULT / "deployment_profiles.npz", task)
    monitor = SpatialGuardMonitor(profile)
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


def test_profile_contains_three_task_modes() -> None:
    with np.load(RESULT / "deployment_profiles.npz", allow_pickle=False) as archive:
        tasks = archive["task_names"].astype(str)
        representations = archive["lock_representations"].astype(str)
        widths = archive["lock_widths"].astype(int)
        confirmations = archive["lock_confirmations"].astype(int)
        directions = archive["lock_directions"].astype(str)
    long = np.char.startswith(tasks, "libero_long/")
    spatial = np.char.startswith(tasks, "libero_spatial/")
    default = ~(long | spatial)
    assert set(zip(representations[long], widths[long], confirmations[long])) == {
        ("L12", 4, 8)
    }
    assert set(
        zip(representations[spatial], widths[spatial], confirmations[spatial])
    ) == {("back_median", 1, 2)}
    assert set(
        zip(representations[default], widths[default], confirmations[default])
    ) == {("all_median", 4, 4)}
    assert set(directions) == {"low"}


def test_fixed_selection_metrics_and_provenance() -> None:
    summary = json.loads((RESULT / "evaluation_summary.json").read_text())
    selected = summary["selection"]["selected"]
    assert summary["selection"]["candidate_count"] == 51_200
    assert summary["selection"]["external_outcomes_read_by_selection"] is False
    assert summary["external_cohort_pristine_holdout"] is False
    assert summary["gpu_sweep"]["visible_physical_devices"] == "6,7"
    assert (
        selected["representation"],
        selected["direction"],
        selected["width"],
        selected["confirmations"],
        selected["quantile"],
    ) == ("back_median", "low", 1, 2, 0.8)
    for name, expected in summary["artifacts"].items():
        if name == "evaluator_sha256":
            path = ROOT / "experiments/evaluate_spatial_guard_v6_gpu.py"
        elif name == "protocol_sha256":
            path = ROOT / "method/ONLINE_SPATIAL_GUARD_V6_PROTOCOL.md"
        else:
            path = RESULT / name.removesuffix("_sha256").replace(
                "sealed_first_alarms", "sealed_first_alarms"
            )
            suffix = ".json" if name == "selection_sha256" else ".npz"
            if name == "outcome_metrics_sha256":
                suffix = ".csv"
            path = path.with_suffix(suffix)
        assert sha256(path) == expected

    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    external = metrics[
        (metrics["cohort"] == "external_8b")
        & (metrics["detector"] == "spatial_guard_v6")
    ].iloc[0]
    assert (int(external["tp"]), int(external["fp"])) == (408, 45)
    assert np.isclose(external["early4_risk_recall"], 354 / 564)
    assert np.isclose(external["precision"], 408 / 453)


def test_v6_dominates_lead_and_improves_v4_operating_point() -> None:
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    external = metrics[metrics["cohort"] == "external_8b"].set_index("detector")
    v4 = external.loc["v4_dual_regime_or"]
    lead = external.loc["lead_guard_v5"]
    v6 = external.loc["spatial_guard_v6"]
    assert v6["risk_recall"] > lead["risk_recall"]
    assert v6["early4_risk_recall"] > lead["early4_risk_recall"]
    assert v6["precision"] > lead["precision"]
    assert v6["timely_fpr"] < lead["timely_fpr"]
    assert v6["early4_risk_recall"] > v4["early4_risk_recall"]
    assert v6["precision"] > v4["precision"]
    assert v6["timely_fpr"] < v4["timely_fpr"]
    assert int(v4["tp"] - v6["tp"]) == 2


def test_runtime_replay_matches_all_profile_modes() -> None:
    sealed = load_npz(RESULT / "sealed_first_alarms.npz")
    cache = load_npz(LAYER_CACHE)
    episode_tasks = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    alarmed = sealed["external_deployed_lock"] >= 0
    for prefix, representation in (
        ("libero_long/", "L12"),
        ("libero_spatial/", "back_median"),
        ("libero_goal/", "all_median"),
    ):
        row = int(
            np.flatnonzero(np.char.startswith(episode_tasks, prefix) & alarmed)[0]
        )
        assert replay_row(row).profile.lock_representation == representation
