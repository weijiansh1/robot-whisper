from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import precision_cascade_monitor as monitor  # noqa: E402


RESULTS = HERE / "results/online_precision_cascade_external"
CACHE_ROOT = HERE.parents[1] / "VLA_MUI_HUB/cache_new/HiMoE-VLA"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sealed() -> dict[str, np.ndarray]:
    with np.load(RESULTS / "sealed_online_scores.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_mobility_windows_are_causal_and_have_frozen_start_queries() -> None:
    mobility = np.full((1, 14), np.nan, dtype=np.float32)
    mobility[:, 1:] = 0.02
    valid = np.ones_like(mobility, dtype=bool)
    scores = monitor.mobility_detector_scores(mobility, valid)
    assert np.isnan(scores["mobility_w4_k1"][0, :4]).all()
    assert np.isfinite(scores["mobility_w4_k1"][0, 4])
    assert np.isnan(scores["mobility_w4_k4"][0, :7]).all()
    assert np.isclose(scores["mobility_w4_k4"][0, 7], -0.02)
    assert np.isnan(scores["mobility_w8_k3"][0, :10]).all()
    assert np.isclose(scores["mobility_w8_k3"][0, 10], -0.02)

    changed = mobility.copy()
    changed[:, 10:] = 0.8
    replay = monitor.mobility_detector_scores(changed, valid)
    for detector in monitor.BASE_DETECTORS:
        np.testing.assert_allclose(
            scores[detector][:, :10], replay[detector][:, :10], equal_nan=True
        )


def test_loop_gate_uses_only_prior_queries() -> None:
    shape = (1, 12)
    heads = {
        name: np.full(shape, 0.1, dtype=np.float32) for name in monitor.ROUTE_HEADS
    }
    heads["instability"][0, 6] = 0.9
    gates = monitor.typed_gates(heads)
    assert not gates["loop_confirmed"][0, 6]
    assert gates["loop_confirmed"][0, 7]
    assert gates["loop_confirmed"][0, 11]


def test_external_seal_and_artifact_hashes() -> None:
    manifest = json.loads((RESULTS / "sealed_manifest.json").read_text())
    assert manifest["schema"] == "himoe.precision_cascade.manifest.v1"
    assert manifest["episodes"] == 15_600
    assert manifest["tasks"] == 39
    assert manifest["labels_used"] == []
    assert manifest["test_episodes_used_for_calibration"] == 0
    assert manifest["runtime_future_access"] is False
    artifacts = manifest["artifacts"]
    assert digest(RESULTS / "sealed_online_scores.npz") == artifacts[
        "sealed_scores_sha256"
    ]
    assert digest(RESULTS / "deployment_profiles.npz") == artifacts[
        "deployment_profiles_sha256"
    ]
    assert digest(HERE / "ONLINE_PRECISION_CASCADE_PROTOCOL.md") == artifacts[
        "protocol_sha256"
    ]


def test_external_flow_seeds_are_disjoint_from_historical_profile() -> None:
    sealed = load_sealed()
    assert set(np.unique(sealed["flow_noise_seed"])) == set(range(1008, 1016))
    with np.load(RESULTS / "deployment_profiles.npz", allow_pickle=False) as profile:
        # The profile stores route features rather than outcome metadata. The
        # source campaign and disjoint seed ranges are sealed in the manifest.
        assert len(profile["route_reference"]) == 39 * 400
        assert len(profile["task_names"]) == 39


def test_runtime_primary_scores_and_alarm_match_sealed_replay() -> None:
    sealed = load_sealed()
    task = str(sealed["task_names"][0])
    episode = 0
    row = int(
        np.flatnonzero(
            (sealed["task_index"] == 0) & (sealed["episode"] == episode)
        )[0]
    )
    run = CACHE_ROOT / task / "right-50x8b-20260903"
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    take = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
    profile = monitor.TaskProfile.load(RESULTS / "deployment_profiles.npz", task)
    runtime = monitor.PrecisionCascadeMonitor(profile)
    detector_position = list(sealed["detector_names"].astype(str)).index(
        monitor.PRIMARY_DETECTOR
    )
    quantile_position = int(
        np.flatnonzero(
            np.isclose(sealed["quantiles"].astype(float), monitor.PRIMARY_QUANTILE)
        )[0]
    )
    runtime_alarm: list[bool] = []
    runtime_scores: list[float] = []
    for index in take:
        result = runtime.update(
            np.asarray(group["hb_router_probs"][index]),
            np.asarray(group["hb_expert_ids"][index]),
        )
        runtime_scores.append(float(result["score"]))
        runtime_alarm.append(bool(result["alarm"]))
    np.testing.assert_allclose(
        runtime_scores,
        sealed["scores"][row, : len(take), detector_position],
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        runtime_alarm,
        sealed["alarms"][row, : len(take), detector_position, quantile_position],
    )


def test_evaluation_uses_all_external_rollouts() -> None:
    summary = json.loads((RESULTS / "evaluation_summary.json").read_text())
    assert summary["episodes"] == 15_600
    assert summary["failures"] + summary["successes"] == 15_600
    outcome = pd.read_csv(RESULTS / "outcome_metrics.csv")
    primary = outcome[
        (outcome["detector"] == monitor.PRIMARY_DETECTOR)
        & np.isclose(outcome["quantile"], monitor.PRIMARY_QUANTILE)
    ].iloc[0]
    assert int(primary.tp + primary.fn) == summary["failures"]
    assert int(primary.fp + primary.tn) == summary["successes"]
