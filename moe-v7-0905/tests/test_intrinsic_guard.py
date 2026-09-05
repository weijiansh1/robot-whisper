from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

from intrinsic_guard_monitor import (  # noqa: E402
    SCHEMA,
    GlobalIntrinsicProfile,
    IntrinsicGuardMonitor,
    hellinger,
    intrinsic_score_arrays,
    normalize_probability,
    weighted_jaccard,
)


RESULT = BUNDLE / "results/intrinsic_guard_v7"


def random_routes(queries: int = 18) -> np.ndarray:
    rng = np.random.default_rng(20260905)
    raw = rng.lognormal(
        mean=-1.0, sigma=1.2, size=(queries, 8, 10, 11, 32)
    ).astype(np.float32)
    return normalize_probability(raw)


def replay(raw: np.ndarray, profile: GlobalIntrinsicProfile) -> tuple[dict[str, np.ndarray], IntrinsicGuardMonitor]:
    monitor = IntrinsicGuardMonitor(profile)
    records = [monitor.update(query) for query in raw]
    streams = {
        "mobility": np.stack([row["layer_mobility"] for row in records]),
        "acceleration": np.asarray([row["route_acceleration"] for row in records]),
        "periodicity": np.asarray([row["lag_periodicity"] for row in records]),
        "freeze": np.asarray([row["freeze_score"] for row in records]),
        "acceleration_persistent": np.asarray(
            [row["acceleration_score"] for row in records]
        ),
        "periodicity_persistent": np.asarray(
            [row["periodicity_score"] for row in records]
        ),
    }
    return streams, monitor


def test_probability_distances_have_expected_identities() -> None:
    left = np.asarray([[2.0, 1.0, -1.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32)
    normalized = normalize_probability(left)
    np.testing.assert_allclose(normalized.sum(axis=-1), 1.0)
    assert (normalized >= 0.0).all()
    np.testing.assert_allclose(hellinger(left, left), 0.0, atol=1e-7)
    np.testing.assert_allclose(weighted_jaccard(left, left), 1.0, atol=1e-7)
    np.testing.assert_allclose(hellinger(left, right), hellinger(right, left))
    np.testing.assert_allclose(weighted_jaccard(left, right), weighted_jaccard(right, left))


def test_stream_scores_equal_batch_scores_and_first_possible_queries() -> None:
    raw = random_routes()
    profile = GlobalIntrinsicProfile(
        freeze_threshold=-1e6,
        acceleration_threshold=-1e6,
        periodicity_threshold=-1e6,
        periodicity_scale=0.02,
    )
    stream, monitor = replay(raw, profile)
    batch = intrinsic_score_arrays(
        stream["mobility"][None],
        stream["acceleration"][None],
        stream["periodicity"][None],
        profile.periodicity_scale,
    )
    for name in ("freeze", "acceleration_persistent", "periodicity_persistent"):
        np.testing.assert_allclose(stream[name], batch[name][0], equal_nan=True, atol=1e-7)
    assert monitor.first_freeze_query == 6
    assert monitor.first_acceleration_query == 9
    assert monitor.first_periodicity_query == 10
    assert monitor.first_turbulence_query == 10
    assert monitor.first_alarm_query == 6


def test_future_queries_cannot_change_an_existing_prefix() -> None:
    raw = random_routes(20)
    changed = raw.copy()
    changed[11:] = np.roll(changed[11:], shift=5, axis=-1)
    profile = GlobalIntrinsicProfile(0.2, 0.1, 0.3, 0.02)
    original, _ = replay(raw, profile)
    counterfactual, _ = replay(changed, profile)
    for name in original:
        np.testing.assert_allclose(
            original[name][:11], counterfactual[name][:11], equal_nan=True, atol=0.0, rtol=0.0
        )


def test_global_profile_rejects_task_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid_profile.npz"
    np.savez(
        path,
        schema=np.asarray(SCHEMA),
        freeze_threshold=np.asarray(0.2),
        acceleration_threshold=np.asarray(0.1),
        periodicity_threshold=np.asarray(0.3),
        periodicity_scale=np.asarray(0.02),
        task_names=np.asarray(["forbidden"]),
    )
    try:
        GlobalIntrinsicProfile.load(path)
    except ValueError as error:
        assert "task metadata" in str(error)
    else:
        raise AssertionError("task-indexed profiles must be rejected")


def test_sealed_artifacts_are_task_free_and_reproduce_primary_counts() -> None:
    with np.load(RESULT / "global_profile.npz", allow_pickle=False) as profile:
        assert "task_names" not in profile.files
        assert "task_index" not in profile.files
        assert float(profile["freeze_quantile"]) == 0.975
        assert float(profile["acceleration_quantile"]) == 0.70
        assert float(profile["periodicity_quantile"]) == 0.65
    manifest = json.loads((RESULT / "sealed_manifest.json").read_text())
    assert manifest["runtime_task_identity"] is False
    assert manifest["task_prototypes"] is False
    assert manifest["learned_weights"] is False
    metrics = pd.read_csv(RESULT / "outcome_metrics.csv")
    external = metrics[
        (metrics["cohort"] == "external_8b")
        & (metrics["detector"] == "intrinsic_guard_v7")
    ].iloc[0]
    assert (int(external["tp"]), int(external["fp"])) == (439, 80)


def test_raw_two_gpu_replay_artifact_passed() -> None:
    audit = json.loads((RESULT / "raw_causal_gpu_verification.json").read_text())
    assert audit["all_passed"] is True
    assert audit["devices"] == [0, 1]
    assert audit["physical_devices_requested"] == [6, 7]
    assert audit["exact_first_alarm_episodes"] == audit["episodes"] == 24
    assert audit["prefix_invariant_episodes"] == audit["episodes"]
