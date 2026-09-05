from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/online_multihead_hub"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALARM = load_module("online_multihead_alarm.py", "online_multihead_alarm_test")
EVALUATE = load_module(
    "evaluate_online_multihead.py", "evaluate_online_multihead_test"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sealed() -> dict[str, np.ndarray]:
    with np.load(RESULTS / "sealed_online_scores.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_online_fold_is_invariant_to_heldout_future() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(400, 12, len(ALARM.FEATURES))).astype(np.float32)
    valid = np.ones((400, 12), dtype=bool)
    init_state = np.repeat(np.arange(50, dtype=np.int16), 8)

    baseline, baseline_winner, _ = ALARM.score_fold(
        features, valid, init_state, held_out=0, query_limit=52
    )
    changed = features.copy()
    changed[init_state == 0, 7:] += 1000.0
    replay, replay_winner, _ = ALARM.score_fold(
        changed, valid, init_state, held_out=0, query_limit=52
    )

    heldout = init_state == 0
    for detector in ALARM.DETECTORS:
        np.testing.assert_allclose(
            baseline[detector][heldout, :7],
            replay[detector][heldout, :7],
            equal_nan=True,
        )
    np.testing.assert_array_equal(
        baseline_winner[heldout, :7], replay_winner[heldout, :7]
    )


def test_all_eight_heldout_draws_are_excluded_from_reference() -> None:
    rng = np.random.default_rng(10)
    features = rng.normal(size=(400, 10, len(ALARM.FEATURES))).astype(np.float32)
    valid = np.ones((400, 10), dtype=bool)
    init_state = np.repeat(np.arange(50, dtype=np.int16), 8)
    baseline, _, train = ALARM.score_fold(
        features, valid, init_state, held_out=0, query_limit=52
    )
    assert train.sum() == 392

    changed = features.copy()
    changed[1] *= -100.0
    replay, _, _ = ALARM.score_fold(
        changed, valid, init_state, held_out=0, query_limit=52
    )
    for detector in ALARM.DETECTORS:
        np.testing.assert_allclose(
            baseline[detector][0], replay[detector][0], equal_nan=True
        )


def test_sealed_manifest_and_hashes() -> None:
    manifest = json.loads((RESULTS / "sealed_manifest.json").read_text())
    assert manifest["episodes"] == 14_800
    assert manifest["tasks"] == 37
    assert manifest["labels_used"] == []
    assert manifest["query_causal"] is True
    assert manifest["future_queries_used_by_online_update"] is False
    assert manifest["raw_score_start_query"] == 2
    assert manifest["warmup_query"] == 4
    artifacts = manifest["artifacts"]
    assert digest(RESULTS / "sealed_online_scores.npz") == artifacts[
        "sealed_scores_sha256"
    ]
    assert digest(RESULTS / "unlabeled_thresholds.csv") == artifacts[
        "thresholds_sha256"
    ]
    assert digest(HERE / "ONLINE_MULTIHEAD_PROTOCOL.md") == artifacts[
        "protocol_sha256"
    ]


def test_scores_start_at_q4_and_alarms_are_latched() -> None:
    sealed = load_sealed()
    detectors = sealed["detector_names"].astype(str)
    scores = sealed["scores"]
    valid = sealed["valid"].astype(bool)
    alarms = sealed["alarms"].astype(bool)
    for detector in (
        "instability",
        "lock_in",
        "flat_narrow_support",
        "feedback_decoupling",
        "dual_mean",
        "dual_max",
        "multi_max",
    ):
        index = int(np.flatnonzero(detectors == detector)[0])
        assert np.isfinite(scores[:, 4, index]).all()
        assert np.isnan(scores[:, :4, index]).all()

    assert not alarms[~valid].any()
    for episode in range(len(valid)):
        length = int(valid[episode].sum())
        for detector in range(alarms.shape[2]):
            for quantile in range(alarms.shape[3]):
                sequence = alarms[episode, :length, detector, quantile]
                if sequence.any():
                    first = int(np.flatnonzero(sequence)[0])
                    assert sequence[first:].all()


def test_unlabeled_threshold_fold_sizes_and_budget() -> None:
    thresholds = pd.read_csv(RESULTS / "unlabeled_thresholds.csv")
    assert len(thresholds) == 37 * 50 * len(ALARM.DETECTORS) * len(
        ALARM.OPERATING_QUANTILES
    )
    assert set(thresholds["unlabeled_reference_episodes"]) == {392}
    assert thresholds["threshold"].notna().all()

    sealed = load_sealed()
    names = sealed["detector_names"].astype(str)
    quantiles = sealed["quantiles"].astype(float)
    q95 = int(np.argmin(np.abs(quantiles - 0.95)))
    for detector in (
        "instability",
        "lock_in",
        "flat_narrow_support",
        "feedback_decoupling",
        "dual_mean",
        "multi_max",
    ):
        index = int(np.flatnonzero(names == detector)[0])
        rate = sealed["alarms"][:, :, index, q95].any(axis=1).mean()
        assert 0.04 <= rate <= 0.07


def test_outcome_table_is_event_based_and_counts_close() -> None:
    outcome = pd.read_csv(RESULTS / "outcome_metrics.csv")
    assert not any("auc" in column.lower() for column in outcome.columns)
    primary = outcome[
        (outcome["detector"] == "multi_max")
        & np.isclose(outcome["quantile"], 0.95)
    ].iloc[0]
    assert int(primary.tp + primary.fn) == 487
    assert int(primary.fp + primary.tn) == 14_313
    assert int(primary.tp) == 83
    assert int(primary.fp) == 773


def test_event_helpers_have_frozen_boundary_semantics() -> None:
    alarm = np.asarray(
        [[False, False, True, True], [False, False, False, False]], dtype=bool
    )
    np.testing.assert_array_equal(EVALUATE.first_alarm_query(alarm), [2, -1])
    assert EVALUATE.longest_true_run([False, True, True, False, True]) == (
        2,
        1,
        3,
    )
    np.testing.assert_array_equal(
        EVALUATE.nonzero_sign_flip_indices([1, 1, -1, -1, 1]), [2, 4]
    )
