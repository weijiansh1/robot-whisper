"""Check exact retrieval, held-out calibration, and streaming behavior."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from knn import ALPHAS, BASE_METHODS, METHODS, ReferenceScorer, calibrate_constant, reference_pairs, score_streams
from knn_monitor import BoundaryKNNMonitor


def profile():
    rng = np.random.default_rng(113)
    success = rng.normal(size=(60, 10))
    mixture = rng.normal(size=(70, 10))
    components = np.linalg.qr(rng.normal(size=(10, 2)))[0].T
    return {"dynamic_center": np.zeros(10), "dynamic_scale": np.ones(10),
        "routing_center": np.zeros(256), "routing_scale": np.ones(256),
        "success_dynamic": success, "mixture_dynamic": mixture,
        "success_routing": rng.normal(size=(60, 256)), "mixture_failure": np.arange(70) % 3 == 0,
        "pca_mean": np.zeros(10), "pca_components": components, "success_pca2": success @ components.T,
        "periodicity_scale": np.asarray(.5), "checkpoint": np.asarray("example"),
        "methods": np.asarray(METHODS), "alphas": np.asarray(ALPHAS),
        "thresholds": np.full((2, len(ALPHAS), len(METHODS)), -1.)}


def test_exact_distance_and_vote_against_direct_sort():
    p = profile()
    rng = np.random.default_rng(55)
    d, r = rng.normal(size=(4, 10)), rng.normal(size=(4, 256))
    actual = ReferenceScorer(p).current(d, r)
    distances = np.sort(cdist(d, p["success_dynamic"]), axis=1)
    for k in (1, 5, 20):
        np.testing.assert_allclose(actual[f"dyn_success_knn_k{k}"], distances[:, :k].mean(1), rtol=1e-6)
    neighbors = np.argsort(cdist(d, p["mixture_dynamic"]), axis=1)[:, :20]
    np.testing.assert_allclose(actual["dyn_vote_knn_k20"], p["mixture_failure"][neighbors].mean(1), rtol=1e-6)
    expected = np.sort(cdist(d @ p["pca_components"].T, p["success_pca2"]), axis=1)[:, :20].mean(1)
    np.testing.assert_allclose(actual["dyn_pca2_success_knn_k20"], expected, rtol=1e-6)


def test_reference_pairs_cannot_include_nonreference_episodes():
    values = np.ones((8, 25, 10))
    values[1, 9] = np.nan
    pairs = reference_pairs(values, np.asarray([1, 3, 5]), cap=20, seed=1)
    assert set(pairs[:, 0]) <= {1, 3, 5}
    assert (pairs[:, 1] >= 7).all() and np.isfinite(values[pairs[:, 0], pairs[:, 1]]).all()
    assert np.unique(pairs, axis=0).shape[0] == 20
    assert max(np.unique(pairs[:, 0], return_counts=True)[1]) <= 8


def test_stream_scores_are_prefix_causal():
    rng = np.random.default_rng(52)
    d = rng.normal(size=(2, 20, 10))
    d[:, :7] = np.nan
    r, direct = rng.normal(size=(2, 20, 256)), rng.normal(size=(2, 20, 6))
    full = score_streams(profile(), d, r, direct)
    for stop in (8, 9, 13):
        prefix = score_streams(profile(), d[:, :stop], r[:, :stop], direct[:, :stop])
        np.testing.assert_allclose(prefix, full[:, :, :stop], equal_nan=True)
    assert np.isnan(full[:, :, :7]).all()
    assert np.isnan(full[METHODS.index("dyn_success_knn_k20_persist3"), :, :9]).all()


def test_failed_calibration_episodes_do_not_change_thresholds():
    rng = np.random.default_rng(5)
    scores = rng.random((len(METHODS), 50, 12)).astype(np.float32)
    labels = np.asarray([0] * 45 + [1] * 5)
    frame = pd.DataFrame({"task": ["a"] * 50, "init_state_id": np.arange(50) // 5})
    first, _, _ = calibrate_constant(scores, labels, frame, scores[:, :3])
    scores[:, labels == 1] += 1000
    second, _, _ = calibrate_constant(scores, labels, frame, scores[:, :3])
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("method,earliest", [("dyn_success_knn_k20", 7), ("dyn_success_knn_k20_persist3", 9)])
def test_monitor_warmup_latching_and_checkpoint(tmp_path, method, earliest):
    path = tmp_path / "profile.npz"
    np.savez(path, **profile())
    with pytest.raises(ValueError, match="checkpoint"):
        BoundaryKNNMonitor(path, "wrong")
    monitor = BoundaryKNNMonitor(path, "example", method=method)
    raw = np.full((8, 10, 11, 32), 1 / 32, np.float32)
    for q in range(12):
        result = monitor.update(raw)
        assert result["first_alarm_query"] == (-1 if q < earliest else earliest)
    monitor.reset()
    assert monitor.query == 0 and monitor.first_alarm_query == -1
