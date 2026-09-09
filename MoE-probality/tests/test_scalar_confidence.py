import inspect

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import expit

from probability.data import sha256
from probability.features import DETECTOR_PATH, detector
from probability.pure_moe import LOCAL_NAMES, WINDOW, features_from_window
from probability.scalar_confidence import (AGGREGATIONS, SCHEMA, SCORE_WINDOW, MoEScalarMonitor,
                                           MonotoneSigmoid, SuccessSupport, aggregate_scores, freeze_risk)


def test_monotone_sigmoid_recovers_probabilities_and_cannot_reverse_ranking():
    score = np.repeat(np.arange(5, dtype=float), 1000)
    rates = expit(2-0.9*np.arange(5))
    y = np.concatenate([np.r_[np.ones(round(p*1000)), np.zeros(1000-round(p*1000))] for p in rates])
    calibrator = MonotoneSigmoid.fit(score, y)
    assert calibrator.alpha == pytest.approx(.9, abs=.005)
    assert calibrator.beta == pytest.approx(2, abs=.01)
    np.testing.assert_allclose(calibrator.predict(np.arange(5)), rates, atol=.001)
    reversed_model = MonotoneSigmoid.fit(-score, y)
    assert reversed_model.alpha == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(reversed_model.predict([-100, 100]), y.mean(), atol=1e-7)
    constant_model = MonotoneSigmoid.fit(np.ones(len(y)), y)
    assert constant_model.alpha == 0
    np.testing.assert_allclose(constant_model.predict([1, 9]), y.mean())


def test_calibration_rejects_invalid_labels_and_unknown_scores():
    for scores, labels in [([0, np.nan], [0, 1]), ([0, 1], [1, 1]), ([0, 1], [0, 2])]:
        with pytest.raises(ValueError):
            MonotoneSigmoid.fit(scores, labels)
    with pytest.raises(ValueError):
        freeze_risk([np.nan])
    np.testing.assert_allclose(freeze_risk([1, .5, 0]), [0, np.log(2), -np.log(1e-8)])


def test_success_library_score_matches_exact_distance_without_fitting_on_queries():
    rng = np.random.default_rng(28)
    reference = rng.normal(size=(30, len(LOCAL_NAMES)))
    query = rng.normal(size=(7, len(LOCAL_NAMES)))
    model = SuccessSupport(neighbors=5).fit(reference)
    center, scale = reference.mean(axis=0), reference.std(axis=0)
    distances = cdist((query-center)/scale, (reference-center)/scale)
    expected = np.log1p(np.sort(distances, axis=1)[:, :5].mean(axis=1))
    np.testing.assert_allclose(model.score(query), expected, rtol=1e-12, atol=1e-12)
    before = model.score(query[:1])
    model.score(query*100)
    np.testing.assert_array_equal(model.score(query[:1]), before)
    np.testing.assert_array_equal(model.scaler.mean_, center)


def test_aggregations_are_prefix_causal_and_reset_between_episodes():
    scores = np.array([1, 10, 2, 3, 1, .5, .2, 4, 5])
    episodes = np.array([0]*7+[1]*2)
    queries = np.r_[np.arange(7, 14), [7, 8]]
    result = aggregate_scores(scores, episodes, queries)
    for length in range(1, len(scores)+1):
        prefix = aggregate_scores(scores[:length], episodes[:length], queries[:length])
        for name in result:
            np.testing.assert_array_equal(result[name][:length], prefix[name])
    assert result["support_window4"][5] < result["support_window4"][4]
    assert result["support_prefix_max"][6] == 10
    assert result["support_prefix_max"][7] == 4
    for ep in (0, 1):
        assert (np.diff(result["support_prefix_max"][episodes == ep]) >= 0).all()
    with pytest.raises(ValueError, match="first ready"):
        aggregate_scores([1, 2], [0, 0], [8, 9])
    with pytest.raises(ValueError, match="missing or reordered"):
        aggregate_scores([1, 2], [0, 0], [7, 9])


def test_streaming_primary_has_bounded_history_and_no_metadata_arguments():
    rng = np.random.default_rng(42)
    raw = rng.uniform(.5, 1.5, (20, 8, 10, 32))
    roots = detector.root_action_routes(raw)
    features = np.concatenate([features_from_window(roots[q-7:q+1]) for q in range(7, len(raw))])
    support = SuccessSupport(neighbors=3).fit(features[:5])
    calibrator = MonotoneSigmoid(1.0, 1.0, 0, 100, 50)
    bundle = dict(schema=SCHEMA, feature_names=LOCAL_NAMES, window=WINDOW, score_window=SCORE_WINDOW,
                  detector_sha256=sha256(DETECTOR_PATH), support=support,
                  calibrators={"support_" + name: calibrator for name in AGGREGATIONS})
    monitor = MoEScalarMonitor(bundle)
    aggregated = aggregate_scores(support.score(features), np.zeros(len(features)), np.arange(7, len(raw)))
    outputs = [monitor.update(chunk) for chunk in raw]
    assert all(not r["ready"] and r["success_probability"] is None for r in outputs[:7])
    for q, result in enumerate(outputs[7:]):
        for name in AGGREGATIONS:
            assert result["probabilities"][name] == pytest.approx(calibrator.predict(aggregated["support_"+name][q]))
    changed = raw.copy()
    changed[:-11] = changed[:-11, :, :, ::-1]
    other = MoEScalarMonitor(bundle)
    for chunk in changed:
        last = other.update(chunk)
    assert last["success_probability"] == pytest.approx(outputs[-1]["success_probability"], abs=1e-12)
    assert list(inspect.signature(MoEScalarMonitor).parameters) == ["bundle"]
    assert list(inspect.signature(MoEScalarMonitor.update).parameters) == ["self", "hb_router_probs"]
