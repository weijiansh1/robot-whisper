import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_core import distance_scores, grouped_folds, route_features, threshold


def test_reference_calibration_and_test_keep_whole_groups_disjoint():
    groups = np.repeat(np.arange(16), 32)
    tested = []
    for ref, cal, test in grouped_folds(groups):
        assert (len(ref), len(cal), len(test)) == (256, 128, 128)
        assert not set(groups[ref]) & set(groups[cal])
        assert not set(groups[ref]) & set(groups[test])
        assert not set(groups[test]) & set(groups[cal])
        tested.extend(test)
    assert sorted(tested) == list(range(512))


def test_reference_scoring_is_independent_of_other_test_samples():
    rng = np.random.default_rng(1)
    ref, target = rng.normal(size=(20, 9)), rng.normal(size=(6, 9))
    original = distance_scores(ref, target)
    changed = distance_scores(ref, np.vstack([target[:1], target[1:] * 1e8]))
    for method in original:
        assert original[method][0] == changed[method][0]
    assert distance_scores(ref, ref[:1])["knn5"][0] > 0


def test_strict_quantile_respects_budget_with_ties():
    values = np.repeat([1, 2, 3, 4], 32)
    for budget in (0.01, 0.03, 0.05, 0.1):
        assert (values > threshold(values, budget)).sum() <= int(len(values) * budget)


def test_token_mi_distinguishes_specialization_from_uniform_routes():
    p = np.ones((2, 10, 4, 10, 32)) / 32
    p[1] = 0
    for token in range(10):
        p[1, :, :, token, token] = 1
    _, scores = route_features(p)
    assert np.allclose(scores["route_token_collapse"][0], 0)
    assert np.allclose(scores["route_token_collapse"][1], -np.log(10))
