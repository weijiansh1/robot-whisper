import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (  # noqa: E402
    binary_metrics, budget_threshold, build_scores, first_from_score,
    fit_stats, prefix_aggregate,
)


def fixture_data(n=48, q=20):
    rng = np.random.default_rng(8)
    valid = np.ones((n, q), dtype=bool)
    return {"raw": rng.uniform(0.02, 0.4, (n, q, 9)).astype(np.float32),
            "mobility": rng.uniform(0.01, 0.1, (n, q, 8)).astype(np.float32),
            "random": rng.random((n, q)).astype(np.float32), "valid": valid}


def test_every_emitted_score_is_prefix_causal():
    data = fixture_data()
    stats = fit_stats(fixture_data(64))
    full = build_scores(data, stats)
    for stop in (1, 4, 5, 6, 8, 12):
        cut = {k: v[:, :stop] for k, v in data.items()}
        short_stats = {"global": stats["global"],
                       "step": tuple(v[:stop] for v in stats["step"])}
        actual = build_scores(cut, short_stats)
        np.testing.assert_allclose(actual, full[:, :, :stop], atol=1e-6, equal_nan=True)


def test_alarm_budget_keeps_tie_groups_and_missing_trajectories():
    x = np.array([[1, 2], [1, 2], [1, 2], [1, 3], [np.nan, np.nan]], dtype=np.float32)
    valid = np.ones_like(x, dtype=bool)
    threshold = budget_threshold(x, valid, 0.4)
    assert threshold == 2
    assert (first_from_score(x, threshold, valid) >= 0).sum() == 1
    assert (first_from_score(x, budget_threshold(x, valid, 0.0), valid) >= 0).sum() == 0


def test_current_and_prefix_max_have_identical_first_alarm():
    x = fixture_data()["raw"][:, :, 0]
    valid = np.ones_like(x, dtype=bool)
    a = prefix_aggregate(x, "max")
    threshold = budget_threshold(x, valid, 0.10)
    assert threshold == budget_threshold(a, valid, 0.10)
    np.testing.assert_array_equal(first_from_score(x, threshold, valid),
                                  first_from_score(a, threshold, valid))


def test_padding_does_not_change_reference_stats_or_previous_scores():
    data = fixture_data()
    before = fit_stats(data)
    original = build_scores(data, before)
    pad = {k: np.concatenate((v, np.full((len(v), 4, *v.shape[2:]),
                                       False if v.dtype == bool else np.nan, dtype=v.dtype)), axis=1)
           for k, v in data.items()}
    after = fit_stats(pad)
    for a, b in zip(before["global"][:2], after["global"][:2]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(build_scores(pad, after)[:, :, :20], original, equal_nan=True)


def test_late_success_alarm_is_counted_and_misses_have_unit_tdet():
    out = binary_metrics(np.array([9, 3, -1]), np.array([False, True, True]),
                         np.array([10, 10, 10]))
    assert out["fp"] == 1 and out["fpr"] == 1
    assert out["tp_lead4"] == 1 and out["recall_lead4"] == 0.5
    assert np.isclose(out["t_det"], (3 / 9 + 1) / 2)
