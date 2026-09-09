import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ranking_pair  # noqa: E402


def test_tied_ranking_agrees_with_standard_metrics():
    rng = np.random.default_rng(71)
    for n in (10, 50, 400):
        truth = rng.random(n) < 0.3
        truth[:2] = [False, True]
        score = rng.integers(0, 7, size=n).astype(float)
        auc, ap = ranking_pair(truth, score)
        assert np.isclose(auc, roc_auc_score(truth, score))
        assert np.isclose(ap, average_precision_score(truth, score))


def test_fixed_query_clock_has_no_within_task_discrimination():
    truth = np.array([False, True, True, False, False])
    auc, ap = ranking_pair(truth, np.full(5, 8.0))
    assert auc == 0.5
    assert ap == truth.mean()
