from __future__ import annotations

import numpy as np
import pytest

from analyze_moe_consensus_audit import (
    medoid_candidate,
    pairwise_topk_jaccard,
    snapshot_bootstrap,
)


def test_topk_jaccard_is_set_invariant_and_site_averaged() -> None:
    ids = np.asarray(
        [
            [[[0, 1], [0, 1]]],
            [[[1, 0], [0, 2]]],
            [[[2, 3], [2, 3]]],
        ],
        dtype=np.int64,
    )
    distance = pairwise_topk_jaccard(ids, n_experts=4)
    assert distance[0, 1] == pytest.approx((0.0 + 2.0 / 3.0) / 2.0)
    assert distance[0, 2] == pytest.approx(1.0)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)


def test_medoid_tie_is_resolved_by_candidate_id() -> None:
    distance = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    candidate, score = medoid_candidate(distance, np.asarray([9, 4, 2]))
    assert candidate == 4
    assert score == pytest.approx(1.0)


def test_snapshot_bootstrap_uses_snapshot_vector_and_refuses_one_unit_ci() -> None:
    result = snapshot_bootstrap([1.0, 3.0], draws=1_000, confidence=0.95, seed=7)
    assert result["mean"] == pytest.approx(2.0)
    assert result["snapshots"] == 2
    assert result["available"]

    singleton = snapshot_bootstrap([0.25], draws=1_000, confidence=0.95, seed=7)
    assert singleton["mean"] == pytest.approx(0.25)
    assert not singleton["available"]
    assert singleton["lower"] is None and singleton["upper"] is None
