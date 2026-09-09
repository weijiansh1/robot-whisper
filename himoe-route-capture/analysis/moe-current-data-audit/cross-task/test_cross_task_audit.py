from __future__ import annotations

import numpy as np

from k32_route_medoid_audit import stable_medoid, topk_jaccard_distance
from task_seed_heldout_increment_audit import pool_standardize, to_seed_folds


def test_expert_id_jaccard_is_order_invariant_and_medoid_ties_use_seed() -> None:
    ids = np.asarray(
        [
            [[[0, 1, 2, 3], [4, 5, 6, 7]]],
            [[[3, 2, 1, 0], [7, 6, 5, 4]]],
            [[[8, 9, 10, 11], [12, 13, 14, 15]]],
        ]
    )
    distance = topk_jaccard_distance(ids)
    np.testing.assert_allclose(
        distance,
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    selected, _score = stable_medoid(distance, np.asarray([1002, 1001, 1000]))
    assert selected == 1


def test_modulo_seed_folds_and_pool_standardization_preserve_axes() -> None:
    values = np.arange(2 * 32 * 3, dtype=np.float64).reshape(2, 32, 3)
    folded = to_seed_folds(values)
    assert folded.shape == (2, 4, 8, 3)
    np.testing.assert_array_equal(folded[:, 2], values[:, 2::4])
    standardized = pool_standardize(folded)
    np.testing.assert_allclose(standardized.mean(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=2), 1.0, atol=1e-12)

