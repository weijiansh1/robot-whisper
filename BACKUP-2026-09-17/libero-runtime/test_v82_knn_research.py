"""Meaningful checks of attribution, history windows and comparison semantics."""

import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from v82_knn_research import (distance_attribution, combine_first, first_crossing,
                              common_prefix_pairs, ReferenceScorer, standardized, auc)


class ResearchTests(unittest.TestCase):
    def test_attribution_uses_mean_distance_not_squared_distance(self):
        point = np.zeros((1, 10))
        bank = np.zeros((2, 10))
        bank[0, 0], bank[1, 8] = 3., 4.
        score, parts, signed = distance_attribution(point, bank, np.array([[0, 1]]))
        np.testing.assert_allclose(score, [3.5])
        np.testing.assert_allclose(parts, [[1.5, 0, 2, 0]])
        self.assertEqual(signed[0, 8], -2.)

    def test_duplicate_zero_distance_neighbors_are_finite(self):
        point, bank = np.zeros((1, 10)), np.zeros((2, 10))
        score, parts, residual = distance_attribution(point, bank, np.array([[0, 1]]))
        np.testing.assert_array_equal(score, [0])
        self.assertTrue(np.isfinite(parts).all() and np.isfinite(residual).all())

    def test_distance_components_sum_for_general_neighbors(self):
        rng = np.random.default_rng(51)
        points, bank = rng.normal(size=(4, 10)), rng.normal(size=(23, 10))
        neighbors = np.argsort(cdist(points, bank), axis=-1)[:, :20]
        score, parts, _ = distance_attribution(points, bank, neighbors)
        np.testing.assert_allclose(parts.sum(-1), score, rtol=1e-12)
        np.testing.assert_allclose(score, np.sort(cdist(points, bank), axis=-1)[:, :20].mean(-1))

    def test_original_neighbor_api_matches_direct_distance(self):
        rng = np.random.default_rng(8)
        points, bank = rng.normal(size=(9, 10)), rng.normal(size=(35, 10))
        distance, ids = ReferenceScorer({'success_dynamic': bank}).neighbors('success_dynamic', points)
        direct = cdist(points, bank)
        np.testing.assert_allclose(distance, np.take_along_axis(direct, ids, axis=-1), rtol=1e-10)
        np.testing.assert_allclose(distance.mean(-1), np.sort(direct)[:, :20].mean(-1), rtol=1e-10)

    def test_latched_and_does_not_require_simultaneous_crossing(self):
        result = combine_first(np.array([8, -1, 12]), np.array([11, 7, -1]))
        np.testing.assert_array_equal(result['OR'], [8, 7, 12])
        np.testing.assert_array_equal(result['AND_latched'], [11, -1, -1])

    def test_current_success_alarm_is_not_dropped_by_termination(self):
        frame = pd.DataFrame(dict(task=['t', 't'], init_state_id=[1, 1], failure=[True, False], length=[52, 12]))
        pair = common_prefix_pairs(frame, {'v82': np.array([20, 11])})
        self.assertEqual(len(pair), 1)
        self.assertFalse(pair.v82_failure_hit.iloc[0])
        self.assertTrue(pair.v82_success_hit.iloc[0])
        self.assertEqual(pair.queries.iloc[0], 12)

    def test_knn_threshold_is_strict_and_padding_not_scored(self):
        score = np.array([[np.nan, 2., 3., np.nan], [np.nan, 2., 2., np.nan]])
        np.testing.assert_array_equal(first_crossing(score, 2.), [2, -1])

    def test_10d_features_do_not_use_future_queries(self):
        rng = np.random.default_rng(81)
        cache = dict(mobility=rng.uniform(.02, .1, (2, 30, 8)).astype(np.float32),
                     acceleration=rng.uniform(.1, .2, (2, 30)).astype(np.float32),
                     periodicity=rng.uniform(-.03, .03, (2, 30)).astype(np.float32),
                     valid=np.ones((2, 30), bool))
        cache['mobility'][:, 0] = np.nan
        cache['periodicity'][:, :2] = np.nan
        profile = dict(periodicity_scale=.02, dynamic_center=np.zeros(10), dynamic_scale=np.ones(10))
        full, _ = standardized(cache, profile)
        prefix, _ = standardized({k: v[:, :14].copy() for k, v in cache.items()}, profile)
        np.testing.assert_array_equal(full[:, :14], prefix)
        self.assertTrue(np.isnan(full[:, :7]).all())

    def test_auc_ties_and_missing_scores(self):
        self.assertEqual(auc([True, False, False], [1., 1., np.nan]), .5)
        self.assertTrue(np.isnan(auc([True, True], [1., 2.])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
