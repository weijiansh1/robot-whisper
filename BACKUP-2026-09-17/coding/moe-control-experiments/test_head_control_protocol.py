"""Controlled-variable and causal-preview checks for head targeting."""

import copy
import unittest

import numpy as np
from head_control_protocol import (HEAD_TOLERANCE, V82Monitor, head_severity,
                                   normalize_scores, preview_pool, tolerant_argmin)


class HeadControlTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.history = rng.dirichlet(np.ones(32), size=(20, 8, 10, 11)).astype(np.float16)
        self.pool = rng.dirichlet(np.ones(32), size=(4, 8, 10, 11)).astype(np.float16)
        self.monitor = V82Monitor()
        for p in self.history:
            self.monitor.update(p)

    def test_preview_does_not_mutate_history(self):
        before = copy.deepcopy(self.monitor)
        result = preview_pool(self.monitor, self.pool, 'freeze', 2)
        self.assertEqual(self.monitor.v7.query, before.v7.query)
        np.testing.assert_array_equal(self.monitor.raw, before.raw)
        np.testing.assert_array_equal(self.monitor.v7._mobility_history, before.v7._mobility_history)
        self.assertEqual(result['query'], 20)
        for index in range(4):
            direct = copy.deepcopy(before).update(self.pool[index])
            self.assertEqual(result['preview_statuses'][index]['freeze_score'], direct['freeze_score'])

    def test_center_and_edge_opposite_extrema(self):
        result = preview_pool(self.monitor, self.pool, 'curvature', 1)
        score, chosen = result['centrality'], result['selected']
        self.assertEqual(chosen['center'], int(np.argmin(score)))
        self.assertEqual(chosen['edge'], int(np.argmax(score)))
        self.assertEqual(chosen['random'], 1)

    def test_head_low_and_high_control_the_same_score(self):
        for head in ('freeze', 'turbulence', 'inversion', 'curvature'):
            result = preview_pool(self.monitor, self.pool, head, 3)
            values, ids = result['target_severity'], result['selected']
            self.assertLessEqual(values[ids['head_low']], values.min() + HEAD_TOLERANCE)
            self.assertGreaterEqual(values[ids['head_high']], values.max() - HEAD_TOLERANCE)

    def test_turbulence_targets_both_current_components(self):
        normalized = np.array([[0, 3, -2, 0, 0], [0, -1, 2, 0, 0]], float)
        np.testing.assert_array_equal(head_severity(normalized, 'turbulence'), [3, 2])

    def test_only_new_head_thresholds_move_with_query(self):
        a, threshold_a, margin_a = normalize_scores(np.zeros(5), 10)
        b, threshold_b, margin_b = normalize_scores(np.zeros(5), 20)
        np.testing.assert_array_equal(threshold_a[:3], threshold_b[:3])
        np.testing.assert_allclose(threshold_b[3:] - threshold_a[3:], -.015)
        np.testing.assert_array_equal(margin_a, margin_b)
        np.testing.assert_array_equal(a[:3], b[:3])

    def test_small_score_ties_do_not_force_a_change(self):
        self.assertEqual(tolerant_argmin(np.array([.5e-6, 0, 2e-6, 3e-6])), 0)
        self.assertEqual(tolerant_argmin(np.array([2e-6, 0, 3e-6, 4e-6])), 1)

    def test_invalid_capture_fails(self):
        with self.assertRaises(ValueError):
            preview_pool(self.monitor, self.pool.astype(np.float32), 'freeze', 0)
        with self.assertRaises(ValueError):
            preview_pool(self.monitor, self.pool, 'freeze', 4)


if __name__ == '__main__':
    unittest.main()
