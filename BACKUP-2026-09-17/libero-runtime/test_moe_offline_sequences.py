"""Chronology and order-control tests, independent of empirical outcomes."""

from itertools import permutations
import unittest

import numpy as np

from moe_offline_sequences import score_window, sequence_scores, first_events, auc


class SequenceTests(unittest.TestCase):
    def test_exact_permutation_mean(self):
        x = np.random.default_rng(4).normal(size=(6, 2))
        direct = []
        for perm in permutations(range(6)):
            block = x[list(perm)]
            direct.append(min(block[:3, 1].min(), -block[3:, 0].max()))
        scores = score_window(x)
        self.assertAlmostEqual(scores[3]-scores[5], np.mean(direct), places=14)

    def test_direction(self):
        x = np.array([[1., 1.]]*3+[[-1., -1.]]*3)
        a, b = score_window(x), score_window(x[::-1])
        self.assertEqual(a[3], 1.)
        self.assertEqual(a[2], -1.)
        self.assertEqual(a[3], b[4])
        self.assertEqual(a[6], -b[6])
        self.assertGreater(a[5], 0.)

    def test_static_values_have_no_order_excess(self):
        scores = score_window(np.array([[-1., 1.]]*6))
        self.assertEqual(scores[3], 1.)
        self.assertEqual(scores[5], 0.)
        self.assertEqual(scores[6], 0.)

    def test_prefix_padding_and_warmup(self):
        x = np.random.default_rng(5).normal(size=(2, 30, 2))
        x[:, :7] = np.nan
        x[1, 18:] = np.nan
        full = sequence_scores(x)
        self.assertTrue(np.isnan(full[:, :12]).all())
        self.assertTrue(np.isnan(full[1, 18:]).all())
        for stop in (8, 12, 13, 17, 24):
            np.testing.assert_array_equal(sequence_scores(x[:, :stop]), full[:, :stop])

    def test_first_events_and_ties(self):
        scores = np.full((3, 20, 1), np.nan)
        scores[0, 12:, 0] = np.log(1.2)
        scores[1, 14:, 0] = .1
        np.testing.assert_array_equal(first_events(scores)[:, 0], [12, -1, -1])
        self.assertEqual(auc([True, False], [2., 2.]), .5)
        self.assertEqual(auc([True, False], [2., 1.]), 1.)


if __name__ == '__main__':
    unittest.main()
