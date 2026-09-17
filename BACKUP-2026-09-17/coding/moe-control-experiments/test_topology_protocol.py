import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from topology_protocol import (classify_membership, delayed_distance, fit_reference, persistence,
                               region_diagnostic, route_distance, synthetic_circle as circle, topology_nulls)


class TopologyTests(unittest.TestCase):
    def test_hellinger_identity_and_bounds(self):
        p = np.zeros((2, 8, 10, 11, 32))
        p[0, ..., 0], p[1, ..., 1] = 1, 1
        for representation in ('back_path', 'full_path', 'back_last'):
            d = route_distance(p, representation)
            np.testing.assert_allclose(d, [[0, 1], [1, 0]])

    def test_delays_match_explicit_vectors(self):
        x = np.random.default_rng(3).normal(size=(15, 7))
        embedded = np.asarray([np.concatenate([x[i - k] for k in range(3)]) for i in range(2, len(x))]) / np.sqrt(3)
        np.testing.assert_allclose(delayed_distance(squareform(pdist(x))), squareform(pdist(embedded)), atol=1e-14)

    def test_constant_is_recurrent_but_has_no_hole(self):
        d = np.zeros((50, 50))
        ref = fit_reference(d, 16)
        self.assertTrue(ref['eligible'])
        self.assertEqual(region_diagnostic(d, 16, ref)['category'], 'no_confirmed_exit')
        self.assertEqual(persistence(d)['normalized_h1'], 0)

    def test_monotonic_drift_is_not_recurrent(self):
        d = delayed_distance(squareform(pdist(np.arange(52)[:, None])))
        self.assertFalse(fit_reference(d, 16)['eligible'])

    def test_cycle_and_causal_reference(self):
        d = delayed_distance(squareform(pdist(circle())))
        reference = fit_reference(d, 16)
        self.assertTrue(reference['eligible'])
        modified = d.copy()
        modified[17:, :] += 100
        modified[:, 17:] += 100
        np.fill_diagonal(modified, 0)
        self.assertEqual(fit_reference(modified, 16), reference)

    def test_geometry_is_order_invariant_without_delays(self):
        d = squareform(pdist(circle()))
        order = np.random.default_rng(8).permutation(len(d))
        a, b = persistence(d), persistence(d[np.ix_(order, order)])
        self.assertGreater(a['normalized_h1'], .3)
        self.assertAlmostEqual(a['normalized_h1'], b['normalized_h1'], places=12)

    def test_no_exit_return_and_finite_nonreturn(self):
        cases = [([True] * 10, 'no_confirmed_exit'),
                 ([True] * 3 + [False] * 4 + [True] * 3, 'exit_then_return'),
                 ([True] * 6 + [False] * 4, 'last_exit_no_observed_return'),
                 ([True] * 7 + [False] * 3, 'insufficient_followup_or_mixed')]
        for mask, label in cases:
            self.assertEqual(classify_membership(mask, np.arange(10) * 10)['category'], label)

    def test_partial_grid_and_outside_start_rejected(self):
        for mask, steps in (([False, False], [0, 10]), ([True, False], [0, 5])):
            with self.assertRaises(ValueError):
                classify_membership(mask, steps)

    def test_synthetic_escape_and_return(self):
        for end, label in ((32, 'exit_then_return'), (52, 'last_exit_no_observed_return')):
            x = circle()
            x[21:end, 0] += 10
            d = delayed_distance(squareform(pdist(x)))
            reference = fit_reference(d, 16)
            self.assertTrue(reference['eligible'])
            self.assertEqual(region_diagnostic(d, 16, reference)['category'], label)

    def test_surrogates_reproducible(self):
        d = squareform(pdist(circle(24)))
        self.assertEqual(topology_nulls(d, 5, count=3), topology_nulls(d, 5, count=3))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            route_distance(np.zeros((1, 8, 10, 11, 32)), 'back_path')
        with self.assertRaises(ValueError):
            delayed_distance(np.asarray([[0, 1], [0, 0]]))


if __name__ == '__main__':
    unittest.main()
