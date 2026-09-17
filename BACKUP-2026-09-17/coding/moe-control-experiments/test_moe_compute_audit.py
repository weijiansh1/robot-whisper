"""Independent formula and endpoint boundary checks for MoE analysis."""

import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from audit_moe_compute_experiment import classify, direct_distance, graph_reference, independent_features
from moe_compute_protocol import feature_sets, port_distance
from topology_protocol import classify_membership, fit_reference


class MoEComputeAuditTests(unittest.TestCase):
    def test_independent_features(self):
        rng = np.random.default_rng(51)
        theta = np.arange(14) * 2 * np.pi / 8
        for points in (np.zeros((14, 2)), np.c_[np.cos(theta), np.sin(theta)], rng.normal(size=(14, 7))):
            raw = squareform(pdist(points))
            a, b = feature_sets(raw), independent_features(raw)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)

    def test_independent_graph(self):
        rng = np.random.default_rng(19)
        for points in (np.zeros((14, 2)), rng.normal(size=(14, 3))):
            raw = squareform(pdist(points))
            for scale in (.75, 1., 1.25):
                a, b = fit_reference(raw, 13, scale), graph_reference(raw, 13, scale)
                for key in ('eligible', 'region_indices', 'labels'):
                    self.assertEqual(a[key], b[key])
                self.assertAlmostEqual(a['epsilon'], b['epsilon'])

    def test_membership_categories_exhaustive(self):
        for mask in range(128):
            inside = np.array([True] + [bool(mask & (1 << i)) for i in range(7)])
            expected = classify_membership(inside, np.arange(8) * 10)['category']
            self.assertEqual(classify(inside), expected)

    def test_direct_distance_independent_axes(self):
        values = np.random.default_rng(1).normal(size=(5, 8, 10, 11, 4))
        for scope in ('back_path', 'full_path', 'back_last'):
            d, _, scales = port_distance(values, scope, 3)
            for a in range(5):
                for b in range(5):
                    self.assertAlmostEqual(d[a, b], direct_distance(values[a], values[b], scope, scales))


if __name__ == '__main__':
    unittest.main()
