"""MoE-only protocol checks without a model or simulator."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import pdist, squareform

from moe_compute_protocol import (combine_distances, equal_parent_metrics, feature_sets, fit_ridge,
                                  noise_grid, parent_weights, port_distance, predict_ridge, select_scope)


class MoEComputeTests(unittest.TestCase):
    def test_port_capture_is_read_only_and_scoped(self):
        import torch
        from moe_compute_capture import MoEPortCapture
        from collection_routes import HB_LAYERS

        class HBMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_experts = torch.nn.Identity()

            def forward(self, value):
                return value + self.shared_experts(value)

        layers = [SimpleNamespace(mlp=HBMoE()) for _ in range(16)]
        model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(layers=layers)))
        value = torch.ones((1, 11, 1024))
        with MoEPortCapture(model) as capture:
            for _ in range(10):
                for index in HB_LAYERS:
                    self.assertTrue(torch.equal(layers[index].mlp(value), value * 2))
        arrays = capture.response()
        self.assertEqual(set(arrays), {'mechanism/input', 'mechanism/shared', 'mechanism/total'})
        self.assertEqual(arrays['mechanism/input'].shape, (8, 10, 11, 1024))
        np.testing.assert_array_equal(arrays['mechanism/input'], 1)
        np.testing.assert_array_equal(arrays['mechanism/total'], 2)
        self.assertTrue(all(not m._forward_hooks for layer in layers for m in layer.mlp.modules()))
        with self.assertRaisesRegex(RuntimeError, 'test cleanup'):
            with MoEPortCapture(model):
                raise RuntimeError('test cleanup')
        self.assertTrue(all(not m._forward_hooks for layer in layers for m in layer.mlp.modules()))

    def test_port_scope_and_direct_distance(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(5, 8, 10, 11, 3))
        distance, raw, scale = port_distance(a, 'back_last', 3)
        x = a[:, 4:, -1:, 1:]
        expected_scale = np.sqrt((x[:3] ** 2).mean(axis=(0, 2, 3, 4)))
        np.testing.assert_allclose(scale, expected_scale)
        expected = np.sqrt(np.mean(((x[1] - x[4]) / scale[:, None, None, None]) ** 2))
        self.assertAlmostEqual(distance[1, 4], expected)
        self.assertAlmostEqual(raw[1, 4], np.sqrt(np.mean((x[1] - x[4]) ** 2)))

    def test_causal_scale_and_distances(self):
        a = np.random.default_rng(2).normal(size=(6, 8, 10, 11, 2))
        d, _, scale = port_distance(a, 'back_path', 3)
        a[3:] *= 100
        changed, _, scale2 = port_distance(a, 'back_path', 3)
        np.testing.assert_array_equal(scale, scale2)
        np.testing.assert_array_equal(d[:3, :3], changed[:3, :3])

    def test_identical_compute_zero(self):
        a = np.ones((4, 8, 10, 11, 2))
        d, _, _ = port_distance(a, 'full_path', 2)
        np.testing.assert_array_equal(d, 0)
        np.testing.assert_array_equal(combine_distances(d, d), 0)

    def test_norm_rescaling_invariance(self):
        a = np.random.default_rng(8).normal(size=(4, 8, 10, 11, 2))
        d, raw, _ = port_distance(a, 'full_path', 2)
        e, raw2, _ = port_distance(a * 3, 'full_path', 2)
        np.testing.assert_allclose(d, e)
        np.testing.assert_allclose(raw * 3, raw2)

    def test_feature_dimensions_and_static(self):
        features = feature_sets(np.zeros((14, 14)))
        self.assertEqual({k: len(v) for k, v in features.items()}, dict(S=12, ST=24, D=91, DT=103))
        np.testing.assert_array_equal(features['S'], 0)
        self.assertTrue(all(np.isfinite(v).all() for v in features.values()))

    def test_cycle_has_nonzero_h1(self):
        theta = np.arange(14) * 2 * np.pi / 8
        raw = squareform(pdist(np.c_[np.cos(theta), np.sin(theta)]))
        self.assertGreater(feature_sets(raw)['ST'][12], .1)

    def test_future_changes_not_in_features(self):
        a = np.random.default_rng(2).normal(size=(30, 2))
        before = feature_sets(squareform(pdist(a[:14])))
        a[14:] += 200
        after = feature_sets(squareform(pdist(a[:14])))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_parent_weight_and_metrics(self):
        parents = np.array(['a', 'b', 'b', 'b'])
        w = parent_weights(parents)
        self.assertAlmostEqual(w[0], .5)
        self.assertAlmostEqual(w[1:].sum(), .5)
        metrics = equal_parent_metrics(np.zeros(4), np.array([2, 0, 0, 0]), parents)
        self.assertEqual(metrics['mae'], 1)
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(2))

    def test_ridge_normal_equations(self):
        x = np.random.default_rng(2).normal(size=(15, 4))
        y = x[:, 0] + 2
        w = parent_weights(['a'] * 5 + ['b'] * 10)
        model = fit_ridge(x, y, w)
        z = (x - model['mean']) / model['scale']
        residual = predict_ridge(model, x) - y
        np.testing.assert_allclose(z.T @ (w * residual) + model['coefficient'], 0, atol=1e-12)
        self.assertAlmostEqual(np.dot(w, residual), 0)

    def test_noise_grid_deterministic(self):
        original = np.ones((10, 24), np.float32)
        a, b = noise_grid(0, original), noise_grid(0, original)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[0], original)
        self.assertFalse(np.array_equal(a[1], noise_grid(1, original)[1]))

    def test_scope_rejects_wrong_axes(self):
        with self.assertRaises(ValueError):
            select_scope(np.zeros((8, 10, 11, 2)), 'back_path')


if __name__ == '__main__':
    unittest.main()
