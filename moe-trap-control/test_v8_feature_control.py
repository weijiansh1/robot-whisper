"""Geometry, causal monitor and actual gate-dispatch regression checks."""

import unittest

import numpy as np
import torch

from collection_routes import HB_LAYERS, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from v8_feature_control import (FeatureCapture, V8Monitor, make_bias, resample_path,
    normalize, flow_features, EFFECTIVE_PROBS, NATIVE_PROBS)


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(32, 12), requires_grad=False)

    def forward(self, value):
        p = torch.nn.functional.linear(value.reshape(-1, 12), self.weight).softmax(-1)
        weight, ids = p.topk(4, sorted=False)
        return ids, weight / weight.sum(-1, keepdim=True), None


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.p = self.rng.dirichlet(np.ones(32) * 3, size=(8, 10, 11)).astype(np.float32)
        self.previous = self.rng.dirichlet(np.ones(32) * 3, size=(8, 10, 11)).astype(np.float32)

    def test_direction_null_matches_each_row_budget(self):
        a = make_bias(self.p, self.previous, "combined", 1., 42)
        b = make_bias(self.p, self.previous, "random", 1., 42)
        np.testing.assert_allclose(np.sort(a, axis=-1), np.sort(b, axis=-1), rtol=0, atol=0)
        np.testing.assert_array_equal(a[:4], 0)
        np.testing.assert_array_equal(a[:, :, 0], 0)
        self.assertLessEqual(np.abs(a).max(), 1.000001)
        self.assertLessEqual(np.sqrt(np.square(a).mean(-1)).max(), .350001)
        np.testing.assert_allclose(make_bias(self.p, self.previous, "combined", .5, 42), a * .5)

    def test_resampling_keeps_endpoints_front_and_state_token(self):
        target = resample_path(self.p)
        np.testing.assert_array_equal(target[:4], self.p[:4])
        np.testing.assert_array_equal(target[:, (0, 9)], self.p[:, (0, 9)])
        np.testing.assert_array_equal(target[:, :, 0], self.p[:, :, 0])
        np.testing.assert_allclose(target.sum(-1), 1, atol=2e-7)

    def test_constant_path_and_zero_direction_are_noop(self):
        p = np.broadcast_to(self.p[:, :1], self.p.shape).copy()
        np.testing.assert_allclose(make_bias(p, p, "combined", 1, 42), 0, atol=3e-7)
        self.assertTrue(np.isfinite(flow_features(p)[0]).all())

    def test_actual_dispatch_and_hook_cleanup(self):
        torch.manual_seed(8)
        gates = [(l, Gate()) for l in HB_LAYERS]
        bias = make_bias(self.p, self.previous, "combined", 1, 42)
        observed = []
        capture = FeatureCapture(gates, bias)
        with capture:
            handles = [g.register_forward_hook(lambda _g, _i, o: observed.append((o[0].clone(), o[1].clone()))) for _, g in gates]
            for _ in range(10):
                for _, g in gates:
                    g(torch.randn(1, 11, 12))
            for h in handles:
                h.remove()
        result = capture.response()
        for i, field in enumerate((EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)):
            expected = torch.stack([r[i] for r in observed]).reshape(10, 8, 11, 4).permute(1, 0, 2, 3).numpy()
            np.testing.assert_array_equal(result[field], expected)
        p = result[EFFECTIVE_PROBS]
        selected = np.take_along_axis(p, result[EFFECTIVE_IDS_KEY].astype(int), -1)
        np.testing.assert_allclose(result[EFFECTIVE_WEIGHTS_KEY], selected / selected.sum(-1, keepdims=True), atol=1e-6)
        np.testing.assert_array_equal(p[:4], result[NATIVE_PROBS][:4])
        self.assertTrue(all(not g._forward_hooks for _, g in gates))

    def test_rejects_unscoped_bias_and_missing_calls(self):
        gates = [(l, Gate()) for l in HB_LAYERS]
        with self.assertRaises(ValueError):
            FeatureCapture(gates, np.ones_like(self.p))
        with self.assertRaises(RuntimeError):
            FeatureCapture(gates, np.zeros_like(self.p)).response()

    def test_two_consecutive_hits_after_earliest_query(self):
        m = V8Monitor()
        p = normalize(self.p)
        # Zero front flow with a moving back path forces the inversion head.
        p[:4] = p[:4, :1]
        for q in range(9):
            status = m.update(p)
            self.assertEqual(int(status["v8_head_first"][0]), -1 if q < 7 else 7)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
