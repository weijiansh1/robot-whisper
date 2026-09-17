import unittest

import numpy as np

from coverage_protocol import (GENERATORS, gate_bias, pool_metrics, request_noise, specifications)


class CoverageProtocolTests(unittest.TestCase):
    def spec(self, generator="state_gate", pool=0, candidate=0, kind="candidate"):
        return dict(kind=kind, generator=generator, pool=pool, candidate=candidate)

    def test_balanced_deterministic_scopes(self):
        bias = gate_bias(self.spec())
        np.testing.assert_array_equal(bias, gate_bias(self.spec()))
        np.testing.assert_array_equal(bias.astype(float).sum(axis=-1), 0)
        mask = np.any(bias != 0, axis=-1)
        np.testing.assert_array_equal(np.count_nonzero(bias[mask] > 0, axis=-1), 16)
        np.testing.assert_array_equal(np.count_nonzero(bias[mask] < 0, axis=-1), 16)
        self.assertEqual(np.count_nonzero(np.any(bias != 0, axis=-1)), 12)
        self.assertFalse(bias[4:].any())
        self.assertFalse(bias[:, :7].any())
        self.assertFalse(bias[:, :, 1:].any())

    def test_matched_energy_and_direction(self):
        state = gate_bias(self.spec())
        action = gate_bias(self.spec("action_gate"))
        matched = gate_bias(self.spec("action_gate_l2"))
        self.assertAlmostEqual(np.linalg.norm(state.astype(float)), np.linalg.norm(matched.astype(float)), places=6)
        np.testing.assert_allclose(action / np.sqrt(10), matched, rtol=1e-7, atol=1e-9)
        self.assertEqual(np.count_nonzero(np.any(action != 0, axis=-1)), 120)

    def test_noise_namespaces_and_source_retention(self):
        parent = dict(benchmark="plus", task=0)
        source = np.zeros((10, 32), np.float32)
        noise = request_noise(parent, source, self.spec("noise"))
        np.testing.assert_array_equal(noise, request_noise(parent, source, self.spec("noise")))
        for other in (self.spec("noise", pool=1), self.spec("noise", kind="target"), self.spec("noise", candidate=1)):
            self.assertFalse(np.array_equal(noise, request_noise(parent, source, other)))
        self.assertFalse(np.array_equal(noise, request_noise(dict(benchmark="pro", task=0), source, self.spec("noise"))))
        np.testing.assert_array_equal(source, request_noise(parent, source, self.spec()))

    def test_balanced_schedule(self):
        specs = specifications()
        self.assertEqual(len(specs), 32)
        self.assertEqual(len({tuple(sorted(x.items())) for x in specs}), 32)
        for generator in GENERATORS:
            self.assertEqual(sum(s["generator"] == generator for s in specs), 8)

    def test_default_only_pool(self):
        default = np.zeros((10, 7))
        result = pool_metrics(default, np.zeros((4, 10, 7)), np.ones((8, 10, 7)), np.ones(7))
        for field in ("coverage_gain", "mean_radius_rms", "participation_rank", "best_positive_cosine"):
            self.assertEqual(result[field], 0)

    def test_exact_target_coverage_and_gripper_exclusion(self):
        default = np.zeros((10, 7))
        candidates = np.ones((4, 10, 7))
        targets = np.ones((8, 10, 7))
        targets[:, :, 6] = 99
        result = pool_metrics(default, candidates, targets, np.ones(7))
        self.assertEqual(result["coverage_gain"], 1)
        self.assertAlmostEqual(result["participation_rank"], 1)
        self.assertAlmostEqual(result["best_positive_cosine"], 1)

    def test_orthogonal_directions_and_zero_target(self):
        candidates = np.zeros((4, 10, 7))
        for i in range(4):
            candidates[i, :, i] = 1
        result = pool_metrics(np.zeros((10, 7)), candidates, np.zeros((1, 10, 7)), np.ones(7))
        self.assertAlmostEqual(result["participation_rank"], 4)
        self.assertIsNone(result["coverage_gain"])
        self.assertEqual(result["valid_targets"], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pool_metrics(np.zeros((9, 7)), np.zeros((4, 10, 7)), np.ones((8, 10, 7)), np.ones(7))
        with self.assertRaises(ValueError):
            gate_bias(self.spec("unregistered"))


if __name__ == "__main__":
    unittest.main()
