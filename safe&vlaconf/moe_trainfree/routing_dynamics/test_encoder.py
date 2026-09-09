"""Mechanism and causality checks for routing dynamics encoding."""

import unittest

import numpy as np

from encoder import (EncoderConfig, FEATURE_NAMES, ReadoutEncoder, RoutingDynamicsEncoder,
                     encode_readouts, query_readout)


def readout(mobility=0.08, acceleration=0.01, path=0.1, diversity=0.003):
    return np.r_[np.full(8, mobility), np.full(8, path), acceleration, np.full(8, diversity)]


class EncoderTests(unittest.TestCase):
    def test_constant_and_zero_signals_do_not_generate_anomaly(self):
        rng = np.random.default_rng(100)
        constants = [readout(), np.zeros(25), *np.exp(rng.normal(-3, 2, (1000, 25)))]
        for x in constants:
            encoder = ReadoutEncoder()
            for _ in range(16):
                result = encoder.update(x)
            for name in ("decoupling", "persistent_decoupling", "decoupling_occupancy", "routing_recovery", "layer_agreement"):
                self.assertEqual(result.as_dict()[name], 0)

    def test_joint_feature_requires_both_changes(self):
        for changed, expected in ((readout(.04, .01), False), (readout(.08, .02), False),
                                  (readout(.04, .02), True), (readout(.12, .02), False)):
            e = ReadoutEncoder()
            for q in range(12):
                result = e.update(readout() if q < 5 else changed)
            self.assertEqual(result.as_dict()["decoupling"] > 0, expected)

    def test_persistence_and_relief_are_finite_memory(self):
        e = ReadoutEncoder()
        signals = []
        for q in range(40):
            x = readout(.03, .03, .2) if 5 <= q < 18 else readout()
            signals.append(e.update(x).as_dict())
        self.assertGreater(signals[15]["persistent_decoupling"], 0)
        self.assertTrue(any(r["routing_recovery"] > 0 for r in signals[18:30]))
        for name in ("decoupling", "persistent_decoupling", "decoupling_occupancy", "routing_recovery"):
            self.assertEqual(signals[-1][name], 0)
        self.assertLessEqual(len(e._history), 8)
        self.assertLessEqual(len(e._recent), 3)
        self.assertEqual(len(e._reference_rows), 0)

    def test_short_excursion_does_not_latch(self):
        e = ReadoutEncoder()
        for q in range(30):
            result = e.update(readout(.02, .04) if q == 8 else readout())
        self.assertEqual(result.as_dict()["recent_decoupling"], 0)

    def test_stream_batch_and_every_prefix_agree(self):
        rng = np.random.default_rng(72)
        x = np.stack([readout() * np.exp(rng.normal(0, .25, 25)) for _ in range(25)])[None]
        batch, layers = encode_readouts(x)
        online = ReadoutEncoder()
        for q in range(x.shape[1]):
            result = online.update(x[0, q])
            prefix, prefix_layers = encode_readouts(x[:, :q + 1])
            np.testing.assert_allclose(result.values, batch[0, q], equal_nan=True, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result.per_layer, layers[0, q], equal_nan=True, atol=1e-12, rtol=0)
            np.testing.assert_allclose(prefix, batch[:, :q + 1], equal_nan=True, atol=1e-12, rtol=0)
            np.testing.assert_allclose(prefix_layers, layers[:, :q + 1], equal_nan=True, atol=1e-12, rtol=0)

    def test_padding_is_not_a_feature(self):
        x = np.tile(readout(), (2, 24, 1))
        valid = np.arange(24)[None] < np.asarray([11, 24])[:, None]
        a, _ = encode_readouts(x, valid)
        x[~valid] = 1e9
        b, _ = encode_readouts(x, valid)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isnan(a[~valid]).all())
        short, _ = encode_readouts(x[:1, :11])
        np.testing.assert_array_equal(a[:1, :11], short)

    def test_availability_and_invalid_input(self):
        e = ReadoutEncoder()
        for q in range(16):
            result = e.update(readout())
            self.assertEqual(result.ready, q >= 7)
            self.assertEqual(np.isfinite(result.values[16:19]).all(), q >= 10)
            self.assertEqual(np.isfinite(result.values[19:]).all(), q >= 14)
        previous = e._seen
        with self.assertRaises(ValueError):
            e.update(np.full(25, np.nan))
        self.assertEqual(e._seen, previous)
        with self.assertRaises(ValueError):
            encode_readouts(np.tile(readout(), (1, 4, 1)), [[True, False, True, False]])
        for config in (dict(history_width=0), dict(epsilon=0), dict(smooth_width=True)):
            with self.assertRaises(ValueError):
                EncoderConfig(**config)

    def test_raw_normalization_permutation_and_reset(self):
        rng = np.random.default_rng(44)
        original, permuted, rescaled = RoutingDynamicsEncoder(), RoutingDynamicsEncoder(), RoutingDynamicsEncoder()
        permutations = np.stack([rng.permutation(32) for _ in range(8)])
        for _ in range(16):
            p = rng.uniform(.02, .1, (8, 10, 11, 32)).astype(np.float32)
            pp = np.take_along_axis(p, permutations[:, None, None, :], axis=-1)
            a, b, c = original.update(p), permuted.update(pp), rescaled.update(p * 8)
            # Reductions of float32 probabilities can vary slightly with expert order.
            names = [i for i, name in enumerate(FEATURE_NAMES) if name not in ("layer_agreement", "decoupling_occupancy")]
            np.testing.assert_allclose(a.values[names], b.values[names], atol=8e-5, rtol=1e-4, equal_nan=True)
            np.testing.assert_allclose(a.values, c.values, atol=1e-12, rtol=0, equal_nan=True)
        original.reset()
        self.assertFalse(original.update(p).ready)
        with self.assertRaises(ValueError):
            original.update(np.zeros_like(p))
        with self.assertRaises(ValueError):
            query_readout(np.ones((8, 10, 10, 32)))


if __name__ == "__main__":
    unittest.main()
