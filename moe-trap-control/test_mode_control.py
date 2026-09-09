"""Check distinctions that can invalidate a paired mode intervention experiment."""

import unittest

import numpy as np

from mode_control import (ARMS, ModeTrace, active_mask, decode_arm, mode_name, noise_for,
    normalized, scalar_cost, select, thresholds_at, vector_cost)


class ModeControlTests(unittest.TestCase):
    def test_frozen_threshold_comparators_and_missingness(self):
        tau = np.ones(5)
        np.testing.assert_array_equal(active_mask(tau, tau), [False, False, False, True, True])
        self.assertEqual(mode_name([np.nan, 2, 2, 2, 2], tau), "unavailable")
        self.assertEqual(mode_name(np.zeros(5), tau), "none")

    def test_vector_preserves_component_hidden_by_scalar_and_abstains(self):
        values = np.full((16, 5), 20.)
        values[0] = [1., 8., -2., -2., -2.]
        values[1] = [.5, 20., -3., -2., -2.]
        values[2] = [.8, -2., -2., -2., -2.]
        self.assertEqual(select("scalar", values, np.zeros(5), np.ones(5), "test", 0, 0)[0], 1)
        self.assertEqual(select("mode", values, np.zeros(5), np.ones(5), "test", 0, 0)[0], 2)
        values[0] = -2.
        self.assertEqual(select("mode", values, np.zeros(5), np.ones(5), "test", 0, 0)[0], 0)
        values[0, 0] = np.nan
        self.assertEqual(select("mode", values, np.zeros(5), np.ones(5), "test", 0, 0)[0], 0)

    def test_new_component_crossing_is_penalized(self):
        r = np.array([[-1., -1., -1., -1., -1.], [-1., -1., -1., 2., -1.]])
        np.testing.assert_array_equal(vector_cost(r, [True, False, False, False, False]), [0., 4.])
        self.assertTrue(np.isinf(scalar_cost([[np.nan]*5])[0]))

    def test_noise_pairing_has_no_global_rng_side_effect(self):
        np.random.seed(87)
        before = np.random.get_state()
        first = noise_for("parent", 0, 2, 0)
        for candidate in range(16):
            noise_for("parent", 0, 2, candidate)
        np.testing.assert_array_equal(first, noise_for("parent", 0, 2, 0))
        self.assertFalse(np.array_equal(first, noise_for("parent", 1, 2, 0)))
        self.assertFalse(np.array_equal(first, noise_for("parent", 0, 2, 1)))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_threshold_motion_is_not_raw_signal_improvement(self):
        trace, scores, base = ModeTrace(), np.array([2., -2., -2., -.02, -.02]), np.zeros(5)
        trace.commit(scores, thresholds_at(20, base), np.ones(5))
        row = trace.commit(scores, thresholds_at(21, base), np.ones(5))
        np.testing.assert_array_equal(row["raw_score_delta"], np.zeros(5))
        np.testing.assert_allclose(row["normalized_delta"][3:], .0015)
        np.testing.assert_array_equal(row["active_streaks"], [2, 0, 0, 2, 2])
        np.testing.assert_array_equal(base, np.zeros(5))

    def test_fixed_arm_population_and_tie_breaking(self):
        self.assertEqual(len(ARMS), 15)
        self.assertEqual(decode_arm("withdraw_r1"), ("withdraw", 1))
        scores = np.ones((16, 5))
        for family in ("scalar", "mode"):
            self.assertEqual(select(family, scores, np.zeros(5), np.ones(5), "test", 0, 0)[0], 0)
        with self.assertRaises(ValueError):
            select("mode", scores[:8], np.zeros(5), np.ones(5), "test", 0, 0)
        with self.assertRaises(ValueError):
            normalized(scores, np.zeros(5), np.zeros(5))


if __name__ == "__main__":
    unittest.main()
