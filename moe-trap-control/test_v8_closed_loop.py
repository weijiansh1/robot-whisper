"""Causal scoring, population confirmation and paired-search contracts."""

import copy
import unittest

import numpy as np

from v8_closed_loop import (RecoveryState, V8Monitor, choose, extra_noises,
                            guided_population, limits, preview, risk)


class ClosedLoopContracts(unittest.TestCase):
    def test_zero_threshold_margin_and_v7_conjunction(self):
        thresholds, margins = limits()
        scores = thresholds-2*margins
        scores[1] = thresholds[1]+10*margins[1]
        self.assertAlmostEqual(float(risk(scores, thresholds, margins)), -2)
        scores[2] = thresholds[2]+3*margins[2]
        self.assertAlmostEqual(float(risk(scores, thresholds, margins)), 3)
        scores[1:3] = thresholds[1:3]-2*margins[1:3]
        scores[3] = 0.
        self.assertEqual(float(risk(scores, thresholds, margins)), 0.)

    def test_missing_evidence_cannot_confirm_recovery(self):
        thresholds, margins = limits()
        scores = thresholds-2*margins
        scores[2] = np.nan
        self.assertTrue(np.isinf(risk(scores, thresholds, margins)))
        state = RecoveryState()
        self.assertFalse(state.commit([-3.])["all_low"])
        self.assertEqual(state.streak, 0)

    def test_full_population_and_consecutive_execution_required(self):
        state = RecoveryState()
        self.assertTrue(state.commit(np.full(16, -2.))["all_low"])
        mixed = np.full(16, -2.)
        mixed[-1] = .01
        self.assertFalse(state.commit(mixed)["all_low"])
        self.assertEqual(state.streak, 0)
        for _ in range(3):
            result = state.commit(np.full(16, -2.))
        self.assertEqual(result["exit_reason"], "population_confirmed")
        with self.assertRaises(ValueError):
            state.commit(np.full(16, -2.))

    def test_unreachable_target_has_finite_budget(self):
        state = RecoveryState()
        for _ in range(12):
            result = state.commit(np.zeros(16))
        self.assertFalse(result["active"])
        self.assertEqual(result["exit_reason"], "recovery_budget_exhausted")

    def test_preview_cannot_clear_latched_alarm_or_advance_history(self):
        monitor = V8Monitor()
        rng = np.random.default_rng(713)
        probabilities = rng.dirichlet(np.ones(32), size=(12, 8, 10, 11)).astype(np.float32)
        for p in probabilities[:8]:
            monitor.update(p)
        monitor.first_alarm = 4
        before = copy.deepcopy(monitor)
        a = preview(monitor, probabilities[8:])
        b = preview(monitor, probabilities[[10, 8, 11, 9]])
        np.testing.assert_array_equal(b, a[[2, 0, 3, 1]])
        self.assertEqual(monitor.v7.query, before.v7.query)
        self.assertEqual(monitor.first_alarm, 4)
        self.assertEqual(len(monitor.raw), len(before.raw))

    def test_generator_pairing_and_finite_cem_variance(self):
        z = extra_noises("parent", 0)
        np.testing.assert_array_equal(z, extra_noises("parent", 0))
        raw = np.concatenate([np.zeros((1, 10, 24)), z[:7]])
        actual, state = guided_population(raw, np.arange(8), z[7:])
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(state["std"] >= .5))
        self.assertEqual(state["elite"], [0, 1, 2, 3])
        scores = np.arange(16, dtype=float)[::-1]
        self.assertEqual(choose("iid_v8", scores, "parent", 0), 15)
        self.assertEqual(choose("iid_random", scores, "parent", 0),
                         choose("guided_random", scores, "parent", 0))


if __name__ == "__main__":
    unittest.main()
