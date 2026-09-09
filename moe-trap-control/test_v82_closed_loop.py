"""Version-specific time counters, causal previews, and matched random streams."""

import copy
import unittest

import numpy as np

from v8_closed_loop import choose, limits, preview, risk
from v82_closed_loop import ARMS, V82Monitor, base_arm, thresholds_at


class V82Contracts(unittest.TestCase):
    def test_dynamic_threshold_changes_only_new_heads(self):
        base, margins = limits()
        dynamic = thresholds_at("iid_v82", 40, base)
        np.testing.assert_array_equal(dynamic[:3], base[:3])
        np.testing.assert_allclose(dynamic[3:], base[3:]-.06, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(thresholds_at("iid_v8", 40, base), base)
        np.testing.assert_array_equal(thresholds_at("guided_random_v82", 40, base), dynamic)
        scores = dynamic-2*margins
        self.assertAlmostEqual(float(risk(scores, dynamic, margins)), -2)
        scores[3] = -.03
        self.assertGreater(float(risk(scores, dynamic, margins)), 0)
        self.assertLess(float(risk(scores, base, margins)), 0)

    def test_candidates_do_not_advance_v82_time_or_latches(self):
        monitor = V82Monitor()
        probabilities = np.random.default_rng(846).dirichlet(np.ones(32), size=(12, 8, 10, 11)).astype(np.float32)
        for p in probabilities[:8]:
            monitor.update(p)
        monitor.first_v82_alarm = 5
        before = copy.deepcopy(monitor)
        first = preview(monitor, probabilities[8:])
        repeated = preview(monitor, probabilities[[11, 8, 10, 9]])
        np.testing.assert_array_equal(repeated, first[[3, 0, 2, 1]])
        self.assertEqual(monitor.v7.query, before.v7.query)
        self.assertEqual(monitor.first_v82_alarm, 5)
        self.assertEqual(monitor.v82_counts, before.v82_counts)
        self.assertEqual(monitor.v82_first, before.v82_first)
        np.testing.assert_array_equal(monitor.raw, before.raw)

    def test_random_controls_and_versions_share_choice_stream(self):
        scores = np.linspace(2, -2, 16)
        choices = []
        for arm in ARMS[1:]:
            value = choose(base_arm(arm), scores, "paired-native-main", 3)
            if "random" in arm:
                choices.append(value)
            else:
                self.assertEqual(value, 15)
        self.assertEqual(len(set(choices)), 1)

    def test_invalid_query_and_arm_rejected(self):
        base, _ = limits()
        for q in (-1, .5):
            with self.assertRaises(ValueError):
                thresholds_at("iid_v82", q, base)
        with self.assertRaises(ValueError):
            base_arm("iid_unknown")


if __name__ == "__main__":
    unittest.main()
