"""Decision, causal-state and malformed-candidate contracts for internal selection."""

import copy
import unittest

import numpy as np

from internal_action_selector import InternalActionSelector, choose


class SelectionContracts(unittest.TestCase):
    def setUp(self):
        self.thresholds = np.ones(7)
        self.scores = np.full((4, 7), .5)
        self.scores[:, 0] = [2., 1.5, 1.4, 1.3]
        self.actions = np.zeros((4, 10, 7))
        self.actions[:, :, 6] = 1.
        self.actions[:, :, 0] = np.asarray([0., .01, .02, .03])[:, None]

    def test_dominance_and_smallest_admissible_action(self):
        result = choose(self.scores, self.actions, self.thresholds, True)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["admissible"], [False, True, False, False])
        self.assertEqual(result["active_signals"], ["freeze"])

    def test_new_violation_or_worsening_active_signal_abstains(self):
        self.scores[1, 1] = 1.2
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)
        self.scores[1, 1] = .5
        self.scores[1, 0] = 2.1
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)

    def test_gripper_guard_covers_entire_executed_chunk(self):
        self.actions[1, 8, 6] = -1.
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True, execute_count=5)["candidate"], 1)

    def test_freeze_is_not_rewarded_and_pure_route_changes_are_rejected(self):
        self.scores[1, 0] = 2.5
        self.scores[1, 4] = 0.
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)
        self.scores[1, 0] = 1.5
        self.actions[1] = self.actions[0]
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)

    def test_untriggered_cold_start_and_no_active_signal(self):
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, False)["reason"], "untriggered")
        self.scores[0, 0] = np.nan
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["reason"], "baseline_evidence_unavailable")
        self.scores[0] = .5
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["reason"], "default_has_no_active_excess")

    def test_invalid_alternative_is_excluded_but_invalid_default_fails(self):
        self.scores[1] = np.nan
        self.assertEqual(choose(self.scores, self.actions, self.thresholds, True)["candidate"], 0)
        self.actions[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            choose(self.scores, self.actions, self.thresholds, True)

    def test_preview_does_not_commit_candidates_or_depend_on_their_order(self):
        selector = InternalActionSelector()
        rng = np.random.default_rng(918)
        probabilities = rng.dirichlet(np.ones(32), size=(12, 8, 10, 11)).astype(np.float32)
        for probability in probabilities[:8]:
            selector.observe(probability)
        before = copy.deepcopy(selector)
        expected = selector.preview(probabilities[8:])
        reordered = selector.preview(probabilities[[11, 9, 10, 8]])
        np.testing.assert_array_equal(reordered, expected[[3, 1, 2, 0]])
        self.assertEqual(selector.monitor.v7.query, before.monitor.v7.query)
        np.testing.assert_array_equal(selector.observe(probabilities[9]), before.observe(probabilities[9]))

    def test_mismatched_observations_are_not_comparable(self):
        selector = InternalActionSelector()
        with self.assertRaises(ValueError):
            selector.propose([None]*4, self.actions, ["a", "a", "b", "a"], True)


if __name__ == "__main__":
    unittest.main()
