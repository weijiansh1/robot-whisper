"""Verify causal dispatch, hybrid choices, bounds and nominal contraction."""

import unittest

import numpy as np

from control_bank import (ARMS, PHYSICAL, ActionFilter, choose_operator, physical_action, waypoints)


class ControlBankTests(unittest.TestCase):
    def test_nominal_cartesian_contraction_and_gripper_bounds(self):
        initial = np.array([.1, -.1, .9])
        history = np.array([[.08, -.1, .9], [.06, -.11, .91], [.04, -.11, .91]])
        scale = np.array([.05]*3)
        for operator in PHYSICAL:
            positions, durations, grips = waypoints(operator, initial, history, 1.)
            self.assertLessEqual(sum(durations), 24)
            for target, grip in zip(positions, grips):
                action = physical_action(operator, initial, target, grip, scale)
                delta = action[:3]*scale
                self.assertLessEqual(np.linalg.norm(delta), .01000000001)
                self.assertEqual(action[6], grip)
                self.assertLessEqual(np.linalg.norm(initial+delta-target), np.linalg.norm(initial-target)+1e-12)

    def test_retrace_uses_only_bounded_historical_targets(self):
        initial, history = np.zeros(3), np.array([[1., 0, 0], [0, 2., 0], [0, 0, 3.]])
        targets, durations, _ = waypoints("retrace", initial, history, -1.)
        np.testing.assert_allclose(np.linalg.norm(targets, axis=1), .08)
        self.assertEqual(durations, [8, 8, 8])

    def test_detour_signs_and_degenerate_history(self):
        initial, history = np.zeros(3), np.zeros((3, 3))
        plus, _, _ = waypoints("side_plus", initial, history, 1.)
        minus, _, _ = waypoints("side_minus", initial, history, 1.)
        np.testing.assert_array_equal(plus[1, :2], -minus[1, :2])
        np.testing.assert_array_equal(plus[[0, 2]], minus[[0, 2]])

    def test_hybrid_guard_priority_and_missingness(self):
        tau = np.zeros(5)
        for scores, expected in (([1,-1,-1,-1,-1], "withdraw4"), ([1,1,-1,-1,-1], "smooth2"),
                ([1,-1,1,-1,-1], "side_plus"), ([-1,-1,-1,0,-1], "replan2"),
                ([-1]*5, "resample"), ([np.nan]*5, "resample")):
            self.assertEqual(choose_operator("moe_switch", scores, tau, "parent", 0), expected)
        self.assertEqual(len(ARMS), 31)

    def test_filter_preserves_gripper_and_stops_at_query_eight(self):
        raw = np.array([.8]*6+[-1.], np.float32)
        for operator in ("damp2", "boost2", "smooth2"):
            f = ActionFilter(operator, np.zeros(7))
            output = f.command(raw, 0)
            self.assertEqual(output[-1], raw[-1])
            self.assertLessEqual(np.abs(output).max(), 1.)
            np.testing.assert_array_equal(f.command(raw, 8), raw)
            self.assertEqual(f.chunk_size(7), 2)
            self.assertEqual(f.chunk_size(8), 10)

    def test_smoothing_updates_on_executed_commands_only(self):
        raw, filt = np.ones(7, np.float32), ActionFilter("smooth2", np.zeros(7))
        before = filt.previous.copy()
        filt.chunk_size(0)
        np.testing.assert_array_equal(filt.previous, before)
        np.testing.assert_allclose(filt.command(raw, 0)[:6], .25)
        np.testing.assert_allclose(filt.command(raw, 0)[:6], .4375)


if __name__ == "__main__":
    unittest.main()
