"""Numerical contracts for the constrained controller and causal disturbance observer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recovery_mpc import RecoveryMPC, project_ball, update_disturbance

MODEL = "design/recovery_dynamics_20260908.json"


class MPCContracts(unittest.TestCase):
    def setUp(self):
        self.controller = RecoveryMPC(MODEL)

    def test_unconstrained_solution_matches_infinite_horizon_lqr(self):
        c = self.controller
        x = np.array([.01, -.02, .03, .003, -.001, .002])
        result = c.solve(x, np.zeros(3))
        self.assertTrue(result["optimizer_accepted"])
        np.testing.assert_allclose(result["plan_cm"][0], -np.asarray(c.proof["K"])@x, atol=2e-5, rtol=0)
        closed = c.A-c.B@np.asarray(c.proof["K"])
        self.assertLess(float(x@closed.T@c.P@closed@x), float(x@c.P@x))

    def test_saturated_plans_and_independent_forward_cost(self):
        c = self.controller
        rng = np.random.default_rng(6143)
        for _ in range(20):
            x = np.r_[rng.uniform(-4, 4, 3), rng.uniform(-.5, .5, 3)]
            d = project_ball(rng.normal(0, .1, 3), .5)
            result = c.solve(x, d)
            self.assertTrue(result["optimizer_accepted"])
            self.assertLessEqual(float(np.linalg.norm(result["plan_cm"], axis=1).max()), 1+1e-12)
            predicted, zero = x.copy(), x.copy()
            cost, zero_cost = 0., 0.
            for k, u in enumerate(result["plan_cm"]):
                predicted = c.A@predicted+c.B@u+c.D@d
                zero = c.A@zero+c.D@d
                np.testing.assert_allclose(result["predicted_states_cm"][k], predicted, atol=1e-12)
                weight = c.P if k == c.horizon-1 else c.Q
                cost += predicted@weight@predicted+u@c.R@u
                zero_cost += zero@weight@zero
            self.assertAlmostEqual(result["objective"], cost, places=9)
            self.assertLessEqual(cost, zero_cost+1e-7)

    def test_observer_uses_previous_transition_and_bounds_disturbance(self):
        c = self.controller
        previous, velocity, u = np.zeros(3), np.array([.2, -.1, .3]), np.array([.3, .2, -.5])
        d = np.array([.03, -.02, .01])
        measured = c.a*velocity+c.b*u+d
        updated = update_disturbance(previous, measured, velocity, u, c.a, c.b)
        np.testing.assert_allclose(updated, d/2, atol=1e-15)
        bounded = update_disturbance(previous, np.ones(3)*100, velocity, u, c.a, c.b)
        self.assertAlmostEqual(float(np.linalg.norm(bounded)), .5)

    def test_failed_optimization_falls_back_to_bounded_p(self):
        fake = SimpleNamespace(x=np.full(18, np.nan), success=False, status=9, nit=120)
        x = np.array([3., -4., 2., 0., 0., 0.])
        with patch("recovery_mpc.minimize", return_value=fake):
            result = self.controller.solve(x, np.zeros(3))
        self.assertTrue(result["fallback"])
        np.testing.assert_allclose(result["plan_cm"][0], project_ball(-x[:3]))
        self.assertTrue(np.isfinite(result["predicted_states_cm"]).all())
        with self.assertRaises(ValueError):
            self.controller.solve(np.full(6, np.nan), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
