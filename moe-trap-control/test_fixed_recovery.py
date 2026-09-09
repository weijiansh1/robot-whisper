import random
import unittest

import numpy as np

from fixed_recovery_control import targets, recovery_action, rng_record, restore_rng, stalled, MODULE


class PhysicalRecoveryTests(unittest.TestCase):
    def test_targets_and_controller_bounds(self):
        current = np.array([.2,-.3,.1])
        recent = np.array([-.5,.7,.4])
        lift,retreat = targets(current,recent)
        np.testing.assert_allclose(lift-current,[0,0,.04])
        self.assertAlmostEqual(np.linalg.norm((retreat-lift)[:2]),.04)
        action = recovery_action("withdraw",current,retreat,-.3,[.05,.05,.05])
        self.assertAlmostEqual(np.linalg.norm(action[:3]*.05),.01)
        np.testing.assert_array_equal(action[3:],[0,0,0,-1])
        np.testing.assert_array_equal(recovery_action("hold",current,retreat,.3,[.05]*3),[0,0,0,0,0,0,1])

    def test_stationary_history_still_produces_lift(self):
        start=np.array([0.,0.,.3])
        t=targets(start,start)
        np.testing.assert_array_equal(t[0],t[1])
        for _ in range(8):
            start += recovery_action("withdraw",start,t[0],1,[.05]*3)[:3]*.05
        np.testing.assert_allclose(start,t[0],atol=1e-15)

    def test_environment_rng_round_trip_including_gaussian_cache(self):
        np.random.seed(41)
        random.seed(42)
        np.random.normal()
        random.gauss(0,1)
        row=rng_record(8)
        expected_np=np.random.normal(size=30)
        expected_py=[random.gauss(0,1) for _ in range(30)]
        restore_rng(row)
        np.testing.assert_array_equal(np.random.normal(size=30),expected_np)
        self.assertEqual([random.gauss(0,1) for _ in range(30)],expected_py)

    def test_stall_needs_full_window_and_detects_excursion(self):
        self.assertFalse(stalled([np.zeros(3)]*5))
        self.assertTrue(stalled([np.zeros(3)]*6))
        self.assertFalse(stalled([np.array([.006,0,0])]+[np.zeros(3)]*5))


if __name__ == "__main__":
    unittest.main()
