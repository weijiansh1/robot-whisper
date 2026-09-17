"""Focused tests for selection, causal scheduling, and paired random streams."""

import unittest
import numpy as np
from online_protocol import (candidate_noises, noise_bank, query_mode, select_candidate,
                             seeded_rng, RECOVERY_QUERIES)


class OnlineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(benchmark='plus', base_task_id=0, init_state_id=26)
        self.bank = noise_bank(20260940)
        rng = np.random.default_rng(1)
        self.routes = rng.dirichlet(np.ones(32), size=(4, 8, 10, 11)).astype(np.float16)

    def test_native_noise_stream(self):
        rng = np.random.default_rng(20260940)
        for step in range(0, 520, 10):
            expected = rng.standard_normal((10, 24)).astype(np.float32)
            np.testing.assert_array_equal(candidate_noises(self.parent, self.bank, step, 4)[0], expected)

    def test_candidate_budget_does_not_advance_native(self):
        for step in (0, 5, 210, 215):
            a = candidate_noises(self.parent, self.bank, step, 1)
            b = candidate_noises(self.parent, self.bank, step, 4)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(b, candidate_noises(self.parent, self.bank, step, 4))
            self.assertFalse(np.array_equal(b[0], b[1]))

    def test_explicit_budget_exit(self):
        self.assertEqual(query_mode('route_short', 0), dict(active=True, chunk=5, candidates=4))
        self.assertTrue(query_mode('short', RECOVERY_QUERIES - 1)['active'])
        self.assertEqual(query_mode('route_short', RECOVERY_QUERIES), dict(active=False, chunk=10, candidates=1))
        self.assertFalse(query_mode('native', 0)['active'])

    def test_route_ties_keep_candidate_zero(self):
        routes = np.repeat(self.routes[:1], 4, axis=0)
        selected, scores = select_candidate('route_short', routes, self.parent, 210)
        self.assertEqual(selected, 0)
        np.testing.assert_allclose(scores, 0, atol=1e-8)

    def test_out_of_scope_routes_cannot_change_ranking(self):
        selected, scores = select_candidate('route_short', self.routes, self.parent, 210)
        changed = self.routes.copy()
        changed[:, :4] = np.roll(changed[:, :4], 1, axis=-1)
        changed[:, :, 3:] = np.roll(changed[:, :, 3:], 2, axis=-1)
        changed[:, :, :, 0] = np.roll(changed[:, :, :, 0], 3, axis=-1)
        actual, actual_scores = select_candidate('route_short', changed, self.parent, 210)
        self.assertEqual(selected, actual)
        np.testing.assert_array_equal(scores, actual_scores)

    def test_random_selection_ignores_scores(self):
        selected, _ = select_candidate('random_short', self.routes, self.parent, 210)
        again, _ = select_candidate('random_short', self.routes[::-1], self.parent, 210)
        self.assertEqual(selected, again)
        self.assertEqual(selected, seeded_rng(self.parent, 210, 100).integers(4))

    def test_invalid_shape_and_mass(self):
        with self.assertRaises(ValueError):
            select_candidate('route_short', self.routes[:3], self.parent, 210)
        with self.assertRaises(ValueError):
            select_candidate('route_short', self.routes * 0, self.parent, 210)


if __name__ == '__main__':
    unittest.main()
