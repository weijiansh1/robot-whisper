"""Causal continuation scoring and real-termination window semantics."""

import unittest
import numpy as np

from alarm_trajectory_study import score_trajectories, physical_window


def line_fixture():
    values = np.zeros((1, 21, 10))
    values[0, :, 0] = np.arange(21)
    values[:, :7] = np.nan
    prefix = np.tile(values[0, 7:10].reshape(1, -1)/np.sqrt(3), (20, 1))
    start, end = np.zeros((20, 10)), np.zeros((20, 10))
    start[:, 0], end[:, 0] = 9, 12
    return values, dict(prefix=prefix, start=start, end=end, delta=end-start, global_rows=np.arange(20))


def static_physics():
    return dict(predicates=np.zeros((4, 2), bool), positions=np.zeros((4, 2, 3)),
                eef=np.tile([1., 0, 0], (4, 1)), grasp=np.zeros((4, 2), bool),
                goal_distance=np.ones((4, 2)))


class TrajectoryTests(unittest.TestCase):
    def test_exact_reference_continuation(self):
        values, bank = line_fixture()
        result = score_trajectories(values, bank)
        self.assertEqual(result['displacement'][0, 12], 0.)
        self.assertEqual(result['endpoint'][0, 12], 0.)
        self.assertEqual(result['displacement'][0, 13], 0.)
        self.assertEqual(result['endpoint'][0, 13], 1.)

    def test_reverse_direction_is_not_identical_continuation(self):
        values, bank = line_fixture()
        values[0, 12, 0] = 6
        result = score_trajectories(values, bank)
        self.assertEqual(result['displacement'][0, 12], 6.)
        self.assertEqual(result['direction_cosine'][0, 12], -1.)

    def test_future_never_changes_past_scores_or_neighbors(self):
        values, bank = line_fixture()
        full = score_trajectories(values, bank)
        prefix = score_trajectories(values[:, :16], bank)
        changed = values.copy()
        changed[:, 16:] = 1000
        other = score_trajectories(changed, bank)
        for key in full:
            np.testing.assert_array_equal(full[key][:, :16], prefix[key])
            np.testing.assert_array_equal(full[key][:, :16], other[key][:, :16])

    def test_unobserved_window_has_no_score(self):
        values, bank = line_fixture()
        result = score_trajectories(values[:, :12], bank)
        self.assertTrue(np.isnan(result['displacement']).all())
        self.assertTrue((result['neighbor_ids'] == -1).all())

    def test_rematching_cannot_raise_minimum_endpoint_distance(self):
        rng = np.random.default_rng(19)
        values = rng.normal(size=(2, 19, 10))
        bank = dict(prefix=rng.normal(size=(37, 30)), start=rng.normal(size=(37, 10)),
                    end=rng.normal(size=(37, 10)), global_rows=np.arange(37))
        bank['delta'] = bank['end']-bank['start']
        result = score_trajectories(values, bank)
        self.assertTrue((result['rematched_endpoint'][:, 12:] <= result['endpoint'][:, 12:]+1e-6).all())

    def test_partial_terminal_success_is_kept(self):
        data = {key: value[:2] for key, value in static_physics().items()}
        data['predicates'][1] = True
        window = physical_window(data, 0, 5)
        self.assertEqual(window['category'], 'completed')
        self.assertEqual(window['observed_queries'], 1)
        self.assertTrue(window['terminal_success'])

    def test_static_is_not_automatically_a_trap(self):
        data = static_physics()
        data['grasp'][:] = True
        window = physical_window(data, 0)
        self.assertEqual(window['category'], 'holding_or_static')
        self.assertNotIn('trap', window)

    def test_return_motion_is_distinct_from_static(self):
        data = static_physics()
        data['eef'][:, 1] = [0, .02, -.02, 0]
        window = physical_window(data, 0)
        self.assertEqual(window['category'], 'return_motion_without_subgoal_gain')
        self.assertEqual(window['eef_net_m'], 0.)
        self.assertGreater(window['eef_path_m'], .03)


if __name__ == '__main__':
    unittest.main(verbosity=2)
