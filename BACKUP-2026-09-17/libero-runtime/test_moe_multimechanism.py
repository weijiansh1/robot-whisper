"""Numerical and external-label regressions for the multimechanism audit."""

import copy
import json
import unittest

import numpy as np

from moe_multimechanism import contribution_contrast, physical_window, phase, FEATURES


def snapshot(seed=1):
    rng = np.random.default_rng(seed)
    ids = np.broadcast_to(np.arange(4), (8, 11, 4)).copy()
    weights = rng.uniform(.1, 1, (8, 11, 4))
    weights /= weights.sum(-1, keepdims=True)
    expert = rng.normal(size=(8, 11, 4, 7))
    routed = (weights[..., None]*expert).sum(-2)
    shared = rng.normal(size=(8, 11, 7))
    return dict(ids=ids, weights=weights, expert_output=expert, routed=routed,
                shared=shared, total=routed+shared, input=rng.normal(size=(8, 11, 7)))


def physics():
    return dict(predicates=np.zeros((4, 2), bool), positions=np.zeros((4, 2, 3)),
        eef=np.zeros((4, 3)), goal_distance=np.ones((4, 2)), unary_distance=np.full((4, 2), np.nan),
        rotations=np.broadcast_to(np.eye(3), (4, 2, 3, 3)).copy(),
        joint_values=np.empty((4, 0)), joint_ranges=np.empty((0, 2)),
        names=np.array(['a', 'b']), goals=np.array([json.dumps(['on', 'a', 'b']), json.dumps(['close', 'b'])]),
        grasp=np.zeros((4, 2), bool))


class ContributionTests(unittest.TestCase):
    def test_id_permutation_changes_nothing(self):
        a, b = snapshot(1), snapshot(2)
        expected, attribution, _ = contribution_contrast(a, b)
        b['ids'] = b['ids'][..., ::-1]
        b['weights'] = b['weights'][..., ::-1]
        b['expert_output'] = b['expert_output'][..., ::-1, :]
        actual, attr, _ = contribution_contrast(a, b)
        for key in FEATURES:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(attr, attribution, atol=1e-12)

    def test_disjoint_support_reconstructs_update(self):
        a, b = snapshot(1), snapshot(2)
        b['ids'] += 4
        values, attribution, audit = contribution_contrast(a, b)
        np.testing.assert_array_equal(values['retained_update_fraction'], 0)
        np.testing.assert_allclose(values['support_change_update_fraction'], 1)
        np.testing.assert_allclose(attribution.sum(-1)+values['shared_update_attribution'], 1, atol=1e-12)
        self.assertLess(audit['decomposition_max_absolute'], 1e-12)

    def test_no_update_is_undefined_not_perfect_cancellation(self):
        a = snapshot()
        values, _, _ = contribution_contrast(a, copy.deepcopy(a))
        self.assertTrue(np.isnan(values['expert_update_cancellation']).all())
        self.assertTrue(np.isnan(values['routed_shared_update_cosine']).all())
        self.assertTrue(np.isnan(values['shared_update_attribution']).all())

    def test_duplicate_ids_are_rejected(self):
        a, b = snapshot(1), snapshot(2)
        b['ids'][0, 0, 1] = b['ids'][0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            contribution_contrast(a, b)


class ExternalTests(unittest.TestCase):
    def test_goal_distance_allows_soft_joint_limit_overshoot(self):
        from restore_multimechanism_physics import interval_for
        class Door:
            def is_close(self, position):
                return position > -.005
        low, high = interval_for(Door(), 'close', -2.094, 0.)
        self.assertAlmostEqual(low, -.005)
        self.assertIsNone(high)

    def test_transient_gain_is_not_endpoint_progress(self):
        data = physics()
        data['predicates'][1, 0] = True
        self.assertEqual(physical_window(data, 0, 3)['local_label'], 'transient_or_regressing')
        data['predicates'][3, 0] = True
        self.assertEqual(physical_window(data, 0, 3)['local_label'], 'verified_progress')
        data['predicates'][0, 1] = True
        self.assertEqual(physical_window(data, 0, 3)['local_label'], 'transient_or_regressing')

    def test_rotation_excludes_static_even_without_translation(self):
        data = physics()
        angle = np.deg2rad(10)
        data['rotations'][-1, 0] = [[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        result = physical_window(data, 0, 3)
        self.assertEqual(result['category'], 'other_motion_no_gain')
        self.assertTrue(result['translation_static_but_internal_motion'])

    def test_grasp_identity_and_censoring(self):
        data = physics()
        data['grasp'][0, 0] = True
        data['grasp'][1, 1] = True
        self.assertNotEqual(phase(data, 0), phase(data, 1))
        self.assertFalse(physical_window(data, 2, 3)['complete_window'])


if __name__ == '__main__':
    unittest.main()
