import unittest
from unittest.mock import patch

import numpy as np

from gate_runtime import BASE
from crossover_rollout_protocol import active, guard, committed_route, maximum_calls, proposal, ARMS
from v8_feature_control import EFFECTIVE_PROBS


class CrossoverRolloutTests(unittest.TestCase):
    def test_fixed_window(self):
        for arm in ARMS:
            self.assertEqual(sum(active(arm, q, 21) for q in range(52)), 0 if arm == 'native' else 12)
        with self.assertRaises(ValueError):
            active('unknown', 21, 21)

    def test_budget(self):
        self.assertEqual(maximum_calls([dict(query=q) for q in (21, 18, 30, 34, 23)]), 876)

    def test_guard_is_independent_of_routing_scores(self):
        native = dict(actions=np.zeros((10, 7), np.float32))
        candidate = dict(actions=native['actions'].copy(), trap_scores=np.full(5, 1e9))
        candidate['actions'][:, :6] = .01
        value = guard(native, candidate, np.ones(6))
        self.assertTrue(value['accepted'])
        self.assertFalse(value['route_scores_used'])
        self.assertEqual(value, guard(native, dict(candidate, trap_scores=np.full(5, -1e9)), np.ones(6)))

    def test_action_and_gripper_guards(self):
        base = dict(actions=np.zeros((10, 7), np.float32))
        candidate = dict(actions=np.full((10, 7), .1, np.float32))
        self.assertEqual(guard(base, candidate, np.ones(6))['reasons'], ['action_rms'])
        candidate['actions'][:] = 0
        candidate['actions'][0, 6] = -1
        self.assertEqual(guard(base, candidate, np.ones(6))['reasons'], ['gripper_change'])

    def test_guard_rejects_invalid_values(self):
        base = dict(actions=np.zeros((10, 7), np.float32))
        for actions in (np.zeros((9, 7)), np.full((10, 7), np.nan)):
            with self.assertRaises(ValueError):
                guard(base, dict(actions=actions), np.ones(6))
        with self.assertRaises(ValueError):
            guard(base, base, np.zeros(6))

    def test_all_arms_use_identical_proposal_rule(self):
        native = {EFFECTIVE_PROBS: np.full((8, 10, 11, 32), 1 / 32, np.float32)}
        previous, parent = native[EFFECTIVE_PROBS].astype(np.float16), dict(query=21)
        with patch('crossover_rollout_protocol.original_proposal', return_value=('bias', 'metadata')) as maker:
            for arm in ARMS[1:]:
                self.assertEqual(proposal(native, previous, parent, 22, arm), ('bias', 'metadata'))
                self.assertEqual(maker.call_args.args[-1], 'mobility_balance')
                self.assertIs(maker.call_args.args[1], previous)
        with self.assertRaises(ValueError):
            proposal(native, previous, parent, 33, 'joint')

    def test_commit_actual_effective_routes_not_donor(self):
        p = np.full((8, 10, 11, 32), 1 / 32, np.float32)
        result = committed_route({EFFECTIVE_PROBS: p, 'donor': p + 1})
        np.testing.assert_array_equal(result, p.astype(np.float16))
        result[:] = 0
        self.assertTrue(np.all(p > 0))
        with self.assertRaises(ValueError):
            committed_route({EFFECTIVE_PROBS: p[0]})


if __name__ == '__main__':
    unittest.main()
