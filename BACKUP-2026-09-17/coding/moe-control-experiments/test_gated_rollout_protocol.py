"""Budget, deterministic live proposals, screening and history isolation."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from head_control_protocol import V82Monitor
from response_matrix_protocol import SHAPE, MASK, bias_bank, specifications
from gated_rollout_protocol import ARMS, LABELS, evaluate, in_window, proposal, target_operator
from v8_feature_control import EFFECTIVE_PROBS


class GatedRolloutTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(409)
        self.history = rng.dirichlet(np.ones(32) * 8, size=(21, 8, 10, 11)).astype(np.float16)
        self.parent = dict(benchmark='plus', base_task_id=3, init_state_id=26, query=20)
        self.prefix = V82Monitor()
        for p in self.history[:20]:
            self.prefix.update(p)
        self.native = {EFFECTIVE_PROBS: self.history[20].astype(np.float32), 'actions': np.zeros((10, 7), np.float32)}

    def test_twelve_query_opportunities_not_twelve_acceptances(self):
        for arm in ARMS:
            active = [q for q in range(52) if in_window(arm, q, 20)]
            self.assertEqual(active, [] if arm == 'native' else list(range(20, 32)))

    def test_invalid_arm_and_queries_rejected(self):
        with self.assertRaises(ValueError):
            in_window('unknown', 20, 20)
        with self.assertRaises(ValueError):
            in_window('mobility', -1, 20)
        with self.assertRaises(ValueError):
            proposal(self.history[20], self.history[19], self.parent, 32, 'mobility')

    def test_all_proposals_match_frozen_matrix(self):
        bank, _ = bias_bank(self.history[20], self.history[19], self.parent)
        specs = specifications()
        for arm, label in LABELS.items():
            bias, info = proposal(self.history[20], self.history[19], self.parent, 20, arm)
            index = next(i for i, s in enumerate(specs) if s['label'] == label)
            np.testing.assert_array_equal(bias, bank[index])
            self.assertEqual(info['label'], label)

    def test_random_preserves_site_energy_and_uses_current_query(self):
        combo, _ = proposal(self.history[20], self.history[19], self.parent, 20, 'mobility_balance')
        random, _ = proposal(self.history[20], self.history[19], self.parent, 20, 'random_balance')
        later, _ = proposal(self.history[20], self.history[19], self.parent, 21, 'random_balance')
        np.testing.assert_array_equal(np.sort(combo, axis=-1), np.sort(random, axis=-1))
        self.assertFalse(np.array_equal(later, random))
        self.assertTrue(np.all(random[~MASK] == 0))
        self.assertLessEqual(np.abs(random).max(), .3000001)

    def test_direction_uses_previous_executed_route(self):
        first, _ = proposal(self.history[20], self.history[19], self.parent, 20, 'mobility')
        second, _ = proposal(self.history[20], self.history[18], self.parent, 20, 'mobility')
        self.assertFalse(np.array_equal(first, second))

    def test_degenerate_direction_falls_back_without_random_search(self):
        flat = np.full(SHAPE, 1 / 32, np.float16)
        bias, info = proposal(flat, flat, self.parent, 20, 'mobility_balance')
        self.assertIsNone(bias)
        self.assertEqual(info['fallback_reason'], 'degenerate_direction')

    def test_native_and_shadow_tie_rejected(self):
        for arm in LABELS:
            decision = evaluate(self.prefix, self.native, self.native, arm, np.ones(6))
            self.assertFalse(decision['accepted'])
            self.assertTrue(decision['reasons'])

    def test_previews_do_not_advance_executed_history(self):
        before = copy.deepcopy(self.prefix)
        changed = dict(self.native, **{EFFECTIVE_PROBS: self.history[18].astype(np.float32)})
        evaluate(self.prefix, self.native, changed, 'mobility_balance', np.ones(6))
        self.assertEqual(self.prefix.v7.query, before.v7.query)
        np.testing.assert_array_equal(self.prefix.raw, before.raw)
        self.prefix.update(self.native[EFFECTIVE_PROBS].astype(np.float16))
        self.assertEqual(self.prefix.v7.query, 20)

    def test_random_and_combo_have_identical_acceptance_rules(self):
        self.assertEqual(target_operator('random_balance'), target_operator('mobility_balance'))
        changed = dict(self.native, **{EFFECTIVE_PROBS: self.history[18].astype(np.float32)})
        first = evaluate(self.prefix, self.native, changed, 'mobility_balance', np.ones(6))
        second = evaluate(self.prefix, self.native, changed, 'random_balance', np.ones(6))
        self.assertEqual(first['accepted'], second['accepted'])
        self.assertEqual(first['reasons'], second['reasons'])

    def test_action_and_gripper_guards_are_enforced(self):
        action = self.native['actions'].copy()
        action[:, :6] = .1
        action[:, 6] = -1
        changed = dict(self.native, actions=action)
        value = evaluate(self.prefix, self.native, changed, 'mobility', np.ones(6))
        self.assertFalse(value['accepted'])
        self.assertIn('action_rms', value['reasons'])
        self.assertIn('gripper_change', value['reasons'])
        with self.assertRaises(ValueError):
            evaluate(self.prefix, self.native, changed, 'mobility', np.zeros(6))

    def test_acceptance_uses_instantaneous_not_historical_score(self):
        base = dict(normalized=np.zeros(5), instantaneous=np.zeros(5), margins=np.ones(5))
        changed = dict(normalized=np.ones(5), instantaneous=np.array([-.1, 0, 0, -.1, 0]), margins=np.ones(5))
        with patch('gated_rollout_protocol.measure', side_effect=[base, changed, base, changed]):
            result = evaluate(self.prefix, self.native, self.native, 'mobility_balance', np.ones(6))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['formal_screen'])
        self.assertTrue(result['fp32_screen'])
        self.assertEqual(result['reasons'], [])

    def test_first_query_matches_archived_development_inputs(self):
        import json
        root = Path(__file__).resolve().parent / 'runs/p3c-response-matrix-20260915-463ir6kj'
        config = json.loads((root / 'config.json').read_text())
        specs = specifications()
        for parent in config['parents']:
            with np.load(root / parent['input_path'], allow_pickle=False) as inputs, \
                    np.load(root / parent['bank_path'], allow_pickle=False) as bank:
                for arm, label in LABELS.items():
                    bias, _ = proposal(inputs['native_hb'], inputs['prefix_hb'][-1], parent, parent['query'], arm)
                    index = next(i for i, s in enumerate(specs) if s['label'] == label)
                    np.testing.assert_array_equal(bias, bank['biases'][index])


if __name__ == '__main__':
    unittest.main()
