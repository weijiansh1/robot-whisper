"""Direction, dose, scope, feedback history and screening invariants."""

import copy
import unittest

import numpy as np

from head_control_protocol import V82Monitor
from response_matrix_protocol import (MASK, OPERATORS, SHAPE, bias_bank, directions,
                                       measure, screen, specifications)


class ResponseMatrixTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.history = rng.dirichlet(np.ones(32) * 8, size=(20, 8, 10, 11)).astype(np.float16)
        self.shadow = self.history[-1]
        self.previous = self.history[-2]
        self.parent = dict(benchmark='plus', base_task_id=3, init_state_id=26, query=19)
        self.bank, self.info = bias_bank(self.shadow, self.previous, self.parent)
        self.specs = specifications()

    def test_frozen_bank_size_and_labels(self):
        self.assertEqual(self.bank.shape, (30,) + SHAPE)
        self.assertEqual(len({s['label'] for s in self.specs}), 30)
        self.assertEqual(sum(s['randomized'] for s in self.specs), 10)

    def test_equal_requested_energy_and_scope(self):
        self.assertTrue(np.all(self.bank[:, ~MASK] == 0))
        self.assertLessEqual(np.max(np.abs(self.bank)), .3000001)
        for bias, spec in zip(self.bank, self.specs):
            np.testing.assert_allclose(np.linalg.norm(bias.astype(float)),
                                       self.info['high_energy'] * spec['fraction'], rtol=1e-7)

    def test_signs_and_doses(self):
        by_label = {s['label']: bias for s, bias in zip(self.specs, self.bank)}
        for operator in OPERATORS:
            np.testing.assert_array_equal(by_label[operator + '-high-plus'], -by_label[operator + '-high-minus'])
            np.testing.assert_array_equal(by_label[operator + '-high-plus'], 2 * by_label[operator + '-low-plus'])

    def test_matched_random_preserves_each_site(self):
        by_label = {s['label']: bias for s, bias in zip(self.specs, self.bank)}
        for operator in OPERATORS:
            original, shuffled = by_label[operator + '-high-plus'], by_label['random-' + operator + '-high']
            np.testing.assert_array_equal(np.sort(original, axis=-1), np.sort(shuffled, axis=-1))
            self.assertFalse(np.array_equal(original, shuffled))

    def test_combinations_have_unit_energy(self):
        value = directions(self.shadow, self.previous)
        for operator in OPERATORS:
            self.assertAlmostEqual(float(np.linalg.norm(value[operator])), 1.)
        pair = value['mobility'] + value['balance']
        np.testing.assert_allclose(value['mobility_balance'], pair / np.linalg.norm(pair))

    def test_deterministic_and_parent_specific_randomization(self):
        repeated, _ = bias_bank(self.shadow, self.previous, self.parent)
        np.testing.assert_array_equal(repeated, self.bank)
        other, _ = bias_bank(self.shadow, self.previous, dict(self.parent, init_state_id=47))
        np.testing.assert_array_equal(other[:20], self.bank[:20])
        self.assertFalse(np.array_equal(other[20:], self.bank[20:]))

    def test_feedback_does_not_mutate_prefix(self):
        prefix = V82Monitor()
        for p in self.history[:-1]:
            prefix.update(p)
        before = copy.deepcopy(prefix)
        baseline = measure(prefix, self.shadow)
        changed = np.roll(self.shadow, 1, axis=-1).copy()
        result = measure(prefix, changed)
        self.assertEqual(prefix.v7.query, before.v7.query)
        np.testing.assert_array_equal(prefix.raw, before.raw)
        delta = result['scores'] - baseline['scores']
        raw_delta = result['instantaneous'] - baseline['instantaneous']
        np.testing.assert_allclose(delta[[0, 3, 4]] * 6, raw_delta[[0, 3, 4]], atol=2e-6)

    def test_degenerate_direction_is_not_invented(self):
        flat = np.full(SHAPE, 1 / 32, np.float16)
        with self.assertRaises(ValueError):
            bias_bank(flat, flat, self.parent)

    def test_screen_requires_targets_and_side_effect_bounds(self):
        self.assertTrue(screen([-.1, 0, 0, -.1, 0], 'mobility_balance', .01, 0))
        self.assertFalse(screen([-.1, .1, 0, -.1, 0], 'mobility_balance', .01, 0))
        self.assertFalse(screen([-.1, 0, 0, 0, 0], 'mobility_balance', .01, 0))
        self.assertFalse(screen([-.1, 0, 0, -.1, 0], 'mobility_balance', .01, 1))


if __name__ == '__main__':
    unittest.main()
