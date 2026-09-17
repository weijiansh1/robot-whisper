"""No simulator or GPU is needed for the execution contract tests."""

import unittest

import numpy as np

from replan_window_protocol import (ANCHORS, ExecutionWindow, TIMING_PARENTS,
                                    maximum_calls, physical_noise)


class WindowTests(unittest.TestCase):
    def test_exact_window_and_return(self):
        window = ExecutionWindow('window5', 180)
        self.assertEqual([window.length(s) for s in (180, 185, 190, 195, 200, 210)],
                         [5, 5, 5, 5, 10, 10])

    def test_native_unchanged(self):
        self.assertEqual(ExecutionWindow('native10', 180).length(180), 10)

    def test_suffix_not_reused_or_rescaled(self):
        actions = np.arange(70, dtype=np.float32).reshape(10, 7)
        window = ExecutionWindow('window5', 180)
        first = window.prefix(actions, 180)
        second = window.prefix(actions + 100, 185)
        np.testing.assert_array_equal(first, actions[:5])
        np.testing.assert_array_equal(second, (actions + 100)[:5])
        self.assertFalse(np.shares_memory(first, actions))

    def test_noise_independent_of_call_count(self):
        parent = dict(benchmark='plus', base_task_id=0, init_state_id=26)
        for r in range(4):
            first = physical_noise(parent, 17, r, 190)
            physical_noise(parent, 17, r, 185)
            np.testing.assert_array_equal(first, physical_noise(parent, 17, r, 190))
        rng = np.random.default_rng(17)
        bank = [rng.standard_normal((10, 24)).astype(np.float32) for _ in range(20)]
        np.testing.assert_array_equal(bank[19], physical_noise(parent, 17, 0, 190))

    def test_validation_and_budget(self):
        self.assertEqual(maximum_calls(), 2080)
        for arm, start in [('bad', 180), ('window5', 185), ('window5', 510)]:
            with self.assertRaises(ValueError):
                ExecutionWindow(arm, start)
        with self.assertRaises(ValueError):
            ExecutionWindow('native10', 180).prefix(np.zeros((5, 7), np.float32), 180)

    def test_split_at_task_init_not_seed_or_variant(self):
        def group(name):
            return tuple(name.split('-')[1:])
        self.assertFalse({group(n) for n, _, _ in ANCHORS} & {group(n) for n in TIMING_PARENTS})


if __name__ == '__main__':
    unittest.main()
