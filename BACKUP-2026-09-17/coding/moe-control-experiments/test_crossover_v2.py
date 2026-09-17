import unittest

import numpy as np

import test_crossover_protocol as fixtures
from crossover_protocol import ACTIVATION_SHAPE, SITE_MASK
from crossover_capture_v2 import WholeOutputScope, OUTPUT_MASK


class WholeBoundaryTests(unittest.TestCase):
    def test_gate_and_replay_scopes_are_distinct(self):
        self.assertEqual(int(SITE_MASK.sum()), 400)
        self.assertEqual(int(OUTPUT_MASK.sum()), 440)
        self.assertTrue(np.all(OUTPUT_MASK[SITE_MASK]))
        self.assertFalse(OUTPUT_MASK[:4].any())

    def test_state_token_replayed_without_mutating_raw_output(self):
        import torch
        model, layers = fixtures.CrossoverTests().fake_model()
        donor = np.full(ACTIVATION_SHAPE, 9., np.float32)
        with WholeOutputScope(model, donor) as scope:
            value = layers[12].mlp(torch.zeros(1, 11, 1024))
            self.assertTrue(torch.all(value == 9))
            self.assertTrue(torch.all(scope.raw[0][1] == 1))
            front = layers[2].mlp(torch.zeros(1, 11, 1024))
            self.assertTrue(torch.all(front == 1))
        self.assertTrue(np.all(donor == 9))
        self.assertFalse(any(m._forward_hooks for m in layers.modules()))


if __name__ == '__main__':
    unittest.main()
