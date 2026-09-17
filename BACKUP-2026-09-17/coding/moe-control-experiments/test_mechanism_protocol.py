import unittest
from types import SimpleNamespace

import numpy as np

from gate_runtime import BASE
from mechanism_protocol import comparison, selected_queries, true_intervals, variants
from mechanism_capture import MechanismCapture
from response_matrix_protocol import SHAPE


class MechanismTests(unittest.TestCase):
    def test_selection_not_best_score(self):
        rows = [dict(query=q, decision=dict(accepted=q in (19, 21, 25))) for q in range(18, 30)]
        self.assertEqual(selected_queries(dict(first_alarm=17), rows),
                         (('before_alarm', 12), ('first_accepted', 19), ('last_accepted', 25)))

    def test_selection_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            selected_queries(dict(first_alarm=17), [dict(query=18, decision=dict(accepted=True))] * 2)

    def test_variants_equal_energy(self):
        bias = np.zeros(SHAPE, np.float32)
        bias[4:, :, 1:] = np.random.default_rng(7).uniform(-.1, .1, size=(4, 10, 10, 32))
        parent = dict(benchmark='plus', base_task_id=3, init_state_id=26)
        bank = variants(bias, parent, 30)
        np.testing.assert_array_equal(bank['opposite'], -bias)
        np.testing.assert_array_equal(np.sort(bank['random'], axis=-1), np.sort(bias, axis=-1))
        np.testing.assert_array_equal(bank['random'], variants(bias, parent, 30)['random'])
        self.assertAlmostEqual(np.linalg.norm(bank['random'].astype(float)), np.linalg.norm(bias.astype(float)))

    def test_variants_rejects_out_of_scope(self):
        with self.assertRaises(ValueError):
            variants(np.ones(SHAPE, np.float32) * .1, {}, 30)

    def test_scale_measurement(self):
        result = comparison(np.array([2., 2.]), np.array([1., 1.]))
        self.assertEqual(result['relative_l2'], 1.)
        self.assertEqual(result['delta_rms'], 1.)
        self.assertIsNone(comparison(np.ones(2), np.zeros(2))['relative_l2'])

    def test_nonfinite_and_misaligned(self):
        for a, b in (([np.nan], [1.]), ([1., 2.], [1.])):
            with self.assertRaises(ValueError):
                comparison(a, b)

    def test_intervals_include_endpoints(self):
        self.assertEqual(true_intervals([1, 1, 0, 1, 0, 1]), [[0, 1], [3, 3], [5, 5]])
        self.assertEqual(true_intervals([]), [])

    def test_hooks_cleaned_on_exception(self):
        import torch
        class HBMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_experts = torch.nn.Identity()
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = HBMoE()
                self.post_attention_layernorm = torch.nn.Identity()
        layers = torch.nn.ModuleList([Block() for _ in range(18)])
        model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(layers=layers)),
                                action_out_proj=torch.nn.Linear(1024, 24))
        with self.assertRaisesRegex(RuntimeError, 'test error'):
            with MechanismCapture(model):
                self.assertTrue(layers[12]._forward_hooks)
                raise RuntimeError('test error')
        for module in list(layers.modules()) + [model.action_out_proj]:
            self.assertFalse(module._forward_hooks)


if __name__ == '__main__':
    unittest.main()
