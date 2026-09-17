import unittest
from types import SimpleNamespace

import numpy as np

from gate_runtime import BASE
from crossover_protocol import (LABELS, DONORS, BIASED, ACTIVATION_SHAPE, BIAS_SHAPE,
                                validate_donor, validate_bias, expected_computation, set_change)
from crossover_capture import OutputReplayScope, replace_action_tokens, parameter_digest, parameter_versions


class CrossoverTests(unittest.TestCase):
    def test_design_and_budget(self):
        self.assertEqual(len(LABELS) * 15, 135)
        self.assertEqual(len(DONORS), 6)
        self.assertIn('route_only', BIASED)
        self.assertNotIn('output_only', BIASED)
        self.assertEqual(expected_computation('route_only'), 'native')
        self.assertEqual(expected_computation('output_only'), 'joint')
        with self.assertRaises(ValueError):
            expected_computation('unknown')

    def test_donor_validation(self):
        donor = np.zeros(ACTIVATION_SHAPE, np.float32)
        self.assertIs(validate_donor(donor), donor)
        for invalid in (donor.astype(np.float16), donor[0], [0.]):
            with self.assertRaises(ValueError):
                validate_donor(invalid)
        donor[4, 0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_donor(donor)

    def test_bias_scope(self):
        bias = np.zeros(BIAS_SHAPE, np.float32)
        with self.assertRaises(ValueError):
            validate_bias(bias)
        bias[4:, :, 1:] = .1
        self.assertIs(validate_bias(bias), bias)
        for where, value in (((0, 0, 1, 0), .1), ((4, 0, 0, 0), .1), ((4, 0, 1, 0), .31)):
            bad = bias.copy()
            bad[where] = value
            with self.assertRaises(ValueError):
                validate_bias(bad)

    def test_exact_token_replay_without_mutating_inputs(self):
        import torch
        donor = np.full(ACTIVATION_SHAPE, 3., np.float32)
        source = torch.ones(1, 11, 1024)
        patched = replace_action_tokens(source, donor, 4, 0)
        self.assertTrue(torch.all(source == 1))
        self.assertTrue(torch.all(patched[:, 0] == 1))
        self.assertTrue(torch.all(patched[:, 1:] == 3))
        self.assertTrue(np.all(donor == 3))
        self.assertNotEqual(patched.data_ptr(), source.data_ptr())
        for slot, step in ((3, 0), (8, 0), (4, 10), (4, -1)):
            with self.assertRaises(RuntimeError):
                replace_action_tokens(source, donor, slot, step)
        with self.assertRaises(RuntimeError):
            replace_action_tokens(source.bfloat16(), donor, 4, 0)

    def fake_model(self):
        import torch
        class HBMoE(torch.nn.Module):
            def forward(self, value):
                return value + 1
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = HBMoE()
        layers = torch.nn.ModuleList([Block() for _ in range(18)])
        model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(layers=layers)))
        return model, layers

    def test_hook_order_and_cleanup_on_exception(self):
        import torch
        model, layers = self.fake_model()
        donor = np.full(ACTIVATION_SHAPE, 9., np.float32)
        seen = []
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with OutputReplayScope(model, donor) as scope:
                handle = layers[12].mlp.register_forward_hook(lambda m, i, o: seen.append(o.clone()))
                try:
                    value = layers[12].mlp(torch.zeros(1, 11, 1024))
                    self.assertTrue(torch.all(scope.raw[0][1] == 1))
                    self.assertTrue(torch.equal(seen[0], value))
                    self.assertTrue(torch.all(value[:, 1:] == 9))
                    raise RuntimeError('intentional')
                finally:
                    handle.remove()
        self.assertFalse(any(m._forward_hooks for m in layers.modules()))

    def test_complete_order_and_missing_calls(self):
        import torch
        from collection_routes import HB_LAYERS
        model, layers = self.fake_model()
        donor = np.full(ACTIVATION_SHAPE, 7., np.float32)
        with OutputReplayScope(model, donor) as scope:
            for step in range(10):
                for index in HB_LAYERS:
                    layers[index].mlp(torch.zeros(1, 11, 1024))
        response = scope.response()
        self.assertEqual(response['crossover/patch_sites'].shape, (40, 2))
        self.assertTrue(np.all(response['crossover/raw_total'] == 1))
        scope.raw.pop()
        with self.assertRaises(RuntimeError):
            scope.response()

    def test_parameter_content_hash_detects_changes(self):
        import torch
        model = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        before, versions = parameter_digest(model), parameter_versions(model)
        model(torch.ones(1, 3))
        self.assertEqual(before, parameter_digest(model))
        self.assertEqual(versions, parameter_versions(model))
        model.weight.add_(1)
        self.assertNotEqual(before, parameter_digest(model))
        self.assertNotEqual(versions, parameter_versions(model))

    def test_set_change_only_measures_target_scope(self):
        base = np.broadcast_to(np.arange(4), (8, 10, 11, 4)).copy()
        other = base[..., ::-1].copy()
        self.assertEqual(set_change(other, base), 0.)
        other[:4, ..., 0] = 7
        self.assertEqual(set_change(other, base), 0.)
        other[4:, :, 1:, 0] = 7
        self.assertEqual(set_change(other, base), 1.)


if __name__ == '__main__':
    unittest.main()
