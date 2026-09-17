"""Checks for functional decomposition, unsupported sites and transparent capture."""

import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from moe_response_analysis import factorial_energy, functional_comparison, symmetric_decomposition
from moe_final_response_recorder import FinalResponseRecorder

sys.path.insert(0, '/data/srv/src')
from moevla.models.modeling_moe import HBMoE


class ToyLayer(nn.Module):
    def __init__(self, moe):
        super().__init__()
        config = SimpleNamespace(hidden_size=6, intermediate_size=12, moe_intermediate_size=8,
                                 hidden_act='silu', pretraining_tp=1, n_routed_experts=8,
                                 n_shared_experts=1, num_experts_per_tok=4, scoring_func='softmax',
                                 aux_loss_alpha=0., seq_aux=False, norm_topk_prob=True)
        self.mlp = HBMoE(config) if moe else nn.Identity()


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(i in (2, 3, 4, 5, 12, 13, 14, 15)) for i in range(16)])

    def forward(self, x, flows=10):
        for flow in range(flows):
            for layer in self.layers:
                if isinstance(layer.mlp, HBMoE):
                    x = x + .1 * layer.mlp(x, None)
        return x


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([[4, 1, 7, 2]])
        self.w0 = np.array([[.2, .3, .1, .4]])
        self.w1 = np.array([[.3, .2, .2, .3]])
        self.e0 = np.arange(12).reshape(1, 4, 3).astype(float) + 1
        self.e1 = self.e0 + np.array([[[1., -1., 2.]]])

    def test_constant_weights_assign_change_to_expert_output(self):
        d = symmetric_decomposition(self.ids, self.w0, self.w0[..., None]*self.e0,
                                    self.ids, self.w0, self.w0[..., None]*self.e1)
        np.testing.assert_allclose(d['weight_term'], 0, atol=1e-14)
        np.testing.assert_allclose(d['expert_term'], [[1, -1, 2]], atol=1e-14)

    def test_constant_expert_outputs_assign_change_to_weights(self):
        d = symmetric_decomposition(self.ids, self.w0, self.w0[..., None]*self.e0,
                                    self.ids, self.w1, self.w1[..., None]*self.e0)
        np.testing.assert_allclose(d['expert_term'], 0, atol=1e-14)
        np.testing.assert_allclose(d['weight_term'], [[-.6, -.6, -.6]], atol=1e-14)

    def test_dispatch_rank_permutation_is_not_expert_change(self):
        order = [2, 0, 3, 1]
        d = symmetric_decomposition(self.ids, self.w0, self.w0[..., None]*self.e0,
                                    self.ids[:, order], self.w0[:, order],
                                    (self.w0[..., None]*self.e0)[:, order])
        self.assertTrue(d['same_support'].all())
        np.testing.assert_allclose(d['expert_term'], 0, atol=0)
        np.testing.assert_allclose(d['weight_term'], 0, atol=0)

    def test_changed_support_is_unavailable(self):
        changed = self.ids.copy()
        changed[0, 0] = 6
        d = symmetric_decomposition(self.ids, self.w0, self.w0[..., None]*self.e0,
                                    changed, self.w1, self.w1[..., None]*self.e1)
        self.assertFalse(d['same_support'].any())
        self.assertTrue(np.isnan(d['expert_term']).all())
        self.assertTrue(np.isnan(d['expert_term_norm_fraction']).all())

    def test_duplicate_or_zero_weight_is_rejected(self):
        invalid = self.ids.copy()
        invalid[0, 1] = invalid[0, 0]
        with self.assertRaises(ValueError):
            symmetric_decomposition(invalid, self.w0, self.w0[..., None]*self.e0,
                                    self.ids, self.w1, self.w1[..., None]*self.e1)
        with self.assertRaises(ValueError):
            symmetric_decomposition(self.ids, self.w0*0, self.e0, self.ids, self.w1, self.e1)

    def test_symmetric_identity_when_both_factors_change(self):
        d = symmetric_decomposition(self.ids, self.w0, self.w0[..., None]*self.e0,
                                    self.ids, self.w1, self.w1[..., None]*self.e1)
        np.testing.assert_allclose(d['weight_term'] + d['expert_term'], d['observed_delta'], atol=1e-14)
        reverse = symmetric_decomposition(self.ids, self.w1, self.w1[..., None]*self.e1,
                                          self.ids, self.w0, self.w0[..., None]*self.e0)
        np.testing.assert_allclose(d['weight_term'], -reverse['weight_term'], atol=1e-14)
        np.testing.assert_allclose(d['expert_term'], -reverse['expert_term'], atol=1e-14)

    def test_zero_input_delta_does_not_create_infinite_gain(self):
        first = dict(ids=self.ids, weights=self.w0, expert_contrib=self.e0,
                     routed=(self.e0*self.w0[..., None]).sum(-2), shared=np.ones((1, 3)), input=np.ones((1, 3)))
        second = dict(first)
        second['routed'] = first['routed'] * 2
        values = functional_comparison(first, second, raw_experts=True)
        self.assertTrue(np.isnan(values['routed_relative_gain']).all())

    def test_factorial_separates_noise_observation_and_interaction(self):
        zero = np.zeros((2, 3))
        one = np.ones((2, 3))
        observation = factorial_energy(zero, zero, one, one)
        np.testing.assert_allclose(observation['fraction'], [[1, 0, 0], [1, 0, 0]])
        noise = factorial_energy(zero, one, zero, one)
        np.testing.assert_allclose(noise['fraction'], [[0, 1, 0], [0, 1, 0]])
        interaction = factorial_energy(one, -one, -one, one)
        np.testing.assert_allclose(interaction['fraction'], [[0, 0, 1], [0, 0, 1]])

    def test_factorial_natural_diagonal_is_sum_of_averaged_input_effects(self):
        rng = np.random.default_rng(5)
        f00, f01, f10, f11 = rng.normal(size=(4, 8, 11, 6))
        result = factorial_energy(f00, f01, f10, f11)
        np.testing.assert_allclose(result['natural_observation_term'] + result['natural_noise_term'],
                                   f11-f00, atol=1e-14)

    def test_recorder_preserves_native_output_and_restores_dispatch(self):
        torch.manual_seed(5)
        model = ToyCore().eval()
        x = torch.randn(1, 11, 6)
        with torch.no_grad():
            native = model(x.clone())
            recorder = FinalResponseRecorder(model).attach()
            try:
                captured = model(x.clone())
                record = recorder.end()
            finally:
                recorder.close()
            after = model(x.clone())
        self.assertTrue(torch.equal(native, captured))
        self.assertTrue(torch.equal(native, after))
        self.assertEqual(record['expert_output'].shape, (8, 11, 4, 6))
        self.assertLess(record['routed_reconstruction_relative'].max(), 1e-5)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertTrue(all('moe_infer' not in b.module.__dict__ for b in recorder.blocks))

    def test_incomplete_capture_fails_and_can_be_closed(self):
        model = ToyCore().eval()
        recorder = FinalResponseRecorder(model).attach()
        try:
            with torch.no_grad():
                model(torch.ones(1, 11, 6), flows=9)
            with self.assertRaises(ValueError):
                recorder.end()
        finally:
            recorder.close()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
