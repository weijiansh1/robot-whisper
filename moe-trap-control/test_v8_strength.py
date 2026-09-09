"""Exact unit-strength compatibility and genuinely larger dispatched biases."""

import unittest

import numpy as np
import torch

from test_v8_feature_control import Gate
from collection_routes import HB_LAYERS, ALL_FIELDS, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from v8_feature_control import FeatureCapture, make_bias, EFFECTIVE_PROBS
from v8_strength_control import ARMS, StrengthCapture, strength_bias, LOGIT_FIELDS


class StrengthTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.p = rng.dirichlet(np.ones(32), size=(8,10,11)).astype(np.float32)
        self.previous = rng.dirichlet(np.ones(32), size=(8,10,11)).astype(np.float32)

    def test_multiplier_is_after_original_clipping(self):
        unit = make_bias(self.p, self.previous, "combined", 1., 7)
        for multiplier in (1,4,16):
            actual = strength_bias(self.p, self.previous, ARMS["combined%d_5"%multiplier], 7)
            np.testing.assert_array_equal(actual, unit*multiplier)
        self.assertGreater(np.abs(actual).max(), 1.)
        np.testing.assert_array_equal(actual[:4], 0)
        np.testing.assert_array_equal(actual[:,:,0], 0)

    def test_random_control_has_identical_per_row_amplitudes(self):
        a = strength_bias(self.p, self.previous, ARMS["combined16_20"], 7)
        b = strength_bias(self.p, self.previous, ARMS["random16_20"], 7)
        np.testing.assert_array_equal(np.sort(a,axis=-1), np.sort(b,axis=-1))

    def test_unit_capture_is_exactly_compatible_and_logits_explain_dispatch(self):
        torch.manual_seed(15)
        gates = [(l,Gate()) for l in HB_LAYERS]
        inputs = [torch.randn(1,11,12) for _ in range(80)]
        bias = strength_bias(self.p,self.previous,ARMS["combined1_5"],7)
        outputs=[]
        for cls in (FeatureCapture,StrengthCapture):
            capture=cls(gates,bias)
            with capture:
                for i,value in enumerate(inputs):
                    gates[i%8][1](value)
            outputs.append(capture.response())
        for key in ALL_FIELDS[:5]:
            np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
        actual=outputs[1]
        np.testing.assert_array_equal(actual[LOGIT_FIELDS[0]]+actual[LOGIT_FIELDS[2]],actual[LOGIT_FIELDS[1]])
        expected=torch.from_numpy(actual[LOGIT_FIELDS[1]]).softmax(-1).numpy()
        np.testing.assert_array_equal(actual[EFFECTIVE_PROBS],expected)
        selected=np.take_along_axis(expected,actual[EFFECTIVE_IDS_KEY].astype(int),-1)
        np.testing.assert_allclose(actual[EFFECTIVE_WEIGHTS_KEY],selected/selected.sum(-1,keepdims=True),atol=1e-6)
        self.assertTrue(all(not gate._forward_hooks for _,gate in gates))

    def test_rejects_out_of_scope_or_excessive_bias(self):
        gates=[(l,Gate()) for l in HB_LAYERS]
        bias=np.zeros_like(self.p)
        bias[4,0,1,0]=17
        with self.assertRaises(ValueError):
            StrengthCapture(gates,bias)
        bias[4,0,1,0]=0
        bias[0,0,1,0]=.1
        with self.assertRaises(ValueError):
            StrengthCapture(gates,bias)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
