import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, "/data/coding/robot-whisper-0909/moe-trap-control")
from collection_routes import HB_LAYERS, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from gate_capture import DtypeFullCapture, DtypeScopeCapture
from gate_protocol import SCOPES, make_gate_bias, probes, scope_mask
from test_v8_feature_control import Gate


class GateProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)

    def test_balanced_scopes_and_matched_signs(self):
        for scope in SCOPES:
            mask = scope_mask(scope)
            self.assertEqual(mask.sum(), 12 if "_state_" in scope else 120)
            positive = make_gate_bias(scope, 0, .1, 1)
            negative = make_gate_bias(scope, 0, .1, -1)
            np.testing.assert_array_equal(negative, -positive)
            np.testing.assert_array_equal(positive[~mask], 0)
            np.testing.assert_allclose(positive.mean(-1), 0, atol=1e-8)
            np.testing.assert_allclose(np.sqrt((positive[mask] ** 2).mean(-1)), .1)
            np.testing.assert_array_equal(make_gate_bias(scope, 0, .05, 1), positive * .5)
        self.assertEqual(len(probes()), 64)
        with self.assertRaises(ValueError):
            make_gate_bias(SCOPES[0], 0, 1, 1)

    def test_scopes_are_disjoint_and_omit_middle(self):
        union = sum(scope_mask(s).astype(int) for s in SCOPES)
        self.assertLessEqual(union.max(), 1)
        np.testing.assert_array_equal(union[:, 3:7], 0)
        np.testing.assert_array_equal(union[:, :3], 1)
        np.testing.assert_array_equal(union[:, 7:], 1)

    def test_actual_dispatch_and_bfloat16_audit(self):
        for dtype in (torch.float32, torch.bfloat16):
            gates = [(index, Gate().to(dtype)) for index in HB_LAYERS]
            bias = make_gate_bias("front_state_early", 0, .1, 1)
            observed = []
            capture = DtypeScopeCapture(gates, bias)
            with capture:
                handles = [gate.register_forward_hook(lambda _g, _i, out: observed.append((out[0].clone(), out[1].clone())))
                           for _, gate in gates]
                for _ in range(10):
                    for _, gate in gates:
                        gate(torch.randn(1, 11, 12, dtype=dtype))
                for handle in handles:
                    handle.remove()
            result = capture.response()
            for i, key in enumerate((EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)):
                expected = torch.stack([o[i] for o in observed]).reshape(10, 8, 11, 4).permute(1, 0, 2, 3)
                np.testing.assert_array_equal(result[key], expected.float().numpy())
            self.assertTrue(all(not gate._forward_hooks for _, gate in gates))

    def test_zero_bias_exact_and_exception_cleanup(self):
        gates = [(index, Gate().to(torch.bfloat16)) for index in HB_LAYERS]
        inputs = [torch.randn(1, 11, 12, dtype=torch.bfloat16) for _ in range(80)]
        results = []
        for capture in (DtypeFullCapture(gates), DtypeScopeCapture(gates, np.zeros((8, 10, 11, 32), np.float32))):
            with capture:
                for index, value in enumerate(inputs):
                    gates[index % 8][1](value)
            results.append(capture.response())
        for key in (EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY, "v8_control/effective_probs_fp32"):
            np.testing.assert_array_equal(results[0][key], results[1][key])
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with DtypeScopeCapture(gates, np.zeros((8, 10, 11, 32), np.float32)):
                raise RuntimeError("intentional")
        self.assertTrue(all(not gate._forward_hooks for _, gate in gates))


if __name__ == "__main__":
    unittest.main()
