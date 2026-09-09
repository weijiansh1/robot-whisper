"""Contract tests for route execution, request cleanup, and paired random streams."""

import unittest

import numpy as np
import torch

from audit_collection_experiment import expected_swap
from collection_protocol import branch_noise, stream_seed
from collection_routes import (FullHBCapture, HB_LAYERS, INTERVENTION_PROBS_KEY,
    NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY, NATIVE_WEIGHTS_KEY, EFFECTIVE_WEIGHTS_KEY, swap1_near,
    FullASCapture, AS_LAYERS, AS_FIELDS)
from collection_mps import PrivateMPS
from run_collection_preflight import replica_port


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(32), requires_grad=False)

    def forward(self, value):
        probability = value.reshape(-1, 32).softmax(-1)
        weights, ids = probability.topk(4, sorted=False)
        return ids, weights / weights.sum(-1, keepdim=True), None


class CollectionContractTest(unittest.TestCase):
    def test_as_preserves_unquantized_actual_weights(self):
        capture = FullASCapture([(index, Gate()) for index in AS_LAYERS])
        probability = torch.tensor([[.9234567, .0500012, .0265421]] * 11)
        ids = torch.zeros(11, 1, dtype=torch.int64)
        weights = probability[:, :1].clone()
        capture.records = [(index, probability, ids, weights) for _ in range(10) for index in AS_LAYERS]
        response = capture.response()
        self.assertEqual(response[AS_FIELDS[2]][0, 0, 0, 0], weights[0, 0].item())
        self.assertNotEqual(float(response[AS_FIELDS[0]][0, 0, 0, 0]), weights[0, 0].item())

    def test_swap_ties_and_token_scope(self):
        probability = torch.ones(11, 32)
        ids = torch.tensor([[7, 3, 9, 5]] * 11)
        effective = swap1_near(probability, ids)
        self.assertTrue(torch.equal(effective[0], ids[0]))
        self.assertTrue(torch.equal(effective[1:], torch.tensor([[7, 0, 9, 5]] * 10)))
        self.assertTrue(torch.equal(ids, torch.tensor([[7, 3, 9, 5]] * 11)))

    def test_actual_hook_output_and_cleanup(self):
        layers = [(index, Gate()) for index in HB_LAYERS]
        value = torch.rand(1, 11, 32, generator=torch.Generator().manual_seed(71))
        native = [gate(value) for _, gate in layers]
        observed = []
        with FullHBCapture(layers, "swap1_near") as capture:
            handles = [gate.register_forward_hook(lambda gate, inputs, output: observed.append(output[0].clone()))
                       for _, gate in layers]
            for _ in range(10):
                for _, gate in layers:
                    gate(value)
            for handle in handles:
                handle.remove()
        response = capture.response()
        np.testing.assert_array_equal(response[EFFECTIVE_IDS_KEY], expected_swap(
            response[INTERVENTION_PROBS_KEY], response[NATIVE_IDS_KEY]))
        np.testing.assert_array_equal(torch.stack(observed).reshape(10, 8, 11, 4).permute(1, 0, 2, 3),
                                      response[EFFECTIVE_IDS_KEY])
        self.assertEqual(np.count_nonzero(response[NATIVE_IDS_KEY] != response[EFFECTIVE_IDS_KEY]), 400)
        np.testing.assert_array_equal(response[NATIVE_WEIGHTS_KEY], response[EFFECTIVE_WEIGHTS_KEY])
        for (_, gate), (ids, weights, _) in zip(layers, native):
            after = gate(value)
            self.assertTrue(torch.equal(ids, after[0]) and torch.equal(weights, after[1]))
            self.assertEqual(len(gate._forward_hooks), 0)

    def test_exception_removes_hooks(self):
        layers = [(index, Gate()) for index in HB_LAYERS]
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with FullHBCapture(layers, "swap1_near"):
                raise RuntimeError("sentinel")
        self.assertTrue(all(not gate._forward_hooks for _, gate in layers))

    def test_paired_noise_independent_of_order(self):
        wanted = branch_noise("main", "event", 2, 19)
        for replicate in range(4):
            for query in range(25):
                branch_noise("main", "event", replicate, query)
        np.testing.assert_array_equal(wanted, branch_noise("main", "event", 2, 19))
        self.assertFalse(np.array_equal(wanted, branch_noise("main", "event", 3, 19)))
        self.assertNotEqual(stream_seed("main", "event", 2, 19, "policy"),
                            stream_seed("main", "event", 2, 19, "environment"))

    def test_gpu_exclusion_and_port_uniqueness(self):
        with self.assertRaises(ValueError):
            PrivateMPS(6, "/tmp/forbidden-mps-test")
        for replicas in (1, 2, 4, 8):
            ports = [replica_port(gpu, 12000, "long", replica, max(10, 5 * replicas))
                     for gpu in (0, 1, 2, 3, 4, 5, 7) for replica in range(replicas)]
            self.assertEqual(len(ports), len(set(ports)))


if __name__ == "__main__":
    unittest.main()
