"""Native-only HB capture for batch-1 collection, without hidden activations."""

from __future__ import annotations

import numpy as np

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
CAPTURE_KEY = "collection/full_hb"
PROBS_KEY = "collection/hb_probs"
NATIVE_IDS_KEY = "collection/hb_native_ids"
NATIVE_WEIGHTS_KEY = "collection/hb_native_weights"
EFFECTIVE_IDS_KEY = "collection/hb_effective_ids"
EFFECTIVE_WEIGHTS_KEY = "collection/hb_effective_weights"
FIELDS = (PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)


class FullHBCapture:
    def __init__(self, layers):
        self.layers = tuple(layers)
        if tuple(index for index, _ in self.layers) != HB_LAYERS:
            raise ValueError("Unexpected HB layer ordering")
        self.records = []
        self.handles = []

    def __enter__(self):
        try:
            for index, gate in self.layers:
                self.handles.append(gate.register_forward_hook(self._hook(index)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer):
        def capture(gate, inputs, output):
            import torch
            from torch.nn import functional as F

            value = inputs[0]
            if tuple(value.shape[:2]) != (1, 11):
                raise ValueError("Full HB collection requires batch=1 and 11 suffix tokens")
            ids, weights, _ = output
            with torch.no_grad():
                logits = F.linear(value.reshape(-1, value.shape[-1]), gate.weight, None)
                probability = logits.softmax(dim=-1)
                recomputed_ids = probability.topk(4, dim=-1, sorted=False).indices
                self.records.append((layer, probability.detach().clone(), ids.detach().clone(),
                                     weights.detach().clone(), recomputed_ids.detach().clone()))
        return capture

    def response(self):
        import torch

        if [record[0] for record in self.records] != list(HB_LAYERS) * 10:
            raise RuntimeError("Missing or reordered HB gate calls")

        def stack(column, dtype):
            value = torch.stack([row[column] for row in self.records])
            return value.reshape(10, 8, 11, -1).permute(1, 0, 2, 3).to(device="cpu", dtype=dtype).numpy()

        probability = stack(1, torch.float32)
        ids = stack(2, torch.int16)
        weights = stack(3, torch.float32)
        recomputed_ids = stack(4, torch.int16)
        if not np.array_equal(np.sort(ids, axis=-1), np.sort(recomputed_ids, axis=-1)):
            raise RuntimeError("Captured probabilities do not reproduce native top-4")
        selected = np.take_along_axis(probability, ids.astype(np.int64), axis=-1)
        expected_weights = selected / selected.sum(axis=-1, keepdims=True)
        error = float(np.max(np.abs(expected_weights - weights)))
        if not np.isfinite(probability).all() or error > 1e-5:
            raise RuntimeError("Full HB capture differs from actual native combine weights")
        return {PROBS_KEY: np.ascontiguousarray(probability, dtype=np.float16),
                NATIVE_IDS_KEY: np.ascontiguousarray(ids), NATIVE_WEIGHTS_KEY: np.ascontiguousarray(weights),
                EFFECTIVE_IDS_KEY: np.ascontiguousarray(ids), EFFECTIVE_WEIGHTS_KEY: np.ascontiguousarray(weights),
                "collection/native_weight_max_error": error, "collection/routing_mode": "native_only"}


class FullHBPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.metadata = policy.metadata
        self.backend_name = policy.backend_name
        self.metadata.update(collection_full_hb_supported=True, collection_hidden_capture=False,
                             collection_routing_mode="native_only", collection_batch_size=1,
                             collection_probability_axes=["layer", "denoise", "suffix_token", "expert"],
                             collection_hb_layers=list(HB_LAYERS))

    def infer(self, observation):
        import torch

        request = dict(observation)
        enabled = request.pop(CAPTURE_KEY, False)
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("Full HB capture flag must be boolean")
        if not enabled:
            return self.policy.infer(request)
        if not request.get("routing/capture", False):
            raise ValueError("Full HB capture requires actual top-4 capture")
        with FullHBCapture(self.policy._routing_layers) as capture:
            response = self.policy.infer(request)
        response.update(capture.response())
        # The existing action-token capture is an independent axis/alignment check.
        ids = response[EFFECTIVE_IDS_KEY][:, :, 1:].transpose(1, 0, 2, 3)
        weights = response[EFFECTIVE_WEIGHTS_KEY][:, :, 1:].transpose(1, 0, 2, 3)
        if not np.array_equal(ids, response["routing/expert_ids"]) or not np.array_equal(weights, response["routing/expert_weights"]):
            raise RuntimeError("Full HB capture disagrees with the existing top-4 hook")
        response["collection/cuda_memory_mib"] = dict(allocated=torch.cuda.memory_allocated() / 1024**2,
            reserved=torch.cuda.memory_reserved() / 1024**2, peak_allocated=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved=torch.cuda.max_memory_reserved() / 1024**2)
        return response
