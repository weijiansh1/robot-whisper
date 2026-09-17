"""Use upstream dispatch hooks with arithmetic-dtype-aware response auditing."""

import numpy as np

from collection_routes import (HB_LAYERS, FullHBCapture, FullHBPolicy, FullASCapture,
    CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
    EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)
from scope_bias_control import ScopeBiasCapture, BIAS_KEY
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS


class DtypeResponse:
    def response(self):
        import torch

        if [row[0] for row in self.records] != list(HB_LAYERS) * 10:
            raise RuntimeError("Missing or reordered gate calls")
        biased = len(self.records[0]) == 8
        for row in self.records:
            _, native, native_ids, native_weights, recomputed, effective_ids = row[:6]
            if not torch.equal(native_ids.sort(-1).values, recomputed.sort(-1).values):
                raise RuntimeError("Native top-k set mismatch")
            effective = row[7] if biased else native
            effective_weights = row[6] if biased else native_weights
            for probabilities, ids, weights in ((native, native_ids, native_weights),
                                                 (effective, effective_ids, effective_weights)):
                selected = probabilities.gather(-1, ids)
                expected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
                if not torch.equal(expected, weights):
                    raise RuntimeError("Actual gate-dtype combine weights mismatch")
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError("Non-finite gate probabilities")

        def stack(column, dtype):
            value = torch.stack([row[column] for row in self.records])
            return value.reshape(10, 8, 11, -1).permute(1, 0, 2, 3).to(device="cpu", dtype=dtype).numpy().copy()

        native = stack(1, torch.float32)
        return {
            PROBS_KEY: native.astype(np.float16), NATIVE_PROBS: native,
            EFFECTIVE_PROBS: stack(7, torch.float32) if biased else native.copy(),
            NATIVE_IDS_KEY: stack(2, torch.int16), NATIVE_WEIGHTS_KEY: stack(3, torch.float32),
            EFFECTIVE_IDS_KEY: stack(5, torch.int16),
            EFFECTIVE_WEIGHTS_KEY: stack(6 if biased else 3, torch.float32),
            "gate_probe/arithmetic_dtype": str(self.records[0][1].dtype),
            "gate_probe/exact_dtype_audit": True,
            "collection/routing_mode": "scope_bias" if biased else "native_only",
        }


class DtypeFullCapture(DtypeResponse, FullHBCapture):
    pass


class DtypeScopeCapture(DtypeResponse, ScopeBiasCapture):
    pass


class GateProbePolicy(FullHBPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.metadata.update(gate_probe_protocol="local_gate_probe.v1", scope_bias_supported=True,
                             gate_probe_max_abs=.30, gate_probe_dtype_audit=True)

    def infer(self, observation):
        request = dict(observation)
        enabled = request.pop(CAPTURE_KEY, False)
        bias = request.pop(BIAS_KEY, None)
        if not enabled:
            if bias is not None:
                raise ValueError("Gate bias requires full capture")
            return self.policy.infer(request)
        if not request.get("routing/capture"):
            raise ValueError("Full gate probe needs independent top-k capture")
        if bias is None:
            capture = DtypeFullCapture(self.policy._routing_layers)
        else:
            bias = np.asarray(bias, np.float32)
            if not np.isfinite(bias).all() or np.max(np.abs(bias)) > .3000001:
                raise ValueError("Gate probe magnitude bound exceeded")
            capture = DtypeScopeCapture(self.policy._routing_layers, bias)
        with capture, FullASCapture(self.as_layers) as as_capture:
            result = self.policy.infer(request)
        result.update(capture.response())
        result.update(as_capture.response())
        for full, wire in ((EFFECTIVE_IDS_KEY, "routing/expert_ids"),
                           (EFFECTIVE_WEIGHTS_KEY, "routing/expert_weights")):
            if not np.array_equal(result[full][:, :, 1:].transpose(1, 0, 2, 3), result[wire]):
                raise RuntimeError("Full dispatch and independent wire hook disagree")
        return result
