"""Full-scope HB gate bias (all eight layers, state token included) for the internal-correction probes."""

from __future__ import annotations

import numpy as np

from collection_routes import HB_LAYERS, FullHBCapture, FullASCapture, FullHBPolicy, CAPTURE_KEY, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from v8_feature_control import FeatureCapture, EFFECTIVE_PROBS, NATIVE_PROBS

PROTOCOL = "moe_control.scope_bias.v1"
BIAS_KEY = "scope_bias/logit_bias"
MAX_ABS = 8.0


class ScopeBiasCapture(FeatureCapture):
    """Same real gate intervention as the v8 capture, without the layer/token scope restriction."""

    def __init__(self, layers, bias):
        FullHBCapture.__init__(self, layers)
        self.bias = np.asarray(bias, np.float32)
        if self.bias.shape != (8, 10, 11, 32) or not np.isfinite(self.bias).all() or np.abs(self.bias).max() > MAX_ABS:
            raise ValueError("Scope bias violates shape or magnitude bounds")

    def response(self):
        result = super().response()
        result["collection/routing_mode"] = "scope_bias"
        return result


class ScopeBiasPolicy(FullHBPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.metadata.update(scope_bias_protocol=PROTOCOL, scope_bias_supported=True, scope_bias_max_abs=MAX_ABS)

    def infer(self, observation):
        if BIAS_KEY not in observation:
            return super().infer(observation)
        request = dict(observation)
        bias = request.pop(BIAS_KEY)
        if not request.pop(CAPTURE_KEY, False) or not request.get("routing/capture"):
            raise ValueError("Scope bias requires full dispatch capture")
        with ScopeBiasCapture(self.policy._routing_layers, bias) as capture, FullASCapture(self.as_layers) as as_capture:
            result = self.policy.infer(request)
        result.update(capture.response())
        result.update(as_capture.response())
        for full, independent in ((EFFECTIVE_IDS_KEY, "routing/expert_ids"), (EFFECTIVE_WEIGHTS_KEY, "routing/expert_weights")):
            np.testing.assert_array_equal(result[full][:, :, 1:].transpose(1, 0, 2, 3), result[independent])
        return result
