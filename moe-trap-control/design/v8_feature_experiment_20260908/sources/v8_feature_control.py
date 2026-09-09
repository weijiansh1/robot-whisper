"""Causal v8 feature targets and bounded, actually dispatched gate interventions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_routes import (HB_LAYERS, FullHBCapture, FullASCapture, FullHBPolicy,
    CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
    EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)

sys.path.insert(0, str(HERE.parent / "moe-v7-0905/method"))
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

PROTOCOL = "moe_control.v8_feature.v1"
BIAS_KEY = "v8_control/logit_bias"
EFFECTIVE_PROBS = "v8_control/effective_probs_fp32"
NATIVE_PROBS = "v8_control/native_probs_fp32"
ARMS = {
    "native": dict(operator="native", strength=0., duration=0),
    "balance": dict(operator="balance", strength=1., duration=5),
    "curvature": dict(operator="curvature", strength=1., duration=5),
    "mobility": dict(operator="mobility", strength=1., duration=5),
    "combined_half": dict(operator="combined", strength=.5, duration=5),
    "combined": dict(operator="combined", strength=1., duration=5),
    "random_half": dict(operator="random", strength=.5, duration=5),
    "random": dict(operator="random", strength=1., duration=5),
}


def normalize(p):
    p = np.maximum(np.asarray(p, np.float32), 0)
    if p.shape != (8, 10, 11, 32) or not np.isfinite(p).all() or (p.sum(-1) <= 0).any():
        raise ValueError("Invalid complete HB probabilities")
    return p / p.sum(-1, keepdims=True)


def flow_features(probability):
    p = normalize(probability)
    speed = (np.linalg.norm(np.diff(np.sqrt(p[:, :, 1:]), axis=1), axis=-1) /
             np.sqrt(2.0)).mean(-1).astype(np.float32)
    path = speed.sum(-1)
    inversion = -np.log(np.maximum(path[:4].mean(), 1e-12) / np.maximum(path[4:].mean(), 1e-12))
    curvature = np.abs(np.diff(speed[4:, [0, 4, 8]], n=2, axis=-1)).mean()
    return np.asarray([inversion, curvature], np.float32), speed


class V8Monitor:
    def __init__(self):
        verify_frozen_alarm()
        self.config = json.loads((HERE / "design/frozen_alarm_comparison_20260908/profiles/parameters.json").read_text())["legacy"]
        self.v7 = IntrinsicGuardMonitor(GlobalIntrinsicProfile(**self.config["v7"]))
        self.raw, self.counts, self.first = [], [0, 0], [-1, -1]
        self.first_alarm = -1

    def update(self, probability):
        status = self.v7.update(probability)
        self.raw.append(flow_features(probability)[0])
        q = self.v7.query
        scores = np.full(2, np.nan, np.float64)
        if q >= 4:
            raw = np.asarray(self.raw, np.float32)
            base = raw[1:5, 1].mean(dtype=np.float32)
            relative = np.log(np.maximum(raw[:, 1], 1e-12) / np.maximum(base, 1e-12))
            scores = np.asarray([raw[-6:, 0].mean(dtype=np.float64), relative[-6:].mean(dtype=np.float64)])
        thresholds = [-self.config["v8_thresholds"]["frontback_flowpath"],
                      self.config["v8_thresholds"]["curvature_3step"]]
        for i in range(2):
            hit = q >= self.config["earliest_v8_head_query"] and scores[i] >= thresholds[i]
            self.counts[i] = self.counts[i] + 1 if hit else 0
            if self.counts[i] >= self.config["v8_confirm"] and self.first[i] < 0:
                self.first[i] = q
        if self.first_alarm < 0 and (status["alarm"] or max(self.first) >= 0):
            self.first_alarm = q
        status.update(v8_first=self.first_alarm, v8_alarm=self.first_alarm >= 0,
                      v8_raw=self.raw[-1], v8_scores=scores,
                      v8_head_first=np.asarray(self.first, np.int32))
        return status


def centered_log(p):
    value = np.log(np.maximum(p, 1e-12))
    return value - value.mean(-1, keepdims=True)


def resample_path(p):
    """Re-time each back-layer path by Hellinger arc length, retaining endpoints."""
    target = p.copy()
    for layer in range(4, 8):
        root = np.sqrt(p[layer, :, 1:]).astype(np.float64)
        length = np.linalg.norm(np.diff(root, axis=0), axis=-1).mean(-1)
        cumulative = np.r_[0., np.cumsum(length)]
        if cumulative[-1] < 1e-12:
            continue
        for k, position in enumerate(np.linspace(0, cumulative[-1], 10)[1:-1], 1):
            j = min(int(np.searchsorted(cumulative, position, side="right")) - 1, 8)
            fraction = (position - cumulative[j]) / max(cumulative[j + 1] - cumulative[j], 1e-12)
            value = (1 - fraction) * root[j] + fraction * root[j + 1]
            square = value * value
            target[layer, k, 1:] = square / square.sum(-1, keepdims=True)
    return target


def make_bias(shadow, previous, operator, strength, seed):
    if operator not in {v["operator"] for v in ARMS.values()} or not 0 <= strength <= 1:
        raise ValueError("Unsupported feature control")
    p = normalize(shadow)
    logp = centered_log(p)
    delta = np.zeros_like(p)
    if operator in ("curvature", "combined", "random"):
        delta += centered_log(resample_path(p)) - logp
    if operator in ("balance", "combined", "random"):
        # Contract denoise variation in back layers, preserving the query mean logits.
        back = logp[4:, :, 1:]
        delta[4:, :, 1:] += -.4 * (back - back.mean(1, keepdims=True))
    if operator in ("mobility", "combined", "random") and previous is not None:
        previous_log = centered_log(normalize(previous))
        direction = logp[4:, -1, 1:] - previous_log[4:, -1, 1:]
        rms = np.sqrt(np.square(direction).mean(-1, keepdims=True))
        # Continue the observed temporal direction; never invent one when it is zero.
        movement = direction * np.minimum(3., .2 / np.maximum(rms, 1e-12))
        delta[4:, :, 1:] += movement[:, None]
    delta[:, :, 0] = 0
    delta -= delta.mean(-1, keepdims=True)
    rms = np.sqrt(np.square(delta).mean(-1, keepdims=True))
    delta *= np.minimum(1., .35 / np.maximum(rms, 1e-12))
    peak = np.max(np.abs(delta), axis=-1, keepdims=True)
    delta *= np.minimum(1., 1. / np.maximum(peak, 1e-12))
    delta *= strength
    if operator == "random":
        rng = np.random.default_rng(seed)
        delta = np.take_along_axis(delta, np.argsort(rng.random(delta.shape), axis=-1), axis=-1)
    return np.ascontiguousarray(delta, dtype=np.float32)


def seed_for(main_id, replicate, index, stream):
    if stream not in ("policy", "environment", "direction") or min(replicate, index) < 0:
        raise ValueError("Invalid feature-control random stream")
    return int(stable_id(PROTOCOL, main_id, replicate, index, stream)[:8], 16)


def noise_for(main_id, replicate, index):
    return np.random.default_rng(seed_for(main_id, replicate, index, "policy")).standard_normal((10, 24)).astype(np.float32)


class FeatureCapture(FullHBCapture):
    def __init__(self, layers, bias):
        super().__init__(layers)
        self.bias = np.asarray(bias, np.float32)
        if (self.bias.shape != (8, 10, 11, 32) or not np.isfinite(self.bias).all() or
                np.abs(self.bias).max() > 1.000001 or np.any(self.bias[:, :, 0] != 0) or
                np.any(self.bias[:4] != 0)):
            raise ValueError("Feature bias violates scope or magnitude bounds")

    def _hook(self, layer):
        def capture(gate, inputs, output):
            import torch
            from torch.nn import functional as F
            value = inputs[0]
            if tuple(value.shape[:2]) != (1, 11):
                raise ValueError("Feature control requires batch=1 and 11 tokens")
            step, slot = divmod(len(self.records), 8)
            if step >= 10 or HB_LAYERS[slot] != layer:
                raise ValueError("Feature hook ordering")
            ids, weights, auxiliary = output
            with torch.no_grad():
                logits = F.linear(value.reshape(-1, value.shape[-1]), gate.weight, None)
                native = logits.softmax(-1)
                bias = torch.as_tensor(self.bias[slot, step], device=logits.device, dtype=logits.dtype)
                effective = (logits + bias).softmax(-1)
                proposed_weights, proposed_ids = effective.topk(4, dim=-1, sorted=False)
                proposed_weights /= proposed_weights.sum(-1, keepdim=True) + 1e-20
                touched = bias.abs().sum(-1, keepdim=True) != 0
                actual_ids = torch.where(touched, proposed_ids, ids)
                actual_weights = torch.where(touched, proposed_weights, weights)
                self.records.append((layer, native.detach().clone(), ids.detach().clone(),
                    weights.detach().clone(), native.topk(4, sorted=False).indices,
                    actual_ids.detach().clone(), actual_weights.detach().clone(), effective.detach().clone()))
            return actual_ids, actual_weights, auxiliary
        return capture

    def response(self):
        import torch
        result = super().response()
        def stack(column):
            return torch.stack([r[column] for r in self.records]).reshape(10, 8, 11, -1).permute(
                1, 0, 2, 3).to(device="cpu", dtype=torch.float32).numpy().copy()
        result[EFFECTIVE_WEIGHTS_KEY] = stack(6)
        result[NATIVE_PROBS], result[EFFECTIVE_PROBS] = stack(1), stack(7)
        selected = np.take_along_axis(result[EFFECTIVE_PROBS], result[EFFECTIVE_IDS_KEY].astype(np.int64), -1)
        np.testing.assert_allclose(result[EFFECTIVE_WEIGHTS_KEY], selected / selected.sum(-1, keepdims=True), atol=1e-5, rtol=0)
        result["collection/routing_mode"] = "v8_feature_bias"
        return result


class V8FeaturePolicy(FullHBPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.metadata.update(v8_feature_protocol=PROTOCOL, v8_feature_bias_supported=True)

    def infer(self, observation):
        if BIAS_KEY not in observation:
            return super().infer(observation)
        request = dict(observation)
        bias = request.pop(BIAS_KEY)
        if not request.pop(CAPTURE_KEY, False) or not request.get("routing/capture"):
            raise ValueError("Feature intervention requires full dispatch capture")
        with FeatureCapture(self.policy._routing_layers, bias) as capture, FullASCapture(self.as_layers) as as_capture:
            result = self.policy.infer(request)
        result.update(capture.response())
        result.update(as_capture.response())
        for full, independent in ((EFFECTIVE_IDS_KEY, "routing/expert_ids"),
                                  (EFFECTIVE_WEIGHTS_KEY, "routing/expert_weights")):
            np.testing.assert_array_equal(result[full][:, :, 1:].transpose(1, 0, 2, 3), result[independent])
        return result
