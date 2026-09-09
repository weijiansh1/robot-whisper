"""Causal, task-independent encoding of MoE dynamics, not failure predictions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


LAYER_NAMES = (2, 3, 4, 5, 12, 13, 14, 15)
FEATURE_NAMES = (
    "back_mobility", "back_flow_path", "flow_acceleration", "back_token_diversity",
    "mobility_log_ratio", "acceleration_log_ratio", "path_log_ratio", "diversity_log_ratio",
    "front_back_path_log_ratio", "decoupling", "both_increasing", "both_decreasing",
    "outer_up_inner_down", "path_decoupling", "layer_agreement", "token_diversity_contraction",
    "recent_decoupling", "persistent_decoupling", "decoupling_occupancy",
    "decoupling_drop", "mobility_rebound", "routing_recovery",
)
LAYER_FEATURE_NAMES = ("mobility_log_ratio", "path_log_ratio", "diversity_log_ratio")
MOBILITY, PATH, ACCELERATION, DIVERSITY = slice(0, 8), slice(8, 16), 16, slice(17, 25)
INSTANT_COUNT = 16


@dataclass(frozen=True)
class EncoderConfig:
    reference_count: int = 4
    smooth_width: int = 3
    history_width: int = 4
    epsilon: float = 1e-6

    def __post_init__(self):
        for value in (self.reference_count, self.smooth_width, self.history_width):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("window sizes must be positive integers")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    @property
    def first_query(self):
        return self.reference_count + self.smooth_width


@dataclass(frozen=True)
class Encoding:
    values: np.ndarray
    per_layer: np.ndarray

    @property
    def ready(self):
        return bool(np.isfinite(self.values[:INSTANT_COUNT]).all())

    def as_dict(self):
        return dict(zip(FEATURE_NAMES, self.values.tolist()))


def query_readout(router_probability, previous_final=None):
    """Return 25 readouts and final action-token routes from one raw query."""
    p = np.array(router_probability, dtype=np.float32, copy=True)
    if p.shape != (8, 10, 11, 32):
        raise ValueError("expected router probabilities [8,10,11,32]")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("probabilities must be finite, nonnegative, with positive row sums")
    p /= p.sum(-1, keepdims=True)
    action = p[:, :, 1:]
    root = np.sqrt(action)
    final = action[:, -1]
    mobility = np.full(8, np.nan, np.float32)
    if previous_final is not None:
        previous = np.asarray(previous_final)
        if previous.shape != final.shape or not np.isfinite(previous).all() or (previous < 0).any():
            raise ValueError("invalid previous final routing")
        mobility = (np.linalg.norm(root[:, -1] - np.sqrt(previous), axis=-1) / np.sqrt(2)).mean(-1)
    speed = (np.linalg.norm(np.diff(root, axis=1), axis=-1) / np.sqrt(2)).mean(-1)
    path = speed.sum(-1)
    second = root[4:, 2:] - 2 * root[4:, 1:-1] + root[4:, :-2]
    acceleration = np.linalg.norm(second, axis=-1).mean() / np.sqrt(2)
    load = final.mean(1)
    entropy = -(final * np.log(np.maximum(final, 1e-30))).sum(-1).mean(-1)
    load_entropy = -(load * np.log(np.maximum(load, 1e-30))).sum(-1)
    diversity = np.maximum(load_entropy - entropy, 0) / np.log(32)
    packed = np.concatenate((mobility, path, [acceleration], diversity)).astype(np.float64)
    return packed, final.copy()


def instant_features(current, reference, config):
    """Shared vectorized primitive, using only a frozen reference and current window."""
    ratio = np.log((current + config.epsilon) / (reference + config.epsilon))
    # Mean and median reductions of identical values can differ by a few ulps.
    ratio = np.where(np.abs(ratio) <= 32 * np.finfo(np.float64).eps, 0., ratio)
    mobility = np.median(ratio[..., MOBILITY][..., 4:], axis=-1)
    acceleration = ratio[..., ACCELERATION]
    path = np.median(ratio[..., PATH][..., 4:], axis=-1)
    diversity = np.median(ratio[..., DIVERSITY][..., 4:], axis=-1)
    slower, faster = np.maximum(-mobility, 0), np.maximum(mobility, 0)
    inner_up, inner_down = np.maximum(acceleration, 0), np.maximum(-acceleration, 0)
    layer_joint = np.minimum(np.maximum(-ratio[..., MOBILITY], 0), np.maximum(ratio[..., PATH], 0))
    layer_support = ((ratio[..., MOBILITY] < 0) & (ratio[..., PATH] > 0)).mean(-1)
    values = np.stack((
        np.median(current[..., MOBILITY][..., 4:], axis=-1),
        current[..., PATH][..., 4:].mean(-1), current[..., ACCELERATION],
        current[..., DIVERSITY][..., 4:].mean(-1),
        mobility, acceleration, path, diversity,
        np.log((current[..., PATH][..., :4].mean(-1) + config.epsilon)
               / (current[..., PATH][..., 4:].mean(-1) + config.epsilon)),
        np.minimum(slower, inner_up), np.minimum(faster, inner_up),
        np.minimum(slower, inner_down), np.minimum(faster, inner_down),
        np.median(layer_joint[..., 4:], axis=-1), layer_support, np.maximum(-diversity, 0),
    ), axis=-1)
    layer = np.stack((ratio[..., MOBILITY], ratio[..., PATH], ratio[..., DIVERSITY]), axis=-1)
    return values, layer


def temporal_features(history, width):
    """Two adjacent finite windows; a decline denotes routing relief, not task success."""
    shape = history.shape[:-2]
    result = np.full((*shape, 6), np.nan, np.float64)
    if history.shape[-2] < width:
        return result
    recent = history[..., -width:, 9]
    good = np.isfinite(recent).all(-1)
    result[..., 0] = np.where(good, recent.mean(-1), np.nan)
    result[..., 1] = np.where(good, recent.min(-1), np.nan)
    result[..., 2] = np.where(good, (recent > 0).mean(-1), np.nan)
    if history.shape[-2] >= 2 * width:
        earlier = history[..., -2 * width:-width, 9]
        before_m = history[..., -2 * width:-width, 4]
        now_m = history[..., -width:, 4]
        good &= np.isfinite(earlier).all(-1) & np.isfinite(before_m).all(-1) & np.isfinite(now_m).all(-1)
        drop = np.maximum(earlier.mean(-1) - recent.mean(-1), 0)
        rebound = np.maximum(now_m.mean(-1) - before_m.mean(-1), 0)
        result[..., 3] = np.where(good, drop, np.nan)
        result[..., 4] = np.where(good, rebound, np.nan)
        result[..., 5] = np.where(good, np.minimum(drop, rebound), np.nan)
    return result


class ReadoutEncoder:
    """Bounded-memory online encoder for aligned, current-query readouts."""

    def __init__(self, config=None):
        self.config = config or EncoderConfig()
        self.reset()

    def reset(self):
        self._seen = 0
        self._reference_rows = []
        self._reference = None
        self._recent = deque(maxlen=self.config.smooth_width)
        self._history = deque(maxlen=2 * self.config.history_width)

    def update(self, readout):
        x = np.array(readout, dtype=np.float64, copy=True)
        if x.shape != (25,):
            raise ValueError("expected 25 current-query readouts")
        required = x if self._seen else x[8:]
        if not np.isfinite(required).all() or (required < 0).any():
            raise ValueError("valid readouts must be finite and nonnegative; only initial mobility may be missing")
        q = self._seen
        self._seen += 1
        self._recent.append(x)
        if 1 <= q <= self.config.reference_count:
            self._reference_rows.append(x)
        if q == self.config.reference_count:
            self._reference = np.median(self._reference_rows, axis=0)
            self._reference_rows.clear()
        empty = Encoding(np.full(len(FEATURE_NAMES), np.nan), np.full((8, 3), np.nan))
        if q < self.config.first_query:
            return empty
        current = np.asarray(self._recent).mean(0)
        instant, layer = instant_features(current, self._reference, self.config)
        self._history.append(instant)
        temporal = temporal_features(np.asarray(self._history), self.config.history_width)
        return Encoding(np.concatenate((instant, temporal)), layer)


class RoutingDynamicsEncoder:
    """Public raw-routing API. It takes no task, outcome, progress, or duration input."""

    def __init__(self, config=None):
        self._encoder = ReadoutEncoder(config)
        self._previous_final = None

    def reset(self):
        self._encoder.reset()
        self._previous_final = None

    def update(self, hb_router_probs):
        readout, final = query_readout(hb_router_probs, self._previous_final)
        encoded = self._encoder.update(readout)
        self._previous_final = final
        return encoded


def encode_readouts(readouts, valid=None, config=None):
    """Batch equivalent of ReadoutEncoder; padding is excluded, never encoded as zero."""
    config = config or EncoderConfig()
    x = np.array(readouts, dtype=np.float64, copy=True)
    if x.ndim != 3 or x.shape[-1] != 25 or x.shape[1] < 1:
        raise ValueError("expected [episode, query, 25] readouts")
    n, qmax, _ = x.shape
    if valid is None:
        valid = np.ones((n, qmax), bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (n, qmax) or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe contiguous observed prefixes")
    if not np.isfinite(x[..., 8:][valid]).all() or (x[..., 8:][valid] < 0).any():
        raise ValueError("nonfinite or negative observed readout")
    if not np.isfinite(x[:, 1:, :8][valid[:, 1:]]).all() or (x[:, 1:, :8][valid[:, 1:]] < 0).any():
        raise ValueError("nonfinite or negative observed mobility after the first query")
    x[~valid] = np.nan
    features = np.full((n, qmax, len(FEATURE_NAMES)), np.nan, np.float64)
    layers = np.full((n, qmax, 8, 3), np.nan, np.float64)
    if qmax <= config.first_query:
        return features, layers
    reference = np.median(x[:, 1:config.reference_count + 1], axis=1)
    for q in range(config.first_query, qmax):
        current = x[:, q - config.smooth_width + 1:q + 1].mean(1)
        instant, layer = instant_features(current, reference, config)
        features[:, q, :INSTANT_COUNT] = instant
        layers[:, q] = layer
        start = max(config.first_query, q - 2 * config.history_width + 1)
        features[:, q, INSTANT_COUNT:] = temporal_features(features[:, start:q + 1, :INSTANT_COUNT], config.history_width)
    features[~valid], layers[~valid] = np.nan, np.nan
    return features, layers
