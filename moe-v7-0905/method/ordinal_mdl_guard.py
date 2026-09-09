#!/usr/bin/env python3
"""Prefix-causal MoE change candidates without threshold calibration or fitting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from intrinsic_guard_monitor import BACK, IntrinsicGuardMonitor


SCHEMA = "himoe.ordinal_mdl_guard.v1"
MODELS = ("stable", "freeze", "turbulence")


class KTCode:
    """Exact Dirichlet-half mixture code, cached by integer symbol counts."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.category = np.asarray(
            [(math.lgamma(n + 0.5) - math.lgamma(0.5)) / math.log(2)
             for n in range(capacity + 1)]
        )
        self.total = np.asarray(
            [(math.lgamma(n + 1.5) - math.lgamma(1.5)) / math.log(2)
             for n in range(capacity + 1)]
        )

    def bits(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts)
        if counts.shape[-1:] != (3,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("counts must be integers with a final ternary axis")
        totals = counts.sum(axis=-1)
        if np.any(counts < 0) or np.any(totals > self.capacity):
            raise ValueError("counts exceed code capacity or are negative")
        return self.total[totals] - self.category[counts].sum(axis=-1)


def _decode_prefix(cumulative: np.ndarray, code: KTCode) -> tuple[np.ndarray, np.ndarray]:
    """Return best gain and split for two branches, shaped [batch, branch]."""
    batch, count = cumulative.shape[:2]
    gains = np.full((batch, 2), -np.inf)
    splits = np.full((batch, 2), -1, dtype=np.int32)
    if count < 2:
        return gains, splits
    whole = cumulative[:, -1]
    before = cumulative[:, :-1]
    after = whole[:, None] - before
    saving = code.bits(whole)[:, None] - code.bits(before) - code.bits(after)
    k = np.arange(1, count, dtype=np.int32)
    prefix_signed = before[..., 2] - before[..., 0]
    suffix_signed = after[..., 2] - after[..., 0]
    # Integer cross multiplication avoids floating comparisons of ordinal means.
    oriented = (suffix_signed > 0) & (
        suffix_signed.astype(np.int64) * k[None, :, None]
        > prefix_signed.astype(np.int64) * (count - k)[None, :, None]
    )
    location_and_model_cost = np.log2(k.astype(np.float64) * (k + 1)) + 1.0
    branch = np.stack((saving[..., 0], saving[..., 1] + saving[..., 2]), axis=1)
    eligible = np.stack((oriented[..., 0], oriented[..., 1] & oriented[..., 2]), axis=1)
    branch = np.where(eligible, branch - location_and_model_cost, -np.inf)
    best = branch.argmax(axis=-1)
    gains = np.take_along_axis(branch, best[..., None], axis=-1)[..., 0]
    splits = np.where(np.isfinite(gains), best + 1, -1).astype(np.int32)
    return gains, splits


def _oriented_features(mobility: np.ndarray, acceleration: np.ndarray,
                       periodicity: np.ndarray) -> np.ndarray:
    return np.stack((-np.median(mobility[..., BACK], axis=-1), acceleration, -periodicity), axis=-1)


def _first(decisions: np.ndarray) -> np.ndarray:
    first = np.full(decisions.shape[0], -1, dtype=np.int32)
    any_decision = decisions.any(axis=1)
    if decisions.shape[1]:
        first[any_decision] = decisions[any_decision].argmax(axis=1)
    return first


def score_feature_arrays(
    layer_mobility: np.ndarray,
    route_acceleration: np.ndarray,
    lag_periodicity: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay cached features. Validity marks present queries, never a horizon feature."""
    mobility = np.asarray(layer_mobility, dtype=np.float64)
    acceleration = np.asarray(route_acceleration, dtype=np.float64)
    periodicity = np.asarray(lag_periodicity, dtype=np.float64)
    valid = np.asarray(valid)
    if mobility.ndim != 3 or mobility.shape[-1] != 8:
        raise ValueError("mobility must have shape [episode, query, 8]")
    shape = mobility.shape[:2]
    if acceleration.shape != shape or periodicity.shape != shape or valid.shape != shape:
        raise ValueError("feature arrays and validity must align")
    if valid.dtype != np.dtype(bool):
        raise ValueError("validity must be Boolean")
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("valid queries must form a contiguous prefix")
    available = valid[:, 2:]
    if (not np.isfinite(mobility[:, 2:][available]).all()
            or not np.isfinite(acceleration[:, 2:][available]).all()
            or not np.isfinite(periodicity[:, 2:][available]).all()):
        raise ValueError("all route features must be finite from q2 onward")
    if np.any(mobility[valid] < 0) or np.any(acceleration[valid] < 0):
        raise ValueError("mobility and acceleration must be nonnegative")

    values = _oriented_features(mobility, acceleration, periodicity)
    episodes, queries = shape
    code = KTCode(queries)
    cumulative = np.zeros((episodes, max(queries - 3, 0), 3, 3), dtype=np.int32)
    gains = np.full((*shape, 2), -np.inf)
    candidates = np.full((*shape, 2), -1, dtype=np.int32)
    for query in range(3, queries):
        history = values[:, 2:query]
        middle = (history.shape[1] - 1) // 2
        reference = np.partition(history, middle, axis=1)[:, middle]
        symbols = (values[:, query] > reference).astype(np.int8) - (
            values[:, query] < reference
        ).astype(np.int8) + 1
        index = query - 3
        cumulative[:, index] = symbols[..., None] == np.arange(3)
        if index:
            cumulative[:, index] += cumulative[:, index - 1]
        current_gain, split = _decode_prefix(cumulative[:, :index + 1], code)
        gains[:, query] = np.where(valid[:, query, None], current_gain, -np.inf)
        candidates[:, query] = np.where(
            valid[:, query, None] & (split >= 0), split + 3, -1
        )

    branch = gains.argmax(axis=-1)
    gain = np.take_along_axis(gains, branch[..., None], axis=-1)[..., 0]
    candidate = np.take_along_axis(candidates, branch[..., None], axis=-1)[..., 0]
    selected = np.where(gain > 0, branch + 1, 0).astype(np.uint8)
    first = _first(selected != 0)
    first_keypoint = np.full(episodes, -1, dtype=np.int32)
    detected = first >= 0
    first_keypoint[detected] = candidate[np.flatnonzero(detected), first[detected]]
    return {
        "freeze_gain_bits": gains[..., 0],
        "turbulence_gain_bits": gains[..., 1],
        "gain_bits": gain,
        "candidate_query": candidate,
        "selected_model": selected,
        "first_freeze_query": _first(gains[..., 0] > 0),
        "first_turbulence_query": _first(gains[..., 1] > 0),
        "first_alarm_query": first,
        "first_keypoint_query": first_keypoint,
    }


class OrdinalFeatureMonitor:
    """Streaming decoder; this interface receives extracted MoE features only."""

    def __init__(self) -> None:
        self.query = -1
        self.first_alarm_query = -1
        self.first_keypoint_query = -1
        self._history: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._code = KTCode(1)

    def update(self, layer_mobility: np.ndarray, route_acceleration: float,
               lag_periodicity: float) -> dict[str, Any]:
        mobility = np.asarray(layer_mobility, dtype=np.float64)
        if mobility.shape != (8,):
            raise ValueError("layer mobility must have shape [8]")
        acceleration = float(route_acceleration)
        periodicity = float(lag_periodicity)
        next_query = self.query + 1
        if next_query >= 2 and (not np.isfinite(mobility).all()
                                or not np.isfinite((acceleration, periodicity)).all()):
            raise ValueError("all route features must be finite from q2 onward")
        if np.any(mobility < 0) or acceleration < 0:
            raise ValueError("mobility and acceleration must be nonnegative")
        self.query = next_query
        gains = np.full(2, -np.inf)
        splits = np.full(2, -1, dtype=np.int32)
        if self.query >= 2:
            current = _oriented_features(mobility, np.asarray(acceleration), np.asarray(periodicity))
            if self._history:
                history = np.asarray(self._history)
                middle = (len(history) - 1) // 2
                reference = np.partition(history, middle, axis=0)[middle]
                symbols = (current > reference).astype(np.int8) - (current < reference).astype(np.int8) + 1
                counts = (symbols[:, None] == np.arange(3)).astype(np.int32)
                if self._counts:
                    counts += self._counts[-1]
                self._counts.append(counts)
                if len(self._counts) > self._code.capacity:
                    self._code = KTCode(max(len(self._counts), 2 * self._code.capacity))
                batch_gains, batch_splits = _decode_prefix(np.asarray(self._counts)[None], self._code)
                gains, splits = batch_gains[0], batch_splits[0]
            self._history.append(current)
        branch = int(gains.argmax())
        gain = float(gains[branch])
        candidate = int(splits[branch] + 3) if splits[branch] >= 0 else -1
        selected = branch + 1 if gain > 0 else 0
        if selected and self.first_alarm_query < 0:
            self.first_alarm_query = self.query
            self.first_keypoint_query = candidate
        return {
            "query": self.query,
            "freeze_gain_bits": float(gains[0]),
            "turbulence_gain_bits": float(gains[1]),
            "gain_bits": gain,
            "candidate_query": candidate,
            "selected_model": selected,
            "model": MODELS[selected],
            "alarm_now": bool(selected),
            "alarm": self.first_alarm_query >= 0,
            "first_alarm_query": self.first_alarm_query,
            "first_keypoint_query": self.first_keypoint_query,
        }


class OrdinalMDLGuard:
    """Drop-in raw-router monitor with no profile, fitted weights, or cutoff."""

    def __init__(self) -> None:
        self._features = OrdinalFeatureMonitor()
        self._previous_action_route: np.ndarray | None = None
        self._back_action_history: list[np.ndarray] = []

    def update(self, hb_router_probs: np.ndarray) -> dict[str, Any]:
        raw = np.asarray(hb_router_probs)
        if raw.shape != (8, 10, 11, 32):
            raise ValueError("router probability must have shape [8, 10, 11, 32]")
        if (not np.isfinite(raw).all() or np.any(raw < 0)
                or np.any(raw.sum(axis=-1, dtype=np.float64) <= 0)):
            raise ValueError("router probabilities must be finite, nonnegative, and nonempty")
        mobility, acceleration, periodicity, final_action = IntrinsicGuardMonitor._query_features(
            raw, self._previous_action_route, self._back_action_history
        )
        record = self._features.update(mobility, acceleration, periodicity)
        self._previous_action_route = final_action
        self._back_action_history.append(final_action[BACK].reshape(40, 32))
        record.update(layer_mobility=mobility, route_acceleration=acceleration,
                      lag_periodicity=periodicity)
        return record
