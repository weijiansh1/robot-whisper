"""Causal ST/AC confirmation with the original frozen v8.2 alarm preserved."""

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

if __package__:
    from .encoder import EncoderConfig, FEATURE_NAMES, ReadoutEncoder, query_readout
    from .guard import (PeakBank, combine_tails, dynamics_scores, intrinsic_score_arrays,
                        persistent, prefix_peak, raw_legacy_features, v8_streams)
else:
    from encoder import EncoderConfig, FEATURE_NAMES, ReadoutEncoder, query_readout
    from guard import (PeakBank, combine_tails, dynamics_scores, intrinsic_score_arrays,
                       persistent, prefix_peak, raw_legacy_features, v8_streams)

SCHEMA = "himoe.state_action_confirmation.v1"
GATE_ALPHA = .10
RELATION_NAMES = ("gap", "state_relative_low", "state_absolute_low", "raw_gap_low")


def state_readout(probability, previous_root=None):
    """Match the audited ST cache's float32 Bhattacharyya implementation."""
    p = np.array(probability, dtype=np.float32, copy=True)
    if p.shape != (8, 10, 11, 32):
        raise ValueError("expected router probabilities [8,10,11,32]")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("invalid routing probabilities")
    p /= p.sum(-1, keepdims=True)
    root = np.sqrt(p[:, -1, 0])
    movement = np.full(8, np.nan, np.float32)
    if previous_root is not None:
        previous_root = np.asarray(previous_root)
        if previous_root.shape != root.shape or not np.isfinite(previous_root).all() or (previous_root < 0).any():
            raise ValueError("invalid previous state root")
        coefficient = np.clip((root * previous_root).sum(-1), 0., 1.)
        movement = np.sqrt(np.clip(1. - coefficient, 0., 1.))
    return movement, root


def relation_features(current_ac, current_st, reference_ac, reference_st, config):
    ac = np.log((current_ac + config.epsilon) / (reference_ac + config.epsilon))
    st = np.log((current_st + config.epsilon) / (reference_st + config.epsilon))
    tolerance = 32 * np.finfo(np.float64).eps
    ac, st = np.where(np.abs(ac) <= tolerance, 0., ac), np.where(np.abs(st) <= tolerance, 0., st)
    return np.stack((np.median((ac - st)[..., 4:], axis=-1),
                     -np.median(st[..., 4:], axis=-1),
                     -np.median(current_st[..., 4:], axis=-1),
                     np.median((current_ac - current_st)[..., 4:], axis=-1)), axis=-1)


def encode_relations(action, state, valid, config=None):
    config = config or EncoderConfig()
    action, state = np.asarray(action, np.float64), np.asarray(state, np.float64)
    valid = np.asarray(valid, bool)
    if action.shape != state.shape or action.shape != (*valid.shape, 8) or valid.ndim != 2:
        raise ValueError("expected aligned [episode,query,8] mobility arrays")
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe observed prefixes")
    for values in (action, state):
        observed = values[:, 1:][valid[:, 1:]]
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError("invalid mobility after the first query")
    result = np.full((*valid.shape, len(RELATION_NAMES)), np.nan, np.float64)
    if valid.shape[1] <= config.first_query:
        return result.astype(np.float32)
    ac, st = [np.where(valid[..., None], values, np.nan) for values in (action, state)]
    ref_ac, ref_st = [np.median(values[:, 1:config.reference_count + 1], axis=1) for values in (ac, st)]
    for q in range(config.first_query, valid.shape[1]):
        current_ac, current_st = [values[:, q - config.smooth_width + 1:q + 1].mean(1) for values in (ac, st)]
        result[:, q] = relation_features(current_ac, current_st, ref_ac, ref_st, config)
    result[~valid] = np.nan
    return result.astype(np.float32)


class RelationEncoder:
    def __init__(self):
        self.config = EncoderConfig()
        self.reset()

    def reset(self):
        self.query = -1
        self.reference = None
        self.opening = []
        self.recent = deque(maxlen=self.config.smooth_width)

    def update(self, action, state):
        values = np.stack((action, state)).astype(np.float64)
        if values.shape != (2, 8):
            raise ValueError("expected two eight-layer readouts")
        if self.query >= 0 and (not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError("invalid mobility after the first query")
        self.query += 1
        self.recent.append(values)
        if 1 <= self.query <= self.config.reference_count:
            self.opening.append(values)
        if self.query == self.config.reference_count:
            self.reference = np.median(self.opening, axis=0)
            self.opening.clear()
        if self.query < self.config.first_query:
            return np.full(len(RELATION_NAMES), np.nan, np.float32)
        current = np.asarray(self.recent).mean(0)
        return relation_features(*current, *self.reference, self.config).astype(np.float32)


def confirmed_relations(features, valid):
    return {name: persistent(np.where(valid, features[..., i], np.nan), 2)
            for i, name in enumerate(RELATION_NAMES)}


def confirmation_tail(ac_tails, gap_tail, gap_score):
    ac = combine_tails(ac_tails, "dynamics_only")
    ready = np.isfinite(gap_tail) & np.isfinite(gap_score)
    accepted = ready & (np.asarray(gap_score) > 0) & (np.asarray(gap_tail) <= GATE_ALPHA)
    # Rejection remains a finite non-alarm; unavailable ST remains unavailable.
    return np.where(ready & np.isfinite(ac), np.where(accepted, ac, 1.), np.nan)


def frozen_triggers(cache, extra, profile):
    """Replay only the published v8.2 rule, including its original query slope."""
    valid = np.asarray(cache["valid"], bool)
    if valid.shape[1] < 7:
        return np.zeros(valid.shape, bool)
    old = intrinsic_score_arrays(cache["mobility"], cache["acceleration"],
                                 cache["periodicity"], profile["periodicity_scale"])
    freeze = old["freeze"] > profile["freeze_threshold"]
    turbulence = ((prefix_peak(old["acceleration_persistent"]) > profile["acceleration_threshold"])
                  & (prefix_peak(old["periodicity_persistent"]) > profile["periodicity_threshold"]))
    extra = v8_streams(extra, valid)
    q = np.arange(valid.shape[1])[None]
    new = [persistent(extra[..., i] + profile["slope"] * q, 2) >= profile[name]
           for i, name in enumerate(("frontback_threshold", "curvature_threshold"))]
    return valid & (freeze | turbulence | new[0] | new[1])


class StateActionGuardMonitor:
    """One-query raw-routing API; no task, outcome, progress, or length input."""

    def __init__(self, profile_path, checkpoint, kind="task_init", alpha=.01):
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get("schema") != SCHEMA or self.profile.get("encoder_config") != asdict(EncoderConfig()):
            raise ValueError("incompatible state/action profile")
        if checkpoint not in self.profile["checkpoints"] or kind not in self.profile["banks"]:
            raise ValueError("unsupported checkpoint or calibration")
        if not np.isfinite(alpha) or not 0 < alpha < 1 or self.profile["gate_alpha"] != GATE_ALPHA:
            raise ValueError("invalid budget or confirmation threshold")
        self.banks = {name: PeakBank.from_list(bank) for name, bank in self.profile["banks"][kind].items()}
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.query = -1
        self.previous_action = self.previous_state = None
        self.action_encoder, self.relation_encoder = ReadoutEncoder(), RelationEncoder()
        self.action_recent, self.relation_recent = deque(maxlen=2), deque(maxlen=2)
        self.legacy_history = deque(maxlen=4)
        self.legacy = {name: [] for name in ("mobility", "acceleration", "periodicity")}
        self.extra = []
        self.first = dict(v82=-1, ac=-1, confirmed=-1, combined=-1)

    def update(self, probability):
        readout, final = query_readout(probability, self.previous_action)
        state, state_root = state_readout(probability, self.previous_state)
        m, a, p, extra, _ = raw_legacy_features(probability, self.previous_action, list(self.legacy_history))
        self.query += 1
        self.previous_action, self.previous_state = final, state_root
        encoded = self.action_encoder.update(readout)
        relation = self.relation_encoder.update(readout[:8], state)
        self.action_recent.append(encoded.values.astype(np.float32))
        self.relation_recent.append(relation)
        scores = {name: np.nan for name in ("acceleration", "decoupling", *RELATION_NAMES)}
        if len(self.action_recent) == 2:
            window = np.asarray(self.action_recent)
            for name, field in (("acceleration", "acceleration_log_ratio"), ("decoupling", "recent_decoupling")):
                scores[name] = float(window[:, FEATURE_NAMES.index(field)].min())
            for i, name in enumerate(RELATION_NAMES):
                scores[name] = float(np.asarray(self.relation_recent)[:, i].min())
        tails = {name: float(self.banks[name].tail(value)) for name, value in scores.items()}
        ac_tail = float(combine_tails(tails, "dynamics_only"))
        gated_tail = float(confirmation_tail(tails, tails["gap"], scores["gap"]))
        for name, value in zip(self.legacy, (m, a, p)):
            self.legacy[name].append(value)
        self.extra.append(extra)
        self.legacy_history.append(final[4:].reshape(40, 32))
        cache = {name: np.asarray(values, np.float32)[None] for name, values in self.legacy.items()}
        cache["valid"] = np.ones((1, self.query + 1), bool)
        original = bool(frozen_triggers(cache, np.asarray(self.extra, np.float32)[None], self.profile["frozen"])[0, -1])
        hits = dict(v82=original, ac=np.isfinite(ac_tail) and ac_tail <= self.alpha,
                    confirmed=np.isfinite(gated_tail) and gated_tail <= self.alpha)
        hits["combined"] = hits["v82"] or hits["confirmed"]
        for name, hit in hits.items():
            if hit and self.first[name] < 0:
                self.first[name] = self.query
        return dict(query=self.query, scores=scores, tails=tails, ac_tail=ac_tail,
                    confirmed_tail=gated_tail, trigger_now=bool(hits["combined"]),
                    confirmed_trigger_now=bool(hits["confirmed"]),
                    ever_alarm=self.first["combined"] >= 0, first_alarm_query=self.first["combined"],
                    first=dict(self.first), state_mobility=state, relation_features=relation)
