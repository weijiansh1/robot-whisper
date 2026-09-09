"""Train-free fusion of a v8.2 routing guard and causal routing dynamics."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .encoder import EncoderConfig, FEATURE_NAMES, RoutingDynamicsEncoder
else:
    from encoder import EncoderConfig, FEATURE_NAMES, RoutingDynamicsEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from monitor import intrinsic_score_arrays, persistent, prefix_peak, scale_values, v8_streams
from intrinsic_guard_monitor import IntrinsicGuardMonitor, normalize_probability

SCHEMA = "himoe.routing_dynamics_guard.v1"
BRANCHES = ("v82", "acceleration", "decoupling")
METHOD_BRANCHES = {
    "v82_reference": ("v82",),
    "dynamics_only": ("acceleration", "decoupling"),
    "v82_integrated": BRANCHES,
}
HEADS = ("freeze", "acceleration_persistent", "periodicity_persistent", "frontback", "curvature")
CONFIRMATIONS = 2
WATCH_ALPHA = .1
CLEAR_QUERIES = 2
ALPHAS = (.005, .01, .02, .05)
KINDS = ("episode", "task_init")


def v82_scores(cache, raw, profile):
    """Only the existing v8.2 routing score; no other historical baselines."""
    valid = np.asarray(cache["valid"], bool)
    old = intrinsic_score_arrays(cache["mobility"], cache["acceleration"],
                                 cache["periodicity"], profile["periodicity_scale"])
    freeze, acc, period = [scale_values(old[name], profile["heads"][name]) for name in HEADS[:3]]
    turbulence = np.minimum(prefix_peak(acc), prefix_peak(period))
    turbulence[~np.isfinite(turbulence)] = np.nan
    extra = v8_streams(raw, valid)
    q = np.arange(valid.shape[1])[None]
    additions = [persistent(scale_values(extra[:, :, i], profile["heads"][name])
                            + .0015 * q / profile["heads"][name]["scale"], 2)
                 for i, name in enumerate(HEADS[3:])]
    result = np.fmax.reduce([freeze, turbulence, *additions]).astype(np.float32)
    result[~valid] = np.nan
    return result


def dynamics_scores(features, valid):
    features = np.asarray(features)
    valid = np.asarray(valid, bool)
    if features.shape != (*valid.shape, len(FEATURE_NAMES)):
        raise ValueError("features must align with [episode, query] validity")
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe observed prefixes")
    result = {}
    for branch, field in (("acceleration", "acceleration_log_ratio"), ("decoupling", "recent_decoupling")):
        values = np.where(valid, features[:, :, FEATURE_NAMES.index(field)], np.nan)
        result[branch] = persistent(values, CONFIRMATIONS)
    return result


class PeakBank:
    """An empirical upper tail of successful-episode (or group) maxima."""

    def __init__(self, peaks):
        values = np.asarray(peaks, dtype=np.float64)
        if values.ndim != 1 or not len(values) or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("expected nonempty finite or negative-infinity peaks")
        self.peaks = np.sort(values)

    def tail(self, values):
        values = np.asarray(values, dtype=np.float64)
        ranks = len(self.peaks) - np.searchsorted(self.peaks, values, side="left")
        result = (1. + ranks) / (len(self.peaks) + 1)
        return np.where(np.isfinite(values), result, np.nan)

    def threshold(self, alpha):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        rank = int(np.ceil((len(self.peaks) + 1) * (1 - alpha)))
        threshold = float(self.peaks[rank - 1]) if rank <= len(self.peaks) else np.inf
        return threshold, rank

    def to_list(self):
        return [float(x) if np.isfinite(x) else None for x in self.peaks]

    @classmethod
    def from_list(cls, values):
        return cls([-np.inf if x is None else x for x in values])


def combine_tails(tails, method):
    if method not in METHOD_BRANCHES:
        raise ValueError("unknown integration method")
    names = METHOD_BRANCHES[method]
    stack = np.stack([np.asarray(tails[name], float) for name in names])
    finite = np.isfinite(stack)
    if ((stack[finite] <= 0) | (stack[finite] > 1)).any():
        raise ValueError("finite tails must lie in (0, 1]")
    result = np.minimum(1., len(names) * np.where(finite, stack, np.inf).min(axis=0))
    return np.where(finite.any(axis=0), result, np.nan)


def first_trigger(tail, valid, alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    tail, valid = np.asarray(tail, float), np.asarray(valid, bool)
    if tail.shape != valid.shape:
        raise ValueError("tail and validity shape mismatch")
    hit = np.isfinite(tail) & valid & (tail <= alpha)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1).astype(np.int16)


class AlarmState:
    """Hysteresis for current state; a cleared signal never erases an alarm."""

    def __init__(self, alpha=.01, watch_alpha=WATCH_ALPHA, clear_queries=CLEAR_QUERIES):
        if not np.isfinite([alpha, watch_alpha]).all() or not 0 < alpha <= watch_alpha < 1:
            raise ValueError("expected 0 < alpha <= watch_alpha < 1")
        if not isinstance(clear_queries, int) or isinstance(clear_queries, bool) or clear_queries < 1:
            raise ValueError("clear_queries must be a positive integer")
        self.alpha, self.watch_alpha, self.clear_queries = alpha, watch_alpha, clear_queries
        self.reset()

    def reset(self):
        self.query, self.below = 0, 0
        self.state = "NORMAL"
        self.first_alarm_query = self.first_watch_query = -1

    def update(self, tail):
        tail = float(tail)
        if np.isinf(tail) or (np.isfinite(tail) and not 0 < tail <= 1):
            raise ValueError("tail must be missing or lie in (0, 1]")
        before, ready = self.state, bool(np.isfinite(tail))
        trigger, watch = ready and tail <= self.alpha, ready and tail <= self.watch_alpha
        if watch:
            self.below = 0
            if self.first_watch_query < 0:
                self.first_watch_query = self.query
            if self.state == "NORMAL":
                self.state = "WATCH"
        elif ready:
            self.below += 1
            if self.below >= self.clear_queries:
                self.state = "NORMAL"
        else:
            self.below = 0
        if trigger:
            self.state = "ALARM"
            if self.first_alarm_query < 0:
                self.first_alarm_query = self.query
        result = dict(query=self.query, tail=tail, score=-np.log(tail) if ready else np.nan,
                      ready=ready, state=self.state, trigger_now=bool(trigger),
                      alarm_active=self.state == "ALARM", ever_alarm=self.first_alarm_query >= 0,
                      first_alarm_query=self.first_alarm_query, first_watch_query=self.first_watch_query,
                      signal_cleared=before != "NORMAL" and self.state == "NORMAL")
        self.query += 1
        return result


def raw_legacy_features(probability, previous, history):
    mobility, acceleration, periodicity, final = IntrinsicGuardMonitor._query_features(probability, previous, history)
    root = np.sqrt(normalize_probability(probability)[:, :, 1:])
    speed = (np.linalg.norm(np.diff(root, axis=1), axis=-1) / np.sqrt(2.)).mean(-1).astype(np.float32)
    path = speed.sum(-1)
    frontback = -np.log(np.maximum(path[:4].mean(), 1e-12) / np.maximum(path[4:].mean(), 1e-12))
    curvature = np.abs(np.diff(speed[4:][:, [0, 4, 8]], n=2, axis=-1)).mean()
    return mobility, acceleration, periodicity, np.asarray([frontback, curvature], np.float32), final


class RoutingGuardMonitor:
    """Each update takes only current raw routing. Profiles contain A calibration."""

    def __init__(self, profile_path, checkpoint, method="v82_integrated", kind="task_init", alpha=.01):
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get("schema") != SCHEMA:
            raise ValueError("incompatible routing guard profile")
        if checkpoint not in self.profile["checkpoints"]:
            raise ValueError("checkpoint not present in reference data")
        if self.profile["encoder_config"] != asdict(EncoderConfig()):
            raise ValueError("incompatible encoder configuration")
        if self.profile["confirmations"] != CONFIRMATIONS or method not in METHOD_BRANCHES or kind not in KINDS:
            raise ValueError("incompatible rule or calibration kind")
        self.method = method
        self.banks = {name: PeakBank.from_list(self.profile["banks"][kind][name]) for name in BRANCHES}
        self.alert = AlarmState(alpha)
        self.encoder = RoutingDynamicsEncoder()
        self.reset()

    def reset(self):
        self.encoder.reset()
        self.alert.reset()
        self.previous = None
        self.history = deque(maxlen=4)
        self.mobility, self.acceleration, self.periodicity, self.extra = [], [], [], []
        self.recent = deque(maxlen=CONFIRMATIONS)

    def update(self, hb_router_probs):
        probability = np.asarray(hb_router_probs)
        if (probability.shape != (8, 10, 11, 32) or not np.isfinite(probability).all()
                or (probability < 0).any() or (probability.sum(-1) <= 0).any()):
            raise ValueError("expected finite nonnegative positive-mass probabilities [8,10,11,32]")
        encoded = self.encoder.update(probability)
        m, a, p, extra, final = raw_legacy_features(probability, self.previous, list(self.history))
        self.mobility.append(m)
        self.acceleration.append(a)
        self.periodicity.append(p)
        self.extra.append(extra)
        self.previous = final
        self.history.append(final[4:].reshape(40, 32))
        branches = dict.fromkeys(BRANCHES, np.nan)
        if len(self.mobility) >= 7:
            cache = dict(mobility=np.asarray(self.mobility, np.float32)[None],
                         acceleration=np.asarray(self.acceleration, np.float32)[None],
                         periodicity=np.asarray(self.periodicity, np.float32)[None],
                         valid=np.ones((1, len(self.mobility)), bool))
            branches["v82"] = float(v82_scores(cache, np.asarray(self.extra, np.float32)[None],
                                               self.profile["v82_profile"])[0, -1])
        # Match the exported encoding precision before applying confirmation.
        values = encoded.values.astype(np.float32)
        self.recent.append([values[FEATURE_NAMES.index("acceleration_log_ratio")],
                            values[FEATURE_NAMES.index("recent_decoupling")]])
        if len(self.recent) == CONFIRMATIONS:
            window = np.asarray(self.recent)
            for i, name in enumerate(BRANCHES[1:]):
                if np.isfinite(window[:, i]).all():
                    branches[name] = float(window[:, i].min())
        tails = {name: float(self.banks[name].tail(value)) for name, value in branches.items()}
        result = self.alert.update(float(combine_tails(tails, self.method)))
        result.update(branch_scores=branches, branch_tails=tails, features=encoded.as_dict(),
                      triggering_branches=[name for name in METHOD_BRANCHES[self.method]
                                           if np.isfinite(tails[name])
                                           and len(METHOD_BRANCHES[self.method]) * tails[name] <= self.alert.alpha])
        return result
