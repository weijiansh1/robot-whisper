"""Route-only guards with absolute freeze support and expiring evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from monitor import (fit_location, intrinsic_score_arrays, persistent, prefix_peak,
                     scale_values, trailing, v8_streams)  # noqa: E402
from intrinsic_guard_monitor import IntrinsicGuardMonitor, normalize_probability  # noqa: E402

METHODS = ("v82_reference", "recent4", "absolute_gate", "recovery", "recovery_fixed",
           "recovery_confirm2", "recovery_confirm3", "reference_confirm3", "recent4_confirm2")
HEADS = ("freeze", "acceleration_persistent", "periodicity_persistent", "frontback",
         "curvature", "absolute_freeze")
SCHEMA = "himoe.v82.recovery.v1"


def recent_peak(values, width=4):
    values = np.asarray(values, float)
    result = np.full_like(values, np.nan)
    for q in range(values.shape[1]):
        window = values[:, max(0, q - width + 1):q + 1]
        peak = np.where(np.isfinite(window), window, -np.inf).max(axis=1)
        result[:, q] = np.where(np.isfinite(peak), peak, np.nan)
    return result


def head_streams(cache, raw, periodicity_scale):
    valid = np.asarray(cache["valid"], bool)
    mobility = np.where(valid[..., None], cache["mobility"], np.nan)
    acceleration = np.where(valid, cache["acceleration"], np.nan)
    periodicity = np.where(valid, cache["periodicity"], np.nan)
    old = intrinsic_score_arrays(mobility, acceleration, periodicity, periodicity_scale)
    new = v8_streams(raw, valid)
    absolute = trailing(np.median(-np.log(np.maximum(mobility[:, :, 4:], 1e-6)), axis=2), 6)
    values = {name: old[name] for name in HEADS[:3]}
    values.update(frontback=new[:, :, 0], curvature=new[:, :, 1], absolute_freeze=absolute)
    for value in values.values():
        value[:, :6] = np.nan
        value[~valid] = np.nan
    return values


def fit_profile(cache, raw, reference):
    period = np.where(cache["valid"][reference], cache["periodicity"][reference], np.nan)
    pscale = float(np.quantile(np.abs(period[np.isfinite(period)]), .75))
    values = head_streams(cache, raw, pscale)
    return dict(schema=SCHEMA, periodicity_scale=pscale,
                heads={name: fit_location(value[reference]) for name, value in values.items()})


def score_streams(cache, raw, profile):
    values = head_streams(cache, raw, float(profile["periodicity_scale"]))
    norm = {name: scale_values(value, profile["heads"][name]) for name, value in values.items()}
    freeze = norm["freeze"]
    gated = np.minimum(freeze, norm["absolute_freeze"])
    acc, period = norm["acceleration_persistent"], norm["periodicity_persistent"]
    latched = np.minimum(prefix_peak(acc), prefix_peak(period))
    latched[~np.isfinite(latched)] = np.nan
    recent = np.minimum(recent_peak(acc), recent_peak(period))
    q = np.arange(freeze.shape[1])[None]
    dynamic, fixed = [], []
    for name in ("frontback", "curvature"):
        fixed.append(persistent(norm[name], 2))
        dynamic.append(persistent(norm[name] + .0015 * q / profile["heads"][name]["scale"], 2))

    def combine(f, t, additions=dynamic):
        return np.fmax.reduce([f, t, *additions])

    baseline = combine(freeze, latched)
    recent_only = combine(freeze, recent)
    current = combine(gated, recent)
    results = (baseline, recent_only, combine(gated, latched), current,
               combine(gated, recent, fixed), persistent(current, 2), persistent(current, 3),
               persistent(baseline, 3), persistent(recent_only, 2))
    scores = np.stack(results).astype(np.float32)
    scores[:, ~cache["valid"]] = np.nan
    return scores


def raw_features(probability, previous, history):
    mobility, acceleration, periodicity, final = IntrinsicGuardMonitor._query_features(probability, previous, history)
    root = np.sqrt(normalize_probability(probability)[:, :, 1:])
    speed = (np.linalg.norm(np.diff(root, axis=1), axis=-1) / np.sqrt(2.)).mean(axis=-1).astype(np.float32)
    path = speed.sum(axis=-1)
    frontback = -np.log(np.maximum(path[:4].mean(), 1e-12) / np.maximum(path[4:].mean(), 1e-12))
    curvature = np.abs(np.diff(speed[4:][:, [0, 4, 8]], n=2, axis=-1)).mean()
    return mobility, acceleration, periodicity, np.asarray([frontback, curvature], np.float32), final


class AlertState:
    """Current signal state can clear; the historical alarm cannot."""

    def __init__(self, threshold, watch_threshold, clear_queries=2):
        if np.isnan(threshold) or np.isnan(watch_threshold) or watch_threshold > threshold:
            raise ValueError("thresholds must be ordered and non-NaN")
        if clear_queries < 1:
            raise ValueError("clear_queries must be positive")
        self.threshold = float(threshold)
        self.watch_threshold = float(watch_threshold)
        self.clear_queries = clear_queries
        self.reset()

    def reset(self):
        self.query = 0
        self.state = "NORMAL"
        self.below = 0
        self.first_alarm_query = -1
        self.first_watch_query = -1

    def update(self, score):
        before = self.state
        ready = bool(np.isfinite(score))
        trigger = ready and score > self.threshold
        watch = ready and score > self.watch_threshold
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
        result = dict(query=self.query, score=float(score), ready=ready, state=self.state,
                      trigger_now=bool(trigger), alarm_active=self.state == "ALARM",
                      ever_alarm=self.first_alarm_query >= 0, first_alarm_query=self.first_alarm_query,
                      first_watch_query=self.first_watch_query,
                      signal_cleared=before != "NORMAL" and self.state == "NORMAL")
        self.query += 1
        return result


class RecoveryGuardMonitor:
    def __init__(self, profile_path, checkpoint):
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get("schema") != SCHEMA:
            raise ValueError("incompatible recovery profile")
        if checkpoint not in self.profile["checkpoints"]:
            raise ValueError("checkpoint absent from the calibrated profile")
        self.method = self.profile["method"]
        if self.method not in METHODS:
            raise ValueError("unknown recovery method")
        threshold = self.profile["threshold"]
        watch = self.profile["watch_threshold"]
        self.alert = AlertState(float("inf") if threshold is None else threshold,
                                float("inf") if watch is None else watch)
        self.reset()

    def reset(self):
        self.previous = None
        self.history = []
        self.mobility, self.acceleration, self.periodicity, self.additions = [], [], [], []
        self.alert.reset()

    def update(self, probabilities):
        probability = np.asarray(probabilities)
        if (probability.shape != (8, 10, 11, 32) or not np.isfinite(probability).all()
                or (probability < 0).any() or (probability.sum(axis=-1) <= 0).any()):
            raise ValueError("expected positive-mass finite nonnegative probabilities [8,10,11,32]")
        m, a, p, extra, final = raw_features(probability, self.previous, self.history)
        self.mobility.append(m)
        self.acceleration.append(a)
        self.periodicity.append(p)
        self.additions.append(extra)
        self.previous = final
        self.history.append(final[4:].reshape(40, 32))
        self.history = self.history[-4:]
        score = np.nan
        if len(self.mobility) >= 7:
            cache = dict(mobility=np.asarray(self.mobility, np.float32)[None],
                         acceleration=np.asarray(self.acceleration, np.float32)[None],
                         periodicity=np.asarray(self.periodicity, np.float32)[None],
                         valid=np.ones((1, len(self.mobility)), bool))
            score = float(score_streams(cache, np.asarray(self.additions, np.float32)[None],
                                        self.profile)[METHODS.index(self.method), 0, -1])
        return self.alert.update(score)
