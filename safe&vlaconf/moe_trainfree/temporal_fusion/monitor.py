"""Replay the frozen fusion candidates one model inference at a time."""

import numpy as np

from fusion import (METHODS, PRIMARY, ReferenceScorer, base_streams, combine_streams,
                    flow_speed, raw_v8_features)
from intrinsic_guard_monitor import IntrinsicGuardMonitor


class TemporalFusionMonitor:
    def __init__(self, profile_path, checkpoint, method=PRIMARY, alpha=.05, grouped=True):
        with np.load(profile_path, allow_pickle=False) as data:
            self.profile = {key: data[key] for key in data.files}
        if str(self.profile["checkpoint"]) != checkpoint:
            raise ValueError("reference profile belongs to another checkpoint")
        if method not in METHODS:
            raise ValueError("unknown fusion method")
        ai = np.flatnonzero(np.isclose(self.profile["alphas"], alpha))
        if len(ai) != 1:
            raise ValueError("alpha absent from profile")
        self.method = method
        self.position = METHODS.index(method)
        self.threshold = float(self.profile["thresholds"][int(grouped), ai[0], self.position])
        self.scorer = ReferenceScorer(self.profile)
        self.reset()

    def reset(self):
        self.query = 0
        self.previous = None
        self.history = []
        self.mobility, self.acceleration, self.periodicity, self.raw_v8 = [], [], [], []
        self.first_alarm_query = -1

    def update(self, probabilities):
        raw = np.asarray(probabilities)
        if raw.shape != (8, 10, 11, 32) or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("expected finite nonnegative all-flow probabilities [8,10,11,32]")
        if self.query >= 52:
            raise ValueError("query exceeds the evaluated 52-query horizon")
        m, a, p, final = IntrinsicGuardMonitor._query_features(raw, self.previous, self.history)
        self.mobility.append(m)
        self.acceleration.append(a)
        self.periodicity.append(p)
        self.raw_v8.append(raw_v8_features(flow_speed(raw)))
        self.history.append(final[4:].reshape(40, 32))
        self.previous = final
        score = np.nan
        if self.query >= 6:
            base = base_streams(self.profile, np.asarray(self.mobility, np.float32)[None],
                np.asarray(self.acceleration, np.float32)[None], np.asarray(self.periodicity, np.float32)[None],
                np.asarray(self.raw_v8, np.float32)[None], self.scorer)
            score = float(combine_streams(base, self.profile)[self.position, 0, -1])
        trigger = bool(np.isfinite(score) and score > self.threshold)
        if trigger and self.first_alarm_query < 0:
            self.first_alarm_query = self.query
        result = dict(query=self.query, score=score, threshold=self.threshold, trigger_now=trigger,
                      alarm=self.first_alarm_query >= 0, first_alarm_query=self.first_alarm_query)
        self.query += 1
        return result
