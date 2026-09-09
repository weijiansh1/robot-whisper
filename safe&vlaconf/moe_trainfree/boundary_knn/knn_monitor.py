"""Read the sealed boundary profile from one routing query at a time."""

import numpy as np

from knn import METHODS, PRIMARY, ReferenceScorer, dynamics
from core import route_features
from intrinsic_guard_monitor import IntrinsicGuardMonitor


class BoundaryKNNMonitor:
    def __init__(self, profile_path, checkpoint, method=PRIMARY, alpha=.05, grouped=True):
        with np.load(profile_path, allow_pickle=False) as z:
            self.profile = {k: z[k] for k in z.files}
        if str(self.profile["checkpoint"]) != checkpoint:
            raise ValueError("reference profile belongs to another checkpoint")
        if method not in METHODS:
            raise ValueError("unknown boundary method")
        alpha_rows = np.flatnonzero(np.isclose(self.profile["alphas"], alpha))
        if len(alpha_rows) != 1:
            raise ValueError("alpha absent from profile")
        self.method = method
        position = list(self.profile["methods"].astype(str)).index(method)
        self.threshold = float(self.profile["thresholds"][int(grouped), alpha_rows[0], position])
        self.scorer = ReferenceScorer(self.profile)
        self.reset()

    def reset(self):
        self.query = 0
        self.previous = None
        self.action_history = []
        self.mobility, self.acceleration, self.periodicity = [], [], []
        self.previous_eef = None
        self.eef_deltas = []
        self.radius_at_seven = None
        self.knn_history = []
        self.first_alarm_query = -1

    def update(self, probabilities, eef_position=None):
        if self.query >= 52:
            raise ValueError("query exceeds the evaluated 52-query horizon")
        raw = np.asarray(probabilities)
        if raw.shape != (8, 10, 11, 32):
            raise ValueError("expected all-flow routing [8,10,11,32]")
        if self.method == "eef_motion_low":
            eef = np.asarray(eef_position, dtype=float)
            if eef.shape != (3,) or not np.isfinite(eef).all():
                raise ValueError("eef_motion_low requires the current finite EEF position")
            if self.previous_eef is not None:
                self.eef_deltas.append(float(np.linalg.norm(eef - self.previous_eef)))
            self.previous_eef = eef.copy()
        m, a, p, final = IntrinsicGuardMonitor._query_features(raw, self.previous, self.action_history)
        self.mobility.append(m)
        self.acceleration.append(a)
        self.periodicity.append(p)
        self.action_history.append(final[4:].reshape(40, 32))
        self.previous = final
        score = np.nan
        if self.query >= 7:
            dynamic, _ = dynamics(np.asarray(self.mobility, np.float32)[None],
                np.asarray(self.acceleration, np.float32)[None], np.asarray(self.periodicity, np.float32)[None],
                float(self.profile["periodicity_scale"]))
            if self.method == "clock_q7":
                score = float(self.query)
            elif self.method == "eef_motion_low":
                score = float(np.float32(-np.mean(self.eef_deltas[-4:])))
            else:
                base = {"dyn_radius_outward": "dyn_radius", "dyn_success_knn_k20_persist3": PRIMARY}.get(self.method, self.method)
                routing = route_features(raw[:, 9][None])["load"]
                score = float(self.scorer.current(dynamic[:, -1], routing, (base,))[base][0])
                if self.method == "dyn_radius_outward":
                    if self.radius_at_seven is None:
                        self.radius_at_seven = score
                    score = float(np.float32(score - self.radius_at_seven))
                elif self.method == "dyn_success_knn_k20_persist3":
                    self.knn_history.append(score)
                    score = float(np.min(self.knn_history[-3:])) if len(self.knn_history) >= 3 else np.nan
        trigger = bool(np.isfinite(score) and score > self.threshold)
        if trigger and self.first_alarm_query < 0:
            self.first_alarm_query = self.query
        result = {"query": self.query, "score": score, "threshold": self.threshold,
                  "trigger_now": trigger, "alarm": self.first_alarm_query >= 0,
                  "first_alarm_query": self.first_alarm_query}
        self.query += 1
        return result
