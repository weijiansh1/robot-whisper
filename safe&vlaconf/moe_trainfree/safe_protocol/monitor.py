"""Deploy the fixed primary statistics-distance detector without model training."""

from __future__ import annotations

import numpy as np

from core import PRIMARY, ReferenceDistance, route_features


class TrainFreeMoEMonitor:
    def __init__(self, profile, checkpoint, alpha=0.05, grouped=False):
        with np.load(profile, allow_pickle=False) as data:
            if str(data["checkpoint"]) != checkpoint:
                raise ValueError("reference profile belongs to a different checkpoint")
            method = list(data["methods"].astype(str)).index(PRIMARY)
            matches = np.flatnonzero(np.isclose(data["alphas"], alpha))
            if len(matches) != 1:
                raise ValueError("alpha absent from profile")
            self.distance = ReferenceDistance(data["stats_success"], data["stats_failure"])
            self.center = data["center"][method].copy()
            self.scale = data["scale"][method].copy()
            self.threshold = float(data["thresholds"][int(grouped), matches[0], method])
        self.reset()

    def reset(self):
        self.query = 0
        self.cumulative = 0.0
        self.first_alarm_query = -1

    def update(self, probabilities):
        if self.query >= len(self.center):
            raise ValueError("query is outside calibrated horizon")
        raw = np.asarray(probabilities)
        if raw.shape == (8, 10, 11, 32):
            raw = raw[:, 9]
        if raw.shape != (8, 11, 32):
            raise ValueError("expected one query of HB route probabilities")
        vector = route_features(raw[None])["stats"]
        current = float(self.distance.score(vector)[0, 0])
        self.cumulative += current
        # Batch aggregation stores its cumulative score as float32.
        score = float(np.float32(self.cumulative))
        standardized = (score - self.center[self.query]) / self.scale[self.query]
        crossed = bool(standardized > self.threshold)
        if crossed and self.first_alarm_query < 0:
            self.first_alarm_query = self.query
        result = {"query": self.query, "current_score": current, "score": score,
                  "standardized_score": float(standardized), "trigger_now": crossed,
                  "alarm": self.first_alarm_query >= 0, "first_alarm_query": self.first_alarm_query}
        self.query += 1
        return result
