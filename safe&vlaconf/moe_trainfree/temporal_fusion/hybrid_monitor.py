"""Combine a recalibrated kNN branch with the existing frozen v7/v8 guard."""

import json
from pathlib import Path
import sys

import numpy as np

from fusion import HERE, ROOT, v8_heads
sys.path.insert(0, str(HERE))
from monitor import TemporalFusionMonitor
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor


class KnNV8Monitor:
    def __init__(self, profile_path, checkpoint, knn_method="knn12_v8", version="v8.2", alpha=.05):
        if knn_method not in ("knn10", "knn12_v8"):
            raise ValueError("the hybrid accepts the 10-D or 12-D kNN branch")
        if version not in ("v8", "v8.2"):
            raise ValueError("supported frozen versions are v8 and v8.2")
        self.knn = TemporalFusionMonitor(profile_path, checkpoint, knn_method, alpha, grouped=True)
        self.v7_profile = GlobalIntrinsicProfile.load(ROOT / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz")
        summary = "v8_full_corpus_summary.json" if version == "v8" else "v82_summary.json"
        self.config = json.loads((ROOT / "moe-v8-0906/results" / summary).read_text())
        if version == "v8.2" and self.config["config"] != dict(baseline=4, width=6, confirm=2, slope=-.0015):
            raise ValueError("frozen v8.2 configuration changed")
        self.relaxation = 0.0 if version == "v8" else .0015
        self.version = version
        self.reset()

    def reset(self):
        self.knn.reset()
        self.v7 = IntrinsicGuardMonitor(self.v7_profile)
        self.confirmations = np.zeros(2, dtype=int)
        self.flow_alarm = False
        self.first_alarm_query = -1

    def update(self, probabilities):
        knn = self.knn.update(probabilities)
        v7 = self.v7.update(probabilities)
        query = knn["query"]
        flow = np.full(2, np.nan)
        if query >= 6:
            flow = v8_heads(np.asarray(self.knn.raw_v8, np.float32)[None])[0, -1]
        thresholds = np.asarray([-self.config["thresholds"]["frontback_flowpath"],
                                  self.config["thresholds"]["curvature_3step"]])
        crossing = np.isfinite(flow) & (flow + self.relaxation*query >= thresholds)
        self.confirmations = np.where(crossing, self.confirmations+1, 0)
        self.flow_alarm |= bool((self.confirmations >= 2).any())
        guard_alarm = bool(v7["alarm"] or self.flow_alarm)
        alarm = bool(knn["alarm"] or guard_alarm)
        if alarm and self.first_alarm_query < 0:
            self.first_alarm_query = query
        return dict(query=query, alarm=alarm, first_alarm_query=self.first_alarm_query,
                    knn_alarm=knn["alarm"], knn_score=knn["score"], knn_threshold=knn["threshold"],
                    v7_alarm=bool(v7["alarm"]), v8_alarm=guard_alarm, version=self.version)
