"""Frozen v8.2 triggering with paired fixed-v8 and time-aware selectors."""

import copy

import numpy as np

from fixed_recovery_control import TriggerMonitor
from v8_closed_loop import SETTINGS as V8_SETTINGS, limits
from v8_feature_control import V8Monitor

PROTOCOL = "moe_control.native_long_v82.v1"
ARMS = ("native", "iid_random_v8", "iid_v8", "guided_random_v8", "guided_v8",
        "iid_random_v82", "iid_v82", "guided_random_v82", "guided_v82")
SETTINGS = dict(copy.deepcopy(V8_SETTINGS), trigger="v82_frozen", slope=-.0015,
    selector_versions=["v8", "v82"], absolute_query="executed query index starting at zero; shared by all candidates",
    margins="fixed original absolute margins; v82 targets move with its threshold",
    paired_noise_protocol="moe_control.v8_closed_loop.v1")


def selector_version(arm):
    if arm not in ARMS:
        raise ValueError("Unknown v8.2 experiment arm")
    return "v82" if arm.endswith("_v82") else "v8"


def base_arm(arm):
    selector_version(arm)
    if "_random_" in arm:
        return arm.split("_")[0]+"_random"
    return arm[:-3]+"v8" if arm.endswith("v82") else arm


def thresholds_at(arm, query, thresholds):
    if query < 0 or int(query) != query:
        raise ValueError("An executed query index is required")
    result = np.asarray(thresholds, float).copy()
    if selector_version(arm) == "v82":
        result[3:] += SETTINGS["slope"]*query
    return result


class V82Monitor(V8Monitor):
    def __init__(self):
        super().__init__()
        if self.config["v82_config"] != dict(baseline=4, width=6, confirm=2, slope=-.0015):
            raise ValueError("Original v8.2 parameters changed")
        self.v82_counts, self.v82_first = [0, 0], [-1, -1]
        self.first_v82_alarm = -1

    def update(self, probability):
        status = super().update(probability)
        q = self.v7.query
        thresholds = np.array([-self.config["v8_thresholds"]["frontback_flowpath"],
            self.config["v8_thresholds"]["curvature_3step"]])+SETTINGS["slope"]*q
        for i in range(2):
            hit = q >= 6 and status["v8_scores"][i] >= thresholds[i]
            self.v82_counts[i] = self.v82_counts[i]+1 if hit else 0
            if self.v82_counts[i] >= 2 and self.v82_first[i] < 0:
                self.v82_first[i] = q
        if self.first_v82_alarm < 0 and (status["alarm"] or max(self.v82_first) >= 0):
            self.first_v82_alarm = q
        status.update(v82_alarm=self.first_v82_alarm >= 0, v82_first=self.first_v82_alarm,
            v82_thresholds=thresholds, v82_head_first=np.asarray(self.v82_first, np.int32))
        return status


class V82TriggerMonitor(TriggerMonitor):
    def __init__(self):
        super().__init__()
        self.v8 = V82Monitor()
        self.first["v82_frozen"] = -1

    def update(self, probabilities, position):
        status = super().update(probabilities, position)
        self.first["v82_frozen"] = self.v8.first_v82_alarm
        return status
