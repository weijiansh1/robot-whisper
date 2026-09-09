"""Fixed train-free routing combinations with separate reference and calibration."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .guard import PeakBank
    from .state_action import StateActionGuardMonitor
else:
    from guard import PeakBank
    from state_action import StateActionGuardMonitor

SCHEMA = "himoe.routing_combinations.v1"
COMPONENTS = ("acceleration", "decoupling", "gap", "state_adjustment")
METHODS = ("ac_recalibrated", "max_or_gap", "soft_sum_gap", "two_of_three",
           "lagged_agreement", "state_adjustment_or")
PRIMARY = "soft_sum_gap"
WINDOW = 4


def component_scores(branches):
    return dict(acceleration=np.asarray(branches["acceleration"], float),
                decoupling=np.asarray(branches["decoupling"], float),
                gap=np.asarray(branches["gap"], float),
                state_adjustment=np.minimum(np.maximum(branches["acceleration"], 0.),
                                            np.maximum(branches["state_relative_low"], 0.)))


def reference_evidence(components, banks):
    values = {name: -np.log(banks[name].tail(components[name])) for name in COMPONENTS}
    values["gap"] = np.where(np.isfinite(components["gap"]),
                             np.where(components["gap"] > 0, values["gap"], 0.), np.nan)
    return values


def combine_instant(evidence):
    a, d, g, s = [np.asarray(evidence[name], float) for name in COMPONENTS]
    if not all(value.shape == a.shape for value in (d, g, s)):
        raise ValueError("component shapes must agree")
    ready = np.isfinite(a) | np.isfinite(d)
    a, d, g, s = [np.where(np.isfinite(value), value, 0.) for value in (a, d, g, s)]
    c = np.maximum(a, d)
    results = dict(ac_recalibrated=c, max_or_gap=np.maximum(c, g), soft_sum_gap=c + g,
                   two_of_three=np.sort(np.stack((a, d, g)), axis=0)[1],
                   state_adjustment_or=np.maximum(c, s))
    return {name: np.where(ready, value, np.nan) for name, value in results.items()}


def recent_max(values):
    values = np.asarray(values, float)
    if values.ndim != 2:
        raise ValueError("expected [episode,query] values")
    result = np.full_like(values, np.nan)
    for q in range(values.shape[1]):
        window = values[:, max(0, q - WINDOW + 1):q + 1]
        maximum = np.where(np.isfinite(window), window, -np.inf).max(axis=1)
        result[:, q] = np.where(np.isfinite(maximum), maximum, np.nan)
    return result


def combination_scores(evidence, valid):
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe [episode,query] observed prefixes")
    if any(np.asarray(value).shape != valid.shape for value in evidence.values()):
        raise ValueError("evidence and validity must align")
    current = {name: np.where(valid, value, np.nan) for name, value in evidence.items()}
    instant = combine_instant(current)
    lagged = np.minimum(recent_max(instant["ac_recalibrated"]), recent_max(current["gap"]))
    instant["lagged_agreement"] = np.where(np.isfinite(instant["ac_recalibrated"]), lagged, np.nan)
    return {name: np.where(valid, instant[name], np.nan) for name in METHODS}


class CombinationMonitor:
    """One-query raw-routing API retaining every original frozen v8.2 alarm."""

    def __init__(self, profile_path, checkpoint, method=PRIMARY, kind="task_init", alpha=.01):
        path = Path(profile_path)
        self.profile = json.loads(path.read_text())
        if self.profile.get("schema") != SCHEMA or self.profile.get("methods") != list(METHODS):
            raise ValueError("incompatible combination profile")
        if method not in METHODS or kind not in self.profile["banks"] or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("invalid method, calibration, or budget")
        base_path = (path.parent / self.profile["base_profile"]).resolve()
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != self.profile["base_profile_sha256"]:
            raise ValueError("underlying routing profile changed")
        self.base = StateActionGuardMonitor(base_path, checkpoint, kind=kind, alpha=alpha)
        self.references = {name: PeakBank.from_list(values) for name, values in self.profile["references"].items()}
        self.banks = {name: PeakBank.from_list(values) for name, values in self.profile["banks"][kind].items()}
        self.method, self.alpha = method, alpha
        self.reset()

    def reset(self):
        self.base.reset()
        self.history = deque(maxlen=WINDOW)
        self.first_added = self.first_alarm = -1

    def update(self, probability):
        current = self.base.update(probability)
        evidence = reference_evidence(component_scores(current["scores"]), self.references)
        instant = combine_instant(evidence)
        self.history.append([float(instant["ac_recalibrated"]), float(evidence["gap"])])
        window = np.asarray(self.history)
        maximum = np.where(np.isfinite(window), window, -np.inf).max(axis=0)
        lagged = float(maximum.min()) if np.isfinite(maximum).all() else np.nan
        instant["lagged_agreement"] = lagged if np.isfinite(instant["ac_recalibrated"]) else np.nan
        scores = {name: float(instant[name]) for name in METHODS}
        tails = {name: float(self.banks[name].tail(value)) for name, value in scores.items()}
        tail = tails[self.method]
        trigger = bool(np.isfinite(tail) and tail <= self.alpha)
        if trigger and self.first_added < 0:
            self.first_added = current["query"]
        if self.first_alarm < 0 and (trigger or current["first"]["v82"] >= 0):
            self.first_alarm = current["query"]
        return dict(query=current["query"], scores=scores, tails=tails,
                    base_scores=current["scores"], evidence={name: float(x) for name, x in evidence.items()},
                    added_trigger_now=trigger, first_added_query=self.first_added,
                    first_alarm_query=self.first_alarm, ever_alarm=self.first_alarm >= 0,
                    frozen_first_query=current["first"]["v82"])
