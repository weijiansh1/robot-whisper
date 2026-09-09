"""Fixed absolute-scale constraints and decaying sequential routing evidence."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .encoder import FEATURE_NAMES
    from .guard import PeakBank, persistent
    from .state_action import StateActionGuardMonitor
else:
    from encoder import FEATURE_NAMES
    from guard import PeakBank, persistent
    from state_action import StateActionGuardMonitor

SCHEMA = "himoe.causal_routing_models.v1"
COMPONENTS = ("acceleration", "decoupling", "absolute_acceleration", "absolute_mobility_low", "joint_stop", "state_trap")
METHODS = ("ac_reference", "absolute_acc", "absolute_both", "persistent_ac", "joint_stop_or",
           "absolute_state_trap", "leaky_ac", "leaky_state_trap")
PRIMARY = "absolute_acc"
DECAY, DRIFT, WINDOW = .75, float(np.log(10.)), 4


def compose_components(branches, absolute_acceleration, absolute_mobility_low, relative_mobility_low):
    a = np.asarray(branches["acceleration"], float)
    joint = np.minimum(np.maximum(relative_mobility_low, 0.), np.maximum(branches["state_relative_low"], 0.))
    return dict(acceleration=a, decoupling=np.asarray(branches["decoupling"], float),
                absolute_acceleration=np.asarray(absolute_acceleration, float),
                absolute_mobility_low=np.asarray(absolute_mobility_low, float),
                joint_stop=joint, state_trap=np.minimum(np.maximum(a, 0.), joint))


def make_components(branches, features, valid):
    values = []
    for name, sign in (("flow_acceleration", 1.), ("back_mobility", -1.), ("mobility_log_ratio", -1.)):
        column = features[..., FEATURE_NAMES.index(name)].astype(float)
        values.append(persistent(np.where(valid, sign * column, np.nan), 2))
    return compose_components(branches, *values)


def reference_evidence(components, banks):
    result = {name: -np.log(banks[name].tail(components[name])) for name in COMPONENTS}
    for name in ("joint_stop", "state_trap"):
        result[name] = np.where(np.isfinite(components[name]),
                                np.where(components[name] > 0, result[name], 0.), np.nan)
    return result


def instant_models(evidence):
    values = [np.asarray(evidence[name], float) for name in COMPONENTS]
    if not all(value.shape == values[0].shape for value in values):
        raise ValueError("component shapes must agree")
    ready = np.isfinite(values[0]) | np.isfinite(values[1])
    a, d, aa, am, joint, trap = [np.where(np.isfinite(x), x, 0.) for x in values]
    core = np.maximum(a, d)
    absolute = np.maximum(np.minimum(a, aa), d)
    result = dict(ac_reference=core, absolute_acc=absolute,
                  absolute_both=np.maximum(np.minimum(a, aa), np.minimum(d, am)),
                  joint_stop_or=np.maximum(core, joint), absolute_state_trap=np.maximum(absolute, trap))
    return {name: np.where(ready, value, np.nan) for name, value in result.items()}


def leaky_evidence(values):
    values = np.asarray(values, float)
    if values.ndim != 2:
        raise ValueError("expected [episode,query] scores")
    result = np.full_like(values, np.nan)
    accumulator = np.zeros(values.shape[0])
    for q in range(values.shape[1]):
        ready = np.isfinite(values[:, q])
        accumulator = np.where(ready, np.maximum(0., DECAY * accumulator + values[:, q] - DRIFT), 0.)
        result[:, q] = np.where(ready, accumulator, np.nan)
    return result


def model_scores(evidence, valid):
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe observed prefixes")
    if any(np.asarray(x).shape != valid.shape for x in evidence.values()):
        raise ValueError("evidence and validity must align")
    current = {name: np.where(valid, value, np.nan) for name, value in evidence.items()}
    result = instant_models(current)
    result["persistent_ac"] = np.fmax(persistent(current["acceleration"], WINDOW), persistent(current["decoupling"], WINDOW))
    result["leaky_ac"] = leaky_evidence(result["ac_reference"])
    result["leaky_state_trap"] = leaky_evidence(result["absolute_state_trap"])
    return {name: np.where(valid, result[name], np.nan) for name in METHODS}


class CausalModelMonitor:
    """Raw-routing monitor with a fixed model and a frozen original alarm path."""

    def __init__(self, profile_path, checkpoint, method=PRIMARY, kind="task_init", alpha=.01):
        path = Path(profile_path)
        self.profile = json.loads(path.read_text())
        if self.profile.get("schema") != SCHEMA or self.profile.get("methods") != list(METHODS):
            raise ValueError("incompatible causal model profile")
        if method not in METHODS or kind not in self.profile["banks"] or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("invalid model or calibration")
        if self.profile.get("decay") != DECAY or self.profile.get("drift") != DRIFT:
            raise ValueError("incompatible accumulator")
        base_path = (path.parent / self.profile["base_profile"]).resolve()
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != self.profile["base_profile_sha256"]:
            raise ValueError("underlying routing profile changed")
        self.base = StateActionGuardMonitor(base_path, checkpoint, kind=kind, alpha=alpha)
        self.references = {name: PeakBank.from_list(bank) for name, bank in self.profile["references"].items()}
        self.banks = {name: PeakBank.from_list(bank) for name, bank in self.profile["banks"][kind].items()}
        self.method, self.alpha = method, alpha
        self.reset()

    def reset(self):
        self.base.reset()
        self.recent = deque(maxlen=WINDOW)
        self.accumulators = dict(leaky_ac=0., leaky_state_trap=0.)
        self.first_added = self.first_alarm = -1

    def update(self, probability):
        base = self.base.update(probability)
        window = np.asarray(self.base.action_recent, float)
        values = []
        for name, sign in (("flow_acceleration", 1.), ("back_mobility", -1.), ("mobility_log_ratio", -1.)):
            values.append(float((sign * window[:, FEATURE_NAMES.index(name)]).min()) if len(window) == 2 else np.nan)
        components = compose_components(base["scores"], *values)
        evidence = reference_evidence(components, self.references)
        scores = instant_models(evidence)
        self.recent.append([float(evidence["acceleration"]), float(evidence["decoupling"])])
        scores["persistent_ac"] = np.nan
        if len(self.recent) == WINDOW:
            window = np.asarray(self.recent)
            minimum = np.where(np.isfinite(window).all(0), window.min(0), np.nan)
            scores["persistent_ac"] = float(np.fmax(minimum[0], minimum[1]))
        for name, source in (("leaky_ac", "ac_reference"), ("leaky_state_trap", "absolute_state_trap")):
            value = float(scores[source])
            ready = np.isfinite(value)
            self.accumulators[name] = max(0., DECAY * self.accumulators[name] + value - DRIFT) if ready else 0.
            scores[name] = self.accumulators[name] if ready else np.nan
        scores = {name: float(scores[name]) for name in METHODS}
        tails = {name: float(self.banks[name].tail(value)) for name, value in scores.items()}
        trigger = bool(np.isfinite(tails[self.method]) and tails[self.method] <= self.alpha)
        if trigger and self.first_added < 0:
            self.first_added = base["query"]
        if self.first_alarm < 0 and (trigger or base["first"]["v82"] >= 0):
            self.first_alarm = base["query"]
        return dict(query=base["query"], components={name: float(x) for name, x in components.items()},
                    evidence={name: float(x) for name, x in evidence.items()}, scores=scores, tails=tails,
                    first_added_query=self.first_added, first_alarm_query=self.first_alarm,
                    frozen_first_query=base["first"]["v82"], ever_alarm=self.first_alarm >= 0)
