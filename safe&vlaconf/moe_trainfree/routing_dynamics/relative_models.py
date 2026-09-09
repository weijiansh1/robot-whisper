"""Fixed relative constraints and causal routing-evidence combinations."""

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .encoder import FEATURE_NAMES
    from .guard import PeakBank, dynamics_scores, persistent
    from .state_action import RELATION_NAMES, StateActionGuardMonitor
else:
    from encoder import FEATURE_NAMES
    from guard import PeakBank, dynamics_scores, persistent
    from state_action import RELATION_NAMES, StateActionGuardMonitor

SCHEMA = "himoe.relative_routing_models.v1"
COMPONENTS = ("acceleration", "decoupling", "relative_curvature", "relative_path_mobility",
              "relative_state", "routing_relief")
RAW_NAMES = (*COMPONENTS, "layer_support")
METHODS = ("ac_reference", "relative_curvature", "relative_path_mobility", "relative_state",
           "relative_both", "relative_soft", "layer_consensus", "relative_recovery", "ac_window_mean")
PRIMARY, WINDOW = "relative_curvature", 4


def instant_components(features, relations):
    features, relations = np.asarray(features, float), np.asarray(relations, float)
    if features.shape[:-1] != relations.shape[:-1] or features.shape[-1] != len(FEATURE_NAMES) or relations.shape[-1] != len(RELATION_NAMES):
        raise ValueError("expected aligned action and state/action features")
    a, p, m, h, v = [features[..., FEATURE_NAMES.index(name)] for name in
                     ("acceleration_log_ratio", "path_log_ratio", "mobility_log_ratio", "layer_agreement", "routing_recovery")]
    s_low = relations[..., RELATION_NAMES.index("state_relative_low")]
    return dict(relative_curvature=np.maximum(a-p, 0.), relative_path_mobility=np.maximum(p-m, 0.),
                relative_state=np.maximum(a+s_low, 0.), routing_relief=np.maximum(v, 0.), layer_support=h)


def make_components(features, relations, valid):
    result = dynamics_scores(features, valid)
    for name, values in instant_components(features, relations).items():
        result[name] = persistent(np.where(valid, values, np.nan), 2)
    return {name: result[name] for name in RAW_NAMES}


def reference_evidence(components, banks):
    result = {name: -np.log(banks[name].tail(components[name])) for name in COMPONENTS}
    for name in COMPONENTS[2:]:
        values = np.asarray(components[name])
        result[name] = np.where(np.isfinite(values), np.where(values > 0, result[name], 0.), np.nan)
    result["layer_support"] = np.asarray(components["layer_support"], float)
    return result


def instant_models(evidence):
    values = [np.asarray(evidence[name], float) for name in RAW_NAMES]
    if not all(value.shape == values[0].shape for value in values):
        raise ValueError("component shapes must agree")
    ready = np.isfinite(values[0]) | np.isfinite(values[1])
    a, d, k, r, s, v, h = [np.where(np.isfinite(value), value, 0.) for value in values]
    if np.any((h < 0) | (h > 1)):
        raise ValueError("layer support must be a proportion")
    core, relative = np.maximum(a, d), np.maximum(np.minimum(a, k), d)
    result = dict(ac_reference=core, relative_curvature=relative,
                  relative_path_mobility=np.maximum(np.minimum(a, r), d),
                  relative_state=np.maximum(np.minimum(a, s), d),
                  relative_both=np.maximum(np.minimum(a, k), np.minimum(d, r)),
                  relative_soft=np.maximum(np.sqrt(a*k), d), layer_consensus=core*h,
                  relative_recovery=np.maximum(relative-v, 0.))
    return {name: np.where(ready, value, np.nan) for name, value in result.items()}


def window_mean(values):
    values = np.asarray(values, float)
    if values.ndim != 2:
        raise ValueError("expected [episode,query] scores")
    result = np.full_like(values, np.nan)
    for q in range(WINDOW-1, values.shape[1]):
        recent = values[:, q-WINDOW+1:q+1]
        result[:, q] = np.where(np.isfinite(recent).all(1), recent.mean(1), np.nan)
    return result


def model_scores(evidence, valid):
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("validity must describe observed prefixes")
    if any(np.asarray(value).shape != valid.shape for value in evidence.values()):
        raise ValueError("evidence and validity must align")
    current = {name: np.where(valid, value, np.nan) for name, value in evidence.items()}
    result = instant_models(current)
    result["ac_window_mean"] = window_mean(result["ac_reference"])
    return {name: np.where(valid, result[name], np.nan) for name in METHODS}


class RelativeModelMonitor:
    """One raw query per update; relative changes never retract historical alarms."""

    def __init__(self, profile_path, checkpoint, method=PRIMARY, kind="task_init", alpha=.01):
        path = Path(profile_path)
        self.profile = json.loads(path.read_text())
        if self.profile.get("schema") != SCHEMA or self.profile.get("methods") != list(METHODS) or self.profile.get("window") != WINDOW:
            raise ValueError("incompatible relative model profile")
        if method not in METHODS or kind not in self.profile["banks"] or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("invalid model or calibration")
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
        self.recent_core = deque(maxlen=WINDOW)
        self.first_added = self.first_alarm = -1

    def update(self, probability):
        base = self.base.update(probability)
        current = instant_components(np.asarray(self.base.action_recent), np.asarray(self.base.relation_recent))
        components = {name: float(values.min()) if len(values) == 2 else np.nan for name, values in current.items()}
        components.update({name: base["scores"][name] for name in ("acceleration", "decoupling")})
        evidence = reference_evidence(components, self.references)
        scores = instant_models(evidence)
        self.recent_core.append(float(scores["ac_reference"]))
        recent = np.asarray(self.recent_core)
        scores["ac_window_mean"] = float(recent.mean()) if len(recent) == WINDOW and np.isfinite(recent).all() else np.nan
        scores = {name: float(scores[name]) for name in METHODS}
        tails = {name: float(self.banks[name].tail(value)) for name, value in scores.items()}
        trigger = bool(np.isfinite(tails[self.method]) and tails[self.method] <= self.alpha)
        if trigger and self.first_added < 0:
            self.first_added = base["query"]
        if self.first_alarm < 0 and (trigger or base["first"]["v82"] >= 0):
            self.first_alarm = base["query"]
        return dict(query=base["query"], components=components,
                    evidence={name: float(value) for name, value in evidence.items()}, scores=scores, tails=tails,
                    first_added_query=self.first_added, first_alarm_query=self.first_alarm,
                    frozen_first_query=base["first"]["v82"], ever_alarm=self.first_alarm >= 0)
