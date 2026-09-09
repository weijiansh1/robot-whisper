"""Post-diagnostic reverse and two-sided relative routing constraints."""

import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .encoder import FEATURE_NAMES
    from .guard import PeakBank, dynamics_scores, persistent
    from .state_action import StateActionGuardMonitor
else:
    from encoder import FEATURE_NAMES
    from guard import PeakBank, dynamics_scores, persistent
    from state_action import StateActionGuardMonitor

SCHEMA = "himoe.relative_direction_models.v1"
COMPONENTS = ("acceleration", "decoupling", "curvature_low", "curvature_magnitude")
RAW_NAMES = COMPONENTS
METHODS = ("ac_reference", "relative_low", "relative_magnitude", "relative_low_soft")
PRIMARY, WINDOW = "relative_low", 2


def direction_components(features):
    features = np.asarray(features, float)
    if features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("expected routing dynamics features")
    difference = features[..., FEATURE_NAMES.index("path_log_ratio")] - features[..., FEATURE_NAMES.index("acceleration_log_ratio")]
    return dict(curvature_low=np.maximum(difference, 0.), curvature_magnitude=np.abs(difference))


def make_components(features, valid):
    result = dynamics_scores(features, valid)
    result.update({name: persistent(np.where(valid, value, np.nan), WINDOW)
                   for name, value in direction_components(features).items()})
    return result


def reference_evidence(components, banks):
    result = {name: -np.log(banks[name].tail(components[name])) for name in COMPONENTS}
    for name in COMPONENTS[2:]:
        value = np.asarray(components[name])
        result[name] = np.where(np.isfinite(value), np.where(value > 0, result[name], 0.), np.nan)
    return result


def instant_models(evidence):
    values = [np.asarray(evidence[name], float) for name in COMPONENTS]
    if not all(value.shape == values[0].shape for value in values):
        raise ValueError("component shapes must agree")
    ready = np.isfinite(values[0]) | np.isfinite(values[1])
    a, d, low, magnitude = [np.where(np.isfinite(value), value, 0.) for value in values]
    result = dict(ac_reference=np.maximum(a,d), relative_low=np.maximum(np.minimum(a,low),d),
                  relative_magnitude=np.maximum(np.minimum(a,magnitude),d), relative_low_soft=np.maximum(np.sqrt(a*low),d))
    return {name: np.where(ready, value, np.nan) for name, value in result.items()}


def model_scores(evidence, valid):
    valid = np.asarray(valid, bool)
    if valid.ndim != 2 or np.any(valid[:,1:] & ~valid[:,:-1]):
        raise ValueError("validity must describe observed prefixes")
    if any(np.asarray(value).shape != valid.shape for value in evidence.values()):
        raise ValueError("evidence and validity must align")
    return instant_models({name: np.where(valid, value, np.nan) for name, value in evidence.items()})


class RelativeDirectionMonitor:
    def __init__(self, profile_path, checkpoint, method=PRIMARY, kind="task_init", alpha=.01):
        path = Path(profile_path)
        self.profile = json.loads(path.read_text())
        if self.profile.get("schema") != SCHEMA or self.profile.get("methods") != list(METHODS) or self.profile.get("window") != WINDOW:
            raise ValueError("incompatible relative direction profile")
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
        self.first_added = self.first_alarm = -1

    def update(self, probability):
        base = self.base.update(probability)
        current = direction_components(np.asarray(self.base.action_recent))
        components = {name: float(value.min()) if len(value) == WINDOW else np.nan for name, value in current.items()}
        components.update({name: base["scores"][name] for name in ("acceleration", "decoupling")})
        evidence = reference_evidence(components, self.references)
        scores = {name: float(value) for name, value in instant_models(evidence).items()}
        tails = {name: float(self.banks[name].tail(value)) for name, value in scores.items()}
        trigger = bool(np.isfinite(tails[self.method]) and tails[self.method] <= self.alpha)
        if trigger and self.first_added < 0:
            self.first_added = base["query"]
        if self.first_alarm < 0 and (trigger or base["first"]["v82"] >= 0):
            self.first_alarm = base["query"]
        return dict(query=base["query"], components=components, evidence={name: float(x) for name,x in evidence.items()},
                    scores=scores, tails=tails, first_added_query=self.first_added, first_alarm_query=self.first_alarm,
                    frozen_first_query=base["first"]["v82"], ever_alarm=self.first_alarm >= 0)
