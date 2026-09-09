"""Training-free candidate selection from frozen routing signals, with abstention."""

from __future__ import annotations

import copy
import json

import numpy as np

from adaptive_control import PARAMETERS, RouteRisk
from v8_feature_control import V8Monitor

PROTOCOL = "moe_control.internal_pareto_selector.v1"
SIGNALS = ("freeze", "acceleration", "periodicity", "frontback_inversion",
           "curvature", "knn_euclidean", "knn_cosine")
RULES = dict(
    training=False, hidden_capture=False, external_critic=False, simulator_lookahead=False,
    threshold_fitting=False, reference="existing frozen success bank; not reference-free",
    candidate_pool="default candidate 0 and three iid-noise candidates at identical observation; batch=1",
    trigger="external frozen alarm; no candidate may redefine the trigger",
    expansion="only if default has an available above-threshold continuous signal",
    dominance="no increase in any positive threshold excess, decrease in at least one active excess",
    action_guard="same gripper signs on the executed prefix; continuous-command RMS change <= finite pool median",
    action_resolution="RMS change must exceed numerical float32 tolerance",
    ranking="fewest remaining above-threshold signals, then smallest action RMS change, then candidate ID",
    abstention="candidate 0 when untriggered, no active signal, missing baseline evidence or no admissible alternative",
    commit="score on private copies of executed history; commit only the dispatched candidate",
    rollout="one selected native-length chunk at q+1, then ordinary policy; original horizon",
    cost="1 forward normally, 4 when expanded; no hidden/probe/critic or additional training",
    interpretation="heuristic signal dominance, not a success-probability or safety certificate",
    numerical_tolerance_float32_eps=64,
)


def tolerance(*values):
    scale = np.maximum.reduce([np.ones_like(np.asarray(values[0], float))]+
                              [np.abs(np.asarray(v, float)) for v in values])
    return RULES["numerical_tolerance_float32_eps"]*np.finfo(np.float32).eps*scale


def choose(scores, actions, thresholds, triggered, execute_count=10):
    """Select a whole candidate; arrays contain current-query evidence only."""
    s, a, limits = np.asarray(scores, float), np.asarray(actions, float), np.asarray(thresholds, float)
    if s.ndim != 2 or s.shape[1] != len(SIGNALS) or not len(s):
        raise ValueError("Expected one seven-signal vector per candidate")
    if a.ndim != 3 or a.shape[0] != len(s) or a.shape[2] < 7 or not 1 <= execute_count <= a.shape[1]:
        raise ValueError("Candidate action shape or execution prefix mismatch")
    if limits.shape != (len(SIGNALS),) or not np.isfinite(limits).all():
        raise ValueError("Invalid frozen signal thresholds")
    if not np.isfinite(a[0]).all():
        raise ValueError("Default action is invalid; cannot abstain into it")
    decision = dict(protocol=PROTOCOL, candidate=0, expanded=False, reason="untriggered",
        active_signals=[], numerical_tolerance=None, action_radius=None,
        admissible=[False]*len(s), action_rms=[None]*len(s), remaining_violations=[None]*len(s))
    if not triggered:
        return decision
    if not np.isfinite(s[0]).all():
        decision["reason"] = "baseline_evidence_unavailable"
        return decision
    eps = tolerance(s[0], limits)
    excess = np.maximum(s-limits, 0.)
    active = excess[0] > eps
    decision.update(active_signals=[name for name, flag in zip(SIGNALS, active) if flag],
                    numerical_tolerance=eps.tolist())
    if not active.any():
        decision["reason"] = "default_has_no_active_excess"
        return decision
    if len(s) == 1:
        decision["reason"] = "additional_candidates_required"
        return decision
    decision["expanded"] = True
    finite_action = np.isfinite(a).all(axis=(1, 2))
    delta = np.full(len(s), np.inf)
    delta[finite_action] = np.sqrt(np.square(a[finite_action, :execute_count, :6]-
                                            a[0, :execute_count, :6]).mean(axis=(1, 2)))
    radius = float(np.median(delta[finite_action]))
    same_gripper = (np.sign(a[:, :execute_count, 6]) == np.sign(a[0, :execute_count, 6])).all(axis=1)
    resolved = float(tolerance(np.max(np.abs(a[0, :execute_count, :6]))))
    action_ok = finite_action & same_gripper & (delta <= radius+resolved) & (delta > resolved)
    finite_signal = np.isfinite(s).all(axis=1)
    no_worse = (excess <= excess[0]+eps).all(axis=1)
    better = (excess[:, active] < excess[0, active]-eps[active]).any(axis=1)
    admissible = finite_signal & action_ok & no_worse & better
    admissible[0] = False
    remaining = (excess > eps).sum(axis=1)
    available = np.flatnonzero(admissible)
    decision.update(action_radius=radius, admissible=admissible.tolist(),
        action_rms=[float(x) if np.isfinite(x) else None for x in delta],
        remaining_violations=[int(x) if finite_signal[i] else None for i, x in enumerate(remaining)])
    if len(available):
        selected = min(available, key=lambda i: (remaining[i], delta[i], int(i)))
        decision.update(candidate=int(selected), reason="admissible_internal_improvement")
    else:
        decision["reason"] = "no_admissible_alternative"
    return decision


class InternalActionSelector:
    """Executed-history monitor; candidate evaluation never advances that history."""

    def __init__(self):
        self.risk, self.monitor = RouteRisk(), V8Monitor()
        config = json.loads(PARAMETERS.read_text())
        legacy, geometry = config["legacy"], config["geometry"]["thresholds"]
        self.thresholds = np.asarray([
            legacy["v7"]["freeze_threshold"], legacy["v7"]["acceleration_threshold"],
            legacy["v7"]["periodicity_threshold"], -legacy["v8_thresholds"]["frontback_flowpath"],
            legacy["v8_thresholds"]["curvature_3step"], geometry["knn20"]["threshold"],
            geometry["cosine_knn20"]["threshold"]], float)
        self.bank = self.risk.reference["success_dynamic"]
        self.bank_norm = np.linalg.norm(self.bank, axis=1)

    def _observe(self, monitor, probabilities):
        status = monitor.update(probabilities)
        vector, euclidean = self.risk.current(monitor.v7)
        cosine = float("nan")
        if np.isfinite(vector).all():
            distances = np.clip(1-(self.bank@vector)/np.maximum(self.bank_norm*np.linalg.norm(vector), 1e-12), 0, 2)
            cosine = float(np.float32(np.partition(distances, 19)[:20].mean()))
        scores = np.asarray([status["freeze_score"], status["acceleration_score"],
            status["periodicity_score"], *status["v8_scores"], euclidean, cosine], float)
        return scores

    def observe(self, probabilities):
        return self._observe(self.monitor, probabilities)

    def preview(self, probabilities):
        rows = []
        for probability in probabilities:
            trial = copy.deepcopy(self.monitor)
            try:
                rows.append(self._observe(trial, probability))
            except (ValueError, FloatingPointError):
                rows.append(np.full(len(SIGNALS), np.nan))
        return np.asarray(rows)

    def needs_candidates(self, default_probability, triggered):
        if not triggered:
            return False
        score = self.preview([default_probability])[0]
        return bool(np.isfinite(score).all() and (score-self.thresholds > tolerance(score, self.thresholds)).any())

    def propose(self, probabilities, actions, observation_ids, triggered, execute_count=10):
        if len(probabilities) != len(actions) or len(observation_ids) != len(actions):
            raise ValueError("Candidate ledger mismatch")
        if not observation_ids or len(set(observation_ids)) != 1 or not observation_ids[0]:
            raise ValueError("All candidates must have identical image/state/instruction input digests")
        scores = self.preview(probabilities)
        return choose(scores, actions, self.thresholds, triggered, execute_count), scores
