"""Frozen-v8 candidate search, finite recovery horizon and population confirmation."""

from __future__ import annotations

import copy
import json

import numpy as np

from adaptive_control import PARAMETERS
from collection_protocol import stable_id, verify_frozen_alarm
from v8_feature_control import V8Monitor

PROTOCOL = "moe_control.v8_closed_loop.v1"
ARMS = ("native", "iid_random", "iid_v8", "guided_random", "guided_v8")
SIGNALS = ("freeze", "acceleration", "periodicity", "frontback_inversion", "curvature")
SETTINGS = dict(population=8, rounds=2, recovery_queries=12, stable_queries=3,
    chunk=10, horizon=520, v7_margin_fraction=.2, inversion_margin=.1,
    curvature_target=.2, elite_count=4, cem_update_fraction=.5, noise_std_floor=.5,
    noise_std_ceiling=1.5, signal_tolerance=1e-6, training=False, hidden_capture=False,
    score="max(freeze, min(acceleration, periodicity), inversion, curvature), in margin units",
    readiness="all five continuous signals finite; otherwise native until history is available",
    convergence="all sixteen unfiltered candidates <= -1 for three consecutive executed-query states",
    search="eight native-noise samples, then eight iid or one smoothed diagonal CEM update; no cross-state optimizer memory",
    selection="uniform among all sixteen or minimum score; whole native chunk, no action averaging or gate edits",
    guards="finite native policy actions; no extra gripper-sign or median-radius veto",
    streams="one original policy draw per executed query; extra noises and choice use separate common random streams",
    exit="population confirmation or twelve-query budget; then native, with re-alarm recorded but no second intervention")


def limits():
    verify_frozen_alarm()
    legacy = json.loads(PARAMETERS.read_text())["legacy"]
    thresholds = np.array([legacy["v7"][name+"_threshold"] for name in SIGNALS[:3]]+
        [-legacy["v8_thresholds"]["frontback_flowpath"], legacy["v8_thresholds"]["curvature_3step"]], float)
    margins = np.r_[thresholds[:3]*SETTINGS["v7_margin_fraction"],
                    SETTINGS["inversion_margin"], thresholds[4]-SETTINGS["curvature_target"]]
    if np.any(margins <= 0):
        raise ValueError("Recovery margins must be positive")
    return thresholds, margins


def score_status(status):
    return np.asarray([status["freeze_score"], status["acceleration_score"],
                       status["periodicity_score"], *status["v8_scores"]], float)


def risk(scores, thresholds, margins):
    values = np.asarray(scores, float)
    if values.shape[-1] != 5:
        raise ValueError("Expected five v8 component scores")
    normalized = (values-thresholds)/margins
    result = np.maximum.reduce([normalized[..., 0], np.minimum(normalized[..., 1], normalized[..., 2]),
                                normalized[..., 3], normalized[..., 4]])
    return np.where(np.isfinite(values).all(axis=-1), result, np.inf)


def preview(monitor, probabilities):
    return np.stack([score_status(copy.deepcopy(monitor).update(p)) for p in probabilities])


def stream(main_id, relative_query, name):
    return np.random.default_rng(int(stable_id(PROTOCOL, main_id, relative_query, name)[:8], 16))


def extra_noises(main_id, index):
    return stream(main_id, index, "extra_noise").standard_normal((15, 10, 24)).astype(np.float32)


def guided_population(noises, scores, standard_noise):
    finite = np.flatnonzero(np.isfinite(scores))
    if len(noises) != 8 or len(standard_noise) != 8 or len(finite) < SETTINGS["elite_count"]:
        raise ValueError("Incomplete first population for CEM")
    elite = sorted(finite, key=lambda i: (scores[i], int(i)))[:SETTINGS["elite_count"]]
    values = np.asarray(noises, float)[elite]
    alpha = SETTINGS["cem_update_fraction"]
    mean = alpha*values.mean(axis=0)
    std = np.clip(np.sqrt((1-alpha)+alpha*values.var(axis=0)),
                  SETTINGS["noise_std_floor"], SETTINGS["noise_std_ceiling"])
    result = (mean+std*np.asarray(standard_noise, float)).astype(np.float32)
    return result, dict(elite=[int(i) for i in elite], mean=mean, std=std)


def choose(arm, scores, main_id, index):
    scores = np.asarray(scores, float)
    if arm not in ARMS[1:] or scores.shape != (16,) or not np.isfinite(scores).all():
        raise ValueError("Complete finite candidate pool required")
    if arm.endswith("random"):
        return int(stream(main_id, index, "selection").integers(16))
    return min(range(16), key=lambda i: (scores[i], i))


class RecoveryState:
    def __init__(self):
        self.active, self.queries, self.streak = True, 0, 0
        self.exit_reason = None

    def commit(self, pool_scores):
        if not self.active:
            raise ValueError("Recovery state already exited")
        scores = np.asarray(pool_scores, float)
        complete = scores.shape == (16,) and np.isfinite(scores).all()
        all_low = bool(complete and np.all(scores <= -1-SETTINGS["signal_tolerance"]))
        self.streak = self.streak+1 if all_low else 0
        self.queries += 1
        if self.streak >= SETTINGS["stable_queries"]:
            self.active, self.exit_reason = False, "population_confirmed"
        elif self.queries >= SETTINGS["recovery_queries"]:
            self.active, self.exit_reason = False, "recovery_budget_exhausted"
        return dict(active=self.active, recovery_queries=self.queries, stable_streak=self.streak,
                    all_low=all_low, exit_reason=self.exit_reason)
