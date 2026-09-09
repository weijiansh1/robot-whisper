"""Fixed-cap success readouts with calibration on disjoint initial states."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import sha256
from .features import BUDGET_NAMES, DETECTOR_PATH, FEATURE_NAMES, PrefixFeatures


MODEL_PARAMS = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
                    min_samples_leaf=50, l2_regularization=5.0,
                    early_stopping=False, random_state=20260907)


def log_odds(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_readout(x_train, y_train, x_cal, y_cal, policy):
    if len(np.unique(y_train)) != 2 or len(np.unique(y_cal)) != 2:
        raise ValueError("training and calibration each require both terminal outcomes")
    estimators = {}
    for name, indices in (("budget", list(range(len(BUDGET_NAMES)))),
                          ("moe", list(range(len(FEATURE_NAMES))))):
        model = HistGradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(x_train[:, indices], y_train)
        calibrator = LogisticRegression(C=1e6, max_iter=200, random_state=20260907)
        calibrator.fit(log_odds(model.predict_proba(x_cal[:, indices])[:, 1]), y_cal)
        estimators[name] = dict(estimator=model, calibrator=calibrator, indices=indices)
    return dict(schema_version=1, feature_names=FEATURE_NAMES, estimators=estimators,
                prior=float(np.mean(y_train)), policy=policy,
                detector_sha256=sha256(DETECTOR_PATH), model_params=MODEL_PARAMS,
                estimand="P(success_by_original_deadline | causal_MoE_history,budget,active_query)",
                rho_status="unavailable: independent trap and escape labels required")


def predict_readout(bundle, x):
    result = {"prior": np.full(len(x), bundle["prior"])}
    for name, item in bundle["estimators"].items():
        p = item["estimator"].predict_proba(x[:, item["indices"]])[:, 1]
        result[name + "_raw"] = p
        result[name] = item["calibrator"].predict_proba(log_odds(p))[:, 1]
    return result


class SuccessMonitor:
    """One instance per rollout; update after inference and before action execution."""

    def __init__(self, bundle, checkpoint_sha256: str):
        if bundle["feature_names"] != FEATURE_NAMES or bundle["detector_sha256"] != sha256(DETECTOR_PATH):
            raise ValueError("model feature/detector version mismatch")
        if checkpoint_sha256 != bundle["policy"]["checkpoint_sha256"]:
            raise ValueError("this readout was calibrated for a different policy checkpoint")
        self.bundle = bundle
        self.features = PrefixFeatures(bundle["policy"]["max_steps"], bundle["policy"]["n_action_steps"])

    def update(self, hb_router_probs):
        x, info = self.features.update(hb_router_probs)
        p = predict_readout(self.bundle, x)
        return dict(query=len(self.features.history) - 1, success_probability=float(p["moe"][0]),
                    budget_baseline_probability=float(p["budget"][0]),
                    remaining_action_steps=int(x[0, 0]), trap_probability=None,
                    natural_escape_probability=None, **info)

    @staticmethod
    def terminal_probability(*, success=False, irreversible_failure=False, remaining_action_steps=0):
        if success and irreversible_failure:
            raise ValueError("conflicting terminal state")
        if success:
            return 1.0
        if irreversible_failure or remaining_action_steps == 0:
            return 0.0
        raise ValueError("nonterminal state requires an inference update")
