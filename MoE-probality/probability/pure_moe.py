"""A fixed-window readout with no budget, clock, or model-selection metadata."""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import sha256
from .features import DETECTOR_PATH, FEATURE_NAMES, SIGNAL_NAMES, detector
from .model import MODEL_PARAMS, log_odds


WINDOW = 8
LOCAL_NAMES = [f"{group}_{signal}_{stat}"
               for group in ("front", "back") for signal in SIGNAL_NAMES
               for stat in ("current", "mean4", "change3")]
LOCAL_INDICES = [FEATURE_NAMES.index(name) for name in LOCAL_NAMES]


def select_local_features(cached):
    x = np.asarray(cached)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected source feature schema")
    local = x[:, LOCAL_INDICES]
    if not np.isfinite(local).all():
        raise ValueError("local readout requires a full eight-query window")
    return local


def features_from_window(roots):
    """Each scalar uses only the eight root-probability tensors in this window."""
    roots = np.asarray(roots)
    if roots.shape != (WINDOW, 8, 10, 32):
        raise ValueError("expected exactly eight [8,10,32] root-probability tensors")
    # Cache extraction rounds each primitive to float32 before window aggregation.
    primitives = detector.features_from_roots(roots).astype(np.float32).astype(np.float64)
    output = []
    for selected in (slice(0, 4), slice(4, 8)):
        for k in range(3):
            signal = np.median(primitives[-4:, selected, k], axis=1)
            output.extend([signal[-1], signal.mean(), signal[-1] - signal[0]])
    result = np.asarray(output, dtype=np.float32)[None]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite routing window")
    return result


def fit_calibrated(x_train, y_train, x_cal, y_cal):
    if len(np.unique(y_train)) != 2 or len(np.unique(y_cal)) != 2:
        raise ValueError("both classes required in training and calibration")
    estimator = HistGradientBoostingClassifier(**MODEL_PARAMS)
    estimator.fit(x_train, y_train)
    calibrator = LogisticRegression(C=1e6, max_iter=200, random_state=20260907)
    calibrator.fit(log_odds(estimator.predict_proba(x_cal)[:, 1]), y_cal)
    return dict(estimator=estimator, calibrator=calibrator)


def probabilities(model, x):
    raw = model["estimator"].predict_proba(x)[:, 1]
    calibrated = model["calibrator"].predict_proba(log_odds(raw))[:, 1]
    return raw, calibrated


class MoEWindowMonitor:
    """A shared model for all suites; no task, budget, or elapsed-step arguments."""

    def __init__(self, bundle):
        if bundle["feature_names"] != LOCAL_NAMES or bundle["window"] != WINDOW:
            raise ValueError("window feature schema mismatch")
        if bundle["detector_sha256"] != sha256(DETECTOR_PATH):
            raise ValueError("routing primitive implementation changed")
        self.bundle = bundle
        self.roots = deque(maxlen=WINDOW)

    def update(self, hb_router_probs):
        root = detector.root_action_routes(hb_router_probs)
        if root.shape != (8, 10, 32):
            raise ValueError("update accepts exactly one router tensor")
        self.roots.append(root)
        if len(self.roots) < WINDOW:
            return dict(ready=False, success_probability=None)
        _, p = probabilities(self.bundle["model"], features_from_window(self.roots))
        return dict(ready=True, success_probability=float(p[0]))
