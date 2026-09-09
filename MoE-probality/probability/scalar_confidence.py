"""Fixed success-support scores and a monotone two-parameter probability map."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import sha256
from .features import DETECTOR_PATH, detector
from .pure_moe import LOCAL_NAMES, WINDOW, features_from_window


SCHEMA = "moe_success_support_platt_v1"
NEIGHBORS = 20
SCORE_WINDOW = 4
AGGREGATIONS = ("current", "window4", "prefix_mean", "prefix_max")
SCALAR_MODELS = tuple("support_" + name for name in AGGREGATIONS) + ("freeze",)


class SuccessSupport:
    """A reference library fitted only on successful training prefixes."""

    def __init__(self, neighbors=NEIGHBORS):
        self.neighbors = neighbors

    def fit(self, references):
        x = np.asarray(references, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != len(LOCAL_NAMES) or not np.isfinite(x).all()
                or not 1 <= self.neighbors <= len(x)):
            raise ValueError("finite success references and enough neighbors required")
        self.scaler = StandardScaler().fit(x)
        self.search = NearestNeighbors(n_neighbors=self.neighbors, algorithm="brute", metric="euclidean")
        self.search.fit(np.ascontiguousarray(self.scaler.transform(x)))
        self.reference_count = len(x)
        return self

    def score(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(LOCAL_NAMES) or not np.isfinite(x).all():
            raise ValueError("finite local MoE features required")
        if not len(x):
            return np.empty(0)
        distances = self.search.kneighbors(np.ascontiguousarray(self.scaler.transform(x)),
                                           return_distance=True)[0]
        return np.log1p(distances.mean(axis=1))


@dataclass(frozen=True)
class MonotoneSigmoid:
    alpha: float
    beta: float
    objective: float
    calibration_rows: int
    calibration_successes: int

    @classmethod
    def fit(cls, score, outcome):
        u, y = np.asarray(score, float), np.asarray(outcome, float)
        if (u.ndim != 1 or u.shape != y.shape or not np.isfinite(u).all()
                or set(np.unique(y)) != {0.0, 1.0}):
            raise ValueError("finite aligned scores and both binary outcomes required")
        center, scale = float(u.mean()), float(u.std())
        base = float(logit(y.mean()))
        if scale < 1e-12:
            loss = np.logaddexp(0, base) - y.mean() * base
            return cls(0.0, base, float(loss), len(y), int(y.sum()))
        x = (u - center) / scale

        def objective(parameters):
            slope, intercept = parameters
            z = intercept - slope * x
            error = expit(z) - y
            return float(np.mean(np.logaddexp(0, z) - y*z)), np.array([-np.mean(error*x), error.mean()])

        result = minimize(objective, [0.0, base], jac=True, method="L-BFGS-B",
                          bounds=[(0.0, None), (None, None)],
                          options=dict(maxiter=1000, ftol=1e-13, gtol=1e-9))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"probability calibration failed: {result.message}")
        alpha = float(result.x[0] / scale)
        beta = float(result.x[1] + alpha * center)
        return cls(alpha, beta, float(result.fun), len(y), int(y.sum()))

    def predict(self, score):
        u = np.asarray(score, float)
        if not np.isfinite(u).all():
            raise ValueError("nonfinite probability score")
        return expit(self.beta - self.alpha * u)


class ScoreHistory:
    def __init__(self):
        self.recent = deque(maxlen=SCORE_WINDOW)
        self.count = 0
        self.total = 0.0
        self.maximum = -np.inf

    def update(self, score):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("nonfinite support score")
        self.recent.append(score)
        self.count += 1
        self.total += score
        self.maximum = max(self.maximum, score)
        return dict(current=score, window4=float(np.mean(self.recent)),
                    prefix_mean=self.total / self.count, prefix_max=self.maximum)


def aggregate_scores(scores, episode_rows, queries):
    """Consume complete ready prefixes, restarting history at each episode."""
    s, ep, q = np.asarray(scores), np.asarray(episode_rows), np.asarray(queries)
    if s.ndim != 1 or s.shape != ep.shape or s.shape != q.shape or not np.isfinite(s).all():
        raise ValueError("aligned finite episode scores required")
    result = {"support_" + name: np.empty(len(s)) for name in AGGREGATIONS}
    previous, seen, history = None, set(), None
    for i in range(len(s)):
        if i == 0 or ep[i] != previous:
            if ep[i] in seen or q[i] != WINDOW-1:
                raise ValueError("each episode must start at its first ready query")
            seen.add(ep[i])
            history = ScoreHistory()
        elif q[i] != q[i-1]+1:
            raise ValueError("missing or reordered query in score history")
        for name, value in history.update(s[i]).items():
            result["support_" + name][i] = value
        previous = ep[i]
    return result


def freeze_risk(score):
    values = np.asarray(score, float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("finite nonnegative existing freeze scores required")
    return -np.log(np.maximum(values, 1e-8))


class MoEScalarMonitor:
    """Main bounded-window support readout, plus declared prefix diagnostics."""

    def __init__(self, bundle):
        if (bundle["schema"] != SCHEMA or bundle["feature_names"] != LOCAL_NAMES
                or bundle["window"] != WINDOW or bundle["score_window"] != SCORE_WINDOW):
            raise ValueError("scalar confidence schema mismatch")
        if bundle["detector_sha256"] != sha256(DETECTOR_PATH):
            raise ValueError("routing primitive implementation changed")
        self.bundle = bundle
        self.roots = deque(maxlen=WINDOW)
        self.history = ScoreHistory()

    def update(self, hb_router_probs):
        root = detector.root_action_routes(hb_router_probs)
        if root.shape != (8, 10, 32):
            raise ValueError("update accepts exactly one router tensor")
        self.roots.append(root)
        if len(self.roots) < WINDOW:
            return dict(ready=False, success_probability=None, natural_escape_probability=None)
        score = float(self.bundle["support"].score(features_from_window(self.roots))[0])
        aggregated = self.history.update(score)
        probabilities = {name: float(self.bundle["calibrators"]["support_" + name].predict(value))
                         for name, value in aggregated.items()}
        return dict(ready=True, success_probability=probabilities["window4"],
                    support_distance_score=score, aggregated_scores=aggregated,
                    probabilities=probabilities, natural_escape_probability=None)
