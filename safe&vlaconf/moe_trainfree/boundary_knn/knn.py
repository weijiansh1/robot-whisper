"""Reference geometry, exact neighbors, and constant trajectory thresholds."""

from pathlib import Path
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_protocol"))
sys.path.insert(0, str(HERE.parent / "feature_geometry"))
from core import ALPHAS, ROOT, SEEDS, conformal_threshold, first_alarm, trajectory_peak
from analyze import dynamics, reference_scaling
from intrinsic_guard_monitor import persistent_score

PRIMARY = "dyn_success_knn_k20"
BASE_METHODS = ("dyn_radius", "dyn_success_knn_k1", "dyn_success_knn_k5", PRIMARY,
                "dyn_vote_knn_k20", "dyn_pca2_success_knn_k20", "route_success_knn_k20")
METHODS = ("dyn_radius", "dyn_radius_outward", "dyn_success_knn_k1", "dyn_success_knn_k5", PRIMARY,
           "dyn_success_knn_k20_persist3", "dyn_vote_knn_k20", "dyn_pca2_success_knn_k20",
           "route_success_knn_k20", "eef_motion_low", "clock_q7")


def reference_pairs(values, rows, cap=4096, seed=0):
    pairs = []
    for row in rows:
        finite = np.flatnonzero(np.isfinite(values[row]).all(-1) & (np.arange(values.shape[1]) >= 7))
        if len(finite):
            queries = finite[np.linspace(0, len(finite) - 1, min(8, len(finite))).astype(int)]
            pairs.extend((int(row), int(q)) for q in queries)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) < 20:
        raise ValueError("fewer than 20 reference points")
    if len(pairs) > cap:
        pairs = pairs[np.random.default_rng(seed).choice(len(pairs), cap, replace=False)]
    return pairs


class ReferenceScorer:
    def __init__(self, profile):
        self.profile = profile
        self.indices = {}

    def neighbors(self, bank_name, values):
        if bank_name not in self.indices:
            bank = np.asarray(self.profile[bank_name], dtype=np.float64)
            self.indices[bank_name] = NearestNeighbors(n_neighbors=20, algorithm="auto", n_jobs=1).fit(bank)
        distances, identities = [], []
        for start in range(0, len(values), 1024):
            distance, identity = self.indices[bank_name].kneighbors(values[start:start + 1024])
            distances.append(distance)
            identities.append(identity)
        return np.concatenate(distances), np.concatenate(identities)

    def current(self, dynamic, routing, methods=BASE_METHODS):
        dynamic, routing = np.asarray(dynamic), np.asarray(routing)
        if dynamic.shape[:-1] != routing.shape[:-1] or dynamic.shape[-1] != 10 or routing.shape[-1] != 256:
            raise ValueError("expected aligned 10-D dynamics and 256-D routing")
        valid = np.isfinite(dynamic).all(-1) & np.isfinite(routing).all(-1)
        output = {name: np.full(dynamic.shape[:-1], np.nan, np.float32) for name in methods}
        if not valid.any():
            return output
        p = self.profile
        d = ((dynamic[valid].astype(np.float64) - p["dynamic_center"]) / p["dynamic_scale"])
        if "dyn_radius" in methods:
            output["dyn_radius"][valid] = np.linalg.norm(d, axis=-1)
        requested = [k for k in (1, 5, 20) if f"dyn_success_knn_k{k}" in methods]
        if requested:
            distance, _ = self.neighbors("success_dynamic", d)
            for k in requested:
                output[f"dyn_success_knn_k{k}"][valid] = distance[:, :k].mean(-1)
        if "dyn_vote_knn_k20" in methods:
            _, neighbors = self.neighbors("mixture_dynamic", d)
            output["dyn_vote_knn_k20"][valid] = p["mixture_failure"][neighbors].mean(-1)
        if "dyn_pca2_success_knn_k20" in methods:
            coordinates = (d - p["pca_mean"]) @ p["pca_components"].T
            distance, _ = self.neighbors("success_pca2", coordinates)
            output["dyn_pca2_success_knn_k20"][valid] = distance.mean(-1)
        if "route_success_knn_k20" in methods:
            r = (routing[valid].astype(np.float64) - p["routing_center"]) / p["routing_scale"]
            distance, _ = self.neighbors("success_routing", r)
            output["route_success_knn_k20"][valid] = distance.mean(-1)
        return output


def score_streams(profile, dynamic, routing, direct):
    values = ReferenceScorer(profile).current(dynamic, routing)
    radius = values["dyn_radius"]
    values["dyn_radius_outward"] = radius - radius[:, 7:8]
    values["dyn_success_knn_k20_persist3"] = persistent_score(values[PRIMARY], 3)
    valid = np.isfinite(radius)
    values["eef_motion_low"] = np.where(valid, direct[..., 5], np.nan)
    values["clock_q7"] = np.where(valid, np.arange(radius.shape[1])[None], np.nan)
    return np.stack([values[name] for name in METHODS]).astype(np.float32)


def calibrate_constant(calibration_scores, labels, frame, test_scores):
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("calibration labels must be known historical outcomes")
    success = labels == 0
    groups = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(ALPHAS), len(METHODS)))
    first = np.empty((*thresholds.shape, test_scores.shape[1]), np.int16)
    records = []
    for method_i, method in enumerate(METHODS):
        peaks = trajectory_peak(calibration_scores[method_i, success])
        grouped = np.asarray([peaks[groups == key].max() for key in sorted(set(groups))])
        for kind_i, (kind, units) in enumerate((("episode", peaks), ("task_init", grouped))):
            for alpha_i, alpha in enumerate(ALPHAS):
                threshold, rank = conformal_threshold(units, alpha)
                exceedances = int((units > threshold).sum())
                assert exceedances <= len(units) + 1 - rank
                thresholds[kind_i, alpha_i, method_i] = threshold
                first[kind_i, alpha_i, method_i] = first_alarm(test_scores[method_i], threshold)
                records.append({"method": method, "kind": kind, "alpha": alpha,
                    "calibration_units": len(units), "rank": rank, "threshold": threshold,
                    "exceedances": exceedances, "success_calibration_episodes": int(success.sum())})
    return thresholds, first, records
