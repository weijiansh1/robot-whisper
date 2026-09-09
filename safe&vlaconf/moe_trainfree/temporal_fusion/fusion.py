"""Causal train-free combinations of success kNN and intrinsic route guards."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "boundary_knn"))
from knn import ReferenceScorer, dynamics, reference_scaling
from core import ALPHAS, conformal_threshold, first_alarm, trajectory_peak
from intrinsic_guard_monitor import causal_trailing_mean, persistent_score

METHODS = (
    "knn10", "knn10_persist3", "knn10_delta_q7", "v7_guard", "v8_guard",
    "v82_guard", "knn12_v8", "knn_and_v7", "knn_and_v8", "knn_or_v8",
)
PRIMARY = "knn_and_v8"
V7_HEADS = ("freeze", "acceleration_persistent", "periodicity_persistent")


def flow_speed(raw):
    p = np.maximum(np.asarray(raw, np.float32), 0)
    if p.shape[-4:] != (8, 10, 11, 32) or not np.isfinite(p).all():
        raise ValueError("expected finite all-flow probabilities [...,8,10,11,32]")
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    root = np.sqrt(p[..., 1:, :])
    delta = np.diff(root, axis=-3)
    return (np.linalg.norm(delta, axis=-1) / np.sqrt(2.0)).mean(axis=-1).astype(np.float32)


def raw_v8_features(speed):
    speed = np.asarray(speed, np.float32)
    if speed.shape[-2:] != (8, 9):
        raise ValueError("expected [...,8,9] flow speed")
    # Ordinary sum preserves missing queries instead of inventing zero paths.
    path = speed.sum(axis=-1)
    front, back = path[..., :4].mean(-1), path[..., 4:].mean(-1)
    inversion = -np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
    sub = speed[..., 4:, :][..., [0, 4, 8]]
    curvature = np.abs(np.diff(sub, n=2, axis=-1)).mean(axis=(-2, -1))
    result = np.stack((inversion, curvature), axis=-1).astype(np.float32)
    result[~np.isfinite(speed).all(axis=(-2, -1))] = np.nan
    return result


def v8_heads(raw):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 2:
        raise ValueError("expected [episode,query,2] v8 inputs")
    output = np.full_like(raw, np.nan)
    if raw.shape[1] < 7:
        return output
    base = raw[:, 1:5, 1].mean(axis=1, dtype=np.float32)
    relative = np.log(np.maximum(raw[..., 1], 1e-12) / np.maximum(base[:, None], 1e-12))
    output[..., 0] = causal_trailing_mean(raw[..., 0], 6)
    output[..., 1] = causal_trailing_mean(relative, 6)
    output[:, :6] = np.nan
    output[~np.isfinite(raw).all(-1)] = np.nan
    return output


def prefix_peak(values):
    return np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf), axis=1)


def base_streams(profile, mobility, acceleration, periodicity, raw, scorer=None):
    dynamic, intrinsic = dynamics(mobility, acceleration, periodicity, float(profile["periodicity_scale"]))
    valid = np.isfinite(raw).all(-1)
    heads = v8_heads(raw)
    normalized_heads = (heads.astype(np.float64) - profile["v8_center"]) / profile["v8_scale"]
    normalized_dynamic = (dynamic.astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
    augmented = np.concatenate((normalized_dynamic, normalized_heads), axis=-1)
    output = {}
    scorer = scorer or ReferenceScorer(profile)
    for name, bank, vectors in (("knn10", "success_dynamic", normalized_dynamic),
                                ("knn12_v8", "success_augmented", augmented)):
        finite = valid & np.isfinite(vectors).all(-1)
        values = np.full(valid.shape, np.nan, np.float32)
        if finite.any():
            distance, _ = scorer.neighbors(bank, vectors[finite])
            values[finite] = distance.mean(-1)
        output[name] = values
    normalized_v7 = [(intrinsic[name] - float(profile["v7_center"][i])) / float(profile["v7_scale"][i])
                     for i, name in enumerate(V7_HEADS)]
    turbulence = np.minimum(prefix_peak(normalized_v7[1]), prefix_peak(normalized_v7[2]))
    turbulence[~np.isfinite(turbulence)] = np.nan
    output["v7_guard"] = np.fmax(normalized_v7[0], turbulence).astype(np.float32)
    for name, relaxation in (("v8_guard", 0.0), ("v82_guard", .0015)):
        relaxed = heads.astype(np.float64) + relaxation * np.arange(heads.shape[1])[None, :, None]
        scaled = (relaxed - profile["v8_center"]) / profile["v8_scale"]
        confirmed = [persistent_score(scaled[..., i], 2) for i in range(2)]
        output[name] = np.fmax(np.fmax(output["v7_guard"], confirmed[0]), confirmed[1])
    for values in output.values():
        values[~valid] = np.nan
    return output


def combine_streams(base, profile):
    result = dict(base)
    k = base["knn10"]
    result["knn10_persist3"] = persistent_score(k, 3)
    result["knn10_delta_q7"] = k - k[:, 7:8] if k.shape[1] >= 8 else np.full_like(k, np.nan)
    scaled = [(base[name] - profile["fusion_center"][i]) / profile["fusion_scale"][i]
              for i, name in enumerate(("knn10", "v7_guard", "v8_guard"))]
    result["knn_and_v7"] = np.minimum(scaled[0], scaled[1])
    result["knn_and_v8"] = np.minimum(scaled[0], scaled[2])
    result["knn_or_v8"] = np.fmax(scaled[0], scaled[2])
    return np.asarray([result[name] for name in METHODS], np.float32)


def calibrate(scores, labels, frame, tests):
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("expected historical calibration outcomes")
    success = np.asarray(labels) == 0
    groups = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(ALPHAS), len(METHODS)))
    first = np.empty((*thresholds.shape, tests.shape[1]), np.int16)
    records = []
    for mi, method in enumerate(METHODS):
        peaks = trajectory_peak(scores[mi, success])
        grouped = np.asarray([peaks[groups == key].max() for key in sorted(set(groups))])
        for ki, (kind, units) in enumerate((("episode", peaks), ("task_init", grouped))):
            for ai, alpha in enumerate(ALPHAS):
                tau, rank = conformal_threshold(units, alpha)
                thresholds[ki, ai, mi] = tau
                first[ki, ai, mi] = first_alarm(tests[mi], tau)
                records.append(dict(method=method, calibration=kind, alpha=alpha, threshold=tau,
                                    rank=rank, units=len(units), exceedances=int((units > tau).sum())))
    return thresholds, first, records
