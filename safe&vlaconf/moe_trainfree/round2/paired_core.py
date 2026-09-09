"""Outcome-free scores and disjoint statistical-reference/calibration splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
SEED = 20260907
BUDGETS = (0.01, 0.03, 0.05, 0.10)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def grouped_folds(groups):
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 4 or len(unique) % 4:
        raise ValueError("expected a multiple of four reference groups")
    fold = inverse % 4
    for f in range(4):
        test = np.flatnonzero(fold == f)
        calibration = np.flatnonzero(fold == (f + 1) % 4)
        reference = np.flatnonzero((fold != f) & (fold != (f + 1) % 4))
        yield reference, calibration, test


def distance_scores(reference, target):
    reference = np.asarray(reference, np.float64)
    target = np.asarray(target, np.float64)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise ValueError("unaligned reference and target matrices")
    if len(reference) < 5 or not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("need five finite reference samples and finite targets")
    center = np.median(reference, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(reference - center), axis=0), 1e-6)
    ref = (reference - center) / scale
    x = (target - center) / scale
    deviation = np.mean(np.abs(x), axis=1)
    distance = np.sqrt(cdist(x, ref, "sqeuclidean") / x.shape[1])
    nearest = np.partition(distance, 4, axis=1)[:, :5].mean(axis=1)
    return {"deviation": deviation, "knn5": nearest}


def threshold(calibration, budget):
    values = np.sort(np.asarray(calibration, np.float64))
    if not len(values) or not np.isfinite(values).all() or not 0 <= budget < 1:
        raise ValueError("invalid calibration scores or budget")
    return float(values[len(values) - int(np.floor(len(values) * budget)) - 1])


def route_features(probabilities):
    """Input [episode, flow, layer, action token, expert]."""
    p = np.asarray(probabilities, np.float64)
    if p.ndim != 5 or np.any(p < 0) or np.any(p.sum(axis=-1) <= 0):
        raise ValueError("invalid router probabilities")
    p = p / p.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1)
    load = p.mean(axis=3)
    load_entropy = -(load * np.log(np.maximum(load, 1e-12))).sum(axis=-1)
    largest = np.partition(p, -2, axis=-1)[..., -2:]
    margin = largest.max(axis=-1) - largest.min(axis=-1)
    return np.sqrt(p).astype(np.float32), {
        "route_entropy_low": -entropy.mean(axis=(2, 3)),
        "route_margin_low": -margin.mean(axis=(2, 3)),
        "route_token_collapse": -(load_entropy - entropy.mean(axis=3)).mean(axis=2),
    }


def flow_views(value):
    return {"d0": value[:, 0], "d9": value[:, -1], "flowmean": value.mean(axis=1)}
