"""Fixed MoE readouts, labeled reference distances, and trajectory calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
RUNS = ("right-50x8-20260903", "right-50x8b-20260903")
SEEDS = (20260907, 20260908, 20260909)
ALPHAS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)
PRIMARY = "stats_contrast__cumsum"
VECTOR_NAMES = ("load", "stats", "history", "behavior")
BASE_NAMES = ("load_contrast", "stats_contrast", "history_contrast", "behavior_contrast",
              "stats_success", "entropy_low", "entropy_high", "token_js_low", "freeze",
              "action_translation_low", "eef_motion_low")
METHODS = tuple(f"{name}__{mode}" for name in BASE_NAMES for mode in ("current", "cumsum")) + ("clock", "random")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def route_features(raw):
    """One episode, final flow [query,8,11,32]; no future normalization."""
    p = np.asarray(raw, np.float64)
    if p.ndim != 4 or p.shape[1:] != (8, 11, 32):
        raise ValueError("expected [query,8,11,32]")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("invalid router probability")
    p = p / p.sum(-1, keepdims=True)
    action = p[:, :, 1:]
    root = np.sqrt(action)
    load = action.mean(2)
    entropy = -(action * np.log(np.maximum(action, 1e-30))).sum(-1).mean(-1)
    load_entropy = -(load * np.log(np.maximum(load, 1e-30))).sum(-1)
    top = np.partition(action, -2, axis=-1)[..., -2:]
    margin = (top[..., 1] - top[..., 0]).mean(-1)
    gap = (np.linalg.norm(root - np.sqrt(p[:, :, :1]), axis=-1) / np.sqrt(2)).mean(-1)
    js = np.maximum(load_entropy - entropy, 0)
    stats = np.stack((entropy / np.log(32), margin, js / np.log(32), gap), axis=-1)
    mobility = np.full((len(p), 8), np.nan)
    mobility[1:] = (np.linalg.norm(root[1:] - root[:-1], axis=-1) / np.sqrt(2)).mean(-1)
    relative = np.full_like(mobility, np.nan)
    if len(p) > 5:
        base = mobility[1:5].mean(0)
        relative[5:] = np.log(np.maximum(mobility[5:], 1e-10) / np.maximum(base, 1e-10))
        relative[:, base <= 1e-10] = np.nan
    history = np.concatenate((stats.reshape(len(p), -1), relative), axis=-1)
    direct = np.stack((-entropy[:, 4:].mean(-1), entropy[:, 4:].mean(-1),
                       -js[:, 4:].mean(-1), -np.median(relative[:, 4:], axis=-1)), axis=-1)
    return {"load": np.sqrt(load).reshape(len(p), -1).astype(np.float32),
            "stats": stats.reshape(len(p), -1).astype(np.float32),
            "history": history.astype(np.float32), "direct": direct.astype(np.float32)}


def behavior_features(state, actions, action_std):
    state, actions = np.asarray(state, float), np.asarray(actions, float)
    scaled = actions / np.asarray(action_std)[None, None]
    translation = np.linalg.norm(scaled[..., :3], axis=-1).mean(-1)
    rotation = np.linalg.norm(scaled[..., 3:6], axis=-1).mean(-1)
    behavior = np.stack((translation, rotation, scaled[..., 6].mean(-1), scaled[..., 6].std(-1)), axis=-1)
    motion = np.full(len(state), np.nan)
    delta = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=-1)
    for q in range(1, len(state)):
        motion[q] = delta[max(0, q - 4):q].mean()
    return behavior.astype(np.float32), np.stack((-translation, -motion), axis=-1).astype(np.float32)


def reference_points(values, rows, cap=4096, seed=0):
    points = []
    for row in rows:
        finite = np.flatnonzero(np.isfinite(values[row]).all(-1))
        if len(finite):
            index = finite[np.unique(np.linspace(0, len(finite) - 1, min(8, len(finite))).astype(int))]
            points.append(values[row, index])
    if not points:
        raise ValueError("empty reference class")
    points = np.concatenate(points)
    if len(points) > cap:
        points = points[np.random.default_rng(seed).choice(len(points), cap, replace=False)]
    return points


class ReferenceDistance:
    def __init__(self, success, failure, normalize=True):
        both = np.concatenate((success, failure))
        self.center = np.median(both, axis=0) if normalize else np.zeros(both.shape[-1])
        self.scale = np.maximum(1.4826 * np.median(np.abs(both - self.center), axis=0), 1e-6) if normalize else np.ones(both.shape[-1])
        self.success = self._index(success)
        self.failure = self._index(failure)

    def _index(self, values):
        return NearestNeighbors(n_neighbors=min(5, len(values)), algorithm="brute", metric="euclidean", n_jobs=1).fit((values - self.center) / self.scale)

    def score(self, values):
        output = np.full((*values.shape[:-1], 2), np.nan, dtype=np.float32)
        valid = np.isfinite(values).all(-1)
        x = (values[valid] - self.center) / self.scale
        found = np.empty((len(x), 2), dtype=np.float32)
        for start in range(0, len(x), 1024):
            batch = x[start:start + 1024]
            ds = self.success.kneighbors(batch, return_distance=True)[0].mean(-1) / np.sqrt(x.shape[-1])
            df = self.failure.kneighbors(batch, return_distance=True)[0].mean(-1) / np.sqrt(x.shape[-1])
            found[start:start + len(batch)] = np.stack((ds - df, ds), axis=-1)
        output[valid] = found
        return output


def aggregate(values, mode):
    if mode == "current":
        return values.copy()
    if mode != "cumsum":
        raise ValueError(mode)
    finite = np.isfinite(values)
    total = np.cumsum(np.where(finite, values, 0), axis=1, dtype=np.float64)
    return np.where(finite, total, np.nan).astype(np.float32)


def reference_band(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("no successful reference observations")
    global_center = np.median(finite)
    floor = max(0.1 * 1.4826 * np.median(np.abs(finite - global_center)), 1e-6)
    center, scale = [], []
    last_center, last_scale = global_center, max(floor * 10, 1e-6)
    counts = np.isfinite(values).sum(axis=0)
    for q in range(values.shape[1]):
        x = values[:, q]
        x = x[np.isfinite(x)]
        if len(x) >= 20:
            last_center = np.median(x)
            last_scale = max(1.4826 * np.median(np.abs(x - last_center)), floor)
        center.append(last_center)
        scale.append(last_scale)
    return np.asarray(center), np.asarray(scale), counts


def trajectory_peak(values):
    return np.max(np.where(np.isfinite(values), values, -np.inf), axis=1)


def conformal_threshold(peaks, alpha):
    if not len(peaks) or not 0 < alpha < 1 or np.isnan(peaks).any():
        raise ValueError("invalid calibration")
    rank = int(np.ceil((len(peaks) + 1) * (1 - alpha)))
    return (float(np.sort(peaks)[rank - 1]) if rank <= len(peaks) else np.inf), rank


def first_alarm(values, threshold):
    crossed = np.isfinite(values) & (values > threshold)
    return np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1).astype(np.int16)
