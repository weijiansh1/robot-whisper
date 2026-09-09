"""Fixed MoE statistics and unlabeled reference calibration for paper audits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V7_METHOD = ROOT / "moe-v7-0905/method"
sys.path.insert(0, str(V7_METHOD))

from intrinsic_guard_monitor import (  # noqa: E402
    first_from_score, intrinsic_score_arrays, persistent_score,
)
from unlabeled_budget_calibration import (  # noqa: E402
    alarms_from_scores, calibrate_reference,
)

SEED = 20260906
BUDGETS = (0.01, 0.03, 0.05, 0.10)
CHECKPOINTS = (0, 3, 7, 11, 15, 19)
PRIMARY = "v7_unlabeled_budget"
BASE_NAMES = (
    "entropy_high", "entropy_low", "margin_low", "token_collapse",
    "flow_path_high", "frontback_inversion", "entropy_front_low",
    "entropy_first_flow_low", "token_front_collapse", "freeze",
    "deviation_global", "deviation_step",
)
METHOD_NAMES = tuple(f"{name}__{agg}" for name in BASE_NAMES
                     for agg in ("current", "mean", "max")) + ("clock", "random")
ONLINE_NAMES = tuple(n for n in METHOD_NAMES if not n.endswith("__max"))
META_FIELDS = ("task", "episode", "init_state_id", "flow_noise_seed", "length", "run_id")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def prefix_aggregate(values: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(x)
    if mode == "current":
        return x.copy()
    if mode == "mean":
        count = np.cumsum(finite, axis=1)
        total = np.cumsum(np.where(finite, x, 0), axis=1, dtype=np.float64)
        out = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    elif mode == "max":
        out = np.maximum.accumulate(np.where(finite, x, -np.inf), axis=1)
        out = np.where(np.isfinite(out), out, np.nan)
    else:
        raise ValueError(mode)
    return out.astype(np.float32)


def peaks(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.max(np.where(valid & np.isfinite(values), values, -np.inf), axis=1)


def budget_threshold(values: np.ndarray, valid: np.ndarray, budget: float) -> float:
    if not 0 <= budget < 1 or len(values) == 0:
        raise ValueError("invalid budget or empty reference")
    ordered = np.sort(peaks(values, valid))
    allowance = int(np.floor(len(ordered) * budget))
    return float(ordered[len(ordered) - allowance - 1])


def robust_stats(x: np.ndarray, by_step: bool):
    if by_step:
        enough = np.isfinite(x).all(axis=-1).sum(axis=0) >= 32
        center = np.full(x.shape[1:], np.nan, dtype=np.float32)
        scale = np.full_like(center, np.nan)
        center[enough] = np.nanmedian(x[:, enough], axis=0)
        scale[enough] = 1.4826 * np.nanmedian(np.abs(x[:, enough] - center[enough]), axis=0)
    else:
        rows = x.reshape(-1, x.shape[-1])
        rows = rows[np.isfinite(rows).all(axis=-1)]
        if len(rows) < 32:
            raise ValueError("insufficient finite reference rows")
        center = np.median(rows, axis=0)
        scale = 1.4826 * np.median(np.abs(rows - center), axis=0)
        enough = np.asarray(len(rows))
    return center.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32), enough


def freeze_score(mobility: np.ndarray) -> np.ndarray:
    n, q, _ = mobility.shape
    if q < 6:
        return np.full((n, q), np.nan, dtype=np.float32)
    unused = np.full((n, q), np.nan, dtype=np.float32)
    scores = intrinsic_score_arrays(mobility, unused, unused, 1.0)["freeze"]
    scores[:, :5] = np.nan
    return scores


def build_scores(data: dict, stats: dict) -> np.ndarray:
    raw = data["raw"]
    # The raw interface is fixed; no label or task identity enters scoring.
    values = [raw[:, :, 0], -raw[:, :, 0], -raw[:, :, 1], -raw[:, :, 2],
              raw[:, :, 3], raw[:, :, 4], -raw[:, :, 6], -raw[:, :, 7],
              -raw[:, :, 8], freeze_score(data["mobility"])]
    for kind in ("global", "step"):
        center, scale, _ = stats[kind]
        values.append(np.mean(np.abs((raw[:, :, :6] - center) / scale), axis=-1))
    output = []
    for value in values:
        for mode in ("current", "mean", "max"):
            aggregate = prefix_aggregate(value, mode)
            output.append(np.where(data["valid"], aggregate, np.nan))
    clock = np.broadcast_to(np.arange(raw.shape[1], dtype=np.float32), raw.shape[:2])
    output.extend([np.where(data["valid"], clock, np.nan), data["random"]])
    return np.stack(output, axis=0).astype(np.float32)


def fit_stats(data: dict) -> dict:
    x = np.where(data["valid"][:, :, None], data["raw"][:, :, :6], np.nan)
    return {"global": robust_stats(x, False), "step": robust_stats(x, True)}


def subset(data: dict, rows: np.ndarray) -> dict:
    n = len(data["valid"])
    return {k: v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v
            for k, v in data.items()}


def v7_alarms(reference: dict, test: dict, budget: float):
    fit = calibrate_reference(reference["mobility"], reference["acceleration"],
                              reference["periodicity"], reference["valid"],
                              alarm_budget=budget)
    scores = intrinsic_score_arrays(test["mobility"], test["acceleration"],
                                    test["periodicity"], fit.profile.periodicity_scale)
    first = alarms_from_scores(scores, test["valid"], fit.profile)["guard"]
    return first, fit


def score_at(values: np.ndarray, position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=int)
    if np.any(position < 0) or np.any(position >= values.shape[1]):
        raise ValueError("invalid checkpoint")
    return values[np.arange(len(values)), position]


def binary_metrics(first: np.ndarray, failure: np.ndarray, length: np.ndarray) -> dict:
    fired = first >= 0
    tp = int((fired & failure).sum())
    fp = int((fired & ~failure).sum())
    p, n = int(failure.sum()), int((~failure).sum())
    recall = tp / p if p else np.nan
    fpr = fp / n if n else np.nan
    lag = np.where(fired, length - 1 - first, -1)
    normalized = np.where(fired, first / np.maximum(length - 1, 1), 1.0)
    result = {"episodes": len(first), "failures": p, "successes": n,
              "tp": tp, "fp": fp, "recall": recall, "fpr": fpr,
              "precision": tp / (tp + fp) if tp + fp else np.nan,
              "balanced_accuracy": (recall + 1 - fpr) / 2,
              "alarm_rate": float(fired.mean()),
              "t_det": float(normalized[failure].mean()) if p else np.nan,
              "median_lead_detected": float(np.median(lag[fired & failure])) if tp else np.nan}
    for lead in (0, 2, 4, 8):
        hits = int((fired & failure & (lag >= lead)).sum())
        result[f"tp_lead{lead}"] = hits
        result[f"recall_lead{lead}"] = hits / p if p else np.nan
    return result
