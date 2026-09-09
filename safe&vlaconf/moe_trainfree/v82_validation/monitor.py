"""Prefix-causal guard replay and trajectory-level calibration primitives."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "moe-v7-0905/method"))
from intrinsic_guard_monitor import intrinsic_score_arrays  # noqa: E402

METHODS = (
    "v7", "v7_frontback", "v7_curvature", "v82", "v8_fixed",
    "v82_no_smoothing", "v82_no_confirmation", "v82_no_curvature_baseline",
    "frontback", "curvature", "eef_motion_low", "clock",
)
ALPHAS = (0.005, 0.01, 0.02, 0.05)


def trailing(values, width):
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    for q in range(width - 1, values.shape[1]):
        window = values[:, q - width + 1:q + 1]
        good = np.isfinite(window).all(axis=1)
        result[good, q] = window[good].mean(axis=1)
    return result


def persistent(values, count):
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    for q in range(count - 1, values.shape[1]):
        window = values[:, q - count + 1:q + 1]
        good = np.isfinite(window).all(axis=1)
        result[good, q] = window[good].min(axis=1)
    return result


def prefix_peak(values):
    return np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf), axis=1)


def first_alarm(values, valid, threshold=0.0, inclusive=False):
    values = np.asarray(values)
    valid = np.asarray(valid, bool)
    if values.shape != valid.shape:
        raise ValueError("score/validity shape mismatch")
    comparison = values >= threshold if inclusive else values > threshold
    hit = comparison & np.isfinite(values) & valid
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1).astype(np.int16)


def union(*alarms):
    stack = np.asarray(alarms, dtype=np.int64)
    values = np.where(stack < 0, 1_000_000, stack).min(axis=0)
    return np.where(values == 1_000_000, -1, values).astype(np.int16)


def first_and(left, right):
    return np.where((left >= 0) & (right >= 0), np.maximum(left, right), -1).astype(np.int16)


def clean_first(first, length):
    first = np.asarray(first)
    return np.where((first >= 0) & (first < length), first, -1).astype(np.int16)


def v8_streams(raw, valid, width=6, relative=True):
    raw = np.asarray(raw, np.float32).copy()
    raw[~valid] = np.nan
    curvature = raw[:, :, 1]
    if relative:
        # The opening baseline is only exposed after it has been observed.
        base = curvature[:, 1:5].mean(axis=1, dtype=np.float32)
        curvature = np.log(np.maximum(curvature, 1e-12) / np.maximum(base[:, None], 1e-12))
    else:
        curvature = np.log(np.maximum(curvature, 1e-12))
    result = np.stack((trailing(raw[:, :, 0], width), trailing(curvature, width)), axis=-1)
    result[:, :6] = np.nan
    result[~valid] = np.nan
    return result


def fit_location(values):
    samples = []
    for row in np.asarray(values):
        queries = np.flatnonzero(np.isfinite(row))
        if len(queries):
            chosen = np.linspace(0, len(queries) - 1, min(8, len(queries))).astype(int)
            samples.extend(row[queries[chosen]])
    if not samples:
        raise ValueError("no finite reference scores")
    sample = np.asarray(samples, np.float64)
    center = float(np.median(sample))
    scale = max(float(1.4826 * np.median(np.abs(sample - center))), 1e-6)
    return dict(center=center, scale=scale, samples=len(sample))


def scale_values(values, profile):
    return (values - profile["center"]) / profile["scale"]


def build_budget_streams(cache, raw, eef, reference):
    valid = cache["valid"]
    period = cache["periodicity"][reference]
    pscale = float(np.quantile(np.abs(period[np.isfinite(period)]), .75))
    v7 = intrinsic_score_arrays(cache["mobility"], cache["acceleration"], cache["periodicity"], pscale)
    profile = {"periodicity_scale": pscale, "heads": {}}

    def normalized(name, values):
        stats = fit_location(values[reference])
        profile["heads"][name] = stats
        return scale_values(values, stats)

    old = [normalized(name, v7[name]) for name in
           ("freeze", "acceleration_persistent", "periodicity_persistent")]
    turbulence = np.minimum(prefix_peak(old[1]), prefix_peak(old[2]))
    turbulence[~np.isfinite(turbulence)] = np.nan
    guard = np.fmax(old[0], turbulence)
    standard = v8_streams(raw, valid)
    normal = [normalized(name, standard[:, :, i]) for i, name in enumerate(("frontback", "curvature"))]
    q = np.arange(valid.shape[1])[None]
    relaxed = [normal[i] + .0015 * q / profile["heads"][name]["scale"]
               for i, name in enumerate(("frontback", "curvature"))]
    confirmed = [persistent(s, 2) for s in relaxed]
    fixed = [persistent(s, 2) for s in normal]

    def combine(*values):
        out = values[0].copy()
        for other in values[1:]:
            out = np.fmax(out, other)
        return out

    short = v8_streams(raw, valid, width=1)
    short_heads = []
    for i, name in enumerate(("frontback", "curvature")):
        s = normalized("unsmoothed_" + name, short[:, :, i])
        s += .0015 * q / profile["heads"]["unsmoothed_" + name]["scale"]
        short_heads.append(persistent(s, 2))
    absolute = v8_streams(raw, valid, relative=False)[:, :, 1]
    absolute = normalized("absolute_curvature", absolute)
    absolute += .0015 * q / profile["heads"]["absolute_curvature"]["scale"]
    outputs = {
        "v7": guard,
        "v7_frontback": combine(guard, confirmed[0]),
        "v7_curvature": combine(guard, confirmed[1]),
        "v82": combine(guard, *confirmed),
        "v8_fixed": combine(guard, *fixed),
        "v82_no_smoothing": combine(guard, *short_heads),
        "v82_no_confirmation": combine(guard, *relaxed),
        "v82_no_curvature_baseline": combine(guard, confirmed[0], persistent(absolute, 2)),
        "frontback": confirmed[0],
        "curvature": confirmed[1],
        "eef_motion_low": normalized("eef_motion_low", eef),
        "clock": np.broadcast_to(q, valid.shape).astype(float),
    }
    scores = np.stack([outputs[name] for name in METHODS]).astype(np.float32)
    scores[:, ~valid] = np.nan
    return scores, profile


def conformal_threshold(peaks, alpha):
    peaks = np.asarray(peaks, np.float64)
    if np.isnan(peaks).any() or not 0 < alpha < 1 or not len(peaks):
        raise ValueError("invalid calibration sample")
    rank = int(np.ceil((len(peaks) + 1) * (1 - alpha)))
    return (float(np.sort(peaks)[rank - 1]) if rank <= len(peaks) else float("inf")), rank


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    good = np.isfinite(scores)
    labels, scores = labels[good], scores[good]
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def cumulative_counts(first, failure, length, q):
    fired = (first >= 0) & (first < length) & (first <= q)
    return dict(tp=int((fired & failure).sum()), fp=int((fired & ~failure).sum()),
                failures=int(failure.sum()), successes=int((~failure).sum()),
                active_failure=int((failure & (length > q)).sum()),
                active_success=int((~failure & (length > q)).sum()))
