#!/usr/bin/env python3
"""Threshold families, the development frontier, and the sealed transfer of it.

The *whole* recall/FPR curve is defined on development.  For every alarm rule the
parameter grid is swept on development, the non-dominated points are kept, and
each surviving parameter set is frozen.  External and legacy are then scored once
at every frozen point.  No outcome outside development ever selects anything, so
the sealed curves are a transfer of the development frontier, not a search on the
sealed cohorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as P
import detector as D


def _quantiles(sample: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    good = sample[np.isfinite(sample)]
    if good.size == 0:
        return np.array([np.inf])
    return np.quantile(good, 1.0 - alphas, method="lower")


def score_thresholds(S: np.ndarray, gmask: np.ndarray) -> np.ndarray:
    """Development quantile thresholds over the gated-chunk score population.

    `method="lower"` returns a value that is actually present in the sample, so
    the `>=` test always fires on at least the whole tie group at that value.
    """
    return _quantiles(S[gmask], D.ALPHA_GRID)


def cusum_peaks(S: np.ndarray, mask: np.ndarray, drift: float) -> np.ndarray:
    n, n_chunk = S.shape
    c = np.zeros(n)
    peak = np.full(n, -np.inf)
    for q in range(n_chunk):
        s = np.where(mask[:, q], np.nan_to_num(S[:, q], nan=0.0) - drift, 0.0)
        c = np.maximum(0.0, c + s)
        peak = np.maximum(peak, np.where(mask[:, q], c, -np.inf))
    return peak


def rule_grid(
    S: dict[str, np.ndarray],
    frames: dict[str, dict],
    gated: dict[str, list[int]],
    best_chunk: dict[str, int],
    drifts: tuple[float, ...],
) -> list[dict]:
    """Every (rule, parameter) point, evaluated on development first."""
    dev = frames[P.FIT_COHORT]
    gmask = {c: D.gate_mask(frames[c], gated) for c in frames}
    window = {
        c: np.arange(frames[c]["alive"].shape[1])[None, :] <= frames[c]["window_end"][:, None]
        for c in frames
    }
    wmask = {c: gmask[c] & window[c] for c in frames}
    amask = {
        c: frames[c]["alive"] & window[c]
        & (np.arange(frames[c]["alive"].shape[1])[None, :] >= P.FIRST_SCORED_CHUNK)
        for c in frames
    }

    taus = score_thresholds(S[P.FIT_COHORT], wmask[P.FIT_COHORT])
    m = {s: len(v) for s, v in gated.items()}
    max_k = max(m.values())

    points: list[dict] = []
    for a, tau in zip(D.ALPHA_GRID, taus):
        points.append({"rule": "single_chunk", "alpha": float(a), "tau": float(tau),
                       "params": {"chunk_of_suite": best_chunk}})
        for k in range(1, max_k + 1):
            points.append({"rule": "k_of_m", "alpha": float(a), "tau": float(tau),
                           "params": {"k": k}})
    for drift in drifts:
        for label, mask in (("cusum_gated", wmask), ("cusum_all", amask)):
            peaks = cusum_peaks(S[P.FIT_COHORT], mask[P.FIT_COHORT], drift)
            hs = _quantiles(peaks, D.ALPHA_GRID)
            for a, h in zip(D.ALPHA_GRID, hs):
                points.append({"rule": label, "alpha": float(a), "tau": float(h),
                               "params": {"drift": float(drift)}})
    return points, wmask, amask


def apply_point(point: dict, S: np.ndarray, frame: dict, wmask, amask) -> np.ndarray:
    rule, tau, params = point["rule"], point["tau"], point["params"]
    if rule == "single_chunk":
        return D.rule_single(S, frame, None, tau, params["chunk_of_suite"])
    if rule == "k_of_m":
        return D.rule_k_of_m(S, frame, wmask, tau, params["k"])
    if rule == "cusum_gated":
        return D.rule_cusum(S, frame, wmask, tau, params["drift"])
    if rule == "cusum_all":
        return D.rule_cusum(S, frame, amask, tau, params["drift"])
    raise ValueError(rule)


def run(
    S: dict[str, np.ndarray],
    frames: dict[str, dict],
    gated: dict[str, list[int]],
    best_chunk: dict[str, int],
    detector_name: str,
    arm: str,
    drifts: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    points, wmask, amask = rule_grid(S, frames, gated, best_chunk, drifts)
    rows = []
    for i, point in enumerate(points):
        rec = {
            "detector": detector_name,
            "arm": arm,
            "rule": point["rule"],
            "alpha": point["alpha"],
            "tau": point["tau"],
            "k": point["params"].get("k", np.nan),
            "drift": point["params"].get("drift", np.nan),
            "point_id": i,
        }
        for cohort, frame in frames.items():
            if cohort not in S:
                continue
            first = apply_point(point, S[cohort], frame, wmask[cohort], amask[cohort])
            got = P.evaluate(first, frame)
            for key, value in got.items():
                rec[f"{cohort}.{key}"] = value
        rows.append(rec)
    return pd.DataFrame(rows)


def frontier(table: pd.DataFrame, cohort: str = P.FIT_COHORT) -> pd.DataFrame:
    """Development-selected non-dominated points, per rule and pooled."""
    x, y = f"{cohort}.iw_fpr", f"{cohort}.iw_recall"
    parts = []
    for rule, sub in table.groupby("rule"):
        front = P.pareto_front(sub, x, y)
        front["scope"] = rule
        parts.append(front)
    pooled = P.pareto_front(table, x, y)
    pooled["scope"] = "any_rule"
    parts.append(pooled)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# reference arms that are not detectors


def survival_curve(frames: dict[str, dict]) -> pd.DataFrame:
    """BASELINE (not a detector): alarm because the episode is still running.

    Two one-parameter families, both using only "is the episode still running"
    and the suite cap:

      ``survivor_phase``  fire at the first chunk whose phase >= theta;
      ``survivor_lead``   fire at chunk ``window_end - d``.

    ``survivor_lead`` at d = 0 is the strongest point either family contains and
    it recalls **every** risk episode by construction: risk means "did not finish
    before the cap", so a risk episode is alive at every chunk below the cap and
    in particular at the window end.  Its task-matched lift is 1.0 by definition,
    so it is the level a detector has to clear, not a detector.
    """
    rows = []
    for theta in np.round(np.arange(0.02, 0.661, 0.005), 4):
        rec = {"detector": "survival_prior", "arm": "baseline",
               "rule": "survivor_phase", "alpha": np.nan, "tau": float(theta),
               "k": np.nan, "drift": np.nan}
        for cohort, frame in frames.items():
            n_chunk = frame["alive"].shape[1]
            q = np.arange(n_chunk)[None, :]
            phase = (q + 1) / frame["cap"][:, None]
            fire = frame["alive"] & (phase >= theta)
            for key, value in P.evaluate(P.first_alarm(fire), frame).items():
                rec[f"{cohort}.{key}"] = value
        rows.append(rec)
    for lead in range(0, 30):
        rec = {"detector": "survival_prior", "arm": "baseline",
               "rule": "survivor_lead", "alpha": np.nan, "tau": float(lead),
               "k": np.nan, "drift": np.nan}
        any_valid = False
        for cohort, frame in frames.items():
            chunk = frame["window_end"] - lead
            ok = chunk >= P.FIRST_SCORED_CHUNK
            fire = np.zeros(frame["alive"].shape, bool)
            rows_ix = np.flatnonzero(ok)
            fire[rows_ix, chunk[ok]] = frame["alive"][rows_ix, chunk[ok]]
            any_valid |= bool(ok.any())
            for key, value in P.evaluate(P.first_alarm(fire), frame).items():
                rec[f"{cohort}.{key}"] = value
        if any_valid:
            rows.append(rec)
    return pd.DataFrame(rows)


def length_leak_row(frames: dict[str, dict]) -> pd.DataFrame:
    """NEGATIVE CONTROL, is_baseline = False.  Reads the final length (an outcome)."""
    rec = {"detector": "length_leak_NOT_a_baseline", "arm": "negative_control",
           "rule": "length_at_cap", "alpha": np.nan, "tau": np.nan,
           "k": np.nan, "drift": np.nan}
    for cohort, frame in frames.items():
        got = P.evaluate(D.rule_length_leak(frame, 0), frame)
        for key, value in got.items():
            rec[f"{cohort}.{key}"] = value
    return pd.DataFrame([rec])
