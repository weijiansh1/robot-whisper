#!/usr/bin/env python3
"""Decision-line variants that remove the corpus-mix dependence of the v7 cuts.

LOSO showed the freeze and acceleration thresholds swing ~20% std when the
calibration corpus horizon mix changes; LOTO showed they barely move when the
mix is preserved. The defect is therefore in how the cut is calibrated, not in
the score definitions. These variants change only the cut.

All three are prefix-causal: the decision at query q uses only queries 0..q.

  pooled_peak       v7 baseline. One scalar per stream, the q-th quantile of
                    per-trajectory peaks pooled over the whole corpus. Length
                    biased: long rollouts have more chances to set a high peak,
                    so the scalar is dominated by the corpus horizon mix.

  query_indexed     One threshold per query index. At query q the rollout's own
                    prefix max is compared against the corpus distribution of
                    prefix maxima *among trajectories that also reached q*.
                    Still corpus derived, but each q is calibrated against
                    comparable rollouts, so the mix no longer sets the line.
                    Causal: at query q the monitor only needs to know q, not the
                    rollout's total length.

  self_normalized   No corpus at all. The line is the rollout's own prefix
                    median plus c robust deviations, so it adapts to both the
                    rollout's scale and its length. c is a design constant, not
                    a value read off a corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

from intrinsic_guard_monitor import quantile_higher, row_max  # noqa: E402


VARIANTS = (
    "pooled_peak",
    "query_indexed",
    "self_normalized",
    "query_normalized_peak",
)
MIN_REFERENCE = 32
SELF_NORM_MIN_QUERY = 8
MAD_TO_SIGMA = 1.4826


def _finite_or(values: np.ndarray, fill: float) -> np.ndarray:
    return np.where(np.isfinite(values), values, fill)


def prefix_max(values: np.ndarray) -> np.ndarray:
    """Running max over the prefix, ignoring NaN. NaN until the first finite."""
    filled = _finite_or(values, -np.inf)
    running = np.maximum.accumulate(filled, axis=1)
    return np.where(np.isfinite(running), running, np.nan)


def pooled_peak_threshold(reference: np.ndarray, quantile: float) -> float:
    """v7: one scalar from the pooled per-trajectory peak distribution."""
    return quantile_higher(row_max(reference), quantile)


def query_indexed_threshold(reference: np.ndarray, quantile: float) -> np.ndarray:
    """One threshold per query index, from prefix maxima of surviving rollouts.

    Trajectories that never reached query q contribute nothing to theta[q], so
    theta[q] is calibrated only against rollouts that are still running at q.
    Queries with too few surviving references get +inf and can never alarm.
    """
    peaks = prefix_max(reference)
    thresholds = np.full(peaks.shape[1], np.inf, dtype=np.float64)
    for query in range(peaks.shape[1]):
        column = peaks[:, query]
        column = column[np.isfinite(column)]
        if len(column) >= MIN_REFERENCE:
            thresholds[query] = quantile_higher(column, quantile)
    return thresholds


def prefix_location_scale(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal prefix median and robust deviation, per rollout and query.

    Split out from the threshold so a whole grid of c values reuses one pass:
    the line is always centre + c * spread.

    Because the line is affine in the score's own location and scale, any
    constant multiplying the score cancels out of the comparison. That is what
    makes this variant corpus free even though z_R is defined with a corpus
    derived periodicity_scale in its denominator.
    """
    episodes, queries = scores.shape
    centre = np.full((episodes, queries), np.nan, dtype=np.float64)
    spread = np.full((episodes, queries), np.nan, dtype=np.float64)
    for query in range(SELF_NORM_MIN_QUERY, queries):
        window = scores[:, : query + 1]
        good = np.isfinite(window).sum(axis=1) >= SELF_NORM_MIN_QUERY
        if not good.any():
            continue
        block = window[good]
        location = np.nanmedian(block, axis=1)
        deviation = np.nanmedian(np.abs(block - location[:, None]), axis=1)
        centre[good, query] = location
        spread[good, query] = deviation * MAD_TO_SIGMA
    return centre, spread


def self_normalized_threshold(
    centre: np.ndarray, spread: np.ndarray, deviations: float
) -> np.ndarray:
    """Per-rollout, per-query line: prefix median plus c robust deviations."""
    line = centre + deviations * spread
    return np.where(np.isfinite(line), line, np.inf)


def query_reference_location_scale(
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query median and IQR of the corpus, among rollouts that reached q.

    This is the *shape* of the decision line: how the typical score level moves
    as a rollout gets longer. It is estimated per query, so the corpus horizon
    mix cannot tilt it the way a single pooled quantile is tilted.
    """
    queries = reference.shape[1]
    location = np.full(queries, np.nan, dtype=np.float64)
    scale = np.full(queries, np.nan, dtype=np.float64)
    for query in range(queries):
        column = reference[:, query]
        column = column[np.isfinite(column)]
        if len(column) < MIN_REFERENCE:
            continue
        low, mid, high = np.quantile(column, (0.25, 0.5, 0.75))
        spread = float(high - low)
        if spread <= 0.0:
            continue
        location[query] = float(mid)
        scale[query] = spread
    return location, scale


def query_normalized(
    scores: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Express each score in per-query corpus units.

    Queries with too few surviving references stay NaN and can never alarm;
    that shrinks coverage for very long rollouts rather than extrapolating a
    line nobody calibrated.
    """
    return ((np.asarray(scores, dtype=np.float64) - location) / scale).astype(
        np.float32
    )


def first_alarm(
    scores: np.ndarray, threshold: float | np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """First query whose score strictly exceeds the (possibly varying) line."""
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    line = np.asarray(threshold, dtype=np.float64)
    if line.ndim == 1:
        line = line[None, :]
    trigger = np.isfinite(scores) & (scores > line) & valid
    any_trigger = trigger.any(axis=1)
    first = np.full(len(scores), -1, dtype=np.int16)
    first[any_trigger] = trigger[any_trigger].argmax(axis=1).astype(np.int16)
    return first
