"""Detector families, all sharing one per-chunk scalar score `z`.

Every family reads `z_q` and the chunk index only.  None of them may see the
cap, the phase or the task.  Thresholds always fire with `>=`.
"""

from __future__ import annotations

import numpy as np

import capfree_common as C

N_CHUNKS = C.N_CHUNKS


def cusum_path(z: np.ndarray, valid: np.ndarray, k: float) -> np.ndarray:
    """S_q = max(0, S_{q-1} + (z_q - k)); frozen once the episode ends."""
    n, Q = z.shape
    S = np.zeros(n, np.float32)
    out = np.empty((n, Q), np.float32)
    for q in range(Q):
        S = np.where(valid[:, q], np.maximum(0.0, S + (z[:, q] - k)), S)
        out[:, q] = S
    return out


def kofm_path(z: np.ndarray, valid: np.ndarray, thr: float, M: int) -> np.ndarray:
    """Count of the last M chunks (valid only) whose z_q >= thr."""
    ind = ((z >= thr) & valid).astype(np.int16)
    cs = np.cumsum(ind, axis=1)
    shifted = np.concatenate([np.zeros((ind.shape[0], M), np.int32),
                              cs[:, :-M].astype(np.int32)], axis=1)
    return (cs - shifted[:, :N_CHUNKS]).astype(np.int16)


def first_crossing(stat: np.ndarray, valid: np.ndarray, thr: float,
                   q_min: int = 0) -> np.ndarray:
    """First chunk with stat >= thr, on a valid chunk, at or after q_min."""
    ok = (stat >= thr) & valid
    if q_min:
        ok[:, :q_min] = False
    hit = ok.any(axis=1)
    return np.where(hit, ok.argmax(axis=1), -1)


def cummax_valid(stat: np.ndarray, valid: np.ndarray, q_min: int = 0) -> np.ndarray:
    """Running max over valid chunks; `-inf` before q_min and on invalid cells.

    Because the result is non-decreasing, the first crossing of any threshold
    is `(cummax < thr).sum(axis=1)`, which lets a whole threshold grid be swept
    without recomputing the statistic.
    """
    m = np.where(valid, stat, -np.inf).astype(np.float32)
    if q_min:
        m[:, :q_min] = -np.inf
    return np.maximum.accumulate(m, axis=1)


def first_from_cummax(cm: np.ndarray, thr: float) -> np.ndarray:
    idx = (cm < thr).sum(axis=1)
    return np.where(idx < cm.shape[1], idx, -1)


def alarm_count_grid(cm: np.ndarray, counts) -> np.ndarray:
    """Thresholds that produce approximately the requested alarm counts."""
    mx = cm[:, -1]
    fin = mx[np.isfinite(mx)]
    n = len(mx)
    qs = np.clip(1.0 - np.asarray(counts, float) / max(1, n), 0.0, 1.0)
    if fin.size < 16:
        return np.array([np.inf])
    return np.unique(np.quantile(fin, qs))


def sweep(cm: np.ndarray, thrs, risk, length, base_at, lead: int,
          fp_lo: float = 0, fp_hi: float = np.inf, extra: dict | None = None):
    """Score a threshold grid on one running-max statistic."""
    rows = []
    for thr in thrs:
        first = first_from_cummax(cm, thr)
        if (first >= 0).sum() == 0:
            continue
        s = C.score(first, risk, length)
        fp = s[f"fp_lead{lead}"]
        if not (fp_lo <= fp <= fp_hi):
            continue
        s["thr"] = float(thr)
        s["excess_tp"] = int(s[f"tp_lead{lead}"] - base_at(fp))
        if extra:
            s.update(extra)
        rows.append(s)
    return rows
