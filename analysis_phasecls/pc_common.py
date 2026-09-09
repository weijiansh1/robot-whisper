"""Shared helpers for the trap-type phase-classification validation.

Hard constraints honoured here:
  * probability vectors only -- never hard top-4 expert ids (bf16 tie jitter)
  * `control_step` in the corpus-A store is a GLOBAL row counter; a branch's
    query index is the RANK of its rows after sorting by control_step.
"""
from __future__ import annotations

import numpy as np

LAYER_ORDER = [2, 3, 4, 5, 12, 13, 14, 15]
FRONT_BLOCK = [0, 1, 2, 3]   # layers 2,3,4,5
BACK_BLOCK = [4, 5, 6, 7]    # layers 12,13,14,15
DENOISE_KEEP = [0, 9]
STATE_TOKENS = [0]
ACTION_TOKENS = list(range(1, 11))


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hellinger distance in [0,1] along the last axis. Soft probs only."""
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    d = sp - sq
    return np.sqrt(np.clip(0.5 * np.einsum("...i,...i->...", d, d), 0.0, 1.0))


def branch_hellinger_series(probs: np.ndarray) -> np.ndarray:
    """probs: (n_q, 8, 10, 11, 32) float -> (n_q-1, 8, len(DENOISE_KEEP), 11)."""
    p = np.asarray(probs, dtype=np.float32)[:, :, DENOISE_KEEP, :, :]
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(s, 1e-8, None)
    return hellinger(p[:-1], p[1:]).astype(np.float32)


def phase_bins(series: np.ndarray, n_bins: int) -> np.ndarray:
    """Resample a 1-D per-transition series onto n_bins of the branch's OWN
    relative phase.  Episode length never enters the feature value."""
    n = len(series)
    if n == 0:
        return np.full(n_bins, np.nan, dtype=np.float64)
    # transition i covers relative phase [i/n, (i+1)/n)
    lo = np.arange(n) / n
    hi = np.arange(1, n + 1) / n
    out = np.empty(n_bins, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b in range(n_bins):
        a, z = edges[b], edges[b + 1]
        w = np.clip(np.minimum(hi, z) - np.maximum(lo, a), 0.0, None)
        tot = w.sum()
        out[b] = float(w @ series / tot) if tot > 1e-12 else np.nan
    if np.isnan(out).any():                      # branches shorter than n_bins
        idx = np.clip(((np.arange(n_bins) + 0.5) / n_bins * n).astype(int), 0, n - 1)
        out = np.where(np.isnan(out), series[idx], out)
    return out


def aggregate(hell: np.ndarray, layer_block: str, denoise: int, tokens: list[int]) -> np.ndarray:
    """(T,8,2,11) -> (T,) mean over the chosen layers / denoise step / tokens."""
    lay = FRONT_BLOCK if layer_block == "front" else BACK_BLOCK
    di = DENOISE_KEEP.index(denoise)
    return hell[:, lay, di, :][:, :, tokens].mean(axis=(1, 2)).astype(np.float64)
