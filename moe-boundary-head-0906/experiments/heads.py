"""Candidate score series and the fast confirmed-first kernel.

Shared by the development sweep and the frozen scoring pass so that both build
the score the same way from the same code.

Naming of a construction is `form|cells`:

    raw|<cell>            the cell as measured
    sb|<cell>             log(cell / that episode's own q1..q4 mean), v7's idiom
    rank|<cell>           the cell's per-chunk percentile against the frozen
                          development reference distribution
    rankdiff|<a>-<s>      rank(action cell) - rank(state cell)
    rawdiff|<a>-<s>       NEGATIVE CONTROL.  Subtracting the raw values is a bug,
                          not a design: front_state is 3.9x front_action, so the
                          difference is just -front_state.  Kept and labelled so
                          the sweep shows what it does.
    logratio|<a>-<s>      log(action / state), scale-free but not chunk-free
    sbdiff|<a>-<s>        difference of the two self-baselined series

Every construction exists in a `bnd_` (chunk boundary: p[q-1,step9] vs
p[q,step0]) and a `wc_` (adjacent-query at step 9: p[q-1,step9] vs p[q,step9])
variant, because the two are different axes for the action tokens - and, as
`decompose_axis.py` shows, the *same* axis for the state token.
"""

from __future__ import annotations

import itertools

import numpy as np

from common import BASE_CELLS, chunk_rank, trailing_mean

BASELINE = slice(1, 5)          # v7's self-baseline window, q1..q4
FAMILIES = ("bnd", "wc")
ACTIONS = ("front_action", "back_action")
STATES = ("front_state", "back_state")
PAIRS = tuple(itertools.product(ACTIONS, STATES))


def self_baseline(series: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        base = np.nanmean(series[:, BASELINE], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.maximum(series, 1e-12) / np.maximum(base, 1e-12))
    out[~np.isfinite(out)] = np.nan
    return out


def _cell(d: dict, family: str, cell: str) -> np.ndarray:
    return d[cell] if family == "bnd" else d[f"wc_{cell}"]


def tag(family: str, act: str, st: str) -> str:
    return f"{family}:{act[0]}a-{st[0]}s"


def constructions(d: dict, ref: dict) -> dict[str, np.ndarray]:
    """All candidate score series.  `ref` supplies the frozen per-chunk rank
    reference (always development), so the transform carries no test-cohort
    information."""
    out: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        raw = {c: _cell(d, family, c) for c in BASE_CELLS}
        ref_raw = {c: _cell(ref, family, c) for c in BASE_CELLS}
        rank = {c: chunk_rank(raw[c], ref_raw[c]) for c in BASE_CELLS}
        sb = {c: self_baseline(raw[c]) for c in BASE_CELLS}
        for c in BASE_CELLS:
            out[f"raw|{family}:{c}"] = raw[c]
            out[f"sb|{family}:{c}"] = sb[c]
            out[f"rank|{family}:{c}"] = rank[c]
        for act, st in PAIRS:
            t = tag(family, act, st)
            out[f"rankdiff|{t}"] = rank[act] - rank[st]
            out[f"rawdiff|{t}"] = raw[act] - raw[st]        # negative control
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(np.maximum(raw[act], 1e-12)
                            / np.maximum(raw[st], 1e-12))
            lr[~np.isfinite(lr)] = np.nan
            out[f"logratio|{t}"] = lr
            out[f"sbdiff|{t}"] = sb[act] - sb[st]
        out[f"rankdiff|{family}:mean"] = (
            0.5 * (rank["front_action"] + rank["back_action"])
            - 0.5 * (rank["front_state"] + rank["back_state"]))
    return out


NEGATIVE_CONTROL_FORMS = ("rawdiff",)


def threshold_of(series: np.ndarray, quantile: float, direction: str) -> float:
    """Order statistic of the UNLABELED pool.  `method="lower"` so the returned
    value is an actually-attained score and the tie group at it can fire."""
    pool = series[np.isfinite(series)]
    level = quantile if direction == "high" else 1.0 - quantile
    return float(np.quantile(pool, level, method="lower"))


def held_runs(smooth: np.ndarray, thr: float, direction: str,
              start: int = 0) -> np.ndarray:
    """Run length of the consecutive crossing ending at each chunk.

    `>=` / `<=`, never strict, so the tie group exactly at the threshold fires.

    `start` is where the detector is allowed to begin looking, and the run
    counter is reset there rather than being allowed to carry in from before.
    This matters and is not a free choice: v8's `confirmed_first` does
    `hit[:, :earliest] = False` *before* counting, so a crossing that began at
    chunk 4 does not already count as confirmed at chunk 6.  Letting the run
    carry in silently weakens the confirmation requirement near the window edge
    and makes the kernel disagree with v8 on real configurations -
    `tests/test_protocol.py::test_fast_kernel_matches_v8_confirmed_first`
    pins the two together.
    """
    hit = (smooth >= thr) if direction == "high" else (smooth <= thr)
    hit &= np.isfinite(smooth)
    if start:
        hit[:, :start] = False
    run = np.zeros(smooth.shape, dtype=np.int16)
    acc = np.zeros(smooth.shape[0], dtype=np.int16)
    for q in range(smooth.shape[1]):
        acc = np.where(hit[:, q], acc + 1, 0)
        run[:, q] = acc
    return run


def window_of(earliest: int, gate) -> tuple[int, int | None]:
    """(start, exclusive upper bound) for an (earliest, gate) pair."""
    if gate is None:
        return earliest, None
    return max(earliest, gate[0]), gate[1]


def first_true_from(held: np.ndarray) -> np.ndarray:
    """`out[i, c]` = smallest q >= c with held[i, q], else n_chunk.

    One backward pass; makes every (earliest, gate) pair an O(1) lookup instead
    of a fresh scan, which is what makes a 10^5-configuration sweep tractable.
    """
    n, m = held.shape
    out = np.full((n, m + 1), m, dtype=np.int16)
    for q in range(m - 1, -1, -1):
        out[:, q] = np.where(held[:, q], q, out[:, q + 1])
    return out


def fire_from(ft: np.ndarray, start: int, hi: int | None, n_chunk: int) -> np.ndarray:
    """First confirmed crossing at or after `start`, before `hi`.

    `ft` must have been built from runs that were themselves reset at `start`.
    """
    bound = n_chunk if hi is None else min(n_chunk, hi)
    first = ft[:, min(start, n_chunk)].astype(np.int64)
    return np.where(first < bound, first, -1)
