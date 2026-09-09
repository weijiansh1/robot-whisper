"""Score variants + one-call runner for a (corpus, score, config, alpha) cell."""
import numpy as np
import pandas as pd

import evalm
import lib
import seq

DATA = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
OUT = "/home/jovyan/work/himoe-vla/analysis_cusum"


def load(tag):
    meta = pd.read_csv(f"{DATA}/{tag}_meta.csv")
    meta["success"] = meta["success"].astype(bool)
    H = np.load(f"{OUT}/H_{tag}.npy")
    cls = lib.load_classes(tag, meta)
    on = lib.load_onsets(tag, meta)
    on = np.where(np.isfinite(on) & (on >= 0), on, np.nan)
    return meta, H, cls, on


def score_real(H, meta):
    B, Tm, _ = H.shape
    S = np.full((B, Tm), np.nan)
    for b, T in enumerate(meta["T"].values):
        S[b, 1:T] = np.diagonal(H[b, :T, :T], 1)
    return S


def score_shuffle(H, meta, seed):
    """SENTINEL 1: permute chunk order inside each branch, then take the same
    consecutive-pair distance along the permuted order."""
    rng = np.random.default_rng(seed)
    B, Tm, _ = H.shape
    S = np.full((B, Tm), np.nan)
    for b, T in enumerate(meta["T"].values):
        p = rng.permutation(T)
        S[b, 1:T] = H[b, p[:-1], p[1:]]
    return S


def score_clock(meta, Tm):
    """SENTINEL 2: a score that is a pure function of t and nothing else."""
    B = len(meta)
    S = np.full((B, Tm), np.nan)
    for b, T in enumerate(meta["T"].values):
        S[b, 1:T] = np.arange(1, T, dtype=float)
    return S


def clock_statistic(meta, Tm):
    """The strongest pure-clock detector: Lambda_t = t.  Under a bound it
    alarms at a fixed index for every branch still alive, so its detection
    rate is exactly the survival contrast."""
    B = len(meta)
    L = np.full((B, Tm), np.nan)
    for b, T in enumerate(meta["T"].values):
        top = min(seq.TMAX, T)
        L[b, seq.ALO:top] = np.arange(seq.ALO, top, dtype=float)
    return L


DEFAULT = dict(W=1, tstd=True, whiten="ar1", dens="gauss", rule="cusum")


def statistic(S, meta, cfg):
    llr, use, aux = seq.build_llr(S, meta, cfg)
    L = seq.accumulate(llr, use, cfg["rule"])
    return L, llr, use, aux


def evaluate(L, meta, cls, on, alpha, tag_extra=None):
    alarm, bounds = seq.logo_alarms(L, meta, alpha)
    rows = []
    base = dict(tag_extra or {})
    base["alpha_nominal"] = alpha
    base["bound_mean"] = float(np.mean([v for v in bounds.values() if np.isfinite(v)]))
    m = evalm.metrics(alarm, meta, onset=on)
    rows.append({**base, "pop": "all_fail", **m})
    for c in cls.columns:
        sel = cls[c].values
        if sel.sum() == 0:
            continue
        m = evalm.metrics(alarm, meta, onset=on, sel=sel)
        rows.append({**base, "pop": c, **m})
    return alarm, rows, bounds


def peak_auc(L, meta):
    """Within-group paired AUC of the peak causal statistic (branch level)."""
    Tb = meta["T"].values
    pk = seq.running_peak(L, Tb)
    pk = np.where(np.isfinite(pk), pk, np.nanmin(pk[np.isfinite(pk)]) - 1)
    a, n = lib.within_group_auc(pk, ~meta["success"].values, meta["group"].values)
    return a, n
