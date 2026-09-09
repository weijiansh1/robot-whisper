"""Divergence-function x temporal-aggregation sweep for the HiMoE-VLA MoE routing tensor.

FIXED (owned by the sibling agent, held constant here):
  * back-block HB layers 12-15   (A_main.npy / B_main.npy axis 1, 4 layers)
  * action tokens 1-10           (axis 2, 10 tokens)
  * denoise step 9               (already sliced out)
  * placement = DIVERGENCE-FIRST (V2): the divergence is evaluated per (layer, token)
    cell on the 32-dim expert probability vector, then averaged over the 40 cells.

VARIED (owned here):
  * stage 1: the divergence functional on the expert axis + what it is measured against
  * stage 2: the temporal aggregation over chunks, and the window width

Signal source is exclusively `hb_router_probs`. No hard top-4 expert IDs anywhere.
"""
import numpy as np

EPS = 1e-12
DATA = "/home/jovyan/work/himoe-vla/analysis_stat_axis"

# ---------------------------------------------------------------- data assembly
def build_dense(tag):
    """Return P (B, Tmax, 40, 32) float32, valid (B, Tmax) bool, meta DataFrame."""
    import pandas as pd
    X = np.load(f"{DATA}/{tag}_main.npy")                 # (N, 4, 10, 32)
    eid = np.load(f"{DATA}/{tag}_episode_id.npy")
    meta = pd.read_csv(f"{DATA}/{tag}_meta.csv")
    meta["success"] = meta["success"].astype(bool)
    C = X.shape[1] * X.shape[2]
    Xf = X.reshape(X.shape[0], C, 32)
    B, Tmax = len(meta), int(meta["T"].max())
    P = np.zeros((B, Tmax, C, 32), np.float32)
    valid = np.zeros((B, Tmax), bool)
    # ascending global row index == temporal order (control_step is a global arange)
    srt = np.argsort(eid, kind="stable")
    e_srt = eid[srt]
    uq = np.unique(e_srt)
    st = np.searchsorted(e_srt, uq, side="left")
    en = np.append(st[1:], len(e_srt))
    order = {int(u): np.sort(srt[s:e]) for u, s, e in zip(uq, st, en)}
    for i, (epid, T) in enumerate(zip(meta["episode_id"].values, meta["T"].values)):
        rows = order[int(epid)]
        assert len(rows) == T, (tag, epid, len(rows), T)
        P[i, :T] = Xf[rows]
        valid[i, :T] = True
    return P, valid, meta


# ---------------------------------------------------------------- stage 1: divergences
# Every f(A, B) takes two (n, C, 32) probability arrays and returns (n, C).
def _hell(a, b):
    return np.sqrt(np.clip(1.0 - np.sqrt(a * b).sum(-1), 0.0, None))

def _bc(a, b):                       # Bhattacharyya COEFFICIENT (a similarity)
    return np.sqrt(a * b).sum(-1)

def _bhatdist(a, b):                 # Bhattacharyya DISTANCE = -log BC
    return -np.log(np.clip(np.sqrt(a * b).sum(-1), EPS, None))

def _symkl(a, b):
    la, lb = np.log(np.clip(a, EPS, None)), np.log(np.clip(b, EPS, None))
    return ((a - b) * (la - lb)).sum(-1)

def _js(a, b):
    m = 0.5 * (a + b)
    lm = np.log(np.clip(m, EPS, None))
    return 0.5 * (a * (np.log(np.clip(a, EPS, None)) - lm)).sum(-1) + \
           0.5 * (b * (np.log(np.clip(b, EPS, None)) - lm)).sum(-1)

def _tv(a, b):                       # total variation = 0.5 * L1 (L1 is a fixed multiple)
    return 0.5 * np.abs(a - b).sum(-1)

def _l1(a, b):
    return np.abs(a - b).sum(-1)

def _l2(a, b):
    return np.sqrt(((a - b) ** 2).sum(-1))

def _cos(a, b):
    num = (a * b).sum(-1)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return 1.0 - num / np.clip(den, EPS, None)

def _chi2(a, b):                     # symmetrised chi-squared, sum (a-b)^2 / ((a+b)/2)
    return ((a - b) ** 2 / np.clip(0.5 * (a + b), EPS, None)).sum(-1)

def _emd(a, b):                      # 1-D Wasserstein under the fixed expert ordering 0..31
    return np.abs(np.cumsum(a, -1) - np.cumsum(b, -1)).sum(-1)

PAIRWISE = {
    "hell": _hell, "bc": _bc, "bhatdist": _bhatdist, "symkl": _symkl, "js": _js,
    "tv": _tv, "l1": _l1, "l2": _l2, "cos": _cos, "chi2": _chi2, "emd": _emd,
}

# non-pairwise scalar functionals of one chunk; the series is the CHANGE of the scalar
def _entropy(p):
    return -(p * np.log(np.clip(p, EPS, None))).sum(-1)

def _top1(p):
    return p.max(-1)

SCALARS = {"ent": _entropy, "top1": _top1}


def divergence_series(P, valid, name):
    """d[b, t] for t >= 1 (NaN elsewhere / where the branch has ended).

    name syntax:
      <metric>|prev     : metric(chunk t, chunk t-1)
      <metric>|runmean  : metric(chunk t, running mean of chunks 0..t-1)
      d<scalar>_abs     : |s(chunk t) - s(chunk t-1)|
      d<scalar>_sgn     : s(chunk t) - s(chunk t-1)  (signed)
    All per-(layer, token) cell, then averaged over the 40 cells.
    """
    B, T, C, E = P.shape
    d = np.full((B, T), np.nan, np.float32)
    if name.startswith("d") and ("_abs" in name or "_sgn" in name):
        key = name[1:].rsplit("_", 1)[0]
        f = SCALARS[key]
        signed = name.endswith("_sgn")
        for t in range(1, T):
            ok = valid[:, t]
            if not ok.any():
                continue
            s1 = f(P[ok, t]).mean(-1)
            s0 = f(P[ok, t - 1]).mean(-1)
            d[ok, t] = (s1 - s0) if signed else np.abs(s1 - s0)
        return d
    metric, ref = name.split("|")
    f = PAIRWISE[metric]
    if ref == "prev":
        for t in range(1, T):
            ok = valid[:, t]
            if ok.any():
                d[ok, t] = f(P[ok, t], P[ok, t - 1]).mean(-1)
    elif ref == "runmean":
        run = P[:, 0].astype(np.float64).copy()            # running SUM of chunks 0..t-1
        for t in range(1, T):
            ok = valid[:, t]
            if ok.any():
                m = (run[ok] / t)
                m = (m / m.sum(-1, keepdims=True)).astype(np.float32)
                d[ok, t] = f(P[ok, t], m).mean(-1)
            run += P[:, t]
    else:
        raise ValueError(ref)
    return d


DIVERGENCES = ([f"{m}|prev" for m in PAIRWISE] +
               [f"{m}|runmean" for m in PAIRWISE] +
               ["dent_abs", "dent_sgn", "dtop1_abs", "dtop1_sgn"])


# ---------------------------------------------------------------- stage 2: temporal aggregation
WINDOWS = (3, 5, 8, 12, 20)
WIN_TEMPORALS = ["mean", "max", "min", "median", "std", "slope",
                 "ewm0.15", "ewm0.30", "ewm0.50",
                 "fracbelow_q10", "fracbelow_q25", "fracabove_q75", "fracabove_q90"]
ALL_TEMPORALS = ["cum_over_t", "ewmall0.15", "ewmall0.30", "ewmall0.50"]


def aggregate(d, t, temporal, W, thr=None):
    """Aggregate the divergence series d (B, T) over chunks, evaluated at query index t.

    Window temporals use the W consecutive values d[t-W+1 .. t]; W=0 means the whole
    history d[1 .. t]. `thr` is the (global, label-free) quantile threshold pair.
    """
    if temporal in ALL_TEMPORALS:
        seg = d[:, 1:t + 1]
        if temporal == "cum_over_t":
            return seg.sum(1) / t
        a = float(temporal.replace("ewmall", ""))
        L = seg.shape[1]
        w = a * (1 - a) ** np.arange(L - 1, -1, -1, dtype=np.float64)
        return (seg * (w / w.sum())).sum(1)
    lo = t - W + 1
    assert lo >= 1, (t, W)
    seg = d[:, lo:t + 1]
    if temporal == "mean":
        return seg.mean(1)
    if temporal == "max":
        return seg.max(1)
    if temporal == "min":
        return seg.min(1)
    if temporal == "median":
        return np.median(seg, 1)
    if temporal == "std":
        return seg.std(1)
    if temporal == "slope":
        x = np.arange(W, dtype=np.float64) - (W - 1) / 2.0
        return (seg * x).sum(1) / (x * x).sum()
    if temporal.startswith("ewm"):
        a = float(temporal.replace("ewm", ""))
        w = a * (1 - a) ** np.arange(W - 1, -1, -1, dtype=np.float64)
        return (seg * (w / w.sum())).sum(1)
    if temporal.startswith("fracbelow_q"):
        return (seg < thr[{"10": 0, "25": 1}[temporal[-2:]]]).mean(1)
    if temporal.startswith("fracabove_q"):
        return (seg > thr[{"75": 2, "90": 3}[temporal[-2:]]]).mean(1)
    raise ValueError(temporal)


def cells():
    """(temporal, window) combinations. window 0 == whole history."""
    out = [(tm, W) for tm in WIN_TEMPORALS for W in WINDOWS]
    out += [(tm, 0) for tm in ALL_TEMPORALS]
    return out


# ---------------------------------------------------------------- AUC / residual
def _avg_rank(x):
    n = len(x)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    r = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        r[o[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return r


def within_group_auc(scores, y, groups):
    """Pooled-by-pair-count within-group AUC = P(score | success > score | failure)."""
    num, den = 0.0, 0
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(scores)
        s, yy = scores[m], y[m]
        npos, nneg = int(yy.sum()), int((~yy).sum())
        if npos == 0 or nneg == 0:
            continue
        r = _avg_rank(s)
        auc = (r[yy].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
        num += auc * npos * nneg
        den += npos * nneg
    return (num / den, den) if den else (np.nan, 0)


def rank_residualise(scores, base, groups):
    """Within group: OLS of rank(scores) on rank(base); return the residual. Label-free."""
    out = np.full(len(scores), np.nan, np.float64)
    for g in np.unique(groups):
        m = np.where((groups == g) & np.isfinite(scores) & np.isfinite(base))[0]
        if len(m) < 3:
            continue
        rs = _avg_rank(scores[m]); rb = _avg_rank(base[m])
        rs -= rs.mean(); rb -= rb.mean()
        den = (rb * rb).sum()
        beta = (rs * rb).sum() / den if den > 1e-12 else 0.0
        r = rs - beta * rb
        # a cell that is an exact rank-duplicate of the baseline leaves only float noise;
        # ranking that noise would resurrect the baseline, so snap it to zero.
        r[np.abs(r) < 1e-8 * len(m)] = 0.0
        out[m] = r
    return out
