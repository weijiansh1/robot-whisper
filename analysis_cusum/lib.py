"""Sequential (CUSUM / SPRT) detector on the MoE routing signal.

Per-step score = Hellinger between consecutive control steps, computed per
(layer, token) cell on the 32-expert probability vector then averaged over the
40 cells (back-block HB layers 12-15, action tokens 1-10, denoise step 9).
This is the established V2 "divergence-first" placement, identical to
analysis_agg_axis/lib.py `hell|prev`.

No hard top-4 expert IDs anywhere.
"""
import os

import numpy as np
import pandas as pd

DATA = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
RUN = ("/home/jovyan/work/himoe-vla/himoe-route-capture/runs/"
       "rolling-star-a100-long-t08-k16-20260828")
HUB = ("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
       "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
OUT = "/home/jovyan/work/himoe-vla/analysis_cusum"
TMP = "/tmp"

EPS = 1e-12


# --------------------------------------------------------------- score series
def build_scores(tag):
    """Return dict with S (B, Tmax) raw per-step Hellinger (NaN where undefined),
    valid (B, Tmax) bool, and meta DataFrame.

    S[b, 0] is NaN by construction (no previous chunk).  S[b, t] for
    1 <= t < T[b] is the Hellinger distance between chunk t and chunk t-1.
    """
    X = np.load(f"{DATA}/{tag}_main.npy", mmap_mode="r")     # (N, 4, 10, 32)
    eid = np.load(f"{DATA}/{tag}_episode_id.npy")
    meta = pd.read_csv(f"{DATA}/{tag}_meta.csv")
    meta["success"] = meta["success"].astype(bool)

    # ascending global row index == temporal order (control_step is a global arange)
    srt = np.argsort(eid, kind="stable")
    e_srt = eid[srt]
    uq = np.unique(e_srt)
    st = np.searchsorted(e_srt, uq, side="left")
    en = np.append(st[1:], len(e_srt))
    order = {int(u): np.sort(srt[s:e]) for u, s, e in zip(uq, st, en)}

    B = len(meta)
    Tmax = int(meta["T"].max())
    S = np.full((B, Tmax), np.nan, np.float64)
    valid = np.zeros((B, Tmax), bool)
    for i, (epid, T) in enumerate(zip(meta["episode_id"].values, meta["T"].values)):
        rows = order[int(epid)]
        assert len(rows) == T, (tag, epid, len(rows), T)
        P = np.asarray(X[rows], np.float32).reshape(T, 40, 32)
        P = P / np.maximum(P.sum(-1, keepdims=True), EPS)
        bc = np.sqrt(P[1:] * P[:-1]).sum(-1)                 # (T-1, 40)
        h = np.sqrt(np.clip(1.0 - bc, 0.0, None)).mean(-1)   # (T-1,)
        S[i, 1:T] = h
        valid[i, :T] = True
    return dict(S=S, valid=valid, meta=meta, tag=tag)


def window_mean(S, W):
    """Trailing window mean of width W.  M[b, t] = mean(S[b, t-W+1 .. t]).
    NaN wherever any element in the window is NaN / out of range.
    W == 1 returns S unchanged."""
    if W == 1:
        return S.copy()
    B, T = S.shape
    M = np.full_like(S, np.nan)
    C = np.nan_to_num(S, nan=0.0)
    ok = np.isfinite(S).astype(np.float64)
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(C, 1)], 1)
    co = np.concatenate([np.zeros((B, 1)), np.cumsum(ok, 1)], 1)
    for t in range(W - 1, T):
        n = co[:, t + 1] - co[:, t + 1 - W]
        s = cs[:, t + 1] - cs[:, t + 1 - W]
        full = n == W
        M[full, t] = s[full] / W
    return M


# --------------------------------------------------------------- failure class
def load_classes(tag, meta):
    """Return DataFrame indexed like meta with boolean columns
    cls_stagnation, cls_loop, cls_other (mutually exclusive, failures only)."""
    n = len(meta)
    out = pd.DataFrame(index=meta.index)
    if tag == "A":
        lab = pd.read_csv(f"{RUN}/analysis/candidate_physical_labels.csv")
        lab = lab.set_index("episode_id")
        st = lab["label_stagnation"].reindex(meta["episode_id"].values).values
        lp = lab["label_loop_or_cycling"].reindex(meta["episode_id"].values).values
        ot = lab["label_non_stagnation_non_loop"].reindex(
            meta["episode_id"].values).values
        st = np.asarray(st, bool)
        lp = np.asarray(lp, bool)
        ot = np.asarray(ot, bool)
    else:
        # corpus B has no physical label file; classes are A-only
        st = lp = ot = np.zeros(n, bool)
    fail = ~meta["success"].values
    out["cls_stagnation"] = st & fail
    out["cls_loop"] = lp & (~st) & fail
    out["cls_other"] = ot & fail
    return out


def load_onsets(tag, meta):
    """Frozen physical loop onset query index per branch, NaN if none."""
    o = np.full(len(meta), np.nan)
    if tag == "A":
        t = pd.read_csv(f"{RUN}/analysis_trap_onset/trap_onset.csv").set_index("episode_id")
        v = t["loop_onset_query"].reindex(meta["episode_id"].values).values
        o = np.asarray(v, np.float64)
        # The frozen table uses -1, not NaN, for branches without an onset.
        # Leaving the sentinel finite contaminates alarm-to-onset delays.
        o[o < 0] = np.nan
    return o


# --------------------------------------------------------------- paired AUC
def within_group_auc(score, y, group):
    """Within-group paired AUC (pos = y True), pooled over groups by pair count."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    group = np.asarray(group)
    conc = 0.0
    npair = 0
    for g in np.unique(group):
        m = (group == g) & np.isfinite(score)
        a = score[m & y]
        b = score[m & ~y]
        if len(a) == 0 or len(b) == 0:
            continue
        d = a[:, None] - b[None, :]
        conc += (d > 0).sum() + 0.5 * (d == 0).sum()
        npair += a.size * b.size
    return (conc / npair if npair else np.nan), npair


# --------------------------------------------------------------- persistence
_CSV = None


def csv_paths(name):
    return [f"{TMP}/{name}", f"{OUT}/{name}"]


def append_rows(rows, name="moe_cusum.csv"):
    """Append rows (list of dict) to both /tmp and analysis_cusum copies."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    for p in csv_paths(name):
        hdr = not os.path.exists(p)
        df.to_csv(p, mode="a", header=hdr, index=False)


def append_md(text, name="moe_cusum.md"):
    for p in csv_paths(name):
        with open(p, "a") as f:
            f.write(text.rstrip("\n") + "\n")
