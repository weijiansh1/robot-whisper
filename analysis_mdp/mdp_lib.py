"""Markov / MDP model over HiMoE-VLA MoE routing states.

Signal source is exclusively `hb_router_probs`, fixed slice:
  back-block HB layers 12-15, action tokens 1-10, denoise step 9
  -> per chunk, a (40, 32) array of expert-probability vectors.

No hard top-4 expert IDs anywhere; probabilities only.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_agg_axis")
import lib as agglib  # build_dense, divergence_series, aggregate, within_group_auc, rank_residualise

DATA = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
OUT = "/home/jovyan/work/himoe-vla/analysis_mdp"
TMP = "/tmp"
TS = (20, 25, 30)
KS = (4, 8, 16, 32, 64)
NFOLD = 5
SEED = 20260829

_cache = {}


def load(tag):
    """P (B,Tmax,40,32) float32, valid (B,Tmax) bool, meta DataFrame (episode_id, group, success, T)."""
    if tag not in _cache:
        _cache[tag] = agglib.build_dense(tag)
    return _cache[tag]


# ------------------------------------------------------------------ features
def features(P, valid, kind):
    """Per-chunk feature vector, label-free.

    cell1280 : sqrt of every (layer, token) 32-simplex, flattened  (Hellinger embedding)
    mean32   : sqrt of the cell-averaged 32-simplex
    """
    B, T, C, E = P.shape
    if kind == "cell1280":
        F = np.sqrt(P, dtype=np.float32).reshape(B, T, C * E)
    elif kind == "mean32":
        m = P.mean(axis=2)
        F = np.sqrt(m, dtype=np.float32)
    else:
        raise ValueError(kind)
    F[~valid] = 0.0
    return F


def fit_states(F, valid, train_b, K, npc=20, seed=SEED):
    """PCA + k-means fitted on the TRAIN branches' chunks only; assign every chunk.

    Returns S (B, Tmax) int16 with -1 on invalid chunks, plus the fitted objects.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    B, T, D = F.shape
    trm = np.zeros(B, bool)
    trm[train_b] = True
    mtr = valid & trm[:, None]
    Xtr = F[mtr]
    if D > npc:
        pca = PCA(n_components=npc, random_state=seed).fit(Xtr)
        Ztr = pca.transform(Xtr)
    else:
        pca = None
        Ztr = Xtr
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(Ztr)
    S = np.full((B, T), -1, np.int16)
    Xall = F[valid]
    Zall = pca.transform(Xall) if pca is not None else Xall
    S[valid] = km.predict(Zall).astype(np.int16)
    return S, pca, km


def folds(meta, nfold=NFOLD, seed=SEED):
    """Branch-level folds stratified by GROUP only (label-free stratification)."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(meta), int)
    for g in meta["group"].unique():
        idx = np.where(meta["group"].values == g)[0]
        idx = idx[rng.permutation(len(idx))]
        f[idx] = np.arange(len(idx)) % nfold
    return f


# ------------------------------------------------------------------ chain
def chain_counts(S, valid, train_b, K):
    """Transient transition counts (label-free) and per-branch terminal state."""
    C = np.zeros((K, K), np.float64)
    term = np.full(S.shape[0], -1, int)
    for b in train_b:
        s = S[b][valid[b]]
        if len(s) < 2:
            continue
        np.add.at(C, (s[:-1], s[1:]), 1.0)
        term[b] = s[-1]
    return C, term


def all_terminals(S, valid):
    term = np.full(S.shape[0], -1, int)
    for b in range(S.shape[0]):
        s = S[b][valid[b]]
        if len(s):
            term[b] = s[-1]
    return term


def absorption(C, term, train_b, fail, K, alpha=0.5):
    """P(eventual FAILURE | current state s) on the chain with 2 absorbing states.

    C     : (K,K) transient transition counts from the training branches
    term  : (B,) terminal routing state of each branch
    fail  : (B,) bool, 1 - success
    alpha : Laplace smoothing spread over the K+2 targets
    """
    R = np.zeros((K, 2), np.float64)          # columns: [to SUCCESS, to FAILURE]
    tb = np.asarray(train_b)
    tb = tb[term[tb] >= 0]
    np.add.at(R, (term[tb], fail[tb].astype(int)), 1.0)
    Ca = C + alpha
    Ra = R + alpha
    tot = Ca.sum(1) + Ra.sum(1)
    Q = Ca / tot[:, None]
    Rn = Ra / tot[:, None]
    h = np.linalg.solve(np.eye(K) - Q, Rn[:, 1])      # absorption prob into FAILURE
    Nfund = np.linalg.inv(np.eye(K) - Q)
    esteps = Nfund.sum(1)                              # expected remaining chunks
    return np.clip(h, 0.0, 1.0), Q, Rn, esteps


def empirical_rates(S, valid, train_b, fail, K, prior_w=5.0):
    """Pooled per-state empirical failure rate (chunk-weighted), shrunk to the global rate."""
    tb = np.asarray(train_b)
    M = np.zeros((K, len(tb)), np.float64)
    for j, b in enumerate(tb):
        s = S[b][valid[b]]
        if len(s):
            np.add.at(M[:, j], s, 1.0)
    f = fail[tb].astype(np.float64)
    n = M.sum(1)
    num = M @ f
    g = f.mean()
    return (num + prior_w * g) / (n + prior_w)


def empirical_rates_t(S, valid, train_b, fail, K, t, prior_w=5.0):
    """Time-local per-state failure rate: training branches alive at t, state at query t."""
    tb = np.asarray(train_b)
    alive = tb[valid[tb, t]]
    if len(alive) == 0:
        return None
    s = S[alive, t]
    f = fail[alive].astype(np.float64)
    n = np.bincount(s, minlength=K).astype(np.float64)
    num = np.bincount(s, weights=f, minlength=K)
    g = f.mean()
    return (num + prior_w * g) / (n + prior_w)


def dwell_at(S, valid, t):
    """Current run length of the state at query t (>=1), per branch; 0 if not alive."""
    B = S.shape[0]
    out = np.zeros(B, np.float64)
    for b in range(B):
        if not valid[b, t]:
            continue
        s = S[b]
        d = 1
        while t - d >= 0 and s[t - d] == s[t]:
            d += 1
        out[b] = d
    return out


# ------------------------------------------------------------------ scoring helpers
def win_mean(vals, t, W):
    """Mean of a per-(branch, chunk) quantity over the window [t-W+1, t]."""
    return vals[:, max(0, t - W + 1):t + 1].mean(1)


def auc_pair(scores, y_succ, groups):
    """(auc_succ, det_auc, npairs). auc_succ = P(score|success > score|failure)."""
    a, n = agglib.within_group_auc(np.asarray(scores, float), np.asarray(y_succ, bool),
                                   np.asarray(groups))
    if not np.isfinite(a):
        return np.nan, np.nan, n
    return a, max(a, 1 - a), n


# ------------------------------------------------------------------ persistence
_ROWS = []


def emit(row):
    _ROWS.append(row)


KEYCOLS = ("corpus", "block", "feat", "K", "t", "variant", "mode", "eps", "stat", "metric")


def flush(name="moe_mdp.csv"):
    """Long format so the schema is stable across heterogeneous blocks."""
    if not _ROWS:
        return
    recs = []
    for r in _ROWS:
        keys = {k: r.get(k, "") for k in KEYCOLS}
        for k, v in r.items():
            if k in KEYCOLS:
                continue
            recs.append({**keys, "name": k, "value": v})
    df = pd.DataFrame(recs)
    for d in (TMP, OUT):
        p = os.path.join(d, name)
        hdr = not os.path.exists(p)
        df.to_csv(p, mode="a", header=hdr, index=False)
    _ROWS.clear()


def note(text, name="moe_mdp.md"):
    for d in (TMP, OUT):
        with open(os.path.join(d, name), "a") as fh:
            fh.write(text.rstrip() + "\n")
    print(text, flush=True)
