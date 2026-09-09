"""Full token-axis grid + within-group permutation null."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import token_axis_eval as E

TS = E.TS
RNG = np.random.default_rng(20260829)


def prep(tag, d):
    """Per t: ranks per group per feature + fail indicators, for fast perms."""
    ymap, gmap = E.meta(tag)
    out = {}
    for t in TS:
        eids, X, feats, alive = E.build_matrix(tag, d, t)
        idx = np.where(alive)[0]
        y = np.array([ymap[int(eids[i])] for i in idx])
        g = np.array([gmap[int(eids[i])] for i in idx])
        Xa = X[idx]
        assert not np.isnan(Xa).any(), (tag, d, t, "nan in features")
        groups = []
        npairs = 0
        for gg in np.unique(g):
            m = g == gg
            ns, nf = int(y[m].sum()), int((1 - y[m]).sum())
            if ns == 0 or nf == 0:
                continue
            R = rankdata(Xa[m], axis=0)                 # (n_g, n_feat)
            groups.append(dict(gid=int(gg), R=R, y=y[m], ns=ns, nf=nf))
            npairs += ns * nf
        out[t] = dict(groups=groups, npairs=npairs, feats=feats,
                      n=len(idx), nsucc=int(y.sum()), nfail=int((1 - y).sum()))
    return out


def pooled_auc(prep_t, labels=None):
    """labels: optional list of per-group y arrays (permuted).  Returns (n_feat,)."""
    num = np.zeros(len(prep_t["feats"]))
    for k, G in enumerate(prep_t["groups"]):
        yy = G["y"] if labels is None else labels[k]
        fail = (yy == 0).astype(np.float64)
        nf, ns = G["nf"], G["ns"]
        num += fail @ G["R"] - nf * (nf + 1) / 2.0
    return num / prep_t["npairs"]


def perm_labels(prep_t, rng):
    return [rng.permutation(G["y"]) for G in prep_t["groups"]]


def sweep(tag, d):
    P = prep(tag, d)
    feats = P[TS[0]]["feats"]
    rows = {}
    for t in TS:
        rows[t] = pooled_auc(P[t])
    df = pd.DataFrame(rows, index=feats)
    return df, P


def permutation_family_p(P, obs_max, n_perm=200, two_sided=True, seed=0):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        m = 0.0
        for t in TS:
            a = pooled_auc(P[t], perm_labels(P[t], rng))
            v = np.maximum(a, 1 - a) if two_sided else a
            m = max(m, v.max())
        null[p] = m
    return (1 + (null >= obs_max).sum()) / (1 + n_perm), null
