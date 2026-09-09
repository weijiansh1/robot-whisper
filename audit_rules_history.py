#!/usr/bin/env python3
"""Audit (c): does the t08 rule increment survive a baseline that also sees
short pose history?

The honest-split increment (+0.011~0.013 over scene+full sim state+action
chunk) was measured against baselines built from the *current* control step
only.  The routing's state token absorbs the prefix with depth, and the rule's
cross-layer conjunction (L2 reads pose nearly raw, L5 reads it prefix-mixed)
could simply be a pose-history predicate.  So: same protocol as
analyze_rules_intention.py (HistGBT residual baselines grouped by episode,
20 scene-splits, top-rule + top-20-union, paired rule-minus-random), plus one
baseline that appends cheap proprio-history features:

    lagged proprio (t-1, t-2, t-4, t-8), step speeds at those lags, cumulative
    path length, gripper transitions so far, steps since last transition,
    min 3-step speed so far.

If the increment over "scene+physical+action+history" collapses to the random
floor, the rule is a pose-history readout and the last anomaly closes.
Writes audit_rules_history.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = pathlib.Path(__file__).resolve().parent
NPZ = (HERE / "himoe-routing-rules-20260819/data"
       / "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
N_REP = 20
TOPK = 20
RNG = np.random.default_rng(0)


def lit_name(i):
    return "L%d:e%d" % (HB_LAYER[i // 32], i % 32)


def cooccur(M):
    n = M.shape[0]
    out = np.zeros((n, 256, 256), bool)
    for i in range(n):
        out[i] = (M[i].T @ M[i]) > 0.5
    return out


def score(fires, fail, p_fail):
    F = fires.astype(np.float32)
    sup = np.einsum("iab->ab", F)
    obs = np.einsum("i,iab->ab", fail.astype(np.float32), F)
    exp = np.einsum("i,iab->ab", p_fail, F)
    var = np.einsum("i,iab->ab", p_fail * (1 - p_fail), F)
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = (obs - exp) / np.sqrt(np.maximum(var, 1e-9))
    return sup, obs, exp, zz


def loso_p(X, yf, ep_of_row):
    p = np.zeros(len(yf))
    for tr, te in GroupKFold(n_splits=8).split(X, yf, groups=ep_of_row):
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=4,
            l2_regularization=1.0, random_state=0).fit(X[tr], yf[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main() -> int:
    z = np.load(NPZ, allow_pickle=True)
    n_rows = z["n_rows"]
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = z["success"].astype(bool)
    scene = z["scene"]
    n = len(y)
    W = int(n_rows.min())
    fail = ~y

    ids = z["state_token_top4"].astype(np.int64)
    hot = np.zeros((ids.shape[0], 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    M = np.zeros((n, W, 256), np.float32)
    prop = np.zeros((n, W, 8), np.float32)
    sim = np.zeros((n, W, z["sim_state"].shape[1]), np.float32)
    act = np.zeros((n, W, 70), np.float32)
    for i in range(n):
        sl = slice(off[i], off[i] + W)
        M[i] = hot[sl].reshape(W, 256)
        prop[i] = z["proprio"][sl]
        sim[i] = z["sim_state"][sl]
        act[i] = z["actions"][sl]

    # ---- history features from proprio only --------------------------------
    def lag(X, k):
        L = np.empty_like(X)
        L[:, k:] = X[:, :-k] if k else X
        L[:, :k] = X[:, :1]
        return L

    speed = np.zeros((n, W), np.float32)
    speed[:, 1:] = np.linalg.norm(np.diff(prop, axis=1), axis=2)
    cum = np.cumsum(speed, axis=1)
    opening = prop[:, :, 6] - prop[:, :, 7]
    closed = opening < 0.04
    tr_ev = np.zeros((n, W), np.float32)
    tr_ev[:, 1:] = (closed[:, 1:] != closed[:, :-1]).astype(np.float32)
    trans_so_far = np.cumsum(tr_ev, axis=1)
    since = np.zeros((n, W), np.float32)
    for t in range(1, W):
        since[:, t] = np.where(tr_ev[:, t] > 0, 0, since[:, t - 1] + 1)
    s3 = np.stack([lag(speed[..., None], k)[..., 0] for k in (0, 1, 2)], -1).mean(-1)
    minslow = np.minimum.accumulate(np.where(np.arange(W) >= 3, s3, np.inf), 1)
    minslow[~np.isfinite(minslow)] = 0.0

    hist_parts = [lag(prop, k).reshape(n * W, 8) for k in (1, 2, 4, 8)]
    hist_parts += [lag(speed[..., None], k).reshape(n * W, 1) for k in (0, 1, 2, 4)]
    hist_parts += [cum.reshape(-1, 1), trans_so_far.reshape(-1, 1),
                   since.reshape(-1, 1), minslow.reshape(-1, 1)]
    HIST = np.concatenate(hist_parts, 1).astype(np.float32)

    ep_row = np.repeat(np.arange(n), W)
    step = np.tile(np.arange(W), n).astype(np.float32)[:, None]
    yf = np.repeat(fail, W).astype(int)
    si = np.unique(scene, return_inverse=True)[1]
    oh = np.zeros((n * W, si.max() + 1), np.float32)
    oh[np.arange(n * W), si[ep_row]] = 1
    PH = np.c_[oh, sim.reshape(n * W, -1), prop.reshape(-1, 8), step]
    AC = act.reshape(-1, 70)

    base = {"phys+act": np.c_[PH, AC],
            "phys+act+hist": np.c_[PH, AC, HIST]}
    P, ll = {}, {}
    print("t08, %d eps x %d steps; history dims = %d" % (n, W, HIST.shape[1]))
    for k, X in base.items():
        P[k] = loso_p(X, yf, ep_row)
        ll[k] = float(-(yf * np.log(np.clip(P[k], 1e-6, 1)) +
                        (1 - yf) * np.log(np.clip(1 - P[k], 1e-6, 1))
                        ).mean() / np.log(2))
        print("  baseline %-14s (%4d dims)  step log-loss %.3f bit"
              % (k, X.shape[1], ll[k]))

    fires = cooccur(M)
    res = {k: (yf - v).reshape(n, W) for k, v in P.items()}
    rand_hit = np.random.default_rng(3).random((n, W)) < 0.10
    acc = {k: {"top": [], "pool": [], "rnd": []} for k in base}
    names = []
    for r in range(N_REP):
        g = np.random.default_rng(100 + r)
        sc = g.permutation(np.unique(scene))
        A = np.isin(scene, sc[:len(sc) // 2])
        B = ~A
        pfA = np.zeros(A.sum(), np.float32)
        scA = scene[A]
        for s in np.unique(scA):
            pfA[scA == s] = fail[A][scA == s].mean()
        sA, oA, eA, zA = score(fires[A], fail[A], pfA)
        ok = (sA >= 10) & (sA <= A.sum() - 3)
        ok[np.tril_indices(256, -1)] = False
        order = np.argsort(np.where(ok, zA, -np.inf).ravel())[::-1]
        a, b = np.unravel_index(order[0], zA.shape)
        names.append("%s & %s" % (lit_name(a), lit_name(b)))
        top = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
        pool = np.zeros((n, W), bool)
        for f in order[:TOPK]:
            u, v = np.unravel_index(f, zA.shape)
            pool |= (M[:, :, u] > 0.5) & (M[:, :, v] > 0.5)
        for k in base:
            acc[k]["top"].append(res[k][B][top[B]].mean()
                                 if top[B].sum() > 20 else np.nan)
            acc[k]["pool"].append(res[k][B][pool[B]].mean()
                                  if pool[B].sum() > 20 else np.nan)
            acc[k]["rnd"].append(res[k][B][rand_hit[B]].mean())

    out = {"task": "libero_long/t08", "n": int(n), "win": int(W),
           "hist_dims": int(HIST.shape[1]), "logloss": ll}
    print("\n  paired rule-minus-random over %d scene-splits:" % N_REP)
    for lab in ("top", "pool"):
        row = "    %-6s" % lab
        for k in base:
            d = np.array(acc[k][lab]) - np.array(acc[k]["rnd"])
            fin = np.isfinite(d)
            m = float(np.nanmean(d))
            se = float(np.nanstd(d) / np.sqrt(fin.sum()))
            out.setdefault(lab, {})[k] = {
                "mean": m, "sem": se,
                "pos": int((d[fin] > 0).sum()), "n": int(fin.sum())}
            row += "   %s %+0.4f ± %.4f (%2d/%2d)" % (
                k, m, se, out[lab][k]["pos"], out[lab][k]["n"])
        print(row)
    cnt = {}
    for nm in names:
        cnt[nm] = cnt.get(nm, 0) + 1
    out["top_rules"] = sorted(cnt.items(), key=lambda t: -t[1])[:3]
    print("  mined rules:", out["top_rules"])

    (HERE / "audit_rules_history.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_rules_history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
