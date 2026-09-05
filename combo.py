"""Does ac_top4mass ADD to the route-change feature when the two are combined?

Two combination modes, both on the baseline organization (action tokens 1-10,
deep block 12-15, d9):

  unfitted   within-group rank-normalise both to [0,1], then H+M / H-M.
             No free parameters, so the within-group label permutation is
             exactly valid -- no selection to price.

  fitted     logistic regression on [H, M] with leave-one-GROUP-out CV
             (A: leave-one-worker-out, B: leave-one-init-state-out), scored
             out-of-fold. Compared against logistic on [H] alone under the
             identical CV, so the increment is apples to apples.

Increment is reported with its per-group sign, because with 4 groups (corpus A)
an exact sign test bottoms out at p=0.0625 -- that is the floor, not a result.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import tokdisc as T
import top4mass as M4

TS = [15, 20, 25, 30]
ORG = "act_all_1_10"
READINGS = ["current", "win8", "d_win8"]
NPERM = 2000
RNG = np.random.default_rng(20260901)


def rank01(v, g):
    """Within-group rank scaled to [0,1]."""
    out = np.empty_like(v, dtype=float)
    for k in np.unique(g):
        m = g == k
        r = pd.Series(v[m]).rank().to_numpy()
        out[m] = (r - 1) / max(len(r) - 1, 1)
    return out


def det_of(score, fail, g):
    a, _, _ = T.grouped_auc_from_ranks(
        T._group_rank(score[:, None], g), fail, g)
    return float(max(a[0], 1 - a[0]))


def perm_det_null(score_fns, fail, g, nperm=NPERM):
    """maxT null over a family of unfitted scores (labels permuted in-group)."""
    Rs = [T._group_rank(s[:, None], g) for s in score_fns]
    idx = [(np.where(g == k)[0], int(fail[g == k].sum())) for k in np.unique(g)]
    idx = [(ix, nf) for ix, nf in idx if nf and len(ix) - nf]
    W = sum(nf * (len(ix) - nf) for ix, nf in idx)
    C = sum(nf * (nf + 1) / 2.0 for _, nf in idx)
    out = np.empty(nperm)
    for p in range(nperm):
        best = 0.0
        picks = [(ix, ix[RNG.choice(len(ix), nf, replace=False)]) for ix, nf in idx]
        for R in Rs:
            tot = sum(R[sel].sum() for _, sel in picks)
            a = (tot - C) / W
            best = max(best, max(a, 1 - a))
        out[p] = best
    return out


def logo_oof(X, fail, g):
    """Out-of-fold logistic probabilities under leave-one-group-out."""
    oof = np.zeros(len(fail))
    for k in np.unique(g):
        te = g == k
        tr = ~te
        if len(np.unique(fail[tr])) < 2:
            oof[te] = 0.5
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X[tr] - mu) / sd, fail[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof


def per_group_det(score, fail, g):
    out = {}
    for k in np.unique(g):
        m = g == k
        nf = int(fail[m].sum())
        ns = int(m.sum() - nf)
        if not nf or not ns:
            continue
        r = pd.Series(score[m]).rank().to_numpy()
        a = (r[fail[m]].sum() - nf * (nf + 1) / 2) / (nf * ns)
        out[int(k)] = a
    return out


def run(tag):
    Mser = M4.top4_series(tag)
    _, _, _, _, _, _, br = M4.load_corpus(tag)
    eids, lens, off, F = T.load_feats(tag)
    hidx = {int(e): i for i, e in enumerate(eids)}
    j_org = M4.ORGS.index(ORG)
    rows = []
    for t in TS:
        alive = [b for b in br if b["T"] > t]
        Hx, _ = T.window_matrix(F, off, lens, t)
        H = np.array([Hx[hidx[int(b["eid"])], T.FAMILY.index(T.BASELINE)]
                      for b in alive])
        Ms = {}
        for b_i, b in enumerate(alive):
            r = b["rows"]
            Ms.setdefault("current", []).append(Mser[r[t], j_org])
            Ms.setdefault("win8", []).append(
                Mser[r[t - M4.WIN + 1:t + 1], j_org].mean())
            Ms.setdefault("d_win8", []).append(
                np.diff(Mser[r[t - M4.WIN:t + 1], j_org]).mean())
        Ms = {k: np.array(v) for k, v in Ms.items()}
        g = np.array([b["group"] for b in alive])
        fail = ~np.array([b["succ"] for b in alive])
        ok = ~np.isnan(H)
        H, g, fail = H[ok], g[ok], fail[ok]
        Ms = {k: v[ok] for k, v in Ms.items()}

        h01 = rank01(H, g)
        det_H = det_of(h01, fail, g)

        fam, names = [], []
        for rd in READINGS:
            m01 = rank01(Ms[rd], g)
            fam += [h01 + m01, h01 - m01]
            names += [f"H+M[{rd}]", f"H-M[{rd}]"]
        null = perm_det_null(fam + [h01], fail, g)
        for nm, sc in zip(names, fam):
            d = det_of(sc, fail, g)
            rows.append(dict(corpus=tag, t=t, mode="unfitted", combo=nm,
                             det=d, det_H=det_H, delta=d - det_H,
                             p_family=float((null >= d).mean()),
                             null_med=float(np.median(null)), n=len(fail)))

        oof_H = logo_oof(H[:, None], fail, g)
        dH = det_of(oof_H, fail, g)
        pgH = per_group_det(oof_H, fail, g)
        for rd in READINGS:
            X = np.column_stack([H, Ms[rd]])
            oof = logo_oof(X, fail, g)
            d = det_of(oof, fail, g)
            pg = per_group_det(oof, fail, g)
            signs = [np.sign(abs(pg[k] - .5) - abs(pgH[k] - .5)) for k in pg]
            rows.append(dict(corpus=tag, t=t, mode="fitted-LOGO",
                             combo=f"H+M[{rd}]", det=d, det_H=dH, delta=d - dH,
                             p_family=np.nan, null_med=np.nan, n=len(fail),
                             groups_improved=int(sum(s > 0 for s in signs)),
                             groups=len(signs)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.concat([run("A"), run("B")], ignore_index=True)
    df.to_csv("/home/jovyan/work/himoe-vla/combo_results.csv", index=False)
    print("wrote combo_results.csv")
