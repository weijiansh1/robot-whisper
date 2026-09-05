"""ac_top4mass vs the token-organization change features.

top-4 mass = sum of the 4 largest renormalised router probabilities per
(layer, token). It uses VALUES only, never the top-4 identities, so the known
23.8% 4th/5th fp16 ties do not affect it.

Reported in three readings so the comparison against |hell is fair:
  current      instantaneous value at query t          (no history)
  win8         8-step trailing window mean             (matches the |hell protocol)
  d_win8       8-step window mean of the step-to-step increment (a change statistic)
"""
import numpy as np
import pandas as pd

import tokdisc as T

CACHE = "/home/jovyan/work/himoe-vla/.layer_axis_cache"
DEEP = slice(4, 8)
NTOK = 11
TOK_SUBSETS = {**{f"tok{k:02d}": [k] for k in range(NTOK)},
               "act1_3": [1, 2, 3], "act4_7": [4, 5, 6, 7], "act8_10": [8, 9, 10],
               "act_all_1_10": list(range(1, 11)), "all11": list(range(11))}
ORGS = list(TOK_SUBSETS)
READINGS = ["current", "win8", "d_win8"]
FEATS = [f"{o}|{r}" for o in ORGS for r in READINGS]
WIN = 8


def load_corpus(tag):
    import sys
    sys.path.insert(0, CACHE)
    import core
    return core.load_corpus(tag)


def top4_series(tag):
    """(N, 16 orgs) top-4 mass per zarr row, deep block averaged."""
    P = np.load(f"{CACHE}/{tag}_d9.npy")[:, DEEP].astype(np.float32)   # (N,4,11,32)
    P = np.clip(P, 1e-12, None)
    P /= P.sum(-1, keepdims=True)
    m = np.partition(P, -4, axis=-1)[..., -4:].sum(-1)                 # (N,4,11)
    return np.stack([m[:, :, TOK_SUBSETS[o]].mean((1, 2)) for o in ORGS], 1)


def build(tag, ts):
    M = top4_series(tag)
    _, _, _, _, _, _, br = load_corpus(tag)
    rows = []
    for t in ts:
        alive = [b for b in br if b["T"] > t]
        X = np.full((len(alive), len(FEATS)), np.nan)
        for i, b in enumerate(alive):
            r = b["rows"]
            w = M[r[t - WIN + 1: t + 1]]                     # (8, 16)
            dw = np.diff(M[r[t - WIN: t + 1]], axis=0)       # (8, 16)
            for j, o in enumerate(ORGS):
                X[i, FEATS.index(f"{o}|current")] = M[r[t], j]
                X[i, FEATS.index(f"{o}|win8")] = w[:, j].mean()
                X[i, FEATS.index(f"{o}|d_win8")] = dw[:, j].mean()
        g = np.array([b["group"] for b in alive])
        fail = ~np.array([b["succ"] for b in alive])
        rows.append((t, X, g, fail, [b["eid"] for b in alive]))
    return rows


def evaluate(tag, ts):
    out = []
    hell_eids, hell_lens, hell_off, hell_F = T.load_feats(tag)
    hidx = {int(e): i for i, e in enumerate(hell_eids)}
    for t, X, g, fail, eids in build(tag, ts):
        # the change baseline, aligned to the same branches
        Hx, _ = T.window_matrix(hell_F, hell_off, hell_lens, t)
        base = np.array([Hx[hidx[int(e)], T.FAMILY.index(T.BASELINE)] for e in eids])

        Xb = np.column_stack([X, base])
        keep = ~np.isnan(Xb).any(1)
        Xb, gk, fk = Xb[keep], g[keep], fail[keep]

        R = T._group_rank(Xb, gk)
        auc, W, _ = T.grouped_auc_from_ranks(R, fk, gk)
        disc = np.maximum(auc, 1 - auc)
        maxT, _ = T.perm_null(R[:, :len(FEATS)], fk, gk)
        p = np.array([(maxT >= d).mean() for d in disc[:len(FEATS)]])

        # residualise every top4mass reading on the change baseline
        Rr = T._group_rank(T.rank_residualise(Xb, gk, len(FEATS)), gk)
        auc_r, _, _ = T.grouped_auc_from_ranks(Rr, fk, gk)
        disc_r = np.maximum(auc_r, 1 - auc_r)
        maxT_r, _ = T.perm_null(Rr[:, :len(FEATS)], fk, gk)
        p_r = np.array([(maxT_r >= d).mean() for d in disc_r[:len(FEATS)]])

        # and the reverse: is the change baseline explained BY ac_top4mass?
        j_ac = FEATS.index("act_all_1_10|win8")
        Rrev = T._group_rank(T.rank_residualise(Xb, gk, j_ac), gk)
        auc_rev, _, _ = T.grouped_auc_from_ranks(Rrev, fk, gk)

        for j, f in enumerate(FEATS):
            org, reading = f.split("|")
            out.append(dict(corpus=tag, t=t, org=org, reading=reading,
                            n=len(fk), pairs=W, auc=auc[j], disc=disc[j],
                            p_family=p[j], disc_resid=disc_r[j],
                            p_family_resid=p_r[j], null_med=np.median(maxT)))
        out.append(dict(corpus=tag, t=t, org="__BASELINE_hell__", reading="win8",
                        n=len(fk), pairs=W, auc=auc[len(FEATS)],
                        disc=disc[len(FEATS)], p_family=np.nan,
                        disc_resid=max(auc_rev[len(FEATS)], 1 - auc_rev[len(FEATS)]),
                        p_family_resid=np.nan, null_med=np.median(maxT)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ts = [10, 15, 20, 25, 30]
    df = pd.concat([evaluate("A", ts), evaluate("B", ts)], ignore_index=True)
    df.to_csv("/home/jovyan/work/himoe-vla/top4mass_results.csv", index=False)
    print("wrote top4mass_results.csv")
