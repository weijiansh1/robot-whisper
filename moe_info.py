"""How much information is in the MoE route tensor?

Four layers, from nominal to usable:
  L0  nominal numbers stored per control step
  L1  free parameters after the simplex constraint
  L2  effective dimensionality actually spanned by the branches at a fixed query
  L3  how many of those dimensions carry outcome discrimination

Embedding is sqrt(p): Euclidean distance there IS the Hellinger distance used
by every other analysis in this project, so the PCA spectrum is commensurate
with the det numbers.
"""
import json
import sys

import numpy as np
import pandas as pd
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
RS = f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
B_DIR = (f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
         "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
ZARR = {"A": f"{RS}/formal/server/routes.zarr", "B": f"{B_DIR}/server/routes.zarr"}

NL, ND, NT, NE = 8, 10, 11, 32
NPERM = 2000
RNG = np.random.default_rng(20260901)


def branches(tag):
    """[(episode_id, zarr row indices in control_step order, group, success)]"""
    z = zarr.open_group(ZARR[tag], mode="r")
    ep = z["episode_id"][:]
    cs = z["control_step"][:]
    out = []
    if tag == "A":
        lab = pd.read_csv(f"{RS}/analysis/candidate_physical_labels.csv")
        order = np.argsort(cs, kind="stable")
        ep_s = ep[order]
        idx_by_ep = {int(e): order[ep_s == e] for e in np.unique(ep)}
        for _, r in lab.iterrows():
            e = int(r["episode_id"])
            if e in idx_by_ep:
                out.append((e, idx_by_ep[e], int(r["worker"]), bool(r["success"])))
    else:
        summ = json.load(open(f"{B_DIR}/client/summaries.json"))
        ic = np.array([s["inference_calls"] for s in summ])
        off = np.concatenate([[0], np.cumsum(ic)])
        for i, s in enumerate(summ):
            out.append((i, np.arange(off[i], off[i + 1]),
                        int(s["init_state_id"]), bool(s["success"])))
    return out


def route_at(tag, t):
    """Full [n_alive, 8,10,11,32] tensor at query t, plus group/success."""
    z = zarr.open_group(ZARR[tag], mode="r")
    rp = z["hb_router_probs"]
    br = [b for b in branches(tag) if len(b[1]) > t]
    rows = np.array([b[1][t] for b in br])
    X = np.empty((len(rows), NL, ND, NT, NE), np.float32)
    for i, r in enumerate(rows):
        X[i] = rp[int(r)]
    X = np.clip(X, 1e-12, None)
    X /= X.sum(-1, keepdims=True)
    g = np.array([b[2] for b in br])
    y = np.array([b[3] for b in br])
    return X, g, y


def spectrum(M):
    """PCA spectrum of already-centred M (n x d). Returns eigenvalues desc."""
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    lam = s ** 2
    return lam[lam > lam.max() * 1e-12]


def describe(lam):
    tot = lam.sum()
    frac = np.cumsum(lam) / tot
    return dict(pr=float(tot ** 2 / (lam ** 2).sum()),
                d90=int(np.searchsorted(frac, 0.90) + 1),
                d99=int(np.searchsorted(frac, 0.99) + 1),
                pc1=float(lam[0] / tot),
                rank=int(len(lam)))


def group_centre(M, g):
    out = M.copy()
    for k in np.unique(g):
        m = g == k
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def grouped_auc(scores, fail, g):
    num = 0.0
    W = 0.0
    C = 0.0
    for k in np.unique(g):
        m = g == k
        nf = int(fail[m].sum())
        ns = int(m.sum() - nf)
        if not nf or not ns:
            continue
        r = pd.Series(scores[m]).rank().to_numpy()
        num += r[fail[m]].sum()
        W += nf * ns
        C += nf * (nf + 1) / 2.0
    return ((num - C) / W if W else np.nan), W


def pc_discrimination(M, g, fail, npc):
    """det of each of the top-npc principal components + within-group maxT null."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :npc] * s[:npc]
    det = np.array([abs(grouped_auc(Z[:, j], fail, g)[0] - 0.5) + 0.5
                    for j in range(npc)])
    R = np.empty_like(Z)
    for k in np.unique(g):
        m = g == k
        R[m] = pd.DataFrame(Z[m]).rank().to_numpy()
    idx = [(np.where(g == k)[0], int(fail[g == k].sum())) for k in np.unique(g)]
    idx = [(ix, nf) for ix, nf in idx if nf and len(ix) - nf]
    W = sum(nf * (len(ix) - nf) for ix, nf in idx)
    C = sum(nf * (nf + 1) / 2.0 for _, nf in idx)
    maxT = np.empty(NPERM)
    for p in range(NPERM):
        tot = np.zeros(npc)
        for ix, nf in idx:
            tot += R[ix[RNG.choice(len(ix), nf, replace=False)]].sum(0)
        a = (tot - C) / W
        maxT[p] = np.maximum(a, 1 - a).max()
    return det, np.array([(maxT >= d).mean() for d in det]), np.median(maxT)


if __name__ == "__main__":
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"query t={t}\n")
    print(f"L0 nominal per control step : {NL*ND*NT*NE:,} numbers "
          f"({NL}x{ND}x{NT}x{NE})")
    print(f"L1 free after simplex rows  : {NL*ND*NT*(NE-1):,}\n")

    for tag in ("A", "B"):
        X, g, y = route_at(tag, t)
        n = len(X)
        S = np.sqrt(X).reshape(n, -1)            # Hellinger embedding
        fail = ~y
        print(f"=== corpus {tag}: n={n} branches, {len(np.unique(g))} groups, "
              f"{fail.sum()} fail / {(~fail).sum()} succ ===")
        print(f"    sample ceiling on rank: n-1 = {n-1}, "
              f"n-#groups = {n-len(np.unique(g))}")

        raw = describe(spectrum(S - S.mean(0, keepdims=True)))
        cen = describe(spectrum(group_centre(S, g)))
        print(f"    raw            PR={raw['pr']:7.2f}  d90={raw['d90']:4d}  "
              f"d99={raw['d99']:4d}  PC1={raw['pc1']*100:5.1f}%")
        print(f"    group-centred  PR={cen['pr']:7.2f}  d90={cen['d90']:4d}  "
              f"d99={cen['d99']:4d}  PC1={cen['pc1']*100:5.1f}%")

        npc = min(20, n - len(np.unique(g)) - 1)
        det, p, nullmed = pc_discrimination(group_centre(S, g), g, fail, npc)
        sig = np.where(p < 0.05)[0]
        print(f"    top-{npc} PCs of the group-centred tensor, det (null median "
              f"{nullmed:.3f}):")
        print("      " + "  ".join(f"PC{j+1}:{det[j]:.3f}{'*' if p[j]<0.05 else ''}"
                                   for j in range(min(10, npc))))
        print(f"    PCs passing family maxT p<0.05 : "
              f"{len(sig)} / {npc}   -> {[int(j+1) for j in sig]}\n")
