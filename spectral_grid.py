"""Within-chunk spectral structure of the (token x denoise) routing grid.

Proposal 2026-09-02: inside ONE control step, take each (token, denoise) cell's
(layer x expert) routing as a 256-dim vector; PCA across the 10 action tokens x
10 denoise steps; arrange PC1 scores as a 10x10 matrix; read the spectrum.
Purely instantaneous -- no history across chunks.

Per control step, X = probs[8 layers, 10 denoise, tokens 1..10, 32] -> (100, 256),
centred on the chunk mean.
  evr1      PC1 explained-variance ratio of the 100x256 cell matrix
  pr_cells  participation ratio of that spectrum (effective # of directions)
  s1frac    10x10 PC1-score matrix (denoise x token) -> sigma_1 / sum sigma
  pr_s      participation ratio of its singular values
  tok_lam1  token Gram (10x10 cosine, mean over denoise) -> lambda_1 / sum
  tok_pr    its participation ratio
  den_lam1  denoise Gram (10x10 cosine, mean over tokens) -> lambda_1 / sum
  den_pr    its participation ratio

Readings and protocol copied from top4mass.py: current / win8 / d_win8;
within-group paired AUC; maxT family null; residualised on the change-rate
baseline; both corpora.
"""
import sys, numpy as np, pandas as pd, zarr
sys.path.insert(0, "/home/jovyan/work/himoe-vla"); sys.path.insert(0, "/home/jovyan/work/himoe-vla/.layer_axis_cache")
import tokdisc as T, core

ZARR = {"A": "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr",
        "B": "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32/server/routes.zarr"}
FEAT_NAMES = ["evr1","pr_cells","s1frac","pr_s","tok_lam1","tok_pr","den_lam1","den_pr"]
READINGS = ["current","win8","d_win8"]
FEATS = [f"{f}|{r}" for f in FEAT_NAMES for r in READINGS]
WIN = 8

def pr(v):
    v = np.asarray(v, float); s = v.sum(); return (s*s)/max((v*v).sum(), 1e-30)

def chunk_features(P):
    """P: (8,10,11,32) one control step -> 8 scalars."""
    X = P[:, :, 1:11, :].astype(np.float32)                 # (8,10,10,32)
    X = np.clip(X, 1e-12, None); X /= X.sum(-1, keepdims=True)
    X = X.transpose(1, 2, 0, 3).reshape(10, 10, 256)          # (denoise, token, layer*expert)
    Xc = X - X.reshape(100, 256).mean(0)                     # centre on chunk mean
    flat = Xc.reshape(100, 256)
    U, s, _ = np.linalg.svd(flat, full_matrices=False)
    ev = s * s
    evr1 = ev[0] / ev.sum(); pr_cells = pr(ev)
    S = (U[:, 0] * s[0]).reshape(10, 10)                     # PC1 score, denoise x token
    sv = np.linalg.svd(S, compute_uv=False)
    s1frac = sv[0] / sv.sum(); pr_s = pr(sv * sv)
    N = Xc / (np.linalg.norm(Xc, axis=-1, keepdims=True) + 1e-12)
    G_tok = np.einsum("dta,dsa->ts", N, N) / 10.0            # token x token
    G_den = np.einsum("dta,eta->de", N, N) / 10.0            # denoise x denoise
    lt = np.clip(np.linalg.eigvalsh(G_tok), 0, None); ld = np.clip(np.linalg.eigvalsh(G_den), 0, None)
    return [evr1, pr_cells, s1frac, pr_s, lt.max()/lt.sum(), pr(lt), ld.max()/ld.sum(), pr(ld)]

def all_features(tag, step=1500):
    import os
    cache = f"/home/jovyan/work/himoe-vla/.layer_axis_cache/spectral_feats_{tag}.npy"
    if os.path.exists(cache): return np.load(cache)
    z = zarr.open_group(ZARR[tag], mode="r")["hb_router_probs"]; n = z.shape[0]
    out = np.empty((n, len(FEAT_NAMES)), np.float32)
    for a in range(0, n, step):
        blk = z[a:a+step]
        for i in range(blk.shape[0]): out[a+i] = chunk_features(blk[i])
        print(f"  {tag} {min(a+step,n)}/{n}", flush=True)
    np.save(cache, out); return out

def evaluate(tag, ts):
    F = all_features(tag)
    _, _, _, _, _, _, br = core.load_corpus(tag)
    hell_eids, hell_lens, hell_off, hell_F = T.load_feats(tag)
    hidx = {int(e): i for i, e in enumerate(hell_eids)}
    out = []
    for t in ts:
        alive = [b for b in br if b["T"] > t]
        X = np.full((len(alive), len(FEATS)), np.nan)
        for i, b in enumerate(alive):
            r = b["rows"]; w = F[r[t-WIN+1:t+1]]; dw = np.diff(F[r[t-WIN:t+1]], axis=0)
            for j, f in enumerate(FEAT_NAMES):
                X[i, FEATS.index(f"{f}|current")] = F[r[t], j]
                X[i, FEATS.index(f"{f}|win8")] = w[:, j].mean()
                X[i, FEATS.index(f"{f}|d_win8")] = dw[:, j].mean()
        g = np.array([b["group"] for b in alive]); fail = ~np.array([b["succ"] for b in alive])
        Hx, _ = T.window_matrix(hell_F, hell_off, hell_lens, t)
        base = np.array([Hx[hidx[int(b["eid"])], T.FAMILY.index(T.BASELINE)] for b in alive])
        Xb = np.column_stack([X, base]); keep = ~np.isnan(Xb).any(1); Xb, gk, fk = Xb[keep], g[keep], fail[keep]
        R = T._group_rank(Xb, gk); auc, W, _ = T.grouped_auc_from_ranks(R, fk, gk); disc = np.maximum(auc, 1-auc)
        maxT, _ = T.perm_null(R[:, :len(FEATS)], fk, gk); p = np.array([(maxT >= d).mean() for d in disc[:len(FEATS)]])
        Rr = T._group_rank(T.rank_residualise(Xb, gk, len(FEATS)), gk)
        auc_r, _, _ = T.grouped_auc_from_ranks(Rr, fk, gk); disc_r = np.maximum(auc_r, 1-auc_r)
        maxT_r, _ = T.perm_null(Rr[:, :len(FEATS)], fk, gk); p_r = np.array([(maxT_r >= d).mean() for d in disc_r[:len(FEATS)]])
        # correlation with baseline (memory: if rho>0.8 it is the same measurement)
        rho = [pd.Series(Xb[:, j]).corr(pd.Series(Xb[:, -1]), method="spearman") for j in range(len(FEATS))]
        for j, f in enumerate(FEATS):
            nm, rd = f.split("|")
            out.append(dict(corpus=tag, t=t, feat=nm, reading=rd, n=len(fk), pairs=W, disc=disc[j], p_family=p[j],
                            disc_resid=disc_r[j], p_family_resid=p_r[j], rho_baseline=rho[j], null_med=np.median(maxT)))
        out.append(dict(corpus=tag, t=t, feat="__BASELINE_hell__", reading="win8", n=len(fk), pairs=W, disc=disc[-1],
                        p_family=np.nan, disc_resid=np.nan, p_family_resid=np.nan, rho_baseline=1.0, null_med=np.median(maxT)))
    return pd.DataFrame(out)

if __name__ == "__main__":
    ts = [15, 20, 25, 30]
    df = pd.concat([evaluate("B", ts), evaluate("A", ts)], ignore_index=True)
    df.to_csv("/home/jovyan/work/himoe-vla/spectral_grid_results.csv", index=False)
    print("wrote spectral_grid_results.csv")
