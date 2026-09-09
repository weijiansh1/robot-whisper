import numpy as np, pandas as pd
from scipy.stats import rankdata
import token_axis_eval as E, token_axis_sweep as S
import zarr

pd.set_option('display.width', 220)

# ---------------------------------------------------------------- 0. is the state token denoise-invariant?
print("### 0. state-token router probs across denoise iterations (are they identical?)")
for tag, p in (("A", E.__dict__["A_LAB"]), ):
    pass
za = zarr.open(store=E.__dict__.get("A_ZARR", "") or
               "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr",
               mode="r")
blk = za["hb_router_probs"][:400, 4:8, :, :, :].astype(np.float32)   # (400,4,10,11,32)
for k in (0, 1, 5, 10):
    v = blk[:, :, :, k, :]
    spread = np.abs(v - v.mean(2, keepdims=True)).max()
    print(f"  corpus A token {k:2d}: max |p(d) - mean_d p| across the 10 denoise iters = {spread:.3e}")
del blk

# ---------------------------------------------------------------- helper
def scores_at(tag, d, t, feats):
    eids, X, allf, alive = E.build_matrix(tag, d, t)
    ymap, gmap = E.meta(tag)
    idx = np.where(alive)[0]
    cols = [allf.index(f) for f in feats]
    return (np.array([int(e) for e in eids])[idx], X[idx][:, cols],
            np.array([ymap[int(eids[i])] for i in idx]),
            np.array([gmap[int(eids[i])] for i in idx]))


def wg_auc(sc, y, g):
    return E.group_auc(sc, y, g)[0]


# ---------------------------------------------------------------- 1. paired bootstrap: best vs baseline
print("\n### 1. paired branch bootstrap, det-AUC(tok00|hell) - det-AUC(act_all_1_10|hell)")
FE = ["tok00|hell", "act_all_1_10|hell", "all11|hell", "act4_7|hell"]
rng = np.random.default_rng(7)
for tag in ("A", "B"):
    for t in (25, 30, 35):
        eid, X, y, g = scores_at(tag, 9, t, FE)
        base = {f: 1 - wg_auc(X[:, j], y, g) for j, f in enumerate(FE)}   # flipped dir
        diffs = {f: [] for f in FE[1:]}
        for b in range(1000):
            take = np.concatenate([rng.choice(np.where(g == gg)[0], (g == gg).sum(), True)
                                   for gg in np.unique(g)])
            yy, gg2 = y[take], g[take]
            if len(np.unique(yy)) < 2:
                continue
            a0 = 1 - wg_auc(X[take, 0], yy, gg2)
            for j, f in enumerate(FE[1:], 1):
                diffs[f].append(a0 - (1 - wg_auc(X[take, j], yy, gg2)))
        print(f"  {tag} t={t}  tok00={base['tok00|hell']:.4f}", end="")
        for f in FE[1:]:
            dv = np.array(diffs[f]); obs = base['tok00|hell'] - base[f]
            print(f" | vs {f}: {base[f]:.4f} d={obs:+.4f} CI95=[{np.percentile(dv,2.5):+.4f},{np.percentile(dv,97.5):+.4f}] P(d>0)={np.mean(dv>0):.3f}", end="")
        print()

# ---------------------------------------------------------------- 2. redundancy: correlation among top scores
print("\n### 2. within-group Spearman corr of scores (t=30, denoise 9)")
FE2 = ["tok00|hell", "act_all_1_10|hell", "act1_3|hell", "act8_10|hell",
       "act_all_1_10|ent", "disp11_hell|ent", "state_vs_actmean_hell|ent"]
for tag in ("A", "B"):
    eid, X, y, g = scores_at(tag, 9, 30, FE2)
    Z = np.zeros_like(X)
    for gg in np.unique(g):
        m = g == gg
        Z[m] = rankdata(X[m], axis=0) / m.sum()
    C = pd.DataFrame(np.corrcoef(Z.T), index=FE2, columns=FE2)
    print(f"  corpus {tag}:"); print(C.round(2).to_string())

# ---------------------------------------------------------------- 3. is the score a proxy for episode length?
print("\n### 3. within-group Spearman corr(score, remaining episode length T) at t=30")
for tag in ("A", "B"):
    eid, X, y, g = scores_at(tag, 9, 30, FE2)
    e2, l, off, F, fn = E.load(tag, 9)
    lm = dict(zip([int(x) for x in e2], l))
    T = np.array([lm[int(e)] for e in eid], float)
    for j, f in enumerate(FE2):
        rs = []
        for gg in np.unique(g):
            m = g == gg
            rs.append(np.corrcoef(rankdata(X[m, j]), rankdata(T[m]))[0, 1] * m.sum())
        print(f"  {tag} {f:28s} rho={np.sum(rs)/len(g):+.3f}   (corr(T,success) "
              f"within group = {np.sum([np.corrcoef(rankdata(T[g==gg]),y[g==gg])[0,1]*(g==gg).sum() for gg in np.unique(g) if len(np.unique(y[g==gg]))>1])/len(g):+.3f})")
    break_ = 0

# ---------------------------------------------------------------- 4. joint (cross-corpus) ranking
print("\n### 4. cross-corpus joint score: min(detA, detB) per org x t (denoise 9)")
dfa = pd.read_csv("/home/jovyan/work/himoe-vla/.tokaxis_grid_A_d9.csv", index_col=0)
dfb = pd.read_csv("/home/jovyan/work/himoe-vla/.tokaxis_grid_B_d9.csv", index_col=0)
da, db = np.maximum(dfa, 1 - dfa), np.maximum(dfb, 1 - dfb)
same = np.sign(dfa - .5) == np.sign(dfb - .5)
joint = np.minimum(da, db).where(same, np.nan)
st = joint.stack().sort_values(ascending=False)
print("  top 15 cells by min(detA,detB) with matching direction:")
for (f, t), v in st[:15].items():
    print(f"    {f:28s} t={t:>2s}  min={v:.4f}  A={da.loc[f,t]:.4f} B={db.loc[f,t]:.4f}")
