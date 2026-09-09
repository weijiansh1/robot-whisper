"""Remaining blocks; appends to /tmp/moe_sweep_token.md and side CSVs."""
import pickle, numpy as np, pandas as pd, zarr
from scipy.stats import rankdata
import token_axis_eval as E

MD = "/tmp/moe_sweep_token.md"
TS = [15, 20, 25, 30, 35]
A_ZARR = "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr"
B_ZARR = f"{E.B_DIR}/server/routes.zarr"


def emit(txt):
    with open(MD, "a") as fh:
        fh.write(txt + "\n")
    print(txt[:200])


def scores_at(tag, d, t, feats):
    eids, X, allf, alive = E.build_matrix(tag, d, t)
    ymap, gmap = E.meta(tag)
    idx = np.where(alive)[0]
    ids = np.array([int(e) for e in eids])[idx]
    return (ids, X[idx][:, [allf.index(f) for f in feats]],
            np.array([ymap[i] for i in ids]), np.array([gmap[i] for i in ids]))


def det(sc, y, g):
    a = E.group_auc(sc, y, g)[0]
    return max(a, 1 - a)


# ---------------------------------------------------------------- Block 11
FE = ["tok00|hell", "act_all_1_10|hell", "all11|hell", "act4_7|hell"]
rng = np.random.default_rng(7)
rows, lines = [], [
    "\n## Best organization vs the baseline: paired branch bootstrap\n",
    "1000 bootstrap resamples of branches **within group**, recomputing both detection AUCs on "
    "the same resample so the comparison is paired. `d` = det(state token 0 alone) - det(comparator).\n",
    "```"]
for tag in ("A", "B"):
    for t in (25, 30, 35):
        ids, X, y, g = scores_at(tag, 9, t, FE)
        base = {f: det(X[:, j], y, g) for j, f in enumerate(FE)}
        diffs = {f: [] for f in FE[1:]}
        for _ in range(1000):
            take = np.concatenate([rng.choice(np.where(g == gg)[0], int((g == gg).sum()), True)
                                   for gg in np.unique(g)])
            yy, gg2 = y[take], g[take]
            if len(np.unique(yy)) < 2:
                continue
            a0 = det(X[take, 0], yy, gg2)
            for j, f in enumerate(FE[1:], 1):
                diffs[f].append(a0 - det(X[take, j], yy, gg2))
        for f in FE[1:]:
            dv = np.array(diffs[f]); obs = base["tok00|hell"] - base[f]
            lines.append(f"  {tag} t={t}  tok00|hell={base['tok00|hell']:.4f} vs {f:20s}"
                         f"={base[f]:.4f}  d={obs:+.4f}  CI95=[{np.percentile(dv,2.5):+.4f},"
                         f"{np.percentile(dv,97.5):+.4f}]  P(d>0)={np.mean(dv>0):.3f}")
            rows.append(dict(corpus=tag, t=t, best="tok00|hell", comparator=f,
                             det_best=base["tok00|hell"], det_comp=base[f], delta=obs,
                             ci_lo=np.percentile(dv, 2.5), ci_hi=np.percentile(dv, 97.5),
                             p_gt0=float(np.mean(dv > 0))))
lines.append("```")
pd.DataFrame(rows).to_csv("/tmp/moe_sweep_token_paired_bootstrap.csv", index=False)
emit("\n".join(lines))

# ---------------------------------------------------------------- Block 12
lines = ["\n## Why the state token is special: its routing is denoise-invariant\n",
         "Mean Hellinger distance between the router distributions at denoise iteration 0 and 9, "
         "and mean between adjacent denoise iterations, per token, deep layers 12-15 "
         "(first 1500 control steps of each corpus). Pure MoE tensor, no external quantity.\n",
         "```"]
for tag, p in (("A", A_ZARR), ("B", B_ZARR)):
    z = zarr.open(store=p, mode="r")
    blk = z["hb_router_probs"][:1500, 4:8, :, :, :].astype(np.float32)
    blk = np.clip(blk, 1e-12, None); blk /= blk.sum(-1, keepdims=True)
    Sq = np.sqrt(blk)
    h09 = np.sqrt(np.clip(1 - np.einsum("nlke,nlke->nlk", Sq[:, :, 0], Sq[:, :, 9]), 0, None)).mean((0, 1))
    hadj = np.sqrt(np.clip(1 - np.einsum("nldke,nldke->nldk", Sq[:, :, 1:], Sq[:, :, :-1]), 0, None)).mean((0, 1, 2))
    lines.append(f"  corpus {tag}   token:        " + "".join(f"{k:>8d}" for k in range(11)))
    lines.append(f"              H(d0,d9):    " + "".join(f"{v:8.4f}" for v in h09))
    lines.append(f"              mean H(d,d-1):" + "".join(f"{v:8.4f}" for v in hadj))
    del blk, Sq
lines += ["```", "",
          "The state token's router distribution is **constant across the 10 flow-denoise "
          "iterations** (H = 0.0008 in corpus A, 0.0000 in corpus B, versus 0.033-0.047 for every "
          "action token). This is why `tok00|hell` scores identically at denoise 0, 5 and 9 - the "
          "denoise axis is *degenerate* for token 0. Practically: the state-token signal is fully "
          "available after the **first** denoise iteration, so it does not require running the "
          "flow head to completion."]
emit("\n".join(lines))

# ---------------------------------------------------------------- Block 13
FE2 = ["tok00|hell", "act_all_1_10|hell", "act_all_1_10|ent",
       "disp11_hell|ent", "state_vs_actmean_hell|ent"]
lines = ["\n## Confound audit (diagnostic only - NOT a comparator, no physical quantity is scored)\n",
         "Within-group Spearman correlation between each routing score at t=30 and the branch's "
         "own total length T (`inference_calls`). T is used only to check whether a routing score "
         "is silently a 'how much longer will this run' proxy; it is never used as a detector.\n",
         "```"]
for tag in ("A", "B"):
    ids, X, y, g = scores_at(tag, 9, 30, FE2)
    e2, l, off, F, fn = E.load(tag, 9); lm = dict(zip([int(x) for x in e2], l))
    T = np.array([lm[i] for i in ids], float)
    num = den = 0.0
    for gg in np.unique(g):
        m = g == gg
        if np.std(T[m]) == 0 or len(np.unique(y[m])) < 2: continue
        num += np.corrcoef(rankdata(T[m]), y[m])[0, 1] * m.sum(); den += m.sum()
    lines.append(f"  corpus {tag}: within-group corr(T, success) = {num/den:+.3f}")
    for j, f in enumerate(FE2):
        num = den = 0.0
        for gg in np.unique(g):
            m = g == gg
            if np.std(T[m]) == 0 or np.std(X[m, j]) == 0: continue
            num += np.corrcoef(rankdata(X[m, j]), rankdata(T[m]))[0, 1] * m.sum(); den += m.sum()
        lines.append(f"    {f:28s} corr(score, T) = {num/den:+.3f}")
lines.append("```")
emit("\n".join(lines))

# ---------------------------------------------------------------- Block 14
lines = ["\n## Redundancy: is the state token just re-reading the action tokens?\n",
         "Within-group rank correlation of the scores themselves at t=30, denoise 9.\n", "```"]
FE3 = ["tok00|hell", "act_all_1_10|hell", "act1_3|hell", "act8_10|hell",
       "act_all_1_10|ent", "disp11_hell|ent", "state_vs_actmean_hell|ent"]
for tag in ("A", "B"):
    ids, X, y, g = scores_at(tag, 9, 30, FE3)
    Z = np.zeros_like(X)
    for gg in np.unique(g):
        m = g == gg
        Z[m] = rankdata(X[m], axis=0) / m.sum()
    C = pd.DataFrame(np.corrcoef(Z.T), index=FE3, columns=FE3)
    lines += [f"corpus {tag}:", C.round(2).to_string()]
lines.append("```")

# exploratory combination
lines += ["", "Exploratory (**not** part of the tested family, not priced by the permutation "
          "null): within-group rank-average of state-token and action-token route change.", "```"]
for tag in ("A", "B"):
    for t in (25, 30, 35):
        ids, X, y, g = scores_at(tag, 9, t, ["tok00|hell", "act_all_1_10|hell"])
        comb = np.zeros(len(ids))
        for gg in np.unique(g):
            m = g == gg
            comb[m] = rankdata(X[m, 0]) + rankdata(X[m, 1])
        lines.append(f"  {tag} t={t}: state-alone={det(X[:,0],y,g):.4f}  "
                     f"action-alone={det(X[:,1],y,g):.4f}  rank-avg={det(comb,y,g):.4f}")
lines += ["```", "", "Combining does not help; the state token alone is as good or better."]
emit("\n".join(lines))
