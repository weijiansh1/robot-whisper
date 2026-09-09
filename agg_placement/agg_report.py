"""Synthesis of the aggregation-method grid -> /tmp/moe_agg_placement.md"""
import numpy as np
import pandas as pd

import agg_core as A

TS = A.TS
NP = 200
BIDX = A.CFG_IDX[A.BASELINE]
OUT = "/tmp/moe_agg_placement.md"
BLK = []


def w(s=""):
    BLK.append(s)
    with open(OUT, "w") as fh:
        fh.write("\n".join(BLK) + "\n")


def factors():
    rec = []
    for c in A.CFGS:
        (ls, lp, lo), (ds, dp, do), (ts_, tp, to) = c
        rec.append(dict(layer_scope=ls, layer_place=lp, layer_op=lo,
                        den_scope=ds, den_place=dp, den_op=do,
                        tok_scope=ts_, tok_place=tp, tok_op=to,
                        placement=A.placement_str(c), name=A.cfg_name(c)))
    return pd.DataFrame(rec)


F = factors()
# fully-crossed subgrid (drops the size-1 scopes that force place/op to be degenerate)
CROSS = ((F.den_scope == "all") & (F.tok_scope != "state")).values


def load(tag):
    z = np.load(f"/tmp/agg_stats_{tag}.npz")
    d = {}
    for nm in ("raw", "res"):
        d[nm] = np.stack([z[f"auc_{nm}_{t}"] for t in TS], -1)          # (NCFG, nt)
        d[nm + "_det"] = np.maximum(d[nm], 1 - d[nm])
        d[nm + "_pw"] = np.stack([z[f"pw_{nm}_{t}"] for t in TS], -1)
        d[nm + "_null"] = z[f"null_{nm}"]
    d["rho"] = np.stack([z[f"rho_{t}"] for t in TS], -1)
    d["npos"], d["nneg"] = z["npos"], z["nneg"]
    return d


def eta2(det, mask):
    """One-way eta^2 of each design factor on detection AUC, on the crossed subgrid."""
    v = det[mask]
    out = {}
    tot = ((v - v.mean()) ** 2).sum()
    for f in ["layer_scope", "layer_place", "layer_op", "den_place", "den_op",
              "tok_scope", "tok_place", "tok_op"]:
        lv = F[f].values[mask]
        ss = 0.0
        for u in np.unique(lv):
            m = lv == u
            ss += m.sum() * (v[m].mean() - v.mean()) ** 2
        out[f] = ss / tot
    return pd.Series(out)


def main():
    Ad, Bd = load("A"), load("B")
    w("# Aggregation-method brute force on the HiMoE-VLA MoE routing tensor")
    w()
    w("Placement x operator x axis-scope, with divergence = Hellinger and temporal "
      "aggregation = mean over an 8-step window held fixed (sibling agent's axis).")
    w()
    w("## 0. Grid definition")
    w()
    w("```")
    w(f"layer   : scope {{all(8), back 12-15, front 2-5}} x place {{Before, After}} x op x6  = 36")
    w(f"denoise : scope all(10)  x place {{B,A}} x op x6  +  scope d9 (size 1, degenerate) = 13")
    w(f"token   : scope {{all11, act 1-10}} x place {{B,A}} x op x6  +  state (size 1)      = 25")
    w(f"total   : 36 x 13 x 25 = {A.NCFG} cells  x  t in {TS}  x  2 corpora")
    w("ops     : mean, max, min, median, std, q75")
    w("place=B : aggregate the 32-dim probability vectors along that axis (re-normalise),")
    w("          then take ONE Hellinger against the previous chunk.")
    w("place=A : one Hellinger per element of that axis, then aggregate the scalars.")
    w("order   : within the Before group and within the After group, axes are reduced")
    w("          layer -> denoise -> token (see agg_order_probe.py for order sensitivity).")
    w("```")
    w()
    w(f"Baseline cell = `{A.cfg_name(A.BASELINE)}` (index {BIDX}): divergence-first over "
      "layer and token, back block, action tokens, denoise 9.")
    w()
    w("| corpus | t | baseline raw auc | baseline det AUC | brief target |")
    w("|---|---|---|---|---|")
    for tag, d, tgt in [("A", Ad, 0.795), ("B", Bd, 0.759)]:
        for j, t in enumerate(TS):
            w(f"| {tag} | {t} | {d['raw'][BIDX,j]:.4f} | {d['raw_det'][BIDX,j]:.4f} | "
              f"{tgt if t==30 else ''} |")
    w()
    w("Reproduction check at t=30: A "
      f"{Ad['raw_det'][BIDX,2]:.4f} vs 0.795, B {Bd['raw_det'][BIDX,2]:.4f} vs 0.759 "
      "-> both within 0.01. Convention: raw `auc` = P(feature_fail > feature_succ); "
      "detection AUC = max(auc, 1-auc). Route change is LOWER on failures at t=30 "
      "(raw auc < 0.5) in both corpora.")
    w()

    # ---------------------------------------------------------------- 1 grid
    w("## 1. The full grid")
    w()
    for nm, lab in [("raw", "RAW"), ("res", "RESIDUALISED on the baseline")]:
        w(f"### {lab} detection AUC, distribution over all {A.NCFG} cells")
        w()
        w("| corpus | t | min | q25 | median | q75 | max | baseline | #cells > baseline |")
        w("|---|---|---|---|---|---|---|---|---|")
        for tag, d in [("A", Ad), ("B", Bd)]:
            for j, t in enumerate(TS):
                v = d[nm + "_det"][:, j]
                b = d["raw_det"][BIDX, j]
                w(f"| {tag} | {t} | {v.min():.3f} | {np.percentile(v,25):.3f} | "
                  f"{np.median(v):.3f} | {np.percentile(v,75):.3f} | {v.max():.3f} | "
                  f"{b:.3f} | {(v > b).sum()} |")
        w()

    # ---------------------------------------------------------------- 2 placement
    w("## 2. Divergence-before vs divergence-after")
    w()
    w("Mean detection AUC by 3-letter placement code (layer, denoise, token; "
      "`-` = size-1 scope so the choice does not exist). Restricted to the fully "
      "crossed subgrid (denoise scope = all, token scope != state).")
    w()
    for j, t in enumerate(TS):
        w(f"**t = {t}**")
        w()
        w("| placement | n | A raw | A resid | B raw | B resid |")
        w("|---|---|---|---|---|---|")
        for p in sorted(F.placement[CROSS].unique()):
            m = CROSS & (F.placement == p).values
            w(f"| {p} | {m.sum()} | {Ad['raw_det'][m,j].mean():.4f} | "
              f"{Ad['res_det'][m,j].mean():.4f} | {Bd['raw_det'][m,j].mean():.4f} | "
              f"{Bd['res_det'][m,j].mean():.4f} |")
        w()
    w("Marginal effect of each axis's placement (mean det AUC, crossed subgrid, t=30):")
    w()
    w("| axis | corpus | Before | After | delta (B-A) |")
    w("|---|---|---|---|---|")
    for ax, col in [("layer", "layer_place"), ("denoise", "den_place"), ("token", "tok_place")]:
        for tag, d in [("A", Ad), ("B", Bd)]:
            mb = CROSS & (F[col] == "B").values
            ma = CROSS & (F[col] == "A").values
            w(f"| {ax} | {tag} | {d['raw_det'][mb,2].mean():.4f} | "
              f"{d['raw_det'][ma,2].mean():.4f} | "
              f"{d['raw_det'][mb,2].mean()-d['raw_det'][ma,2].mean():+.4f} |")
    w()

    # ---------------------------------------------------------------- 3 eta2
    w("## 3. Which choice matters most (one-way eta^2 on detection AUC)")
    w()
    w("Fraction of the grid's variance in detection AUC explained by each design "
      "factor alone, on the fully-crossed subgrid.")
    w()
    tab = {}
    for tag, d in [("A", Ad), ("B", Bd)]:
        for nm in ("raw", "res"):
            for j, t in enumerate(TS):
                tab[(tag, nm, t)] = eta2(d[nm + "_det"][:, j], CROSS)
    E = pd.DataFrame(tab)
    w("```")
    w(E.round(3).to_string())
    w("```")
    w()

    # ---------------------------------------------------------------- 4 top cells
    w("## 4. Top cells")
    w()
    for nm, lab in [("raw", "RAW"), ("res", "RESIDUAL")]:
        for tag, d, o in [("A", Ad, Bd), ("B", Bd, Ad)]:
            v = d[nm + "_det"]
            fl = v.ravel()
            idx = np.argsort(fl)[::-1][:10]
            w(f"**{lab} grid, corpus {tag}, top 10** (other corpus shown for the same cell)")
            w()
            w("```")
            for k in idx:
                i, j = np.unravel_index(k, v.shape)
                sgn = "same" if np.sign(d[nm][i, j] - .5) == np.sign(o[nm][i, j] - .5) else "FLIP"
                w(f"  {F.name[i]:52s} t={TS[j]}  det={v[i,j]:.4f} "
                  f"pw_p={d[nm+'_pw'][i,j]:.3f}  other={o[nm+'_det'][i,j]:.4f} {sgn}")
            w("```")
            w()

    # ---------------------------------------------------------------- 5 perm
    w("## 5. Permutation family-wise p (200 draws, labels shuffled within group)")
    w()
    w("Family statistic = max detection AUC over the whole grid and all three t. "
      "One permutation per draw at branch level, reused across t.")
    w()
    w("| corpus | grid | observed max | null mean | null p95 | null max | family p |")
    w("|---|---|---|---|---|---|---|")
    for tag, d in [("A", Ad), ("B", Bd)]:
        for nm in ("raw", "res"):
            obs = d[nm + "_det"].max()
            nl = d[nm + "_null"]
            p = (1 + (nl >= obs).sum()) / (1 + NP)
            w(f"| {tag} | {nm} | {obs:.4f} | {nl.mean():.4f} | {np.percentile(nl,95):.4f} | "
              f"{nl.max():.4f} | {p:.4f} |")
    w()
    # family with the baseline removed from the raw grid
    keep = np.ones(A.NCFG, bool)
    keep[BIDX] = False
    w("Raw grid with the baseline cell removed (the previous round's lesson: a family "
      "that contains the baseline only certifies the baseline):")
    w()
    for tag, d in [("A", Ad), ("B", Bd)]:
        obs = d["raw_det"][keep].max()
        nl = d["raw_null"]
        w(f"* corpus {tag}: observed {obs:.4f}, family p = {(1+(nl>=obs).sum())/(1+NP):.4f} "
          "-- unchanged, because the raw family is saturated with near-copies of the "
          f"baseline (median |within-group rank rho| with the baseline = "
          f"{np.median(np.abs(d['rho'][:,2])):.3f} at t=30).")
    w()

    # ---------------------------------------------------------------- 6 joint
    w("## 6. Cross-corpus agreement")
    w()
    sgnA = np.sign(Ad["res"] - .5)
    sgnB = np.sign(Bd["res"] - .5)
    agree = (sgnA == sgnB)
    w(f"**Sign agreement of the RESIDUALS**: {agree.sum()}/{agree.size} = "
      f"**{agree.mean()*100:.1f}%** of cells (chance = 50%).")
    w()
    w("| slice | cells | sign agreement |")
    w("|---|---|---|")
    for j, t in enumerate(TS):
        w(f"| t={t} | {agree[:,j].size} | {agree[:,j].mean()*100:.1f}% |")
    for f in ["layer_place", "den_place", "tok_place"]:
        for u in ["B", "A"]:
            m = (F[f] == u).values
            w(f"| {f}={u} | {agree[m].size} | {agree[m].mean()*100:.1f}% |")
    for f in ["layer_scope", "tok_scope", "den_scope"]:
        for u in F[f].unique():
            m = (F[f] == u).values
            w(f"| {f}={u} | {agree[m].size} | {agree[m].mean()*100:.1f}% |")
    w()
    sgnAr = np.sign(Ad["raw"] - .5)
    sgnBr = np.sign(Bd["raw"] - .5)
    w(f"(Raw grid, for reference: {(sgnAr==sgnBr).mean()*100:.1f}% sign agreement.)")
    w()

    # joint null: a cell must clear in BOTH corpora
    dA = np.load("/tmp/agg_draws_A.npz")
    dB = np.load("/tmp/agg_draws_B.npz")
    w("**Joint criterion** -- a cell counts only if it clears in both corpora. "
      "Statistic = min(det_A, det_B) with matching direction; null pairs draw i of "
      "corpus A with draw i of corpus B (independent corpora).")
    w()
    w("| grid | observed max min(det_A,det_B) | cell | null p95 | joint family p |")
    w("|---|---|---|---|---|")
    for nm in ("raw", "res"):
        obs = np.where(np.sign(Ad[nm] - .5) == np.sign(Bd[nm] - .5),
                       np.minimum(Ad[nm + "_det"], Bd[nm + "_det"]), 0.0)
        k = np.unravel_index(obs.argmax(), obs.shape)
        nulls = np.minimum(dA[nm], dB[nm]).reshape(NP, -1).max(1)
        p = (1 + (nulls >= obs.max()).sum()) / (1 + NP)
        w(f"| {nm} | {obs.max():.4f} | `{F.name[k[0]]}` t={TS[k[1]]} | "
          f"{np.percentile(nulls,95):.4f} | {p:.4f} |")
    w()
    for nm in ("raw", "res"):
        nulls = np.minimum(dA[nm], dB[nm]).reshape(NP, -1).max(1)
        thr = np.percentile(nulls, 95)
        obs = np.where(np.sign(Ad[nm] - .5) == np.sign(Bd[nm] - .5),
                       np.minimum(Ad[nm + "_det"], Bd[nm + "_det"]), 0.0)
        n = (obs >= thr).sum()
        w(f"* {nm} grid: **{n}** of {obs.size} cells clear the joint FWER-95 threshold "
          f"({thr:.4f}) in both corpora.")
        if n:
            idx = np.argsort(obs.ravel())[::-1][:min(n, 15)]
            w("```")
            for kk in idx:
                i, j = np.unravel_index(kk, obs.shape)
                w(f"  {F.name[i]:52s} t={TS[j]}  A={Ad[nm+'_det'][i,j]:.4f} "
                  f"B={Bd[nm+'_det'][i,j]:.4f}")
            w("```")
    w()

    # ---------------------------------------------------------------- 7 degenerate
    w("## 7. Degenerate / pruned cells")
    w()
    for tag in ("A", "B"):
        z = np.load(f"/tmp/agg_cube_{tag}.npz")
        c = z["cube"]
        const = (c.std(0) < 1e-12).any(1)
        w(f"* corpus {tag}: {int(const.sum())} of {A.NCFG} cells have a *constant* "
          "per-branch score at some t (zero variance -> AUC pinned at 0.5).")
        if const.sum():
            sub = F[const]
            w("```")
            w(sub.groupby(["layer_place", "layer_op", "den_place", "den_op",
                           "tok_place", "tok_op"]).size().to_string())
            w("```")
    w()
    pd.DataFrame(dict(A_raw=Ad["raw_det"][:, 2], A_res=Ad["res_det"][:, 2],
                      B_raw=Bd["raw_det"][:, 2], B_res=Bd["res_det"][:, 2],
                      rho_A=Ad["rho"][:, 2], rho_B=Bd["rho"][:, 2],
                      name=F.name)).to_csv("/tmp/moe_agg_placement_t30.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
