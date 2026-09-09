"""Family-corrected permutation test + cross-corpus sign agreement.

The family is every (latent config, statistic) cell of the SSM read-out. The baseline is
NOT in the family: the increment is measured on the baseline-residualised score, so a
family null cannot re-certify the baseline. p_family uses the max |AUC-0.5| over the
family under within-group branch-level label permutation.

Usage: python ssm_family.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L
from ssm_run import CONFIGS
from ssm_eval import build_stats

TS = (20, 25, 30)
NDRAW = 200
DROP = ("base_w8", "base_cum")


def cells(corpus, variant, t):
    C = np.load(f"{L.CACHE}/prep/{corpus}_{variant}_common.npz")
    base, y, grp = C["base"], C["y"], C["grp"]
    S = np.load(f"{L.CACHE}/series_{corpus}_{variant}.npz")
    names, raw, res = [], [], []
    bs = base[:, t - 7:t + 1].mean(1)
    for cfg in CONFIGS:
        for nm, v in build_stats(S, base, t, cfg).items():
            if nm in DROP:
                continue
            names.append(f"r{cfg[0]}d{cfg[1]}:{nm}")
            raw.append(v)
            res.append(L.rank_residualise(v, bs, grp))
    return names, np.array(raw).T, np.array(res).T, y, grp, bs


def auc_mat(Smat, y, grp):
    """within-group AUC for every column of Smat."""
    out = np.zeros(Smat.shape[1])
    den_tot = 0.0
    num = np.zeros(Smat.shape[1])
    for g in np.unique(grp):
        m = grp == g
        yy = y[m]
        npos, nneg = int(yy.sum()), int((~yy).sum())
        if npos == 0 or nneg == 0:
            continue
        sub = Smat[m]
        rk = np.apply_along_axis(L._avg_rank, 0, sub)
        a = (rk[yy].sum(0) - npos * (npos + 1) / 2.0) / (npos * nneg)
        num += a * npos * nneg
        den_tot += npos * nneg
    out = num / den_tot
    return out


def main():
    rows = []
    store = {}
    for corpus in ("A", "B"):
        for t in TS:
            names, raw, res, y, grp, bs = cells(corpus, "real", t)
            a_raw = auc_mat(raw, y, grp)
            a_res = auc_mat(res, y, grp)
            a_base, _ = L.within_group_auc(bs, y, grp)
            store[(corpus, t)] = dict(names=names, a_res=a_res, a_raw=a_raw)
            rng = np.random.default_rng(L.SEED)
            null_raw, null_res = np.zeros(NDRAW), np.zeros(NDRAW)
            for i in range(NDRAW):
                yp = y.copy()
                for g in np.unique(grp):
                    idx = np.where(grp == g)[0]
                    yp[idx] = y[idx][rng.permutation(len(idx))]
                null_raw[i] = np.abs(auc_mat(raw, yp, grp) - 0.5).max()
                null_res[i] = np.abs(auc_mat(res, yp, grp) - 0.5).max()
            obs_raw = np.abs(a_raw - 0.5).max()
            obs_res = np.abs(a_res - 0.5).max()
            p_raw = (1 + int((null_raw >= obs_raw).sum())) / (NDRAW + 1)
            p_res = (1 + int((null_res >= obs_res).sum())) / (NDRAW + 1)
            best_raw = names[int(np.argmax(np.abs(a_raw - 0.5)))]
            best_res = names[int(np.argmax(np.abs(a_res - 0.5)))]
            rows.append(dict(corpus=corpus, t=t, ncells=len(names),
                             base_det=max(a_base, 1 - a_base),
                             best_raw=best_raw, best_raw_det=0.5 + obs_raw,
                             p_family_raw=p_raw, null95_raw=0.5 + np.quantile(null_raw, .95),
                             best_res=best_res, best_res_det=0.5 + obs_res,
                             p_family_res=p_res, null95_res=0.5 + np.quantile(null_res, .95)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{L.OUT}/family_corrected.csv", index=False)
    df.to_csv("/tmp/family_corrected.csv", index=False)

    # cross-corpus sign agreement of the residualised cells
    L.note("\n### Family-corrected increment over the baseline (residualised, "
           "max over 117 SSM cells, 200-draw within-group label permutation)")
    for _, r in df.iterrows():
        L.note(f"  {r['corpus']} t={int(r['t'])}: baseline det {r['base_det']:.3f} | "
               f"best residual cell {r['best_res']} det {r['best_res_det']:.3f} "
               f"(family null 95% = {r['null95_res']:.3f}) p_family={r['p_family_res']:.3f}")
    L.note("\n### Cross-corpus sign agreement of residualised cells (auc_succ vs 0.5)")
    for t in TS:
        sa = np.sign(store[("A", t)]["a_res"] - 0.5)
        sb = np.sign(store[("B", t)]["a_res"] - 0.5)
        agree = float((sa == sb).mean())
        # restrict to cells that are at least modestly separating in A
        strong = np.abs(store[("A", t)]["a_res"] - 0.5) > 0.05
        agr_s = float((sa[strong] == sb[strong]).mean()) if strong.any() else np.nan
        L.note(f"  t={t}: sign agreement {agree:.3f} over {len(sa)} cells; "
               f"{agr_s:.3f} over the {int(strong.sum())} cells with |AUC-0.5|>0.05 in A")
        L.emit(corpus="AB", t=t, stat="sign_agreement", metric="resid",
               agree=agree, agree_strong=agr_s, ncells=len(sa), nstrong=int(strong.sum()))
    L.flush("moe_ssm_probe.csv")


if __name__ == "__main__":
    main()
