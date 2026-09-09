"""Stage 3: score, evaluate and persist. Raw + baseline-residualised within-group AUC,
within-group permutation nulls, redundancy check.

Usage: python ssm_eval.py <A|B> <real|shuffle>
"""
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L
from ssm_run import CONFIGS, MAIN

TS = (20, 25, 30)
NPERM = 200


def build_stats(S, base, t, cfg):
    r, d = cfg
    out = {}
    for W in (1, 3, 8, 12):
        sl = slice(t - W + 1, t + 1)
        out[f"innov_w{W}"] = S[f"innov__{r}_{d}"][:, sl].mean(1)
        out[f"nis_w{W}"] = S[f"nis__{r}_{d}"][:, sl].mean(1)
    for W in (3, 8, 12):
        out[f"dzf_w{W}"] = S[f"dzf__{r}_{d}"][:, t - W + 1:t + 1].mean(1)
    out["nll_w8"] = S[f"nll__{r}_{d}"][:, t - 7:t + 1].mean(1)
    out["zfnorm_w8"] = S[f"zfnorm__{r}_{d}"][:, t - 7:t + 1].mean(1)
    out["pcaz_w8"] = S[f"pcaz__{r}_{d}"][:, t - 7:t + 1].mean(1)
    out["innov_cum"] = S[f"innov__{r}_{d}"][:, 1:t + 1].mean(1)
    out["nis_cum"] = S[f"nis__{r}_{d}"][:, 1:t + 1].mean(1)
    return out


def main():
    corpus, variant = sys.argv[1], sys.argv[2]
    C = np.load(f"{L.CACHE}/prep/{corpus}_{variant}_common.npz")
    base, y, grp = C["base"], C["y"], C["grp"]
    S = np.load(f"{L.CACHE}/series_{corpus}_{variant}.npz")
    for t in TS:
        bs = base[:, t - 7:t + 1].mean(1)
        bcum = base[:, 1:t + 1].mean(1)
        for nm, v in (("base_w8", bs), ("base_cum", bcum)):
            a, n = L.within_group_auc(v, y, grp)
            L.emit(corpus=corpus, variant=variant, r=0, d=0, t=t, stat=nm, metric="raw",
                   auc_succ=a, det=max(a, 1 - a), npairs=n,
                   p_perm=L.perm_p(v, y, grp, max(a, 1 - a), NPERM))
        for cfg in CONFIGS:
            r, d = cfg
            for nm, v in build_stats(S, base, t, cfg).items():
                a, n = L.within_group_auc(v, y, grp)
                det = max(a, 1 - a)
                rho, rhov = L.within_group_spearman(v, bs, grp)
                rv = L.rank_residualise(v, bs, grp)
                ar, nr = L.within_group_auc(rv, y, grp)
                detr = max(ar, 1 - ar)
                main_stat = nm in ("nis_w8", "nis_w12", "innov_w8", "dzf_w8")
                nd = NPERM if (cfg == MAIN or main_stat) else 0
                L.emit(corpus=corpus, variant=variant, r=r, d=d, t=t, stat=nm,
                       metric="raw", auc_succ=a, det=det, npairs=n, rho_base=rho,
                       rho_min=float(np.nanmin(rhov)), rho_max=float(np.nanmax(rhov)),
                       **({"p_perm": L.perm_p(v, y, grp, det, nd)} if nd else {}))
                L.emit(corpus=corpus, variant=variant, r=r, d=d, t=t, stat=nm,
                       metric="resid", auc_succ=ar, det=detr, npairs=nr,
                       **({"p_perm": L.perm_p(rv, y, grp, detr, nd)} if nd else {}))
        L.flush()
        print(f"eval {corpus}/{variant} t={t} done", flush=True)


if __name__ == "__main__":
    main()
