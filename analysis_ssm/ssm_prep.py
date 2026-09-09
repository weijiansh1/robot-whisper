"""Stage 1: clr emission, protocol baseline, per-fold (leave-one-group-out) PCA basis.

Usage: python ssm_prep.py <A|B> <real|shuffle>

Writes to cache/prep/:  {corpus}_{variant}_f{g}_X.npy   (Bn, T, RMAX) whitened PCA scores
                        {corpus}_{variant}_f{g}_U.npy   (Bn, T, 70) standardised actions
                        {corpus}_{variant}_common.npz   base, anorm, valid, y, grp
The PCA mean/basis/scale and the action standardisation use ONLY the training groups.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L

RMAX = 64
BACK40 = [i * 11 + t for i in (4, 5, 6, 7) for t in range(1, 11)]
PREP = f"{L.CACHE}/prep"


def main():
    corpus, variant = sys.argv[1], sys.argv[2]
    torch.set_num_threads(32)
    os.makedirs(PREP, exist_ok=True)
    P, Aact, valid, meta = L.load(corpus, "all88")
    if variant == "shuffle":
        P, Aact = L.shuffle_within_branch(P, Aact, valid)
    y = meta["success"].values.astype(bool)
    grp = meta["group"].values
    Bn, Tmax = valid.shape

    base = np.nan_to_num(L.hellinger_prev(P[:, :, BACK40, :], valid), nan=0.0)
    for t in (20, 25, 30):
        a, n = L.within_group_auc(base[:, t - 7:t + 1].mean(1), y, grp)
        L.note(f"[{corpus}/{variant}] BASELINE t={t}: auc_succ={a:.4f} "
               f"det={max(a,1-a):.4f} npairs={n}")

    Y = L.clr(P, valid)
    del P
    U = Aact.reshape(Bn, Tmax, 70).astype(np.float64)
    anorm = np.linalg.norm(Aact.reshape(Bn, Tmax, 70), axis=-1)
    del Aact
    np.savez(f"{PREP}/{corpus}_{variant}_common.npz", base=base, anorm=anorm,
             valid=valid, y=y, grp=grp)

    for g in np.unique(grp):
        tr = grp != g
        trm = valid & tr[:, None]
        mu, V, sd = L.fit_pca(Y[trm], RMAX)
        X = L.apply_pca(Y, mu, V, sd).astype(np.float32)
        um, us = U[trm].mean(0), U[trm].std(0) + 1e-8
        np.save(f"{PREP}/{corpus}_{variant}_f{g}_X.npy", X)
        np.save(f"{PREP}/{corpus}_{variant}_f{g}_U.npy", ((U - um) / us).astype(np.float32))
        print(f"prep {corpus}/{variant} fold {g} ntr={int(tr.sum())} X{X.shape}", flush=True)


if __name__ == "__main__":
    main()
