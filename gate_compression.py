"""What does the 1024 -> 32 gate throw away?

The gate is exactly softmax(W h) with rank(W) = 31 (verified R^2 = 0.9997).
So the input splits cleanly and analytically:

    h = h_row  (31 dims,  the ONLY thing the router can respond to)
      + h_null (993 dims, analytically invisible to the router)

For every (layer, token) site we ask the same question of both halves, under
the identical protocol used everywhere else in this project: within-init-state
paired det, family maxT by within-group label permutation.

To keep the comparison apples to apples the null half is reduced to its top 31
PCs, so both halves are 31-dimensional families.
"""
import json

import numpy as np
import pandas as pd
import zarr

ROOT = "/home/jovyan/work/himoe-vla"
B_DIR = (f"{ROOT}/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
         "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
DEEP = slice(4, 8)
LAYERS = [12, 13, 14, 15]
DENOISE = 9
NDIM = 31
NPERM = 1000
T_EVAL = 30
RNG = np.random.default_rng(20260901)


def fit_W(nfit=512):
    """Recover the gate projection per deep layer from (hidden, log-route)."""
    h = zarr.open_group(f"{B_DIR}/server/hidden.zarr", mode="r")["hb_hidden"]
    r = zarr.open_group(f"{B_DIR}/server/routes.zarr", mode="r")["hb_router_probs"]
    H = h[:nfit, DEEP, DENOISE, :, :].astype(np.float32)
    P = r[:nfit, DEEP, DENOISE, :, :].astype(np.float32)
    bases, r2s = [], []
    for i in range(4):
        X = H[:, i].reshape(-1, 1024)
        p = np.clip(P[:, i].reshape(-1, 32), 1e-12, None)
        p /= p.sum(-1, keepdims=True)
        Y = np.log(p)
        Y -= Y.mean(-1, keepdims=True)
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        W, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
        r2s.append(1 - ((Yc - Xc @ W) ** 2).sum() / (Yc ** 2).sum())
        U, s, _ = np.linalg.svd(W, full_matrices=False)   # W: (1024, 32)
        bases.append(U[:, :NDIM])                          # orthonormal, 1024 x 31
    return bases, r2s


def eval_rows(t=T_EVAL):
    summ = json.load(open(f"{B_DIR}/client/summaries.json"))
    ic = np.array([s["inference_calls"] for s in summ])
    off = np.concatenate([[0], np.cumsum(ic)])
    keep = [i for i, s in enumerate(summ) if ic[i] > t]
    rows = np.array([off[i] + t for i in keep])
    g = np.array([summ[i]["init_state_id"] for i in keep])
    fail = np.array([not summ[i]["success"] for i in keep])
    return rows, g, fail


def group_rank(X, g):
    R = np.empty_like(X, dtype=float)
    for k in np.unique(g):
        m = g == k
        R[m] = pd.DataFrame(X[m]).rank().to_numpy()
    return R


def det_family(X, g, fail, nperm=NPERM):
    """det per column + maxT family p, within-group paired."""
    R = group_rank(X, g)
    idx = [(np.where(g == k)[0], int(fail[g == k].sum())) for k in np.unique(g)]
    idx = [(ix, nf) for ix, nf in idx if nf and len(ix) - nf]
    W_ = sum(nf * (len(ix) - nf) for ix, nf in idx)
    C = sum(nf * (nf + 1) / 2.0 for _, nf in idx)
    obs = sum(R[ix[fail[ix]]].sum(0) for ix, _ in idx)
    auc = (obs - C) / W_
    det = np.maximum(auc, 1 - auc)
    maxT = np.empty(nperm)
    for p in range(nperm):
        tot = np.zeros(X.shape[1])
        for ix, nf in idx:
            tot += R[ix[RNG.choice(len(ix), nf, replace=False)]].sum(0)
        a = (tot - C) / W_
        maxT[p] = np.maximum(a, 1 - a).max()
    return det, np.array([(maxT >= d).mean() for d in det]), float(np.median(maxT))


if __name__ == "__main__":
    bases, r2s = fit_W()
    print("gate recovery R^2 per deep layer:", [f"{v:.5f}" for v in r2s])

    rows, g, fail = eval_rows()
    print(f"eval: {len(rows)} episodes at t={T_EVAL}, "
          f"{len(np.unique(g))} init states, {fail.sum()} fail\n")

    h = zarr.open_group(f"{B_DIR}/server/hidden.zarr", mode="r")["hb_hidden"]
    order = np.argsort(rows)
    H = np.empty((len(rows), 4, 11, 1024), np.float32)
    for j, oi in enumerate(order):
        H[oi] = h[int(rows[oi]), DEEP, DENOISE, :, :]
        if j % 100 == 0:
            print(f"  read {j}/{len(rows)}", flush=True)

    out = []
    for i, L in enumerate(LAYERS):
        U = bases[i]
        for tok in range(11):
            X = H[:, i, tok, :]
            Xc = X - X.mean(0)
            row = Xc @ U                          # 31 dims the router sees
            null = Xc - row @ U.T                 # 993 dims it discards
            v_row = (row ** 2).sum()
            v_null = (null ** 2).sum()
            Un, sn, _ = np.linalg.svd(null, full_matrices=False)
            nullpc = Un[:, :NDIM] * sn[:NDIM]     # top 31 PCs, matched family size

            d_r, p_r, nm = det_family(row, g, fail)
            d_n, p_n, _ = det_family(nullpc, g, fail)
            out.append(dict(layer=L, token=tok,
                            var_frac_row=v_row / (v_row + v_null),
                            det_row_max=d_r.max(), p_row=p_r.min(),
                            n_row_sig=int((p_r < 0.05).sum()),
                            det_null_max=d_n.max(), p_null=p_n.min(),
                            n_null_sig=int((p_n < 0.05).sum()),
                            null_med=nm))
        print(f"  layer {L} done", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(f"{ROOT}/gate_compression.csv", index=False)
    print("\nwrote gate_compression.csv")
