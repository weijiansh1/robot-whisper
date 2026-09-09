"""Stage 2: fit the LGSSM for ONE held-out group and store the causal per-chunk series.

Usage: python ssm_run.py <A|B> <real|shuffle> <group>

The model (PCA basis, F, B, W, Q, R, m0, P0) is estimated from the OTHER groups only,
then the Kalman filter is run forward over the held-out branches. Every stored value at
chunk k depends only on chunks 0..k of that branch and on parameters that branch did not
contribute to.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L

PREP = f"{L.CACHE}/prep"
CONFIGS = [(8, 4), (16, 4), (16, 8), (32, 4), (32, 8), (32, 16),
           (64, 8), (64, 16), (64, 32)]
MAIN = (32, 8)                      # declared a priori, before seeing any AUC
N_ITER, TOL = 40, 1e-6
KEYS = ("innov", "nis", "nll", "zfnorm", "dzf", "pcaz")


def main():
    corpus, variant, gsel = sys.argv[1], sys.argv[2], sys.argv[3]
    torch.set_num_threads(int(os.environ.get("EM_THREADS", "8")))
    os.makedirs(f"{L.CACHE}/em", exist_ok=True)
    outp = f"{L.CACHE}/em/{corpus}_{variant}_f{gsel}.npz"
    if os.path.exists(outp):
        print("skip", outp)
        return
    t00 = time.time()
    C = np.load(f"{PREP}/{corpus}_{variant}_common.npz")
    valid, grp = C["valid"], C["grp"]
    Mt = torch.as_tensor(valid)
    tr = grp.astype(str) != gsel
    ho = ~tr
    X64 = np.load(f"{PREP}/{corpus}_{variant}_f{gsel}_X.npy").astype(np.float64)
    Ut = torch.as_tensor(np.load(f"{PREP}/{corpus}_{variant}_f{gsel}_U.npy").astype(np.float64))
    res, diag, zfm = {}, [], None
    _cf = [MAIN] if os.environ.get("ONLY_MAIN") else CONFIGS
    for (r, d) in _cf:
        Xt = torch.as_tensor(np.ascontiguousarray(X64[:, :, :r]))
        p, lls = L.em_fit(Xt[tr], Ut[tr], Mt[tr], r, d, n_iter=N_ITER, tol=TOL, dev="cpu")
        f = L.kalman_filter(Xt[ho], Ut[ho], Mt[ho], p, want_seq=False)
        zf = f["zf"].numpy()
        Xh = X64[ho][:, :, :r]
        S = {"innov": f["innov"].norm(dim=-1).numpy(),
             "nis": f["nis"].numpy(), "nll": f["nll"].numpy(),
             "zfnorm": np.linalg.norm(zf, axis=-1),
             "dzf": np.zeros(zf.shape[:2]), "pcaz": np.zeros(zf.shape[:2])}
        S["dzf"][:, 1:] = np.linalg.norm(zf[:, 1:] - zf[:, :-1], axis=-1)
        S["pcaz"][:, 1:] = np.linalg.norm(Xh[:, 1:] - Xh[:, :-1], axis=-1)
        for k in KEYS:
            res[f"{k}__{r}_{d}"] = S[k]
        if (r, d) == MAIN:
            zfm = zf.astype(np.float32)
        mono = all(lls[i + 1] >= lls[i] - 1e-5 for i in range(len(lls) - 1))
        ev = np.abs(np.linalg.eigvals(p.F.numpy()))
        diag.append(dict(g=gsel, r=r, d=d, niter=len(lls), ll0=lls[0], ll1=lls[-1],
                         mono=mono, rho_F=float(ev.max()),
                         Bnorm=float(np.linalg.norm(p.Bm.numpy())),
                         Wnorm=float(np.linalg.norm(p.W.numpy())),
                         Qtr=float(np.trace(p.Q.numpy())),
                         Rtr=float(np.trace(p.R.numpy()))))
        print(f"  {corpus}/{variant} g{gsel} r{r} d{d} it{len(lls)} "
              f"ll {lls[0]:.0f}->{lls[-1]:.0f} mono={mono} ({time.time()-t00:.0f}s)", flush=True)
    np.savez_compressed(outp, ho=ho, zf_main=zfm,
                        diag=pd.DataFrame(diag).to_csv(index=False), **res)
    print(f"{corpus}/{variant} g={gsel} DONE {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
