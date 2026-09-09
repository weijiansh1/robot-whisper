"""Stage 2: EM-fit the LGSSM on one fold's training groups, filter the held-out group.

Usage: python ssm_em.py <A|B> <real|shuffle> <fold_group> <r> <d> [n_iter]

The model never sees the held-out group. The Kalman filter is forward-only, so every
series value at chunk k uses chunks 0..k of that branch alone.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L

PREP = f"{L.CACHE}/prep"
EMD = f"{L.CACHE}/em"


def main():
    corpus, variant, g, r, d = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    nit = int(sys.argv[6]) if len(sys.argv) > 6 else 25
    nth = int(os.environ.get("EM_THREADS", "6"))
    torch.set_num_threads(nth)
    os.makedirs(EMD, exist_ok=True)
    out = f"{EMD}/{corpus}_{variant}_f{g}_r{r}_d{d}.npz"
    if os.path.exists(out):
        print("skip", out)
        return
    t0 = time.time()
    C = np.load(f"{PREP}/{corpus}_{variant}_common.npz")
    valid, grp = C["valid"], C["grp"]
    X = np.load(f"{PREP}/{corpus}_{variant}_f{g}_X.npy")[:, :, :r].astype(np.float64)
    U = np.load(f"{PREP}/{corpus}_{variant}_f{g}_U.npy").astype(np.float64)
    tr = grp.astype(str) != g
    ho = ~tr
    Xt = torch.as_tensor(X)
    Ut = torch.as_tensor(U)
    Mt = torch.as_tensor(valid)
    p, lls = L.em_fit(Xt[tr], Ut[tr], Mt[tr], r, d, n_iter=nit, dev="cpu", batch=1024)
    f = L.kalman_filter(Xt[ho], Ut[ho], Mt[ho], p, want_seq=False)
    zf = f["zf"].numpy()
    Xh = X[ho]
    pz = np.zeros(zf.shape[:2])
    pz[:, 1:] = np.linalg.norm(Xh[:, 1:] - Xh[:, :-1], axis=-1)
    dzf = np.zeros(zf.shape[:2])
    dzf[:, 1:] = np.linalg.norm(zf[:, 1:] - zf[:, :-1], axis=-1)
    np.savez_compressed(out,
                        ho=ho,
                        innov=f["innov"].norm(dim=-1).numpy(),
                        nis=f["nis"].numpy(),
                        nll=f["nll"].numpy(),
                        zfnorm=np.linalg.norm(zf, axis=-1),
                        dzf=dzf, pcaz=pz, zf=zf.astype(np.float32),
                        ll=np.array(lls),
                        Fnorm=np.array([float(torch.linalg.matrix_norm(p.F, 2)),
                                        float(torch.linalg.matrix_norm(p.Bm, 2))]))
    mono = all(lls[i + 1] >= lls[i] - 1e-5 for i in range(len(lls) - 1))
    print(f"{corpus}/{variant} f{g} r{r} d{d}: {time.time()-t0:.0f}s ll {lls[0]:.0f}->{lls[-1]:.0f} "
          f"mono={mono} nho={int(ho.sum())}", flush=True)


if __name__ == "__main__":
    main()
