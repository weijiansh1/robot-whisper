#!/usr/bin/env python3
"""
HB-gradient directional audit, logit space, from the corpus alone (no backward pass).

The exact per-token gradient is
    dL/dz_{j,u} = (s_{j,u}/U) (f_j - ell_u),      ell_u = sum_i f_i s_{i,u}
and centered logits are exactly recoverable from the stored softmax:
    zhat = log s - mean_j(log s)      (log s = z - logsumexp(z), a per-token shift)

The pure-temperature (flatten) direction in logit space is -zhat.  A descent step
-eta*g flattens iff <g, zhat> > 0.  So this converts "a rank-preserving descent path
exists" into a measurement of how much of the ACTUAL gradient lies along it.

Reported per HB layer:
  cos(g, zhat)          alignment of the gradient with the flatten direction
  frac_flat             ||proj of g onto zhat||^2 / ||g||^2
  P(flatten)            fraction of sites where the step reduces logit spread
  dStd/deta, dM4/deta   directional derivatives along the descent step
  top4 survival         does an actual finite step change the dispatch?
Also splits the gradient by hard load c_j, where the sign flip lives.
"""
import json
from pathlib import Path

import numpy as np
import zarr

HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
N, K, U, ND, NL = 32, 4, 11, 10, 8
HB = [2, 3, 4, 5, 12, 13, 14, 15]
BLOCK, STRIDE = 64, 24
ETA = 1.0          # finite step size for the top-4 survival probe


def main():
    tasks = sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json"))
    print("descent step is -eta*g;  <g, zhat> > 0  =>  the step FLATTENS\n")
    grand = {}
    for run in tasks:
        meta = json.loads((run / "meta.json").read_text())
        name = f"{meta['benchmark']}/t{meta['task_id']}"
        z = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
        T = z["hb_router_probs"].shape[0]

        acc = {k: np.zeros(NL) for k in
               ("cos", "fracflat", "pflat", "dstd", "dm4", "surv", "n")}
        gby = np.zeros((NL, 8)); gcnt = np.zeros((NL, 8))
        for b0 in range(0, T, BLOCK * STRIDE):
            b1 = min(b0 + BLOCK, T)
            p = z["hb_router_probs"][b0:b1].astype(np.float64)
            ids = z["hb_expert_ids"][b0:b1].astype(np.int64)
            p /= p.sum(-1, keepdims=True)
            B = p.shape[0]
            M = B * NL * ND
            s = p.reshape(M, U, N)
            idf = ids.reshape(M, U, K)

            c = np.zeros((M, N))
            np.add.at(c, (np.arange(M)[:, None], idf.reshape(M, U * K)), 1.0)
            f = (N / (K * U)) * c                              # (M,32)
            ell = np.einsum("mun,mn->mu", s, f)                # (M,U)
            g = s * (f[:, None, :] - ell[:, :, None]) / U      # (M,U,32)

            lg = np.log(np.clip(s, 1e-30, None))
            zh = lg - lg.mean(-1, keepdims=True)               # centered logits

            gf = g.reshape(M, -1); zf = zh.reshape(M, -1)
            dot = (gf * zf).sum(1)
            ng = np.linalg.norm(gf, axis=1); nz = np.linalg.norm(zf, axis=1)
            cos = dot / (ng * nz + 1e-300)
            fracflat = (dot / (nz + 1e-300)) ** 2 / (ng ** 2 + 1e-300)

            # d/deta of Std_j(z) along -g   (sign only needs -<g,zhat>)
            dstd = -dot / (N * U)
            # d/deta of top-K mass along -g
            ds = -s * (g - (s * g).sum(-1, keepdims=True))     # softmax Jacobian
            dm4 = np.take_along_axis(ds, idf, axis=2).sum(-1).mean(-1)

            # finite step: does the dispatch change at all?
            z2 = zh - ETA * g
            new = np.argsort(-z2, axis=2)[:, :, :K]
            surv = (np.sort(new, axis=2) == np.sort(idf, axis=2)).all(-1).mean(-1)

            lay = np.tile(np.repeat(np.arange(NL), ND), B)
            def pl(v): return np.bincount(lay, weights=v, minlength=NL)
            acc["cos"] += pl(cos); acc["fracflat"] += pl(fracflat)
            acc["pflat"] += pl((dot > 0).astype(float))
            acc["dstd"] += pl(dstd); acc["dm4"] += pl(dm4)
            acc["surv"] += pl(surv); acc["n"] += np.bincount(lay, minlength=NL)

            # gradient sign by hard load bucket
            cb = np.clip(c, 0, 7).astype(int)
            for l in range(NL):
                sel = lay == l
                gm = g[sel].sum(1)                              # sum over tokens
                for k in range(8):
                    mk = cb[sel] == k
                    if mk.any():
                        gby[l, k] += gm[mk].sum(); gcnt[l, k] += mk.sum()

        n = acc["n"]
        print("=" * 100)
        print(f"{name}  {meta['task_name'][:52]}")
        print(f"  {'layer':>6} {'cos(g,zhat)':>12} {'frac_flat':>10} "
              f"{'P(flatten)':>11} {'dStd/deta':>11} {'dM4/deta':>11} "
              f"{'top4 survives':>14}")
        for l in range(NL):
            print(f"  L{HB[l]:>5} {acc['cos'][l]/n[l]:+12.4f} "
                  f"{acc['fracflat'][l]/n[l]:10.4f} "
                  f"{acc['pflat'][l]/n[l]:11.4f} {acc['dstd'][l]/n[l]:+11.3e} "
                  f"{acc['dm4'][l]/n[l]:+11.3e} {acc['surv'][l]/n[l]:14.4f}")
        gm = (gby / np.maximum(gcnt, 1)).mean(0)
        print(f"\n  mean dL/dz summed over tokens, by hard load c_j "
              f"(descent RAISES the logit where this is negative):")
        print("    c_j     " + " ".join(f"{k:>10}" for k in range(6)))
        print("    grad    " + " ".join(f"{gm[k]:+10.3e}" for k in range(6)))
        print(f"    f_j     " + " ".join(f"{(N/(K*U))*k:>10.3f}" for k in range(6)))
        print(f"    -> sign flips between c_j=1 (f={N/(K*U):.3f}) and c_j=2 "
              f"(f={2*N/(K*U):.3f}); the loss wants c_j = {K*U/N} for every expert")
        grand[name] = (acc["cos"] / n).mean()
        print()

    print("=" * 100)
    print("task-mean cos(g, zhat):  " +
          "  ".join(f"{k.split('/')[-1]}={v:+.4f}" for k, v in grand.items()))


if __name__ == "__main__":
    main()
