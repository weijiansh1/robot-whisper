#!/usr/bin/env python3
"""
Top-4 survival vs the RELATIVE logit perturbation  rho = ||dz|| / ||zhat||,
restricted to tokens whose top-4 boundary is strictly separated in the stored fp16.

30.1% of tokens have an exact s_(4) == s_(5) tie at fp16 storage precision (median
gap 2.75e-4 ~ 18 fp16 ulp, 10th pct exactly 0).  For those the baseline argsort is
arbitrary, so ANY perturbation flips them and "survival" reports a storage artifact
rather than a property of the gradient.  Conditioning on gap > 0 removes it.
rho is optimizer-agnostic: it does not assume an eta, a learning rate, or Adam.
"""
import json
from pathlib import Path

import numpy as np
import zarr

N, K, U, ND, NL = 32, 4, 11, 10, 8
HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
HB = [2, 3, 4, 5, 12, 13, 14, 15]
RHO = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

for run in sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json")):
    meta = json.loads((run / "meta.json").read_text())
    z = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    T = z["hb_router_probs"].shape[0]
    surv = np.zeros((NL, len(RHO))); n = np.zeros(NL); tie = np.zeros(NL); tn = np.zeros(NL)
    for b0 in range(0, T, 64 * 40):
        b1 = min(b0 + 64, T)
        s = z["hb_router_probs"][b0:b1].astype(np.float64)
        s /= s.sum(-1, keepdims=True)
        B = s.shape[0]; M = B * NL * ND
        s = s.reshape(M, U, N)
        order = np.argsort(-s, axis=2)
        idf = np.sort(order[:, :, :K], axis=2)
        srt = np.take_along_axis(s, order, axis=2)
        strict = srt[:, :, K - 1] > srt[:, :, K]              # (M,U) usable tokens

        c = np.zeros((M, N))
        np.add.at(c, (np.arange(M)[:, None], idf.reshape(M, U * K)), 1.0)
        f = (N / (K * U)) * c
        g = s * (f[:, None, :] - np.einsum("mun,mn->mu", s, f)[:, :, None]) / U
        lg = np.log(np.clip(s, 1e-30, None)); zh = lg - lg.mean(-1, keepdims=True)
        ng = np.linalg.norm(g.reshape(M, -1), axis=1)
        nz = np.linalg.norm(zh.reshape(M, -1), axis=1)
        lay = np.tile(np.repeat(np.arange(NL), ND), B)

        for ri, rho in enumerate(RHO):
            z2 = zh - (rho * nz / (ng + 1e-300))[:, None, None] * g
            new = np.sort(np.argsort(-z2, axis=2)[:, :, :K], axis=2)
            ok = (new == idf).all(-1) & strict
            surv[:, ri] += np.bincount(lay, weights=ok.sum(1), minlength=NL)
        n += np.bincount(lay, weights=strict.sum(1), minlength=NL)
        tie += np.bincount(lay, weights=(~strict).sum(1), minlength=NL)
        tn += np.bincount(lay, minlength=NL) * U

    print(f"\n{meta['benchmark']}/t{meta['task_id']}  {meta['task_name'][:44]}")
    print("  fp16 rank-4/5 exact-tie rate per layer: " +
          " ".join(f"L{HB[l]}={100*tie[l]/tn[l]:.0f}%" for l in range(NL)))
    print(f"  {'rho':>8} " + " ".join(f"L{l:>5}" for l in HB) +
          "    <- top-4 survival | strict boundary")
    for ri, rho in enumerate(RHO):
        print(f"  {rho:8.4f} " + " ".join(f"{surv[l,ri]/n[l]:6.4f}" for l in range(NL)))
print("\nrho is the relative size of the logit update; the real per-step value depends")
print("on lambda_HB, the flow gradient and the Adam preconditioner, so it must come")
print("from the backward attribution, not from this table.")
