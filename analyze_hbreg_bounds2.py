#!/usr/bin/env python3
"""
(1) Verify the regular-dispatch construction and the closed form
        L_HB -> N K / ( m (N - m + K) )
    for a pattern in which m experts are used, each by exactly KU/m tokens, and
    the remaining N-m experts never enter any top-K.  Strictly top-K consistent.

(2) Reparametrize the top-4 survival probe by the RELATIVE logit perturbation
        rho = ||dz|| / ||zhat||
    instead of an arbitrary eta, so it is optimizer-agnostic.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import zarr

N, K, U = 32, 4, 11
HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
HB = [2, 3, 4, 5, 12, 13, 14, 15]
ND, NL = 10, 8


def L_of(s, strict=True):
    idx = np.argsort(-s, axis=1)[:, :K]
    srt = np.sort(s, axis=1)[:, ::-1]
    if strict:
        assert (srt[:, K - 1] > srt[:, K]).all(), "top-K not strict"
        assert (s > 0).all() and np.allclose(s.sum(1), 1), "invalid softmax"
    c = np.bincount(idx.ravel(), minlength=N).astype(float)
    return (N / (K * U ** 2)) * (c * s.sum(0)).sum(), c


def regular_pattern(m):
    """m used experts, each with count KU/m; token u takes a contiguous block."""
    if (K * U) % m or m < K or m > N:
        return None
    cnt = (K * U) // m
    sel = [[(u * K + k) % m for k in range(K)] for u in range(U)]   # cyclic blocks
    c = np.bincount(np.array(sel).ravel(), minlength=N)
    if not (c[:m] == cnt).all() or c[m:].any():
        return None
    return sel


print("=" * 100)
print("(1) regular dispatch over m used experts;  closed form N K / (m (N-m+K))")
print("=" * 100)
print(f"  {'m':>4} {'count':>6} {'q=1/(N-m+K)':>13} {'closed form':>13} "
      f"{'numeric':>11} {'diff':>10} {'dead experts':>13}")
for m in range(K, N + 1):
    sel = regular_pattern(m)
    if sel is None:
        continue
    closed = Fraction(N * K, m * (N - m + K))
    best = None
    for d in (1e-4, 1e-6, 1e-8):
        eps = d ** 2
        q = (1 - K * d - (m - K) * eps) / (N - m + K)
        s = np.empty((U, N))
        for u in range(U):
            s[u, :m] = eps                       # used-but-not-mine
            s[u, m:] = q                         # never used by anyone
            s[u, sel[u]] = q + d                 # my top-K
        s /= s.sum(1, keepdims=True)
        best, c = L_of(s)
    print(f"  {m:>4} {(K*U)//m:>6} {1/(N-m+K):13.6f} "
          f"{float(closed):13.6f} {best:11.6f} {abs(best-float(closed)):10.2e} "
          f"{N-m:>13}")

print(f"\n  m must divide KU={K*U} and satisfy K <= m <= N, so m in "
      f"{[m for m in range(K, N+1) if (K*U) % m == 0]}")
print(f"  m(N-m+K) is maximised at m=(N+K)/2={(N+K)/2:.0f}, which is not a divisor;")
print(f"  among the divisors the best is m=22 -> L -> 32/77 = "
      f"{float(Fraction(32,77)):.6f}")
print(f"  m=4 recovers total collapse onto one shared top-4 -> L -> "
      f"{float(Fraction(N*K, 4*(N-4+K))):.4f}")
print("\n  These are WITNESSES, not an infimum: no claim that any of them is minimal.")

print("\n" + "=" * 100)
print("(2) top-4 survival vs RELATIVE logit perturbation rho = ||dz|| / ||zhat||")
print("=" * 100)
tasks = sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json"))
RHO = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
for run in tasks[:3]:
    meta = json.loads((run / "meta.json").read_text())
    z = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    T = z["hb_router_probs"].shape[0]
    surv = np.zeros((NL, len(RHO))); dm4 = np.zeros((NL, len(RHO))); n = np.zeros(NL)
    disagree_acc = []
    for b0 in range(0, T, 64 * 40):
        b1 = min(b0 + 64, T)
        p = z["hb_router_probs"][b0:b1].astype(np.float64)
        ids = z["hb_expert_ids"][b0:b1].astype(np.int64)
        p /= p.sum(-1, keepdims=True)
        B = p.shape[0]; M = B * NL * ND
        s = p.reshape(M, U, N)
        rec = ids.reshape(M, U, K)
        # self-consistent baseline: the top-K of the STORED probs.  Comparing a
        # perturbation against the recorded bf16 ids instead would inherit the
        # fp16-recompute disagreement at the known 4th/5th near-ties, which puts a
        # spurious 15-35% floor on "survival" even as rho -> 0.
        idf = np.sort(np.argsort(-s, axis=2)[:, :, :K], axis=2)
        agree = (idf == np.sort(rec, axis=2)).all(-1).mean()
        disagree_acc.append(1 - agree)
        c = np.zeros((M, N)); np.add.at(c, (np.arange(M)[:, None],
                                            idf.reshape(M, U * K)), 1.0)
        f = (N / (K * U)) * c
        g = s * (f[:, None, :] - np.einsum("mun,mn->mu", s, f)[:, :, None]) / U
        lg = np.log(np.clip(s, 1e-30, None)); zh = lg - lg.mean(-1, keepdims=True)
        ng = np.linalg.norm(g.reshape(M, -1), axis=1)
        nz = np.linalg.norm(zh.reshape(M, -1), axis=1)
        lay = np.tile(np.repeat(np.arange(NL), ND), B)
        for ri, rho in enumerate(RHO):
            eta = rho * nz / (ng + 1e-300)
            z2 = zh - eta[:, None, None] * g
            new = np.sort(np.argsort(-z2, axis=2)[:, :, :K], axis=2)
            sv = (new == idf).all(-1).mean(-1)
            s2 = np.exp(z2 - z2.max(-1, keepdims=True))
            s2 /= s2.sum(-1, keepdims=True)
            d = (np.take_along_axis(s2, idf, 2).sum(-1)
                 - np.take_along_axis(s, idf, 2).sum(-1)).mean(-1)
            surv[:, ri] += np.bincount(lay, weights=sv, minlength=NL)
            dm4[:, ri] += np.bincount(lay, weights=d, minlength=NL)
        n += np.bincount(lay, minlength=NL)
    print(f"\n  {meta['benchmark']}/t{meta['task_id']}"
          f"   (fp16-recompute vs recorded bf16 top-4 disagreement: "
          f"{100*np.mean(disagree_acc):.1f}% of tokens -- the near-tie rate, "
          f"excluded from the sweep below by using a self-consistent baseline)")
    print(f"    {'rho':>8} " + " ".join(f"L{l:>5}" for l in HB) + "   <- top-4 survival")
    for ri, rho in enumerate(RHO):
        print(f"    {rho:8.4f} " + " ".join(f"{surv[l,ri]/n[l]:6.3f}" for l in range(NL)))
    print(f"    {'rho':>8} " + " ".join(f"L{l:>5}" for l in HB) + "   <- dM4 (sign)")
    for ri in (0, 3, 6):
        print(f"    {RHO[ri]:8.4f} " +
              " ".join(f"{dm4[l,ri]/n[l]:+6.0e}"[:6] for l in range(NL)))
