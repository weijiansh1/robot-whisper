#!/usr/bin/env python3
"""
HB-Reg exact decomposition + blindness test.

Two identities (exact, no approximation):

 (I1)  L_HB - 1 = (N^2 / (K U^2)) * Cov_i(c_i, S_i)
       where c_i = # tokens selecting expert i (sum = K U),
             S_i = sum_u s_{i,u}              (sum = U).
       => the loss is a functional of the COVARIANCE between hard load and soft
          mass ONLY.  It does not see Var(c) -- the actual imbalance.  Any
          hard-load distribution, including total collapse, gives L_HB = 1
          whenever Cov = 0.

 (I2)  L_HB - 1 = (N/(K U)) * (M_K_bar - K/N)                    [SELF term]
                + (N/(K U^2)) * sum_{u != v} (A(u,v) - K/N)      [CROSS term]
       where M_K_bar = mean per-token top-K mass, A(u,v) = mass token v puts on
       token u's selected set.
       The SELF term is pure per-token sharpness: it is nonzero even if every
       token routes to a different expert set, and carries ZERO information about
       whether experts are shared/overused.  Its weight in the pair budget is
       1/U, so it is 9% of all pairs at U=11 and 0.9% at U=110.

 Blindness test: hold the observed scores fixed, swap in counterfactual hard
 loads, and see whether the loss can tell them apart.
"""
import json
from pathlib import Path

import numpy as np
import zarr

HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
N, K, U = 32, 4, 11
NDENOISE, NLAYER = 10, 8
HB = [2, 3, 4, 5, 12, 13, 14, 15]
BLOCK = 128
STRIDE = 6          # subsample control steps; every stat here is a mean


def min_sumsq(slots, n):
    q, r = divmod(slots, n)
    return r * (q + 1) ** 2 + (n - r) * q ** 2


def var_min(slots, n):
    c = np.full(n, slots // n, float)
    c[:slots % n] += 1
    return c.var()


def main():
    tasks = sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json"))
    print("N=%d K=%d U=%d  |  K*U=%d slots, target load %.3f/expert (infeasible)"
          % (N, K, U, K * U, K * U / N))
    print("Var_min(c) at U=11 = %.4f   (perfectly balanced = 12 experts x2, 20 x1)"
          % var_min(K * U, N))
    print("L_floor for a maximally sharp router = %.5f\n"
          % (N * min_sumsq(K * U, N) / (K ** 2 * U ** 2)))

    grand = []
    for run in tasks:
        meta = json.loads((run / "meta.json").read_text())
        name = f"{meta['benchmark']}/t{meta['task_id']}"
        z = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
        T = z["hb_router_probs"].shape[0]

        acc = {k: np.zeros(NLAYER) for k in
               ("L", "self", "cross", "cov", "varc", "vars", "m4",
                "Lbal", "Lcol_lo", "Lcol_hi", "Lperm", "n")}
        for b0 in range(0, T, BLOCK * STRIDE):
            b1 = min(b0 + BLOCK, T)
            p = z["hb_router_probs"][b0:b1].astype(np.float32)
            ids = z["hb_expert_ids"][b0:b1].astype(np.int64)
            B = p.shape[0]
            p /= p.sum(-1, keepdims=True)
            M = B * NLAYER * NDENOISE
            pf = p.reshape(M, U, N)
            idf = ids.reshape(M, U, K)

            c = np.zeros((M, N), np.float32)
            np.add.at(c, (np.arange(M)[:, None], idf.reshape(M, U * K)), 1.0)
            S = pf.sum(1)                                    # (M,32), sums to U
            L = (N / (K * U ** 2)) * (c * S).sum(1)

            cov = ((c - c.mean(1, keepdims=True)) *
                   (S - S.mean(1, keepdims=True))).mean(1)
            # (I2) self / cross split
            m4 = np.take_along_axis(pf, idf, axis=2).sum(-1)   # (M,U)
            self_t = (N / (K * U)) * (m4.mean(1) - K / N)
            cross_t = (L - 1) - self_t

            # ---- blindness: same scores S, counterfactual hard loads
            order = np.argsort(S, axis=1)                     # ascending soft mass
            bal = np.ones((M, N), np.float32)
            rng = np.random.default_rng(0)
            extra = rng.permuted(np.tile(np.arange(N), (M, 1)), axis=1)[:, :12]
            np.add.at(bal, (np.arange(M)[:, None], extra), 1.0)   # 12x2 + 20x1
            col_lo = np.zeros((M, N), np.float32)
            np.put_along_axis(col_lo, order[:, :K], float(U), axis=1)
            col_hi = np.zeros((M, N), np.float32)
            np.put_along_axis(col_hi, order[:, -K:], float(U), axis=1)
            perm = rng.permuted(c, axis=1)                     # same c multiset

            def LL(cc):
                return (N / (K * U ** 2)) * (cc * S).sum(1)

            lay = np.tile(np.repeat(np.arange(NLAYER), NDENOISE), B)

            def pl(v):
                return np.bincount(lay, weights=v, minlength=NLAYER)

            acc["L"] += pl(L); acc["self"] += pl(self_t); acc["cross"] += pl(cross_t)
            acc["cov"] += pl(cov); acc["varc"] += pl(c.var(1)); acc["vars"] += pl(S.var(1))
            acc["m4"] += pl(m4.mean(1))
            acc["Lbal"] += pl(LL(bal)); acc["Lcol_lo"] += pl(LL(col_lo))
            acc["Lcol_hi"] += pl(LL(col_hi)); acc["Lperm"] += pl(LL(perm))
            acc["n"] += np.bincount(lay, minlength=NLAYER)

        n = acc["n"]
        print("=" * 104)
        print(f"{name}  {meta['task_name'][:56]}")
        print(f"  {'layer':>5} {'L_HB':>8} {'checkI1':>9} {'SELF':>9} {'CROSS':>9} "
              f"{'self%':>6} {'Var(c)':>7} {'/Varmin':>8} {'Cov(c,S)':>9} {'M4':>7}")
        for l in range(NLAYER):
            L = acc["L"][l] / n[l]
            cov = acc["cov"][l] / n[l]
            chk = 1 + N ** 2 * cov / (K * U ** 2)
            se, cr = acc["self"][l] / n[l], acc["cross"][l] / n[l]
            vc = acc["varc"][l] / n[l]
            print(f"  L{HB[l]:>4} {L:8.5f} {chk:9.5f} {se:+9.5f} {cr:+9.5f} "
                  f"{100*se/(L-1):6.1f} {vc:7.3f} {vc/var_min(K*U,N):8.2f} "
                  f"{cov:9.5f} {acc['m4'][l]/n[l]:7.4f}")

        print(f"\n  BLINDNESS: identical scores, different hard loads "
              f"(all 8 layers pooled)")
        w = n / n.sum()
        obs = (acc["L"] / n * w).sum()
        for lbl, key, vcf in [
            ("observed routing", "L", (acc["varc"] / n * w).sum()),
            ("perfectly balanced (12x2,20x1)", "Lbal", var_min(K * U, N)),
            ("same load multiset, shuffled", "Lperm", (acc["varc"] / n * w).sum()),
            ("COLLAPSED onto 4 lowest-mass", "Lcol_lo", np.var([U]*K + [0]*(N-K))),
            ("COLLAPSED onto 4 highest-mass", "Lcol_hi", np.var([U]*K + [0]*(N-K))),
        ]:
            v = (acc[key] / n * w).sum()
            print(f"    {lbl:<32} L_HB = {v:8.5f}  ({100*(v-1):+7.3f}%)   "
                  f"Var(c) = {vcf:7.3f}  ({vcf/var_min(K*U,N):6.1f}x floor)")
        grand.append((name, obs))
        print()

    print("=" * 104)
    print("A router that sends ALL 11 tokens to the SAME 4 experts (100% collapse,")
    print("28 of 32 experts dead) scores BETTER on per-sequence HB-Reg than the")
    print("actual checkpoint, provided those 4 experts carry below-average soft mass.")
    print("The loss cannot separate collapse from balance at U=11.")


if __name__ == "__main__":
    main()
