#!/usr/bin/env python3
"""
HB-Reg statistical-scale audit.

Question: HiMoE-VLA computes the DeepSeek load-balancing loss per sequence
(`seq_aux=True`, paper Sec 3.3: "Statistics are computed per sequence and averaged
across the batch").  The expert tower's gates see ONLY the 11 suffix tokens
(1 state + 10 action), against N=32 experts with K=4.  So each sequence has
K*U = 44 dispatch slots for 32 experts.

    f_i = (N/(K U)) sum_u r_{i,u}      P_i = (1/U) sum_u s_{i,u}
    L_HB = sum_i f_i P_i               (paper: "=1 at the balanced configuration
                                        f_i=1, P_i=1/N")

Closed-form facts this script checks against the data:

 (F1) f_i = 1 is INFEASIBLE at U=11: f_i in (N/KU)*Z = (32/44)*Z = {0, .727, 1.455,...}
      The loss's own reference point is off-lattice.  Its implied target hard load
      is K*U/N = 1.375 tokens/expert -- an integer-infeasible value.

 (F2) L_HB == 1 EXACTLY whenever P_i == 1/N, for ANY hard load, including total
      collapse onto 4 experts:  sum_i f_i (1/N) = N/N = 1.
      So "L_HB near 1" carries zero information about expert utilization.

 (F3) Top-K is invariant to temperature, so f_i is too.  Sharpening the router
      therefore cannot change utilization -- it can only change P.  With a
      maximally sharp router (mass 1/K on each selected expert) L = N*S/(K^2 U^2)
      where S = sum_i c_i^2 >= (KU)^2/N.  At U=11: min S = 68 -> L >= 1.1240.
      => the most balanced SHARP router achievable scores WORSE (1.124) than a
         maximally indecisive router (1.000).  The penalty for being decisive is
         12.4% of the loss.  At U'=110 (aggregating the 10 flow steps) the same
         penalty is 0.10%.  It decays as O(N^2/(K U')^2).

Measured here per task, per HB layer, at four aggregation scales.
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr

HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
N, K, U = 32, 4, 11
NDENOISE, NLAYER = 10, 8
HB_MODEL_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
BLOCK = 256


# ---------------------------------------------------------------- closed form
def cv_min(n_slots, n_experts):
    """CV of the most balanced integer allocation of n_slots to n_experts."""
    q, r = divmod(n_slots, n_experts)
    c = np.full(n_experts, q, float)
    c[:r] += 1
    return c.std() / c.mean()


def min_sumsq(n_slots, n_experts):
    q, r = divmod(n_slots, n_experts)
    return r * (q + 1) ** 2 + (n_experts - r) * q ** 2


def L_sharp_floor(u):
    """Lowest L_HB reachable by a maximally sharp router at sequence length u."""
    return N * min_sumsq(K * u, N) / (K ** 2 * u ** 2)


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis)


# ---------------------------------------------------------------- accumulators
class Acc:
    """Per-layer accumulators at several aggregation scales."""

    def __init__(self):
        # sequence scale (U=11): running moments of L and friends
        self.n = 0
        self.sums = {}
        # coarse scales need only (counts c_i, score sums S_i)
        self.c_task = np.zeros((NLAYER, N))
        self.S_task = np.zeros((NLAYER, N))
        self.u_task = 0
        self.c_ep, self.S_ep, self.u_ep = {}, {}, {}      # episode -> arrays
        self.L_query = [[] for _ in range(NLAYER)]
        self.L_seq_all = [[] for _ in range(NLAYER)]
        self.cv_seq = [[] for _ in range(NLAYER)]
        self.load_hist = np.zeros((NLAYER, 12))            # c_i histogram
        self.temp = {}                                     # T -> [NLAYER] sums
        self.temp_n = 0

    def add(self, key, val):
        """val: (NLAYER,) or (NLAYER, ...) summed over sites."""
        if key not in self.sums:
            self.sums[key] = np.zeros_like(val, dtype=float)
        self.sums[key] += val


TEMPS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 4.0]


def process_task(run_dir, temp_stride=4):
    z = zarr.open(str(run_dir / "server" / "routes.zarr"), mode="r")
    S_total = z["hb_router_probs"].shape[0]
    ep_all = z["episode_id"][:]
    acc = Acc()
    ntok_sites = 0

    for b0 in range(0, S_total, BLOCK):
        b1 = min(b0 + BLOCK, S_total)
        p = z["hb_router_probs"][b0:b1].astype(np.float32)      # (B,8,10,11,32)
        ids = z["hb_expert_ids"][b0:b1].astype(np.int64)         # (B,8,10,11,4)
        B = p.shape[0]
        eps = ep_all[b0:b1]

        # renormalise fp16 rounding drift (rows sum to 1 +/- 1e-3)
        p /= p.sum(-1, keepdims=True)

        M = B * NLAYER * NDENOISE                               # sequence sites
        pf = p.reshape(M, U, N)
        idf = ids.reshape(M, U, K)

        # ---- hard load c_i per sequence (44 slots -> 32 experts)
        c = np.zeros((M, N), np.float32)
        np.add.at(c, (np.arange(M)[:, None], idf.reshape(M, U * K)), 1.0)

        # ---- P_i, L_HB per sequence
        P = pf.mean(1)                                          # (M,32)
        L_seq = (N / (K * U)) * (c * P).sum(1)

        # ---- top-K mass per token, entropies
        m4 = np.take_along_axis(pf, idf, axis=2).sum(-1)         # (M,U)
        H_tok = entropy(pf)                                     # (M,U)
        H_load = entropy(c / (K * U))                            # (M,)
        H_soft_mix = entropy(P)                                 # (M,)   H(E)
        # sorted scores for the rank-4/5 gap
        srt = np.sort(pf, axis=2)[:, :, ::-1]
        d45 = srt[:, :, K - 1] - srt[:, :, K]

        lay = np.repeat(np.arange(NLAYER), NDENOISE)
        lay = np.tile(lay, B)                                    # (M,) layer of each site

        def per_layer(v):
            """sum v (M,) into (NLAYER,)"""
            return np.bincount(lay, weights=v, minlength=NLAYER)

        acc.add("L_seq", per_layer(L_seq))
        acc.add("L_seq_sq", per_layer(L_seq ** 2))
        acc.add("m4_state", per_layer(m4[:, 0]))
        acc.add("m4_act", per_layer(m4[:, 1:].mean(1)))
        acc.add("H_tok_state", per_layer(H_tok[:, 0]))
        acc.add("H_tok_act", per_layer(H_tok[:, 1:].mean(1)))
        acc.add("H_tok", per_layer(H_tok.mean(1)))
        acc.add("H_load", per_layer(H_load))
        acc.add("H_soft_mix", per_layer(H_soft_mix))
        acc.add("d45", per_layer(d45.mean(1)))
        acc.add("n_used", per_layer((c > 0).sum(1).astype(float)))
        acc.add("cv_hard", per_layer(c.std(1) / c.mean(1)))
        # distinct top-K sets among the 11 tokens
        packed = np.zeros((M, U), np.int64)
        for k in range(K):
            packed |= (1 << idf[:, :, k])
        n_distinct = np.array([len(np.unique(row)) for row in packed], float)
        acc.add("n_distinct_sets", per_layer(n_distinct))
        acc.n += M // NLAYER

        for l in range(NLAYER):
            sel = lay == l
            acc.L_seq_all[l].append(L_seq[sel].astype(np.float32))
        for l in range(NLAYER):
            acc.load_hist[l] += np.bincount(
                c[lay == l].astype(int).ravel(), minlength=12)[:12]

        # ---- coarse scales: accumulate (c_i, S_i)
        Ssum = pf.sum(1)                                        # (M,32) score sums
        for l in range(NLAYER):
            sel = lay == l
            acc.c_task[l] += c[sel].sum(0)
            acc.S_task[l] += Ssum[sel].sum(0)
        acc.u_task += (M // NLAYER) * U

        # query scale: one control step, all 10 flow steps, U'=110
        c_q = c.reshape(B, NLAYER, NDENOISE, N).sum(2)
        S_q = Ssum.reshape(B, NLAYER, NDENOISE, N).sum(2)
        Uq = U * NDENOISE
        L_q = (N / (K * Uq ** 2)) * (c_q * S_q).sum(-1)          # (B,NLAYER)
        for l in range(NLAYER):
            acc.L_query[l].append(L_q[:, l].astype(np.float32))

        # episode scale
        for e in np.unique(eps):
            m = eps == e
            if e not in acc.c_ep:
                acc.c_ep[e] = np.zeros((NLAYER, N))
                acc.S_ep[e] = np.zeros((NLAYER, N))
                acc.u_ep[e] = 0
            acc.c_ep[e] += c_q[m].sum(0)
            acc.S_ep[e] += S_q[m].sum(0)
            acc.u_ep[e] += m.sum() * Uq

        # ---- temperature sweep (subsampled)
        if b0 % (BLOCK * temp_stride) == 0:
            pt_base = pf[::7]                                    # thin within block
            ct = c[::7]
            lt = lay[::7]
            for T in TEMPS:
                q = pt_base ** (1.0 / T)
                q /= q.sum(-1, keepdims=True)
                Pq = q.mean(1)
                Lq = (N / (K * U)) * (ct * Pq).sum(1)
                key = f"T{T}"
                if key not in acc.temp:
                    acc.temp[key] = np.zeros(NLAYER)
                acc.temp[key] += np.bincount(lt, weights=Lq, minlength=NLAYER)
            acc.temp_n += np.bincount(lt, minlength=NLAYER)[0]

    return acc, S_total


def scale_stats(c, S, u):
    """L_HB, hard-load CV, H_load, I(E;X) proxy at an aggregation scale."""
    slots = K * u
    L = (N / (K * u ** 2)) * (c * S).sum(-1)
    cv = c.std(-1) / c.mean(-1)
    P = S / u
    return dict(L=L, cv=cv, cv_ratio=cv / cv_min(int(slots), N),
                H_load=entropy(c / slots) / np.log(N),
                H_soft=entropy(P) / np.log(N))


def main():
    tasks = sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json"))
    print(f"{len(tasks)} task runs\n")
    print("=" * 100)
    print("CLOSED FORM (N=32, K=4)")
    print("=" * 100)
    for u, lbl in [(11, "sequence (as trained: seq_aux=True)"),
                   (110, "policy query = 11 tok x 10 flow steps"),
                   (1100, "10 control steps"),
                   (11000, "~episode")]:
        print(f"  U={u:<6} slots={K*u:<6} target load K*U/N={K*u/N:8.3f}  "
              f"integer-feasible={'YES' if (K*u) % N == 0 else 'NO ':<3}  "
              f"CV_min={cv_min(K*u, N):.4f}  "
              f"L_floor(sharp)={L_sharp_floor(u):.5f}  "
              f"decisiveness penalty={100*(L_sharp_floor(u)-1):+.3f}%")
    print()

    rows = []
    for run in tasks:
        meta = json.loads((run / "meta.json").read_text())
        summ = json.loads((run / "client" / "summaries.json").read_text())
        name = f"{meta['benchmark']}/t{meta['task_id']}"
        acc, nsteps = process_task(run)
        n = acc.n
        s = acc.sums
        print("=" * 100)
        print(f"{name}   {meta['task_name'][:60]}")
        print(f"  {nsteps} control steps, {len(summ)} episodes, "
              f"success {sum(e['success'] for e in summ)}/{len(summ)}, "
              f"{n*NLAYER} sequence-sites")
        print(f"  {'layer':>6} {'L_seq':>8} {'sd':>7} {'M4_act':>7} {'M4_st':>7} "
              f"{'H_tok/lnN':>9} {'H_load/lnN':>10} {'I(E;X)':>7} {'nused':>6} "
              f"{'sets':>5} {'CVobs/CVmin':>11} {'d45':>8}")
        for l in range(NLAYER):
            L = s["L_seq"][l] / n
            sd = np.sqrt(max(s["L_seq_sq"][l] / n - L ** 2, 0))
            H_tok = s["H_tok"][l] / n
            H_mix = s["H_soft_mix"][l] / n
            print(f"  L{HB_MODEL_LAYERS[l]:>5} {L:8.5f} {sd:7.5f} "
                  f"{s['m4_act'][l]/n:7.4f} {s['m4_state'][l]/n:7.4f} "
                  f"{H_tok/np.log(N):9.4f} {s['H_load'][l]/n/np.log(N):10.4f} "
                  f"{(H_mix-H_tok):7.4f} {s['n_used'][l]/n:6.2f} "
                  f"{s['n_distinct_sets'][l]/n:5.2f} "
                  f"{(s['cv_hard'][l]/n)/cv_min(K*U,N):11.4f} "
                  f"{s['d45'][l]/n:8.5f}")

        # ---- aggregation scales
        print(f"\n  aggregation scale -> L_HB (mean over sites), layer-averaged:")
        seq_L = np.array([s["L_seq"][l] / n for l in range(NLAYER)])
        q_L = np.array([np.concatenate(acc.L_query[l]).mean() for l in range(NLAYER)])
        ep_L, ep_cv = [], []
        for l in range(NLAYER):
            v = [scale_stats(acc.c_ep[e][l], acc.S_ep[e][l], acc.u_ep[e])
                 for e in acc.c_ep]
            ep_L.append(np.mean([x["L"] for x in v]))
            ep_cv.append(np.mean([x["cv_ratio"] for x in v]))
        tk = [scale_stats(acc.c_task[l], acc.S_task[l], acc.u_task)
              for l in range(NLAYER)]
        u_ep_mean = np.mean(list(acc.u_ep.values()))
        print(f"    {'scale':<26} {'U':>8} {'L_HB':>9} {'excess%':>9} "
              f"{'L_floor(sharp)':>15} {'headroom':>9} {'CVobs/CVmin':>11}")
        for lbl, uu, LL, cvr in [
            ("sequence  (as trained)", U, seq_L.mean(),
             np.mean([s['cv_hard'][l]/n for l in range(NLAYER)])/cv_min(K*U, N)),
            ("policy query (10 flow)", U*NDENOISE, q_L.mean(), np.nan),
            ("episode", u_ep_mean, np.mean(ep_L), np.mean(ep_cv)),
            ("whole task (512 eps)", acc.u_task, np.mean([x["L"] for x in tk]),
             np.mean([x["cv_ratio"] for x in tk])),
        ]:
            fl = L_sharp_floor(int(uu))
            print(f"    {lbl:<26} {int(uu):>8} {LL:9.5f} {100*(LL-1):+9.3f} "
                  f"{fl:15.5f} {100*(fl-LL):+9.3f} {cvr:11.4f}")

        # ---- temperature sweep
        print(f"\n  temperature sweep at sequence scale (top-K set FIXED, so f_i fixed):")
        print(f"    {'T':>6} {'L_HB':>9}  (T<1 sharper, T>1 flatter)")
        for T in TEMPS:
            v = acc.temp[f"T{T}"] / acc.temp_n
            print(f"    {T:6.2f} {v.mean():9.5f}")

        # ---- hard-load histogram
        hh = acc.load_hist.sum(0)
        hh = hh / hh.sum()
        print(f"\n  hard load per expert per sequence (44 slots / 32 experts, "
              f"loss target = {K*U/N:.3f}):")
        print("    c_i =  " + " ".join(f"{i:>6}" for i in range(8)))
        print("    frac  " + " ".join(f"{hh[i]:6.3f}" for i in range(8)))
        rows.append((name, seq_L.mean(), q_L.mean(), np.mean(ep_L),
                     np.mean([x["L"] for x in tk]),
                     s["m4_act"].mean() / n, s["m4_state"].mean() / n))
        print()

    print("=" * 100)
    print("SUMMARY  (L_HB by aggregation scale; flat-router value = 1.00000)")
    print("=" * 100)
    print(f"  {'task':<22} {'L_seq(U=11)':>12} {'L_query':>10} {'L_episode':>10} "
          f"{'L_task':>10} {'M4_act':>8} {'M4_state':>9}")
    for r in rows:
        print(f"  {r[0]:<22} {r[1]:12.5f} {r[2]:10.5f} {r[3]:10.5f} {r[4]:10.5f} "
              f"{r[5]:8.4f} {r[6]:9.4f}")
    print(f"\n  uniform-router reference: M4 = K/N = {K/N:.4f}, L_HB = 1.00000")
    print(f"  sharp-router floor at U=11: L_HB = {L_sharp_floor(11):.5f}")


if __name__ == "__main__":
    main()
