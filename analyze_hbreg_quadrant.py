#!/usr/bin/env python3
"""
Which quadrant is the HB router in?

  global utilization balanced?  x  per-token routing sharp?
    balanced + sharp  = healthy balanced specialization,  I(E;X) high
    balanced + flat   = "balanced by indecision",         I(E;X) ~ 0
    unbalanced + sharp = collapse / over-specialisation
    unbalanced + flat  = both degenerate

Measures, per HB layer, at task scale (512 episodes):
  - global hard-load CV, H/lnN, share of the busiest/idlest expert
  - I(E;X)/lnN for X = token position, X = flow step, X = observation
    (nested, so the increments are the variance attributable to each factor)
  - cross-task correlation of the global load vector
"""
import json
from pathlib import Path

import numpy as np
import zarr

HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
N, K, U, NDENOISE, NLAYER = 32, 4, 11, 10, 8
HB = [2, 3, 4, 5, 12, 13, 14, 15]
BLOCK, STRIDE = 128, 6
LN = np.log(N)


def H(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis)


def main():
    tasks = sorted(p.parent for p in HUB.glob("*/*/right-16x32/meta.json"))
    loads, names = [], []
    for run in tasks:
        meta = json.loads((run / "meta.json").read_text())
        name = f"{meta['benchmark']}/t{meta['task_id']}"
        z = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
        T = z["hb_router_probs"].shape[0]

        c_g = np.zeros((NLAYER, N))                     # global hard load
        P_g = np.zeros((NLAYER, N))                     # global mean score
        P_pos = np.zeros((NLAYER, U, N))                # by token position
        P_den = np.zeros((NLAYER, NDENOISE, N))         # by flow step
        Hcond = np.zeros(NLAYER)                        # E_x[H(p_x)] per token
        nsite = 0
        for b0 in range(0, T, BLOCK * STRIDE):
            b1 = min(b0 + BLOCK, T)
            p = z["hb_router_probs"][b0:b1].astype(np.float32)
            ids = z["hb_expert_ids"][b0:b1].astype(np.int64)
            p /= p.sum(-1, keepdims=True)
            B = p.shape[0]
            oh = np.zeros((B, NLAYER, NDENOISE, U, N), np.float32)
            np.put_along_axis(oh, ids, 1.0, axis=-1)
            c_g += oh.sum((0, 2, 3))
            P_g += p.sum((0, 2, 3))
            P_pos += p.sum((0, 2))
            P_den += p.sum((0, 3))
            Hcond += H(p).sum((0, 2, 3))
            nsite += B * NDENOISE * U
        P_g /= nsite
        Hcond /= nsite
        loads.append(c_g / c_g.sum(-1, keepdims=True))
        names.append(name)

        print("=" * 100)
        print(f"{name}  {meta['task_name'][:52]}   ({T} control steps)")
        print(f"  {'layer':>5} {'CV_load':>8} {'H_load/lnN':>10} {'busiest%':>9} "
              f"{'idlest%':>8} {'<1/64 sh':>8} | {'H(E)/lnN':>9} {'H(E|x)/lnN':>11} "
              f"{'I(E;X)/lnN':>11} {'I_pos':>7} {'I_flow':>7} {'I_obs':>7}")
        for l in range(NLAYER):
            f = c_g[l] / c_g[l].sum()
            Hg = H(P_g[l])
            Ppos = P_pos[l] / P_pos[l].sum(-1, keepdims=True)
            Pden = P_den[l] / P_den[l].sum(-1, keepdims=True)
            I_pos = Hg - H(Ppos).mean()
            I_den = Hg - H(Pden).mean()
            I_tot = Hg - Hcond[l]
            print(f"  L{HB[l]:>4} {f.std()/f.mean():8.4f} {H(f)/LN:10.4f} "
                  f"{100*f.max():9.3f} {100*f.min():8.3f} "
                  f"{(f < 1/(2*N)).sum():8d} | {Hg/LN:9.5f} {Hcond[l]/LN:11.5f} "
                  f"{I_tot/LN:11.5f} {I_pos/LN:7.4f} {I_den/LN:7.4f} "
                  f"{(I_tot-I_pos-I_den)/LN:7.4f}")
        print(f"  uniform load = {100/N:.3f}% per expert")

    print("=" * 100)
    print("cross-task correlation of the global load vector (is utilization task-specific?)")
    L = np.stack(loads)                                  # (task, layer, 32)
    for l in range(NLAYER):
        M = np.corrcoef(L[:, l, :])
        off = M[np.triu_indices(len(names), 1)]
        print(f"  L{HB[l]:>4}  mean pairwise r = {off.mean():+.4f}  "
              f"[min {off.min():+.3f}, max {off.max():+.3f}]")


if __name__ == "__main__":
    main()
