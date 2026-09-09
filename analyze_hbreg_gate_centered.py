#!/usr/bin/env python3
"""
Centered router geometry.  Adding a common vector to every expert row shifts all
logits by the same scalar, so softmax is invariant to it: raw ||w_bar||/||resid||
and raw pairwise cosine are gauge-contaminated.  The gauge-invariant object is

    W~ = W - 1 w_bar^T          (expert-row-centered gate)

and the quantity that actually sets routing contrast on real data is

    E_h[ Var_j(z_j) ] = (1/N) Tr( W~ M2 W~^T ),     M2 = E[h h^T]

which splits exactly into an observation-INDEPENDENT and a conditional part:

    E_h[Var_j(z_j)] = Var_j(w~_j . h_bar)  +  (1/N) Tr( W~ Cov(h) W~^T )
                      \_ fixed contrast _/    \_ conditional contrast _/

The fixed part is identical for every token and observation, so it produces a
constant expert preference and carries ZERO conditional information.  Only the
second term can supply I(E;X).
"""
import math
from pathlib import Path

import numpy as np
import torch
import zarr

CKPT = Path("/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/"
            "cache/checkpoints/HiMoE-VLA-Libero-Spatial/pytorch_model.pth")
RUN = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_spatial/"
           "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate/right-16x32")
HB = [2, 3, 4, 5, 12, 13, 14, 15]
N, D = 32, 1024


def eff_rank(S):
    """singular-value entropy effective rank"""
    q = S ** 2 / (S ** 2).sum()
    q = q[q > 0]
    return math.exp(-(q * np.log(q)).sum())


def geom(W):
    wb = W.mean(0)
    Wt = W - wb
    S = np.linalg.svd(Wt, compute_uv=False)
    return dict(w_bar=np.linalg.norm(wb), Wt_F=np.linalg.norm(Wt),
                erank=eff_rank(S), s1=S[0], Wt=Wt)


def main():
    print("loading gate weights (mmap)...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    Ws = {}
    for lay in HB:
        k = [x for x in sd if f"layers.{lay}.mlp.gate.weight" in x][0]
        Ws[lay] = sd[k].float().numpy().astype(np.float64)

    rng = np.random.default_rng(0)
    Wk = rng.uniform(-1 / 32, 1 / 32, size=(N, D))

    # ---- second moments of the real router inputs, per HB layer
    h = zarr.open(str(RUN / "server" / "hidden.zarr"), mode="r")["hb_hidden"]
    idx = np.linspace(0, h.shape[0] - 1, 256).astype(int)
    print(f"accumulating h statistics over {len(idx)} control steps "
          f"x 10 flow x 11 tokens ...")
    Hs = h.oindex[idx].astype(np.float32)              # (256,8,10,11,1024)

    print("\n" + "=" * 104)
    print("CENTERED GATE GEOMETRY  (softmax-invariant common component removed)")
    print("=" * 104)
    gk = geom(Wk)
    print(f"  {'gate':<8} {'||w_bar||':>10} {'||W~||_F':>9} {'vs init':>8} "
          f"{'eff_rank(W~)':>13} {'sigma1/||W~||':>14}")
    print(f"  {'kaiming':<8} {gk['w_bar']:10.4f} {gk['Wt_F']:9.4f} "
          f"{1.0:8.3f} {gk['erank']:13.2f} {gk['s1']/gk['Wt_F']:14.4f}")
    G = {}
    for lay in HB:
        g = geom(Ws[lay]); G[lay] = g
        print(f"  L{lay:<7} {g['w_bar']:10.4f} {g['Wt_F']:9.4f} "
              f"{g['Wt_F']/gk['Wt_F']:8.3f} {g['erank']:13.2f} "
              f"{g['s1']/g['Wt_F']:14.4f}")

    print("\n" + "=" * 104)
    print("LOGIT CONTRAST ON REAL DATA, split into fixed vs conditional")
    print("=" * 104)
    print(f"  {'gate':<7} {'E[Var_j z]':>11} {'fixed':>10} {'conditional':>12} "
          f"{'cond frac':>10} | {'kaiming tot':>12} {'ratio':>7} {'k cond frac':>12}")
    for li, lay in enumerate(HB):
        hh = Hs[:, li].reshape(-1, D).astype(np.float64)
        hbar = hh.mean(0)
        dh = hh - hbar
        C = (dh.T @ dh) / len(dh)                       # Cov(h)
        out = []
        for W, g in ((Ws[lay], G[lay]), (Wk, gk)):
            Wt = g["Wt"]
            fixed = np.var(Wt @ hbar)
            cond = np.trace(Wt @ C @ Wt.T) / N
            tot = np.mean(np.var(hh @ Wt.T, axis=1))
            out.append((tot, fixed, cond))
        (t, fx, cd), (tk, fxk, cdk) = out
        print(f"  L{lay:<6} {t:11.5f} {fx:10.5f} {cd:12.5f} {cd/t:10.4f} | "
              f"{tk:12.5f} {t/tk:7.3f} {cdk/tk:12.4f}")
        if lay == HB[0]:
            print(f"         (identity check: fixed+cond = {fx+cd:.5f} vs "
                  f"E[Var_j z] = {t:.5f})")

    print("\n  E[Var_j z] is the SQUARE of the logit sd; sqrt gives the earlier table.")
    print("  'cond frac' is the share of routing contrast that can carry any")
    print("  observation dependence at all -- compare with I(E;X)/lnN = 0.004-0.025.")


if __name__ == "__main__":
    main()
