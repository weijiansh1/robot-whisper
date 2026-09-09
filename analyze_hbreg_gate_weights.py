#!/usr/bin/env python3
"""
Was the HB router FLATTENED, or was it never SHARPENED?

Both produce near-uniform softmax scores, but they leave different fingerprints
in the gate weight W (32 x 1024).

The HB-Reg gradient is  dL/dz_j = (s_j/U)(f_j - L_u), so  dL/dW_j = dL/dz_j * h.
It is a function of LOAD ONLY -- it never references the direction of h.  Its
fixed point is z_j equal for all j, i.e. all 32 rows of W collapsing toward a
common vector.  So flattening pressure shows up as:
   (a) shrunk logit spread relative to kaiming init on the same hidden states
   (b) rows of W collapsing onto a shared direction: ||w_bar|| >> ||w_j - w_bar||
An untrained / never-sharpened router shows neither: its rows stay mutually
near-orthogonal at the init scale (mean pairwise cosine ~ 0, ||w_bar||/||resid||
~ 1/sqrt(32) ~ 0.18).

Init used by the repo: kaiming_uniform_(a=sqrt(5)) on (32, 1024)
  -> bound = sqrt(6/((1+5)*1024)) = 1/32, std = (1/32)/sqrt(3) = 0.018042
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
AS = [0, 1, 16, 17]
N = 32


def row_stats(W):
    """W: (N, d) float64"""
    wb = W.mean(0)
    resid = W - wb
    nrm = np.linalg.norm(W, axis=1)
    Wn = W / nrm[:, None]
    cos = Wn @ Wn.T
    off = cos[np.triu_indices(len(W), 1)]
    return dict(
        row_norm_mean=nrm.mean(), row_norm_cv=nrm.std() / nrm.mean(),
        w_std=W.std(),
        shared=np.linalg.norm(wb) / np.linalg.norm(resid, axis=1).mean(),
        cos_mean=off.mean(), cos_sd=off.std(),
    )


def main():
    print("loading gate weights (mmap)...")
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(sd, dict) or "model" in sd:
        sd = sd.get("model", sd)
    keys = [k for k in sd if ".mlp.gate.weight" in k]
    print(f"  {len(keys)} gate weights found\n")

    KA_STD = (1 / 32) / math.sqrt(3)
    print(f"kaiming_uniform_(a=sqrt(5)) on (32,1024): elementwise std = {KA_STD:.6f}, "
          f"row norm = {KA_STD*math.sqrt(1024):.4f}")
    print(f"random-row reference: shared/resid = 1/sqrt(32) = {1/math.sqrt(32):.4f}, "
          f"cos_mean = 0\n")

    rng = np.random.default_rng(0)
    Wk = rng.uniform(-1 / 32, 1 / 32, size=(N, 1024))
    ks = row_stats(Wk)
    print("=" * 104)
    print(f"{'gate':<12} {'kind':>4} {'elem_std':>9} {'vs init':>8} {'rownorm':>8} "
          f"{'rn_CV':>7} {'shared/resid':>12} {'cos_mean':>9} {'cos_sd':>7}")
    print("-" * 104)
    print(f"{'KAIMING INIT':<12} {'--':>4} {ks['w_std']:9.6f} {1.0:8.3f} "
          f"{ks['row_norm_mean']:8.4f} {ks['row_norm_cv']:7.4f} "
          f"{ks['shared']:12.4f} {ks['cos_mean']:+9.4f} {ks['cos_sd']:7.4f}")

    Ws = {}
    for lay in HB + AS:
        k = [x for x in keys if f"layers.{lay}.mlp.gate.weight" in x]
        if not k:
            continue
        W = sd[k[0]].float().numpy().astype(np.float64)
        Ws[lay] = W
        st = row_stats(W)
        kind = "HB" if lay in HB else "AS"
        print(f"L{lay:<11} {kind:>4} {st['w_std']:9.6f} {st['w_std']/KA_STD:8.3f} "
              f"{st['row_norm_mean']:8.4f} {st['row_norm_cv']:7.4f} "
              f"{st['shared']:12.4f} {st['cos_mean']:+9.4f} {st['cos_sd']:7.4f}")

    # ---- logit spread on REAL hidden states, actual W vs kaiming W
    print("\n" + "=" * 104)
    print("logit spread on the REAL router inputs (hidden.zarr, 1024-d), "
          "actual W vs kaiming-reinit W")
    print("  a uniform softmax over 32 needs logit sd -> 0; "
          "logit sd ~ 1 already gives a strongly peaked router")
    h = zarr.open(str(RUN / "server" / "hidden.zarr"), mode="r")["hb_hidden"]
    idx = np.linspace(0, h.shape[0] - 1, 64).astype(int)
    Hs = h.oindex[idx].astype(np.float64)          # (64,8,10,11,1024)
    print(f"\n  {'gate':<8} {'|h|':>8} {'logit_sd(actual)':>17} "
          f"{'logit_sd(kaiming)':>18} {'ratio':>7} {'M4(actual)':>11} {'M4(kaiming)':>12}")
    for li, lay in enumerate(HB):
        hh = Hs[:, li].reshape(-1, 1024)
        z_a = hh @ Ws[lay].T
        z_k = hh @ Wk.T
        def m4(z):
            z = z - z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            return np.sort(p, 1)[:, -4:].sum(1).mean()
        print(f"  L{lay:<7} {np.linalg.norm(hh,axis=1).mean():8.2f} "
              f"{z_a.std(1).mean():17.4f} {z_k.std(1).mean():18.4f} "
              f"{z_a.std(1).mean()/z_k.std(1).mean():7.3f} "
              f"{m4(z_a):11.4f} {m4(z_k):12.4f}")
    print(f"\n  uniform reference: M4 = 4/32 = {4/32:.4f}")


if __name__ == "__main__":
    main()
