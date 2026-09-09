#!/usr/bin/env python3
"""What the router reads: the geometry of the gate, not the shape of its output.

Everything measured so far is downstream of the gate -- probabilities, top-4
sets, entropy, load.  The gate itself is a linear map from a 1024-dim hidden
state to 32 logits, and `hidden.zarr` holds every one of its inputs, so the
question of *what direction it reads* is answerable offline.

Asked at the state token, which is the only place the routing depends on the
input at all, and specifically at layers 4-5, where that dependence is a bimodal
gate keyed on the gripper:

  1. the gate matrix W (32 x 1024): its singular spectrum.  If it is low rank the
     router is not choosing in 32 directions, it is choosing along a few.
  2. the hidden states h: how much of their variance lies in the subspace W can
     see at all.  Variance the gate is blind to cannot influence routing however
     large it is.
  3. the switch direction: h averaged over the ON steps minus h averaged over the
     OFF steps.  Is it one direction, and does the gate's own row for the
     favourite expert point along it?
  4. and is that direction a gripper readout -- does projecting h onto it
     recover the gripper opening?
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import torch
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
CKPT = "HiMoE-VLA-Libero-10"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
CH = 1024


def gates():
    sd = torch.load(BRIDGE / "checkpoints" / CKPT / "pytorch_model.pth",
                    map_location="cpu", weights_only=True, mmap=True)
    out = {}
    for k in sd:
        if k.endswith("mlp.gate.weight") and sd[k].shape == (32, 1024):
            out[int(re.search(r"layers\.(\d+)\.", k).group(1))] = sd[k].float().numpy()
    return out


def main() -> int:
    W = gates()
    print("=== 1. is the gate low rank? singular spectrum of W (32 x 1024) ===")
    print("layer   sigma1   sigma1/sigma32   effective rank   90%% of the energy in")
    for L in HB_LAYER:
        s = np.linalg.svd(W[L], compute_uv=False)
        e = s ** 2 / (s ** 2).sum()
        eff = float(np.exp(-(e * np.log(e)).sum()))
        k90 = int(np.searchsorted(np.cumsum(e), 0.90) + 1)
        print("  %2d    %.4f      %6.2f          %5.1f / 32        %d directions"
              % (L, s[0], s[0] / s[-1], eff, k90))

    # ---- the state token's hidden states -------------------------------
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    h5 = zarr.open(str(RUN / "server/hidden.zarr"), mode="r")
    n = h5["hb_hidden"].shape[0]
    rows = np.sort(np.concatenate([off + t for t in range(0, 35, 2)]))
    rows = rows[rows < n]
    H = np.empty((len(rows), 8, 1024), np.float32)
    for a in range(0, len(rows), 256):
        b = min(a + 256, len(rows))
        H[a:b] = np.asarray(h5["hb_hidden"].oindex[rows[a:b], :, 0, 0, :], np.float32)
    print("\n  read %d state-token hidden states" % len(H))

    print("\n=== 2. how much of h does the gate even see? ===")
    print("layer   var of h   var inside the row space of W   share")
    for i, L in enumerate(HB_LAYER):
        X = H[:, i] - H[:, i].mean(0)
        tot = (X ** 2).sum(1).mean()
        Q = np.linalg.qr(W[L].T)[0]                    # 1024 x 32 orthonormal
        inside = ((X @ Q) ** 2).sum(1).mean()
        print("  %2d    %.4f     %.4f                       %5.1f%%   "
              "(32 of 1024 dims at random would give %.1f%%)"
              % (L, tot, inside, 100 * inside / tot, 100 * 32 / 1024))

    # ---- the switch direction at L4 and L5 ------------------------------
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    P = np.empty((len(rows), 8, 32), np.float32)
    for a in range(0, len(rows), 256):
        b = min(a + 256, len(rows))
        P[a:b] = np.asarray(z["hb_router_probs"].oindex[rows[a:b], :, 0, 0, :],
                            np.float32)
    star = P.mean(0).argmax(-1)

    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i, :n_rows[i]] = d[:n_rows[i]]
    ep = np.searchsorted(off, rows, "right") - 1
    st = rows - off[ep]
    gap = prop[ep, st, 6] - prop[ep, st, 7]

    print("\n=== 3. the switch direction, and whether the gate points along it ===")
    print("layer  favourite  on%%    |d| / |h|   cos(d, gate row of e*)   "
          "cos(d, the other 31 rows)")
    for i, L in enumerate(HB_LAYER):
        p = P[:, i, star[i]]
        on, offm = p > 0.5, p < 2 / 32
        if on.sum() < 50 or offm.sum() < 50:
            print("  %2d      e%-2d     %4.1f%%   -- not bimodal here" % (L, star[i], 100 * on.mean()))
            continue
        d = H[on, i].mean(0) - H[offm, i].mean(0)
        dn = d / np.linalg.norm(d)
        w = W[L] / np.linalg.norm(W[L], axis=1, keepdims=True)
        c = w @ dn
        others = np.delete(c, star[i])
        print("  %2d      e%-2d     %4.1f%%   %.4f      %+.4f                 "
              "%+.4f +- %.4f"
              % (L, star[i], 100 * on.mean(),
                 np.linalg.norm(d) / np.linalg.norm(H[:, i].mean(0)),
                 c[star[i]], others.mean(), others.std()))

        # ---- 4. is that direction the gripper? --------------------------
        proj = (H[:, i] - H[:, i].mean(0)) @ dn
        r = np.corrcoef(proj, gap)[0, 1]
        r_best = max(abs(np.corrcoef((H[:, i] - H[:, i].mean(0)) @
                                     (W[L][j] / np.linalg.norm(W[L][j])), gap)[0, 1])
                     for j in range(32))
        print("         projecting h on that direction vs the gripper opening: "
              "r = %+.3f   (best single gate row: %.3f)" % (r, r_best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
