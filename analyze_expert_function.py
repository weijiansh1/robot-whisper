#!/usr/bin/env python3
"""What does an individual HB expert compute?

The word "specialisation" has been used for the routing all along, but never for
the thing being routed to.  Known so far: the 32 experts' outputs are pairwise
near-orthogonal (cosine +0.03..+0.05) so they are not copies, and swapping the
selected set for a random one costs only 4% of the action so the *choice* barely
matters.  Neither says what any one expert does.

An expert here is a SwiGLU, 1024 -> 1024, writing into the residual stream:

    E_e(h) = down_proj( silu(gate_proj(h)) * up_proj(h) )

so "what it computes" is "what vector it writes, and how much that depends on h".
Three questions, all answerable from `hidden.zarr` plus the checkpoint:

  1. Is an expert a function or a bias?  Split E_e(h) into its mean over inputs
     and the input-dependent residual.  If the mean dominates, "which expert" is
     just "which constant vector", which would explain why a random substitution
     is nearly free.
  2. Does an expert do something different on the inputs it is *selected* for?
     That is what specialisation would have to mean at this level: compare
     E_e(h) on h where e is in the top-4 against h where it is not.
  3. How do the 32 mean directions sit relative to each other, and to the shared
     expert, which runs on every token regardless.
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
LAYERS = [5, 15]              # one from the front block, one from the back
N_H = 3000


def expert_fn(sd, layer, tag):
    p = "paligemma_with_expert.gemma_expert.layers.%d.mlp.%s." % (layer, tag)
    g = sd[p + "gate_proj.weight"].float()
    u = sd[p + "up_proj.weight"].float()
    d = sd[p + "down_proj.weight"].float()
    return lambda h: (torch.nn.functional.silu(h @ g.T) * (h @ u.T)) @ d.T


def main() -> int:
    sd = torch.load(BRIDGE / "checkpoints" / CKPT / "pytorch_model.pth",
                    map_location="cpu", weights_only=True, mmap=True)
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    h5 = zarr.open(str(RUN / "server/hidden.zarr"), mode="r")
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = h5["hb_hidden"].shape[0]
    rng = np.random.default_rng(5)
    rows = np.sort(rng.choice(n, N_H, replace=False))

    for L in LAYERS:
        i = HB_LAYER.index(L)
        H = np.empty((len(rows), 1024), np.float32)
        ids = np.empty((len(rows), 4), np.int64)
        for a in range(0, len(rows), 256):
            b = min(a + 256, len(rows))
            H[a:b] = np.asarray(h5["hb_hidden"].oindex[rows[a:b], i, 0, 0, :],
                                np.float32)
            ids[a:b] = np.asarray(z["hb_expert_ids"].oindex[rows[a:b], i, 0, 0, :])
        h = torch.from_numpy(H)
        sel = np.zeros((len(rows), 32), bool)
        np.put_along_axis(sel, ids, True, 1)

        print("\n=== layer %d, state token, %d hidden states ===" % (L, len(rows)))
        outs, means = [], []
        for e in range(32):
            y = expert_fn(sd, L, "experts.%d" % e)(h)
            outs.append(y)
            means.append(y.mean(0))
        Y = torch.stack(outs)                       # [32, N, 1024]
        M = torch.stack(means)                      # [32, 1024]
        sh = expert_fn(sd, L, "shared_experts")(h)

        # 1. function or bias?
        var = (Y - M[:, None]).pow(2).sum(-1).mean(1)     # per expert
        bias = M.pow(2).sum(-1)
        print("  1. is an expert a function or a constant?")
        print("     ||mean output||^2 / ||input-dependent part||^2 :  "
              "median %.2f   range %.2f - %.2f"
              % (float((bias / var).median()), float((bias / var).min()),
                 float((bias / var).max())))
        print("     the shared expert: %.2f"
              % float(sh.mean(0).pow(2).sum() / (sh - sh.mean(0)).pow(2).sum(-1).mean()))
        print("     (>1 means the expert is mostly a fixed vector it always writes)")

        # 2. does it behave differently where it is selected?
        d_norm, d_cos, n_ok = [], [], 0
        for e in range(32):
            m = sel[:, e]
            if m.sum() < 50 or (~m).sum() < 50:
                continue
            n_ok += 1
            a, b = Y[e][m], Y[e][~m]
            d_norm.append(float(a.norm(dim=1).mean() / b.norm(dim=1).mean()))
            ca = a.mean(0) / a.mean(0).norm()
            cb = b.mean(0) / b.mean(0).norm()
            d_cos.append(float(ca @ cb))
        print("  2. on the inputs it is selected for vs the ones it is not "
              "(%d experts with both):" % n_ok)
        print("     output norm ratio  %.3f +- %.3f      "
              "cosine of the two mean directions  %.4f +- %.4f"
              % (np.mean(d_norm), np.std(d_norm), np.mean(d_cos), np.std(d_cos)))
        print("     (1.000 and 1.0000 = the expert does the same thing either way)")

        # 3. how the 32 sit relative to each other and to the shared expert
        Mn = M / M.norm(dim=1, keepdim=True)
        C = (Mn @ Mn.T).numpy()
        iu = np.triu_indices(32, 1)
        shm = sh.mean(0) / sh.mean(0).norm()
        print("  3. geometry of the 32 mean output directions:")
        print("     pairwise cosine  %+.4f +- %.4f   (random in d=1024: 0 +- %.4f)"
              % (C[iu].mean(), C[iu].std(), 1 / np.sqrt(1024)))
        print("     cosine with the shared expert's mean  %+.4f +- %.4f"
              % (float((Mn @ shm).mean()), float((Mn @ shm).std())))
        print("     ||expert mean|| / ||shared mean||  median %.3f"
              % float((M.norm(dim=1) / sh.mean(0).norm()).median()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
