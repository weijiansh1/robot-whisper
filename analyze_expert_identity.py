#!/usr/bin/env python3
"""Two follow-ups: what the router is selecting for, and whether expert identity
survives across checkpoints.

A. The experts turned out to write a 30% larger vector on the inputs they are
   routed to (layer 5; 7% at layer 15), which is what specialisation would look
   like -- but it could equally be an artefact of the gate row and the expert's
   own input projections having been trained together and ending up reading
   similar directions.  That is testable and, if true, is itself the mechanism:
   correlate the gate's score for expert e against the size of what e would
   write, over the same inputs.  A positive correlation means the router picks
   whichever expert is about to do the most.

B. The four LIBERO checkpoints were fine-tuned from one heterogeneous pretrain.
   For the AS gates the difference was confined to the seven columns LIBERO's
   data_mask turns on.  For HB nothing is confined, so: is expert e in the Goal
   checkpoint still the same function as expert e in Spatial, or did fine-tuning
   permute what the slots mean?  If identity survives, "expert 12" is a thing one
   can talk about across the suite; if not, it is a slot index and nothing more.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
CKPT = {"Goal": "HiMoE-VLA-Libero-Goal", "Spatial": "HiMoE-VLA-Libero-Spatial",
        "Object": "HiMoE-VLA-Libero-Object", "Long": "HiMoE-VLA-Libero-10"}
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
P = "paligemma_with_expert.gemma_expert.layers.%d.mlp.%s.%s.weight"


def load(name):
    return torch.load(BRIDGE / "checkpoints" / name / "pytorch_model.pth",
                      map_location="cpu", weights_only=True, mmap=True)


def efn(sd, L, tag):
    g = sd[P % (L, tag, "gate_proj")].float()
    u = sd[P % (L, tag, "up_proj")].float()
    d = sd[P % (L, tag, "down_proj")].float()
    return lambda h: (torch.nn.functional.silu(h @ g.T) * (h @ u.T)) @ d.T


def main() -> int:
    sd = load(CKPT["Long"])
    h5 = zarr.open(str(RUN / "server/hidden.zarr"), mode="r")
    rng = np.random.default_rng(5)
    rows = np.sort(rng.choice(h5["hb_hidden"].shape[0], 3000, replace=False))

    print("=== A. does the router pick whichever expert will write the most? ===")
    print("layer   corr(gate score for e, ||E_e(h)||)   over 32 experts")
    for L in (2, 3, 4, 5, 12, 13, 14, 15):
        i = HB_LAYER.index(L)
        H = np.empty((len(rows), 1024), np.float32)
        for a in range(0, len(rows), 256):
            b = min(a + 256, len(rows))
            H[a:b] = np.asarray(h5["hb_hidden"].oindex[rows[a:b], i, 0, 0, :],
                                np.float32)
        h = torch.from_numpy(H)
        W = sd["paligemma_with_expert.gemma_expert.layers.%d.mlp.gate.weight" % L].float()
        score = (h @ W.T).numpy()                     # [N, 32] pre-softmax
        rs = []
        for e in range(32):
            y = efn(sd, L, "experts.%d" % e)(h).norm(dim=1).numpy()
            rs.append(np.corrcoef(score[:, e], y)[0, 1])
        rs = np.array(rs)
        print("  %2d      %+.3f +- %.3f     (min %+.3f, max %+.3f, %d of 32 positive)"
              % (L, rs.mean(), rs.std(), rs.min(), rs.max(), int((rs > 0).sum())))
    print("  a positive correlation means the gate's winner is also the expert")
    print("  about to write the largest vector into the residual stream")

    print("\n=== B. is expert e the same expert across the four checkpoints? ===")
    ref = load(CKPT["Goal"])
    L = 5
    i = HB_LAYER.index(L)
    H = np.empty((len(rows), 1024), np.float32)
    for a in range(0, len(rows), 256):
        b = min(a + 256, len(rows))
        H[a:b] = np.asarray(h5["hb_hidden"].oindex[rows[a:b], i, 0, 0, :], np.float32)
    h = torch.from_numpy(H)
    Yref = torch.stack([efn(ref, L, "experts.%d" % e)(h) for e in range(32)])
    Yref = Yref / Yref.norm(dim=2, keepdim=True)

    for tag, name in list(CKPT.items())[1:]:
        other = load(name)
        dW = max(float((other[P % (L, "experts.%d" % e, p)].float()
                        - ref[P % (L, "experts.%d" % e, p)].float()).abs().max())
                 for e in range(32) for p in ("gate_proj", "up_proj", "down_proj"))
        Y = torch.stack([efn(other, L, "experts.%d" % e)(h) for e in range(32)])
        Y = Y / Y.norm(dim=2, keepdim=True)
        # cosine between expert e here and every expert j in the reference
        C = torch.einsum("end,fnd->ef", Y, Yref) / len(h)
        diag = C.diagonal()
        offd = C[~torch.eye(32, dtype=bool)]
        match = int((C.argmax(1) == torch.arange(32)).sum())
        print("  %-8s vs Goal   max|dW| %.4f   cos(e, same e) %.3f +- %.3f   "
              "cos(e, other e) %+.3f +- %.3f   nearest match is itself for %d/32"
              % (tag, dW, float(diag.mean()), float(diag.std()),
                 float(offd.mean()), float(offd.std()), match))
    print("  layer %d, state token, %d inputs; 1.000 on the diagonal would mean "
          "fine-tuning left the expert untouched" % (L, len(h)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
