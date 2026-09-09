#!/usr/bin/env python3
"""Two follow-ups the first denoise pass forced.

A. The top-4 identity churns hard along the denoise axis -- only 1.2-1.7 of 4
   experts survive from d=0 to d=9, against a chance baseline of 0.5 -- while
   the router distribution itself moves by a total variation of 0.04.  Either
   the churn is real reselection among distinguishable experts, or it is ties
   flipping.  Measure the probability gap at each swap: for every expert that
   drops out of the top-4 between d and d+1, how far is it from the one that
   replaced it?  A gap at the bf16 floor is a tie, not a decision.

B. 68-94% of the router's variance sits on the suffix-token axis, and the state
   token (index 0) is 0.25 nats sharper than the ten action tokens.  So ask
   which token the late-window outcome signal actually comes from.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
STEPS = [0, 13, 26, 34]
HB_LAYER_IDX = [2, 3, 4, 5, 12, 13, 14, 15]
BF16_EPS = 2.0 ** -8


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    off = np.concatenate([[0], np.cumsum([s["inference_calls"] for s in S])[:-1]])
    mixed = [int(s) for s in np.unique(scene)
             if 0 < y[scene == s].sum() < (scene == s).sum()]

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = len(S)
    P = np.empty((n, len(STEPS), 8, 10, 11, 32), np.float32)
    ent = np.empty((n, len(STEPS), 8, 10, 11), np.float32)
    for i in range(n):
        rows = off[i] + np.array(STEPS)
        P[i] = np.asarray(z["hb_router_probs"].oindex[rows, :, :, :, :], np.float32)
        ent[i] = np.asarray(z["hb_entropy"].oindex[rows, :, :, :], np.float32)
        if i % 128 == 0:
            print("  read %d/%d" % (i, n), flush=True)

    # ---- A. is the denoise churn a decision or a tie? ----------------------
    print("\n=== A. the probability gap at every top-4 swap along d ===")
    srt = np.sort(P, -1)[..., ::-1]
    gap45 = srt[..., 3] - srt[..., 4]           # 4th minus 5th at each site
    print("gap between the 4th and 5th expert, all sites:")
    print("  median %.6f   mean %.6f   frac < bf16 eps (%.5f): %.1f%%   frac == 0: %.1f%%"
          % (np.median(gap45), gap45.mean(), BF16_EPS,
             100 * (gap45 < BF16_EPS).mean(), 100 * (gap45 == 0).mean()))

    order = np.argsort(-P, -1)
    top4 = np.sort(order[..., :4], -1)
    hot = np.zeros(P.shape[:-1] + (32,), bool)
    np.put_along_axis(hot, top4, True, axis=-1)
    left = hot[:, :, :, :-1] & ~hot[:, :, :, 1:]      # dropped out between d,d+1
    came = ~hot[:, :, :, :-1] & hot[:, :, :, 1:]      # came in
    pa, pb = P[:, :, :, :-1], P[:, :, :, 1:]
    # the swap gap, measured on the distribution *before* the move
    gl = np.where(left, pa, -np.inf).max(-1)          # best expert that left
    gc = np.where(came, pa, -np.inf).max(-1)          # best expert that arrived
    swapped = np.isfinite(gl) & np.isfinite(gc)
    d = gl[swapped] - gc[swapped]     # subset first: -inf - -inf would be a nan
    print("\nswaps between adjacent denoise steps: %d sites of %d (%.1f%%)"
          % (swapped.sum(), swapped.size, 100 * swapped.mean()))
    print("  at the moment of the swap the outgoing expert led the incoming one by")
    print("  median %.6f   90th pct %.6f   frac < bf16 eps: %.1f%%"
          % (np.median(d), np.percentile(d, 90), 100 * (d < BF16_EPS).mean()))

    # ---- B. which suffix token carries the outcome? -----------------------
    print("\n=== B. outcome AUC by suffix token (entropy, mean over layers, d) ===")

    def sauc(x):
        num = den = 0.0
        for s in mixed:
            m = scene == s
            n1, n0 = int(y[m].sum()), int((~y[m]).sum())
            r = rankdata(x[m])
            num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
            den += n1 * n0
        return num / den

    print("step   state(s0)   action tokens s1..s10")
    for ti, t in enumerate(STEPS):
        a0 = sauc(ent[:, ti, :, :, 0].mean((1, 2)))
        aa = [sauc(ent[:, ti, :, :, s].mean((1, 2))) for s in range(1, 11)]
        print("%4d      %.3f      " % (t, a0) + " ".join("%.3f" % v for v in aa))
    print("\n  action-token block as one feature (mean over s1..s10):")
    for ti, t in enumerate(STEPS):
        print("    step %2d: %.3f" % (t, sauc(ent[:, ti, :, :, 1:].mean((1, 2, 3)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
