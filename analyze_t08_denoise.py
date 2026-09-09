#!/usr/bin/env python3
"""What the router does across the 10 flow-matching denoise iterations.

Every analysis so far averaged the denoise axis away.  It is the one axis where
the input is known by construction: `dt = -1/num_steps` (serve_flow_trace.py:73)
so time runs 1 -> 0, and at d=0 the action tokens are *pure Gaussian noise*
drawn from flow_noise_seed, while by d=9 they are one Euler step from the chunk
the policy emits.  The prefix -- image, wrist, prompt, state token -- is
identical across all ten.

That makes d=0 a free experiment.  Within one scene the 32 episodes share a
bit-identical observation and differ only in the noise seed, so routing spread
at d=0 within a scene is the router's response to noise alone, and spread across
scenes at d=0 is its response to the observation.  The ratio says what the
router is actually listening to.

Four questions:

  1. does the router sharpen as the chunk denoises?  (entropy, top-4 mass vs d)
  2. how much does the choice move along d?  (top-4 overlap, TV distance)
  3. where does the variance live -- denoise, suffix token, control step,
     episode, scene?
  4. is any single denoise iteration enough for the outcome signal, or does the
     average over d carry more?

The suffix is 11 tokens: index 0 is the state token, 1..10 the action chunk.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
STEPS = [0, 5, 13, 20, 26, 34]
HB_LAYER_IDX = [2, 3, 4, 5, 12, 13, 14, 15]      # decoder layers behind axis 1


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    off = np.concatenate([[0], np.cumsum([s["inference_calls"] for s in S])[:-1]])

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n, T = len(S), len(STEPS)
    probs = np.empty((n, T, 8, 10, 11, 32), np.float16)
    ent = np.empty((n, T, 8, 10, 11), np.float32)
    ids = np.empty((n, T, 8, 10, 11, 4), np.uint8)
    for i in range(n):
        rows = off[i] + np.array(STEPS)
        probs[i] = z["hb_router_probs"].oindex[rows, :, :, :, :]
        ent[i] = np.asarray(z["hb_entropy"].oindex[rows, :, :, :], np.float32)
        ids[i] = z["hb_expert_ids"].oindex[rows, :, :, :, :]
        if i % 128 == 0:
            print("  read %d/%d" % (i, n), flush=True)
    P = probs.astype(np.float32)

    # ---- 1. sharpening along the denoise axis -----------------------------
    print("\n=== 1. entropy by denoise iteration (d=0 is pure noise) ===")
    print("        " + "  ".join("d%-4d" % d for d in range(10)) + "   d9-d0")
    for L in range(8):
        e = ent[:, :, L].mean((0, 1, 3))                   # over ep, step, suffix
        print("layer%2d " % HB_LAYER_IDX[L] + "  ".join("%.3f" % v for v in e)
              + "   %+.4f" % (e[9] - e[0]))
    e_all = ent.mean((0, 1, 2, 4))
    print("mean    " + "  ".join("%.3f" % v for v in e_all) + "   %+.4f" % (e_all[9] - e_all[0]))
    m = P.max(-1).mean((0, 1, 2, 4))
    print("top1 p  " + "  ".join("%.3f" % v for v in m) + "   %+.4f" % (m[9] - m[0]))

    print("\n  entropy by suffix token (0 = state token, 1..10 = action chunk):")
    es = ent.mean((0, 1, 2, 3))
    print("        " + "  ".join("s%-4d" % s for s in range(11)))
    print("        " + "  ".join("%.3f" % v for v in es))

    # ---- 2. how far does the choice move along d? -------------------------
    print("\n=== 2. movement along the denoise axis ===")
    ov_adj, ov_far, tv_adj, tv_far = [], [], [], []
    for L in range(8):
        a = ids[:, :, L]                                    # [n, T, 10, 11, 4]
        # set intersection as a membership matrix -- 32 experts fit in one axis,
        # so the overlap count is an AND and a sum rather than a python loop
        hot = np.zeros(a.shape[:-1] + (32,), bool)
        np.put_along_axis(hot, a.astype(np.int64), True, axis=-1)
        adj = (hot[:, :, :-1] & hot[:, :, 1:]).sum(-1).mean()
        far = (hot[:, :, 0:1] & hot[:, :, 9:10]).sum(-1).mean()
        p = P[:, :, L]
        tva = 0.5 * np.abs(p[:, :, :-1] - p[:, :, 1:]).sum(-1).mean()
        tvf = 0.5 * np.abs(p[:, :, 0] - p[:, :, 9]).sum(-1).mean()
        ov_adj.append(adj); ov_far.append(far); tv_adj.append(tva); tv_far.append(tvf)
        print("layer%2d  top4 shared d->d+1 %.2f/4   d0 vs d9 %.2f/4   "
              "TV(d,d+1) %.4f   TV(d0,d9) %.4f"
              % (HB_LAYER_IDX[L], adj, far, tva, tvf))
    # reference: two random top-4 sets out of 32 share 4*4/32 = 0.5
    print("  reference: two independent top-4 sets out of 32 share 0.50 experts")

    # ---- 3. where does the variance live? ---------------------------------
    print("\n=== 3. variance decomposition of the router distribution ===")
    # nested means over each axis of P [ep, step, layer, denoise, suffix, expert]
    tot = P.var((0, 1, 3, 4), ddof=0).mean(-1)              # per layer, over everything
    v_den = P.mean((4,)).var(3, ddof=0).mean((0, 1, 3))     # spread across d
    v_suf = P.mean((3,)).var(3, ddof=0).mean((0, 1, 3))     # spread across suffix
    v_step = P.mean((3, 4)).var(1, ddof=0).mean((0, 2))     # spread across control step
    v_ep = P.mean((1, 3, 4)).var(0, ddof=0).mean(-1)        # spread across episode
    print("layer   total      denoise    suffix     ctrl step  episode")
    for L in range(8):
        print("  %2d   %.3e  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%"
              % (HB_LAYER_IDX[L], tot[L], 100 * v_den[L] / tot[L], 100 * v_suf[L] / tot[L],
                 100 * v_step[L] / tot[L], 100 * v_ep[L] / tot[L]))

    # ---- 4. noise vs observation, measured at d=0 -------------------------
    print("\n=== 4. at d=0 the action tokens are pure noise: what moves the router? ===")
    d0 = P[:, :, :, 0].mean(3)                              # [n, T, layer, expert]
    for ti, t in enumerate(STEPS):
        within = np.mean([d0[scene == s, ti].var(0, ddof=1).mean()
                          for s in np.unique(scene)])
        between = np.array([d0[scene == s, ti].mean(0)
                            for s in np.unique(scene)]).var(0, ddof=1).mean()
        print("  control step %2d:  within-scene (noise) %.3e   "
              "between-scene (observation) %.3e   ratio %6.1fx"
              % (t, within, between, between / within))
    print("  note: at control step 0 the observation is the initial state itself,")
    print("        so between-scene there is the purest observation effect available")

    # ---- 5. does any single d carry the outcome? --------------------------
    print("\n=== 5. outcome AUC from one denoise iteration vs the average ===")
    from scipy.stats import rankdata
    mixed = [int(s) for s in np.unique(scene)
             if 0 < y[scene == s].sum() < (scene == s).sum()]

    def sauc(x):
        num = den = 0.0
        for s in mixed:
            m = scene == s
            n1, n0 = int(y[m].sum()), int((~y[m]).sum())
            r = rankdata(x[m])
            num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
            den += n1 * n0
        return num / den

    print("step   " + "  ".join("d%-4d" % d for d in range(10)) + "   mean(d)")
    for ti, t in enumerate(STEPS):
        row = [sauc(ent[:, ti, :, d].mean((1, 2))) for d in range(10)]
        print("%4d   " % t + "  ".join("%.3f" % v for v in row)
              + "   %.3f" % sauc(ent[:, ti].mean((1, 2, 3))))
    print("  (entropy averaged over layers and suffix; 0.5 = nothing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
