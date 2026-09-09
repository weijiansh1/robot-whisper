#!/usr/bin/env python3
"""The flicker looks random.  How much of it is?

Watching the viewer, the lit cells jump around with no visible pattern.  That is
a real observation and most of it is correct, but "random" can mean three
different things and they are separable:

  1. memoryless -- the next draw is independent of this one
  2. uniform    -- every expert is equally likely at a site
  3. meaningless -- nothing about the world changes it

A process can be memoryless and still be far from uniform, and can be uniform in
aggregate while being sharply structured per token.  So:

  A. is it memoryless?  Compare the top-4 overlap between consecutive draws
     against the overlap between two draws of the same site picked at random from
     the whole episode, which is what an IID process would give.  Done at both
     scales the viewer shows: consecutive denoise iterations, and consecutive
     control steps.
  B. is it uniform?  Per site, how far the long-run selection frequency sits
     from 4/32, and how many frames of averaging it takes to see that.
  C. is it meaningless?  How much of the per-site distribution is set by which
     token it is, and how much by the gripper being open or closed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
RNG = np.random.default_rng(11)


def hot(ids):
    h = np.zeros(ids.shape[:-1] + (32,), bool)
    np.put_along_axis(h, ids.astype(np.int64), True, -1)
    return h


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    # one long episode, so the per-site sequences are long enough to characterise
    i = int(np.argmax(np.where([s["success"] for s in S], n_rows, 0)))
    T = int(n_rows[i])
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    ids = np.asarray(z["hb_expert_ids"][off[i]:off[i] + T])      # [T,8,10,11,4]
    P = np.asarray(z["hb_router_probs"][off[i]:off[i] + T, :, :, :, :], np.float32)
    d = np.load(RUN / ("client/episode_%02d.npz" % S[i]["episode_index"]),
                allow_pickle=True)
    gap = (d["state"][:T, 6] - d["state"][:T, 7]).astype(np.float32)
    print("episode %d, %d control steps, %d sites per step\n"
          % (S[i]["episode_index"], T, 8 * 10 * 11))

    H = hot(ids)                                   # [T,8,10,11,32]

    print("=== A. is the flicker memoryless? ===")
    print("  (top-4 experts shared out of 4; 'IID' = two draws of the SAME site")
    print("   taken at random from the whole episode, which is what a memoryless")
    print("   process with that site's own distribution would give)")
    print("  block        consecutive denoise   consecutive control step   IID")
    for tag, sl in (("front 2-5", slice(0, 4)), ("back 12-15", slice(4, 8))):
        Q = H[:, sl]
        lag_d = (Q[:, :, :-1] & Q[:, :, 1:]).sum(-1).mean()
        lag_t = (Q[:-1] & Q[1:]).sum(-1).mean()
        a = RNG.integers(0, T, 4000)
        b = RNG.integers(0, T, 4000)
        iid = (Q[a] & Q[b]).sum(-1).mean()
        print("  %-11s      %.2f                  %.2f                %.2f"
              % (tag, lag_d, lag_t, iid))

    print("\n=== B. is it uniform? ===")
    print("  per site, the long-run chance an expert is in the top-4 "
          "(uniform = 0.125)")
    print("  block        top expert   4th        16th       32nd   "
          "effective experts")
    for tag, sl in (("front 2-5", slice(0, 4)), ("back 12-15", slice(4, 8))):
        f = H[:, sl].mean((0, 2))                  # [layers, token, 32]
        s = np.sort(f, -1)[..., ::-1]
        p = f / f.sum(-1, keepdims=True)
        eff = np.exp(-(p * np.log(np.where(p > 0, p, 1))).sum(-1))
        print("  %-11s  %.3f       %.3f      %.3f      %.3f      %.1f / 32"
              % (tag, s[..., 0].mean(), s[..., 3].mean(), s[..., 15].mean(),
                 s[..., 31].mean(), eff.mean()))

    print("\n  how many frames of averaging before the skeleton is visible?")
    print("  (correlation between the frequency estimated from n frames and the")
    print("   whole-episode one, front block)")
    full = H[:, :4].reshape(-1, 4, 11, 32).mean(0)
    flat = H[:, :4].reshape(T * 10, 4, 11, 32)
    for n in (1, 2, 5, 10, 30, 80, 200):
        if n > len(flat):
            continue
        r = []
        for _ in range(40):
            k = RNG.integers(0, len(flat) - n + 1)
            est = flat[k:k + n].mean(0)
            r.append(np.corrcoef(est.ravel(), full.ravel())[0, 1])
        print("     %3d 帧 (%.1f 个控制步): r = %.3f" % (n, n / 10, np.mean(r)))

    print("\n=== C. what sets the per-site distribution? ===")
    on = P[:, 3, 0, 0].argmax(-1)                  # not used; kept for clarity
    fav = P[:, 3, :, 0, :].mean((0, 1)).argmax()
    gate = P[:, 3, 0, 0, fav] > 0.5
    print("  the L5 gate is open on %d of %d control steps" % (gate.sum(), T))
    for tag, sl in (("front 2-5", slice(0, 4)), ("back 12-15", slice(4, 8))):
        f = H[:, sl].mean((0, 2))                  # [L, token, 32]
        # how much does the token you are explain?
        gm = f.mean(1, keepdims=True)
        ss_tok = ((f - gm) ** 2).sum()
        ss_tot = ((f - f.mean((1, 2), keepdims=True)) ** 2).sum()
        # and the gripper, on the state token only
        st_on = H[gate][:, sl][:, :, :, 0].mean((0, 2))
        st_off = H[~gate][:, sl][:, :, :, 0].mean((0, 2))
        # these vectors sum to 4 (four experts selected each draw), so the
        # disjoint upper bound is 8; normalise to the usual 0-2 scale
        dg = (np.abs(st_on - st_off).sum(-1) / 4).mean()
        print("  %-11s  token 身份解释了 %4.1f%% 的位点间差异   "
              "夹爪开/关让状态 token 的分布移动 L1 = %.3f / 2.0"
              % (tag, 100 * ss_tok / ss_tot, dg))
    print("  (归一化到和为 1 之后；L1 = 2.0 表示两个分布完全不重叠)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
