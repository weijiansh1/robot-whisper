#!/usr/bin/env python3
"""Only the top-1 expert: is it better resolved, and does it split the outcome?

The top-4 boundary is decided by nothing -- 97.8% of sites have the 4th and 5th
expert within one bf16 ulp of each other, and 27.3% are exactly tied -- so the
*set* is mostly arithmetic noise.  The winner is a different question: it has to
beat 31 others, not just the runner-up at the cut, so its margin may be real
where the cut's is not.  If so, top-1 could carry outcome signal that every
top-4-based summary has been throwing away.

  A. is the winner better resolved than the cut?
  B. is it more stable in time and more concentrated per site?
  C. within a scene, does it separate the rollouts that succeed from the ones
     that fail -- as an identity, as a probability, and as a margin?
  D. one concrete scene and site, spelled out.

Everything scene-stratified and at a fixed control step, because scene difficulty
runs from 0/32 to 32/32 and every failure runs the full horizon.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
STEPS = [0, 4, 8, 13, 20, 26, 34]
BF16_EPS = 2.0 ** -8


def sauc(x, y, scene):
    num = den = 0.0
    for s in np.unique(scene):
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(x[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else float("nan")


def loso(X, y, scene):
    num = den = 0.0
    for held in np.unique(scene):
        tr, te = scene != held, scene == held
        if not (0 < y[tr].sum() < tr.sum()) or not (0 < y[te].sum() < te.sum()):
            continue
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s = f.decision_function(sc.transform(X[te]))
        n1 = int(y[te].sum())
        r = rankdata(s)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * int((~y[te]).sum())
    return num / den if den else float("nan")


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    mixed = [s for s in np.unique(scene)
             if 0 < y[scene == s].sum() < (scene == s).sum()]
    keep = np.isin(scene, mixed)
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    # ---- A/B on a sample of sites ---------------------------------------
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(z["hb_router_probs"].shape[0], 1200, replace=False))
    P = np.asarray(z["hb_router_probs"].oindex[rows], np.float32)   # [n,8,10,11,32]
    srt = np.sort(P, -1)[..., ::-1]
    d12, d45 = srt[..., 0] - srt[..., 1], srt[..., 3] - srt[..., 4]
    print("=== A. the winner's margin vs the cut's ===")
    print("            median      p10        frac < bf16 ulp   frac exactly 0")
    for tag, d in (("1st - 2nd", d12), ("4th - 5th", d45)):
        print("  %-10s %.6f   %.6f      %5.1f%%           %5.1f%%"
              % (tag, np.median(d), np.percentile(d, 10),
                 100 * (d < BF16_EPS).mean(), 100 * (d == 0).mean()))
    print("  -> the winner is decided by a margin %.0fx the cut's"
          % (np.median(d12) / np.median(d45)))

    print("\n=== B. is the winner more stable? ===")
    n_ep = 200
    a = off[:n_ep]
    Q = np.asarray(z["hb_router_probs"].oindex[
        np.sort(np.concatenate([a + t for t in (4, 5)])), :, 0, :, :], np.float32)
    Q = Q.reshape(-1, 2, 8, 11, 32) if Q.shape[0] == 2 * n_ep else None
    # simpler: consecutive control steps of the same episode
    t0 = np.asarray(z["hb_router_probs"].oindex[off[:n_ep] + 4, :, 0, :, :], np.float32)
    t1 = np.asarray(z["hb_router_probs"].oindex[off[:n_ep] + 5, :, 0, :, :], np.float32)
    same1 = (t0.argmax(-1) == t1.argmax(-1)).mean()
    s0 = np.sort(np.argsort(-t0, -1)[..., :4], -1)
    s1 = np.sort(np.argsort(-t1, -1)[..., :4], -1)
    same4 = (s0 == s1).all(-1).mean()
    ov4 = np.mean([len(np.intersect1d(s0[i, l, k], s1[i, l, k]))
                   for i in range(0, n_ep, 4) for l in range(8) for k in range(11)])
    print("  from control step 4 to 5, at the same site:")
    print("     the top-1 expert is unchanged        %5.1f%%   (chance 3.1%%)" % (100 * same1))
    print("     the whole top-4 set is unchanged     %5.1f%%" % (100 * same4))
    print("     top-4 experts shared                 %.2f / 4" % ov4)

    # ---- C. outcome -----------------------------------------------------
    print("\n=== C. within a scene, does the winner separate success from failure? ===")
    print("  %d mixed scenes, %d episodes; scene-stratified AUC at a fixed step"
          % (len(mixed), keep.sum()))
    print("  step   top-1 id(one-hot)  top-1 prob   1st-2nd margin   "
          "full 32-dim probs")
    for t in STEPS:
        idx = np.flatnonzero(keep)
        R = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, 0, :],
                       np.float32)                       # state token [n,8,32]
        arg = R.argmax(-1)                               # [n,8]
        oh = np.zeros((len(idx), 8 * 32), np.float32)
        for l in range(8):
            oh[np.arange(len(idx)), l * 32 + arg[:, l]] = 1
        sr = np.sort(R, -1)[..., ::-1]
        print("  %4d      %.3f            %.3f        %.3f            %.3f"
              % (t, loso(oh, y[idx], scene[idx]),
                 sauc(sr[..., 0].mean(1), y[idx], scene[idx]),
                 sauc((sr[..., 0] - sr[..., 1]).mean(1), y[idx], scene[idx]),
                 loso(R.reshape(len(idx), -1), y[idx], scene[idx])))

    # ---- D. one scene, spelled out --------------------------------------
    print("\n=== D. one scene, one site, spelled out ===")
    sc = max(mixed, key=lambda s: min(int(y[scene == s].sum()),
                                      int((~y[scene == s]).sum())))
    idx = np.flatnonzero(scene == sc)
    t = 26
    R = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, 3, 0, 0, :], np.float32)
    arg = R.argmax(-1)
    ok = y[idx]
    print("  场景 %d，控制步 %d，L5 状态 token：%d 成功 / %d 失败"
          % (sc, t, ok.sum(), (~ok).sum()))
    for lab, m in (("成功", ok), ("失败", ~ok)):
        u, c = np.unique(arg[m], return_counts=True)
        o = np.argsort(-c)
        print("     %s 的 top-1 专家: %s"
              % (lab, "  ".join("e%d×%d" % (u[k], c[k]) for k in o[:6])))
    print("     两组用到的专家集合是否重叠: %s"
          % (set(arg[ok].tolist()) & set(arg[~ok].tolist()) or "不重叠"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
