#!/usr/bin/env python3
"""One scene, its 32 rollouts: does routing separate the successes?

This is the cleanest version of the question.  Inside a scene the initial state
and the prompt are bit-identical and only the flow-noise seed differs, so scene
difficulty -- which ran from 0/32 to 32/32 and forced every earlier comparison to
be stratified -- simply is not present.  What is left is n = 32.

At 14 successes against 18 failures there are 252 discordant pairs and the AUC's
null standard error is about 0.10, so a two-sided 95% band is roughly
[0.30, 0.70].  Anything inside that band is indistinguishable from nothing no
matter how suggestive it looks, and that has to be printed next to every number
or the table invites over-reading.  A within-scene label permutation gives the
band without assuming normality.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
STEPS = [0, 4, 8, 13, 18, 24, 30, 34]
NPERM = 2000
RNG = np.random.default_rng(4)


def auc(x, y):
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cv_auc(X, y, folds=4):
    """Cross-validated probe score -- with 32 samples a fitted probe would
    otherwise just memorise them."""
    if min(int(y.sum()), int((~y).sum())) < folds:
        return float("nan")
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        s[te] = f.decision_function(sc.transform(X[te]))
    return auc(s, y)


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    d0 = np.load(RUN / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.full((len(S), n_rows.max(), d0.shape[1]), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["sim_state"]
        sim[i, :n_rows[i]] = d[:n_rows[i]]

    mixed = [int(s) for s in np.unique(scene)
             if 4 <= y[scene == s].sum() <= 28]
    print("t08 的 16 个场景里，成败混合到够做统计的有 %d 个" % len(mixed))
    print("每个场景 32 条 rollout，初始状态逐位相同，只差流噪声种子\n")

    # the null band, from permuting labels inside one scene
    ref = max(mixed, key=lambda s: min(int(y[scene == s].sum()),
                                       int((~y[scene == s]).sum())))
    yy = y[scene == ref]
    null = []
    for _ in range(NPERM):
        null.append(auc(RNG.permutation(np.arange(32).astype(float)), yy))
    lo, hi = np.percentile(null, [2.5, 97.5])
    print("零假设带（场景 %d，%d 成功 / %d 失败，%d 次标签置换）: AUC 落在 "
          "[%.2f, %.2f] 之内就是「什么都没有」\n" % (ref, yy.sum(), (~yy).sum(),
                                                    NPERM, lo, hi))

    print("每个场景，单独看它的 32 条：状态 token 路由的 AUC（交叉验证）")
    print("场景  成功   " + "  ".join("步%-3d" % t for t in STEPS))
    hits = 0
    for s in mixed:
        idx = np.flatnonzero(scene == s)
        row = []
        for t in STEPS:
            R = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, 0, :],
                           np.float32).reshape(len(idx), -1)
            a = cv_auc(R, y[idx])
            row.append(a)
            if a < lo or a > hi:
                hits += 1
        print(" s%-3d %2d/32  " % (s, y[idx].sum())
              + "  ".join(("%5.2f%s" % (v, "*" if (v < lo or v > hi) else " "))
                          for v in row))
    print("  * = 落在零假设带之外。共 %d / %d 个格子越界，"
          "纯随机下期望约 %.1f 个" % (hits, len(mixed) * len(STEPS),
                                      0.05 * len(mixed) * len(STEPS)))

    print("\n同样的表，换成 47 维物理状态：")
    print("场景  成功   " + "  ".join("步%-3d" % t for t in STEPS))
    hits2 = 0
    for s in mixed:
        idx = np.flatnonzero(scene == s)
        row = []
        for t in STEPS:
            a = cv_auc(sim[idx, t], y[idx])
            row.append(a)
            if a < lo or a > hi:
                hits2 += 1
        print(" s%-3d %2d/32  " % (s, y[idx].sum())
              + "  ".join(("%5.2f%s" % (v, "*" if (v < lo or v > hi) else " "))
                          for v in row))
    print("  * 共 %d / %d 个格子越界" % (hits2, len(mixed) * len(STEPS)))

    print("\n把这些场景合起来看（这才是有功效的做法）:")
    keep = np.isin(scene, mixed)
    for t in STEPS:
        idx = np.flatnonzero(keep)
        R = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, 0, :],
                       np.float32).reshape(len(idx), -1)
        num = den = 0.0
        for s in mixed:                       # leave-one-scene-out, pooled
            tr, te = scene[idx] != s, scene[idx] == s
            sc = StandardScaler().fit(R[tr])
            f = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(R[tr]),
                                                             y[idx][tr])
            v = f.decision_function(sc.transform(R[te]))
            n1 = int(y[idx][te].sum())
            r = rankdata(v)
            num += r[y[idx][te]].sum() - n1 * (n1 + 1) / 2.0
            den += n1 * int((~y[idx][te]).sum())
        print("  步 %-3d  合并 %d 局，AUC = %.3f" % (t, keep.sum(), num / den))
    return 0


if __name__ == "__main__":
    sys.exit(main())
