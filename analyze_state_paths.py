#!/usr/bin/env python3
"""Trajectories through the unsupervised routing state space.

The point of a state space is that it removes the control-step problem: two
rollouts can be in the same state at different times, so they are compared by the
path they take, not by what they were doing at step 17.

Descriptor is the state token's routing, 8 layers x 32 experts.  Chosen over the
gate heights because it is far more stable under resampling (ARI 0.967 vs 0.759)
even though its silhouette is slightly lower, and a state space that reshuffles
when you resample is not a state space.  Adding the ten action tokens was tried
and makes it worse on both counts -- they carry a fixed positional code, so they
dilute.

Everything is built without labels.  Labels appear only in the last section, and
next to a control that matters: the same analysis run on the gripper alone.  The
routing state is largely a gripper readout, so if the gripper reproduces every
result, the state space has added nothing.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
K = 6
# Every failure runs the full 52 control steps and the shortest success takes 35,
# so a dwell *fraction* has episode length in its denominator and separates the
# classes perfectly without any routing being involved -- state 1 sits at mean
# step 0.8 and scored AUC 0.999 that way.  Truncating every rollout to the window
# where all 512 are still running removes it.
WIN = 34
RNG = np.random.default_rng(0)


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


def perm_p(x, y, scene, n=2000):
    obs = abs(sauc(x, y, scene) - .5)
    hit = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(scene):
            i = np.flatnonzero(scene == s)
            yp[i] = y[RNG.permutation(i)]
        hit += abs(sauc(x, yp, scene) - .5) >= obs
    return (1 + hit) / (n + 1)


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    N = len(ep)
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    A = np.empty((N, 8 * 32), np.float32)
    for a in range(0, N, 2048):
        b = min(a + 2048, N)
        A[a:b] = np.asarray(z["hb_router_probs"][a:b, :, 0, 0, :],
                            np.float32).reshape(b - a, -1)
    Xs = (A - A.mean(0)) / (A.std(0) + 1e-9)
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(Xs)          # no labels

    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i, :n_rows[i]] = d[:n_rows[i]]
    step = np.concatenate([np.arange(k) for k in n_rows])
    gap = prop[ep, step, 6] - prop[ep, step, 7]
    # the control: the gripper alone, cut into K bins by quantile
    glab = np.digitize(gap, np.quantile(gap, np.linspace(0, 1, K + 1)[1:-1]))

    print("t08, %d rollouts.  状态空间用状态 token 路由，k=%d，无监督聚类" % (len(S), K))
    print("所有轨迹特征只用前 %d 个控制步（全部 512 局都还在跑），杜绝局长泄漏" % WIN)
    print("对照：把夹爪开合度按分位数切成同样 %d 档\n" % K)

    print("=== 1. 状态和夹爪的关系（还没看标签）===")
    print("  状态  占比    平均夹爪     平均控制步   与夹爪分档的一致性")
    from sklearn.metrics import adjusted_rand_score
    for c in range(K):
        m = lab == c
        print("   %d   %5.1f%%   %.4f       %5.1f" %
              (c, 100 * m.mean(), gap[m].mean(), step[m].mean()))
    print("  路由状态 vs 夹爪分档的 ARI = %.3f  (1.0 = 完全等价)"
          % adjusted_rand_score(lab, glab))

    print("\n=== 2. 每条 rollout 的轨迹特征（仍然无监督构造）===")
    feats = {}
    for name, L in (("路由状态", lab), ("夹爪分档", glab)):
        cut = [L[ep == e][:WIN] for e in range(len(S))]      # 等长窗口
        dwell = np.stack([np.bincount(c, minlength=K) / len(c) for c in cut])
        nvis = np.array([len(set(c.tolist())) for c in cut])
        path = [len([1 for i in range(1, len(c)) if c[i] != c[i-1]]) for c in cut]
        feats[name] = dict(dwell=dwell, nvis=nvis, switch=np.array(path))
        print("  %s: 每条访问 %.1f / %d 个状态，切换 %.1f 次"
              % (name, nvis.mean(), K, np.mean(path)))

    print("\n=== 3. 现在看标签：场景内分层 AUC + 场景内置换检验 ===")
    print("  特征                          路由状态           夹爪分档")
    rows = [("访问的不同状态数", "nvis"), ("状态切换次数", "switch")]
    for lbl, key in rows:
        a = sauc(feats["路由状态"][key].astype(float), y, scene)
        b = sauc(feats["夹爪分档"][key].astype(float), y, scene)
        pa = perm_p(feats["路由状态"][key].astype(float), y, scene)
        pb = perm_p(feats["夹爪分档"][key].astype(float), y, scene)
        print("  %-22s   %.3f (p=%.3f)   %.3f (p=%.3f)" % (lbl, a, pa, b, pb))
    for c in range(K):
        a = sauc(feats["路由状态"]["dwell"][:, c], y, scene)
        b = sauc(feats["夹爪分档"]["dwell"][:, c], y, scene)
        pa = perm_p(feats["路由状态"]["dwell"][:, c], y, scene)
        print("  在状态 %d 的停留比例      %.3f (p=%.3f)   %.3f" % (c, a, pa, b))

    print("\n=== 4. 有没有只被一类走过的路径？ ===")
    for name, L in (("路由状态", lab), ("夹爪分档", glab)):
        cnt = collections.Counter()
        for e in range(len(S)):
            s_ = L[ep == e][:WIN]
            p = tuple(s_[i] for i in range(len(s_)) if i == 0 or s_[i] != s_[i-1])
            cnt[p[:4]] += 1
        pure = 0
        tot = 0
        for p, n in cnt.items():
            if n < 8:
                continue
            eps = [e for e in range(len(S))
                   if tuple(x for i, x in enumerate(L[ep == e][:WIN])
                            if i == 0 or x != L[ep == e][:WIN][i-1])[:4] == p]
            r = y[eps].mean()
            tot += 1
            if r > .85 or r < .30:
                pure += 1
        print("  %s: 出现 >=8 次的前 4 步路径有 %d 种，其中成功率极端（>85%% 或 "
              "<30%%）的有 %d 种（全体 %.0f%%）" % (name, tot, pure, 100*y.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
