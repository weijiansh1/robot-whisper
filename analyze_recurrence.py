#!/usr/bin/env python3
"""The routing state as a continuum: trajectory shape and cross-recurrence.

The k-means state space had a silhouette of 0.21, i.e. no natural clusters, so k
was a choice of mine and "number of state switches" -- the strongest feature at
0.760 -- inherited that choice.  This drops the discretisation entirely.

Two parameter-light descriptions of the same 256-dim state token trajectory:

  1. shape.  Path length, spread, and mean step size are the continuous analogues
     of switch count and dwell, with no k in them.

  2. cross-recurrence.  This is the direct form of the original idea -- "several
     rollouts activate the same MoE state".  For two rollouts A and B, take every
     pair (a step of A, a step of B) and ask how often the two states are within
     eps of each other.  No time alignment: a step 8 of A may match a step 21 of
     B.  Then the question becomes whether two rollouts that both succeed recur
     with each other more than a success does with a failure.

Everything inside one scene, so the initial state is bit-identical and only the
noise seed differs; and inside the first 34 control steps, where all 512 rollouts
are still running, so episode length cannot leak.  Labels are used only to group
the pairs at the end, and a scene-internal permutation gives the null.  The
physical state runs alongside as the control throughout.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
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
    nS = len(S)
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    R = np.empty((nS, WIN, 8 * 32), np.float32)
    for i in range(nS):
        R[i] = np.asarray(z["hb_router_probs"].oindex[
            off[i] + np.arange(WIN), :, 0, 0, :], np.float32).reshape(WIN, -1)
    d0 = np.load(RUN / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    P = np.empty((nS, WIN, d0.shape[1]), np.float32)
    for i, s in enumerate(S):
        P[i] = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                       allow_pickle=True)["sim_state"][:WIN]
    # standardise each channel on the pooled data so distances are comparable
    for X in (R, P):
        X -= X.reshape(-1, X.shape[-1]).mean(0)
        X /= X.reshape(-1, X.shape[-1]).std(0) + 1e-9

    print("t08, %d rollouts, 前 %d 个控制步；场景内比较，无时间对齐\n" % (nS, WIN))

    print("=== 1. 轨迹形状（连续，没有 k）===")
    print("  特征                        路由状态            物理状态")
    for lbl, fn in (("路径长度 Σ‖Δx‖", lambda X: np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)),
                    ("平均步长", lambda X: np.linalg.norm(np.diff(X, axis=1), axis=2).mean(1)),
                    ("散布 Σ‖x-x̄‖", lambda X: np.linalg.norm(X - X.mean(1, keepdims=True), axis=2).mean(1)),
                    ("端点距离 ‖x_T-x_0‖", lambda X: np.linalg.norm(X[:, -1] - X[:, 0], axis=1)),
                    ("弯曲度 路径长/端点距", lambda X: (np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)
                                                 / (np.linalg.norm(X[:, -1] - X[:, 0], axis=1) + 1e-9)))):
        a, b = fn(R), fn(P)
        print("  %-24s  %.3f (p=%.3f)   %.3f (p=%.3f)"
              % (lbl, sauc(a, y, scene), perm_p(a, y, scene),
                 sauc(b, y, scene), perm_p(b, y, scene)))

    print("\n=== 2. 交叉递归：两条 rollout 有没有走过同一个状态 ===")
    print("  eps 取场景内所有点对距离的 5% 分位数；CRR = 落在 eps 内的 (a步, b步) 对的比例")
    mixed = [int(s) for s in np.unique(scene) if 4 <= y[scene == s].sum() <= 28]
    for name, X in (("路由状态", R), ("物理状态", P)):
        ss, sf, ff, per_scene = [], [], [], []
        for s in mixed:
            idx = np.flatnonzero(scene == s)
            F = X[idx].reshape(len(idx) * WIN, -1)
            D = np.linalg.norm(F[:, None] - F[None], axis=2)
            eps = np.quantile(D[np.triu_indices(len(F), 1)], 0.05)
            B = (D < eps).reshape(len(idx), WIN, len(idx), WIN).mean((1, 3))
            yy = y[idx]
            iu = np.triu_indices(len(idx), 1)
            same_ok = yy[iu[0]] & yy[iu[1]]
            same_no = (~yy[iu[0]]) & (~yy[iu[1]])
            diff = yy[iu[0]] ^ yy[iu[1]]
            v = B[iu]
            ss.append(v[same_ok].mean()); ff.append(v[same_no].mean())
            sf.append(v[diff].mean())
            # per-rollout: mean CRR with the successes of its own scene
            per_scene.append((idx, B, yy))
        print("  %s:  成功-成功 %.4f   失败-失败 %.4f   成功-失败 %.4f"
              % (name, np.mean(ss), np.mean(ff), np.mean(sf)))
        print("           同类比异类高 %+.4f  （成功-成功 与 成功-失败 之差 %+.4f）"
              % ((np.mean(ss) + np.mean(ff)) / 2 - np.mean(sf),
                 np.mean(ss) - np.mean(sf)))
        # turn it into a per-rollout score, leave-one-out so a rollout never
        # recurs with itself, and test it like any other feature
        score = np.full(nS, np.nan)
        for idx, B, yy in per_scene:
            for a in range(len(idx)):
                o = [b for b in range(len(idx)) if b != a and yy[b]]
                score[idx[a]] = B[a, o].mean()
        m = np.isfinite(score)
        print("           与本场景其他成功局的平均 CRR，作为单特征: "
              "AUC %.3f (p=%.3f)"
              % (sauc(score[m], y[m], scene[m]), perm_p(score[m], y[m], scene[m])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
