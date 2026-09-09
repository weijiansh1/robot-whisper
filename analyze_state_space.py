#!/usr/bin/env python3
"""An unsupervised MoE state space: what descriptor, and does it have structure?

Every earlier probe aligned on the control step and used the labels.  This does
neither.  The idea is that a rollout passes through a sequence of routing
*states*, that two rollouts can be in the same state at different control steps,
and that the states can be found without knowing which rollout succeeded.

The first thing to settle is what a "state" is, because a single frame of raw
routing is 84% noise against its own stable part.  Four candidates, judged on
whether they produce structure at all -- clusters that are stable under
resampling and separated at all, before any label is looked at:

    A  state token, all 8 HB layers   [8 x 32 = 256]
    B  state token, top-1 per layer   [8 symbols]
    C  the gate heights               [8 numbers]  -- the favourite's probability
    D  every token, all layers        [8 x 11 x 32 = 2816]

The state token needs no averaging over the denoise axis: its routing is
bit-identical across all ten iterations, so one control step already is one
vector.  The action tokens are a fixed positional code, so D mostly adds a
constant.

Labels are used at the very end and only to report what the clusters turned out
to be -- never to build them.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
RNG = np.random.default_rng(0)


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    N = len(ep)
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    print("t08: %d rollouts, %d (rollout, control step) pairs total" % (len(S), N))
    print("descriptors are built with no reference to the labels\n")

    # ---- build the four descriptors --------------------------------------
    CH = 2048
    A = np.empty((N, 8, 32), np.float32)
    D = np.empty((N, 8, 11, 32), np.float32)
    for a in range(0, N, CH):
        b = min(a + CH, N)
        P = np.asarray(z["hb_router_probs"][a:b, :, 0, :, :], np.float32)
        A[a:b] = P[:, :, 0]
        D[a:b] = P
    A2 = A.reshape(N, -1)
    B = A.argmax(-1)                                  # [N, 8] symbols
    Bh = np.zeros((N, 8 * 32), np.float32)
    for l in range(8):
        Bh[np.arange(N), l * 32 + B[:, l]] = 1
    C = A.max(-1)                                     # [N, 8]
    D2 = D.reshape(N, -1)

    desc = {"A 状态token 全分布 (256)": A2, "B 状态token top-1 (8符号)": Bh,
            "C 门高度 (8)": C, "D 全部token (2816)": D2}

    print("=== 1. 每个描述子有没有结构？（完全不看标签）===")
    print("描述子                    PCA 前2维占比  最佳k  轮廓系数  "
          "重采样稳定性(ARI)")
    best = None
    for name, X in desc.items():
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        pc = PCA(min(20, Xs.shape[1])).fit(Xs)
        sub = RNG.choice(N, 6000, replace=False)
        rows = []
        for k in (2, 3, 4, 6, 8):
            lab = KMeans(k, n_init=6, random_state=0).fit_predict(Xs[sub])
            rows.append((silhouette_score(Xs[sub], lab), k))
        sil, k = max(rows)
        # stability: cluster two disjoint halves, compare on a common held-out set
        h1, h2 = RNG.permutation(N)[:8000].reshape(2, -1)
        com = RNG.choice(N, 4000, replace=False)
        m1 = KMeans(k, n_init=6, random_state=1).fit(Xs[h1])
        m2 = KMeans(k, n_init=6, random_state=2).fit(Xs[h2])
        ari = adjusted_rand_score(m1.predict(Xs[com]), m2.predict(Xs[com]))
        print("  %-26s %5.1f%%       %d     %.3f     %.3f"
              % (name, 100 * pc.explained_variance_ratio_[:2].sum(), k, sil, ari))
        if best is None or sil > best[1]:
            best = (name, sil, Xs, k)

    name, sil, Xs, k = best
    print("\n  -> 用 %s，k=%d（轮廓系数最高）" % (name, k))

    # ---- 2. what are the states, described without labels -----------------
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
    print("\n=== 2. 这 %d 个状态是什么（仍然不看成败）===" % k)
    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i, :n_rows[i]] = d[:n_rows[i]]
    gap = prop[ep, step, 6] - prop[ep, step, 7]
    print("  状态   占比    平均控制步   夹爪开合度      每条 rollout 平均访问次数")
    for c in range(k):
        m = lab == c
        visits = np.mean([ (lab[ep == e] == c).sum() for e in range(len(S)) ])
        print("   %d    %5.1f%%   %6.1f      %.4f +- %.4f      %.1f"
              % (c, 100 * m.mean(), step[m].mean(), gap[m].mean(), gap[m].std(),
                 visits))

    # ---- 3. only now, the labels ------------------------------------------
    print("\n=== 3. 现在才看标签：这些状态和成败有关系吗 ===")
    print("  状态   成功局访问的比例   失败局访问的比例   差")
    for c in range(k):
        f_ok = np.mean([(lab[ep == e] == c).mean() for e in np.flatnonzero(y)])
        f_no = np.mean([(lab[ep == e] == c).mean() for e in np.flatnonzero(~y)])
        print("   %d      %6.3f            %6.3f          %+.3f"
              % (c, f_ok, f_no, f_ok - f_no))
    # is any state visited by essentially only one class?
    print("\n  有没有哪个状态几乎只被一类访问？")
    for c in range(k):
        eps = np.unique(ep[lab == c])
        r = y[eps].mean()
        print("   状态 %d 被 %3d 条 rollout 访问过，其中 %.1f%% 是成功局 "
              "（全体 %.1f%%）" % (c, len(eps), 100 * r, 100 * y.mean()))
    np.savez("/tmp/t08_states.npz", lab=lab, ep=ep, step=step, y=y, scene=scene,
             gap=gap, k=k)
    print("\n  saved /tmp/t08_states.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
