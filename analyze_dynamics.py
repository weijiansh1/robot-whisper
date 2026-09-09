#!/usr/bin/env python3
"""Basins, attractors and symbolic dynamics of the state token's routing.

Two prior results bound this question before it starts.  Density-based basins
were already tried and failed -- the densest routing region *preferentially
retains failures* on four bundles.  And routing was found near-memoryless across
control steps, excess MI 0.080 bit at lag 1 falling to ~0.012 by lag 3, which was
read as "a routing automaton has almost no edges worth estimating".

That second one was measured on the **action** tokens, and those turned out to
carry a fixed positional code with all the input dependence living on the single
state token instead.  The state token is a pose code, and pose is temporally
continuous, so its symbolic dynamics is a different object and has not been
measured.

The framing that keeps this honest: the state token's routing is a quantiser of
the arm's own 8 numbers, so its "dynamics" is the closed-loop policy's dynamics
seen through a partition.  Attractors and basins found in it are the *task's*,
not the router's -- unless the router's partition carries more of them than an
ordinary partition of the same resolution.  So every measurement below is run
twice, once on k-means cells of the routing and once on k-means cells of the
proprioception at matched K, and it is the gap that is the finding.

    part 1  contraction or divergence -- do same-scene rollouts converge, and
            does the same/different-outcome split show up in routing before it
            shows up in the physical state
    part 2  symbolic dynamics -- dwell time, determinism I(c_t; c_t+k), and how
            many recurrent classes the transition graph has, against a
            within-episode time-shuffle null
    part 3  are the basins irreversible -- once on the failing side, do
            trajectories come back, and is entering a cell absorbing in the
            Markov sense (does history still matter once you know the cell)

Window is the shortest episode, so all 512 rollouts are live throughout and
nothing here can be reading "still running" as "will fail".
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "himoe-routing-rules-20260819/data"
T08 = "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz"
KS = (16, 64, 256)
RNG = np.random.default_rng(0)


def read(path):
    d = np.load(path, allow_pickle=True)
    nr = d["n_rows"].astype(int)
    W = int(nr.min())
    off = np.concatenate([[0], np.cumsum(nr)[:-1]])
    take = (off[:, None] + np.arange(W)[None, :]).ravel()
    n = len(nr)
    ids = d["state_token_top4"][take].astype(np.int64).reshape(n, W, 8, 4)
    hot = np.zeros((n, W, 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    return dict(task=str(d["task"]), n=n, W=W, y=d["success"], scene=d["scene"],
                hot=hot, prop=d["proprio"][take].reshape(n, W, 8),
                probs=d["state_token_probs"][take].astype(np.float32).reshape(n, W, 256))


def cells(X, K, n, W):
    """K-means labels over all (episode, step) rows, shaped back to (n, W)."""
    km = MiniBatchKMeans(K, random_state=0, n_init=10, batch_size=2048).fit(X)
    return km.labels_.reshape(n, W)


def dwell(c):
    """Mean run length of a symbol along the control-step axis."""
    runs = []
    for row in c:
        ch = np.flatnonzero(np.diff(row)) + 1
        runs.extend(np.diff(np.concatenate([[0], ch, [len(row)]])))
    return float(np.mean(runs))


def determinism(c, k):
    a, b = c[:, :-k].ravel(), c[:, k:].ravel()
    return mutual_info_score(a, b) / np.log(2)


def recurrent(c, K, thr=3):
    """Strongly connected components of the observed transition graph."""
    a, b = c[:, :-1].ravel(), c[:, 1:].ravel()
    m = coo_matrix((np.ones(len(a)), (a, b)), shape=(K, K)).tocsr()
    m.data = (m.data >= thr).astype(float)
    m.eliminate_zeros()
    ncc, lab = connected_components(m, directed=True, connection="strong")
    sz = np.bincount(lab, minlength=ncc)
    # terminal = component with no outgoing edge to another component
    out = np.zeros(ncc, bool)
    src, dst = m.nonzero()
    for s, t in zip(lab[src], lab[dst]):
        if s != t:
            out[s] = True
    return ncc, int((sz > 1).sum()), int((~out).sum())


def main() -> int:
    D = read(DATA / (sys.argv[1] if len(sys.argv) > 1 else T08))
    n, W, y, scene = D["n"], D["W"], D["y"], D["scene"]
    fail = ~y
    print("%s\n%d 局 x %d 控制步（窗口=最短局，全员在场），成功率 %.1f%%\n"
          % (D["task"], n, W, 100 * y.mean()))

    # ---------- part 1 : contraction / divergence -----------------------------
    P = (D["prop"] - D["prop"].reshape(-1, 8).mean(0)) / \
        (D["prop"].reshape(-1, 8).std(0) + 1e-9)
    H = D["hot"]
    print("=== 1. 同场景两条 rollout 之间的距离随控制步怎么变 ===")
    print("  （路由距离 = 8 层 top-4 集合的平均 Jaccard 距离；位姿距离 = 8 维标准化欧氏）")
    print("  步    路由:同结局  异结局   差      位姿:同结局  异结局   差")
    same_r, diff_r, same_p, diff_p = {}, {}, {}, {}
    for s in np.unique(scene):
        m = np.flatnonzero(scene == s)
        if len(m) < 4:
            continue
        i, j = np.triu_indices(len(m), 1)
        a, b = m[i], m[j]
        so = y[a] == y[b]
        inter = (H[a] * H[b]).sum(-1)
        dr = 1.0 - inter / (8.0 - inter)          # |A n B| / |A u B|, |A|=|B|=4
        dr = dr.mean(-1)                          # over the 8 layers
        dp = np.linalg.norm(P[a] - P[b], axis=-1)
        for t in range(W):
            same_r.setdefault(t, []).extend(dr[so, t])
            diff_r.setdefault(t, []).extend(dr[~so, t])
            same_p.setdefault(t, []).extend(dp[so, t])
            diff_p.setdefault(t, []).extend(dp[~so, t])
    first_r = first_p = None
    for t in range(W):
        sr, dr_, sp, dp_ = (np.mean(same_r[t]), np.mean(diff_r[t]),
                            np.mean(same_p[t]), np.mean(diff_p[t]))
        gr, gp = dr_ - sr, dp_ - sp
        # first step at which the outcome split exceeds 10% of the same-outcome level
        if first_r is None and gr > 0.10 * sr:
            first_r = t
        if first_p is None and gp > 0.10 * sp:
            first_p = t
        if t % 4 == 0 or t == W - 1:
            print("  %3d      %.4f     %.4f  %+.4f      %.3f      %.3f  %+.3f"
                  % (t, sr, dr_, gr, sp, dp_, gp))
    print("  同结局与异结局分开（超过同结局水平的 10%%）：路由第 %s 步，位姿第 %s 步"
          % (first_r, first_p))
    d0 = np.mean(same_r[0]) + np.mean(diff_r[0])
    dE = np.mean(same_r[W - 1]) + np.mean(diff_r[W - 1])
    print("  路由总距离 首步 %.4f -> 末步 %.4f （%s）"
          % (d0 / 2, dE / 2, "收缩" if dE < d0 else "发散"))
    d0p = (np.mean(same_p[0]) + np.mean(diff_p[0])) / 2
    dEp = (np.mean(same_p[W - 1]) + np.mean(diff_p[W - 1])) / 2
    print("  位姿总距离 首步 %.3f -> 末步 %.3f （%s）"
          % (d0p, dEp, "收缩" if dEp < d0p else "发散"))

    # ---------- part 2 : symbolic dynamics, routing vs matched pose ----------
    print("\n=== 2. 符号动力学：路由划分 vs 同分辨率位姿划分 ===")
    print("  K    划分      停留步长  I(c_t;c_t+1)  +2      +5    强连通分量  非平凡  终止类")
    CELLS = {}
    for K in KS:
        for nm, X in (("路由", D["probs"].reshape(-1, 256)),
                      ("位姿", P.reshape(-1, 8))):
            c = cells(X, K, n, W)
            CELLS[(K, nm)] = c
            cs = c.copy()
            for r in range(n):
                cs[r] = c[r][RNG.permutation(W)]
            ncc, nt, term = recurrent(c, K)
            print("  %3d  %s      %5.2f      %.3f     %.3f   %.3f     %4d      %3d     %3d"
                  % (K, nm, dwell(c), determinism(c, 1), determinism(c, 2),
                     determinism(c, 5), ncc, nt, term))
        c = CELLS[(K, "路由")]
        cs = np.array([c[r][RNG.permutation(W)] for r in range(n)])
        print("  %3d  打乱时序 %5.2f      %.3f     %.3f   %.3f"
              % (K, dwell(cs), determinism(cs, 1), determinism(cs, 2),
                 determinism(cs, 5)))

    # ---------- part 3 : are the basins irreversible? ------------------------
    print("\n=== 3. 盆地是否不可逆 ===")
    print("  给每个格子按「进入它的局里有多少失败」定边（场景内比较），"
          "再问越过之后还回不回得来")
    print("  K    划分   纯失败格数/有支持格数  被它们吸收的局  其中真失败  返回率  场景内置换上限")
    for K in KS:
        for nm in ("路由", "位姿"):
            c = CELLS[(K, nm)]
            ent = np.zeros((n, K), bool)
            for r in range(n):
                ent[r, np.unique(c[r])] = True
            sup = ent.sum(0)
            pf = np.array([fail[ent[:, k]].mean() if sup[k] else np.nan
                           for k in range(K)])
            good = sup >= 20
            pure = good & (pf == 1.0)
            hit = ent[:, pure].any(1) if pure.any() else np.zeros(n, bool)
            # permutation ceiling: shuffle outcome inside each scene
            best = []
            for _ in range(200):
                fp = fail.copy()
                for s in np.unique(scene):
                    m = np.flatnonzero(scene == s)
                    fp[m] = fail[m][RNG.permutation(len(m))]
                q = np.array([fp[ent[:, k]].mean() if sup[k] else np.nan
                              for k in range(K)])
                best.append(int((good & (q == 1.0)).sum()))
            # reversibility: after first entry into a pure-failure cell, does the
            # episode ever return to a cell whose failure rate is below 1/2?
            back = np.nan
            if pure.any():
                lo = good & (pf < 0.5)
                ret = []
                for r in np.flatnonzero(hit):
                    t0 = np.min([np.argmax(c[r] == k) for k in np.flatnonzero(pure)
                                 if (c[r] == k).any()])
                    ret.append(bool(np.isin(c[r][t0 + 1:], np.flatnonzero(lo)).any()))
                back = float(np.mean(ret)) if ret else np.nan
            print("  %3d  %s      %3d / %3d           %3d        %5.1f%%   %5.1f%%    %d"
                  % (K, nm, int(pure.sum()), int(good.sum()), int(hit.sum()),
                     100 * fail[hit].mean() if hit.any() else 0.0,
                     100 * back if back == back else float("nan"),
                     int(np.quantile(best, 0.95))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
