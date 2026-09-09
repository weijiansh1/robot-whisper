#!/usr/bin/env python3
"""Dynamics of the routing, take two: tested separation, trapping, irreversibility.

The first pass established the part that needs no defending -- the symbolic
dynamics is strong (I(c_t;c_t+1) = 2.2-5.9 bit against a time-shuffle null of
0.06-2.3) but the routing partition is *uniformly weaker* than k-means on the raw
proprioception at matched K, so the router has not learned a dynamics-adapted
quantisation.  It also reported "routing separates outcomes at step 3, pose at
step 11", which is not reportable: the criterion was relative to a distance that
is still near zero at step 3, and the relative gap is smaller at step 8 than at
step 4, so it is non-monotone and untested.

Redone here:

  A  expansion or contraction, with the local rate ln d(t+1)/d(t) so a
     contracting phase would show up even though the endpoints diverge, and the
     same/different-outcome gap tested against outcome permuted inside each
     scene, max-statistic over steps so the "first significant step" is honest.

  B  trapping, which is the attractor story that would actually explain failure:
     an episode that cannot finish because it is stuck in a cycle should visit
     fewer distinct cells, revisit them more, and dwell longer.  Per episode,
     scene-adjusted, routing against a matched pose partition.

  C  irreversibility, with the pure-failure cell count scored against its own
     within-scene permutation ceiling and against the escape rate -- a basin you
     leave half the time is not a basin.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_dynamics import DATA, T08, cells, read  # noqa: E402

N_PERM = 300
RNG = np.random.default_rng(0)


def scene_z(v, fail, scene, n_perm=N_PERM, rng=None):
    """Episode-level statistic v: z of the fail-minus-succeed gap, scene-adjusted."""
    rng = rng or np.random.default_rng(1)

    def gap(f):
        g = []
        for s in np.unique(scene):
            m = scene == s
            if 0 < f[m].sum() < m.sum():
                g.append(v[m & f].mean() - v[m & ~f].mean())
        return np.mean(g) if g else np.nan

    obs = gap(fail)
    null = []
    for _ in range(n_perm):
        fp = fail.copy()
        for s in np.unique(scene):
            m = np.flatnonzero(scene == s)
            fp[m] = fail[m][rng.permutation(len(m))]
        null.append(gap(fp))
    null = np.array(null)
    return obs, (obs - np.nanmean(null)) / (np.nanstd(null) + 1e-12)


def main() -> int:
    D = read(DATA / (sys.argv[1] if len(sys.argv) > 1 else T08))
    n, W, y, scene = D["n"], D["W"], D["y"], D["scene"]
    fail = ~y
    P = (D["prop"] - D["prop"].reshape(-1, 8).mean(0)) / \
        (D["prop"].reshape(-1, 8).std(0) + 1e-9)
    H = D["hot"]
    print("%s\n%d 局 x %d 控制步，成功率 %.1f%%\n"
          % (D["task"], n, W, 100 * y.mean()))

    # ---- A : expansion rate, and a tested outcome split ---------------------
    DR, DP, SO, SC = [], [], [], []
    for s in np.unique(scene):
        m = np.flatnonzero(scene == s)
        i, j = np.triu_indices(len(m), 1)
        a, b = m[i], m[j]
        inter = (H[a] * H[b]).sum(-1)
        DR.append((1.0 - inter / (8.0 - inter)).mean(-1))
        DP.append(np.linalg.norm(P[a] - P[b], axis=-1))
        SO.append(y[a] == y[b])
        SC.append(np.full(len(a), s))
    DR, DP = np.vstack(DR), np.vstack(DP)
    SO, SC = np.concatenate(SO), np.concatenate(SC)
    ep_a = np.concatenate([np.triu_indices(32, 1)[0] for _ in np.unique(scene)])

    print("=== A. 扩张还是收缩 ===")
    mr, mp = DR.mean(0), DP.mean(0)
    lam_r = np.diff(np.log(mr + 1e-9))
    lam_p = np.diff(np.log(mp + 1e-9))
    print("  平均距离 首步->末步：路由 %.4f -> %.4f，位姿 %.3f -> %.3f"
          % (mr[0], mr[-1], mp[0], mp[-1]))
    print("  局部增长率 ln d(t+1)/d(t) 为负（收缩）的步数：路由 %d/%d，位姿 %d/%d"
          % ((lam_r < 0).sum(), len(lam_r), (lam_p < 0).sum(), len(lam_p)))
    print("  收缩发生在: 路由 %s" % np.flatnonzero(lam_r < 0)[:12].tolist())
    print("               位姿 %s" % np.flatnonzero(lam_p < 0)[:12].tolist())

    def gapstat(perm_y=None):
        yy = y if perm_y is None else perm_y
        so = np.concatenate([(yy[np.flatnonzero(scene == s)][np.triu_indices(32, 1)[0]]
                              == yy[np.flatnonzero(scene == s)][np.triu_indices(32, 1)[1]])
                             for s in np.unique(scene)])
        return DR[~so].mean(0) - DR[so].mean(0), DP[~so].mean(0) - DP[so].mean(0)

    gr, gp = gapstat()
    nr_, np_ = [], []
    for _ in range(N_PERM):
        yp = y.copy()
        for s in np.unique(scene):
            m = np.flatnonzero(scene == s)
            yp[m] = y[m][RNG.permutation(len(m))]
        a, b = gapstat(yp)
        nr_.append(a)
        np_.append(b)
    nr_, np_ = np.array(nr_), np.array(np_)
    zr = (gr - nr_.mean(0)) / (nr_.std(0) + 1e-12)
    zp = (gp - np_.mean(0)) / (np_.std(0) + 1e-12)
    thr_r = np.quantile(np.abs(nr_ - nr_.mean(0)).max(1) / (nr_.std(0).mean() + 1e-12), .95)
    cr = np.quantile(np.nanmax((nr_ - nr_.mean(0)) / (nr_.std(0) + 1e-12), axis=1), .95)
    cp = np.quantile(np.nanmax((np_ - np_.mean(0)) / (np_.std(0) + 1e-12), axis=1), .95)
    fr = next((t for t in range(W) if zr[t] > cr), None)
    fp_ = next((t for t in range(W) if zp[t] > cp), None)
    print("\n  异结局比同结局多分开多少（场景内置换 %d 次，最大统计量控制多重比较）" % N_PERM)
    print("  步     路由差   z      位姿差   z")
    for t in range(0, W, 4):
        print("  %3d   %+.4f  %5.2f   %+.3f  %5.2f" % (t, gr[t], zr[t], gp[t], zp[t]))
    print("  最大统计量 95%% 阈值：路由 z=%.2f，位姿 z=%.2f" % (cr, cp))
    print("  第一个显著步：路由 %s，位姿 %s" % (fr, fp_))

    # ---- B : trapping ------------------------------------------------------
    print("\n=== B. 失败是不是卡在环里（吸引子式的解释）===")
    print("  K    划分   量               失败均值  成功均值   场景校正 z   置换阈值")
    for K in (64, 256):
        for nm, X in (("路由", D["probs"].reshape(-1, 256)),
                      ("位姿", P.reshape(-1, 8))):
            c = cells(X, K, n, W)
            nuniq = np.array([len(np.unique(r)) for r in c], float)
            nswitch = (np.diff(c, axis=1) != 0).sum(1).astype(float)
            revisit = np.array([1.0 - len(np.unique(r)) / W for r in c])
            maxdwell = np.array([np.max(np.diff(np.concatenate(
                [[0], np.flatnonzero(np.diff(r)) + 1, [W]]))) for r in c], float)
            for lab, v in (("访问过的不同格子数", nuniq), ("切换次数", nswitch),
                           ("重访率", revisit), ("最长停留", maxdwell)):
                o, z = scene_z(v, fail, scene, rng=np.random.default_rng(2))
                print("  %3d  %s   %-16s %8.2f  %8.2f   %+7.2f      ±2.6"
                      % (K, nm, lab, v[fail].mean(), v[y].mean(), z))

    # ---- C : irreversibility ----------------------------------------------
    print("\n=== C. 纯失败格子：超过置换上限吗，进去还出得来吗 ===")
    print("  K    划分   纯失败格  置换上限95%  吸收的局  逃逸率（回到低失败格）")
    for K in (256, 512):
        for nm, X in (("路由", D["probs"].reshape(-1, 256)),
                      ("位姿", P.reshape(-1, 8))):
            c = cells(X, K, n, W)
            ent = np.zeros((n, K), bool)
            for r in range(n):
                ent[r, np.unique(c[r])] = True
            sup = ent.sum(0)
            good = sup >= 20
            pf = np.array([fail[ent[:, k]].mean() if sup[k] else np.nan
                           for k in range(K)])
            pure = good & (pf == 1.0)
            best = []
            for _ in range(200):
                fp = fail.copy()
                for s in np.unique(scene):
                    m = np.flatnonzero(scene == s)
                    fp[m] = fail[m][RNG.permutation(len(m))]
                q = np.array([fp[ent[:, k]].mean() if sup[k] else np.nan
                              for k in range(K)])
                best.append(int((good & (q == 1.0)).sum()))
            hit = ent[:, pure].any(1) if pure.any() else np.zeros(n, bool)
            esc = np.nan
            if pure.any():
                lo = np.flatnonzero(good & (pf < 0.5))
                r_ = []
                for e in np.flatnonzero(hit):
                    t0 = min(int(np.argmax(c[e] == k)) for k in np.flatnonzero(pure)
                             if (c[e] == k).any())
                    r_.append(bool(np.isin(c[e][t0 + 1:], lo).any()))
                esc = float(np.mean(r_)) if r_ else np.nan
            print("  %3d  %s   %6d      %6d       %5d      %s"
                  % (K, nm, int(pure.sum()), int(np.quantile(best, .95)),
                     int(hit.sum()),
                     "%.1f%%" % (100 * esc) if esc == esc else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
