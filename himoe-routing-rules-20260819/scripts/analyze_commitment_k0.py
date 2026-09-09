#!/usr/bin/env python3
"""Commitment at k=0, and where the routing sits on the same axis.

The earlier fork experiment asked how much of the outcome is already fixed after
k control steps, by restoring one state and re-drawing the future.  This corpus
answers the k=0 case at higher resolution without any forking: 16 initial scenes
are the prefixes and 32 flow-noise seeds per scene are the futures, against the
fork run's 8 x 8.  The caveat is that the two are not the same quantity -- a fork
prefix is one state advanced k steps, this one is "which scene" -- so the columns
are comparable in shape, not in magnitude.

Then the part that is about the routing.  H(Y | routing at step t) is estimated
by cross-validated log-loss, which is an upper bound on the conditional entropy,
so the "resolved" fraction it yields is a lower bound.  Put beside H(Y | physical
state at step t) it says whether routing is an extra channel or a lossy copy.

Survivorship is the trap here: failures always run to the horizon while successes
stop early, so any window that outlives the shortest success turns "still
running" into "will fail".  The window is checked and printed rather than
assumed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB/cache/HiMoE-VLA"
TASKS = [
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"),
    ("libero_long", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"),
]
FOCUS = ("libero_long", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove")


def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def meta(run):
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    nr = np.array([s["inference_calls"] for s in S])
    return (S, nr, np.concatenate([[0], np.cumsum(nr)[:-1]]),
            np.array([s["success"] for s in S], bool),
            np.array([s["init_state_id"] for s in S]))


def cond_H(X, y, groups=None):
    """Cross-validated log-loss in bits -- an upper bound on H(Y|X)."""
    p = np.zeros(len(y))
    folds = ([(groups != h, groups == h) for h in np.unique(groups)]
             if groups is not None
             else StratifiedKFold(8, shuffle=True, random_state=0).split(X, y))
    for tr, te in folds:
        sd = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=0.2, max_iter=3000).fit(sd.transform(X[tr]), y[tr])
        p[te] = f.predict_proba(sd.transform(X[te]))[:, 1]
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(q) + (~y) * np.log2(1 - q)).mean()


def main() -> int:
    print("=== 1. k=0 承诺：前缀 = 初始场景，未来 = 32 个流匹配噪声抽样 ===")
    print("  任务                        局数  成功率  H(Y)   H(Y|前缀)  已解决   锁死的前缀")
    for suite, task in TASKS:
        run = HUB / suite / task / "right-16x32"
        _, _, _, y, sc = meta(run)
        HY = Hb(y.mean())
        rates = np.array([y[sc == s].mean() for s in np.unique(sc)])
        HYX = np.mean([Hb(y[sc == s].mean()) for s in np.unique(sc)])
        print("  %-26s %4d  %5.1f%%  %.3f   %.3f     %5.1f%%   %d/%d"
              % (task[:26], len(y), 100 * y.mean(), HY, HYX,
                 100 * (1 - HYX / HY) if HY > 0 else 0.0,
                 int(((rates == 0) | (rates == 1)).sum()), len(rates)))
    print("\n  对照：原来那次 fork 实验（LIBERO-Goal t0 init24，8 前缀 x 8 未来）")
    for k, h, r in ((4, 0.906, 8.3), (8, 0.601, 39.9), (11, 0.203, 78.7),
                    (12, 0.101, 89.1)):
        print("    k=%-3d  H(Y|prefix)=%.3f  已解决 %4.1f%%%s"
              % (k, h, r, "   （不显著）" if k == 4 else ""))
    print("  注：fork 的前缀是「同一初态推进 k 步」，这里的前缀是「哪个场景」，"
          "两列同形不同量。")

    suite, task = FOCUS
    run = HUB / suite / task / "right-16x32"
    S, nr, off, y, sc = meta(run)
    print("\n=== 2. 生存偏差检查（%s）===" % task[:34])
    print("  失败局长度 %d-%d，成功局长度 %d-%d"
          % (nr[~y].min(), nr[~y].max(), nr[y].min(), nr[y].max()))
    print("   步   仍在跑  占比    其中成功率")
    for t in (0, 8, 13, 20, 26, 34, 40):
        m = nr > t
        print("   %3d   %4d  %5.1f%%   %5.1f%%"
              % (t, m.sum(), 100 * m.mean(), 100 * y[m].mean()))
    print("  => 窗口取到 %d 步为止，512 局全在场" % (nr[y].min() - 1))

    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.full((len(S), nr.max(), d0.shape[1]), np.nan, np.float32)
    for i, s in enumerate(S):
        sim[i, :nr[i]] = np.load(
            run / ("client/episode_%02d.npz" % s["episode_index"]),
            allow_pickle=True)["sim_state"][:nr[i]]
    HY = Hb(y.mean())
    oh = np.zeros((len(S), 16), np.float32)
    oh[np.arange(len(S)), np.unique(sc, return_inverse=True)[1]] = 1
    print("\n=== 3. 把路由放到承诺这根轴上   H(Y) = %.3f bit ===" % HY)
    print("  （H(Y|X) 用交叉验证对数损失做上界，所以「已解决」是下界）")
    h_sc = cond_H(oh, y)
    print("  只知道场景（k=0 的前缀）    H=%.3f   已解决 %5.1f%%"
          % (h_sc, 100 * (1 - h_sc / HY)))
    print("\n  控制步   只知路由   路由+场景   只知物理状态   物理+场景")
    for t in (0, 8, 13, 20, 26, 34):
        idx = np.flatnonzero(nr > t)
        ST = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, 0, :],
                        np.float32).reshape(len(idx), -1)
        PH, O, yy = sim[idx, t], oh[idx], y[idx]
        v = [cond_H(ST, yy), cond_H(np.c_[ST, O], yy),
             cond_H(PH, yy), cond_H(np.c_[PH, O], yy)]
        print("   %3d     %.3f      %.3f       %.3f        %.3f" % (t, *v))
        print("           %5.1f%%     %5.1f%%      %5.1f%%       %5.1f%%"
              % tuple(100 * (1 - a / HY) for a in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
