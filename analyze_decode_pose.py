#!/usr/bin/env python3
"""Invert it: can the arm's pose be read back out of the routing?

Forward is settled -- whether an expert is in the state token's top-4 is
predictable from the 8-dim proprioception at AUC 0.94-0.999, and shuffling the
pairing takes it to 0.50.  If the routing really is a partition of the pose
space, the map should run backwards too: knowing which cell you are in should
pin down the pose to within the cell.

Measured as leave-one-scene-out R^2, and then converted to physical units,
because an R^2 does not say whether the reconstruction is worth anything -- a
millimetre and a centimetre both look like 0.99 on a task that spans 30 cm.

Two forms of "the routing", because they carry different amounts:
    binary   which four experts are in the top-4     8 x 32 indicators
    weights  the full router distribution            8 x 32 probabilities
and front block against back block, since the front sits closer to
state_proj(proprio) and should invert better.

Control: the same regression with the routing rows shuffled inside each scene.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from sklearn.linear_model import Ridge

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
DIM = ["eef x", "eef y", "eef z", "轴角1", "轴角2", "轴角3", "指 a", "指 b"]
UNIT = ["m", "m", "m", "rad", "rad", "rad", "m", "m"]
RNG = np.random.default_rng(0)


def loso_r2(X, Y, scene):
    """Leave-one-scene-out R^2 and RMSE per target column."""
    pred = np.zeros_like(Y)
    for h in np.unique(scene):
        tr, te = scene != h, scene == h
        m = Ridge(alpha=10.0).fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot, np.sqrt(((Y - pred) ** 2).mean(0))


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    scene_ep = np.array([s["init_state_id"] for s in S])
    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i, :n_rows[i]] = d[:n_rows[i]]
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = z["hb_expert_ids"].shape[0]
    rows = np.sort(RNG.choice(n, 9000, replace=False))
    ids = np.asarray(z["hb_expert_ids"].oindex[rows, :, 0, 0, :]).astype(np.int64)
    prob = np.asarray(z["hb_router_probs"].oindex[rows, :, 0, 0, :], np.float32)
    hot = np.zeros((len(rows), 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    Y = prop[ep[rows], step[rows]]
    sc = scene_ep[ep[rows]]
    print("t08，%d 个控制步样本，留一场景交叉验证\n" % len(rows))

    sets = {
        "二值 top-4，全 8 层 (256)": hot.reshape(len(rows), -1),
        "二值 top-4，前块 2-5 (128)": hot[:, :4].reshape(len(rows), -1),
        "二值 top-4，后块 12-15 (128)": hot[:, 4:].reshape(len(rows), -1),
        "连续概率，全 8 层 (256)": prob.reshape(len(rows), -1),
        "连续概率，前块 2-5 (128)": prob[:, :4].reshape(len(rows), -1),
    }
    print("=== 从路由重建 8 维位姿：R² ===")
    print("特征                          " + "  ".join("%-6s" % d for d in DIM))
    res = {}
    for name, X in sets.items():
        r2, rmse = loso_r2(X, Y, sc)
        res[name] = (r2, rmse)
        print("  %-28s" % name + "  ".join("%6.3f" % v for v in r2))

    # control
    Xs = sets["连续概率，全 8 层 (256)"].copy()
    for s in np.unique(sc):
        m = np.flatnonzero(sc == s)
        Xs[m] = Xs[RNG.permutation(m)]
    r2s, _ = loso_r2(Xs, Y, sc)
    print("  %-28s" % "对照：场景内打乱行序" + "  ".join("%6.3f" % v for v in r2s))

    print("\n=== 换成物理单位：重建误差 ===")
    name = "连续概率，全 8 层 (256)"
    r2, rmse = res[name]
    print("  用 %s" % name)
    print("  维        RMSE          该维的标准差    误差 / 标准差")
    for j in range(8):
        u = 1000 if UNIT[j] == "m" else 1
        s = "mm" if UNIT[j] == "m" else "rad"
        print("  %-9s %7.2f %-4s  %7.2f %-4s   %.3f"
              % (DIM[j], rmse[j] * u, s, Y[:, j].std() * u, s, rmse[j] / Y[:, j].std()))

    gap = Y[:, 6] - Y[:, 7]
    Xg = sets[name]
    pred = np.zeros(len(gap))
    for h in np.unique(sc):
        tr, te = sc != h, sc == h
        pred[te] = Ridge(alpha=10.0).fit(Xg[tr], gap[tr]).predict(Xg[te])
    r2g = 1 - ((gap - pred) ** 2).sum() / ((gap - gap.mean()) ** 2).sum()
    print("\n  夹爪开合度（指a − 指b，真正的那一个自由度）:")
    print("     R² = %.3f，RMSE = %.2f mm，量程 %.1f mm"
          % (r2g, 1000 * np.sqrt(((gap - pred) ** 2).mean()),
             1000 * (gap.max() - gap.min())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
