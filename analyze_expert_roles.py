#!/usr/bin/env python3
"""What is each expert for?  A systematic sweep instead of one lucky find.

The gripper gate was found backwards: a bimodal histogram turned up, and only
then was the variable behind it identified.  That leaves the obvious question
unanswered -- is it the only expert with a job, or the only one anyone looked at?

So sweep all 8 HB layers x 32 experts.  For each, take "is this expert in the
state token's top-4 at this control step" and ask how well the robot's own
8-dim proprioception predicts it, and which single dimension does most of the
work.  Leave-one-scene-out, so an expert that merely fires in one kitchen scores
nothing.

    dims 0-2  end effector position x, y, z
    dims 3-5  end effector orientation
    dims 6-7  the two gripper fingers

Then the cross-checkpoint question that the expert-identity result makes
meaningful: expert slots survive fine-tuning intact (nearest match 32/32 across
all five checkpoints), so if slot 12 of layer 5 is the gripper expert on one task
it can be asked whether it is the gripper expert on the others too.
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
RUN_ID = "right-16x32"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
DIM = ["eef x", "eef y", "eef z", "rot 1", "rot 2", "rot 3", "指 a", "指 b"]
TASKS = [("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "Long"),
         ("libero_spatial",
          "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", "Spatial"),
         ("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside", "Goal")]


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
        f = LogisticRegression(C=1.0, max_iter=1500).fit(sc.transform(X[tr]), y[tr])
        s = f.decision_function(sc.transform(X[te]))
        n1 = int(y[te].sum())
        r = rankdata(s)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * int((~y[te]).sum())
    return num / den if den else float("nan")


def load(suite, task, hub_dir, sub=6000):
    run = HUB / "cache/HiMoE-VLA" / hub_dir / task / RUN_ID
    if not (run / "server/routes.zarr").exists():
        return None
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    scene = np.array([s["init_state_id"] for s in S])[ep]
    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i, :n_rows[i]] = d[:n_rows[i]]
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    n = z["hb_expert_ids"].shape[0]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(n, min(sub, n), replace=False))
    ids = np.asarray(z["hb_expert_ids"].oindex[rows, :, 0, 0, :]).astype(np.int64)
    hot = np.zeros((len(rows), 8, 32), bool)
    np.put_along_axis(hot, ids, True, -1)
    return hot, prop[ep[rows], step[rows]], scene[rows]


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl

    store = {}
    for suite, task, tag in TASKS:
        d = load(suite, task, cl.HUB_DIR[suite])
        if d:
            store[tag] = d
            print("loaded %-8s %s  (%d sites)" % (tag, task[:40], len(d[0])))

    tag0 = "Long"
    hot, prop, scene = store[tag0]
    print("\n=== %s：每层最可被本体感受预测的 3 个专家 ===" % tag0)
    print("层  专家  选中率   8维本体感受   最强的单个维度        单维 AUC")
    best = {}
    for i, L in enumerate(HB_LAYER):
        rank = []
        for e in range(32):
            y = hot[:, i, e]
            if not (0.02 < y.mean() < 0.98):
                continue
            a = loso(prop, y, scene)
            rank.append((abs(a - .5), a, e))
        rank.sort(reverse=True)
        best[L] = [r[2] for r in rank[:3]]
        for _, a, e in rank[:3]:
            y = hot[:, i, e]
            singles = [(abs(sauc(prop[:, k], y, scene) - .5), k) for k in range(8)]
            _, k = max(singles)
            print("  %2d  e%-3d  %5.1f%%   %.3f          %-8s (dim %d)   %.3f"
                  % (L, e, 100 * y.mean(), a, DIM[k], k,
                     sauc(prop[:, k], y, scene)))

    print("\n=== 同一个专家槽，在别的任务/checkpoint 上还是同一个角色吗 ===")
    print("  取 %s 上每层最可预测的那个专家，看它在别的任务上的表现" % tag0)
    print("层  专家   " + "".join("%-26s" % t for t in store))
    for i, L in enumerate(HB_LAYER):
        e = best[L][0]
        cells = []
        for tag, (h2, p2, s2) in store.items():
            y = h2[:, i, e]
            if not (0.02 < y.mean() < 0.98):
                cells.append("选中率 %4.1f%% 太极端" % (100 * y.mean()))
                continue
            a = loso(p2, y, s2)
            sing = [(abs(sauc(p2[:, k], y, s2) - .5), k) for k in range(8)]
            _, k = max(sing)
            cells.append("%.3f  %-6s(%4.1f%%)" % (a, DIM[k], 100 * y.mean()))
        print("  %2d  e%-3d  " % (L, e) + "".join("%-26s" % c for c in cells))
    print("\n  每格：8 维本体感受的 LOSO AUC、最强单维、该专家的选中率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
