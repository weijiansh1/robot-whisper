#!/usr/bin/env python3
"""Does the pose-code result hold on the other tasks and checkpoints?

On t08 the state token's routing turned out to be ~96% a nonlinear function of
the robot's own 8 numbers, invertible back to the pose at R^2 0.87-0.97, with
object poses adding nothing.  The forward direction was already spot-checked on
three tasks; the inverse and the budget were not.  Goal, Spatial and Long are
three separately fine-tuned checkpoints, so agreeing across them is the cheapest
available test that this is a property of the architecture rather than of one
weight set.

Three measurements per task, all leave-one-scene-out:

    forward  median |AUC-0.5| over the experts of each layer, predicting
             "is this expert in the state token's top-4" from proprio
    inverse  R^2 decoding each pose dimension back out of the top-4 indicators
    budget   fraction of routing variance a linear and an RBF fit on proprio
             account for, and whether object poses add anything on top

RBF rather than gradient boosting for the nonlinear fit: on t08 they landed
within a point of each other (95.1 vs 96.3) and this runs in seconds per task.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN_ID = "right-16x32"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
DIM = ["eef x", "eef y", "eef z", "rot1", "rot2", "rot3", "指a", "指b"]
RNG = np.random.default_rng(0)


def loso_auc(X, y, s):
    num = den = 0.0
    for h in np.unique(s):
        tr, te = s != h, s == h
        if not (0 < y[tr].sum() < tr.sum()) or not (0 < y[te].sum() < te.sum()):
            continue
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=1.0, max_iter=1500).fit(sc.transform(X[tr]), y[tr])
        v = f.decision_function(sc.transform(X[te]))
        n1 = int(y[te].sum())
        r = rankdata(v)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * int((~y[te]).sum())
    return num / den if den else float("nan")


def loso_pred(make, X, Y, s):
    pred = np.zeros_like(Y)
    for h in np.unique(s):
        tr, te = s != h, s == h
        sc = StandardScaler().fit(X[tr])
        v = make(sc.transform(X[tr]), Y[tr], sc.transform(X[te]))
        pred[te] = v.reshape(pred[te].shape)
    return pred


LIN = lambda A, Y, B: Ridge(alpha=10.0).fit(A, Y).predict(B)


def RBF(A, Y, B):
    ny = Nystroem(gamma=0.3, n_components=500, random_state=0).fit(A)
    return Ridge(alpha=1.0).fit(ny.transform(A), Y).predict(ny.transform(B))


def frac(Y, p):
    return 1 - ((Y - p) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def one(suite, task, hub_dir, sub=6000):
    run = HUB / "cache/HiMoE-VLA" / hub_dir / task / RUN_ID
    if not (run / "server/routes.zarr").exists():
        return None
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    scene = np.array([s["init_state_id"] for s in S])[ep]
    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.full((len(S), n_rows.max(), d0.shape[1]), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        prop[i, :n_rows[i]] = d["state"][:n_rows[i]]
        sim[i, :n_rows[i]] = d["sim_state"][:n_rows[i]]
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    n = z["hb_expert_ids"].shape[0]
    rows = np.sort(RNG.choice(n, min(sub, n), replace=False))
    ids = np.asarray(z["hb_expert_ids"].oindex[rows, :, 0, 0, :]).astype(np.int64)
    hot = np.zeros((len(rows), 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    prob = np.asarray(z["hb_router_probs"].oindex[rows, :, 0, 0, :],
                      np.float32).reshape(len(rows), -1)
    X8 = prop[ep[rows], step[rows]]
    X55 = np.c_[X8, sim[ep[rows], step[rows]]]
    SC = scene[rows]

    fwd = []
    for i in range(8):
        v = [abs(loso_auc(X8, hot[:, i, e] > 0, SC) - .5) for e in range(32)
             if 0.05 < (hot[:, i, e] > 0).mean() < 0.95]
        fwd.append(np.median(v) if v else np.nan)
    inv, _ = None, None
    P = loso_pred(LIN, hot.reshape(len(rows), -1), X8, SC)
    inv = 1 - ((X8 - P) ** 2).sum(0) / ((X8 - X8.mean(0)) ** 2).sum(0)
    gap = X8[:, 6] - X8[:, 7]
    pg = loso_pred(LIN, hot.reshape(len(rows), -1), gap[:, None], SC)[:, 0]
    r2g = 1 - ((gap - pg) ** 2).sum() / ((gap - gap.mean()) ** 2).sum()
    bud = (frac(prob, loso_pred(LIN, X8, prob, SC)),
           frac(prob, loso_pred(RBF, X8, prob, SC)),
           frac(prob, loso_pred(LIN, X55, prob, SC)))
    return dict(task=task, n=len(rows), fwd=fwd, inv=inv, r2g=r2g, bud=bud,
                rmse_g=float(np.sqrt(((gap - pg) ** 2).mean())),
                rng_g=float(gap.max() - gap.min()))


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    out = []
    for suite, ck in (("libero_goal", "Goal"), ("libero_spatial", "Spatial"),
                      ("libero_10", "Long")):
        base = HUB / "cache/HiMoE-VLA" / cl.HUB_DIR[suite]
        for task in sorted(p.name for p in base.iterdir()):
            r = one(suite, task, cl.HUB_DIR[suite])
            if r:
                r["ck"] = ck
                out.append(r)
                print("done %-8s %s" % (ck, task[:44]), flush=True)

    print("\n=== 正向：位姿 → 选中哪些专家（每层专家的中位 |AUC-0.5|，0.5=确定）===")
    print("ckpt     任务                     " + " ".join("L%-4d" % L for L in HB_LAYER))
    for r in out:
        print("  %-7s %-24s " % (r["ck"], r["task"][:24])
              + " ".join("%.3f" % v for v in r["fwd"]))

    print("\n=== 反向：路由 → 位姿（LOSO R²）===")
    print("ckpt     任务                     " + " ".join("%-6s" % d for d in DIM))
    for r in out:
        print("  %-7s %-24s " % (r["ck"], r["task"][:24])
              + " ".join("%6.3f" % v for v in r["inv"]))
    print("\n  夹爪开合度单独看：")
    for r in out:
        print("    %-7s %-26s R² %.3f   误差 %.2f mm / 量程 %.1f mm"
              % (r["ck"], r["task"][:26], r["r2g"], 1000 * r["rmse_g"],
                 1000 * r["rng_g"]))

    print("\n=== 预算：路由方差被解释掉多少 ===")
    print("ckpt     任务                     本体8线性  本体8非线性  本体+物体线性")
    for r in out:
        print("  %-7s %-24s  %5.1f%%     %5.1f%%      %5.1f%%"
              % (r["ck"], r["task"][:24], 100 * r["bud"][0], 100 * r["bud"][1],
                 100 * r["bud"][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
