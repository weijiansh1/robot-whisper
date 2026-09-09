#!/usr/bin/env python3
"""Does the trajectory-shape result replicate on the other tasks?

On t08 the routing trajectory's path length beat the full physical state
(0.808 vs 0.743, scene bootstrap CI [+0.013, +0.109]) and its tortuosity had the
opposite sign (0.786 vs 0.285), both surviving a gripper-cycle control.  One
task and one checkpoint is not a result, so run the identical pipeline on every
captured task.

Two things differ per task and both have to be set from the data, not assumed:

  window   every failure runs to the horizon and successes end when they finish,
           so the comparison window is the longest prefix in which all 512
           rollouts are still running.  On t08 that is 34 steps; on the spatial
           tasks it is closer to 9, which is a much thinner trajectory to
           characterise and should be read with that in mind.
  power    t00 succeeded 512/512, so it has no contrast at all and is skipped;
           t05 has only 12 failures, so its intervals will be wide.
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
RUN_ID = "right-16x32"
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


def perm_p(x, y, scene, n=1000):
    obs = abs(sauc(x, y, scene) - .5)
    hit = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(scene):
            i = np.flatnonzero(scene == s)
            yp[i] = y[RNG.permutation(i)]
        hit += abs(sauc(x, yp, scene) - .5) >= obs
    return (1 + hit) / (n + 1)


def one(suite, task, hub_dir):
    run = HUB / "cache/HiMoE-VLA" / hub_dir / task / RUN_ID
    if not (run / "server/routes.zarr").exists():
        return None
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    if y.all() or (~y).all():
        return dict(task=task, skip="no contrast (%d/%d success)" % (y.sum(), len(y)))
    W = int(n_rows.min())
    mixed = [int(s) for s in np.unique(scene)
             if 2 <= y[scene == s].sum() <= (scene == s).sum() - 2]
    keep = np.isin(scene, mixed)
    if keep.sum() < 60 or W < 6:
        return dict(task=task, skip="window %d, %d usable episodes" % (W, keep.sum()))

    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    nS = len(S)
    R = np.empty((nS, W, 256), np.float32)
    for i in range(nS):
        R[i] = np.asarray(z["hb_router_probs"].oindex[off[i] + np.arange(W),
                                                      :, 0, 0, :],
                          np.float32).reshape(W, -1)
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    P = np.empty((nS, W, d0.shape[1]), np.float32)
    G = np.empty((nS, W), np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        P[i] = d["sim_state"][:W]
        G[i] = d["state"][:W, 6] - d["state"][:W, 7]
    for X in (R, P):
        X -= X.reshape(-1, X.shape[-1]).mean(0)
        X /= X.reshape(-1, X.shape[-1]).std(0) + 1e-9

    plen = lambda X: np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)
    endp = lambda X: np.linalg.norm(X[:, -1] - X[:, 0], axis=1) + 1e-9
    pr, pp = plen(R), plen(P)
    tr, tp = pr / endp(R), pp / endp(P)
    ntr = np.array([int(np.abs(np.diff((G[i] < 0.04).astype(int))).sum())
                    for i in range(nS)])

    def resid(v):
        r = v.astype(float).copy()
        for s in np.unique(scene):
            m = scene == s
            A = np.c_[ntr[m], np.ones(m.sum())]
            r[m] = v[m] - A @ np.linalg.lstsq(A, v[m], rcond=None)[0]
        return r

    boot = []
    for _ in range(1500):
        pick = RNG.choice(mixed, len(mixed), replace=True)
        m = np.isin(scene, pick)
        try:
            boot.append(sauc(pr[m], y[m], scene[m]) - sauc(pp[m], y[m], scene[m]))
        except Exception:
            pass
    boot = np.array(boot)
    return dict(task=task, W=W, n=int(keep.sum()), nfail=int((~y[keep]).sum()),
                nsc=len(mixed),
                pr=sauc(pr[keep], y[keep], scene[keep]),
                pp=sauc(pp[keep], y[keep], scene[keep]),
                p_pr=perm_p(pr[keep], y[keep], scene[keep]),
                tr=sauc(tr[keep], y[keep], scene[keep]),
                tp=sauc(tp[keep], y[keep], scene[keep]),
                grip=sauc(ntr.astype(float)[keep], y[keep], scene[keep]),
                pr_r=sauc(resid(pr)[keep], y[keep], scene[keep]),
                pp_r=sauc(resid(pp)[keep], y[keep], scene[keep]),
                dif=float(np.mean(boot)),
                ci=tuple(np.percentile(boot, [2.5, 97.5])))


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    out = []
    for suite in ("libero_goal", "libero_spatial", "libero_10"):
        base = HUB / "cache/HiMoE-VLA" / cl.HUB_DIR[suite]
        for task in sorted(p.name for p in base.iterdir()):
            r = one(suite, task, cl.HUB_DIR[suite])
            if r:
                r["suite"] = suite
                out.append(r)

    print("路径长度：路由 vs 物理状态，每个任务独立跑同一套流程\n")
    print("任务                        窗口  可用局/失败  路由    物理    差 (95%CI)"
          "            路由 p")
    for r in out:
        if "skip" in r:
            print("  %-26s -- 跳过：%s" % (r["task"][:26], r["skip"]))
            continue
        print("  %-26s %3d   %3d/%-3d   %.3f  %.3f  %+.3f [%+.3f,%+.3f]  %.3f"
              % (r["task"][:26], r["W"], r["n"], r["nfail"], r["pr"], r["pp"],
                 r["dif"], r["ci"][0], r["ci"][1], r["p_pr"]))

    print("\n弯曲度（路径长/端点距）—— t08 上路由和物理是反号的")
    print("任务                        路由    物理")
    for r in out:
        if "skip" in r:
            continue
        print("  %-26s %.3f  %.3f%s" % (r["task"][:26], r["tr"], r["tp"],
              "   <- 反号" if (r["tr"] - .5) * (r["tp"] - .5) < 0 else ""))

    print("\n扣掉夹爪开合次数之后（场景内回归残差）")
    print("任务                        开合次数单独  路由 原始→残差   物理 原始→残差")
    for r in out:
        if "skip" in r:
            continue
        print("  %-26s   %.3f       %.3f→%.3f     %.3f→%.3f"
              % (r["task"][:26], r["grip"], r["pr"], r["pr_r"], r["pp"], r["pp_r"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
