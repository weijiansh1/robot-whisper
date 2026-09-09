#!/usr/bin/env python3
"""Is the rule's residual the world, or the model's own intention?

Honestly split, the mined conjunctions keep a small but consistent excess over a
baseline that already knows the scene and the full 47-dim simulator state:
+0.017 +- 0.014 (18/20 splits) within scenes, +0.019 +- 0.013 (19/20) across
them, against a random-rule floor of +0.002.  Small, but it is the first thing
in this investigation where the routing beats the physical state.

There is one obvious candidate for what it is.  The routing is computed from an
observation the simulator state also describes, but it is computed *inside* the
same forward pass that emits the action chunk -- so it can carry what the model
is about to do, which the world state does not contain.  An earlier fork pilot
already found the action chunk to be a 2.4x better outcome proxy than routing at
a fixed state, which is what that would look like.

So put the emitted chunk (10 x 7) into the baseline and rerun the identical
honest loop.  If the routing excess dies there, the routing is a lossy shadow of
the model's own intention and nothing more.  If it survives, it is carrying
something neither the world nor the action reveals.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_rules import HUB, cooccur, lit_name, load, score  # noqa: E402
from analyze_rules_increment import loso_p  # noqa: E402

N_REP = 20
TOPK = 20


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    suite = sys.argv[2] if len(sys.argv) > 2 else "libero_long"
    run = HUB / "cache/HiMoE-VLA" / suite / task / "right-16x32"
    # third argument truncates the window, so a long task can be compared with a
    # short one at matched length rather than at matched task
    win = int(sys.argv[3]) if len(sys.argv) > 3 else None
    D = load(run, win=win)
    y, scene, M, prop, W = D["y"], D["scene"], D["M"], D["prop"], D["win"]
    fail, n = ~y, len(y)
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)
    ns = d0["sim_state"].shape[1]
    sim = np.zeros((n, W, ns), np.float32)
    act = np.zeros((n, W, 70), np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        sim[i] = d["sim_state"][:W]
        act[i] = d["actions"][:W].reshape(W, -1)
    ep_row = np.repeat(np.arange(n), W)
    step = np.tile(np.arange(W), n).astype(np.float32)[:, None]
    yf = np.repeat(fail, W).astype(int)
    si = np.unique(scene, return_inverse=True)[1]
    oh = np.zeros((n * W, si.max() + 1), np.float32)
    oh[np.arange(n * W), si[ep_row]] = 1
    PH = np.c_[oh, sim.reshape(-1, ns), prop.reshape(-1, 8), step]
    AC = act.reshape(-1, 70)
    print("%s，%d 局 x %d 步\n" % (task, n, W))

    base = {"场景+物理状态": PH,
            "场景+动作块": np.c_[oh, AC, step],
            "场景+物理+动作块": np.c_[PH, AC]}
    P = {}
    for k, X in base.items():
        P[k] = loso_p(X, yf, ep_row)
        ll = -(yf * np.log(np.clip(P[k], 1e-6, 1)) +
               (1 - yf) * np.log(np.clip(1 - P[k], 1e-6, 1))).mean() / np.log(2)
        print("  基线「%s」(%d 维) 步级对数损失 %.3f bit" % (k, X.shape[1], ll))

    fires = cooccur(M)
    res = {k: (yf - v).reshape(n, W) for k, v in P.items()}
    rand_hit = np.random.default_rng(3).random((n, W)) < 0.10
    print("\n=== 按场景挖一半、验另一半（%d 次）；数字是留出集上的残差超额失败率 ==="
          % N_REP)
    print("  被打分的东西            " + "".join("%-24s" % k for k in base))
    acc = {k: {"top": [], "pool": [], "rnd": []} for k in base}
    names = []
    for r in range(N_REP):
        g = np.random.default_rng(100 + r)
        sc = g.permutation(np.unique(scene))
        A = np.isin(scene, sc[:len(sc) // 2])
        B = ~A
        pfA = np.zeros(A.sum(), np.float32)
        scA = scene[A]
        for s in np.unique(scA):
            pfA[scA == s] = fail[A][scA == s].mean()
        sA, oA, eA, zA = score(fires[A], fail[A], pfA)
        ok = (sA >= 10) & (sA <= A.sum() - 3)
        ok[np.tril_indices(256, -1)] = False
        if ok.sum() < TOPK:
            continue
        order = np.argsort(np.where(ok, zA, -np.inf).ravel())[::-1]
        a, b = np.unravel_index(order[0], zA.shape)
        names.append("%s & %s" % (lit_name(a), lit_name(b)))
        top = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
        pool = np.zeros((n, W), bool)
        for f in order[:TOPK]:
            u, v = np.unravel_index(f, zA.shape)
            pool |= (M[:, :, u] > 0.5) & (M[:, :, v] > 0.5)
        for k in base:
            acc[k]["top"].append(res[k][B][top[B]].mean()
                                 if top[B].sum() > 20 else np.nan)
            acc[k]["pool"].append(res[k][B][pool[B]].mean()
                                  if pool[B].sum() > 20 else np.nan)
            acc[k]["rnd"].append(res[k][B][rand_hit[B]].mean())
    for lab, key in (("挖出的最强规则", "top"), ("前 %d 条并集" % TOPK, "pool"),
                     ("对照：随机规则", "rnd")):
        row = "  %-22s " % lab
        for k in base:
            v = np.array(acc[k][key])
            row += "%+.4f±%.4f (%2d/%2d) " % (np.nanmean(v), np.nanstd(v),
                                              int(np.nansum(v > 0)), len(v))
        print(row)
    # the random arm is not exactly zero -- the residual has a small positive
    # tilt on any subset -- so the statistic that matters is the paired
    # difference on the same split
    print("\n  规则减去随机对照（同一划分内配对）:")
    for lab, key in (("最强规则", "top"), ("前 %d 条并集" % TOPK, "pool")):
        row = "    %-14s " % lab
        for k in base:
            d = np.array(acc[k][key]) - np.array(acc[k]["rnd"])
            se = np.nanstd(d) / np.sqrt(np.isfinite(d).sum())
            row += "%+.4f ± %.4f (%2d/%2d 正) " % (
                np.nanmean(d), se, int(np.nansum(d > 0)), np.isfinite(d).sum())
        print(row)

    c = {}
    for nm in names:
        c[nm] = c.get(nm, 0) + 1
    print("\n  挖出 %d 条不同规则；最常出现: %s"
          % (len(c), "，".join("%s x%d" % kv
                               for kv in sorted(c.items(), key=lambda t: -t[1])[:3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
