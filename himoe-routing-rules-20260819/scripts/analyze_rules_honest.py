#!/usr/bin/env python3
"""The rule chosen on one set of episodes, scored on another.

Against an episode-grouped, scene-aware, nonlinear baseline the top conjunction
kept a +0.091 [+0.052,+0.140] excess failure rate on top of the *full* 47-dim
simulator state.  That number is not yet honest: those rules were picked by their
scene-adjusted z on the same 512 episodes they were then scored on, so the
selection is free to chase the residual it is later credited with.

Here the selection and the scoring never see the same episode.  The baseline
predictions are computed once, episode-grouped, so no row ever sees its own
outcome; then 20 times over, the search runs on half the episodes and the excess
is read off the other half.

Two split kinds, because they ask different questions:
    按局   both halves contain all 16 scenes -- does the rule generalise to new
           noise draws of a situation it has seen?
    按场景 the halves are disjoint in scenes -- does it mean anything general?

A same-rate random rule goes through the identical loop; whatever it scores is
the floor, and the gap above it is what the routing is worth.
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
    D = load(run)
    y, scene, M, prop, W = D["y"], D["scene"], D["M"], D["prop"], D["win"]
    fail, n = ~y, len(y)
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.zeros((n, W, d0.shape[1]), np.float32)
    for i, s in enumerate(S):
        sim[i] = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                         allow_pickle=True)["sim_state"][:W]
    ep_row = np.repeat(np.arange(n), W)
    step = np.tile(np.arange(W), n).astype(np.float32)[:, None]
    yf = np.repeat(fail, W).astype(int)
    si = np.unique(scene, return_inverse=True)[1]
    oh = np.zeros((n * W, si.max() + 1), np.float32)
    oh[np.arange(n * W), si[ep_row]] = 1
    X = np.c_[oh, sim.reshape(-1, d0.shape[1]), prop.reshape(-1, 8), step]
    P = loso_p(X, yf, ep_row)
    ll = -(yf * np.log(np.clip(P, 1e-6, 1)) +
           (1 - yf) * np.log(np.clip(1 - P, 1e-6, 1))).mean() / np.log(2)
    print("%s\n%d 局 x %d 步；基线 = 场景 + 完整仿真状态 %d 维 + 相位，"
          "按局分折，对数损失 %.3f bit\n" % (task, n, W, d0.shape[1], ll))

    fires = cooccur(M)
    res = (yf - P).reshape(n, W)
    rng = np.random.default_rng(3)
    rand_hit = rng.random((n, W)) < 0.10

    for kind in ("按局（两边都含全部场景）", "按场景（两边场景不重叠）"):
        e1, ek, er, names = [], [], [], []
        for r in range(N_REP):
            g = np.random.default_rng(100 + r)
            if kind.startswith("按局"):
                A = np.zeros(n, bool)
                for s in np.unique(scene):
                    m = np.flatnonzero(scene == s)
                    A[g.permutation(m)[:len(m) // 2]] = True
            else:
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
            hit = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
            e1.append(res[B][hit[B]].mean() if hit[B].sum() > 20 else np.nan)
            pool = np.zeros((n, W), bool)
            for f in order[:TOPK]:
                u, v = np.unravel_index(f, zA.shape)
                pool |= (M[:, :, u] > 0.5) & (M[:, :, v] > 0.5)
            ek.append(res[B][pool[B]].mean() if pool[B].sum() > 20 else np.nan)
            er.append(res[B][rand_hit[B]].mean())
        e1, ek, er = np.array(e1), np.array(ek), np.array(er)
        print("=== %s，%d 次重复 ===" % (kind, len(e1)))
        print("  挖出的最强规则   留出残差超额 %+.4f ± %.4f   (%d/%d 次为正)"
              % (np.nanmean(e1), np.nanstd(e1), int(np.nansum(e1 > 0)), len(e1)))
        print("  前 %d 条并集      留出残差超额 %+.4f ± %.4f   (%d/%d 次为正)"
              % (TOPK, np.nanmean(ek), np.nanstd(ek),
                 int(np.nansum(ek > 0)), len(ek)))
        print("  对照：随机规则    留出残差超额 %+.4f ± %.4f"
              % (np.nanmean(er), np.nanstd(er)))
        c = {}
        for nm in names:
            c[nm] = c.get(nm, 0) + 1
        print("  挖出 %d 条不同规则，最常出现的: %s\n"
              % (len(c), "，".join("%s x%d" % kv for kv in
                                   sorted(c.items(), key=lambda t: -t[1])[:3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
