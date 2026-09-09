#!/usr/bin/env python3
"""Do the "always fails" routing rules survive, and do they beat a pose predicate?

The first pass found conjunctions with a real scene-adjusted excess -- the best
reached z=6.30 against a permuted-label ceiling of 5.57 -- but the single
held-out check collapsed from z=5.22 to z=1.20, and the strongest rule decoded
to "the arm is 1.2 sd low and 1.2 sd toward +y", which is a pose statement.

Two things settle it, and both need the mining to be rerun inside the loop
rather than scored after the fact:

  transfer   20 random 8/8 splits of the scenes.  Mine on one half, score the
             top rule and the top-20 average on the other.  Under a null
             procedure held-out z is centred on 0 with sd 1, so the honest
             comparison is against that, not against the mining-half z.

  novelty    the identical search run over 32 *pose* literals (8 proprio dims x
             quartile), giving 528 conjunctions instead of 32,896.  If a pose
             predicate transfers at least as well as a routing predicate, the
             routing rule is a lossy restatement of it and carries nothing of
             its own.

A label-permuted arm runs through the whole pipeline as the calibration.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_rules import HUB, cooccur, lit_name, load, score  # noqa: E402

N_SPLIT = 20
TOPK = 20
RNG = np.random.default_rng(1)


def pose_literals(prop):
    """8 proprio dims x quartile -> 32 binary literals per (episode, step)."""
    n, w, d = prop.shape
    flat = prop.reshape(-1, d)
    out = np.zeros((n * w, d * 4), np.float32)
    for j in range(d):
        q = np.quantile(flat[:, j], [.25, .5, .75])
        b = np.digitize(flat[:, j], q)
        out[np.arange(n * w), j * 4 + b] = 1.0
    return out.reshape(n, w, d * 4)


def pose_name(i):
    return "%s Q%d" % (["x", "y", "z", "r1", "r2", "r3", "指a", "指b"][i // 4],
                       i % 4 + 1)


def one_family(fires, fail, p_fail, scene, name, namer, min_sup_mine=10):
    L = fires.shape[1]
    tri = np.tril_indices(L, -1)
    sc = np.unique(scene)
    t1, tk, best_rules = [], [], []
    for r in range(N_SPLIT):
        perm = RNG.permutation(sc)
        A, B = np.isin(scene, perm[:len(sc) // 2]), np.isin(scene, perm[len(sc) // 2:])
        sA, oA, eA, zA = score(fires[A], fail[A], p_fail[A])
        sB, oB, eB, zB = score(fires[B], fail[B], p_fail[B])
        ok = (sA >= min_sup_mine) & (sA <= A.sum() - 3) & (sB >= 5)
        ok[tri] = False
        if ok.sum() < TOPK:
            continue
        zm = np.where(ok, zA, -np.inf)
        order = np.argsort(zm.ravel())[::-1]
        b = np.unravel_index(order[0], zA.shape)
        t1.append(zB[b])
        best_rules.append((namer(b[0]), namer(b[1]), zA[b], zB[b]))
        idx = np.unravel_index(order[:TOPK], zA.shape)
        tk.append(np.nanmean(zB[idx]))
    t1, tk = np.array(t1), np.array(tk)
    print("  %-22s  留出 z：最强规则 %+.2f ± %.2f，前%d平均 %+.2f ± %.2f   "
          "(%d/%d 个划分留出 z>1.96)"
          % (name, t1.mean(), t1.std(), TOPK, tk.mean(), tk.std(),
             (t1 > 1.96).sum(), len(t1)))
    return t1, tk, best_rules


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    suite = sys.argv[2] if len(sys.argv) > 2 else "libero_long"
    D = load(HUB / "cache/HiMoE-VLA" / suite / task / "right-16x32")
    y, scene, M, prop = D["y"], D["scene"], D["M"], D["prop"]
    fail = ~y
    p_fail = np.zeros(len(y), np.float32)
    for s in np.unique(scene):
        p_fail[scene == s] = fail[scene == s].mean()
    print("%s\n%d 局 / %d 场景，%d 次随机 8-8 场景划分\n"
          % (task, len(y), len(np.unique(scene)), N_SPLIT))

    fr_route = cooccur(M)
    fr_pose = cooccur(pose_literals(prop))
    print("=== 挖一半场景，验另一半（留出 z 的零期望是 0±1）===")
    global RNG
    RNG = np.random.default_rng(1)
    r1, rk, rb = one_family(fr_route, fail, p_fail, scene, "路由合取 (32896 条)", lit_name)
    RNG = np.random.default_rng(1)
    p1, pk, pb = one_family(fr_pose, fail, p_fail, scene, "位姿合取 (528 条)", pose_name)

    fs = fail.copy()
    for s in np.unique(scene):
        m = np.flatnonzero(scene == s)
        fs[m] = fail[m][np.random.default_rng(7).permutation(len(m))]
    RNG = np.random.default_rng(1)
    one_family(fr_route, fs, p_fail, scene, "对照：打乱标签的路由", lit_name)

    print("\n=== 每次划分挖出来的最强路由规则（看它稳不稳定）===")
    print("  规则                     挖集 z   留出 z")
    for a, b, za, zb in rb[:10]:
        print("  %-24s  %5.2f   %+5.2f" % ("%s & %s" % (a, b), za, zb))
    same = len({(a, b) for a, b, _, _ in rb})
    print("  %d 次划分挖出 %d 条不同的规则" % (len(rb), same))

    print("\n=== 位姿谓词那边挖出来的 ===")
    for a, b, za, zb in pb[:6]:
        print("  %-24s  %5.2f   %+5.2f" % ("%s & %s" % (a, b), za, zb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
