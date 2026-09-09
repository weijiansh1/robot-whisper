#!/usr/bin/env python3
"""Does the rule say anything the arm's pose does not already say?

The mined conjunctions transfer -- held-out z +2.17 +- 1.37 against a permuted
ceiling of +0.49 -- and `L2:e5` recurs in most splits.  But the state token's
routing is ~96% a nonlinear function of the 8 proprio numbers, so "L2:e5 &
L5:e8 fires" may be nothing more than a coarse way of saying "the arm is here".

Matched pairs settle it.  For each control step where the rule fires, find the
nearest control step in the *same scene*, at a comparable phase of the episode,
where the arm is in almost the same pose and the rule does *not* fire.  Then
compare the two episodes' outcomes.  If the routing rule carries anything of its
own, the firing side fails more often than its pose-matched twin.

Two numbers come out, and both matter:

  separability  the fraction of firing steps with no pose-matched twin at all.
                A rule that is exactly a pose predicate has no twins -- the
                match failing IS the finding, not a bug.
  excess        among the steps that do have a twin, the paired difference in
                failure rate, bootstrapped over episodes since one episode
                contributes many steps.

Run for the recurring rule and for the top-20 pooled, plus a shuffled-rule arm.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_rules import HB_LAYER, HUB, cooccur, lit_name, load, score  # noqa: E402

PHASE = 3          # twin must sit within +-3 control steps
RNG = np.random.default_rng(0)


def matched(hit, prop, scene, fail, dmax_q=0.02):
    """Pair each firing (episode, step) with the closest non-firing twin."""
    n, w, _ = prop.shape
    Z = (prop - prop.reshape(-1, 8).mean(0)) / (prop.reshape(-1, 8).std(0) + 1e-9)
    ep = np.repeat(np.arange(n), w)
    st = np.tile(np.arange(w), n)
    F = Z.reshape(-1, 8)
    h = hit.ravel()
    sc = scene[ep]
    pairs, dists = [], []
    for s in np.unique(scene):
        m = sc == s
        gi = np.flatnonzero(m & h)
        gj = np.flatnonzero(m & ~h)
        if not len(gi) or not len(gj):
            continue
        D = ((F[gi][:, None, :] - F[gj][None, :, :]) ** 2).sum(-1)
        D[np.abs(st[gi][:, None] - st[gj][None, :]) > PHASE] = np.inf
        k = np.argmin(D, 1)
        d = D[np.arange(len(gi)), k]
        for a, b, dd in zip(gi, gj[k], d):
            if np.isfinite(dd):
                pairs.append((ep[a], ep[b]))
                dists.append(np.sqrt(dd))
    if not pairs:
        return None
    pairs = np.array(pairs)
    dists = np.array(dists)
    # a "twin" only counts if the poses really are close; the threshold is the
    # 2nd percentile of within-scene, phase-matched distances at large
    thr = np.quantile(dists, dmax_q) if len(dists) > 50 else np.inf
    keep = dists <= max(thr, 0.25)
    return pairs, dists, keep


def report(name, hit, prop, scene, fail):
    out = matched(hit, prop, scene, fail)
    if out is None:
        print("  %-26s 无法配对" % name)
        return
    pairs, dists, keep = out
    n_fire = int(hit.sum())
    a, b = fail[pairs[keep, 0]], fail[pairs[keep, 1]]
    diff = a.astype(float).mean() - b.astype(float).mean()
    # bootstrap over the firing episode, which is the clustering unit
    eps = np.unique(pairs[keep, 0])
    bs = []
    for _ in range(2000):
        pick = RNG.choice(eps, len(eps))
        m = np.concatenate([np.flatnonzero(pairs[keep, 0] == e) for e in pick])
        bs.append(a[m].mean() - b[m].mean())
    lo, hi = np.quantile(bs, [.025, .975])
    print("  %-26s 触发 %5d 步，配上孪生 %5d 步（位姿距离中位 %.2f sd）"
          % (name, n_fire, keep.sum(), np.median(dists[keep])))
    print("      触发侧失败率 %.3f   孪生侧 %.3f   差 %+.3f  [%+.3f, %+.3f]  %s"
          % (a.mean(), b.mean(), diff, lo, hi,
             "显著" if lo > 0 or hi < 0 else "含 0"))


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
    print("%s，%d 局，窗口 %d 步，同场景 + 相位 ±%d 步 + 位姿最近邻配对\n"
          % (task, len(y), D["win"], PHASE))

    fires = cooccur(M)
    sup, obs, exp, z = score(fires, fail, p_fail)
    ok = (sup >= 20) & (sup <= len(y) - 5)
    ok[np.tril_indices(256, -1)] = False
    order = np.argsort(np.where(ok, z, -np.inf).ravel())[::-1]

    print("=== 位姿配对后，规则还剩多少 ===")
    for f in order[:5]:
        a, b = np.unravel_index(f, z.shape)
        hit = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
        report("%s & %s" % (lit_name(a), lit_name(b)), hit, prop, scene, fail)

    pool = np.zeros(M.shape[:2], bool)
    for f in order[:20]:
        a, b = np.unravel_index(f, z.shape)
        pool |= (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
    report("前 20 条并集", pool, prop, scene, fail)

    # control: a rule with the same firing rate but drawn at random
    rate = pool.mean()
    rnd = RNG.random(M.shape[:2]) < rate
    report("对照：同频率随机触发", rnd, prop, scene, fail)

    # what does the recurring literal L2:e5 do on its own?
    i5 = HB_LAYER.index(2) * 32 + 5
    report("单独 L2:e5", M[:, :, i5] > 0.5, prop, scene, fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
