#!/usr/bin/env python3
"""Does the detector see a trap, or does it see failure in general?

The detector is trained exactly as everywhere else - stasis traps against
successes - but here every rollout is scored, including the 19 failures that
were excluded from the main analysis because they are not stasis traps
(reached-then-lost, misplaced). Those 19 never enter any training fold.

If the detector separates them from successes about as well as it separates
stasis traps, it is a general failure detector. If it cannot tell them apart,
it is specifically a "the robot is stuck" readout, and the other tasks - whose
dominant failure modes are different - should not be expected to transfer.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (20, 27, 34)
DRAWS = 20000
MOE = ("route_identity", "route_geometry", "route_temporal",
       "hidden_identity", "hidden_structure", "hidden_temporal")


def extract_action(run, episodes):
    out = np.empty((len(episodes), len(core.HORIZONS), 70 * 5), np.float32)
    for ep in episodes:
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            a = np.asarray(p["actions"][: max(core.HORIZONS) + 1], np.float32).reshape(-1, 70)
        for hi, h in enumerate(core.HORIZONS):
            out[ep.index, hi] = core.physical_history(a[h - core.HISTORY + 1: h + 1])
    return out


def folds_scoring_everyone(state, seed, included):
    """Train only on included rollouts; put every rollout in some test fold."""
    sg = [np.asarray(x) for x in np.array_split(np.unique(state), 4)]
    dg = [np.asarray(x) for x in np.array_split(np.unique(seed), 4)]
    out = []
    for hs in sg:
        for ds in dg:
            test = np.isin(state, hs) & np.isin(seed, ds)
            train = included & ~np.isin(state, hs) & ~np.isin(seed, ds)
            out.append((np.flatnonzero(train), np.flatnonzero(test), {}))
    seen = np.zeros(len(state), int)
    for _, t, _ in out:
        seen[t] += 1
    assert (seen == 1).all(), "every rollout must be tested exactly once"
    return out


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(lab, sc, groups):
    num = den = 0.0
    for i in groups:
        y = lab[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, sc[i]) * w
        den += w
    return num / den if den else np.nan


def boot(fn, groups, seed=20260827):
    rng = np.random.default_rng(seed)
    v = []
    while len(v) < DRAWS:
        ch = rng.integers(0, len(groups), len(groups))
        d = [rng.choice(groups[a], len(groups[a]), replace=True) for a in ch]
        x = fn(d)
        if np.isfinite(x):
            v.append(x)
    v = np.asarray(v)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=core.RUN)
    ap.add_argument("--cache", type=pathlib.Path,
                    default=HERE / "token-dynamics/early_structure_features.npz")
    ap.add_argument("--modes", type=pathlib.Path, default=HERE / "failure_modes.json")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    episodes = core.load_episodes(args.run)
    sims = core.load_sim(args.run, episodes)
    stasis, included, _, _ = core.build_targets(episodes, sims)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)
    fail = np.asarray([e.failure for e in episodes], np.int8)

    other = (fail == 1) & (stasis == 0)          # the 19 excluded failures
    succ = fail == 0
    print("成功 %d ｜ 停滞陷入 %d ｜ 非停滞失败 %d"
          % (succ.sum(), stasis.sum(), other.sum()), flush=True)

    modes = json.loads(args.modes.read_text())
    key = [k for k in modes if "KITCHEN_SCENE8" in k][0]
    by_ep = {d["episode"]: d["mode"] for d in modes[key]["detail"]}
    from collections import Counter
    print("这 19 条的物理模式:", dict(Counter(by_ep[e] for e in np.flatnonzero(other))))

    folds = folds_scoring_everyone(state, seeds, included.astype(bool))
    sc = {}
    for h in HORIZONS:
        idx = core.HORIZONS.index(h)
        for name, fam in (("base", ("proprio", "action")),
                          ("moe", ("proprio", "action") + MOE)):
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            sc[(h, name)] = core.predict(m, stasis)      # trained on stasis vs success
        print("  scored t%d" % h, flush=True)

    def groups_for(mask):
        g = [np.flatnonzero((succ | mask) & (state == v)) for v in np.unique(state)]
        return [x for x in g if len(np.unique(mask[x])) == 2]

    rows = []
    for label, mask in (("停滞陷入 vs 成功", stasis.astype(bool)),
                        ("非停滞失败 vs 成功", other)):
        gs = groups_for(mask)
        lab = mask.astype(int)
        print("\n%s   可用 state %d" % (label, len(gs)))
        for h in HORIZONS:
            a_b = weighted(lab, sc[(h, "base")], gs)
            a_m = weighted(lab, sc[(h, "moe")], gs)
            rows.append({"contrast": label, "horizon": h,
                         "base_auc": a_b, "moe_auc": a_m, "states": len(gs)})
            print("  t%-3d 非特权基线 %.3f   加 MoE %.3f" % (h, a_b, a_m))
        fn = lambda b: float(np.mean([weighted(lab, sc[(h, "moe")], b) for h in HORIZONS]))
        pt = fn(gs)
        lo, hi = boot(fn, gs)
        print("  t20–t34 合并 加 MoE AUC %.3f  95%% CI [%.3f, %.3f]%s"
              % (pt, lo, hi, "  排除 0.5" if lo > .5 else "  含 0.5"))
        rows.append({"contrast": label, "horizon": "pooled", "moe_auc": pt,
                     "ci": [lo, hi], "states": len(gs)})

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
