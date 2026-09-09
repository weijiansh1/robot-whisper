#!/usr/bin/env python3
"""Spread the result over every case instead of two hand-picked exemplars.

Cross-task validation is structurally blocked: in every other task the failures
run to the step cap while the successes finish far earlier, so the cohort in
which all rollouts are still active ends at t8-t16 - before the window where
the effect appears at all.  What is available is the full within-task spread.

Exports, for every initial state and every rollout:
  * the per-state MoE increment at each horizon
  * leave-one-state-out stability of the pooled estimate
  * per-rollout cross-fitted scores, so the viewer can show all 512 curves
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (7, 12, 20, 27, 34)
POOL = (20, 27, 34)
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


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def delta(labels, L, R, groups):
    a = b = 0.0
    den = 0
    for i in groups:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        a += auc(y, L[i]) * w
        b += auc(y, R[i]) * w
        den += w
    return (a - b) / den if den else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=core.RUN)
    ap.add_argument("--cache", type=pathlib.Path,
                    default=HERE / "token-dynamics/early_structure_features.npz")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    episodes = core.load_episodes(args.run)
    sims = core.load_sim(args.run, episodes)
    labels, included, onset, meta = core.build_targets(episodes, sims)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)
    folds = core.double_holdout_folds(state, seeds, included)

    groups = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    groups = [g for g in groups if len(np.unique(labels[g])) == 2]

    sc = {}
    for h in HORIZONS:
        idx = core.HORIZONS.index(h)
        for name, fam in (("base", ("proprio", "action")),
                          ("moe", ("proprio", "action") + MOE)):
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            sc[(h, name)] = core.predict(m, labels)
        print("  scored t%d" % h, flush=True)

    # per-state curves
    per_state = []
    for g in groups:
        y = labels[g]
        per_state.append({
            "state": int(state[g[0]]),
            "n_success": int((1 - y).sum()), "n_stasis": int(y.sum()),
            "pairs": int(y.sum()) * int((1 - y).sum()),
            "delta": [round(float(delta(labels, sc[(h, "moe")], sc[(h, "base")], [g])), 4)
                      for h in HORIZONS],
            "base_auc": [round(float(auc(y, sc[(h, "base")][g])), 4) for h in HORIZONS],
            "moe_auc": [round(float(auc(y, sc[(h, "moe")][g])), 4) for h in HORIZONS],
        })
    order = np.argsort([-r["pairs"] for r in per_state])
    per_state = [per_state[i] for i in order]

    def pooled(gs):
        return float(np.mean([delta(labels, sc[(h, "moe")], sc[(h, "base")], gs) for h in POOL]))

    full = pooled(groups)
    rng = np.random.default_rng(20260827)
    bs = []
    while len(bs) < DRAWS:
        ch = rng.integers(0, len(groups), len(groups))
        v = pooled([rng.choice(groups[a], len(groups[a]), replace=True) for a in ch])
        if np.isfinite(v):
            bs.append(v)
    bs = np.asarray(bs)
    ci = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]

    loo = []
    for k in range(len(groups)):
        rest = [g for j, g in enumerate(groups) if j != k]
        loo.append({"dropped": int(state[groups[k][0]]), "delta": round(pooled(rest), 4)})
    lv = [r["delta"] for r in loo]
    print("\npooled t20-t34 increment  %+.3f  CI [%+.3f, %+.3f]" % (full, ci[0], ci[1]))
    print("leave-one-state-out range %+.3f .. %+.3f   (all %s)"
          % (min(lv), max(lv), "positive" if min(lv) > 0 else "NOT all positive"))

    # restricted to states with a decent number of each class
    strong = [g for g in groups
              if min(int(labels[g].sum()), int((1 - labels[g]).sum())) >= 5]
    sfull = pooled(strong)
    bs2 = []
    rng = np.random.default_rng(20260827)
    while len(bs2) < DRAWS:
        ch = rng.integers(0, len(strong), len(strong))
        v = pooled([rng.choice(strong[a], len(strong[a]), replace=True) for a in ch])
        if np.isfinite(v):
            bs2.append(v)
    bs2 = np.asarray(bs2)
    sci = [float(np.quantile(bs2, .025)), float(np.quantile(bs2, .975))]
    print("restricted to %d states with >=5 per class: %+.3f  CI [%+.3f, %+.3f]  p=%.4f"
          % (len(strong), sfull, sci[0], sci[1], float((bs2 <= 0).mean())))

    rollouts = []
    for e in range(len(episodes)):
        if not included[e]:
            continue
        rollouts.append({
            "state": int(state[e]), "stasis": int(labels[e]), "onset": int(onset[e]),
            "base": [round(float(sc[(h, "base")][e]), 4) for h in HORIZONS],
            "moe": [round(float(sc[(h, "moe")][e]), 4) for h in HORIZONS],
        })

    args.out.write_text(json.dumps({
        "horizons": list(HORIZONS), "pool": list(POOL), "draws": DRAWS,
        "n_states": len(groups), "n_rollouts": len(rollouts),
        "target": {k: meta[k] for k in ("n_success", "n_failure", "n_stasis_trap",
                                        "n_ambiguous_failure_excluded")},
        "pooled": {"delta": full, "ci": ci, "p_one_sided": float((bs <= 0).mean())},
        "restricted": {"states": len(strong), "delta": sfull, "ci": sci,
                       "p_one_sided": float((bs2 <= 0).mean())},
        "leave_one_out": loo, "per_state": per_state, "rollouts": rollouts,
    }, separators=(",", ":")))
    print("\nwrote %s (%.0f KB)" % (args.out, args.out.stat().st_size / 1024))


if __name__ == "__main__":
    main()
