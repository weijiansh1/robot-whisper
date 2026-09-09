#!/usr/bin/env python3
"""Is "only t34 is significant" a real timing fact or just low power?

Twelve initial states carry both outcome classes, so a cluster bootstrap over
them has very little power, and testing four horizons separately throws away
the fact that all four point estimates point the same way.

Three sharper readings of the same scores:
  A  per-state decomposition - is the increment consistent across states?
  B  pooled contrasts - average increment over horizons, and its trend
  C  onset-aligned - the increment at a fixed offset from each rollout's own
     trap onset instead of a fixed control step
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (12, 20, 27, 34)
DRAWS = 20000
MOE = ("route_identity", "route_geometry", "route_temporal",
       "hidden_identity", "hidden_structure", "hidden_temporal")


def extract_action(run, episodes):
    out = np.empty((len(episodes), len(core.HORIZONS), 70 * 5), np.float32)
    for ep in episodes:
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            act = np.asarray(p["actions"][: max(core.HORIZONS) + 1], np.float32).reshape(-1, 70)
        for hi, h in enumerate(core.HORIZONS):
            out[ep.index, hi] = core.physical_history(act[h - core.HISTORY + 1: h + 1])
    return out


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def delta_on(labels, L, R, blocks):
    """Pair-weighted AUC(L) - AUC(R) over a list of index groups."""
    a = b = 0.0
    den = 0
    for i in blocks:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        a += auc(y, L[i]) * w
        b += auc(y, R[i]) * w
        den += w
    return (a - b) / den if den else np.nan


def boot(fn, blocks, seed=20260827):
    rng = np.random.default_rng(seed)
    v = []
    while len(v) < DRAWS:
        ch = rng.integers(0, len(blocks), len(blocks))
        d = [rng.choice(blocks[a], len(blocks[a]), replace=True) for a in ch]
        x = fn(d)
        if np.isfinite(x):
            v.append(x)
    v = np.asarray(v)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975)), float((v <= 0).mean())


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
    labels, included, onset, _ = core.build_targets(episodes, sims)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)
    folds = core.double_holdout_folds(state, seeds, included)

    groups = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    groups = [b for b in groups if len(np.unique(labels[b])) == 2]
    print("usable initial states (both classes): %d of %d\n"
          % (len(groups), len(np.unique(state))), flush=True)

    sc = {}
    for h in HORIZONS:
        idx = core.HORIZONS.index(h)
        for name, fam in (("base", ("proprio", "action")),
                          ("moe", ("proprio", "action") + MOE)):
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            sc[(h, name)] = core.predict(m, labels)

    # ---- A: per-state decomposition -------------------------------------
    print("A  per-state increment (MoE over runtime baseline)")
    print("%6s %4s %4s %s" % ("state", "nS", "nF", "".join("%9s" % f"t{h}" for h in HORIZONS)))
    per_state = {}
    for gi, grp in enumerate(groups):
        row = []
        for h in HORIZONS:
            row.append(delta_on(labels, sc[(h, "moe")], sc[(h, "base")], [grp]))
        per_state[int(state[grp[0]])] = row
        y = labels[grp]
        print("%6d %4d %4d %s" % (state[grp[0]], int((1 - y).sum()), int(y.sum()),
                                  "".join("%+9.3f" % v for v in row)))
    arr = np.array(list(per_state.values()))
    print("%6s %4s %4s %s" % ("mean", "", "", "".join("%+9.3f" % v for v in arr.mean(0))))
    print("%6s %4s %4s %s" % ("pos/12", "", "", "".join("%9s" % f"{int((arr[:,i]>0).sum())}/{len(arr)}"
                                                        for i in range(len(HORIZONS)))))

    # ---- B: pooled over horizons ----------------------------------------
    print("\nB  pooled contrasts (cluster bootstrap over the same 12 states)")
    out_b = []
    for label, hs in (("mean over t12-t34", HORIZONS),
                      ("mean over t20-t34", (20, 27, 34)),
                      ("mean over t12/t20/t27 (pre-t34 only)", (12, 20, 27))):
        fn = lambda b, hs=hs: float(np.mean(
            [delta_on(labels, sc[(h, "moe")], sc[(h, "base")], b) for h in hs]))
        pt = fn(groups)
        lo, hi, p = boot(fn, groups)
        flag = "  excludes 0" if lo > 0 else ""
        print("  %-38s %+7.3f  [%+.3f, %+.3f]  p=%.4f%s" % (label, pt, lo, hi, p, flag))
        out_b.append({"contrast": label, "delta": pt, "ci": [lo, hi], "p_one_sided": p})

    # ---- C: onset-aligned ------------------------------------------------
    print("\nC  increment at a fixed offset from each rollout's own trap onset")
    print("   (failures shifted by their onset; successes by a peer's onset)")
    rng = np.random.default_rng(20260827)
    peer = {}
    for v in np.unique(state):
        cand = [onset[k] for k in np.flatnonzero(state == v) if onset[k] >= 0]
        peer[int(v)] = cand
    out_c = []
    for off in (-8, -4, 0, +4, +8, +12):
        # choose, per episode, the horizon closest to its own onset+off
        pick = {}
        for e in range(len(episodes)):
            base = onset[e] if onset[e] >= 0 else (
                int(rng.choice(peer[int(state[e])])) if peer[int(state[e])] else -1)
            if base < 0:
                continue
            target = base + off
            pick[e] = min(HORIZONS, key=lambda h: abs(h - target))
        L = np.full(len(labels), np.nan)
        R = np.full(len(labels), np.nan)
        for e, h in pick.items():
            L[e] = sc[(h, "moe")][e]
            R[e] = sc[(h, "base")][e]
        ok = np.isfinite(L) & np.isfinite(R) & included
        grp = [g[np.isin(g, np.flatnonzero(ok))] for g in groups]
        grp = [g for g in grp if len(np.unique(labels[g])) == 2]
        if len(grp) < 4:
            continue
        fn = lambda b: delta_on(labels, L, R, b)
        pt = fn(grp)
        lo, hi, p = boot(fn, grp)
        flag = "  excludes 0" if lo > 0 else ""
        print("  onset%+3d  %+7.3f  [%+.3f, %+.3f]  p=%.4f  (%d states)%s"
              % (off, pt, lo, hi, p, len(grp), flag))
        out_c.append({"offset": off, "delta": pt, "ci": [lo, hi],
                      "p_one_sided": p, "states": len(grp)})

    args.out.write_text(json.dumps(
        {"draws": DRAWS, "usable_states": len(groups),
         "per_state": {str(k): v for k, v in per_state.items()},
         "pooled": out_b, "onset_aligned": out_c}, indent=2) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
