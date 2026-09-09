#!/usr/bin/env python3
"""Is MoE state useful for runtime failure monitoring?

The simulator-state controls use privileged information - object poses and
joint angles - that a deployed robot does not have.  For runtime monitoring the
honest baseline is what the robot can actually read at inference time: its own
proprioception and the action chunk it just emitted.  Image features are not a
separate baseline here, since obtaining them means running the same network.

Same folds, same readout, same cluster bootstrap as every other contrast.
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
ROUTE = ("route_identity", "route_geometry", "route_temporal")


def extract_action(run: pathlib.Path, episodes) -> np.ndarray:
    out = np.empty((len(episodes), len(core.HORIZONS), 10 * 7 * 5), dtype=np.float32)
    for ep in episodes:
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            act = np.asarray(p["actions"][: max(core.HORIZONS) + 1],
                             dtype=np.float32).reshape(-1, 70)
        for hi, h in enumerate(core.HORIZONS):
            out[ep.index, hi] = core.physical_history(act[h - core.HISTORY + 1: h + 1])
    return out


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(labels, scores, blocks):
    num, den = 0.0, 0
    for i in blocks:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, scores[i]) * w
        den += w
    return num / den


def paired(labels, L, R, blocks):
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
    return (a - b) / den


def ci(fn, blocks, seed=20260827):
    rng = np.random.default_rng(seed)
    v = []
    while len(v) < DRAWS:
        ch = rng.integers(0, len(blocks), len(blocks))
        d = [rng.choice(blocks[a], len(blocks[a]), replace=True) for a in ch]
        x = fn(d)
        if np.isfinite(x):
            v.append(x)
    return tuple(float(q) for q in np.quantile(v, [0.025, 0.975]))


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
    labels, included, _, _ = core.build_targets(episodes, sims)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([e.state for e in episodes], dtype=np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], dtype=np.int16)
    folds = core.double_holdout_folds(state, seeds, included)
    keep = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    keep = [b for b in keep if len(np.unique(labels[b])) == 2]

    FAMS = {
        "runtime (proprio+action)": ("proprio", "action"),
        "runtime + MoE": ("proprio", "action") + MOE,
        "runtime + route only": ("proprio", "action") + ROUTE,
        "MoE alone": MOE,
        "privileged sim": ("sim",),
    }
    scores, table = {}, []
    for h in HORIZONS:
        idx = core.HORIZONS.index(h)
        for name, fam in FAMS.items():
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            scores[(h, name)] = core.predict(m, labels)
            a = weighted(labels, scores[(h, name)], keep)
            table.append({"horizon": h, "family": name, "auc": a})
        print("t%-3d " % h + "  ".join(
            "%s %.3f" % (n.split(" (")[0], weighted(labels, scores[(h, n)], keep))
            for n in FAMS), flush=True)

    print("\npaired increments over the deployable baseline (%d-draw cluster bootstrap)" % DRAWS)
    print("%-52s %8s  %s" % ("contrast", "delta", "95% CI"))
    out = []
    for h in HORIZONS:
        for label, l, r in (
            ("MoE over runtime-available signals", "runtime + MoE", "runtime (proprio+action)"),
            ("route only over runtime-available signals", "runtime + route only", "runtime (proprio+action)"),
        ):
            L, R = scores[(h, l)], scores[(h, r)]
            pt = paired(labels, L, R, keep)
            lo, hi = ci(lambda b: paired(labels, L, R, b), keep)
            flag = "  excludes 0" if (lo > 0 or hi < 0) else ""
            print("%-52s %+8.3f  [%+.3f, %+.3f]%s" % (label + f"  @ t{h}", pt, lo, hi, flag))
            out.append({"horizon": h, "contrast": label, "delta": pt,
                        "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)})

    args.out.write_text(json.dumps({"draws": DRAWS, "auc": table, "increments": out},
                                   indent=2) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
