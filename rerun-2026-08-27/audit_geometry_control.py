#!/usr/bin/env python3
"""Does the MoE increment survive a physically-featurised control?

The published physical baseline feeds raw simulator state through PCA and a
linear ridge, so the readout cannot form relative geometry: gripper-to-object
distance, grasp proxies or object tilt are all nonlinear in raw qpos.  The HB
hidden state has already computed such quantities, so the reported
`sim + MoE` minus `sim` increment may be "nonlinear features of the physical
state" rather than "information the physical state does not contain".

This rebuilds the physical block with explicit relative geometry and reruns the
same paired contrast under the identical folds, readout and bootstrap.

state  : [eef_pos(3), eef_axis_angle(3), finger(2)]
sim    : [time, robot qpos(9), obj1 pos(3) quat(4), obj2 pos(3) quat(4), ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
O1, O2 = slice(10, 13), slice(17, 20)
Q1, Q2 = slice(13, 17), slice(20, 24)
HORIZONS = (12, 27, 34)
DRAWS = 20000


def tilt(quat: np.ndarray) -> np.ndarray:
    """z-component of the body z-axis: 1 means upright, -1 fully inverted."""
    w, x, y, z = quat.T
    return 1.0 - 2.0 * (x * x + y * y)


def rich_features(state: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """[T, D] raw physical state plus the relative geometry a linear map cannot form."""
    eef = state[:, :3]
    grip = (state[:, 6] - state[:, 7])[:, None]
    p1, p2 = sim[:, O1], sim[:, O2]
    d1, d2 = p1 - eef, p2 - eef
    d12 = p2 - p1
    n1 = np.linalg.norm(d1, axis=1, keepdims=True)
    n2 = np.linalg.norm(d2, axis=1, keepdims=True)
    n12 = np.linalg.norm(d12, axis=1, keepdims=True)
    near = np.minimum(n1, n2)
    return np.column_stack([
        sim[:, 1:],                                   # everything the old control had
        eef, state[:, 3:6], grip,
        d1, d2, d12, n1, n2, n12, near,
        tilt(sim[:, Q1])[:, None], tilt(sim[:, Q2])[:, None],
        grip * n1, grip * n2,                         # grasp proxies
        np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=1)[:, None],
        np.linalg.norm(np.diff(p1, axis=0, prepend=p1[:1]), axis=1)[:, None],
        np.linalg.norm(np.diff(p2, axis=0, prepend=p2[:1]), axis=1)[:, None],
    ]).astype(np.float32)


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def pair_weighted(labels, scores, blocks):
    num = 0.0
    den = 0
    for i in blocks:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, scores[i]) * w
        den += w
    return num / den if den else float("nan")


def paired(labels, left, right, blocks):
    a = b = 0.0
    den = 0
    for i in blocks:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        a += auc(y, left[i]) * w
        b += auc(y, right[i]) * w
        den += w
    return (a - b) / den if den else float("nan")


def cluster_ci(fn, blocks, seed=20260827):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < DRAWS:
        chosen = rng.integers(0, len(blocks), len(blocks))
        drawn = [rng.choice(blocks[a], len(blocks[a]), replace=True) for a in chosen]
        v = fn(drawn)
        if np.isfinite(v):
            out.append(v)
    return tuple(float(x) for x in np.quantile(out, [0.025, 0.975]))


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
    state = np.asarray([e.state for e in episodes], dtype=np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], dtype=np.int16)

    rich = np.empty((len(episodes), len(core.HORIZONS),
                     rich_features(np.zeros((3, 8), np.float32),
                                   np.zeros((3, 47), np.float32)).shape[1] * 5),
                    dtype=np.float32)
    for ep in episodes:
        with np.load(core.episode_path(args.run, ep), allow_pickle=True) as p:
            st = np.asarray(p["state"][: max(core.HORIZONS) + 1], dtype=np.float32)
        sm = sims[ep.index][: max(core.HORIZONS) + 1]
        feat = rich_features(st, sm)
        for hi, h in enumerate(core.HORIZONS):
            lo = h - core.HISTORY + 1
            rich[ep.index, hi] = core.physical_history(feat[lo:h + 1])
    blocks["sim_rich"] = rich
    print("sim_rich block: %s (raw sim block was %s)"
          % (rich.shape, blocks["sim"].shape), flush=True)

    folds = core.double_holdout_folds(state, seeds, included)
    keep = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    keep = [b for b in keep if len(np.unique(labels[b])) == 2]

    MOE = ("route_identity", "route_geometry", "route_temporal",
           "hidden_identity", "hidden_structure", "hidden_temporal")
    FAMS = {
        "sim": ("sim",),
        "sim_rich": ("sim_rich",),
        "sim_plus_moe": ("sim",) + MOE,
        "sim_rich_plus_moe": ("sim_rich",) + MOE,
    }
    scores, rows = {}, []
    for hi, h in enumerate(core.HORIZONS):
        if h not in HORIZONS:
            continue
        idx = core.HORIZONS.index(h)
        for name, fam in FAMS.items():
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 977 * idx)
            scores[(h, name)] = core.predict(m, labels)
            a = pair_weighted(labels, scores[(h, name)], keep)
            rows.append({"horizon": h, "family": name, "auc": a})
            print("  t%-3d %-20s AUC %.3f" % (h, name, a), flush=True)

    print("\npaired increments (cluster bootstrap over initial states, %d draws)" % DRAWS)
    print("%-46s %8s  %s" % ("contrast", "delta", "95% CI"))
    out = []
    for h in HORIZONS:
        for label, l, r in (
            ("MoE over raw physical", "sim_plus_moe", "sim"),
            ("MoE over geometry-featurised physical", "sim_rich_plus_moe", "sim_rich"),
            ("geometry featurisation alone", "sim_rich", "sim"),
        ):
            L, R = scores[(h, l)], scores[(h, r)]
            pt = paired(labels, L, R, keep)
            lo, hi_ = cluster_ci(lambda b: paired(labels, L, R, b), keep)
            flag = "  excludes 0" if (lo > 0 or hi_ < 0) else ""
            print("%-46s %+8.3f  [%+.3f, %+.3f]%s" % (label + f"  @ t{h}", pt, lo, hi_, flag))
            out.append({"horizon": h, "contrast": label, "delta": pt,
                        "ci": [lo, hi_], "excludes_zero": bool(lo > 0 or hi_ < 0)})

    args.out.write_text(json.dumps(
        {"draws": DRAWS, "auc": rows, "increments": out,
         "rich_dim": int(rich.shape[2]), "raw_dim": int(blocks["sim"].shape[2])},
        indent=2) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
