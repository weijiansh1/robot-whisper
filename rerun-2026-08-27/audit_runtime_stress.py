#!/usr/bin/env python3
"""Stress the runtime-monitoring result the same way the sim result was stressed.

The simulator increment collapsed once the linear readout was handed the
relative geometry it could not compute itself.  The analogous attack here is to
hand the deployable baseline explicit "I am spinning in place" quantities:
end-effector path length and straightness, action-chunk self-similarity, and the
residual between commanded and achieved displacement.  All are computable from
proprioception and the emitted actions alone - no privileged state.

If the MoE increment survives this, it is an information gap rather than a
readout gap.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (27, 34)
DRAWS = 20000
MOE = ("route_identity", "route_geometry", "route_temporal",
       "hidden_identity", "hidden_structure", "hidden_temporal")


def runtime_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """[T, D] from proprioception and emitted actions only."""
    eef = state[:, :3]
    rot = state[:, 3:6]
    grip = (state[:, 6] - state[:, 7])[:, None]
    step = np.diff(eef, axis=0, prepend=eef[:1])
    speed = np.linalg.norm(step, axis=1, keepdims=True)

    flat = action.reshape(len(action), -1)                     # [T, 70]
    prev = np.vstack([flat[:1], flat[:-1]])
    dot = (flat * prev).sum(1)
    denom = np.linalg.norm(flat, axis=1) * np.linalg.norm(prev, axis=1)
    self_sim = np.divide(dot, denom, out=np.zeros(len(flat)), where=denom > 1e-9)[:, None]
    chunk_delta = np.linalg.norm(flat - prev, axis=1, keepdims=True)

    commanded = action[:, :, :3].sum(axis=1)                   # [T, 3] intended xyz motion
    residual = np.linalg.norm(commanded - step, axis=1, keepdims=True)

    # trailing-window path length vs net displacement: straightness of the wrist
    path = np.zeros((len(eef), 1))
    net = np.zeros((len(eef), 1))
    for t in range(len(eef)):
        lo = max(0, t - core.HISTORY + 1)
        path[t] = np.linalg.norm(np.diff(eef[lo:t + 1], axis=0), axis=1).sum()
        net[t] = np.linalg.norm(eef[t] - eef[lo])
    straight = np.divide(net, path, out=np.zeros_like(net), where=path > 1e-9)

    return np.column_stack([
        eef, rot, grip, step, speed,
        action.mean(axis=1), action.std(axis=1), np.linalg.norm(flat, axis=1, keepdims=True),
        self_sim, chunk_delta, commanded, np.linalg.norm(commanded, axis=1, keepdims=True),
        residual, path, net, straight,
    ]).astype(np.float32)


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(labels, sc, blocks):
    num, den = 0.0, 0
    for i in blocks:
        y = labels[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, sc[i]) * w
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
    state = np.asarray([e.state for e in episodes], dtype=np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], dtype=np.int16)

    probe = runtime_features(np.zeros((9, 8), np.float32), np.zeros((9, 10, 7), np.float32))
    rich = np.empty((len(episodes), len(core.HORIZONS), probe.shape[1] * 5), np.float32)
    act = np.empty((len(episodes), len(core.HORIZONS), 70 * 5), np.float32)
    for ep in episodes:
        with np.load(core.episode_path(args.run, ep), allow_pickle=False) as p:
            st = np.asarray(p["state"][: max(core.HORIZONS) + 1], np.float32)
            ac = np.asarray(p["actions"][: max(core.HORIZONS) + 1], np.float32)
        feat = runtime_features(st, ac)
        flat = ac.reshape(len(ac), -1)
        for hi, h in enumerate(core.HORIZONS):
            lo = h - core.HISTORY + 1
            rich[ep.index, hi] = core.physical_history(feat[lo:h + 1])
            act[ep.index, hi] = core.physical_history(flat[lo:h + 1])
    blocks["action"] = act
    blocks["runtime_rich"] = rich
    print("runtime_rich block %s" % (rich.shape,), flush=True)

    folds = core.double_holdout_folds(state, seeds, included)
    keep = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    keep = [b for b in keep if len(np.unique(labels[b])) == 2]

    FAMS = {
        "runtime": ("proprio", "action"),
        "runtime_rich": ("proprio", "action", "runtime_rich"),
        "runtime_rich + MoE": ("proprio", "action", "runtime_rich") + MOE,
    }
    sc = {}
    for h in HORIZONS:
        idx = core.HORIZONS.index(h)
        for name, fam in FAMS.items():
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            sc[(h, name)] = core.predict(m, labels)
        print("t%-3d " % h + "  ".join("%s %.3f" % (n, weighted(labels, sc[(h, n)], keep))
                                       for n in FAMS), flush=True)

    print("\npaired increments (%d-draw cluster bootstrap)" % DRAWS)
    out = []
    for h in HORIZONS:
        for label, l, r in (
            ("MoE over stress-hardened runtime baseline", "runtime_rich + MoE", "runtime_rich"),
            ("stuck-detector featurisation alone", "runtime_rich", "runtime"),
        ):
            L, R = sc[(h, l)], sc[(h, r)]
            pt = paired(labels, L, R, keep)
            lo, hi = ci(lambda b: paired(labels, L, R, b), keep)
            flag = "  excludes 0" if (lo > 0 or hi < 0) else ""
            print("%-46s %+7.3f  [%+.3f, %+.3f]%s" % (label + f" @ t{h}", pt, lo, hi, flag))
            out.append({"horizon": h, "contrast": label, "delta": pt, "ci": [lo, hi],
                        "excludes_zero": bool(lo > 0 or hi < 0)})
    args.out.write_text(json.dumps({"draws": DRAWS, "increments": out,
        "auc": {f"t{h}|{n}": weighted(labels, sc[(h, n)], keep)
                for h in HORIZONS for n in FAMS}}, indent=2) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
