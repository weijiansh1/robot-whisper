#!/usr/bin/env python3
"""Build the sticky-core dataset: the router's most persistent expert.

For a given HB layer and action token the router picks 4 of 32 experts at every
control step.  Occupancy is the fraction of a trailing window in which the most
persistent expert stayed inside that Top-4, averaged over the 10 action tokens.

Every window sits inside the common cohort - the last control step at which all
32 rollouts of the initial state are still running - so a successful rollout
ending early can never thin the comparison group.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr


HUB = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB"
RUN = HUB / "cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
DEEP = (4, 5, 6, 7)
DENOISE = 9
TOKENS = slice(1, 11)
N_EXPERTS = 32
WINDOW = 10
GOAL_RADIUS_M = 0.05
PROGRESS_EPS_M = 0.01
POSE = np.r_[np.arange(10, 13), np.arange(17, 20)]


def masks(store, lo: int, hi: int) -> np.ndarray:
    ids = np.asarray(store["hb_expert_ids"][lo:hi, :, DENOISE, TOKENS, :], dtype=np.int64)
    m = np.zeros(ids.shape[:-1] + (N_EXPERTS,), dtype=bool)
    np.put_along_axis(m, ids, True, axis=-1)
    return m                                   # [T, layer, token, expert]


def occupancy(mask: np.ndarray, end: int, window: int) -> np.ndarray:
    """[layer] mean over tokens of the most persistent expert's occupancy."""
    piece = mask[end - window + 1: end + 1]
    return piece.mean(0).max(-1).mean(-1)


def rolling_occupancy(mask: np.ndarray, window: int) -> np.ndarray:
    """[T, layer] occupancy of a trailing window ending at each step."""
    out = np.full((len(mask), mask.shape[1]), np.nan)
    for t in range(window - 1, len(mask)):
        out[t] = occupancy(mask, t, window)
    return out


def trap_onset(distance: np.ndarray) -> int:
    best = np.minimum.accumulate(distance)
    future = np.minimum.accumulate(distance[::-1])[::-1]
    ok = np.flatnonzero((best - future <= PROGRESS_EPS_M) & (best > GOAL_RADIUS_M))
    return int(ok[0]) if len(ok) else -1


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    wins = float((pos[:, None] > neg[None, :]).sum())
    wins += 0.5 * float((pos[:, None] == neg[None, :]).sum())
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--layer", type=int, default=15, help="HB layer shown in the ribbon")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    lengths = np.asarray([r["inference_calls"] for r in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    scenes = np.asarray([int(r["init_state_id"]) for r in rows])
    seeds = np.asarray([int(r["flow_noise_seed"]) for r in rows])
    success = np.asarray([bool(r["success"]) for r in rows])
    index = np.flatnonzero(scenes == args.scene)
    common = int(lengths[index].min()) - 1
    axis = HB_LAYERS.index(args.layer)

    sims = []
    for e in range(len(rows)):
        with np.load(args.run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
            sims.append(np.asarray(p["sim_state"], dtype=np.float32))
    terminal = [(scenes[i], seeds[i], sims[i][-1, POSE]) for i in range(len(rows)) if success[i]]

    store = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    M, roll, final, dist = {}, {}, {}, {}
    for n, e in enumerate(index, 1):
        lo, hi = int(offsets[e]), int(offsets[e] + lengths[e])
        M[e] = masks(store, lo, hi)
        roll[e] = rolling_occupancy(M[e], WINDOW)
        final[e] = occupancy(M[e], common, WINDOW)
        reference = np.stack([p for s, d, p in terminal if s != args.scene and d != seeds[e]])
        from scipy.spatial.distance import cdist
        dist[e] = cdist(sims[e][:, POSE], reference).min(axis=1)
        print("  %02d/%d" % (n, len(index)), end="\r", flush=True)
    print()

    deep_final = {e: float(final[e][list(DEEP)].mean()) for e in index}
    S = np.array([deep_final[e] for e in index if success[e]])
    F = np.array([deep_final[e] for e in index if not success[e]])

    exemplar = {}
    for label, want in (("success", True), ("stasis", False)):
        group = [e for e in index if success[e] == want]
        med = float(np.median([deep_final[e] for e in group]))
        exemplar[label] = int(min(group, key=lambda e: abs(deep_final[e] - med)))

    # Per-token grids: at one layer and one action token exactly 4 experts are
    # selected each step, so a persistent expert reads as an unbroken run. The
    # token-aggregated view cannot show this - roughly 12 experts are on at any
    # step once the 10 tokens are pooled, and runs are long in both groups.
    ribbons = {}
    for label, e in exemplar.items():
        ribbons[label] = {
            "episode": int(e), "seed": int(seeds[e]), "length": int(lengths[e]),
            "onset": -1 if success[e] else trap_onset(dist[e]),
            "occupancy": deep_final[e],
            "tokens": [M[e][:, axis, tok].T.astype(np.uint8).tolist()   # [expert][t]
                       for tok in range(M[e].shape[2])],
            "rolling": np.round(roll[e][:, list(DEEP)].mean(1), 4).tolist(),
            "distance": np.round(dist[e], 4).tolist(),
        }

    # Longest unbroken Top-4 run per token, so the page can pick a token whose
    # contrast is typical rather than the most flattering one.
    def longest(grid, upto):
        best = 0
        for row in grid:
            n = 0
            for t in range(upto + 1):
                n = n + 1 if row[t] else 0
                best = max(best, n)
        return best
    token_runs = [
        {"token": tok,
         "success": longest(ribbons["success"]["tokens"][tok], common),
         "stasis": longest(ribbons["stasis"]["tokens"][tok], common)}
        for tok in range(len(ribbons["success"]["tokens"]))
    ]

    payload = {
        "task": "libero_long / KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "prompt": rows[int(index[0])].get("prompt", ""),
        "scene": int(args.scene), "layer": int(args.layer), "layers": list(HB_LAYERS),
        "n": len(index), "n_success": int(success[index].sum()),
        "common": common, "window": WINDOW, "max_len": int(lengths[index].max()),
        "goal_radius_m": GOAL_RADIUS_M,
        "headline": {
            "success_mean": float(S.mean()), "stasis_mean": float(F.mean()),
            "gap": float(F.mean() - S.mean()), "auc": auc(F, S),
            "success_range": [float(S.min()), float(S.max())],
            "stasis_range": [float(F.min()), float(F.max())],
        },
        "by_layer": [
            {"layer": L,
             "success": round(float(np.mean([final[e][a] for e in index if success[e]])), 4),
             "stasis":  round(float(np.mean([final[e][a] for e in index if not success[e]])), 4)}
            for a, L in enumerate(HB_LAYERS)
        ],
        "rollouts": [
            {"episode": int(e), "success": bool(success[e]), "occupancy": deep_final[e],
             "onset": -1 if success[e] else trap_onset(dist[e]),
             "rolling": np.round(roll[e][:, list(DEEP)].mean(1), 4).tolist()}
            for e in index
        ],
        "exemplars": ribbons,
        "token_runs": token_runs,
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote %s (%.0f KB) | occupancy %.3f -> %.3f  AUC %.3f | common t<=%d | exemplars %s"
          % (args.out, args.out.stat().st_size / 1024, S.mean(), F.mean(),
             payload["headline"]["auc"], common,
             {k: v["episode"] for k, v in ribbons.items()}))
    print("longest unbroken Top-4 run per action token (success vs stasis):")
    for r in token_runs:
        print("  token%d: %2d vs %2d" % (r["token"], r["success"], r["stasis"]))


if __name__ == "__main__":
    main()
