#!/usr/bin/env python3
"""Extract one initial state's 32 rollouts as an animatable trap sequence.

Everything exported here is recorded data, not a schematic.

* The piano roll is the authoritative Top-4 expert selection at the final
  denoise step: for each HB layer and expert, how many of the 10 action tokens
  selected that expert at that control step (0..10).
* Route churn is 1 - Jaccard of consecutive control steps' Top-4 sets,
  averaged over the 10 action tokens, kept separately per HB layer.
* The goal distance reuses `analyze_early_structure.build_targets`: held-out
  nearest-neighbour distance to the successful-terminal pose set, with
  same-state and same-seed successes removed from the reference so no rollout
  ever scores against itself.

The two exemplar rollouts are the ones whose final-10-step deep-layer churn is
closest to their own group's median, so neither is a hand-picked extreme.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr
from scipy.spatial.distance import cdist


HUB = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB"
RUN = HUB / "cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
DEEP = (4, 5, 6, 7)          # axes of layers 12/13/14/15
DENOISE = 9
TOKENS = slice(1, 11)
N_EXPERTS = 32
POSE = np.r_[np.arange(10, 13), np.arange(17, 20)]
GOAL_RADIUS_M = 0.05
PROGRESS_EPS_M = 0.01
TAIL = 10


def selection_mask(store, lo: int, hi: int) -> np.ndarray:
    """[T, layer, token, expert] boolean Top-4 selection."""
    ids = np.asarray(store["hb_expert_ids"][lo:hi, :, DENOISE, TOKENS, :], dtype=np.int64)
    mask = np.zeros(ids.shape[:-1] + (N_EXPERTS,), dtype=bool)
    np.put_along_axis(mask, ids, True, axis=-1)
    return mask


def churn(mask: np.ndarray) -> np.ndarray:
    """[T-1, layer] fraction of the Top-4 set replaced since the previous step."""
    intersection = np.logical_and(mask[1:], mask[:-1]).sum(-1)
    union = np.logical_or(mask[1:], mask[:-1]).sum(-1)
    return (1.0 - intersection / union).mean(axis=2)


def trap_onset(distance: np.ndarray) -> int:
    best = np.minimum.accumulate(distance)
    future = np.minimum.accumulate(distance[::-1])[::-1]
    eligible = np.flatnonzero((best - future <= PROGRESS_EPS_M) & (best > GOAL_RADIUS_M))
    return int(eligible[0]) if len(eligible) else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, default=RUN)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    lengths = np.asarray([r["inference_calls"] for r in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    scenes = np.asarray([int(r["init_state_id"]) for r in rows])
    seeds = np.asarray([int(r["flow_noise_seed"]) for r in rows])
    success = np.asarray([bool(r["success"]) for r in rows])
    index = np.flatnonzero(scenes == args.scene)

    sims = []
    for episode in range(len(rows)):
        with np.load(args.run / ("client/episode_%02d.npz" % episode), allow_pickle=True) as p:
            sims.append(np.asarray(p["sim_state"], dtype=np.float32))
    terminal = [(scenes[i], seeds[i], sims[i][-1, POSE])
                for i in range(len(rows)) if success[i]]

    store = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    masks, distances = {}, {}
    for n, episode in enumerate(index, 1):
        lo, hi = int(offsets[episode]), int(offsets[episode] + lengths[episode])
        masks[episode] = selection_mask(store, lo, hi)
        reference = np.stack([p for s, d, p in terminal
                              if s != args.scene and d != seeds[episode]])
        distances[episode] = cdist(sims[episode][:, POSE], reference).min(axis=1)
        print("  extracted %02d/%d" % (n, len(index)), end="\r", flush=True)
    print()

    deep_tail = {e: float(churn(masks[e])[-TAIL:, DEEP].mean()) for e in index}
    exemplar = {}
    for label, want in (("success", True), ("stasis", False)):
        group = [e for e in index if success[e] == want]
        median = float(np.median([deep_tail[e] for e in group]))
        exemplar[label] = int(min(group, key=lambda e: abs(deep_tail[e] - median)))

    episodes = []
    for episode in index:
        mask = masks[episode]
        episodes.append({
            "episode": int(episode),
            "seed": int(seeds[episode]),
            "success": bool(success[episode]),
            "onset": -1 if success[episode] else trap_onset(distances[episode]),
            "length": int(lengths[episode]),
            "distance": np.round(distances[episode], 4).tolist(),
            "churn": np.round(churn(mask), 4).tolist(),
        })

    rolls = {}
    for label, episode in exemplar.items():
        counts = masks[episode].sum(axis=2).astype(np.uint8)   # [T, layer, expert] 0..10
        rolls[label] = {
            "episode": int(episode),
            "seed": int(seeds[episode]),
            "length": int(lengths[episode]),
            "onset": next(e["onset"] for e in episodes if e["episode"] == episode),
            "grid": counts.transpose(1, 2, 0).tolist(),        # [layer][expert][t]
        }

    layer_gap = []
    for axis, layer in enumerate(HB_LAYERS):
        s = float(np.mean([churn(masks[e])[-TAIL:, axis].mean() for e in index if success[e]]))
        f = float(np.mean([churn(masks[e])[-TAIL:, axis].mean() for e in index if not success[e]]))
        layer_gap.append({"layer": layer, "success": round(s, 4),
                          "stasis": round(f, 4), "gap": round(f - s, 4)})

    payload = {
        "task": "libero_long / KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "prompt": rows[int(index[0])].get("prompt", ""),
        "scene": int(args.scene),
        "layers": list(HB_LAYERS),
        "deep_axes": list(DEEP),
        "denoise_step": DENOISE,
        "n": len(index),
        "n_success": int(success[index].sum()),
        "max_len": int(lengths[index].max()),
        "goal_radius_m": GOAL_RADIUS_M,
        "progress_eps_m": PROGRESS_EPS_M,
        "tail_steps": TAIL,
        "layer_gap": layer_gap,
        "exemplars": rolls,
        "episodes": episodes,
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    onsets = [e["onset"] for e in episodes if e["onset"] >= 0]
    print("wrote %s (%.0f KB) | success %d stasis %d | onset median t%d | exemplars %s"
          % (args.out, args.out.stat().st_size / 1024,
             payload["n_success"], len(onsets), int(np.median(onsets)),
             {k: v["episode"] for k, v in rolls.items()}))


if __name__ == "__main__":
    main()
