#!/usr/bin/env python3
"""Check the within-chunk denoise profile across every hub task.

On the rolling-star run the executed mixture moves more between the last two
denoise steps than anywhere in the middle of the schedule, which is the reverse
of what a settling flow-matching decoder would do.  That measurement is label
free and needs no outcome, so it can be repeated on all five 16x32 corpora to
see whether it is a property of the model or of one task.

Reported per task: the mean Hellinger step between consecutive denoise steps of
the executed mixture, split by layer group, plus the same profile computed on
the state token so a token-specific artefact would show up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

DEEP_AXES = [4, 5, 6, 7]
FRONT_AXES = [0, 1, 2, 3]
ACTION_TOKENS = slice(1, 11)
STATE_TOKEN = slice(0, 1)
BLOCK = 128

TASKS = {
    "spatial/stove": "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "spatial/ramekin": "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "goal/top_drawer": "libero_goal/open_the_top_drawer_and_put_the_bowl_inside",
    "goal/middle_drawer": "libero_goal/open_the_middle_drawer_of_the_cabinet",
    "long/SCENE8": "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
}


def executed_mixture(weight: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    mixture = np.zeros(weight.shape[:-1] + (32,), dtype=np.float32)
    weight = weight / np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    np.put_along_axis(mixture, chosen, weight, axis=-1)
    return mixture


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))
    return np.sqrt(np.clip(0.5 * (root ** 2).sum(axis=-1), 0.0, 1.0))


def profile(root: Path, axes: list[int], tokens: slice) -> np.ndarray:
    store = zarr.open(str(root / "server/routes.zarr"), mode="r")
    n_row = store["hb_expert_ids"].shape[0]
    total = None
    count = 0
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        weight = store["hb_selected_prob"][start:stop][:, axes, :, tokens, :].astype(
            np.float32
        )
        chosen = store["hb_expert_ids"][start:stop][:, axes, :, tokens, :].astype(
            np.int64
        )
        mixture = executed_mixture(weight, chosen)
        step = hellinger(mixture[:, :, :-1], mixture[:, :, 1:])   # (b, layer, d-1, token)
        block = step.mean(axis=(0, 1, 3))
        total = block * (stop - start) if total is None else total + block * (stop - start)
        count += stop - start
    return total / count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {}
    for name, relative in TASKS.items():
        root = args.hub / relative / "right-16x32"
        if not (root / "server/routes.zarr").exists():
            print(f"  skip {name}")
            continue
        entry = {
            "action_deep": profile(root, DEEP_AXES, ACTION_TOKENS).tolist(),
            "action_front": profile(root, FRONT_AXES, ACTION_TOKENS).tolist(),
            "state_deep": profile(root, DEEP_AXES, STATE_TOKEN).tolist(),
        }
        report[name] = entry
        deep = np.asarray(entry["action_deep"])
        print(f"\n=== {name} ===")
        for label, values in entry.items():
            values = np.asarray(values)
            print(f"  {label:>13}: " + " ".join(f"{v:.3f}" for v in values))
        print(
            f"  last step / middle mean = "
            f"{deep[-1] / deep[2:5].mean():.2f}"
        )

    with open(args.out / "denoise_profile.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
