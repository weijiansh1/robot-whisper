#!/usr/bin/env python3
"""Repeat the horizontal/vertical routing comparison across hub tasks.

The rolling-star run showed that a branch differs from its own previous
control step more than it differs from a sibling at the same step, which means
routing variation is spent mostly on tracking task phase rather than on branch
identity.  That was one task.  The 16x32 hub corpora hold five tasks, each with
sixteen initial states and thirty-two flow-noise draws per state, so the same
two axes can be measured wherever siblings share an initial state.

Distances are 1 - Jaccard over Top-4 expert sets at the final denoise step,
averaged over action tokens, reported against the 0.925 expected distance
between two independent random Top-4 sets.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
DEEP_AXES = [4, 5, 6, 7]
FRONT_AXES = [0, 1, 2, 3]
MAX_LAG = 8
MAX_QUERY = 20
RANDOM_BASELINE = 0.9250
BLOCK = 512

TASKS = {
    "spatial/stove": "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "spatial/ramekin": "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "goal/top_drawer": "libero_goal/open_the_top_drawer_and_put_the_bowl_inside",
    "goal/middle_drawer": "libero_goal/open_the_middle_drawer_of_the_cabinet",
    "long/SCENE8": "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
}


def to_masks(expert_ids: np.ndarray) -> np.ndarray:
    masks = np.zeros(expert_ids.shape[:-1], dtype=np.uint32)
    for slot in range(expert_ids.shape[-1]):
        masks |= np.uint32(1) << expert_ids[..., slot].astype(np.uint32)
    return masks


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.bitwise_count(left & right).astype(np.float64)
    union = np.bitwise_count(left | right).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = 1.0 - np.where(union > 0, intersection / union, 1.0)
    return float(value.mean())


def analyse(root: Path, axes: list[int]) -> pd.DataFrame:
    store = zarr.open(str(root / "server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)
    ids = np.zeros((n_row, 8, 10, 4), dtype=np.uint8)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        ids[start:stop] = store["hb_expert_ids"][start:stop][
            :, :, DENOISE, ACTION_TOKENS, :
        ]
    masks = to_masks(ids)
    order = np.lexsort((step_of_row, episode_of_row))
    masks = masks[order]
    episode_sorted = episode_of_row[order]
    by_episode = {
        int(e): masks[episode_sorted == e] for e in np.unique(episode_sorted)
    }

    summaries = pd.DataFrame(json.load(open(root / "client/summaries.json")))
    groups = summaries.groupby("init_state_id").episode_index.apply(list)

    rows = []
    for init_state, episodes in groups.items():
        for query in range(MAX_QUERY):
            present = [e for e in episodes if len(by_episode.get(e, [])) > query]
            if len(present) < 8:
                break
            sample = present[:16]  # cap pair count; siblings are exchangeable
            horizontal = float(
                np.mean(
                    [
                        jaccard(
                            by_episode[a][query][axes], by_episode[b][query][axes]
                        )
                        for a, b in combinations(sample, 2)
                    ]
                )
            )
            record = {
                "init_state_id": int(init_state),
                "query": query,
                "branches": len(present),
                "horizontal": horizontal,
            }
            for lag in range(1, MAX_LAG + 1):
                if query - lag < 0:
                    continue
                record[f"vertical_lag{lag}"] = float(
                    np.mean(
                        [
                            jaccard(
                                by_episode[e][query][axes],
                                by_episode[e][query - lag][axes],
                            )
                            for e in present
                        ]
                    )
                )
            rows.append(record)
    return pd.DataFrame(rows)


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
        frame = analyse(root, DEEP_AXES)
        frame.to_csv(args.out / f"axes_{name.replace('/', '_')}.csv", index=False)
        print(f"\n===== {name} =====")
        print(f"{'query':>6} {'inits':>6} {'horizontal':>11} {'lag1':>8} {'lag2':>8} "
              f"{'h<lag1':>8}")
        entries = []
        for query, part in frame.groupby("query"):
            horizontal = float(part.horizontal.mean())
            lag1 = (
                float(part["vertical_lag1"].mean())
                if "vertical_lag1" in part and part["vertical_lag1"].notna().any()
                else float("nan")
            )
            lag2 = (
                float(part["vertical_lag2"].mean())
                if "vertical_lag2" in part and part["vertical_lag2"].notna().any()
                else float("nan")
            )
            below = (
                int((part.horizontal < part["vertical_lag1"]).sum())
                if "vertical_lag1" in part
                else 0
            )
            entries.append(
                {
                    "query": int(query),
                    "horizontal": horizontal,
                    "vertical_lag1": lag1,
                    "vertical_lag2": lag2,
                    "inits_horizontal_below_lag1": below,
                    "inits": int(len(part)),
                }
            )
            print(
                f"{query:>6} {len(part):>6} {horizontal:>11.4f} {lag1:>8.4f} "
                f"{lag2:>8.4f} {below:>5}/{len(part):<2}"
            )
        report[name] = entries

    with open(args.out / "axes_hub_summary.json", "w") as handle:
        json.dump(
            {"random_baseline": RANDOM_BASELINE, "tasks": report},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
