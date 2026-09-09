#!/usr/bin/env python3
"""Put the two axes of routing variation on one scale.

Horizontal variation is the distance between two different branches at the
same control step.  Vertical variation is the distance between one branch and
its own past.  Because every sibling starts from a bit-identical simulator
state, the two are directly comparable, and their ratio says what the router is
tracking: if a branch resembles its siblings more than its own recent past,
routing is following task phase rather than branch identity.

The reported crossing lag is the number of steps back a branch must look before
its own past is as far away as a sibling is right now.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

import analyze_rolling_star_experiment as rolling

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
DEEP_AXES = [4, 5, 6, 7]
FRONT_AXES = [0, 1, 2, 3]
BLOCK = 512
MAX_QUERY = 24
MAX_LAG = 8


def to_masks(expert_ids: np.ndarray) -> np.ndarray:
    """Pack Top-4 expert sets into one 32-bit mask per (row, layer, token)."""
    masks = np.zeros(expert_ids.shape[:-1], dtype=np.uint32)
    for slot in range(expert_ids.shape[-1]):
        masks |= (np.uint32(1) << expert_ids[..., slot].astype(np.uint32))
    return masks


def jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """1 - Jaccard over packed masks, averaged over the trailing token axis."""
    intersection = np.bitwise_count(left & right).astype(np.float64)
    union = np.bitwise_count(left | right).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = 1.0 - np.where(union > 0, intersection / union, 1.0)
    return value.mean(axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_route_axes"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
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

    by_snapshot = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)

    rows = []
    for key, group in sorted(by_snapshot.items()):
        for query in range(MAX_QUERY):
            present = [
                c for c in group if len(by_episode.get(c.episode_id, [])) > query
            ]
            if len(present) < 8:
                break
            for axes, name in ((DEEP_AXES, "deep"), (FRONT_AXES, "front")):
                horizontal = [
                    float(
                        jaccard(
                            by_episode[a.episode_id][query][axes],
                            by_episode[b.episode_id][query][axes],
                        ).mean()
                    )
                    for a, b in combinations(present, 2)
                ]
                record = {
                    "snapshot_key": key,
                    "query": query,
                    "layers": name,
                    "branches": len(present),
                    "horizontal": float(np.mean(horizontal)),
                }
                for lag in range(1, MAX_LAG + 1):
                    values = [
                        float(
                            jaccard(
                                by_episode[c.episode_id][query][axes],
                                by_episode[c.episode_id][query - lag][axes],
                            ).mean()
                        )
                        for c in present
                        if query - lag >= 0
                    ]
                    record[f"vertical_lag{lag}"] = (
                        float(np.mean(values)) if values else np.nan
                    )
                rows.append(record)
        print(f"  {key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "route_axes.csv", index=False)

    deep = frame[frame.layers == "deep"]
    print("\n=== horizontal versus vertical, deep layers ===")
    print(f"{'query':>6} {'horizontal':>11} {'lag1':>8} {'lag2':>8} {'lag4':>8} "
          f"{'lag8':>8} {'crossing lag':>13}")
    summary = []
    for query, part in deep.groupby("query"):
        horizontal = float(part.horizontal.mean())
        verticals = {
            lag: float(part[f"vertical_lag{lag}"].mean())
            for lag in range(1, MAX_LAG + 1)
            if part[f"vertical_lag{lag}"].notna().any()
        }
        crossing = next(
            (lag for lag in sorted(verticals) if verticals[lag] >= horizontal), None
        )
        summary.append(
            {
                "query": int(query),
                "horizontal": horizontal,
                "vertical": {str(k): v for k, v in verticals.items()},
                "crossing_lag": crossing,
            }
        )
        print(
            f"{query:>6} {horizontal:>11.4f} "
            f"{verticals.get(1, float('nan')):>8.4f} "
            f"{verticals.get(2, float('nan')):>8.4f} "
            f"{verticals.get(4, float('nan')):>8.4f} "
            f"{verticals.get(8, float('nan')):>8.4f} "
            f"{str(crossing) if crossing else '>8':>13}"
        )

    with open(out_dir / "route_axes_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
