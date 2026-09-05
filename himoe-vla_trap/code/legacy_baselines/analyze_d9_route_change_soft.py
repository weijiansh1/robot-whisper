#!/usr/bin/env python3
"""Re-measure routing change on the soft distribution instead of Top-4 ids.

Every churn and divergence number produced so far used Top-4 set overlap.  In
this cache 28.6% of action-token selections sit on a p4=p5 boundary, so a set
can turn over while the underlying distribution barely moves, and set-based
change is partly a readout of quantization noise.

Hellinger and total-variation distance on the full 32-expert distribution are
invariant to which of two tied experts is called fourth, so they measure change
in the routing decision itself.  The same two axes are reported as before --
one branch against its own past, and two branches against each other at the
same step -- plus an unrelated-pair baseline drawn from different branches at
different steps, which fixes the top of the scale empirically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-route-capture"))

import analyze_rolling_star_experiment as rolling  # noqa: E402

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
DEEP_AXES = [4, 5, 6, 7]
MAX_QUERY = 24
MAX_LAG = 8
BLOCK = 512
SEED = 20260829
ABSORBING = {"w01_s002", "w01_s003", "w01_s004"}


def hellinger(left: np.ndarray, right: np.ndarray) -> float:
    """Hellinger distance, averaged over the leading layer and token axes."""
    root = np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))
    return float(np.sqrt(np.clip(0.5 * (root ** 2).sum(axis=-1), 0.0, 1.0)).mean())


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float((0.5 * np.abs(left - right).sum(axis=-1)).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root / "analysis_route_change_soft").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)
    probs = np.zeros((n_row, len(DEEP_AXES), 10, 32), dtype=np.float32)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        block = store["hb_router_probs"][start:stop][
            :, DEEP_AXES, DENOISE, ACTION_TOKENS, :
        ].astype(np.float32)
        probs[start:stop] = block / np.maximum(
            block.sum(axis=-1, keepdims=True), 1e-12
        )
    order = np.lexsort((step_of_row, episode_of_row))
    probs = probs[order]
    episode_sorted = episode_of_row[order]
    by_episode = {
        int(e): probs[episode_sorted == e] for e in np.unique(episode_sorted)
    }

    by_snapshot = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)

    rng = np.random.default_rng(SEED)
    episodes = list(by_episode)
    unrelated = []
    for _ in range(3000):
        a, b = rng.choice(episodes, 2, replace=False)
        left, right = by_episode[int(a)], by_episode[int(b)]
        unrelated.append(
            hellinger(
                left[rng.integers(len(left))], right[rng.integers(len(right))]
            )
        )
    ceiling = float(np.mean(unrelated))
    print(f"unrelated-pair Hellinger ceiling: {ceiling:.4f}")

    rows = []
    for key, group in sorted(by_snapshot.items()):
        for query in range(MAX_QUERY):
            present = [
                c for c in group if len(by_episode.get(c.episode_id, [])) > query
            ]
            if len(present) < 8:
                break
            horizontal = float(
                np.mean(
                    [
                        hellinger(
                            by_episode[a.episode_id][query],
                            by_episode[b.episode_id][query],
                        )
                        for a, b in combinations(present, 2)
                    ]
                )
            )
            record = {
                "snapshot_key": key,
                "query": query,
                "regime": "absorbing" if key in ABSORBING else "other",
                "branches": len(present),
                "horizontal": horizontal,
            }
            for lag in range(1, MAX_LAG + 1):
                if query - lag < 0:
                    continue
                record[f"vertical_lag{lag}"] = float(
                    np.mean(
                        [
                            hellinger(
                                by_episode[c.episode_id][query],
                                by_episode[c.episode_id][query - lag],
                            )
                            for c in present
                        ]
                    )
                )
            rows.append(record)
        print(f"  {key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "route_change_soft.csv", index=False)

    print("\n=== soft-distribution change, all snapshots ===")
    print(f"{'query':>6} {'horizontal':>11} {'lag1':>8} {'lag2':>8} {'lag4':>8} "
          f"{'h<lag1':>8}")
    summary = []
    for query, part in frame.groupby("query"):
        horizontal = float(part.horizontal.mean())
        lags = {
            lag: float(part[f"vertical_lag{lag}"].mean())
            for lag in (1, 2, 4)
            if f"vertical_lag{lag}" in part and part[f"vertical_lag{lag}"].notna().any()
        }
        below = (
            int((part.horizontal < part["vertical_lag1"]).sum())
            if "vertical_lag1" in part
            else 0
        )
        summary.append(
            {
                "query": int(query),
                "horizontal": horizontal,
                "vertical": lags,
                "snapshots_horizontal_below_lag1": below,
                "snapshots": int(len(part)),
            }
        )
        print(
            f"{query:>6} {horizontal:>11.4f} {lags.get(1, float('nan')):>8.4f} "
            f"{lags.get(2, float('nan')):>8.4f} {lags.get(4, float('nan')):>8.4f} "
            f"{below:>5}/{len(part):<2}"
        )

    print("\n=== absorbing versus the rest ===")
    for regime, part in frame.groupby("regime"):
        usable = part[part["query"] >= 4]
        print(
            f"  {regime:>10}: snapshots={part.snapshot_key.nunique():>2} "
            f"horizontal={usable.horizontal.mean():.4f} "
            f"lag1={usable['vertical_lag1'].mean():.4f} "
            f"lag4={usable['vertical_lag4'].mean():.4f}"
        )

    with open(out_dir / "route_change_soft_summary.json", "w") as handle:
        json.dump(
            {"unrelated_ceiling": ceiling, "per_query": summary},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
