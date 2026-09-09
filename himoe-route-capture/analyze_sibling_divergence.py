#!/usr/bin/env python3
"""Compare siblings that share a trunk state and differ only in flow noise.

Every snapshot restores one simulator state and launches sixteen branches whose
only difference is a recorded noise stream.  Mixed snapshots therefore give a
paired contrast in which initial geometry, task phase and prompt are held
exactly fixed, which is the one design in this cache that cannot be explained
by initial-state difficulty.

Two questions are asked per query index, always inside a snapshot:

* divergence -- how far each branch's Top-4 routing has moved away from its own
  siblings, measured as mean pairwise Jaccard distance;
* churn -- how much each branch's own routing moves between consecutive steps.

Both are compared between the branches that eventually succeed and those that
eventually fail within the same snapshot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import stats

import analyze_rolling_star_experiment as rolling

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
STATE_TOKEN = 0
FRONT_AXES = [0, 1, 2, 3]
DEEP_AXES = [4, 5, 6, 7]
BLOCK = 512
MAX_QUERY = 24


def jaccard_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Mean 1 - Jaccard of Top-4 sets over the token axis."""
    values = []
    for token in range(left.shape[0]):
        a = set(left[token].tolist())
        b = set(right[token].tolist())
        union = len(a | b)
        values.append(1.0 - (len(a & b) / union if union else 1.0))
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_sibling"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)

    action_ids = np.zeros((n_row, 8, 10, 4), dtype=np.uint8)
    state_ids = np.zeros((n_row, 8, 4), dtype=np.uint8)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        block = store["hb_expert_ids"][start:stop][:, :, DENOISE, :, :]
        action_ids[start:stop] = block[:, :, ACTION_TOKENS, :]
        state_ids[start:stop] = block[:, :, STATE_TOKEN, :]
        if start % (BLOCK * 8) == 0:
            print(f"  ids {stop}/{n_row}", flush=True)

    order = np.lexsort((step_of_row, episode_of_row))
    action_ids = action_ids[order]
    state_ids = state_ids[order]
    episode_sorted = episode_of_row[order]
    action_by_episode = {}
    state_by_episode = {}
    for episode in np.unique(episode_sorted):
        mask = episode_sorted == episode
        action_by_episode[int(episode)] = action_ids[mask]
        state_by_episode[int(episode)] = state_ids[mask]

    by_snapshot: dict[str, list] = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)

    rows = []
    for snapshot_key, siblings in sorted(by_snapshot.items()):
        outcomes = np.asarray([c.success for c in siblings])
        if outcomes.all() or not outcomes.any():
            continue  # only mixed snapshots carry the paired contrast
        for query in range(MAX_QUERY):
            present = [
                c
                for c in siblings
                if len(action_by_episode.get(c.episode_id, [])) > query
            ]
            if len(present) < 8:
                break
            action_stack = np.stack(
                [action_by_episode[c.episode_id][query] for c in present]
            )
            for axes, name in ((DEEP_AXES, "deep"), (FRONT_AXES, "front")):
                for index, candidate in enumerate(present):
                    others = [j for j in range(len(present)) if j != index]
                    divergence = float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        jaccard_distance(
                                            action_stack[index, layer],
                                            action_stack[j, layer],
                                        )
                                        for layer in axes
                                    ]
                                )
                                for j in others
                            ]
                        )
                    )
                    churn = np.nan
                    if query > 0:
                        previous = action_by_episode[candidate.episode_id][query - 1]
                        current = action_by_episode[candidate.episode_id][query]
                        churn = float(
                            np.mean(
                                [
                                    jaccard_distance(previous[layer], current[layer])
                                    for layer in axes
                                ]
                            )
                        )
                    rows.append(
                        {
                            "snapshot_key": snapshot_key,
                            "worker": candidate.worker,
                            "candidate": candidate.candidate,
                            "episode_id": candidate.episode_id,
                            "success": candidate.success,
                            "query": query,
                            "layers": name,
                            "sibling_divergence": divergence,
                            "self_churn": churn,
                        }
                    )
        print(f"  {snapshot_key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "sibling_divergence.csv", index=False)

    print("\n=== paired within-snapshot contrast, deep layers ===")
    print(f"{'query':>6} {'snaps':>6} {'succ_div':>9} {'fail_div':>9} {'diff':>8} "
          f"{'snaps+':>7} {'p_sign':>8}")
    summary = []
    deep = frame[frame.layers == "deep"]
    for query in sorted(deep["query"].unique()):
        part = deep[deep["query"] == query]
        paired = []
        for snapshot_key, group in part.groupby("snapshot_key"):
            s = group[group.success].sibling_divergence
            f = group[~group.success].sibling_divergence
            if len(s) and len(f):
                paired.append(float(s.mean() - f.mean()))
        if len(paired) < 6:
            continue
        paired = np.asarray(paired)
        positive = int((paired > 0).sum())
        p_value = float(
            stats.binomtest(positive, len(paired), 0.5).pvalue
        )
        succ = float(part[part.success].sibling_divergence.mean())
        fail = float(part[~part.success].sibling_divergence.mean())
        summary.append(
            {
                "query": int(query),
                "snapshots": len(paired),
                "success_divergence": succ,
                "failure_divergence": fail,
                "paired_difference": float(paired.mean()),
                "snapshots_positive": positive,
                "sign_test_p": p_value,
            }
        )
        print(f"{query:>6} {len(paired):>6} {succ:>9.4f} {fail:>9.4f} "
              f"{paired.mean():>+8.4f} {positive:>4}/{len(paired):<2} {p_value:>8.4f}")

    with open(out_dir / "sibling_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
