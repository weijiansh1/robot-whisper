#!/usr/bin/env python3
"""Ask whether routing encodes which object a branch will commit to.

Engaging moka_pot_1 first succeeds 93% of the time and engaging moka_pot_2
first succeeds 2%, so the outcome of a branch is largely decided by a discrete
ordering choice rather than by control quality.  Snapshots that contain both
choices hold initial geometry exactly fixed, so they isolate that choice.

Two clocks are compared:

* physical commitment -- the first query at which the two objects' cumulative
  motion separates by more than ``COMMIT_MARGIN_M``;
* routing separation -- whether, at query k, sibling pairs that will choose
  different objects already route more differently than pairs that will choose
  the same object.

Routing that separates before the physical clock would be actionable, because
the noise sample could be redrawn.  Routing that separates only afterwards is
a readout of a commitment already made.
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
from scipy import stats

import analyze_rolling_star_experiment as rolling

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
FRONT_AXES = [0, 1, 2, 3]
DEEP_AXES = [4, 5, 6, 7]
COMMIT_MARGIN_M = 0.02
BLOCK = 512
MAX_QUERY = 20


def jaccard_distance(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for token in range(left.shape[0]):
        a = set(left[token].tolist())
        b = set(right[token].tolist())
        union = len(a | b)
        values.append(1.0 - (len(a & b) / union if union else 1.0))
    return float(np.mean(values))


def per_query_object_paths(
    arrays: dict[str, np.ndarray], targets: list[dict]
) -> np.ndarray:
    """Cumulative path of every target object, sampled at query boundaries."""
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    paths = []
    for target in targets:
        lo = int(target["state_lo"])
        position = sim[:, lo : lo + 3]
        step = np.linalg.norm(np.diff(position, axis=0), axis=1)
        paths.append(np.r_[0.0, np.cumsum(step)])
    stacked = np.stack(paths, axis=1)
    starts = np.r_[
        0, np.flatnonzero(np.diff(arrays["control_query_index"]) != 0) + 1
    ]
    return stacked[starts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_mode_commitment"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    targets, _ = rolling.load_layout(run_root)
    trajectories = {
        c.episode_id: rolling.load_trajectory(c) for c in candidates
    }

    mode = {}
    commit = {}
    for candidate in candidates:
        paths = per_query_object_paths(trajectories[candidate.episode_id], targets)
        final = paths[-1]
        mode[candidate.episode_id] = "pot1" if final[0] > final[1] else "pot2"
        margin = paths[:, 0] - paths[:, 1]
        fired = np.flatnonzero(np.abs(margin) >= COMMIT_MARGIN_M)
        commit[candidate.episode_id] = int(fired[0]) if len(fired) else -1

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
    order = np.lexsort((step_of_row, episode_of_row))
    ids = ids[order]
    episode_sorted = episode_of_row[order]
    by_episode = {
        int(e): ids[episode_sorted == e] for e in np.unique(episode_sorted)
    }

    by_snapshot = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)
    mixed = {
        key: group
        for key, group in by_snapshot.items()
        if len({mode[c.episode_id] for c in group}) > 1
    }
    print(f"mixed-mode snapshots: {len(mixed)}/{len(by_snapshot)}")

    commit_values = [
        commit[c.episode_id] for c in candidates if commit[c.episode_id] >= 0
    ]
    print(
        f"physical commitment query: median {np.median(commit_values):.0f}, "
        f"p10 {np.percentile(commit_values, 10):.0f}, "
        f"p90 {np.percentile(commit_values, 90):.0f}"
    )

    rows = []
    for key, group in sorted(mixed.items()):
        for query in range(MAX_QUERY):
            present = [
                c for c in group if len(by_episode.get(c.episode_id, [])) > query
            ]
            if len(present) < 8:
                break
            for left, right in combinations(present, 2):
                same = mode[left.episode_id] == mode[right.episode_id]
                for axes, name in ((DEEP_AXES, "deep"), (FRONT_AXES, "front")):
                    value = float(
                        np.mean(
                            [
                                jaccard_distance(
                                    by_episode[left.episode_id][query][layer],
                                    by_episode[right.episode_id][query][layer],
                                )
                                for layer in axes
                            ]
                        )
                    )
                    rows.append(
                        {
                            "snapshot_key": key,
                            "query": query,
                            "layers": name,
                            "same_mode": same,
                            "divergence": value,
                        }
                    )
        print(f"  {key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "mode_pair_divergence.csv", index=False)

    print("\n=== different-mode minus same-mode pair divergence, deep layers ===")
    print(f"{'query':>6} {'snaps':>6} {'same':>8} {'diff':>8} {'delta':>8} "
          f"{'snaps+':>8} {'p_sign':>8}")
    summary = []
    deep = frame[frame.layers == "deep"]
    for query in sorted(deep["query"].unique()):
        part = deep[deep["query"] == query]
        paired = []
        for key, group in part.groupby("snapshot_key"):
            same = group[group.same_mode].divergence
            diff = group[~group.same_mode].divergence
            if len(same) >= 3 and len(diff) >= 3:
                paired.append(float(diff.mean() - same.mean()))
        if len(paired) < 6:
            continue
        paired = np.asarray(paired)
        positive = int((paired > 0).sum())
        p_value = float(stats.binomtest(positive, len(paired), 0.5).pvalue)
        summary.append(
            {
                "query": int(query),
                "snapshots": len(paired),
                "same_mode_divergence": float(part[part.same_mode].divergence.mean()),
                "different_mode_divergence": float(
                    part[~part.same_mode].divergence.mean()
                ),
                "paired_delta": float(paired.mean()),
                "snapshots_positive": positive,
                "sign_test_p": p_value,
            }
        )
        print(
            f"{query:>6} {len(paired):>6} "
            f"{part[part.same_mode].divergence.mean():>8.4f} "
            f"{part[~part.same_mode].divergence.mean():>8.4f} "
            f"{paired.mean():>+8.4f} {positive:>5}/{len(paired):<2} {p_value:>8.4f}"
        )

    with open(out_dir / "mode_commitment_summary.json", "w") as handle:
        json.dump(
            {
                "physical_commitment_median": float(np.median(commit_values)),
                "mixed_mode_snapshots": len(mixed),
                "commit_margin_m": COMMIT_MARGIN_M,
                "per_query": summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
