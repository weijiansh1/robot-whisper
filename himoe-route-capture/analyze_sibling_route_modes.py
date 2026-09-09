#!/usr/bin/env python3
"""Ask whether same-state siblings occupy discrete routing modes.

Every snapshot restores one simulator state and launches sixteen branches that
differ only in a recorded flow-noise stream, so at a fixed query offset the
task, the initial state and the phase are all held fixed by construction.  No
phase normalization is needed and none is used, which is what separates this
from the earlier relative-phase grammar.

For each snapshot and query the sixteen sibling routing vectors are centered on
their own mean, leaving only the noise-driven deviation.  Three things are then
measured, none of which uses an outcome label:

* discreteness -- how well a two-way split explains the sixteen deviations,
  scored against a within-snapshot permutation null;
* a flow-noise control -- the same pipeline applied to the noise vectors that
  actually drove the branches, since a routing split that merely mirrors a
  noise split carries nothing new;
* motor alignment -- whether the routing split separates the end-effector
  displacement that the branches go on to execute.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import analyze_rolling_star_experiment as rolling

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
DEEP_AXES = [4, 5, 6, 7]
FRONT_AXES = [0, 1, 2, 3]
BLOCK = 512
MAX_QUERY = 24
PERMUTATIONS = 400
SEED = 20260829


def two_way_silhouette(matrix: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """Silhouette of the best two-way split, with its labels."""
    if len(matrix) < 4:
        return float("nan"), np.zeros(len(matrix), dtype=int)
    model = KMeans(n_clusters=2, n_init=10, random_state=seed)
    labels = model.fit_predict(matrix)
    if len(np.unique(labels)) < 2:
        return float("nan"), labels
    return float(silhouette_score(matrix, labels)), labels


def discreteness(matrix: np.ndarray, rng: np.random.Generator, seed: int) -> dict:
    """Two-way silhouette against a null that destroys joint structure.

    The null shuffles every coordinate independently across branches.  That
    keeps each coordinate's marginal spread but removes any agreement between
    coordinates, so a surviving split reflects branches moving together rather
    than one loud dimension.
    """
    observed, labels = two_way_silhouette(matrix, seed)
    if not np.isfinite(observed):
        return {"silhouette": np.nan, "null_mean": np.nan, "p": np.nan}
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = matrix.copy()
        for column in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, column])
        value, _ = two_way_silhouette(shuffled, seed)
        if np.isfinite(value):
            null.append(value)
    null = np.asarray(null)
    return {
        "silhouette": observed,
        "null_mean": float(null.mean()) if len(null) else np.nan,
        "p": float((null >= observed).mean()) if len(null) else np.nan,
        "labels": labels,
    }


def motor_alignment(
    labels: np.ndarray, displacement: np.ndarray, rng: np.random.Generator
) -> dict:
    """Separation of executed end-effector displacement between the two groups."""
    left = displacement[labels == 0]
    right = displacement[labels == 1]
    if len(left) < 2 or len(right) < 2:
        return {"separation": np.nan, "null_mean": np.nan, "p": np.nan}
    observed = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    null = []
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(labels)
        a = displacement[permuted == 0]
        b = displacement[permuted == 1]
        if len(a) >= 2 and len(b) >= 2:
            null.append(float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))))
    null = np.asarray(null)
    return {
        "separation": observed,
        "null_mean": float(null.mean()) if len(null) else np.nan,
        "p": float((null >= observed).mean()) if len(null) else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_route_modes"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    trajectories = {
        c.episode_id: rolling.load_trajectory(c) for c in candidates
    }

    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)
    deep = np.zeros((n_row, len(DEEP_AXES) * 32), dtype=np.float32)
    front = np.zeros((n_row, len(FRONT_AXES) * 32), dtype=np.float32)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        block = store["hb_router_probs"][start:stop][
            :, :, DENOISE, ACTION_TOKENS, :
        ].astype(np.float32)
        block = block.mean(axis=2)
        block = block / np.maximum(block.sum(axis=-1, keepdims=True), 1e-12)
        deep[start:stop] = block[:, DEEP_AXES].reshape(stop - start, -1)
        front[start:stop] = block[:, FRONT_AXES].reshape(stop - start, -1)
        if start % (BLOCK * 8) == 0:
            print(f"  probs {stop}/{n_row}", flush=True)

    order = np.lexsort((step_of_row, episode_of_row))
    deep = deep[order]
    front = front[order]
    episode_sorted = episode_of_row[order]
    deep_by = {}
    front_by = {}
    for episode in np.unique(episode_sorted):
        mask = episode_sorted == episode
        deep_by[int(episode)] = deep[mask]
        front_by[int(episode)] = front[mask]

    by_snapshot = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)

    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in sorted(by_snapshot.items()):
        group = sorted(group, key=lambda c: c.candidate)
        for query in range(MAX_QUERY):
            present = [
                c for c in group if len(deep_by.get(c.episode_id, [])) > query + 1
            ]
            if len(present) < 12:
                break
            route = np.stack([deep_by[c.episode_id][query] for c in present])
            route = route - route.mean(axis=0, keepdims=True)
            noise = np.stack(
                [
                    np.asarray(
                        trajectories[c.episode_id]["flow_noise"][query],
                        dtype=np.float32,
                    ).reshape(-1)
                    for c in present
                ]
            )
            noise = noise - noise.mean(axis=0, keepdims=True)
            # displacement executed during this query, from the recorded control stream
            displacement = []
            for c in present:
                arrays = trajectories[c.episode_id]
                index = np.flatnonzero(arrays["control_query_index"] == query)
                eef = np.asarray(arrays["control_eef_position"], dtype=np.float32)
                displacement.append(eef[index[-1] + 1] - eef[index[0]])
            displacement = np.stack(displacement)

            route_result = discreteness(route, rng, SEED)
            noise_result = discreteness(noise, rng, SEED)
            motor = (
                motor_alignment(route_result["labels"], displacement, rng)
                if isinstance(route_result.get("labels"), np.ndarray)
                else {"separation": np.nan, "p": np.nan, "null_mean": np.nan}
            )
            rows.append(
                {
                    "snapshot_key": key,
                    "query": query,
                    "branches": len(present),
                    "route_silhouette": route_result["silhouette"],
                    "route_null": route_result["null_mean"],
                    "route_p": route_result["p"],
                    "noise_silhouette": noise_result["silhouette"],
                    "noise_null": noise_result["null_mean"],
                    "noise_p": noise_result["p"],
                    "motor_separation": motor["separation"],
                    "motor_null": motor["null_mean"],
                    "motor_p": motor["p"],
                }
            )
        print(f"  {key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "route_modes.csv", index=False)

    print("\n=== discreteness of sibling routing, by query ===")
    print(f"{'query':>6} {'snaps':>6} {'route_sil':>10} {'null':>7} {'p<.05':>7} "
          f"{'noise_sil':>10} {'p<.05':>7} {'motor_p<.05':>12}")
    summary = []
    for query, part in frame.groupby("query"):
        if len(part) < 6:
            continue
        record = {
            "query": int(query),
            "snapshots": int(len(part)),
            "route_silhouette": float(part.route_silhouette.mean()),
            "route_null": float(part.route_null.mean()),
            "route_significant": int((part.route_p < 0.05).sum()),
            "noise_silhouette": float(part.noise_silhouette.mean()),
            "noise_significant": int((part.noise_p < 0.05).sum()),
            "motor_significant": int((part.motor_p < 0.05).sum()),
        }
        summary.append(record)
        print(
            f"{query:>6} {len(part):>6} {record['route_silhouette']:>10.4f} "
            f"{record['route_null']:>7.4f} {record['route_significant']:>4}/{len(part):<2} "
            f"{record['noise_silhouette']:>10.4f} "
            f"{record['noise_significant']:>4}/{len(part):<2} "
            f"{record['motor_significant']:>9}/{len(part):<2}"
        )

    with open(out_dir / "route_modes_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
