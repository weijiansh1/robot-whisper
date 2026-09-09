#!/usr/bin/env python3
"""Select how many routing modes the sixteen siblings actually occupy.

The previous pass imposed a two-way split.  This one lets the data choose,
including the possibility of no structure at all, using Tibshirani's gap
statistic with a principal-axis uniform reference.  Because k=1 is in the
candidate set, a snapshot whose siblings are simply spread out will be scored
as unstructured rather than forced into two groups.

The same selection is run on the flow-noise vectors that drove the branches.
Routing that is genuinely organized should choose k>1 where noise chooses k=1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

import analyze_rolling_star_experiment as rolling

DENOISE = 9
ACTION_TOKENS = slice(1, 11)
DEEP_AXES = [4, 5, 6, 7]
BLOCK = 512
MAX_QUERY = 24
MAX_K = 4
REFERENCES = 100
BOOTSTRAPS = 40
COMPONENTS = 8
SEED = 20260829


def dispersion(matrix: np.ndarray, k: int, seed: int) -> float:
    """Log pooled within-cluster sum of squares."""
    if k == 1:
        centre = matrix.mean(axis=0, keepdims=True)
        return float(np.log(max(((matrix - centre) ** 2).sum(), 1e-12)))
    model = KMeans(n_clusters=k, n_init=3, random_state=seed)
    labels = model.fit_predict(matrix)
    total = 0.0
    for label in np.unique(labels):
        part = matrix[labels == label]
        total += ((part - part.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(np.log(max(total, 1e-12)))


def gap_select(matrix: np.ndarray, rng: np.random.Generator, seed: int) -> dict:
    """Tibshirani gap statistic over a principal-axis uniform reference."""
    n_component = min(COMPONENTS, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=n_component, random_state=seed).fit_transform(matrix)
    lo = reduced.min(axis=0)
    hi = reduced.max(axis=0)
    gaps = []
    deviations = []
    for k in range(1, MAX_K + 1):
        observed = dispersion(reduced, k, seed)
        reference = []
        for _ in range(REFERENCES):
            draw = rng.uniform(lo, hi, size=reduced.shape)
            reference.append(dispersion(draw, k, seed))
        reference = np.asarray(reference)
        gaps.append(float(reference.mean() - observed))
        deviations.append(
            float(reference.std() * np.sqrt(1.0 + 1.0 / REFERENCES))
        )
    gaps = np.asarray(gaps)
    deviations = np.asarray(deviations)
    chosen = MAX_K
    for index in range(len(gaps) - 1):
        if gaps[index] >= gaps[index + 1] - deviations[index + 1]:
            chosen = index + 1
            break
    return {"k": int(chosen), "gaps": gaps.tolist()}


def stability(matrix: np.ndarray, k: int, rng: np.random.Generator, seed: int) -> float:
    """Median ARI between clusterings of overlapping subsamples."""
    if k < 2 or len(matrix) < 8:
        return float("nan")
    scores = []
    size = max(4, int(len(matrix) * 0.8))
    for _ in range(BOOTSTRAPS):
        left = rng.choice(len(matrix), size, replace=False)
        right = rng.choice(len(matrix), size, replace=False)
        shared = np.intersect1d(left, right)
        if len(shared) < 4:
            continue
        a = KMeans(k, n_init=3, random_state=seed).fit_predict(matrix[left])
        b = KMeans(k, n_init=3, random_state=seed).fit_predict(matrix[right])
        index_a = {value: position for position, value in enumerate(left)}
        index_b = {value: position for position, value in enumerate(right)}
        scores.append(
            adjusted_rand_score(
                [a[index_a[v]] for v in shared], [b[index_b[v]] for v in shared]
            )
        )
    return float(np.median(scores)) if scores else float("nan")


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
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        block = store["hb_router_probs"][start:stop][
            :, :, DENOISE, ACTION_TOKENS, :
        ].astype(np.float32).mean(axis=2)
        block = block / np.maximum(block.sum(axis=-1, keepdims=True), 1e-12)
        deep[start:stop] = block[:, DEEP_AXES].reshape(stop - start, -1)
    order = np.lexsort((step_of_row, episode_of_row))
    deep = deep[order]
    episode_sorted = episode_of_row[order]
    deep_by = {
        int(e): deep[episode_sorted == e] for e in np.unique(episode_sorted)
    }

    by_snapshot = defaultdict(list)
    for candidate in candidates:
        by_snapshot[candidate.snapshot_key].append(candidate)

    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in sorted(by_snapshot.items()):
        group = sorted(group, key=lambda c: c.candidate)
        for query in range(MAX_QUERY):
            present = [
                c for c in group if len(deep_by.get(c.episode_id, [])) > query
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
            route_choice = gap_select(route, rng, SEED)
            noise_choice = gap_select(noise, rng, SEED)
            rows.append(
                {
                    "snapshot_key": key,
                    "query": query,
                    "route_k": route_choice["k"],
                    "noise_k": noise_choice["k"],
                    "route_stability": stability(route, route_choice["k"], rng, SEED),
                }
            )
        print(f"  {key} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "mode_count.csv", index=False)

    print("\n=== selected number of routing modes ===")
    print(f"{'query':>6} {'snaps':>6} {'route k=1':>10} {'k=2':>6} {'k=3':>6} "
          f"{'k>=4':>6} {'noise k=1':>10} {'stability':>10}")
    summary = []
    for query, part in frame.groupby("query"):
        counts = Counter(part.route_k)
        noise_counts = Counter(part.noise_k)
        record = {
            "query": int(query),
            "snapshots": int(len(part)),
            "route_k1": counts.get(1, 0),
            "route_k2": counts.get(2, 0),
            "route_k3": counts.get(3, 0),
            "route_k4plus": counts.get(4, 0),
            "noise_k1": noise_counts.get(1, 0),
            "median_stability": float(part.route_stability.median()),
        }
        summary.append(record)
        print(
            f"{query:>6} {len(part):>6} {record['route_k1']:>10} "
            f"{record['route_k2']:>6} {record['route_k3']:>6} "
            f"{record['route_k4plus']:>6} {record['noise_k1']:>10} "
            f"{record['median_stability']:>10.3f}"
        )

    with open(out_dir / "mode_count_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
