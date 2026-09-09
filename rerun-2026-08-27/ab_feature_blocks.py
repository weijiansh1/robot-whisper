#!/usr/bin/env python3
"""A/B the rollout-trend scalar block against its de-duplicated version.

`analyze_moe_rollout_trend.feature_blocks` puts routed commitment/adjacent into
BOTH the `routed` block and the `scalar` block, so `routed_full` and
`base_routed` reduce those columns twice through two independent PCA-12 stages.
This runs the primary seed-held-out contrast (k8 minus k0) with the shipped
block layout and with a de-duplicated scalar block, on the same compact cache.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_moe_rollout_trend as trend


CHUNKS = (0, 8)
EVALUATION = "seed_heldout"
REPORTED = ("routed_full", "routed_scalar", "hidden_identity", "shared_identity", "base", "base_routed")


SHIPPED = trend.feature_blocks


def shipped_blocks(data, chunk):
    return SHIPPED(data, chunk)


def deduplicated_blocks(data, chunk):
    """Same as shipped, except the scalar block holds only the scalar columns."""
    blocks = {}
    rows = len(data.failure)
    for branch in trend.BRANCHES:
        identity = getattr(data, branch)[:, chunk].reshape(rows, -1)
        commitment = getattr(data, branch + "_commitment")[:, chunk].reshape(rows, -1)
        adjacent = getattr(data, branch + "_adjacent")[:, chunk].reshape(rows, -1)
        blocks[branch] = np.column_stack((identity, commitment, adjacent))
    blocks["scalar"] = data.scalar[:, chunk].reshape(rows, -1)
    return blocks


def run_variant(tasks, builder, bootstrap, seed):
    """Return per-task conditional AUC at each chunk for every model family."""
    original = trend.feature_blocks
    trend.feature_blocks = builder
    try:
        results = {}
        for task_axis, data in enumerate(tasks):
            rng = np.random.default_rng(seed + task_axis * 10000)
            n_scenes = len(np.unique(data.scenes))
            draws = rng.integers(0, n_scenes, size=(bootstrap, n_scenes))
            for chunk in CHUNKS:
                predictions, manifold, strata = trend.crossfit_query(
                    data, chunk, EVALUATION, seed + task_axis * 10000
                )
                for family, score in predictions.items():
                    wins, pairs = trend.conditional_scene_stats(
                        data.failure, score, data.scenes, strata
                    )
                    results[("classifier", data.task, chunk, family)] = (
                        trend._auc(wins, pairs),
                        trend._bootstrap_auc(wins, pairs, draws),
                    )
                for family, score in manifold.items():
                    wins, pairs = trend.conditional_scene_stats(
                        data.failure, score, data.scenes, strata
                    )
                    results[("manifold", data.task, chunk, family)] = (
                        trend._auc(wins, pairs),
                        trend._bootstrap_auc(wins, pairs, draws),
                    )
                print("  %s chunk %d done" % (data.task, chunk), flush=True)
        return results
    finally:
        trend.feature_blocks = original


def summarize(results, task_names):
    output = {}
    for kind in ("classifier", "manifold"):
        for family in trend.MODEL_FAMILIES:
            points = {
                chunk: [results[(kind, task, chunk, family)][0] for task in task_names]
                for chunk in CHUNKS
            }
            boots = {
                chunk: np.nanmean(
                    [results[(kind, task, chunk, family)][1] for task in task_names],
                    axis=0,
                )
                for chunk in CHUNKS
            }
            contrast = boots[8] - boots[0]
            contrast = contrast[np.isfinite(contrast)]
            per_task = [points[8][i] - points[0][i] for i in range(len(task_names))]
            output[kind + ":" + family] = {
                "auc_k0": float(np.mean(points[0])),
                "auc_k8": float(np.mean(points[8])),
                "estimate": float(np.mean(per_task)),
                "ci95": [
                    float(np.percentile(contrast, 2.5)),
                    float(np.percentile(contrast, 97.5)),
                ],
                "tasks_positive": int(sum(v > 0 for v in per_task)),
                "per_task": per_task,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    tasks = []
    for path in sorted(args.compact_dir.glob("*.npz")):
        data = trend._load_cached(path, trend.N_CHUNKS, trend.PROJECTED_DIM)
        if data is None:
            raise ValueError("incompatible compact cache %s" % path)
        tasks.append(data)
    if len(tasks) != 4:
        raise ValueError("expected four compact caches, found %d" % len(tasks))
    task_names = [task.task for task in tasks]
    print("tasks: %s" % task_names, flush=True)

    print("== variant A: shipped feature_blocks ==", flush=True)
    a = summarize(run_variant(tasks, shipped_blocks, args.bootstrap, args.seed), task_names)
    print("== variant B: de-duplicated scalar block ==", flush=True)
    b = summarize(run_variant(tasks, deduplicated_blocks, args.bootstrap, args.seed), task_names)

    print()
    header = "%-34s %8s %8s %8s | %8s %8s %8s" % (
        "readout:family", "A k0", "A k8", "A d", "B k0", "B k8", "B d"
    )
    print(header)
    print("-" * len(header))
    lines = []
    for key in a:
        kind, family = key.split(":")
        if family not in REPORTED:
            continue
        row = "%-34s %8.3f %8.3f %+8.3f | %8.3f %8.3f %+8.3f" % (
            key,
            a[key]["auc_k0"], a[key]["auc_k8"], a[key]["estimate"],
            b[key]["auc_k0"], b[key]["auc_k8"], b[key]["estimate"],
        )
        print(row)
        lines.append(row)
    args.out.write_text(
        json.dumps(
            {
                "chunks": list(CHUNKS),
                "evaluation": EVALUATION,
                "bootstrap": args.bootstrap,
                "tasks": task_names,
                "shipped": a,
                "deduplicated": b,
            },
            indent=2,
        )
        + "\n"
    )
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
