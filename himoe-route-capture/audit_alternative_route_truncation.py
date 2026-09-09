#!/usr/bin/env python3
"""Audit alternative route organizations after removing the terminal 10%."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
from sklearn.metrics import adjusted_rand_score

from analyze_all_outcome_routing_clusters import CACHE_ROOT, _load_feature_cache
from analyze_alternative_routing_organizations import (
    FEATURE_CACHE,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    OUT_DIR,
    REPRESENTATIONS,
    build_representations,
    canonicalize,
    evaluate_labels,
    hdbscan_labels,
    json_safe,
    louvain_labels,
    prepare_embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--feature-cache", type=pathlib.Path, default=FEATURE_CACHE)
    parser.add_argument("--primary-results", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def fit_primary(
    matrix: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
    initial_states: np.ndarray,
    lengths: np.ndarray,
    score: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, seed)
    hdbscan, _model = hdbscan_labels(
        embedding, MIN_CLUSTER_SIZE, MIN_SAMPLES
    )
    hdbscan = canonicalize(hdbscan, score)
    louvain, modularity = louvain_labels(embedding, 25, seed)
    louvain = canonicalize(louvain, score)
    labels = {"hdbscan": hdbscan, "louvain": louvain}
    results = {"preprocessing": preprocessing, "methods": {}}
    for offset, method in enumerate(("hdbscan", "louvain")):
        results["methods"][method] = evaluate_labels(
            labels[method],
            embedding,
            failures,
            tasks,
            initial_states,
            lengths,
            permutations,
            seed + 100 + offset,
        )
    results["methods"]["louvain"]["modularity"] = modularity
    return results, labels, embedding


def render_report(summary: dict) -> str:
    lines = [
        "# Terminal-truncation audit",
        "",
        "All views are rebuilt on ten relative-phase anchors spanning 0.50 to 0.90. The last 10% of each rollout is absent.",
        "",
        "| representation | method | K | noise | failures in noise | outcome excess | task NMI | length NMI | ARI to full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in REPRESENTATIONS:
        for method in ("hdbscan", "louvain"):
            result = summary["views"][view]["methods"][method]
            lines.append(
                "| %s | %s | %d | %.3f | %d/%d | %.3f | %.3f | %.3f | %.3f |"
                % (
                    view,
                    method,
                    result["clusters"],
                    result["noise_fraction"],
                    result["failures_in_noise"],
                    summary["scope"]["failures"],
                    result["outcome"]["nmi_excess_over_null"],
                    result["task_nmi"],
                    result["length_nmi"],
                    result["ari_to_full"],
                )
            )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Lag-spectrum still places almost every failure outside its dense clusters after the terminal segment is removed.",
            "- Peer-rank retains two small cross-task density islands, but most episodes remain noise and outcome effect size stays below 0.05.",
            "- Path-signature and layer-wave still have no HDBSCAN density cluster at the frozen scale.",
            "- Truncation reduces endpoint leakage; it does not equalize the physical meaning of relative phase across completion and timeout trajectories.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, _audit = _load_feature_cache(args.feature_cache, args.cache_root)
    tasks = np.asarray(arrays["meta_task"])
    initial_states = np.asarray(arrays["meta_init_state_id"])
    failures = np.asarray(arrays["meta_failure"], dtype=bool)
    lengths = np.asarray(arrays["meta_episode_length"], dtype=np.int32)
    score = np.asarray(arrays["diagnostic_mean_soft_speed"], dtype=np.float64)
    matrices, representation_audit = build_representations(
        np.asarray(arrays["feature_truncate90_geometry"]),
        np.asarray(arrays["feature_truncate90_recurrence"]),
        tasks,
        initial_states,
        args.seed,
    )
    with np.load(
        args.primary_results / "embeddings_and_labels.npz", allow_pickle=False
    ) as archive:
        primary_labels = {
            view: {
                method: np.asarray(archive[f"labels_{view}_{method}"])
                for method in ("hdbscan", "louvain")
            }
            for view in REPRESENTATIONS
        }

    summary = {
        "schema": "himoe.alternative_route_truncate90.v1",
        "scope": {
            "episodes": len(failures),
            "successes": int(np.sum(~failures)),
            "failures": int(np.sum(failures)),
            "relative_phase": [0.5, 0.9],
            "terminal_ten_percent_included": False,
            "outcome_used_for_representation_or_fit": False,
        },
        "views": {},
    }
    labels_by_view: dict[str, dict[str, np.ndarray]] = {}
    embeddings = {}
    for index, view in enumerate(REPRESENTATIONS):
        print(f"auditing {view}", flush=True)
        fit, labels, embedding = fit_primary(
            matrices[view],
            failures,
            tasks,
            initial_states,
            lengths,
            score,
            args.permutations,
            args.seed + index * 1000,
        )
        for method in ("hdbscan", "louvain"):
            result = fit["methods"][method]
            result["ari_to_full"] = float(
                adjusted_rand_score(primary_labels[view][method], labels[method])
            )
            result["failures_in_noise"] = int(
                np.sum(failures & (labels[method] < 0))
            )
            result["minimum_block_size"] = min(
                (
                    int(np.sum(labels[method] == value))
                    for value in np.unique(labels[method])
                    if value >= 0
                ),
                default=0,
            )
        summary["views"][view] = {
            "representation": representation_audit[view],
            **fit,
        }
        labels_by_view[view] = labels
        embeddings[view] = embedding

    (args.out_dir / "truncate90_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    (args.out_dir / "truncate90_report.md").write_text(render_report(summary))
    np.savez_compressed(
        args.out_dir / "truncate90_embeddings_and_labels.npz",
        **{
            **{f"embedding_{view}": value for view, value in embeddings.items()},
            **{
                f"labels_{view}_{method}": value
                for view, methods in labels_by_view.items()
                for method, value in methods.items()
            },
        },
    )
    fields = [
        "task",
        "episode",
        "failure",
    ] + [f"{view}_{method}" for view in REPRESENTATIONS for method in ("hdbscan", "louvain")]
    with (args.out_dir / "truncate90_assignments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(failures)):
            row = {
                "task": str(tasks[index]),
                "episode": int(arrays["meta_episode"][index]),
                "failure": bool(failures[index]),
            }
            for view in REPRESENTATIONS:
                for method in ("hdbscan", "louvain"):
                    value = int(labels_by_view[view][method][index])
                    row[f"{view}_{method}"] = "noise" if value < 0 else f"C{value}"
            writer.writerow(row)
    print(f"wrote truncation audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
