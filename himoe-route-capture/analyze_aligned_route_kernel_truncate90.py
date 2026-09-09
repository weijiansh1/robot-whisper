#!/usr/bin/env python3
"""Test aligned route-kernel sensitivity after removing the final 10% phase."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from analyze_aligned_route_kernel import (
    CANDIDATE_K,
    LANDMARKS,
    MAX_SHIFT,
    PERMUTATIONS,
    SEED,
    SUBSAMPLES,
    _load_source,
    _metadata,
    add_bh,
    aligned_landmark_distances,
    annotate_cross_task_blocks,
    association_bundle,
    build_anchor_signatures,
    canonicalize,
    cluster_landmark_view,
    cluster_profiles,
    minimum_cluster_size,
    normalize_trajectory_shape,
    per_task_associations,
    residualize_sequences,
)
from analyze_failure_routing_clusters import matched_cluster_jaccard, sha256_file


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE / "analysis/all-outcome-routing-clusters"
REFERENCE_DIR = HERE / "analysis/aligned-route-kernel"
OUT_DIR = HERE / "analysis/aligned-route-kernel-truncate90"
METHOD_SOURCE = OUT_DIR / "METHOD.md"

VARIANTS = ("raw", "task_residual")
REFERENCE_K = {"raw": 6, "task_residual": 2}
REFERENCE_RAW_BLOCK = 1
PERSISTENCE_MIN = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, default=SOURCE_DIR)
    parser.add_argument("--reference-dir", type=pathlib.Path, default=REFERENCE_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--subsamples", type=int, default=SUBSAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_truncate90_source(
    source_dir: pathlib.Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    arrays, manifest, completion = _load_source(source_dir)
    item = manifest["arrays"]["feature_truncate90_geometry"]
    path = source_dir / "feature_cache" / item["file"]
    if sha256_file(path) != item["sha256"]:
        raise ValueError("truncate90 geometry hash mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != item["shape"] or str(values.dtype) != item["dtype"]:
        raise ValueError("truncate90 geometry metadata mismatch")
    arrays["feature_truncate90_geometry"] = values
    return arrays, manifest, completion


def load_reference(
    reference_dir: pathlib.Path,
    source_dir: pathlib.Path,
    metadata: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    completion_path = reference_dir / "completion.json"
    completion = json.loads(completion_path.read_text())
    if completion.get("run_class") != "formal_exploratory":
        raise ValueError("reference aligned route-kernel run is not formal")
    for name in ("summary.json", "embeddings_labels_landmarks.npz", "assignments.csv"):
        if sha256_file(reference_dir / name) != completion["output_sha256"].get(name):
            raise ValueError(f"reference output hash mismatch: {name}")
    summary = json.loads((reference_dir / "summary.json").read_text())
    if summary["provenance"]["source_dir"] != str(source_dir.resolve()):
        raise ValueError("reference source directory differs from sensitivity source")
    if summary["representation"]["phase_window"] != [0.5, 1.0]:
        raise ValueError("reference does not use the expected full phase window")
    bundle = np.load(
        reference_dir / "embeddings_labels_landmarks.npz", allow_pickle=False
    )
    labels = {
        variant: np.asarray(bundle[f"label_{variant}"], dtype=np.int32)
        for variant in VARIANTS
    }
    with (reference_dir / "assignments.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(metadata["episode"]):
        raise ValueError("reference assignment count mismatch")
    for index, row in enumerate(rows):
        expected = (
            str(metadata["task"][index]),
            int(metadata["episode"][index]),
            int(metadata["init_state_id"][index]),
            int(metadata["flow_noise_seed"][index]),
        )
        actual = (
            row["task"],
            int(row["episode"]),
            int(row["init_state_id"]),
            int(row["flow_noise_seed"]),
        )
        if actual != expected:
            raise ValueError(f"reference row-order mismatch at row {index}")
    raw_profile = summary["variants"]["raw"]["profiles"][REFERENCE_RAW_BLOCK]
    if (
        summary["variants"]["raw"]["reported_k"] != REFERENCE_K["raw"]
        or REFERENCE_RAW_BLOCK
        not in summary["variants"]["raw"]["cross_task_failure_blocks"]
        or raw_profile["episodes"] != 252
        or raw_profile["failures"] != 246
    ):
        raise ValueError("reference raw C1 does not match the frozen 246/252 target")
    return summary, labels, completion


def fixed_ward_labels(
    embedding: np.ndarray, clusters: int, score: np.ndarray
) -> np.ndarray:
    tree = linkage(embedding, method="ward", optimal_ordering=False)
    labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
    if len(np.unique(labels)) != clusters:
        raise RuntimeError(f"Ward returned fewer than {clusters} clusters")
    return canonicalize(labels, score)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_overlap(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    true_positive = int(np.sum(reference & candidate))
    false_positive = int(np.sum(~reference & candidate))
    false_negative = int(np.sum(reference & ~candidate))
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "jaccard": ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
    }


def match_reference_block(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    reference_cluster: int,
) -> dict[str, Any]:
    reference = reference_labels == reference_cluster
    candidates = []
    for cluster in np.unique(candidate_labels):
        candidate = candidate_labels == cluster
        metrics = binary_overlap(reference, candidate)
        candidates.append({"cluster": int(cluster), **metrics})
    matched = max(
        candidates,
        key=lambda row: (row["jaccard"], row["true_positive"], -row["cluster"]),
    )
    matched_cluster = matched["cluster"]
    candidate = candidate_labels == matched_cluster
    failure = np.asarray(metadata["failure"], dtype=bool)
    failure_metrics = binary_overlap(reference & failure, candidate & failure)
    profiles = cluster_profiles(candidate_labels, metadata)
    profile = next(row for row in profiles if row["cluster"] == matched_cluster)
    tasks = sorted(set(metadata["task"][reference]) | set(metadata["task"][candidate]))
    composition = []
    for task in tasks:
        task_mask = metadata["task"] == task
        overlap = reference & candidate & task_mask
        reference_cell = reference & task_mask
        candidate_cell = candidate & task_mask
        candidate_profile = next(
            (
                row
                for row in profile["task_cells"]
                if row["task"] == str(task)
            ),
            None,
        )
        composition.append(
            {
                "task": str(task),
                "reference_episodes": int(reference_cell.sum()),
                "reference_failures": int((reference_cell & failure).sum()),
                "candidate_episodes": int(candidate_cell.sum()),
                "candidate_failures": int((candidate_cell & failure).sum()),
                "overlap_episodes": int(overlap.sum()),
                "overlap_failures": int((overlap & failure).sum()),
                "reference_episode_recall": ratio(
                    int(overlap.sum()), int(reference_cell.sum())
                ),
                "candidate_failure_rate_delta": (
                    candidate_profile.get("failure_rate_delta")
                    if candidate_profile is not None
                    else None
                ),
            }
        )
    reference_distribution = [
        {
            "candidate_cluster": int(cluster),
            "reference_episodes": int(np.sum(reference & (candidate_labels == cluster))),
            "reference_failures": int(
                np.sum(reference & failure & (candidate_labels == cluster))
            ),
        }
        for cluster in np.unique(candidate_labels)
    ]
    strong = bool(
        matched["precision"] >= PERSISTENCE_MIN
        and matched["recall"] >= PERSISTENCE_MIN
        and failure_metrics["precision"] >= PERSISTENCE_MIN
        and failure_metrics["recall"] >= PERSISTENCE_MIN
        and profile["all_eligible_deltas_positive"]
    )
    return {
        "reference_cluster": reference_cluster,
        "reference_episodes": int(reference.sum()),
        "reference_failures": int((reference & failure).sum()),
        "matched_candidate_cluster": matched_cluster,
        "candidate_episodes": profile["episodes"],
        "candidate_failures": profile["failures"],
        "candidate_failure_rate": profile["failure_rate"],
        "episode_overlap": matched,
        "failure_only_overlap": failure_metrics,
        "candidate_profile": profile,
        "per_task_composition": composition,
        "reference_distribution_across_candidate_clusters": reference_distribution,
        "strong_persistence": strong,
        "threshold": PERSISTENCE_MIN,
    }


def analyze_outcomes(
    labels: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
    fixed_raw: np.ndarray,
    metadata: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    for offset, variant in enumerate(VARIANTS):
        result = results[variant]
        result["associations"] = association_bundle(
            labels[variant], metadata, permutations, seed + 50000 + 1000 * offset
        )
        result["per_task"] = per_task_associations(
            labels[variant], metadata, permutations, seed + 60000 + 1000 * offset
        )
        result["profiles"] = cluster_profiles(labels[variant], metadata)

    fixed_trial = next(
        row
        for row in results["raw"]["trials"]
        if row["clusters"] == REFERENCE_K["raw"]
    )
    fixed_same_as_selected = bool(np.array_equal(fixed_raw, labels["raw"]))
    if fixed_same_as_selected:
        fixed_associations = results["raw"]["associations"]
    else:
        fixed_associations = association_bundle(
            fixed_raw, metadata, permutations, seed + 90000
        )
    fixed_result = {
        "status": "stable_partition" if fixed_trial["stable"] else "unstable_fixed_partition",
        "reported_k": REFERENCE_K["raw"],
        "reported_sizes": np.bincount(
            fixed_raw, minlength=REFERENCE_K["raw"]
        ).tolist(),
        "associations": fixed_associations,
        "profiles": cluster_profiles(fixed_raw, metadata),
        "same_as_selected_raw": fixed_same_as_selected,
        "selection_trial": copy.deepcopy(fixed_trial),
    }
    global_entries = [
        association
        for variant in VARIANTS
        for association in results[variant]["associations"].values()
    ]
    if not fixed_same_as_selected:
        global_entries.extend(fixed_associations.values())
    add_bh(global_entries)
    add_bh(
        [
            association
            for variant in VARIANTS
            for association in results[variant]["per_task"].values()
            if association["status"] == "estimated"
        ]
    )
    for variant in VARIANTS:
        annotate_cross_task_blocks(results[variant])
    annotate_cross_task_blocks(fixed_result)
    return fixed_result


def short_task(task: str) -> str:
    return task.split("/", 1)[-1]


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    persistence = summary["raw_c1_persistence"]
    fixed = summary["fixed_raw_k6"]
    candidate_cluster = persistence["matched_candidate_cluster"]
    ordinary_block = candidate_cluster in fixed["cross_task_failure_blocks"]
    if persistence["strong_persistence"] and ordinary_block:
        answer = (
            "Yes. Full-window raw C1 strongly persists after removing the final "
            "10% of relative phase and remains a cross-task failure block."
        )
    elif persistence["strong_persistence"]:
        answer = (
            "The episode set strongly overlaps full-window raw C1, but the matched "
            "truncate-90 cluster does not pass the ordinary cross-task block gate."
        )
    else:
        answer = (
            "No. Full-window raw C1 does not meet the frozen strong-persistence "
            "criterion after removing the final 10% of relative phase."
        )
    lines = [
        "# Truncate-90 aligned route-kernel sensitivity",
        "",
        f"Run class: `{summary['provenance']['run_class']}`.",
        "",
        answer,
        "",
        "## Raw C1 persistence",
        "",
        "| comparison | value |",
        "|---|---:|",
        f"| full raw K6 vs truncate-90 raw K6 ARI | {fmt(summary['agreement']['raw_fixed_k6_ari'])} |",
        f"| matched truncate-90 cluster | C{candidate_cluster} |",
        f"| reference episodes/failures | {persistence['reference_episodes']}/{persistence['reference_failures']} |",
        f"| candidate episodes/failures | {persistence['candidate_episodes']}/{persistence['candidate_failures']} |",
        f"| episode precision/recall/Jaccard | {fmt(persistence['episode_overlap']['precision'])}/{fmt(persistence['episode_overlap']['recall'])}/{fmt(persistence['episode_overlap']['jaccard'])} |",
        f"| failure precision/recall/Jaccard | {fmt(persistence['failure_only_overlap']['precision'])}/{fmt(persistence['failure_only_overlap']['recall'])}/{fmt(persistence['failure_only_overlap']['jaccard'])} |",
        f"| strong persistence | {str(persistence['strong_persistence']).lower()} |",
        f"| ordinary cross-task failure block retained | {str(ordinary_block).lower()} |",
        "",
        "### Per-task composition",
        "",
        "| task | reference n/fail | truncate-90 n/fail | overlap n/fail | reference recall | candidate failure-rate delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in persistence["per_task_composition"]:
        delta = row["candidate_failure_rate_delta"]
        lines.append(
            "| %s | %d/%d | %d/%d | %d/%d | %s | %s |"
            % (
                short_task(row["task"]),
                row["reference_episodes"],
                row["reference_failures"],
                row["candidate_episodes"],
                row["candidate_failures"],
                row["overlap_episodes"],
                row["overlap_failures"],
                fmt(row["reference_episode_recall"]),
                "n/a" if delta is None else fmt(delta),
            )
        )
    lines.extend(
        [
            "",
            "## Refit model selection",
            "",
            "| variant | status | selected/reported K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | failure/success blocks | full-label ARI |",
            "|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for variant in VARIANTS:
        result = summary["variants"][variant]
        trial = next(
            row for row in result["trials"] if row["clusters"] == result["reported_k"]
        )
        outcome = result["associations"]["outcome_within_task_initial_state"]
        lines.append(
            "| %s | %s | %s/%d | %s | %s | %s/%s | %s | %s | %s | %s | %s/%s | %s |"
            % (
                variant,
                result["status"],
                "none" if result["selected_k"] is None else result["selected_k"],
                result["reported_k"],
                "/".join(map(str, result["reported_sizes"])),
                fmt(trial["subsample_silhouette_mean"]),
                fmt(trial["ari_median"]),
                fmt(trial["ari_p10"]),
                fmt(trial["consensus_pac_01_09"]),
                fmt(result["associations"]["task"]["nmi"]),
                fmt(outcome["nmi_excess_over_null"]),
                fmt(outcome["fdr_bh_q"], 4),
                ",".join(map(str, result["cross_task_failure_blocks"])) or "none",
                ",".join(map(str, result["cross_task_success_blocks"])) or "none",
                fmt(summary["agreement"][f"{variant}_selected_ari"]),
            )
        )
    task_result = summary["variants"]["task_residual"]
    lines.extend(
        [
            "",
            "## Task-residual sensitivity",
            "",
            f"The fixed K2 full-vs-truncate ARI is {fmt(summary['agreement']['task_residual_fixed_k2_ari'])}. "
            f"The truncate-90 partition status is `{task_result['status']}` and its "
            f"cross-task failure/success blocks are {task_result['cross_task_failure_blocks']}/"
            f"{task_result['cross_task_success_blocks']}.",
            "",
            "## Interpretation limits",
            "",
            "- Raw persistence is a robustness result for the episode set, not evidence that the block is task-independent.",
            "- The candidate remains subject to task, episode-length, checkpoint, and timeout proxy explanations.",
            "- Reference matching is posthoc but occurs only after truncate-90 labels are frozen.",
            "- Stability is conditional on full-cohort anchor-PCA and landmark bases.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_model_selection(
    path: pathlib.Path, results: dict[str, dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, variant in zip(axes, VARIANTS):
        rows = results[variant]["trials"]
        k = [row["clusters"] for row in rows]
        axis.plot(k, [row["subsample_silhouette_mean"] for row in rows], "o-", label="silhouette")
        axis.plot(k, [row["ari_median"] for row in rows], "s-", label="ARI median")
        axis.plot(k, [row["ari_p10"] for row in rows], "^-", label="ARI p10")
        axis.plot(k, [row["consensus_pac_01_09"] for row in rows], "x-", label="PAC")
        axis.set_title(variant.replace("_", " "))
        axis.set_xlabel("K")
        axis.set_xticks(k)
        axis.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_raw_overlap(
    path: pathlib.Path, reference: np.ndarray, candidate: np.ndarray
) -> None:
    matrix = np.zeros(
        (int(reference.max()) + 1, int(candidate.max()) + 1), dtype=np.int64
    )
    for left, right in zip(reference, candidate):
        matrix[left, right] += 1
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xlabel("truncate-90 raw K6")
    axis.set_ylabel("full-window raw K6")
    axis.set_xticks(range(matrix.shape[1]), [f"C{i}" for i in range(matrix.shape[1])])
    axis.set_yticks(range(matrix.shape[0]), [f"C{i}" for i in range(matrix.shape[0])])
    axis.set_title("Episode overlap between raw K6 partitions")
    figure.colorbar(image, ax=axis, label="episodes")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_c1_composition(path: pathlib.Path, persistence: dict[str, Any]) -> None:
    rows = persistence["per_task_composition"]
    tasks = [short_task(row["task"]) for row in rows]
    position = np.arange(len(rows))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(
        position - width / 2,
        [row["reference_episodes"] for row in rows],
        width,
        label="full C1",
    )
    axes[0].bar(
        position + width / 2,
        [row["candidate_episodes"] for row in rows],
        width,
        label="truncate match",
    )
    axes[1].bar(
        position - width / 2,
        [row["reference_failures"] for row in rows],
        width,
        label="full C1",
    )
    axes[1].bar(
        position + width / 2,
        [row["candidate_failures"] for row in rows],
        width,
        label="truncate match",
    )
    for axis, title in zip(axes, ("Episodes by task", "Failures by task")):
        axis.set_xticks(position, tasks, rotation=25, ha="right")
        axis.set_title(title)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_assignments(
    path: pathlib.Path,
    metadata: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    fixed: dict[str, np.ndarray],
) -> None:
    fields = [
        "task",
        "checkpoint_sha256",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "episode_length",
        "outcome",
        "full_raw_cluster",
        "truncate90_raw_selected_cluster",
        "truncate90_raw_k6_cluster",
        "full_task_residual_cluster",
        "truncate90_task_residual_selected_cluster",
        "truncate90_task_residual_k2_cluster",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(metadata["episode"])):
            writer.writerow(
                {
                    "task": str(metadata["task"][index]),
                    "checkpoint_sha256": str(metadata["checkpoint_sha256"][index]),
                    "episode": int(metadata["episode"][index]),
                    "init_state_id": int(metadata["init_state_id"][index]),
                    "flow_noise_seed": int(metadata["flow_noise_seed"][index]),
                    "episode_length": int(metadata["episode_length"][index]),
                    "outcome": "failure" if metadata["failure"][index] else "success",
                    "full_raw_cluster": int(reference["raw"][index]),
                    "truncate90_raw_selected_cluster": int(selected["raw"][index]),
                    "truncate90_raw_k6_cluster": int(fixed["raw"][index]),
                    "full_task_residual_cluster": int(reference["task_residual"][index]),
                    "truncate90_task_residual_selected_cluster": int(
                        selected["task_residual"][index]
                    ),
                    "truncate90_task_residual_k2_cluster": int(
                        fixed["task_residual"][index]
                    ),
                }
            )


def self_test() -> None:
    reference = np.asarray([1, 1, 1, 0, 0, 2])
    candidate = np.asarray([2, 2, 0, 0, 0, 1])
    metadata = {
        "task": np.asarray(["a", "a", "a", "a", "a", "a"]),
        "failure": np.asarray([True, True, True, False, False, False]),
        "episode_length": np.ones(6),
    }
    result = match_reference_block(reference, candidate, metadata, 1)
    assert result["matched_candidate_cluster"] == 2
    assert result["episode_overlap"]["true_positive"] == 2
    assert np.isclose(result["episode_overlap"]["precision"], 1.0)
    assert np.isclose(result["episode_overlap"]["recall"], 2 / 3)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.subsamples < 10 or args.permutations < 100:
        raise ValueError("too few subsamples or permutations")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "completion.json").unlink(missing_ok=True)
    method_target = args.out_dir / "METHOD.md"
    if METHOD_SOURCE.resolve() != method_target.resolve():
        shutil.copyfile(METHOD_SOURCE, method_target)

    arrays, source_manifest, source_completion = load_truncate90_source(args.source_dir)
    metadata = _metadata(arrays)
    reference_summary, reference_labels, reference_completion = load_reference(
        args.reference_dir, args.source_dir, metadata
    )
    score = np.asarray(arrays["diagnostic_mean_soft_speed"])
    formal = bool(
        args.source_dir.resolve() == SOURCE_DIR.resolve()
        and args.reference_dir.resolve() == REFERENCE_DIR.resolve()
        and args.out_dir.resolve() == OUT_DIR.resolve()
        and args.subsamples == SUBSAMPLES
        and args.permutations == PERMUTATIONS
        and args.seed == SEED
    )

    distances: dict[str, np.ndarray] = {}
    landmarks: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    labels: dict[str, np.ndarray] = {}
    embeddings: dict[str, np.ndarray] = {}
    fixed: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=4):
        sequences, signature_audit = build_anchor_signatures(
            arrays["feature_truncate90_geometry"], args.seed
        )
        for offset, variant in enumerate(VARIANTS):
            print(f"building truncate90 aligned distances: {variant}", flush=True)
            variant_sequences = normalize_trajectory_shape(
                residualize_sequences(sequences, metadata, variant)
            )
            matrix, landmark_index, distance_audit = aligned_landmark_distances(
                variant_sequences, args.seed + 1000 * offset
            )
            distances[variant] = matrix
            landmarks[variant] = landmark_index
            print(f"clustering truncate90 distances: {variant}", flush=True)
            result, variant_labels, embedding = cluster_landmark_view(
                matrix, score, args.subsamples, args.seed + 10000 * offset
            )
            result["distance_audit"] = distance_audit
            results[variant] = result
            labels[variant] = variant_labels
            embeddings[variant] = embedding
            fixed[variant] = fixed_ward_labels(
                embedding, REFERENCE_K[variant], score
            )

    print("truncate90 labels frozen; revealing outcome metadata", flush=True)
    fixed_raw_result = analyze_outcomes(
        labels, results, fixed["raw"], metadata, args.permutations, args.seed
    )
    persistence = match_reference_block(
        reference_labels["raw"], fixed["raw"], metadata, REFERENCE_RAW_BLOCK
    )
    persistence["ordinary_cross_task_failure_block_retained"] = bool(
        persistence["matched_candidate_cluster"]
        in fixed_raw_result["cross_task_failure_blocks"]
    )
    agreement = {
        "raw_selected_ari": float(
            adjusted_rand_score(reference_labels["raw"], labels["raw"])
        ),
        "raw_fixed_k6_ari": float(
            adjusted_rand_score(reference_labels["raw"], fixed["raw"])
        ),
        "raw_fixed_k6_matched_jaccard": float(
            matched_cluster_jaccard(reference_labels["raw"], fixed["raw"])
        ),
        "task_residual_selected_ari": float(
            adjusted_rand_score(
                reference_labels["task_residual"], labels["task_residual"]
            )
        ),
        "task_residual_fixed_k2_ari": float(
            adjusted_rand_score(
                reference_labels["task_residual"], fixed["task_residual"]
            )
        ),
        "task_residual_fixed_k2_matched_jaccard": float(
            matched_cluster_jaccard(
                reference_labels["task_residual"], fixed["task_residual"]
            )
        ),
    }
    summary = {
        "schema": "himoe.aligned_route_kernel_truncate90.v1",
        "method": str(method_target.resolve()),
        "provenance": {
            "run_class": "formal_sensitivity" if formal else "nonformal",
            "source_dir": str(args.source_dir.resolve()),
            "source_completion_sha256": sha256_file(
                args.source_dir / "completion.json"
            ),
            "source_feature_manifest_sha256": sha256_file(
                args.source_dir / "feature_cache_manifest.json"
            ),
            "reference_dir": str(args.reference_dir.resolve()),
            "reference_completion_sha256": sha256_file(
                args.reference_dir / "completion.json"
            ),
            "reference_summary_sha256": sha256_file(
                args.reference_dir / "summary.json"
            ),
            "analysis_script_sha256": sha256_file(pathlib.Path(__file__)),
            "method_sha256": sha256_file(method_target),
            "seed": args.seed,
            "subsamples": args.subsamples,
            "permutations": args.permutations,
            "numeric_thread_limit": 4,
        },
        "source_audit": source_manifest["audit"],
        "source_completion_schema": source_completion["schema"],
        "reference_completion_schema": reference_completion["schema"],
        "reference": {
            "phase_window": reference_summary["representation"]["phase_window"],
            "raw_reported_k": reference_summary["variants"]["raw"]["reported_k"],
            "raw_failure_blocks": reference_summary["variants"]["raw"][
                "cross_task_failure_blocks"
            ],
            "raw_c1_episodes": 252,
            "raw_c1_failures": 246,
            "task_residual_reported_k": reference_summary["variants"][
                "task_residual"
            ]["reported_k"],
            "task_residual_status": reference_summary["variants"][
                "task_residual"
            ]["status"],
        },
        "representation": {
            **signature_audit,
            "source_feature": "feature_truncate90_geometry",
            "anchors": 10,
            "phase_window": [0.5, 0.9],
            "landmarks": LANDMARKS,
            "integer_shift_range": [-MAX_SHIFT, MAX_SHIFT],
            "outcome_used_before_label_freeze": False,
        },
        "selection": {
            "candidate_k": list(CANDIDATE_K),
            "minimum_cluster_size": minimum_cluster_size(len(metadata["failure"])),
            "subsamples": args.subsamples,
            "subsample_fraction": 0.80,
            "fixed_comparisons": REFERENCE_K,
        },
        "persistence_criterion": {
            "minimum_episode_precision": PERSISTENCE_MIN,
            "minimum_episode_recall": PERSISTENCE_MIN,
            "minimum_failure_precision": PERSISTENCE_MIN,
            "minimum_failure_recall": PERSISTENCE_MIN,
            "minimum_mixed_tasks": 3,
            "minimum_candidate_task_cell_episodes": 10,
            "same_direction": "positive_failure_rate_delta",
        },
        "variants": results,
        "fixed_raw_k6": fixed_raw_result,
        "agreement": agreement,
        "raw_c1_persistence": persistence,
    }

    write_assignments(
        args.out_dir / "assignments.csv", metadata, reference_labels, labels, fixed
    )
    np.savez_compressed(
        args.out_dir / "embeddings_labels_landmarks.npz",
        **{
            f"embedding_{variant}": embeddings[variant].astype(np.float32)
            for variant in VARIANTS
        },
        **{
            f"label_selected_{variant}": labels[variant].astype(np.int16)
            for variant in VARIANTS
        },
        **{
            f"label_fixed_{variant}": fixed[variant].astype(np.int16)
            for variant in VARIANTS
        },
        **{
            f"label_reference_{variant}": reference_labels[variant].astype(np.int16)
            for variant in VARIANTS
        },
        **{
            f"landmark_index_{variant}": landmarks[variant].astype(np.int32)
            for variant in VARIANTS
        },
        **{
            f"landmark_distance_{variant}": distances[variant].astype(np.float32)
            for variant in VARIANTS
        },
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    plot_model_selection(args.out_dir / "model_selection.png", results)
    plot_raw_overlap(
        args.out_dir / "raw_k6_overlap.png", reference_labels["raw"], fixed["raw"]
    )
    plot_c1_composition(args.out_dir / "raw_c1_composition.png", persistence)
    outputs = [
        "METHOD.md",
        "assignments.csv",
        "embeddings_labels_landmarks.npz",
        "summary.json",
        "report.md",
        "model_selection.png",
        "raw_k6_overlap.png",
        "raw_c1_composition.png",
    ]
    completion = {
        "schema": "himoe.aligned_route_kernel_truncate90.completion.v1",
        "run_class": "formal_sensitivity" if formal else "nonformal",
        "expected_outputs": outputs,
        "output_sha256": {
            name: sha256_file(args.out_dir / name) for name in outputs
        },
    }
    temporary = args.out_dir / "completion.json.tmp"
    temporary.write_text(json.dumps(completion, indent=2, allow_nan=False))
    temporary.replace(args.out_dir / "completion.json")
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
