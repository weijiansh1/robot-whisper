#!/usr/bin/env python3
"""Evaluate perturbation robustness and offline expert disagreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


BUNDLE = Path(__file__).resolve().parent.parent
PROJECT = BUNDLE.parent
FORK_CLIENT = PROJECT / "himoe-route-capture/runs/fork-pilot-n32-client"
FORK_SERVER = PROJECT / "himoe-route-capture/runs/fork-pilot-n32"
EXPERT_ROOT = PROJECT / "himoe-route-capture/analysis/expert-activation-hidden-matched"
STATE_IMPACT = PROJECT / "himoe-route-capture/analysis/moe-state-impact"
OUTPUT = BUNDLE / "results/routes_5_6"
RNG = np.random.default_rng(20260905)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def reconstruct_noise(episode: int, fork_step: int, candidate: int) -> np.ndarray:
    seed = (episode * 1000 + fork_step) * 100 + candidate
    return np.random.default_rng(seed).standard_normal((10, 24)).astype(np.float32)


def grouped_statistic(groups, outcomes, values, statistic) -> tuple[float, int]:
    per_group = []
    for group in np.unique(groups):
        selected = groups == group
        value = statistic(values[selected], outcomes[selected])
        if np.isfinite(value):
            per_group.append(value)
    return mean_finite(per_group), len(per_group)


def candidate_spearman(values: np.ndarray, outcomes: np.ndarray) -> float:
    return spearman(values, outcomes)


def pairwise_outcome_spearman(distance: np.ndarray, outcomes: np.ndarray) -> float:
    return spearman(squareform(distance, checks=False), pdist(outcomes[:, None]))


def grouped_permutation(groups, outcomes, values, statistic, permutations: int) -> dict:
    observed, used = grouped_statistic(groups, outcomes, values, statistic)
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        permuted = outcomes.copy()
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            permuted[selected] = RNG.permutation(permuted[selected])
        null[iteration], _ = grouped_statistic(groups, permuted, values, statistic)
    valid = null[np.isfinite(null)]
    p = (1 + np.count_nonzero(np.abs(valid) >= abs(observed))) / (1 + len(valid))
    return {
        "estimate": observed,
        "groups": used,
        "permutation_p_two_sided": float(p),
        "null_mean": mean_finite(valid),
        "null_q025": float(np.quantile(valid, 0.025)),
        "null_q975": float(np.quantile(valid, 0.975)),
    }


def grouped_distance_statistic(groups, outcomes, distances) -> tuple[float, int]:
    per_group = []
    for group in np.unique(groups):
        selected = groups == group
        value = pairwise_outcome_spearman(distances[int(group)], outcomes[selected])
        if np.isfinite(value):
            per_group.append(value)
    return mean_finite(per_group), len(per_group)


def grouped_distance_permutation(groups, outcomes, distances, permutations: int) -> dict:
    observed, used = grouped_distance_statistic(groups, outcomes, distances)
    null = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        permuted = outcomes.copy()
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            permuted[selected] = RNG.permutation(permuted[selected])
        null[iteration], _ = grouped_distance_statistic(groups, permuted, distances)
    valid = null[np.isfinite(null)]
    p = (1 + np.count_nonzero(np.abs(valid) >= abs(observed))) / (1 + len(valid))
    return {
        "estimate": observed,
        "groups": used,
        "permutation_p_two_sided": float(p),
        "null_mean": mean_finite(valid),
        "null_q025": float(np.quantile(valid, 0.025)),
        "null_q975": float(np.quantile(valid, 0.975)),
    }


def route5(permutations: int, output: Path) -> dict:
    record_path = FORK_CLIENT / "fork_records.json"
    records = json.loads(record_path.read_text(encoding="utf-8"))

    def keep(record) -> bool:
        spread = float(record.get("candidate_spread", 0.0))
        return spread <= 0 or float(record.get("rerun_drift", 0.0)) <= 0.05 * spread

    records = [record for record in records if keep(record)]
    row = np.asarray([record["trace_row"] for record in records], dtype=np.int64)
    group = np.asarray(
        [record["episode"] * 1000 + record["fork_step"] for record in records],
        dtype=np.int64,
    )
    outcome = np.asarray([record["drawer_delta"] for record in records], dtype=np.float64)
    success = np.asarray([record["success_in_window"] for record in records], dtype=bool)
    noise = np.stack(
        [
            reconstruct_noise(record["episode"], record["fork_step"], record["candidate"])
            for record in records
        ]
    ).reshape(len(records), -1)
    store = zarr.open(str(FORK_SERVER / "routes.zarr"), mode="r")
    probability = np.asarray(store["hb_router_probs"][row], dtype=np.float32)
    probability = probability[:, :, :, 1:, :]
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    sites = int(np.prod(probability.shape[1:-1]))
    full_root = np.sqrt(probability).reshape(len(records), -1) / np.sqrt(2.0 * sites)
    terminal = probability[:, :, -1, :, :]
    terminal_sites = int(np.prod(terminal.shape[1:-1]))
    terminal_root = np.sqrt(terminal).reshape(len(records), -1) / np.sqrt(
        2.0 * terminal_sites
    )

    local_sensitivity = np.empty(len(records), dtype=np.float64)
    terminal_sensitivity = np.empty(len(records), dtype=np.float64)
    snapshot_rows = []
    full_distance = {}
    for group_id in np.unique(group):
        selected = np.flatnonzero(group == group_id)
        noise_distance = squareform(pdist(noise[selected] / np.sqrt(noise.shape[1])))
        route_distance = squareform(pdist(full_root[selected]))
        terminal_distance = squareform(pdist(terminal_root[selected]))
        np.fill_diagonal(noise_distance, np.inf)
        nearest = noise_distance.argmin(axis=1)
        denominator = noise_distance[np.arange(len(selected)), nearest]
        local_sensitivity[selected] = route_distance[np.arange(len(selected)), nearest] / denominator
        terminal_sensitivity[selected] = (
            terminal_distance[np.arange(len(selected)), nearest] / denominator
        )
        np.fill_diagonal(noise_distance, 0.0)
        full_distance[group_id] = route_distance

        labels = AgglomerativeClustering(
            n_clusters=2, metric="precomputed", linkage="average"
        ).fit_predict(terminal_distance)
        counts = np.bincount(labels, minlength=2)
        silhouette = silhouette_score(terminal_distance, labels, metric="precomputed")
        snapshot_rows.append(
            {
                "snapshot": int(group_id),
                "route_dispersion": float(np.median(squareform(route_distance))),
                "terminal_route_dispersion": float(np.median(squareform(terminal_distance))),
                "noise_route_pair_spearman": spearman(
                    squareform(noise_distance, checks=False),
                    squareform(route_distance, checks=False),
                ),
                "outcome_spread": float(np.ptp(outcome[selected])),
                "successes": int(success[selected].sum()),
                "candidates": len(selected),
                "forced_two_basin_dominant_mass": float(counts.max() / counts.sum()),
                "forced_two_basin_silhouette": float(silhouette),
            }
        )

    route_outcome = grouped_distance_permutation(
        group, outcome, full_distance, permutations
    )
    sensitivity_outcome = grouped_permutation(
        group, outcome, local_sensitivity, candidate_spearman, permutations
    )
    snapshot_route = np.asarray([row["route_dispersion"] for row in snapshot_rows])
    snapshot_outcome = np.asarray([row["outcome_spread"] for row in snapshot_rows])
    snapshot_correlation = spearman(snapshot_route, snapshot_outcome)
    snapshot_null = np.asarray(
        [spearman(snapshot_route, RNG.permutation(snapshot_outcome)) for _ in range(permutations)]
    )
    snapshot_p = (1 + np.count_nonzero(np.abs(snapshot_null) >= abs(snapshot_correlation))) / (
        1 + permutations
    )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "route5_snapshot_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    np.savez_compressed(
        output / "route5_candidate_metrics.npz",
        group=group,
        local_route_sensitivity=local_sensitivity,
        terminal_route_sensitivity=terminal_sensitivity,
        drawer_delta=outcome,
        success_in_window=success,
    )
    return {
        "design": "same snapshot, 32 independent Gaussian flow-noise candidates",
        "not_measured": "infinitesimal perturbation robustness",
        "candidates": len(records),
        "snapshots": len(np.unique(group)),
        "mixed_success_snapshots": int(
            sum(0 < success[group == item].sum() < (group == item).sum() for item in np.unique(group))
        ),
        "route_distance_to_outcome_distance": route_outcome,
        "local_sensitivity_to_progress": sensitivity_outcome,
        "snapshot_route_dispersion_to_outcome_spread": {
            "spearman": snapshot_correlation,
            "permutation_p_two_sided": float(snapshot_p),
        },
        "mean_forced_two_basin_dominant_mass": mean_finite(
            [row["forced_two_basin_dominant_mass"] for row in snapshot_rows]
        ),
        "mean_forced_two_basin_silhouette": mean_finite(
            [row["forced_two_basin_silhouette"] for row in snapshot_rows]
        ),
        "source_sha256": {"fork_records.json": sha256(record_path)},
    }


def pool_spearman(predictor: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([spearman(predictor[pool], target[pool]) for pool in range(len(predictor))])


def hierarchical_bootstrap(per_task: list[np.ndarray], draws: int) -> tuple[float, float]:
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_ids = RNG.integers(0, len(per_task), len(per_task))
        task_values = []
        for task_id in task_ids:
            pools = per_task[task_id]
            selected = RNG.integers(0, len(pools), len(pools))
            task_values.append(np.nanmean(pools[selected]))
        values[draw] = np.nanmean(task_values)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def route6(bootstrap: int) -> dict:
    predictors = ("disagreement", "cancellation", "conflict")
    targets = ("block_sensitivity", "routed_sensitivity", "input_sensitivity")
    task_files = sorted(EXPERT_ROOT.glob("*/candidate_proxy_values.npz"))
    per_metric: dict[tuple[str, str], list[np.ndarray]] = {
        (predictor, target): [] for predictor in predictors for target in targets
    }
    validations = []
    source_hashes = {}
    for path in task_files:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
        for predictor in predictors:
            for target in targets:
                per_metric[(predictor, target)].append(
                    pool_spearman(arrays[predictor], arrays[target])
                )
        summary_path = path.with_name("summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validations.append(summary["offline_reconstruction_validation"])
        source_hashes[str(path.relative_to(PROJECT))] = sha256(path)

    metrics = []
    for (predictor, target), task_values in per_metric.items():
        task_means = [mean_finite(values) for values in task_values]
        low, high = hierarchical_bootstrap(task_values, bootstrap)
        metrics.append(
            {
                "predictor": predictor,
                "target": target,
                "task_mean_spearman": mean_finite(task_means),
                "hierarchical_bootstrap_low": low,
                "hierarchical_bootstrap_high": high,
                "task_spearman": task_means,
            }
        )

    with (STATE_IMPACT / "outcome.csv").open(newline="", encoding="utf-8") as handle:
        outcome_rows = list(csv.DictReader(handle))
    confirmation = [
        row
        for row in outcome_rows
        if not row["task"].startswith("libero_long/")
        and row["family"] in {"routed_full", "hidden_identity", "shared_identity"}
        and int(row["denoise"]) in {0, 8}
    ]
    outcome_summary = []
    for family in ("routed_full", "shared_identity", "hidden_identity"):
        for denoise in (0, 8):
            rows = [
                row
                for row in confirmation
                if row["family"] == family and int(row["denoise"]) == denoise
            ]
            outcome_summary.append(
                {
                    "family": family,
                    "denoise": denoise,
                    "confirmation_tasks": len(rows),
                    "task_mean_failure_auc": mean_finite(
                        [float(row["conditional_auc"]) for row in rows]
                    ),
                }
            )
    return {
        "design": "offline top-4 expert reconstruction at HB L5, denoise 0",
        "status": "mechanistic proxy, not a runtime causal intervention",
        "tasks": len(task_files),
        "pools": 16 * len(task_files),
        "candidates": 32 * 16 * len(task_files),
        "metrics": metrics,
        "reconstruction": {
            "mean_top4_set_match": mean_finite([row["top4_set_match"] for row in validations]),
            "mean_selected_probability_mae": mean_finite(
                [row["selected_probability_mae"] for row in validations]
            ),
        },
        "closed_loop_failure_controls": outcome_summary,
        "source_sha256": source_hashes,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "himoe.assurance.routes_5_6.v1",
        "route5": route5(args.permutations, args.output),
        "route6": route6(args.bootstrap),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
