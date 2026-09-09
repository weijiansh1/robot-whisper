#!/usr/bin/env python3
"""Learn a separate label-blind loop-motif vocabulary inside each task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from itertools import combinations
from typing import Any

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist, pdist, squareform

from analyze_unsupervised_replanning_loops import stratified_test, write_csv


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE / "analysis/unsupervised-replanning-loops"
OUT_DIR = HERE / "analysis/within-task-unsupervised-loops"
DEFAULT_PERMUTATIONS = 5000
DEFAULT_BOOTSTRAPS = 5000
N_RESTARTS = 5
K_VALUES = tuple(range(2, 9))
MAX_SILHOUETTE_SAMPLE = 1500
STABILITY_GATE = 0.8

FIT_FEATURES = (
    "physical_seed_strength_pct",
    "route_return_gain_pct",
    "action_return_gain_pct",
    "replay_physical_distance_pct",
    "replay_route_overlap_pct",
    "replay_action_distance_pct",
    "effect_distance_pct",
    "physical_closure_fraction",
    "route_recovery_fraction",
    "action_recovery_fraction",
    "log_physical_excursion_scale",
    "path_inefficiency",
)

FLOAT_FIELDS = tuple(
    dict.fromkeys(
        FIT_FEATURES
        + (
            "loop_score",
            "minimum_xra_recovery_fraction",
            "lag_fraction",
            "phase",
            "physical_closure_distance",
            "physical_excursion_distance",
            "physical_excursion_scale",
            "route_return_overlap",
            "route_return_gain",
            "action_return_distance",
            "action_return_gain",
            "replay_physical_distance",
            "replay_route_overlap",
            "replay_action_distance",
            "effect_distance",
        )
    )
)

INT_FIELDS = (
    "episode",
    "scene",
    "flow_noise_seed",
    "episode_length",
    "shared_prefix",
    "current_query",
    "past_query",
    "turn_query",
    "lag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-pairs", type=pathlib.Path, default=SOURCE_DIR / "seed_pairs.csv"
    )
    parser.add_argument(
        "--episode-scores",
        type=pathlib.Path,
        default=SOURCE_DIR / "episode_scores.csv",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def stable_seed(base: int, task: str, *parts: int) -> int:
    payload = f"{base}|{task}|" + "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_label_blind_events(path: pathlib.Path) -> list[dict[str, Any]]:
    events = []
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            # Deliberately construct a fresh record without source["failure"].
            row: dict[str, Any] = {
                "task": str(source["task"]),
                "positive_topology": source["positive_topology"] == "True",
            }
            row.update({name: int(source[name]) for name in INT_FIELDS})
            row.update({name: float(source[name]) for name in FLOAT_FIELDS})
            events.append(row)
    if not events:
        raise ValueError(f"no seed pairs in {path}")
    return events


def adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    left_labels, left_inverse = np.unique(left, return_inverse=True)
    right_labels, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((len(left_labels), len(right_labels)), np.int64)
    np.add.at(table, (left_inverse, right_inverse), 1)

    def choose_two(values: np.ndarray) -> float:
        values = np.asarray(values, np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    joint = choose_two(table)
    left_pairs = choose_two(table.sum(axis=1))
    right_pairs = choose_two(table.sum(axis=0))
    total_pairs = len(left) * (len(left) - 1.0) / 2.0
    if total_pairs <= 0:
        return 1.0
    expected = left_pairs * right_pairs / total_pairs
    maximum = 0.5 * (left_pairs + right_pairs)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if np.array_equal(left_inverse, right_inverse) else 0.0
    return float((joint - expected) / denominator)


def silhouette_from_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    values = np.zeros(len(labels), np.float64)
    for index in range(len(labels)):
        own = np.flatnonzero(labels == labels[index])
        own = own[own != index]
        if not len(own):
            continue
        within = float(distance[index, own].mean())
        between = min(
            float(distance[index, np.flatnonzero(labels == label)].mean())
            for label in unique
            if label != labels[index]
        )
        values[index] = (between - within) / max(within, between, 1e-12)
    return float(values.mean())


def fit_restarts(
    matrix: np.ndarray,
    count: int,
    task: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    runs = []
    for restart in range(N_RESTARTS):
        rng = np.random.default_rng(stable_seed(base_seed, task, count, restart))
        centroids, labels = kmeans2(
            matrix,
            count,
            iter=50,
            minit="++",
            missing="raise",
            rng=rng,
        )
        residual = matrix - centroids[labels]
        runs.append(
            {
                "centroids": np.asarray(centroids, np.float64),
                "labels": np.asarray(labels, np.int32),
                "inertia": float(np.square(residual).sum()),
            }
        )
    return runs


def fit_task_model(
    task: str,
    events: list[dict[str, Any]],
    base_seed: int,
) -> dict[str, Any]:
    training = [
        row
        for row in events
        if bool(row["positive_topology"])
        and int(row["current_query"]) < int(row["shared_prefix"])
    ]
    full_positive = [row for row in events if bool(row["positive_topology"])]
    if len(training) < 2 * 20:
        raise ValueError(f"too few positive-topology fit events for {task}")

    raw = np.asarray(
        [[float(row[name]) for name in FIT_FEATURES] for row in training],
        np.float64,
    )
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    keep = scale > 1e-10
    matrix = (raw[:, keep] - center[keep]) / scale[keep]
    sample_rng = np.random.default_rng(stable_seed(base_seed, task, 999))
    sample = np.sort(
        sample_rng.choice(
            len(matrix),
            size=min(len(matrix), MAX_SILHOUETTE_SAMPLE),
            replace=False,
        )
    )
    sample_distance = squareform(pdist(matrix[sample]))
    minimum_size = max(20, int(math.ceil(0.02 * len(training))))

    trials = []
    runtime = []
    for count in K_VALUES:
        restarts = fit_restarts(matrix, count, task, base_seed)
        best = min(restarts, key=lambda row: float(row["inertia"]))
        stability_values = [
            adjusted_rand(left["labels"], right["labels"])
            for left, right in combinations(restarts, 2)
        ]
        sizes = [
            int(np.sum(best["labels"] == label))
            for label in range(len(best["centroids"]))
        ]
        silhouette = silhouette_from_distance(sample_distance, best["labels"][sample])
        trial = {
            "clusters": count,
            "silhouette": silhouette,
            "mean_restart_ari": float(np.mean(stability_values)),
            "minimum_restart_ari": float(np.min(stability_values)),
            "sizes": sizes,
            "size_guard": bool(min(sizes) >= minimum_size),
            "stability_guard": bool(np.mean(stability_values) >= STABILITY_GATE),
            "best_inertia": float(best["inertia"]),
        }
        trials.append(trial)
        runtime.append({"trial": trial, "best": best})

    stable = [
        row
        for row in runtime
        if row["trial"]["size_guard"] and row["trial"]["stability_guard"]
    ]
    size_valid = [row for row in runtime if row["trial"]["size_guard"]]
    if stable:
        pool = stable
        status = "ok"
    elif size_valid:
        pool = size_valid
        status = "low_stability"
    else:
        pool = runtime[:1]
        status = "size_guard_failed"
    selected = max(pool, key=lambda row: float(row["trial"]["silhouette"]))
    best = selected["best"]
    labels = np.asarray(best["labels"], np.int32)

    ordered = sorted(
        range(len(best["centroids"])),
        key=lambda label: (
            float(
                np.median(
                    [
                        float(row["minimum_xra_recovery_fraction"])
                        for row, value in zip(training, labels)
                        if value == label
                    ]
                )
            ),
            float(
                np.median(
                    [
                        float(row["loop_score"])
                        for row, value in zip(training, labels)
                        if value == label
                    ]
                )
            ),
        ),
    )
    centroids = np.asarray([best["centroids"][label] for label in ordered])
    full_raw = np.asarray(
        [[float(row[name]) for name in FIT_FEATURES] for row in full_positive],
        np.float64,
    )
    full_matrix = (full_raw[:, keep] - center[keep]) / scale[keep]
    assigned = np.argmin(cdist(full_matrix, centroids), axis=1) + 1
    for row in events:
        row["within_task_cluster"] = 0
    for row, cluster in zip(full_positive, assigned):
        row["within_task_cluster"] = int(cluster)

    return {
        "task": task,
        "status": status,
        "training_events": len(training),
        "full_positive_events": len(full_positive),
        "minimum_cluster_size": minimum_size,
        "selected_clusters": int(len(centroids)),
        "selected_silhouette": float(selected["trial"]["silhouette"]),
        "selected_mean_restart_ari": float(selected["trial"]["mean_restart_ari"]),
        "active_features": [name for name, active in zip(FIT_FEATURES, keep) if active],
        "feature_center": center[keep].tolist(),
        "feature_scale": scale[keep].tolist(),
        "centroids": centroids.tolist(),
        "trials": trials,
    }


def label_blind_profiles(
    task: str, events: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    profiles = {}
    descriptive = (
        "loop_score",
        "minimum_xra_recovery_fraction",
        "physical_closure_fraction",
        "route_recovery_fraction",
        "action_recovery_fraction",
        "physical_excursion_scale",
        "path_inefficiency",
        "route_return_overlap",
        "route_return_gain",
        "action_return_distance",
        "action_return_gain",
        "replay_physical_distance",
        "replay_route_overlap",
        "replay_action_distance",
        "effect_distance",
        "lag",
        "lag_fraction",
        "phase",
    )
    for cluster in range(1, int(model["selected_clusters"]) + 1):
        full = [row for row in events if int(row["within_task_cluster"]) == cluster]
        training = [
            row for row in full if int(row["current_query"]) < int(row["shared_prefix"])
        ]
        profiles[str(cluster)] = {
            "task": task,
            "cluster": cluster,
            "training_events": len(training),
            "full_events": len(full),
            "full_episodes": len({int(row["episode"]) for row in full}),
            "strong_5pct_fraction": float(
                np.mean(
                    [
                        float(row["minimum_xra_recovery_fraction"]) >= 0.05
                        for row in full
                    ]
                )
            ),
            "medians": {
                name: float(np.median([float(row[name]) for row in full]))
                for name in descriptive
            },
        }
    return profiles


def load_outcomes(path: pathlib.Path) -> dict[tuple[str, int], bool]:
    outcomes = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            outcomes[(str(row["task"]), int(row["episode"]))] = row["failure"] == "True"
    if not outcomes:
        raise ValueError(f"no outcomes in {path}")
    return outcomes


def build_catalog(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalog = {}
    for task in sorted({str(row["task"]) for row in events}):
        subset = [row for row in events if str(row["task"]) == task]
        episode_rows = {}
        for row in subset:
            episode_rows.setdefault(int(row["episode"]), row)
        catalog[task] = {
            "episodes": len(episode_rows),
            "queries": sum(int(row["episode_length"]) for row in episode_rows.values()),
            "prefix": int(next(iter(episode_rows.values()))["shared_prefix"]),
            "scenes": {
                episode: int(row["scene"]) for episode, row in episode_rows.items()
            },
            "seeds": {
                episode: int(row["flow_noise_seed"])
                for episode, row in episode_rows.items()
            },
            "lengths": {
                episode: int(row["episode_length"])
                for episode, row in episode_rows.items()
            },
        }
    return catalog


def build_episode_records(
    task: str,
    events: list[dict[str, Any]],
    model: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    outcomes: dict[tuple[str, int], bool],
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict]] = {}
    for row in events:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    records = []
    for episode in range(int(catalog[task]["episodes"])):
        prefix = int(catalog[task]["prefix"])
        selected = [
            row for row in by_episode[episode] if int(row["current_query"]) < prefix
        ]
        if not selected:
            raise ValueError(f"no equal-prefix events for {task} episode {episode}")
        positive_count = sum(bool(row["positive_topology"]) for row in selected)
        record: dict[str, Any] = {
            "task": task,
            "episode": episode,
            "scene": int(catalog[task]["scenes"][episode]),
            "flow_noise_seed": int(catalog[task]["seeds"][episode]),
            "episode_length": int(catalog[task]["lengths"][episode]),
            "shared_prefix": prefix,
            "event_opportunities": len(selected),
            "positive_topology_rate": float(positive_count / len(selected)),
            "failure": bool(outcomes[(task, episode)]),
        }
        for cluster in range(1, int(model["selected_clusters"]) + 1):
            group = [
                row for row in selected if int(row["within_task_cluster"]) == cluster
            ]
            record[f"cluster_{cluster}_rate"] = float(len(group) / len(selected))
            record[f"cluster_{cluster}_share_given_positive"] = (
                float(len(group) / positive_count) if positive_count else 0.0
            )
            record[f"cluster_{cluster}_present"] = float(bool(group))
            record[f"cluster_{cluster}_max_score"] = (
                float(max(float(row["loop_score"]) for row in group)) if group else 0.0
            )
            record[f"cluster_{cluster}_max_recovery"] = (
                float(max(float(row["minimum_xra_recovery_fraction"]) for row in group))
                if group
                else 0.0
            )
        records.append(record)
    return records


def posthoc_composition_controls(
    task: str,
    episode_records: list[dict[str, Any]],
    model: dict[str, Any],
    permutations: int,
    bootstraps: int,
    base_seed: int,
    task_correction: int,
) -> dict[str, Any]:
    """Separate recurrence prevalence from cluster composition after unblinding."""
    topology_validation = stratified_test(
        episode_records,
        ("positive_topology_rate",),
        permutations,
        bootstraps,
        np.random.default_rng(stable_seed(base_seed, task, 20_001)),
    )
    positive_records = [
        row for row in episode_records if float(row["positive_topology_rate"]) > 0.0
    ]
    share_metrics = tuple(
        f"cluster_{cluster}_share_given_positive"
        for cluster in range(1, int(model["selected_clusters"]) + 1)
    )
    composition_validation = stratified_test(
        positive_records,
        share_metrics,
        permutations,
        bootstraps,
        np.random.default_rng(stable_seed(base_seed, task, 20_002)),
    )

    correction = max(task_correction, 1)
    for validation in (topology_validation, composition_validation):
        if validation.get("status") != "ok":
            continue
        for metric in validation["metrics"].values():
            metric["four_task_bonferroni_familywise_p"] = float(
                min(metric["familywise_p_one_sided"] * correction, 1.0)
            )

    failures = [row for row in episode_records if bool(row["failure"])]
    successes = [row for row in episode_records if not bool(row["failure"])]
    positive_failures = [row for row in positive_records if bool(row["failure"])]
    positive_successes = [row for row in positive_records if not bool(row["failure"])]
    shares = {}
    for cluster, metric_name in enumerate(share_metrics, start=1):
        shares[str(cluster)] = {
            "failure_mean": float(
                np.mean([float(row[metric_name]) for row in positive_failures])
            )
            if positive_failures
            else None,
            "success_mean": float(
                np.mean([float(row[metric_name]) for row in positive_successes])
            )
            if positive_successes
            else None,
            "test": composition_validation["metrics"][metric_name]
            if composition_validation.get("status") == "ok"
            else None,
        }
    return {
        "analysis_role": "posthoc diagnostic; not a preregistered primary endpoint",
        "positive_topology_rate": {
            "failure_mean": float(
                np.mean([float(row["positive_topology_rate"]) for row in failures])
            )
            if failures
            else None,
            "success_mean": float(
                np.mean([float(row["positive_topology_rate"]) for row in successes])
            ),
            "test": topology_validation["metrics"]["positive_topology_rate"]
            if topology_validation.get("status") == "ok"
            else None,
        },
        "composition_population": {
            "episodes_with_positive_topology": len(positive_records),
            "failures_with_positive_topology": len(positive_failures),
            "successes_with_positive_topology": len(positive_successes),
        },
        "cluster_share_given_positive": shares,
        "topology_validation": topology_validation,
        "composition_validation": composition_validation,
    }


def enrich_profiles_with_outcomes(
    task: str,
    profiles: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    validation: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    outcomes: dict[tuple[str, int], bool],
    task_correction: int,
) -> dict[str, dict[str, Any]]:
    enriched = json.loads(json.dumps(profiles))
    scene_rates = {}
    for scene in sorted(set(catalog[task]["scenes"].values())):
        episodes = [
            episode
            for episode, value in catalog[task]["scenes"].items()
            if value == scene
        ]
        scene_rates[scene] = float(
            np.mean([outcomes[(task, episode)] for episode in episodes])
        )
    task_correction = max(task_correction, 1)
    for cluster, profile in enriched.items():
        cluster_id = int(cluster)
        subset = [
            row for row in events if int(row["within_task_cluster"]) == cluster_id
        ]
        episode_ids = sorted({int(row["episode"]) for row in subset})
        failures = [outcomes[(task, episode)] for episode in episode_ids]
        expected = [
            scene_rates[int(catalog[task]["scenes"][episode])]
            for episode in episode_ids
        ]
        failure_records = [row for row in episode_records if bool(row["failure"])]
        success_records = [row for row in episode_records if not bool(row["failure"])]
        metric = f"cluster_{cluster_id}_rate"
        if validation.get("status") == "ok":
            test = validation["metrics"][metric]
            test["four_task_bonferroni_familywise_p"] = float(
                min(test["familywise_p_one_sided"] * task_correction, 1.0)
            )
        else:
            test = None
        profile["unblinded_outcome"] = {
            "failure_episodes_full": int(sum(failures)),
            "success_episodes_full": int(len(failures) - sum(failures)),
            "failure_fraction_full": float(np.mean(failures)),
            "matched_scene_baseline_full": float(np.mean(expected)),
            "equal_prefix_rate_failure_mean": float(
                np.mean([float(row[metric]) for row in failure_records])
            )
            if failure_records
            else None,
            "equal_prefix_rate_success_mean": float(
                np.mean([float(row[metric]) for row in success_records])
            ),
            "rate_test": test,
        }
    return enriched


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# 任务内无监督 loop motif",
        "",
        "每个任务独立在等长前缀的 positive-topology 事件上拟合 motif。",
        "聚类完成后才读取 outcome；phase、period、task identity 和 outcome 均未进入拟合。",
        "",
        "## 拟合概览",
        "",
        "| task | episodes / failures | fit / full positive events | selected K | silhouette | restart ARI | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, model in result["models"].items():
        coverage = result["coverage_by_task"][task]
        lines.append(
            "| %s | %d / %d | %d / %d | %d | %s | %s | %s |"
            % (
                task,
                coverage["episodes"],
                coverage["failures"],
                model["training_events"],
                model["full_positive_events"],
                model["selected_clusters"],
                fmt(model["selected_silhouette"]),
                fmt(model["selected_mean_restart_ari"]),
                model["status"],
            )
        )

    lines.extend(
        [
            "",
            "Cluster rate 的检验只使用等长前缀，并在该任务的 initial-state 内比较。",
            "`task FWER p` 校正该任务的全部 cluster rate；`4-task p` 再对四个含失败任务作 Bonferroni 校正。",
        ]
    )
    for task, model in result["models"].items():
        lines.extend(
            [
                "",
                f"## {task}",
                "",
                "| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        profiles = result["profiles_unblinded"][task]
        for cluster, profile in profiles.items():
            medians = profile["medians"]
            outcome = profile["unblinded_outcome"]
            test = outcome["rate_test"]
            if test:
                auc = fmt(test["auc_failure"])
                family_p = f"{test['familywise_p_one_sided']:.4f}"
                cross_task_p = f"{test['four_task_bonferroni_familywise_p']:.4f}"
            else:
                auc = family_p = cross_task_p = "-"
            lines.append(
                "| %s | %d / %d | %s | %.2f%% | %.1f%% | %s / %s | %s / %s | %s | %s | %s |"
                % (
                    cluster,
                    profile["training_events"],
                    profile["full_events"],
                    fmt(medians["loop_score"]),
                    100 * medians["minimum_xra_recovery_fraction"],
                    100 * profile["strong_5pct_fraction"],
                    fmt(medians["lag"]),
                    fmt(medians["phase"]),
                    fmt(outcome["equal_prefix_rate_failure_mean"]),
                    fmt(outcome["equal_prefix_rate_success_mean"]),
                    auc,
                    family_p,
                    cross_task_p,
                )
            )

    lines.extend(
        [
            "",
            "## 事后组成效应检查",
            "",
            "这一节不是预注册主检验。它检查显著 cluster rate 是否只是在复述 positive X/R/A-return topology 的总体频率。",
            "",
            "| task | positive-topology rate failure / success | failure AUC | task p | 4-task p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, control in result["posthoc_composition_controls"].items():
        topology = control["positive_topology_rate"]
        test = topology["test"]
        if test:
            auc = fmt(test["auc_failure"])
            family_p = f"{test['familywise_p_one_sided']:.4f}"
            cross_task_p = f"{test['four_task_bonferroni_familywise_p']:.4f}"
        else:
            auc = family_p = cross_task_p = "-"
        lines.append(
            "| %s | %s / %s | %s | %s | %s |"
            % (
                task,
                fmt(topology["failure_mean"]),
                fmt(topology["success_mean"]),
                auc,
                family_p,
                cross_task_p,
            )
        )

    significant = result["significant_failure_clusters"]
    lines.extend(["", "## 结论", ""])
    if significant:
        lines.append(
            "任务内无监督发现了以下 failure-associated motif（cluster-rate AUC > 0.5 且 task FWER p < 0.05）："
        )
        for row in significant:
            lines.append(
                "- %s / cluster %d: AUC=%s, task FWER p=%.4f, median min recovery=%.2f%%."
                % (
                    row["task"],
                    row["cluster"],
                    fmt(row["auc_failure"]),
                    row["task_familywise_p"],
                    100 * row["median_minimum_recovery"],
                )
            )
    else:
        lines.append(
            "没有 task-internal cluster rate 在任务内多重校正后与失败稳定关联。"
        )
    robust = [row for row in significant if row["four_task_bonferroni_p"] < 0.05]
    if robust:
        lines.append(
            "%d 个 motif 在进一步的四任务 Bonferroni 校正后仍保留。" % len(robust)
        )
    elif significant:
        lines.append("这些 motif 在额外的四任务校正后均未保留。")
    if significant:
        lines.extend(
            [
                "",
                "对上述显著 motif，再只看至少含一个 positive-topology 事件的 episode，并比较该 motif 在所有 positive-topology 事件中的占比：",
            ]
        )
        for row in significant:
            control = result["posthoc_composition_controls"][row["task"]]
            population = control["composition_population"]
            share = control["cluster_share_given_positive"][str(row["cluster"])]
            test = share["test"]
            if test:
                lines.append(
                    "- %s / cluster %d: %d episodes (%d failures), share AUC=%s, task FWER p=%.4f, 4-task p=%.4f."
                    % (
                        row["task"],
                        row["cluster"],
                        population["episodes_with_positive_topology"],
                        population["failures_with_positive_topology"],
                        fmt(test["auc_failure"]),
                        test["familywise_p_one_sided"],
                        test["four_task_bonferroni_familywise_p"],
                    )
                )
        lines.append(
            "若 composition effect 不保留，cluster-rate 关联主要应解释为回返拓扑总体更频繁，而不是某个无监督几何类型特异地增多。"
        )
    lines.extend(
        [
            "簇是任务内几何类型，不是滑落、空抓等语义失败标签。",
            "恢复幅度仍需单独检查；高 AUC 的 micro-return 不能称为强 basin loop。",
            "",
            "## 限制",
            "",
            "- 聚类是 transductive：使用所有 episode 的无标签等长前缀。",
            "- X 仍是 qpos，而不是 RGB/vision embedding。",
            "- 事件使用下一 query，只能用于发现而不是提前预警。",
            "- Outcome 关联不能证明 routing recurrence 导致失败。",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    left = np.asarray([0, 0, 1, 1, 2, 2])
    permuted = np.asarray([2, 2, 0, 0, 1, 1])
    different = np.asarray([0, 1, 0, 1, 0, 1])
    assert abs(adjusted_rand(left, permuted) - 1.0) < 1e-12
    assert adjusted_rand(left, different) < 0.0
    distance = squareform(pdist(np.asarray([[0.0], [0.1], [2.0], [2.1]])))
    assert silhouette_from_distance(distance, np.asarray([0, 0, 1, 1])) > 0.8
    assert stable_seed(1, "task", 2, 3) == stable_seed(1, "task", 2, 3)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.permutations < 1 or args.bootstraps < 1:
        raise ValueError("permutations and bootstraps must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = load_label_blind_events(args.seed_pairs)
    catalog = build_catalog(events)
    by_task = {
        task: [row for row in events if str(row["task"]) == task]
        for task in sorted(catalog)
    }
    models = {}
    profiles_label_blind = {}
    for task, task_events in by_task.items():
        model = fit_task_model(task, task_events, args.seed)
        models[task] = model
        profiles_label_blind[task] = label_blind_profiles(task, task_events, model)
        print(
            "fit %s: n=%d, K=%d, silhouette=%.3f, ARI=%.3f, %s"
            % (
                task,
                model["training_events"],
                model["selected_clusters"],
                model["selected_silhouette"],
                model["selected_mean_restart_ari"],
                model["status"],
            )
        )
    print("label-blind within-task models and assignments complete")

    # Outcome access begins only after every task model and assignment exists.
    outcomes = load_outcomes(args.episode_scores)
    for row in events:
        row["failure"] = bool(outcomes[(str(row["task"]), int(row["episode"]))])
    failing_task_count = sum(
        any(outcomes[(task, episode)] for episode in range(info["episodes"]))
        for task, info in catalog.items()
    )
    rng = np.random.default_rng(args.seed)
    validations = {}
    all_episode_records = []
    profiles_unblinded = {}
    for task, task_events in by_task.items():
        records = build_episode_records(
            task, task_events, models[task], catalog, outcomes
        )
        all_episode_records.extend(records)
        metrics = tuple(
            f"cluster_{cluster}_rate"
            for cluster in range(1, int(models[task]["selected_clusters"]) + 1)
        )
        validation = stratified_test(
            records,
            metrics,
            args.permutations,
            args.bootstraps,
            rng,
        )
        validations[task] = validation
        profiles_unblinded[task] = enrich_profiles_with_outcomes(
            task,
            profiles_label_blind[task],
            task_events,
            records,
            validation,
            catalog,
            outcomes,
            failing_task_count,
        )

    records_by_task = {
        task: [row for row in all_episode_records if str(row["task"]) == task]
        for task in by_task
    }
    posthoc_controls = {
        task: posthoc_composition_controls(
            task,
            records_by_task[task],
            models[task],
            args.permutations,
            args.bootstraps,
            args.seed,
            failing_task_count,
        )
        for task in by_task
    }

    significant = []
    for task, profiles in profiles_unblinded.items():
        for cluster, profile in profiles.items():
            test = profile["unblinded_outcome"]["rate_test"]
            if (
                test
                and test["auc_failure"] > 0.5
                and test["familywise_p_one_sided"] < 0.05
            ):
                significant.append(
                    {
                        "task": task,
                        "cluster": int(cluster),
                        "auc_failure": float(test["auc_failure"]),
                        "task_familywise_p": float(test["familywise_p_one_sided"]),
                        "four_task_bonferroni_p": float(
                            test["four_task_bonferroni_familywise_p"]
                        ),
                        "median_minimum_recovery": float(
                            profile["medians"]["minimum_xra_recovery_fraction"]
                        ),
                    }
                )

    significant_keys = {(row["task"], int(row["cluster"])) for row in significant}
    review = [
        row
        for row in events
        if bool(row["failure"])
        and int(row["current_query"]) < int(row["shared_prefix"])
        and (str(row["task"]), int(row["within_task_cluster"])) in significant_keys
    ]
    review = sorted(review, key=lambda row: float(row["loop_score"]), reverse=True)[
        :200
    ]

    coverage = {}
    for task, info in catalog.items():
        coverage[task] = {
            "episodes": int(info["episodes"]),
            "failures": int(
                sum(
                    outcomes[(task, episode)]
                    for episode in range(int(info["episodes"]))
                )
            ),
            "queries": int(info["queries"]),
            "shared_prefix": int(info["prefix"]),
        }
    result = {
        "schema": "within-task-unsupervised-loops/1",
        "source": str(args.seed_pairs),
        "preregistered_spec": str(args.out_dir / "PREREG.md"),
        "label_blind_models_completed_before_outcome_access": True,
        "definitions": {
            "fit_population": "within-task equal-prefix positive-X/R/A-topology seed pairs",
            "fit_features": list(FIT_FEATURES),
            "cluster_count": "K=2..8 by silhouette with size and five-restart ARI guards",
            "primary_external_endpoint": "within-task equal-prefix cluster rate",
            "outcome_role": "accessed only after all models and assignments",
        },
        "coverage_by_task": coverage,
        "models": models,
        "validations": validations,
        "posthoc_composition_controls": posthoc_controls,
        "profiles_unblinded": profiles_unblinded,
        "significant_failure_clusters": significant,
        "review_queue_events": len(review),
        "limitations": [
            "transductive unsupervised fit on all equal-prefix episodes",
            "qpos rather than RGB or a vision embedding",
            "next-query event discovery rather than early warning",
            "geometric motifs are not semantic failure-cause labels",
            "association is not a causal routing intervention",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(result))
    write_csv(args.out_dir / "assigned_events.csv", events)
    write_csv(args.out_dir / "episode_cluster_rates.csv", all_episode_records)
    write_csv(args.out_dir / "review_queue.csv", review)
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
