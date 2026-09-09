#!/usr/bin/env python3
"""Audit whether peer-rank HDBSCAN C1 is a cross-task failure block.

This is deliberately separate from the exploratory clustering pipeline.  It
targets the already named C1 block, compares the full and truncate90 fits, and
keeps all outcome calculations post hoc.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy.stats import fisher_exact
from sklearn.cluster import HDBSCAN

from analyze_all_outcome_routing_clusters import CACHE_ROOT, _load_feature_cache
from analyze_alternative_routing_organizations import FEATURE_CACHE, peer_rank_features


HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "analysis/alternative-routing-organizations"
OUT_DIR = RESULTS_DIR / "peer-rank-c1-audit"
MIN_CLUSTER_GRID = (16, 24, 32, 40, 48, 64, 80)
MIN_SAMPLES_GRID = (4, 8, 12, 16, 24, 32)
CURVE_LAYOUT = (("point", 10), ("speed", 9), ("bend", 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=pathlib.Path, default=FEATURE_CACHE)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        return value.item() if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stratum_keys(*columns: np.ndarray) -> np.ndarray:
    return np.asarray(["::".join(map(str, row)) for row in zip(*columns)])


def wilson_interval(successes: int, total: int, level: float = 0.95) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    z = NormalDist().inv_cdf(0.5 + level / 2)
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2))
    half /= denominator
    return [float(center - half), float(center + half)]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def deterministic_expectation(
    membership: np.ndarray, failures: np.ndarray, strata: np.ndarray
) -> float:
    expected = 0.0
    for value in np.unique(strata):
        index = strata == value
        expected += float(membership[index].sum() * failures[index].mean())
    return expected


def stratified_enrichment(
    membership: np.ndarray,
    failures: np.ndarray,
    strata: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    membership = np.asarray(membership, dtype=bool)
    failures = np.asarray(failures, dtype=bool)
    observed = int(np.sum(membership & failures))
    selected = int(membership.sum())
    null = np.zeros(draws, dtype=np.int32)
    expected = 0.0
    mh_numerator = 0.0
    mh_denominator = 0.0
    covered = 0
    informative = 0
    positive = negative = tied = 0
    rng = np.random.default_rng(seed)
    for value in np.unique(strata):
        index = strata == value
        total = int(index.sum())
        selected_local = int(membership[index].sum())
        failure_local = int(failures[index].sum())
        if selected_local:
            covered += 1
        expected += selected_local * failure_local / total
        null += rng.hypergeometric(
            failure_local,
            total - failure_local,
            selected_local,
            size=draws,
        )

        a = int(np.sum(index & membership & failures))
        b = selected_local - a
        c = failure_local - a
        d = total - a - b - c
        mh_numerator += a * d / total
        mh_denominator += b * c / total
        if 0 < selected_local < total and 0 < failure_local < total:
            informative += 1
            difference = a / selected_local - c / (total - selected_local)
            if difference > 0:
                positive += 1
            elif difference < 0:
                negative += 1
            else:
                tied += 1

    rate = observed / selected
    expected_rate = expected / selected
    return {
        "selected": selected,
        "observed_failures": observed,
        "observed_failure_rate": rate,
        "failure_rate_wilson_95ci": wilson_interval(observed, selected),
        "expected_failures": expected,
        "expected_failure_rate": expected_rate,
        "failure_count_ratio": observed / expected if expected else None,
        "failure_rate_delta": rate - expected_rate,
        "mantel_haenszel_common_odds_ratio": (
            mh_numerator / mh_denominator if mh_denominator else None
        ),
        "permutation_draws": draws,
        "permutation_null_failure_count_mean": float(null.mean()),
        "permutation_null_failure_count_95ci": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "permutation_p_one_sided": float((1 + np.sum(null >= observed)) / (draws + 1)),
        "strata_total": int(len(np.unique(strata))),
        "strata_with_selected_episode": covered,
        "informative_strata": informative,
        "informative_strata_positive_negative_tied": [positive, negative, tied],
    }


def task_profiles(
    membership: np.ndarray, failures: np.ndarray, tasks: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for task in np.unique(tasks):
        task_mask = tasks == task
        selected = task_mask & membership
        complement = task_mask & ~membership
        selected_n = int(selected.sum())
        selected_failures = int(np.sum(selected & failures))
        task_failures = int(np.sum(task_mask & failures))
        complement_failures = int(np.sum(complement & failures))
        odds = fisher_exact(
            [
                [selected_failures, selected_n - selected_failures],
                [complement_failures, int(complement.sum()) - complement_failures],
            ],
            alternative="greater",
        )
        selected_rate = selected_failures / selected_n
        task_rate = task_failures / task_mask.sum()
        complement_rate = complement_failures / complement.sum()
        rows.append(
            {
                "task": str(task),
                "episodes": int(task_mask.sum()),
                "task_failures": task_failures,
                "task_failure_rate": float(task_rate),
                "c1_episodes": selected_n,
                "c1_failures": selected_failures,
                "c1_failure_rate": float(selected_rate),
                "c1_failure_rate_wilson_95ci": wilson_interval(
                    selected_failures, selected_n
                ),
                "delta_vs_task_rate": float(selected_rate - task_rate),
                "non_c1_failure_rate": float(complement_rate),
                "delta_vs_non_c1": float(selected_rate - complement_rate),
                "task_failure_recall": (
                    float(selected_failures / task_failures) if task_failures else None
                ),
                "fisher_exact_odds_ratio": float(odds.statistic),
                "fisher_exact_p_one_sided": float(odds.pvalue),
            }
        )
    adjusted = holm_adjust([row["fisher_exact_p_one_sided"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["fisher_exact_p_holm"] = value
    return rows


def set_overlap(left: np.ndarray, right: np.ndarray, universe: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=bool) & universe
    right = np.asarray(right, dtype=bool) & universe
    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    left_n = int(left.sum())
    right_n = int(right.sum())
    return {
        "left_n": left_n,
        "right_n": right_n,
        "intersection": intersection,
        "union": union,
        "jaccard": float(intersection / union) if union else None,
        "overlap_coefficient": (
            float(intersection / min(left_n, right_n)) if min(left_n, right_n) else None
        ),
        "left_retained_in_right": float(intersection / left_n) if left_n else None,
        "right_confirmed_by_left": float(intersection / right_n) if right_n else None,
    }


def overlap_audit(
    full: np.ndarray,
    truncated: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
) -> dict[str, Any]:
    all_episodes = np.ones(len(full), dtype=bool)
    return {
        "all_episodes": set_overlap(full, truncated, all_episodes),
        "failures_only": set_overlap(full, truncated, failures),
        "successes_only": set_overlap(full, truncated, ~failures),
        "by_task": [
            {
                "task": str(task),
                **set_overlap(full, truncated, tasks == task),
            }
            for task in np.unique(tasks)
        ],
    }


def phase_coordinates(view: str) -> dict[str, np.ndarray]:
    endpoint = 1.0 if view == "full" else 0.9
    point = np.linspace(0.5, endpoint, 10)
    return {
        "point": point,
        "speed": 0.5 * (point[:-1] + point[1:]),
        "bend": point[1:-1],
    }


def curve_profiles(
    view: str,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    c1 = labels == 1
    offset = 0
    summary = []
    long_rows = []
    phases = phase_coordinates(view)
    for name, width in CURVE_LAYOUT:
        curve = np.asarray(matrix[:, offset : offset + width], dtype=np.float64)
        offset += width
        means = {
            "c1": curve[c1].mean(axis=0),
            "non_c1": curve[~c1].mean(axis=0),
            "noise": curve[labels < 0].mean(axis=0),
            "c0": curve[labels == 0].mean(axis=0),
        }
        delta = means["c1"] - means["non_c1"]
        summary.append(
            {
                "curve": name,
                "c1_mean_rank": float(means["c1"].mean()),
                "non_c1_mean_rank": float(means["non_c1"].mean()),
                "mean_rank_delta": float(delta.mean()),
                "minimum_phase_delta": float(delta.min()),
                "maximum_phase_delta": float(delta.max()),
                "c1_peak_phase": float(phases[name][np.argmax(means["c1"])]),
                "c1_first_phase_rank": float(means["c1"][0]),
                "c1_last_phase_rank": float(means["c1"][-1]),
                "c1_curve": means["c1"].tolist(),
                "non_c1_curve": means["non_c1"].tolist(),
                "delta_curve": delta.tolist(),
            }
        )
        for index, phase in enumerate(phases[name]):
            long_rows.append(
                {
                    "view": view,
                    "curve": name,
                    "phase": float(phase),
                    "c1_mean_rank": float(means["c1"][index]),
                    "non_c1_mean_rank": float(means["non_c1"][index]),
                    "noise_mean_rank": float(means["noise"][index]),
                    "c0_mean_rank": float(means["c0"][index]),
                    "c1_minus_non_c1": float(delta[index]),
                }
            )
    return summary, long_rows


def high_deviation_cluster(labels: np.ndarray, raw_curve_score: np.ndarray) -> np.ndarray | None:
    clusters = np.unique(labels[labels >= 0])
    if not len(clusters):
        return None
    selected = max(clusters, key=lambda value: float(raw_curve_score[labels == value].mean()))
    return labels == selected


def sensitivity_grid(
    view: str,
    embedding: np.ndarray,
    matrix: np.ndarray,
    reference: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
    task_initial: np.ndarray,
    task_initial_length: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray | None]]:
    raw_curve_score = np.asarray(matrix[:, :27], dtype=np.float64).mean(axis=1)
    rows = []
    masks: dict[tuple[int, int], np.ndarray | None] = {}
    for minimum in MIN_CLUSTER_GRID:
        for samples in MIN_SAMPLES_GRID:
            model = HDBSCAN(
                min_cluster_size=minimum,
                min_samples=samples,
                cluster_selection_method="eom",
                allow_single_cluster=False,
                n_jobs=-1,
                copy=True,
            ).fit(embedding)
            labels = np.asarray(model.labels_, dtype=np.int32)
            candidate = high_deviation_cluster(labels, raw_curve_score)
            masks[(minimum, samples)] = candidate
            row: dict[str, Any] = {
                "view": view,
                "min_cluster_size": minimum,
                "min_samples": samples,
                "clusters": int(len(np.unique(labels[labels >= 0]))),
                "noise_fraction": float(np.mean(labels < 0)),
                "high_deviation_block_exists": candidate is not None,
            }
            if candidate is not None:
                intersection = int(np.sum(candidate & reference))
                union = int(np.sum(candidate | reference))
                expected = deterministic_expectation(candidate, failures, task_initial)
                expected_length = deterministic_expectation(
                    candidate, failures, task_initial_length
                )
                observed = int(np.sum(candidate & failures))
                counts = Counter(tasks[candidate].tolist())
                row.update(
                    {
                        "block_size": int(candidate.sum()),
                        "failure_count": observed,
                        "failure_rate": float(failures[candidate].mean()),
                        "task_count": len(counts),
                        "dominant_task_fraction": float(max(counts.values()) / candidate.sum()),
                        "raw_curve_mean_rank": float(raw_curve_score[candidate].mean()),
                        "jaccard_to_primary_c1": float(intersection / union),
                        "primary_c1_recall": float(intersection / reference.sum()),
                        "candidate_precision_against_primary_c1": float(
                            intersection / candidate.sum()
                        ),
                        "task_initial_failure_count_ratio": (
                            float(observed / expected) if expected else None
                        ),
                        "task_initial_length_failure_count_ratio": (
                            float(observed / expected_length) if expected_length else None
                        ),
                    }
                )
            rows.append(row)
    return rows, masks


def summarize_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [row for row in rows if row["high_deviation_block_exists"]]
    return {
        "settings": len(rows),
        "viable_settings": len(viable),
        "no_cluster_settings": len(rows) - len(viable),
        "jaccard_to_primary_median": float(
            np.median([row["jaccard_to_primary_c1"] for row in viable])
        ),
        "jaccard_to_primary_range": [
            float(min(row["jaccard_to_primary_c1"] for row in viable)),
            float(max(row["jaccard_to_primary_c1"] for row in viable)),
        ],
        "failure_rate_range": [
            float(min(row["failure_rate"] for row in viable)),
            float(max(row["failure_rate"] for row in viable)),
        ],
        "block_size_range": [
            int(min(row["block_size"] for row in viable)),
            int(max(row["block_size"] for row in viable)),
        ],
        "all_viable_blocks_cover_all_five_tasks": bool(
            all(row["task_count"] == 5 for row in viable)
        ),
    }


def cross_view_sensitivity(
    full_masks: dict[tuple[int, int], np.ndarray | None],
    truncated_masks: dict[tuple[int, int], np.ndarray | None],
    failures: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for setting in full_masks:
        full = full_masks[setting]
        truncated = truncated_masks[setting]
        if full is None or truncated is None:
            continue
        rows.append(
            {
                "min_cluster_size": setting[0],
                "min_samples": setting[1],
                "all_episode_jaccard": set_overlap(
                    full, truncated, np.ones(len(full), dtype=bool)
                )["jaccard"],
                "failure_jaccard": set_overlap(full, truncated, failures)["jaccard"],
            }
        )
    return {
        "jointly_viable_settings": len(rows),
        "all_episode_jaccard_median": float(
            np.median([row["all_episode_jaccard"] for row in rows])
        ),
        "all_episode_jaccard_range": [
            float(min(row["all_episode_jaccard"] for row in rows)),
            float(max(row["all_episode_jaccard"] for row in rows)),
        ],
        "failure_jaccard_median": float(
            np.median([row["failure_jaccard"] for row in rows])
        ),
        "failure_jaccard_range": [
            float(min(row["failure_jaccard"] for row in rows)),
            float(max(row["failure_jaccard"] for row in rows)),
        ],
        "settings": rows,
    }


def short_task(task: str) -> str:
    return task.split("/", 1)[-1]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Peer-rank HDBSCAN C1 audit",
        "",
        "## Verdict",
        "",
        "C1 is a real, persistently high peer-deviation density island and its core",
        "replicates after terminal truncation. It is descriptively failure-enriched",
        "after task x initial-state conditioning, but the enrichment vanishes after",
        "conditioning on exact episode length. The unqualified phrase `cross-task",
        "high-deviation failure-enriched block` therefore overstates the result.",
        "",
        "A defensible description is: `a cross-task, high peer-deviation density",
        "island whose membership is associated with failure in this corpus, largely",
        "through longer or timeout trajectories`.",
        "",
        "## Conditional enrichment",
        "",
        "| view | C1 n | failures | precision | task-init expected | ratio | delta | p | task-init-length expected | ratio | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in ("full", "truncate90"):
        result = summary["views"][view]
        first = result["enrichment_task_initial"]
        second = result["enrichment_task_initial_length"]
        lines.append(
            "| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                view,
                first["selected"],
                first["observed_failures"],
                fmt(first["observed_failure_rate"]),
                fmt(first["expected_failures"], 2),
                fmt(first["failure_count_ratio"]),
                fmt(first["failure_rate_delta"]),
                fmt(first["permutation_p_one_sided"], 5),
                fmt(second["expected_failures"], 2),
                fmt(second["failure_count_ratio"]),
                fmt(second["permutation_p_one_sided"], 5),
            )
        )

    lines.extend(
        [
            "",
            "The task-init test uses conditional hypergeometric permutations, preserving",
            "the failure count in every one of the 80 task x initial-state cells.",
            "Exact length is outcome-proximal and is included as a leakage/confounding",
            "diagnostic, not as a causal adjustment.",
            "",
            "## Per-task precision",
            "",
            "| view | task | C1 n | C1 failures | C1 rate | task rate | delta | failure recall | Fisher p | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for view in ("full", "truncate90"):
        for row in summary["views"][view]["task_profiles"]:
            lines.append(
                "| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s |"
                % (
                    view,
                    short_task(row["task"]),
                    row["c1_episodes"],
                    row["c1_failures"],
                    fmt(row["c1_failure_rate"]),
                    fmt(row["task_failure_rate"]),
                    fmt(row["delta_vs_task_rate"]),
                    fmt(row["task_failure_recall"]),
                    fmt(row["fisher_exact_p_one_sided"], 5),
                    fmt(row["fisher_exact_p_holm"], 5),
                )
            )

    overlap = summary["full_truncate90_overlap"]
    lines.extend(
        [
            "",
            "## Full versus truncate90",
            "",
            "| subset | full n | trunc n | intersection | Jaccard | full retained | trunc confirmed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("all_episodes", "failures_only", "successes_only"):
        row = overlap[name]
        lines.append(
            "| %s | %d | %d | %d | %s | %s | %s |"
            % (
                name,
                row["left_n"],
                row["right_n"],
                row["intersection"],
                fmt(row["jaccard"]),
                fmt(row["left_retained_in_right"]),
                fmt(row["right_confirmed_by_left"]),
            )
        )

    lines.extend(
        [
            "",
            "## Feature curves",
            "",
            "The raw curve values are within-task x initial-state peer ranks. A value",
            "near 0.84 means that C1 stays near the high-deviation end of its 32-rollout",
            "sibling group; it is not an absolute routing-distance measurement.",
            "",
            "| view | curve | C1 mean rank | non-C1 | delta | min phase delta | first | last |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for view in ("full", "truncate90"):
        for row in summary["views"][view]["curve_profiles"]:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    view,
                    row["curve"],
                    fmt(row["c1_mean_rank"]),
                    fmt(row["non_c1_mean_rank"]),
                    fmt(row["mean_rank_delta"]),
                    fmt(row["minimum_phase_delta"]),
                    fmt(row["c1_first_phase_rank"]),
                    fmt(row["c1_last_phase_rank"]),
                )
            )

    lines.extend(
        [
            "",
            "All three curves are elevated at every phase. Full C1 weakens slightly at",
            "the terminal anchor instead of being created by a terminal-only spike; the",
            "truncate90 curves remain elevated through phase 0.90.",
            "",
            "## HDBSCAN sensitivity",
            "",
            "Each candidate below is selected label-blind as the discovered cluster with",
            "the highest mean raw peer-rank curve. The full grid is in `sensitivity.csv`.",
            "",
            "| view | viable/grid | Jaccard to default median (range) | size range | failure-rate range | all viable cover 5 tasks |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for view in ("full", "truncate90"):
        row = summary["views"][view]["sensitivity_summary"]
        lines.append(
            "| %s | %d/%d | %s (%s-%s) | %d-%d | %s-%s | %s |"
            % (
                view,
                row["viable_settings"],
                row["settings"],
                fmt(row["jaccard_to_primary_median"]),
                fmt(row["jaccard_to_primary_range"][0]),
                fmt(row["jaccard_to_primary_range"][1]),
                row["block_size_range"][0],
                row["block_size_range"][1],
                fmt(row["failure_rate_range"][0]),
                fmt(row["failure_rate_range"][1]),
                str(row["all_viable_blocks_cover_all_five_tasks"]).lower(),
            )
        )
    cross = summary["sensitivity_cross_view"]
    lines.extend(
        [
            "",
            (
                "Across %d settings where both views produce clusters, the "
                "full/truncate90 high-block Jaccard has median %s (range %s-%s); "
                "failure-only Jaccard has median %s (range %s-%s)."
            )
            % (
                cross["jointly_viable_settings"],
                fmt(cross["all_episode_jaccard_median"]),
                fmt(cross["all_episode_jaccard_range"][0]),
                fmt(cross["all_episode_jaccard_range"][1]),
                fmt(cross["failure_jaccard_median"]),
                fmt(cross["failure_jaccard_range"][0]),
                fmt(cross["failure_jaccard_range"][1]),
            ),
            "",
            "Some conservative settings return no cluster at all. With",
            "`allow_single_cluster=False`, raising min_cluster_size above the small",
            "low-deviation companion island can invalidate the entire two-island fit,",
            "even though the high-deviation candidate itself is much larger.",
            "",
            "## Limits",
            "",
            "- This is a targeted post-hoc audit of a named block, not a discovery p-value.",
            "- Four tasks contain failures; the middle-drawer task has zero failures and cannot validate enrichment.",
            "- The strong evidence is concentrated in the two spatial bowl tasks; direction is positive but weaker in top-drawer and moka-pot.",
            "- Full and truncate90 use relative-phase anchors with different terminal semantics.",
            "- No unseen task is available, so cross-task means within-corpus coverage, not task generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    failures = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    selected = np.asarray([0, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    strata = np.asarray(["a"] * 4 + ["b"] * 4)
    result = stratified_enrichment(selected, failures, strata, 1000, 8)
    assert result["selected"] == 4
    assert result["observed_failures"] == 3
    assert set_overlap(selected, selected, np.ones(8, dtype=bool))["jaccard"] == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.permutations < 10_000:
        raise ValueError("use at least 10,000 conditional permutation draws")

    arrays, cache_audit = _load_feature_cache(args.feature_cache, args.cache_root)
    tasks = np.asarray(arrays["meta_task"]).astype(str)
    initial_states = np.asarray(arrays["meta_init_state_id"], dtype=np.int32)
    lengths = np.asarray(arrays["meta_episode_length"], dtype=np.int32)
    failures = np.asarray(arrays["meta_failure"], dtype=bool)
    task_initial = stratum_keys(tasks, initial_states)
    task_initial_length = stratum_keys(tasks, initial_states, lengths)

    with np.load(args.results_dir / "representation_features.npz", allow_pickle=False) as archive:
        full_matrix = np.asarray(archive["peer_rank"], dtype=np.float32)
    recomputed_full, _audit = peer_rank_features(
        np.asarray(arrays["feature_primary_geometry"]).reshape(len(failures), 10, -1),
        tasks,
        initial_states,
    )
    if not np.allclose(full_matrix, recomputed_full, atol=1e-6, rtol=1e-6):
        raise ValueError("stored full peer-rank features do not reproduce")
    truncated_matrix, _audit = peer_rank_features(
        np.asarray(arrays["feature_truncate90_geometry"]).reshape(len(failures), 10, -1),
        tasks,
        initial_states,
    )

    with np.load(args.results_dir / "embeddings_and_labels.npz", allow_pickle=False) as archive:
        full_embedding = np.asarray(archive["embedding_peer_rank"], dtype=np.float64)
        full_labels = np.asarray(archive["labels_peer_rank_hdbscan"], dtype=np.int32)
    with np.load(
        args.results_dir / "truncate90_embeddings_and_labels.npz", allow_pickle=False
    ) as archive:
        truncated_embedding = np.asarray(archive["embedding_peer_rank"], dtype=np.float64)
        truncated_labels = np.asarray(archive["labels_peer_rank_hdbscan"], dtype=np.int32)
    for labels in (full_labels, truncated_labels):
        if set(np.unique(labels)) != {-1, 0, 1}:
            raise ValueError("expected noise, C0, and C1 labels")

    matrices = {"full": full_matrix, "truncate90": truncated_matrix}
    embeddings = {"full": full_embedding, "truncate90": truncated_embedding}
    labels_by_view = {"full": full_labels, "truncate90": truncated_labels}
    memberships = {view: labels == 1 for view, labels in labels_by_view.items()}
    curve_rows = []
    sensitivity_rows = []
    sensitivity_masks = {}
    views = {}
    for offset, view in enumerate(("full", "truncate90")):
        labels = labels_by_view[view]
        membership = memberships[view]
        curves, long_rows = curve_profiles(view, matrices[view], labels)
        curve_rows.extend(long_rows)
        grid, masks = sensitivity_grid(
            view,
            embeddings[view],
            matrices[view],
            membership,
            failures,
            tasks,
            task_initial,
            task_initial_length,
        )
        sensitivity_rows.extend(grid)
        sensitivity_masks[view] = masks
        raw_curve_score = matrices[view][:, :27].mean(axis=1)
        cluster_curve_means = {
            f"C{cluster}": float(raw_curve_score[labels == cluster].mean())
            for cluster in (0, 1)
        }
        if cluster_curve_means["C1"] <= cluster_curve_means["C0"]:
            raise ValueError("named C1 is not the high-deviation island")
        views[view] = {
            "c1_episodes": int(membership.sum()),
            "c1_failures": int(np.sum(membership & failures)),
            "c1_failure_rate": float(failures[membership].mean()),
            "all_episode_failure_rate": float(failures.mean()),
            "cluster_raw_curve_mean_rank": cluster_curve_means,
            "enrichment_task_initial": stratified_enrichment(
                membership,
                failures,
                task_initial,
                args.permutations,
                args.seed + 100 * offset,
            ),
            "enrichment_task_initial_length": stratified_enrichment(
                membership,
                failures,
                task_initial_length,
                args.permutations,
                args.seed + 100 * offset + 1,
            ),
            "task_profiles": task_profiles(membership, failures, tasks),
            "curve_profiles": curves,
            "sensitivity_summary": summarize_sensitivity(grid),
        }

    cross_sensitivity = cross_view_sensitivity(
        sensitivity_masks["full"], sensitivity_masks["truncate90"], failures
    )
    summary = {
        "schema": "himoe.peer_rank_hdbscan_c1_audit.v1",
        "scope": {
            "episodes": int(len(failures)),
            "failures": int(failures.sum()),
            "tasks": int(len(np.unique(tasks))),
            "task_initial_state_cells": int(len(np.unique(task_initial))),
            "conditional_permutation_draws": args.permutations,
            "outcome_used_for_representation_or_clustering": False,
            "outcome_used_for_targeted_audit": True,
        },
        "inputs": {
            "feature_cache": str(args.feature_cache.resolve()),
            "feature_cache_audit": cache_audit,
            "primary_results": str(args.results_dir.resolve()),
        },
        "views": views,
        "full_truncate90_overlap": overlap_audit(
            memberships["full"], memberships["truncate90"], failures, tasks
        ),
        "sensitivity_protocol": {
            "min_cluster_size_grid": MIN_CLUSTER_GRID,
            "min_samples_grid": MIN_SAMPLES_GRID,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "candidate_selection": "highest_mean_raw_peer_rank_curve",
            "candidate_selection_uses_outcome": False,
        },
        "sensitivity_cross_view": cross_sensitivity,
        "judgment": {
            "high_peer_deviation": True,
            "present_in_all_five_observed_tasks": True,
            "task_initial_state_conditioned_failure_enrichment": True,
            "failure_enrichment_independent_of_exact_episode_length": False,
            "unseen_task_generalization_tested": False,
            "verdict": "qualified_descriptive_island_not_independent_cross_task_failure_block",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    with (args.out_dir / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    fields = sorted({key for row in sensitivity_rows for key in row})
    with (args.out_dir / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
