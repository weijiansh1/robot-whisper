#!/usr/bin/env python3
"""Seed robustness and positive-control power audit for the residual-failure
subtype negative result.

The claim under test is that HDBSCAN over nine routing representations of the
89 residual long-task failures yields mutually inconsistent partitions
(pairwise ARI median 0.018).  A negative result of that shape is only
informative if the same pipeline WOULD have recovered a real cluster structure
of comparable sample size and dimensionality.  This script therefore runs:

1. seed sweep of the exact view-level clustering step,
2. a real positive control (long-task successes vs. 3-vote consensus core),
3. a graded synthetic positive control that interpolates the real residual
   feature vectors toward two real routing anchors (no Gaussian noise),
4. the same pairwise-ARI statistic recomputed under the planted structure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import HDBSCAN

from analyze_residual_failure_subtypes import (
    ANALYSIS,
    LONG_TASK,
    consensus_votes,
    density_labels,
    label_summary,
    load_tables,
    load_views,
    robust_pca,
)
from threadpoolctl import threadpool_limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ANALYSIS / "residual-failure-subtypes-repro",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260829, 11, 20250101, 987654321],
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60],
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def cohort():
    aligned, events, alternative = load_tables()
    _flags, votes = consensus_votes(aligned, events, alternative)
    failure = aligned["outcome"].to_numpy() == "failure"
    task = aligned["task"].to_numpy()
    long_mask = task == LONG_TASK
    residual = failure & (votes < 2) & long_mask
    core = failure & (votes >= 2) & long_mask
    success = (~failure) & long_mask
    return {
        "residual": np.flatnonzero(residual),
        "core": np.flatnonzero(core),
        "success": np.flatnonzero(success),
        "aligned": aligned,
        "votes": votes,
    }


def pairwise_ari(labels: dict[str, np.ndarray]) -> dict[str, float]:
    viable = [name for name, value in labels.items() if len(set(value) - {-1}) >= 2]
    pairs = []
    for left, right in combinations(viable, 2):
        pairs.append(
            {
                "left": left,
                "right": right,
                "ari": float(adjusted_rand_score(labels[left], labels[right])),
            }
        )
    values = [item["ari"] for item in pairs]
    independent = [
        item["ari"]
        for item in pairs
        if {item["left"], item["right"]} != {"aligned_raw", "aligned_task_init"}
    ]
    return {
        "viable_views": viable,
        "n_pairs": len(pairs),
        "median": float(np.median(values)) if values else float("nan"),
        "maximum": float(np.max(values)) if values else float("nan"),
        "maximum_excluding_aligned_variants": (
            float(np.max(independent)) if independent else float("nan")
        ),
        "pairs": pairs,
    }


def seed_sweep(views, index, seeds):
    rows = []
    for seed in seeds:
        labels = {
            name: density_labels(values[index], seed + offset)
            for offset, (name, values) in enumerate(views.items())
        }
        summary = {name: label_summary(value) for name, value in labels.items()}
        stats = pairwise_ari(labels)
        rows.append(
            {
                "seed": int(seed),
                "clusters_per_view": {
                    name: item["clusters"] for name, item in summary.items()
                },
                "coverage_per_view": {
                    name: round(item["coverage"], 4) for name, item in summary.items()
                },
                "all_noise_views": [
                    name for name, item in summary.items() if item["clusters"] == 0
                ],
                "pairwise_ari_median": stats["median"],
                "pairwise_ari_maximum": stats["maximum"],
                "pairwise_ari_maximum_excluding_aligned_variants": stats[
                    "maximum_excluding_aligned_variants"
                ],
                "viable_views": stats["viable_views"],
            }
        )
    return rows


def real_positive_control(views, groups, replicates, seed):
    """Can the identical pipeline recover a real, known two-group routing split
    at exactly n=89 within one task?"""
    per_view = {name: {"ari": [], "clusters": []} for name in views}
    cross_median, cross_max, cross_max_excl = [], [], []
    for draw in range(replicates):
        rng = np.random.default_rng(seed + draw)
        first = rng.choice(groups["success"], 45, replace=False)
        second = rng.choice(groups["core"], 44, replace=False)
        index = np.r_[first, second]
        truth = np.r_[np.zeros(45, int), np.ones(44, int)]
        draw_labels = {}
        for name, values in views.items():
            labels = density_labels(values[index], seed + draw)
            draw_labels[name] = labels
            per_view[name]["ari"].append(float(adjusted_rand_score(truth, labels)))
            per_view[name]["clusters"].append(int(len(set(labels) - {-1})))
        stats = pairwise_ari(draw_labels)
        cross_median.append(stats["median"])
        cross_max.append(stats["maximum"])
        cross_max_excl.append(stats["maximum_excluding_aligned_variants"])
    out = {
        name: {
            "ari_median": float(np.median(item["ari"])),
            "ari_min": float(np.min(item["ari"])),
            "ari_max": float(np.max(item["ari"])),
            "cluster_count_median": float(np.median(item["clusters"])),
        }
        for name, item in per_view.items()
    }
    return out, {
        "cross_view_pairwise_ari_median": float(np.median(cross_median)),
        "cross_view_pairwise_ari_maximum": float(np.median(cross_max)),
        "cross_view_pairwise_ari_max_excluding_aligned": float(
            np.median(cross_max_excl)
        ),
    }


def observed_statistic_robustness(views, index, draws, seed):
    """The 0.018 headline is a single number.  Perturb the cohort (80%
    subsample) and the density hyper-parameters to see how wide it is."""
    rng = np.random.default_rng(seed)
    size = int(np.ceil(0.8 * len(index)))
    subsample = {"median": [], "max_excl": [], "n_noise_views": []}
    for draw in range(draws):
        pick = np.sort(rng.choice(len(index), size, replace=False))
        labels = {
            name: density_labels(values[index][pick], seed + draw)
            for name, values in views.items()
        }
        stats = pairwise_ari(labels)
        subsample["median"].append(stats["median"])
        subsample["max_excl"].append(stats["maximum_excluding_aligned_variants"])
        subsample["n_noise_views"].append(
            sum(1 for value in labels.values() if len(set(value) - {-1}) == 0)
        )
    grid = []
    for min_cluster_size in (5, 8, 12, 15):
        for min_samples in (3, 4, 6):
            labels = {}
            for name, values in views.items():
                embedded = robust_pca(values[index].astype(np.float64), seed)
                labels[name] = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    copy=True,
                ).fit_predict(embedded)
            stats = pairwise_ari(labels)
            grid.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "n_noise_views": sum(
                        1 for value in labels.values() if len(set(value) - {-1}) == 0
                    ),
                    "pairwise_ari_median": stats["median"],
                    "pairwise_ari_max_excluding_aligned": stats[
                        "maximum_excluding_aligned_variants"
                    ],
                }
            )
    return {
        "subsample80": {
            "draws": draws,
            "median_of_medians": float(np.median(subsample["median"])),
            "median_q10": float(np.quantile(subsample["median"], 0.10)),
            "median_q90": float(np.quantile(subsample["median"], 0.90)),
            "max_excl_median": float(np.median(subsample["max_excl"])),
            "max_excl_q90": float(np.quantile(subsample["max_excl"], 0.90)),
            "n_noise_views_median": float(np.median(subsample["n_noise_views"])),
        },
        "hdbscan_grid": grid,
        "hdbscan_grid_median_range": [
            float(np.min([row["pairwise_ari_median"] for row in grid])),
            float(np.max([row["pairwise_ari_median"] for row in grid])),
        ],
    }


def standardized_separation(values, planted, seed):
    """Between-centroid distance divided by the pooled within-group RMS radius,
    measured in the same RobustScaler+PCA embedding the clusterer sees."""
    embedded = robust_pca(np.asarray(values, dtype=np.float64), seed)
    first = embedded[planted == 0]
    second = embedded[planted == 1]
    centre0 = first.mean(axis=0)
    centre1 = second.mean(axis=0)
    within = np.sqrt(
        (
            np.sum((first - centre0) ** 2) + np.sum((second - centre1) ** 2)
        )
        / (len(first) + len(second))
    )
    if within <= 1e-12:
        return float("inf")
    return float(np.linalg.norm(centre1 - centre0) / within)


def separation_scale(values, index, groups):
    """Report the real success-vs-core routing gap in robust units so the
    interpolation coefficient alpha has an interpretable meaning."""
    residual = values[index]
    spread = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    spread = np.where(spread > 1e-12, spread, np.nan)
    gap = np.median(values[groups["success"]], axis=0) - np.median(
        values[groups["core"]], axis=0
    )
    ratio = np.abs(gap) / spread
    return float(np.nanmedian(ratio))


def synthetic_positive_control(views, index, groups, alphas, replicates, seed, sizes):
    """Interpolate real residual feature vectors toward two REAL routing
    anchors (long-task success median, long-task consensus-core median).

    x'_i = (1 - alpha) * x_i + alpha * anchor_{g(i)}

    No Gaussian noise is added; every planted centroid is a convex combination
    of measured routing features, so the planted clusters lie on the observed
    routing manifold.
    """
    results = []
    for size_label, first_size in sizes:
        for alpha in alphas:
            per_view_labels_by_draw = []
            per_view_ari = {name: [] for name in views}
            per_view_clusters = {name: [] for name in views}
            separations: dict[str, list[float]] = {name: [] for name in views}
            for draw in range(replicates):
                rng = np.random.default_rng(seed + 1000 * draw + int(alpha * 1000))
                planted = np.zeros(len(index), dtype=int)
                planted[
                    rng.choice(len(index), first_size, replace=False)
                ] = 1
                draw_labels = {}
                for offset, (name, values) in enumerate(views.items()):
                    base = values[index].astype(np.float64)
                    anchor0 = np.median(values[groups["success"]], axis=0)
                    anchor1 = np.median(values[groups["core"]], axis=0)
                    anchors = np.where(planted[:, None] == 1, anchor1, anchor0)
                    injected = (1.0 - alpha) * base + alpha * anchors
                    labels = density_labels(injected, seed + offset + draw)
                    draw_labels[name] = labels
                    separations[name].append(
                        standardized_separation(injected, planted, seed)
                    )
                    per_view_ari[name].append(
                        float(adjusted_rand_score(planted, labels))
                    )
                    per_view_clusters[name].append(int(len(set(labels) - {-1})))
                per_view_labels_by_draw.append((planted, draw_labels))
            cross = [pairwise_ari(item[1]) for item in per_view_labels_by_draw]
            results.append(
                {
                    "planted_sizes": size_label,
                    "alpha": float(alpha),
                    "recovery_ari_median_by_view": {
                        name: float(np.median(value))
                        for name, value in per_view_ari.items()
                    },
                    "cluster_count_median_by_view": {
                        name: float(np.median(value))
                        for name, value in per_view_clusters.items()
                    },
                    "views_recovering_ari_ge_0.5": [
                        name
                        for name, value in per_view_ari.items()
                        if float(np.median(value)) >= 0.5
                    ],
                    "planted_separation_by_view": {
                        name: float(np.median(value))
                        for name, value in separations.items()
                    },
                    "cross_view_pairwise_ari_median": float(
                        np.median([item["median"] for item in cross])
                    ),
                    "cross_view_pairwise_ari_max_excluding_aligned": float(
                        np.median(
                            [item["maximum_excluding_aligned_variants"] for item in cross]
                        )
                    ),
                }
            )
    return results


def self_test() -> None:
    rng = np.random.default_rng(0)
    good = np.r_[rng.normal(-3, 0.1, (20, 4)), rng.normal(3, 0.1, (20, 4))]
    labels = density_labels(good, 0)
    truth = np.r_[np.zeros(20, int), np.ones(20, int)]
    assert adjusted_rand_score(truth, labels) > 0.9
    stats = pairwise_ari({"a": truth, "b": truth, "c": np.zeros(40, int)})
    assert stats["median"] == 1.0
    print("self-test passed")


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = cohort()
    index = groups["residual"]
    assert len(index) == 89, len(index)
    views, _truncated = load_views()

    sweep = seed_sweep(views, index, args.seeds)
    robustness = observed_statistic_robustness(views, index, args.replicates * 3, 5150)
    real, real_cross = real_positive_control(views, groups, args.replicates, 4242)
    scales = {
        name: separation_scale(values, index, groups) for name, values in views.items()
    }
    synthetic = synthetic_positive_control(
        views,
        index,
        groups,
        args.alphas,
        args.replicates,
        777,
        [("balanced_45_44", 44), ("imbalanced_20_69", 20)],
    )

    summary = {
        "cohort": {
            "residual_long_failures": int(len(index)),
            "long_consensus_core": int(len(groups["core"])),
            "long_successes": int(len(groups["success"])),
        },
        "seed_sweep": sweep,
        "seed_sweep_summary": {
            "pairwise_ari_median_across_seeds": [
                round(row["pairwise_ari_median"], 4) for row in sweep
            ],
            "max_excluding_aligned_across_seeds": [
                round(row["pairwise_ari_maximum_excluding_aligned_variants"], 4)
                for row in sweep
            ],
            "all_noise_view_counts": [len(row["all_noise_views"]) for row in sweep],
        },
        "observed_statistic_robustness": robustness,
        "real_positive_control": {
            "design": "45 real long-task successes + 44 real long-task 3/2-vote core failures, same n=89, same pipeline",
            "by_view": real,
            "cross_view_statistic_under_real_two_group_truth": real_cross,
        },
        "anchor_gap_in_robust_units": scales,
        "synthetic_positive_control": {
            "design": "x' = (1-alpha)*x + alpha*anchor_g, anchors are real long-task success / consensus-core medians",
            "results": synthetic,
        },
    }
    path = args.out_dir / "seed_and_power_audit.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path}")

    print("\n== seed sweep ==")
    print(
        pd.DataFrame(
            [
                {
                    "seed": row["seed"],
                    "pairwise_ari_median": round(row["pairwise_ari_median"], 4),
                    "max_excl_aligned": round(
                        row["pairwise_ari_maximum_excluding_aligned_variants"], 4
                    ),
                    "n_all_noise_views": len(row["all_noise_views"]),
                    "viable_views": len(row["viable_views"]),
                }
                for row in sweep
            ]
        ).to_string(index=False)
    )
    print("\n== observed-statistic robustness (80% subsample / hdbscan grid) ==")
    print(json.dumps(robustness["subsample80"], indent=1))
    print(
        pd.DataFrame(robustness["hdbscan_grid"]).round(4).to_string(index=False)
    )
    print("\n== real positive control (success vs core, n=89) ==")
    print(
        pd.DataFrame(real)
        .T[["ari_median", "ari_min", "ari_max", "cluster_count_median"]]
        .round(3)
        .to_string()
    )
    print("cross-view statistic under REAL two-group truth:", real_cross)
    print("\n== synthetic positive control ==")
    print(
        pd.DataFrame(
            [
                {
                    "sizes": row["planted_sizes"],
                    "alpha": row["alpha"],
                    "n_views_ari>=0.5": len(row["views_recovering_ari_ge_0.5"]),
                    "xview_med": round(row["cross_view_pairwise_ari_median"], 4),
                    "xview_max_excl": round(
                        row["cross_view_pairwise_ari_max_excluding_aligned"], 4
                    ),
                    "sep_alignedraw": round(
                        row["planted_separation_by_view"]["aligned_raw"], 2
                    ),
                    "sep_event": round(row["planted_separation_by_view"]["event"], 2),
                    **{
                        f"ari_{k}": round(v, 3)
                        for k, v in row["recovery_ari_median_by_view"].items()
                    },
                }
                for row in synthetic
            ]
        ).to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
