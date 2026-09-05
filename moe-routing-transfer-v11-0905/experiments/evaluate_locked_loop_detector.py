#!/usr/bin/env python3
"""Evaluate the main-corpus-selected conditional loop axis on the held-out corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_transfer import (
    BASELINES,
    LEADS,
    bootstrap_mean_ci,
    empirical_percentile,
    inventory,
    load_task,
    matched_pairs,
    scalar_event_auc,
)


BUNDLE = Path(__file__).resolve().parent.parent
PROFILE_ROOT = BUNDLE / "results/conditional_profiles"
OUTPUT = BUNDLE / "results/locked_loop_detector"
LOCKED_AXIS = "query|action_partial|front_acceleration"
LOCKED_DIRECTION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_weighted_auc(score: np.ndarray, pairs: list[tuple[int, np.ndarray]]) -> tuple[float, int]:
    wins = 0.0
    count = 0
    for positive, negatives in pairs:
        valid = np.isfinite(score[negatives]) & np.isfinite(score[positive])
        negative = score[negatives][valid]
        wins += float(np.sum(score[positive] > negative))
        wins += 0.5 * float(np.sum(score[positive] == negative))
        count += len(negative)
    return (wins / count if count else float("nan")), count


def paired_event_increment(
    baseline: np.ndarray,
    fusion: np.ndarray,
    pairs: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    difference = []
    for positive, negatives in pairs:
        valid = (
            np.isfinite(baseline[positive])
            & np.isfinite(fusion[positive])
            & np.isfinite(baseline[negatives])
            & np.isfinite(fusion[negatives])
        )
        if not valid.any():
            continue
        selected = negatives[valid]
        baseline_auc = float(
            np.mean(baseline[positive] > baseline[selected])
            + 0.5 * np.mean(baseline[positive] == baseline[selected])
        )
        fusion_auc = float(
            np.mean(fusion[positive] > fusion[selected])
            + 0.5 * np.mean(fusion[positive] == fusion[selected])
        )
        difference.append(fusion_auc - baseline_auc)
    return np.asarray(difference, dtype=np.float64)


def plot_leads(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), sharex=True, sharey=True)
    colors = {"baseline": "#6d7780", "conditional": "#2c758e", "fusion": "#c34e3c"}
    for panel, corpus in zip(axes, ("main16x32", "grid50x8")):
        for score in ("baseline", "conditional", "fusion"):
            selected = sorted(
                [row for row in rows if row["corpus"] == corpus and row["score"] == score],
                key=lambda row: row["lead"],
            )
            panel.plot(
                [row["lead"] for row in selected],
                [row["event_mean_auc"] for row in selected],
                marker="o",
                lw=1.8,
                color=colors[score],
                label=score,
            )
        panel.axhline(0.5, color="#777777", lw=0.8, ls="--")
        panel.set_title(corpus)
        panel.set_xlabel("query offset from loop onset")
        panel.set_ylim(0.40, 0.82)
        panel.grid(alpha=0.18)
    axes[0].set_ylabel("event-mean matched AUC")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "locked_loop_leads.png", dpi=180)
    fig.savefig(output / "locked_loop_leads.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = inventory(args.profile_root)
    loaded = []
    transfer_reference = []
    baseline_reference = []
    baseline_axis, baseline_direction = BASELINES["loop"]
    for task in tasks:
        names, matrix, baseline_names, baseline_matrix, metadata = load_task(task)
        loaded.append((task, names, matrix, baseline_names, baseline_matrix, metadata))
        if task.corpus == "main16x32":
            indices = {name: index for index, name in enumerate(names)}
            baseline_indices = {name: index for index, name in enumerate(baseline_names)}
            transfer_reference.append(matrix[:, indices[LOCKED_AXIS]])
            baseline_reference.append(baseline_matrix[:, baseline_indices[baseline_axis]])
    transfer_reference_array = np.concatenate(transfer_reference)
    baseline_reference_array = np.concatenate(baseline_reference)

    event_parts: dict[tuple[str, int, str], list[np.ndarray]] = {}
    pair_totals: dict[tuple[str, int, str], list[float]] = {}
    delta_parts: dict[tuple[str, int], list[np.ndarray]] = {}
    for task, names, matrix, baseline_names, baseline_matrix, metadata in loaded:
        indices = {name: index for index, name in enumerate(names)}
        baseline_indices = {name: index for index, name in enumerate(baseline_names)}
        conditional = empirical_percentile(
            transfer_reference_array, matrix[:, indices[LOCKED_AXIS]]
        )
        if LOCKED_DIRECTION < 0:
            conditional = 1.0 - conditional
        baseline = empirical_percentile(
            baseline_reference_array, baseline_matrix[:, baseline_indices[baseline_axis]]
        )
        if baseline_direction < 0:
            baseline = 1.0 - baseline
        fusion = 0.5 * (baseline + conditional)
        for lead in LEADS:
            pairs = matched_pairs(task, metadata, "loop", lead)
            scores = {"baseline": baseline, "conditional": conditional, "fusion": fusion}
            for name, score in scores.items():
                event_auc = scalar_event_auc(score, pairs)
                if len(event_auc):
                    event_parts.setdefault((task.corpus, lead, name), []).append(event_auc)
                weighted_auc, pair_count = pair_weighted_auc(score, pairs)
                target = pair_totals.setdefault((task.corpus, lead, name), [0.0, 0.0])
                if pair_count:
                    target[0] += weighted_auc * pair_count
                    target[1] += pair_count
            increment = paired_event_increment(baseline, fusion, pairs)
            if len(increment):
                delta_parts.setdefault((task.corpus, lead), []).append(increment)

    rows = []
    for ordinal, (key, parts) in enumerate(sorted(event_parts.items())):
        corpus, lead, name = key
        value = np.concatenate(parts)
        low, high = bootstrap_mean_ci(value, args.bootstrap, 6203 + ordinal)
        weighted = pair_totals[key]
        rows.append(
            {
                "corpus": corpus,
                "lead": lead,
                "score": name,
                "event_mean_auc": float(value.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "events": len(value),
                "pair_weighted_auc": float(weighted[0] / weighted[1]),
                "pairs": int(weighted[1]),
            }
        )
    write_csv(args.output / "locked_loop_auc.csv", rows)

    delta_rows = []
    for ordinal, (key, parts) in enumerate(sorted(delta_parts.items())):
        corpus, lead = key
        value = np.concatenate(parts)
        low, high = bootstrap_mean_ci(value, args.bootstrap, 9109 + ordinal)
        delta_rows.append(
            {
                "corpus": corpus,
                "lead": lead,
                "fusion_minus_baseline_event_auc": float(value.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "events": len(value),
                "fraction_events_improved": float(np.mean(value > 0.0)),
            }
        )
    write_csv(args.output / "paired_increment.csv", delta_rows)
    plot_leads(rows, args.output)

    q_minus_2 = [row for row in rows if row["lead"] == -2]
    q_minus_2_delta = [row for row in delta_rows if row["lead"] == -2]
    summary = {
        "schema": "himoe.locked_conditional_loop_detector.v1",
        "selection": (
            "The single conditional axis and its high-risk direction were selected by main16x32 "
            "q-2 event-mean AUC. grid50x8 was held out for direction-locked evaluation."
        ),
        "training": False,
        "locked_axis": LOCKED_AXIS,
        "locked_direction": LOCKED_DIRECTION,
        "baseline_axis": baseline_axis,
        "fusion": "equal mean of main-reference empirical percentiles",
        "q_minus_2": q_minus_2,
        "q_minus_2_paired_increment": q_minus_2_delta,
        "all_leads": rows,
        "caveat": (
            "The axis was selected from a 584-axis discovery scan. Held-out direction replication "
            "is meaningful, but a fresh corpus is required for an unbiased final estimate."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"q_minus_2": q_minus_2, "increment": q_minus_2_delta}, indent=2))


if __name__ == "__main__":
    main()
