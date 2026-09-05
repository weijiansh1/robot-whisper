#!/usr/bin/env python3
"""Evaluate locked train-free routing-transfer hypotheses on matched controls."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
PROFILE_ROOT = BUNDLE / "results/transfer_profiles"
BASELINE_ROOT = PROJECT / "moe-graph-homeostasis-v10-0905/results/structural_profiles"
EVENT_ROOT = PROJECT / "analysis_moe_phenotype/events"
OUTPUT = BUNDLE / "results/evaluation"
LEADS = (-8, -6, -4, -2, 0)
LOOP_BLIND_TASKS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "open_the_middle_drawer_of_the_cabinet",
}

# These axes and directions are mechanism-locked before looking at v11 outcomes.
FIXED_DEFINITIONS = {
    "loop_conditional_overcorrection": {
        "event": "loop",
        "axes": (
            ("flow|action_conditional|gap_curvature", 1),
            ("query|action_conditional|gap_acceleration", 1),
            ("query|action_partial|gap_return_advantage", 1),
            (
                "query_scalar|flow|action_conditional|terminal_alignment|acceleration",
                1,
            ),
        ),
    },
    "loop_transfer_overcorrection": {
        "event": "loop",
        "axes": (
            ("flow|action_shape|gap_curvature", 1),
            ("flow|action_shape|velocity_opposition_fraction", 1),
            ("query|action_shape|gap_acceleration", 1),
            ("query|action_shape|gap_return_advantage", 1),
        ),
    },
    "loop_edge_structure_handoff": {
        "event": "loop",
        "axes": (
            ("channel|front_edge_to_back_action_shape|roughness_difference", 1),
            (
                "query_channel|front_edge_to_back_action_shape|target_source_acceleration_log_ratio",
                1,
            ),
            (
                "query_scalar|channel|front_edge_to_back_action_shape|handoff_delay|acceleration",
                1,
            ),
            (
                "query_scalar|channel|front_edge_to_back_action_shape|speed_correlation|acceleration",
                1,
            ),
        ),
    },
    "static_transfer_arrest": {
        "event": "static",
        "axes": (
            ("query|action_shape|back_lag4", -1),
            ("query|action_shape|gap_lag4", -1),
            ("query_channel|front_edge_to_back_action_shape|target_lag4", -1),
            ("query_scalar|flow|action_shape|terminal_alignment|lag4", -1),
        ),
    },
    "static_conditional_arrest": {
        "event": "static",
        "axes": (
            ("query|action_conditional|back_lag4", -1),
            ("query|action_conditional|gap_lag4", -1),
            ("query|action_partial|back_lag4", -1),
            (
                "query_channel|front_state_to_back_action_conditional|target_lag4",
                -1,
            ),
        ),
    },
    "static_conditional_decoupling": {
        "event": "static",
        "axes": (
            ("query|action_conditional|lag4_log_ratio", -1),
            ("query|action_partial|lag4_log_ratio", -1),
            (
                "query_channel|front_state_to_back_action_conditional|target_source_lag4_log_ratio",
                -1,
            ),
            (
                "query_scalar|flow|action_conditional|terminal_alignment|lag4",
                -1,
            ),
        ),
    },
    "static_structure_decoupling": {
        "event": "static",
        "axes": (
            ("query|action_relation|lag4_log_ratio", -1),
            ("query|action_shape|lag4_log_ratio", -1),
            ("query|state_shape|lag4_log_ratio", -1),
            (
                "query_channel|front_edge_to_back_action_shape|target_source_lag4_log_ratio",
                -1,
            ),
        ),
    },
}
BASELINES = {
    "loop": ("flow|all|edge_action|curvature", 1),
    "static": ("query|back|edge_state|lag4", -1),
}
PRIMARY_TRANSFER = {
    "loop": "loop_conditional_overcorrection",
    "static": "static_conditional_arrest",
}


@dataclass(frozen=True)
class Task:
    corpus: str
    suite: str
    task: str
    profile: Path
    baseline: Path
    events: tuple[dict[str, int | str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--profile-root", type=Path, default=PROFILE_ROOT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def parse_event(row: dict[str, str]) -> dict[str, int | str]:
    integer = {
        "episode_id",
        "scene",
        "repeat",
        "success",
        "n_queries",
        "loop_onset_q",
        "static_onset_q",
        "trap_onset_q",
    }
    return {key: int(value) if key in integer else value for key, value in row.items()}


def inventory(profile_root: Path) -> list[Task]:
    tasks = []
    for profile in sorted(profile_root.glob("*/*/*.npz")):
        corpus, suite = profile.parts[-3:-1]
        task = profile.stem
        baseline = BASELINE_ROOT / corpus / suite / profile.name
        event_path = EVENT_ROOT / corpus / suite / task / "events.csv"
        if not baseline.is_file() or not event_path.is_file():
            raise FileNotFoundError(f"missing paired input for {corpus}/{suite}/{task}")
        with event_path.open(newline="", encoding="utf-8") as handle:
            events = tuple(parse_event(row) for row in csv.DictReader(handle))
        tasks.append(Task(corpus, suite, task, profile, baseline, events))
    observed = {
        corpus: sum(task.corpus == corpus for task in tasks)
        for corpus in ("main16x32", "grid50x8")
    }
    if observed != {"main16x32": 5, "grid50x8": 40}:
        raise RuntimeError(f"profile inventory mismatch: {observed}")
    return tasks


def _load_matrix(path: Path) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive["within_feature_names"].astype(str)) + tuple(
            archive["query_feature_names"].astype(str)
        )
        matrix = np.column_stack(
            (
                np.asarray(archive["within_features"], dtype=np.float32),
                np.asarray(archive["query_features"], dtype=np.float32),
            )
        )
        metadata = {
            name: np.asarray(archive[name])
            for name in ("episode_id", "query", "scene", "repeat")
        }
    return names, matrix, metadata


def load_task(
    task: Task,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    names, matrix, metadata = _load_matrix(task.profile)
    baseline_names, baseline_matrix, baseline_metadata = _load_matrix(task.baseline)
    for key in metadata:
        if not np.array_equal(metadata[key], baseline_metadata[key]):
            raise ValueError(f"v10/v11 row mismatch for {task.task}: {key}")
    return names, matrix, baseline_names, baseline_matrix, metadata


def event_onset(task: Task, event: dict[str, int | str], event_type: str) -> int:
    if event_type == "loop" and task.task in LOOP_BLIND_TASKS:
        return -1
    return int(event[f"{event_type}_onset_q"])


def matched_pairs(
    task: Task,
    metadata: dict[str, np.ndarray],
    event_type: str,
    lead: int,
) -> list[tuple[int, np.ndarray]]:
    if event_type == "loop" and task.task in LOOP_BLIND_TASKS:
        return []
    episode_id = metadata["episode_id"]
    query = metadata["query"]
    lookup = {
        (int(episode), int(q)): row
        for row, (episode, q) in enumerate(zip(episode_id, query))
    }
    controls: dict[tuple[int, int], list[int]] = {}
    for event in task.events:
        if int(event["loop_onset_q"]) >= 0 or int(event["static_onset_q"]) >= 0:
            continue
        episode = int(event["episode_id"])
        scene = int(event["scene"])
        for q in query[episode_id == episode]:
            controls.setdefault((scene, int(q)), []).append(lookup[(episode, int(q))])
    pairs = []
    for event in task.events:
        onset = event_onset(task, event, event_type)
        target = onset + lead
        if onset < 0 or target < 0:
            continue
        positive = lookup.get((int(event["episode_id"]), target))
        negative = controls.get((int(event["scene"]), target), ())
        if positive is not None and negative:
            pairs.append((positive, np.asarray(negative, dtype=np.int64)))
    return pairs


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)].astype(np.float64))
    output = np.full(np.shape(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if len(reference):
        rank = np.searchsorted(reference, np.asarray(values)[valid], side="right")
        output[valid] = (rank + 0.5) / (len(reference) + 1.0)
    return output


def score_definition(
    definition: dict[str, Any],
    matrix: np.ndarray,
    indices: dict[str, int],
    references: dict[str, np.ndarray],
) -> np.ndarray:
    columns = []
    for axis, direction in definition["axes"]:
        percentile = empirical_percentile(references[axis], matrix[:, indices[axis]])
        columns.append(percentile if direction > 0 else 1.0 - percentile)
    stack = np.column_stack(columns)
    score = np.full(len(matrix), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(stack), axis=1)
    score[valid] = stack[valid].mean(axis=1)
    return score


def event_auc_vector(matrix: np.ndarray, pairs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    values = []
    for positive, negatives in pairs:
        positive_value = matrix[positive]
        negative_value = matrix[negatives]
        valid = np.isfinite(negative_value) & np.isfinite(positive_value)[None, :]
        wins = np.sum(
            (positive_value[None, :] > negative_value) & valid, axis=0
        ).astype(np.float64)
        wins += 0.5 * np.sum((positive_value[None, :] == negative_value) & valid, axis=0)
        count = valid.sum(axis=0)
        event_auc = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        np.divide(wins, count, out=event_auc, where=count > 0)
        values.append(event_auc)
    return np.vstack(values) if values else np.empty((0, matrix.shape[1]))


def scalar_event_auc(score: np.ndarray, pairs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    values = []
    for positive, negatives in pairs:
        valid = np.isfinite(score[negatives]) & np.isfinite(score[positive])
        if not valid.any():
            continue
        negative = score[negatives][valid]
        values.append(float(np.mean(score[positive] > negative) + 0.5 * np.mean(score[positive] == negative)))
    return np.asarray(values, dtype=np.float64)


def bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_auc(
    destination: dict[tuple[str, str, int], list[np.ndarray]],
    key: tuple[str, str, int],
    value: np.ndarray,
) -> None:
    if len(value):
        destination.setdefault(key, []).append(value)


def _summarize_scalar(
    values: dict[tuple[str, str, int], list[np.ndarray]],
    draws: int,
    label: str,
) -> list[dict[str, Any]]:
    rows = []
    for ordinal, (key, parts) in enumerate(sorted(values.items())):
        corpus, name, lead = key
        event_auc = np.concatenate(parts)
        low, high = bootstrap_mean_ci(event_auc, draws, 1913 + ordinal)
        rows.append(
            {
                "corpus": corpus,
                "event": FIXED_DEFINITIONS.get(name, {}).get("event", name.split("_", 1)[0]),
                "lead": lead,
                "score": name,
                "score_type": label,
                "event_mean_auc": float(np.mean(event_auc)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "events": len(event_auc),
            }
        )
    return rows


def plot_primary(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row
        for row in rows
        if row["lead"] == -2
        and row["score"]
        in {
            "loop_baseline",
            PRIMARY_TRANSFER["loop"],
            "loop_equal_fusion",
            "static_baseline",
            PRIMARY_TRANSFER["static"],
            "static_equal_fusion",
        }
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    colors = {"main16x32": "#315f8c", "grid50x8": "#c14f3d"}
    for panel, event in zip(axes, ("loop", "static")):
        names = [f"{event}_baseline", PRIMARY_TRANSFER[event], f"{event}_equal_fusion"]
        x = np.arange(len(names), dtype=np.float64)
        for offset, corpus in ((-0.18, "main16x32"), (0.18, "grid50x8")):
            values = []
            lows = []
            highs = []
            for name in names:
                row = next(
                    item for item in selected if item["corpus"] == corpus and item["score"] == name
                )
                values.append(row["event_mean_auc"])
                lows.append(row["bootstrap_95_low"])
                highs.append(row["bootstrap_95_high"])
            values_array = np.asarray(values)
            panel.bar(x + offset, values_array, width=0.34, color=colors[corpus], label=corpus)
            panel.errorbar(
                x + offset,
                values_array,
                yerr=np.vstack((values_array - lows, np.asarray(highs) - values_array)),
                fmt="none",
                color="#202020",
                lw=0.8,
                capsize=2,
            )
        panel.axhline(0.5, color="#777777", lw=0.8, ls="--")
        panel.set_xticks(x, ("v10 baseline", "transfer", "equal fusion"))
        panel.set_title(f"{event} at q-2")
        panel.set_ylim(0.4, 1.0)
        panel.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("event-mean matched AUC")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "transfer_increment.png", dpi=180)
    fig.savefig(output / "transfer_increment.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = inventory(args.profile_root)
    canonical_names = None
    canonical_baseline_names = None
    reference_parts: list[np.ndarray] = []
    baseline_reference_parts: dict[str, list[np.ndarray]] = {event: [] for event in BASELINES}
    axis_event_auc: dict[tuple[str, str, int], list[np.ndarray]] = {}

    # Pass one: audit every axis and build outcome-free main-corpus references.
    for task in tasks:
        names, matrix, baseline_names, baseline_matrix, metadata = load_task(task)
        if canonical_names is None:
            canonical_names = names
            canonical_baseline_names = baseline_names
        if names != canonical_names or baseline_names != canonical_baseline_names:
            raise ValueError("feature schema mismatch")
        if task.corpus == "main16x32":
            reference_parts.append(matrix)
            baseline_index = {name: index for index, name in enumerate(baseline_names)}
            for event, (axis, _) in BASELINES.items():
                baseline_reference_parts[event].append(baseline_matrix[:, baseline_index[axis]])
        for event in ("loop", "static"):
            for lead in LEADS:
                value = event_auc_vector(matrix, matched_pairs(task, metadata, event, lead))
                _append_auc(axis_event_auc, (task.corpus, event, lead), value)

    assert canonical_names is not None and canonical_baseline_names is not None
    reference_matrix = np.vstack(reference_parts)
    references = {
        name: reference_matrix[:, index] for index, name in enumerate(canonical_names)
    }
    baseline_references = {
        event: np.concatenate(parts) for event, parts in baseline_reference_parts.items()
    }

    axis_rows = []
    for key, parts in sorted(axis_event_auc.items()):
        corpus, event, lead = key
        values = np.vstack(parts)
        for index, axis in enumerate(canonical_names):
            finite = values[:, index][np.isfinite(values[:, index])]
            axis_rows.append(
                {
                    "corpus": corpus,
                    "event": event,
                    "lead": lead,
                    "axis": axis,
                    "event_mean_auc_high": float(np.mean(finite)) if len(finite) else float("nan"),
                    "events": len(finite),
                }
            )
    write_csv(args.output / "all_axis_matched_auc.csv", axis_rows)

    lookup = {
        (row["corpus"], row["event"], row["lead"], row["axis"]): row
        for row in axis_rows
    }
    replicated = {}
    for event in ("loop", "static"):
        candidates = []
        for row in axis_rows:
            if row["corpus"] != "main16x32" or row["event"] != event or row["lead"] != -2:
                continue
            if row["events"] < 50 or not np.isfinite(row["event_mean_auc_high"]):
                continue
            external = lookup[("grid50x8", event, -2, row["axis"])]
            direction = 1 if row["event_mean_auc_high"] >= 0.5 else -1
            main_auc = max(row["event_mean_auc_high"], 1.0 - row["event_mean_auc_high"])
            grid_auc = (
                external["event_mean_auc_high"]
                if direction > 0
                else 1.0 - external["event_mean_auc_high"]
            )
            candidates.append(
                {
                    "axis": row["axis"],
                    "direction": direction,
                    "main_auc": main_auc,
                    "grid_locked_auc": grid_auc,
                    "main_events": row["events"],
                    "grid_events": external["events"],
                    "replicated_advantage": min(main_auc, grid_auc) - 0.5,
                }
            )
        replicated[event] = sorted(
            candidates, key=lambda item: item["replicated_advantage"], reverse=True
        )[:30]

    composite_auc: dict[tuple[str, str, int], list[np.ndarray]] = {}
    complement_rows = []
    correlations: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for task in tasks:
        names, matrix, baseline_names, baseline_matrix, metadata = load_task(task)
        indices = {name: index for index, name in enumerate(names)}
        baseline_indices = {name: index for index, name in enumerate(baseline_names)}
        transfer_scores = {
            name: score_definition(definition, matrix, indices, references)
            for name, definition in FIXED_DEFINITIONS.items()
        }
        for event, (axis, direction) in BASELINES.items():
            raw = baseline_matrix[:, baseline_indices[axis]]
            percentile = empirical_percentile(baseline_references[event], raw)
            baseline_score = percentile if direction > 0 else 1.0 - percentile
            transfer_name = PRIMARY_TRANSFER[event]
            transfer_score = transfer_scores[transfer_name]
            fusion_score = 0.5 * (baseline_score + transfer_score)
            score_map = {
                f"{event}_baseline": baseline_score,
                transfer_name: transfer_score,
                f"{event}_equal_fusion": fusion_score,
            }
            correlations.setdefault((task.corpus, event), []).append(
                (baseline_score, transfer_score)
            )
            for lead in LEADS:
                pairs = matched_pairs(task, metadata, event, lead)
                for name, score in score_map.items():
                    _append_auc(
                        composite_auc,
                        (task.corpus, name, lead),
                        scalar_event_auc(score, pairs),
                    )
                if lead == -2:
                    for positive, negatives in pairs:
                        valid = (
                            np.isfinite(baseline_score[positive])
                            & np.isfinite(transfer_score[positive])
                            & np.isfinite(baseline_score[negatives])
                            & np.isfinite(transfer_score[negatives])
                        )
                        for negative in negatives[valid]:
                            base_margin = baseline_score[positive] - baseline_score[negative]
                            transfer_margin = transfer_score[positive] - transfer_score[negative]
                            fusion_margin = fusion_score[positive] - fusion_score[negative]
                            complement_rows.append(
                                {
                                    "corpus": task.corpus,
                                    "event": event,
                                    "baseline_correct": int(base_margin > 0.0),
                                    "transfer_correct": int(transfer_margin > 0.0),
                                    "fusion_correct": int(fusion_margin > 0.0),
                                    "transfer_rescues_baseline": int(
                                        base_margin <= 0.0 and fusion_margin > 0.0
                                    ),
                                    "transfer_breaks_baseline": int(
                                        base_margin > 0.0 and fusion_margin <= 0.0
                                    ),
                                }
                            )
        for name, definition in FIXED_DEFINITIONS.items():
            event = definition["event"]
            if name == PRIMARY_TRANSFER[event]:
                continue
            for lead in LEADS:
                _append_auc(
                    composite_auc,
                    (task.corpus, name, lead),
                    scalar_event_auc(
                        transfer_scores[name], matched_pairs(task, metadata, event, lead)
                    ),
                )

    composite_rows = _summarize_scalar(composite_auc, args.bootstrap, "fixed_train_free")
    write_csv(args.output / "fixed_composite_matched_auc.csv", composite_rows)

    complement_summary = []
    for corpus in ("main16x32", "grid50x8"):
        for event in ("loop", "static"):
            selected = [
                row for row in complement_rows if row["corpus"] == corpus and row["event"] == event
            ]
            complement_summary.append(
                {
                    "corpus": corpus,
                    "event": event,
                    "pairs": len(selected),
                    "baseline_correct_fraction": float(np.mean([row["baseline_correct"] for row in selected])),
                    "transfer_correct_fraction": float(np.mean([row["transfer_correct"] for row in selected])),
                    "fusion_correct_fraction": float(np.mean([row["fusion_correct"] for row in selected])),
                    "transfer_rescues_baseline_fraction": float(
                        np.mean([row["transfer_rescues_baseline"] for row in selected])
                    ),
                    "transfer_breaks_baseline_fraction": float(
                        np.mean([row["transfer_breaks_baseline"] for row in selected])
                    ),
                }
            )
    write_csv(args.output / "fusion_pair_audit.csv", complement_summary)

    correlation_rows = []
    for (corpus, event), parts in sorted(correlations.items()):
        baseline = np.concatenate([part[0] for part in parts])
        transfer = np.concatenate([part[1] for part in parts])
        valid = np.isfinite(baseline) & np.isfinite(transfer)
        correlation, p_value = spearmanr(baseline[valid], transfer[valid])
        correlation_rows.append(
            {
                "corpus": corpus,
                "event": event,
                "rows": int(valid.sum()),
                "spearman": float(correlation),
                "p_value": float(p_value),
            }
        )
    write_csv(args.output / "baseline_transfer_correlation.csv", correlation_rows)
    plot_primary(composite_rows, args.output)

    primary_q2 = [
        row
        for row in composite_rows
        if row["lead"] == -2
        and row["score"]
        in {
            "loop_baseline",
            PRIMARY_TRANSFER["loop"],
            "loop_equal_fusion",
            "static_baseline",
            PRIMARY_TRANSFER["static"],
            "static_equal_fusion",
        }
    ]
    summary = {
        "schema": "himoe.routing_transfer_evaluation.v2",
        "tasks": len(tasks),
        "profile_root": str(args.profile_root),
        "axes": len(canonical_names),
        "leads": list(LEADS),
        "outcome_weight_training": False,
        "feature_selection_training": False,
        "percentile_reference": "all main16x32 query rows; outcomes not used",
        "matched_control": "same task, scene, and absolute query; neither loop nor static",
        "fixed_definitions": FIXED_DEFINITIONS,
        "baseline_definitions": BASELINES,
        "primary_q_minus_2": primary_q2,
        "fusion_pair_audit_q_minus_2": complement_summary,
        "baseline_transfer_correlation": correlation_rows,
        "post_hoc_replicated_axis_audit": replicated,
        "caveat": (
            "Fixed composites test the transfer hypothesis. The full-axis audit is descriptive and "
            "is not evidence without confirmation or multiplicity control."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "primary_q_minus_2": primary_q2,
                "fusion_pair_audit_q_minus_2": complement_summary,
                "top_replicated_axes": {event: rows[:5] for event, rows in replicated.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
