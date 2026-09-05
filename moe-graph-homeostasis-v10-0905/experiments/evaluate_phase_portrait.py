#!/usr/bin/env python3
"""Evaluate fixed graph-homeostasis hypotheses with matched healthy controls."""

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


BUNDLE = Path(__file__).resolve().parent.parent
PROJECT = BUNDLE.parent
PROFILE_ROOT = BUNDLE / "results/structural_profiles"
EVENT_ROOT = PROJECT / "analysis_moe_phenotype/events"
OUTPUT = BUNDLE / "results/phase_portrait"
LEADS = (-8, -6, -4, -2, 0)
LOOP_BLIND_TASKS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "open_the_middle_drawer_of_the_cabinet",
}
DEFINITIONS = {
    "loop_flow_instability": {
        "event": "loop",
        "axes": (
            ("state|flow_nonsettling", 1),
            ("state|flow_curvature", 1),
            ("state|flow_directness", -1),
            ("state|flow_late_motion", 1),
        ),
    },
    "loop_dynamic_homeostasis": {
        "event": "loop",
        "axes": (
            ("state|flow_nonsettling", 1),
            ("state|query_acceleration", 1),
            ("state|query_jerk", 1),
            ("state|query_return", 1),
        ),
    },
    "static_rigid_consensus": {
        "event": "static",
        "axes": (
            ("state|query_response", -1),
            ("state|query_acceleration", -1),
            ("state|query_jerk", -1),
            ("state|query_turn", -1),
            ("state|token_coherence", 1),
            ("state|diversity", -1),
        ),
    },
}
LOCKED_DISCOVERY = {
    "main_ranked_loop_top4": {
        "event": "loop",
        "axes": (
            ("flow|all|edge_action|curvature", 1),
            ("flow|front|expert_load|curvature", 1),
            ("flow|front|edge_action|settling_log_ratio", 1),
            ("flow|all|expert_load|curvature", 1),
        ),
    },
    "main_ranked_static_top4": {
        "event": "static",
        "axes": (
            ("query|back|edge_state|lag4", -1),
            ("query|all|expert_load|lag4", -1),
            ("query|back|expert_load|lag4", -1),
            ("query|back|edge_action|lag4", -1),
        ),
    },
}
ALL_DEFINITIONS = {**DEFINITIONS, **LOCKED_DISCOVERY}
PHASE_AXES = (
    "state|flow_nonsettling",
    "state|query_response",
    "state|token_coherence",
)


@dataclass(frozen=True)
class Task:
    corpus: str
    suite: str
    task: str
    profile: Path
    events: tuple[dict[str, int | str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
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


def inventory() -> list[Task]:
    tasks = []
    for profile in sorted(PROFILE_ROOT.glob("*/*/*.npz")):
        corpus, suite = profile.parts[-3:-1]
        task = profile.stem
        event_path = EVENT_ROOT / corpus / suite / task / "events.csv"
        with event_path.open(newline="", encoding="utf-8") as handle:
            events = tuple(parse_event(row) for row in csv.DictReader(handle))
        tasks.append(Task(corpus, suite, task, profile, events))
    observed = {
        corpus: sum(task.corpus == corpus for task in tasks)
        for corpus in ("main16x32", "grid50x8")
    }
    if observed != {"main16x32": 5, "grid50x8": 40}:
        raise RuntimeError(f"structural profile inventory mismatch: {observed}")
    return tasks


def load_profile(task: Task) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    with np.load(task.profile, allow_pickle=False) as archive:
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


def empty_accumulator(axis_count: int) -> dict[str, np.ndarray]:
    return {
        "wins": np.zeros(axis_count, dtype=np.float64),
        "pairs": np.zeros(axis_count, dtype=np.int64),
        "events": np.zeros(axis_count, dtype=np.int64),
    }


def accumulate(accumulator: dict[str, np.ndarray], matrix: np.ndarray, pairs) -> None:
    for positive, negatives in pairs:
        positive_value = matrix[positive]
        negative_value = matrix[negatives]
        valid = np.isfinite(negative_value) & np.isfinite(positive_value)[None, :]
        accumulator["wins"] += np.sum((positive_value[None, :] > negative_value) & valid, axis=0)
        accumulator["wins"] += 0.5 * np.sum((positive_value[None, :] == negative_value) & valid, axis=0)
        accumulator["pairs"] += valid.sum(axis=0)
        accumulator["events"] += valid.any(axis=0)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)].astype(np.float64))
    output = np.full(np.shape(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
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


def scalar_auc(score: np.ndarray, pairs) -> dict[str, float | int]:
    wins = 0.0
    count = 0
    events = 0
    for positive, negatives in pairs:
        positive_value = score[positive]
        negative_value = score[negatives]
        valid = np.isfinite(negative_value) & np.isfinite(positive_value)
        if not valid.any():
            continue
        wins += np.sum(positive_value > negative_value[valid])
        wins += 0.5 * np.sum(positive_value == negative_value[valid])
        count += int(valid.sum())
        events += 1
    return {
        "auc_event_high": float(wins / count) if count else float("nan"),
        "events": events,
        "pairs": count,
    }


def bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase(rows: list[dict[str, Any]], output: Path) -> None:
    colors = {"loop": "#c6403d", "static": "#246e8a"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=True, sharey=True)
    for panel, corpus in zip(axes, ("main16x32", "grid50x8")):
        for event_type in ("loop", "static"):
            selected = [
                row for row in rows if row["corpus"] == corpus and row["event"] == event_type
            ]
            if not selected:
                continue
            control_x = np.mean([row["control_responsiveness"] for row in selected])
            control_y = np.mean([row["control_settling"] for row in selected])
            event_x = np.mean([row["event_responsiveness"] for row in selected])
            event_y = np.mean([row["event_settling"] for row in selected])
            panel.scatter(control_x, control_y, marker="o", s=55, facecolor="white", edgecolor=colors[event_type])
            panel.scatter(event_x, event_y, marker="o", s=65, color=colors[event_type], label=f"{event_type} q-2")
            panel.annotate(
                "",
                xy=(event_x, event_y),
                xytext=(control_x, control_y),
                arrowprops={"arrowstyle": "->", "color": colors[event_type], "lw": 1.6},
            )
        panel.axhline(0.5, color="#b5b5b5", lw=0.7, ls="--")
        panel.axvline(0.5, color="#b5b5b5", lw=0.7, ls="--")
        panel.set_title(corpus)
        panel.set_xlabel("cross-query responsiveness percentile")
        panel.set_xlim(0.0, 0.85)
        panel.set_ylim(0.2, 0.8)
        panel.grid(alpha=0.16)
    axes[0].set_ylabel("within-flow settling percentile")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("MoE graph homeostasis: matched event shift at q-2", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "phase_portrait.png", dpi=180)
    fig.savefig(output / "phase_portrait.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = inventory()
    canonical_names: tuple[str, ...] | None = None
    reference_axes = sorted(
        set(PHASE_AXES)
        | {axis for definition in ALL_DEFINITIONS.values() for axis, _ in definition["axes"]}
    )
    reference_parts = {axis: [] for axis in reference_axes}
    accumulators: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}

    for task in tasks:
        names, matrix, metadata = load_profile(task)
        if canonical_names is None:
            canonical_names = names
            for corpus in ("main16x32", "grid50x8"):
                for event in ("loop", "static"):
                    for lead in LEADS:
                        accumulators[(corpus, event, lead)] = empty_accumulator(len(names))
        if names != canonical_names:
            raise ValueError(f"feature schema mismatch: {task.profile}")
        indices = {name: index for index, name in enumerate(names)}
        if task.corpus == "main16x32":
            for axis in reference_axes:
                reference_parts[axis].append(matrix[:, indices[axis]])
        for event in ("loop", "static"):
            for lead in LEADS:
                pairs = matched_pairs(task, metadata, event, lead)
                accumulate(accumulators[(task.corpus, event, lead)], matrix, pairs)

    assert canonical_names is not None
    references = {
        axis: np.sort(
            np.concatenate(values)[np.isfinite(np.concatenate(values))].astype(np.float32)
        )
        for axis, values in reference_parts.items()
    }
    axis_rows = []
    for (corpus, event, lead), accumulator in accumulators.items():
        for index, axis in enumerate(canonical_names):
            pairs = int(accumulator["pairs"][index])
            axis_rows.append(
                {
                    "corpus": corpus,
                    "event": event,
                    "lead": lead,
                    "axis": axis,
                    "auc_event_high": float(accumulator["wins"][index] / pairs) if pairs else float("nan"),
                    "events": int(accumulator["events"][index]),
                    "pairs": pairs,
                }
            )
    write_csv(args.output / "all_axis_matched_auc.csv", axis_rows)

    composite_accumulator: dict[tuple[str, str, int], dict[str, float | int]] = {}
    phase_rows: list[dict[str, Any]] = []
    for task in tasks:
        names, matrix, metadata = load_profile(task)
        indices = {name: index for index, name in enumerate(names)}
        scores = {
            name: score_definition(definition, matrix, indices, references)
            for name, definition in ALL_DEFINITIONS.items()
        }
        for name, definition in ALL_DEFINITIONS.items():
            event = definition["event"]
            for lead in LEADS:
                result = scalar_auc(scores[name], matched_pairs(task, metadata, event, lead))
                key = (task.corpus, name, lead)
                target = composite_accumulator.setdefault(
                    key, {"wins": 0.0, "events": 0, "pairs": 0}
                )
                target["wins"] += float(result["auc_event_high"]) * int(result["pairs"]) if result["pairs"] else 0.0
                target["events"] += int(result["events"])
                target["pairs"] += int(result["pairs"])

        coordinate = {
            "settling": 1.0 - empirical_percentile(
                references["state|flow_nonsettling"], matrix[:, indices["state|flow_nonsettling"]]
            ),
            "responsiveness": empirical_percentile(
                references["state|query_response"], matrix[:, indices["state|query_response"]]
            ),
            "coherence": empirical_percentile(
                references["state|token_coherence"], matrix[:, indices["state|token_coherence"]]
            ),
        }
        for event in ("loop", "static"):
            for positive, negatives in matched_pairs(task, metadata, event, -2):
                values = [coordinate[name][positive] for name in coordinate]
                controls = [np.nanmean(coordinate[name][negatives]) for name in coordinate]
                if not np.all(np.isfinite(values + controls)):
                    continue
                phase_rows.append(
                    {
                        "corpus": task.corpus,
                        "suite": task.suite,
                        "task": task.task,
                        "event": event,
                        "event_settling": values[0],
                        "control_settling": controls[0],
                        "event_responsiveness": values[1],
                        "control_responsiveness": controls[1],
                        "event_coherence": values[2],
                        "control_coherence": controls[2],
                    }
                )

    composite_rows = []
    for (corpus, name, lead), value in sorted(composite_accumulator.items()):
        pairs = int(value["pairs"])
        composite_rows.append(
            {
                "corpus": corpus,
                "event": ALL_DEFINITIONS[name]["event"],
                "lead": lead,
                "score": name,
                "status": "prior_fixed" if name in DEFINITIONS else "main16x32_rank_locked",
                "auc_event_high": float(value["wins"] / pairs) if pairs else float("nan"),
                "events": int(value["events"]),
                "pairs": pairs,
            }
        )
    write_csv(args.output / "composite_matched_auc.csv", composite_rows)
    write_csv(
        args.output / "fixed_composite_matched_auc.csv",
        [row for row in composite_rows if row["status"] == "prior_fixed"],
    )
    write_csv(
        args.output / "locked_discovery_matched_auc.csv",
        [row for row in composite_rows if row["status"] == "main16x32_rank_locked"],
    )
    write_csv(args.output / "phase_matched_points.csv", phase_rows)

    phase_summary = []
    for ordinal, corpus in enumerate(("main16x32", "grid50x8")):
        for event_index, event in enumerate(("loop", "static")):
            selected = [row for row in phase_rows if row["corpus"] == corpus and row["event"] == event]
            for coordinate in ("settling", "responsiveness", "coherence"):
                event_value = np.asarray([row[f"event_{coordinate}"] for row in selected])
                control_value = np.asarray([row[f"control_{coordinate}"] for row in selected])
                difference = event_value - control_value
                low, high = bootstrap_mean_ci(
                    difference, args.bootstrap, 3109 + 100 * ordinal + 10 * event_index + len(coordinate)
                )
                phase_summary.append(
                    {
                        "corpus": corpus,
                        "event": event,
                        "coordinate": coordinate,
                        "events": len(selected),
                        "event_mean": float(event_value.mean()),
                        "control_mean": float(control_value.mean()),
                        "paired_difference": float(difference.mean()),
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
    write_csv(args.output / "phase_summary.csv", phase_summary)
    plot_phase(phase_rows, args.output)

    lookup = {
        (row["corpus"], row["event"], row["lead"], row["axis"]): row for row in axis_rows
    }
    stable_axes = {}
    for event in ("loop", "static"):
        candidates = []
        for row in axis_rows:
            if row["corpus"] != "main16x32" or row["event"] != event or row["lead"] != -2:
                continue
            if row["events"] < 50 or not np.isfinite(row["auc_event_high"]):
                continue
            external = lookup[("grid50x8", event, -2, row["axis"])]
            direction = 1 if row["auc_event_high"] >= 0.5 else -1
            main_detection = max(row["auc_event_high"], 1.0 - row["auc_event_high"])
            external_locked = external["auc_event_high"] if direction > 0 else 1.0 - external["auc_event_high"]
            candidates.append(
                {
                    "axis": row["axis"],
                    "direction": direction,
                    "main_auc": main_detection,
                    "grid_locked_auc": external_locked,
                    "main_events": row["events"],
                    "grid_events": external["events"],
                    "replicated_advantage": min(main_detection, external_locked) - 0.5,
                }
            )
        stable_axes[event] = sorted(
            candidates, key=lambda row: row["replicated_advantage"], reverse=True
        )[:20]

    summary = {
        "schema": "himoe.graph_homeostasis_evaluation.v1",
        "tasks": len(tasks),
        "axes": len(canonical_names),
        "leads": list(LEADS),
        "outcome_weight_training": False,
        "percentile_reference": "all main16x32 query rows; outcome labels not used",
        "matched_control": "same task, scene, and absolute query; neither loop nor static",
        "fixed_definitions": DEFINITIONS,
        "fixed_composites": [row for row in composite_rows if row["status"] == "prior_fixed"],
        "locked_discovery_method": (
            "Top four full-coverage q-2 axes and directions selected by main16x32 matched AUC; "
            "the grid50x8 result is direction-locked confirmation, not weight training."
        ),
        "locked_discovery_definitions": LOCKED_DISCOVERY,
        "locked_discovery_composites": [
            row for row in composite_rows if row["status"] == "main16x32_rank_locked"
        ],
        "phase_summary": phase_summary,
        "post_hoc_axis_audit": stable_axes,
        "caveat": "The axis scan is descriptive discovery; only fixed composites test the prior hypothesis.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fixed_q_minus_2": [
                    row for row in composite_rows
                    if row["lead"] == -2 and row["status"] == "prior_fixed"
                ],
                "locked_q_minus_2": [
                    row for row in composite_rows
                    if row["lead"] == -2 and row["status"] == "main16x32_rank_locked"
                ],
                "phase_q_minus_2": phase_summary,
                "top_replicated_axes": {key: value[:3] for key, value in stable_axes.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
