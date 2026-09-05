#!/usr/bin/env python3
"""Scan full-router dynamic axes and externally test label-locked composites."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys


BUNDLE = Path(__file__).resolve().parent.parent
PROJECT = BUNDLE.parent
PROFILE_ROOT = BUNDLE / "results/dynamic_profiles"
EVENT_ROOT = PROJECT / "analysis_moe_phenotype/events"
OUTPUT = BUNDLE / "results/dynamic_evaluation"
LEADS = (-8, -6, -4, -2, 0)
sys.path.insert(0, str(BUNDLE / "dynamic"))

from scoring import SCHEMA as SCORE_REFERENCE_SCHEMA  # noqa: E402
LOOP_BLIND_TASKS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "open_the_middle_drawer_of_the_cabinet",
}
PREDECLARED = {
    "loop": [
        ("aggregate|late|action|flow_accel_mean", 1),
        ("aggregate|late|action|flow_turn_instability", 1),
        ("aggregate|late|flow_graph_curvature_mean", 1),
        ("aggregate|late|expert_load_tv_mean", 1),
    ],
    "static": [
        ("query_aggregate|late|action|recurrence_lag1", -1),
        ("query_aggregate|late|action|query_accel", -1),
        ("query_aggregate|late|action|query_jerk", -1),
        ("query_aggregate|late|graph_d1_abs", -1),
    ],
}


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
    parser.add_argument("--composite-size", type=int, default=8)
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
    observed = {corpus: sum(task.corpus == corpus for task in tasks) for corpus in ("main16x32", "grid50x8")}
    if observed != {"main16x32": 5, "grid50x8": 40}:
        raise RuntimeError(f"dynamic profile inventory mismatch: {observed}")
    return tasks


def load_profile(task: Task) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    with np.load(task.profile, allow_pickle=False) as archive:
        within_names = tuple(archive["within_feature_names"].astype(str))
        query_names = tuple(archive["query_feature_names"].astype(str))
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
    return within_names + query_names, matrix, metadata


def event_onset(task: Task, event: dict[str, int | str], event_type: str) -> int:
    if event_type == "loop" and task.task in LOOP_BLIND_TASKS:
        return -1
    return int(event[f"{event_type}_onset_q"])


def matched_pairs(
    task: Task, metadata: dict[str, np.ndarray], event_type: str, lead: int
) -> list[tuple[int, np.ndarray]]:
    if event_type == "loop" and task.task in LOOP_BLIND_TASKS:
        return []
    episode_id = metadata["episode_id"]
    query = metadata["query"]
    lookup = {(int(episode), int(q)): row for row, (episode, q) in enumerate(zip(episode_id, query))}
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
        target_query = onset + lead
        if onset < 0 or target_query < 0:
            continue
        positive = lookup.get((int(event["episode_id"]), target_query))
        negative = controls.get((int(event["scene"]), target_query), ())
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


def axis_family(axis: str) -> str:
    parts = axis.split("|")
    if parts[0] == "query_aggregate" and parts[-1].startswith("recurrence_lag"):
        return "query_aggregate|recurrence"
    if parts[0] in {"aggregate", "query_aggregate"}:
        return f"{parts[0]}|{parts[-1]}"
    if parts[0] == "graph_level":
        return f"graph_level|{parts[-1]}"
    return parts[0]


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)].astype(np.float64))
    output = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    rank = np.searchsorted(reference, values[valid], side="right")
    output[valid] = (rank + 0.5) / (len(reference) + 1.0)
    return np.clip(output, 1e-8, 1.0 - 1e-8)


def scalar_auc(score: np.ndarray, pairs) -> dict[str, float | int]:
    wins = 0.0
    count = 0
    events = 0
    for positive, negatives in pairs:
        value = score[positive]
        controls = score[negatives]
        controls = controls[np.isfinite(controls)]
        if not np.isfinite(value) or not len(controls):
            continue
        wins += float(np.count_nonzero(value > controls))
        wins += 0.5 * float(np.count_nonzero(value == controls))
        count += len(controls)
        events += 1
    auc = wins / count if count else float("nan")
    return {"auc_event_high": auc, "events": events, "pairs": count}


def score_definitions(
    selected_full: dict[str, list[dict[str, Any]]],
    selected_history: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    definitions = {
        f"predeclared_{event}": axes for event, axes in PREDECLARED.items()
    }
    for event, rows in selected_full.items():
        definitions[f"discovery_full_{event}"] = [
            (row["axis"], row["direction"]) for row in rows
        ]
    for event, rows in selected_history.items():
        definitions[f"discovery_history_{event}"] = [
            (row["axis"], row["direction"]) for row in rows
        ]
    return definitions


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = inventory()
    first_names, first_matrix, _ = load_profile(tasks[0])
    axis_names = first_names
    axis_count = len(axis_names)
    del first_matrix
    accumulators = {
        (corpus, event, lead): empty_accumulator(axis_count)
        for corpus in ("main16x32", "grid50x8")
        for event in ("loop", "static")
        for lead in LEADS
    }

    task_metadata: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for position, task in enumerate(tasks, 1):
        names, matrix, metadata = load_profile(task)
        if names != axis_names:
            raise ValueError(f"feature schema mismatch: {task.profile}")
        task_metadata[(task.corpus, task.suite, task.task)] = metadata
        for event in ("loop", "static"):
            for lead in LEADS:
                pairs = matched_pairs(task, metadata, event, lead)
                accumulate(accumulators[(task.corpus, event, lead)], matrix, pairs)
        print(f"[scan {position}/{len(tasks)}] {task.corpus}/{task.suite}/{task.task}", flush=True)
        del matrix

    rows: list[dict[str, Any]] = []
    for (corpus, event, lead), accumulator in accumulators.items():
        for index, axis in enumerate(axis_names):
            pair_count = int(accumulator["pairs"][index])
            auc = float(accumulator["wins"][index] / pair_count) if pair_count else float("nan")
            rows.append(
                {
                    "corpus": corpus,
                    "event": event,
                    "lead": lead,
                    "axis": axis,
                    "family": axis_family(axis),
                    "auc_event_high": auc,
                    "auc_detection": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                    "events": int(accumulator["events"][index]),
                    "pairs": pair_count,
                }
            )
    axis_csv = args.output / "all_axis_matched_auc.csv"
    with axis_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["corpus"], row["event"], row["lead"], row["axis"]): row for row in rows}
    selected_full: dict[str, list[dict[str, Any]]] = {}
    selected_history: dict[str, list[dict[str, Any]]] = {}
    stable: dict[str, list[dict[str, Any]]] = {}
    for event in ("loop", "static"):
        main_rows = [
            row
            for row in rows
            if row["corpus"] == "main16x32" and row["event"] == event and row["lead"] == -2
            and row["events"] >= 50
        ]
        main_rows.sort(key=lambda row: row["auc_detection"], reverse=True)
        max_events = max(row["events"] for row in main_rows)
        max_pairs = max(row["pairs"] for row in main_rows)
        chosen_full = []
        chosen_history = []
        full_families = set()
        history_families = set()
        all_stable = []
        for row in main_rows:
            external = lookup[("grid50x8", event, -2, row["axis"])]
            direction = 1 if row["auc_event_high"] >= 0.5 else -1
            external_locked = external["auc_event_high"] if direction == 1 else 1.0 - external["auc_event_high"]
            candidate = {
                "axis": row["axis"],
                "family": row["family"],
                "direction": direction,
                "main_detection_auc": row["auc_detection"],
                "grid_locked_auc": external_locked,
                "stable_advantage": min(row["auc_detection"] - 0.5, external_locked - 0.5),
                "main_events": row["events"],
                "grid_events": external["events"],
            }
            all_stable.append(candidate)
            if (
                row["events"] == max_events
                and row["pairs"] == max_pairs
                and row["family"] not in full_families
                and len(chosen_full) < args.composite_size
            ):
                chosen_full.append(candidate)
                full_families.add(row["family"])
            if (
                row["family"] not in history_families
                and len(chosen_history) < args.composite_size
            ):
                chosen_history.append(candidate)
                history_families.add(row["family"])
        selected_full[event] = chosen_full
        selected_history[event] = chosen_history
        stable[event] = sorted(all_stable, key=lambda row: row["stable_advantage"], reverse=True)[:30]

    definitions = score_definitions(selected_full, selected_history)
    used_axes = sorted({axis for definition in definitions.values() for axis, _ in definition})
    indices = {axis: axis_names.index(axis) for axis in used_axes}
    references: dict[str, list[np.ndarray]] = {axis: [] for axis in used_axes}
    for task in tasks:
        if task.corpus != "main16x32":
            continue
        _names, matrix, _metadata = load_profile(task)
        for axis, index in indices.items():
            references[axis].append(matrix[:, index])
    sorted_reference = {}
    for axis, values in references.items():
        joined = np.concatenate(values).astype(np.float64)
        sorted_reference[axis] = np.sort(joined[np.isfinite(joined)])

    reference_names = tuple(sorted_reference)
    reference_offsets = [0]
    reference_values = []
    for axis in reference_names:
        reference_values.append(sorted_reference[axis].astype(np.float32))
        reference_offsets.append(reference_offsets[-1] + len(reference_values[-1]))
    reference_index = {axis: index for index, axis in enumerate(reference_names)}
    score_offsets = [0]
    score_reference_indices = []
    score_directions = []
    for definition in definitions.values():
        for axis, direction in definition:
            score_reference_indices.append(reference_index[axis])
            score_directions.append(direction)
        score_offsets.append(len(score_reference_indices))
    np.savez_compressed(
        args.output / "score_reference.npz",
        schema=np.asarray(SCORE_REFERENCE_SCHEMA),
        feature_names=np.asarray(axis_names),
        reference_names=np.asarray(reference_names),
        reference_offsets=np.asarray(reference_offsets, dtype=np.int64),
        reference_values=np.concatenate(reference_values),
        score_names=np.asarray(tuple(definitions)),
        score_offsets=np.asarray(score_offsets, dtype=np.int32),
        score_reference_indices=np.asarray(score_reference_indices, dtype=np.int32),
        score_directions=np.asarray(score_directions, dtype=np.int8),
    )

    composite_rows = []
    score_root = args.output / "scores"
    for position, task in enumerate(tasks, 1):
        _names, matrix, metadata = load_profile(task)
        task_scores = {}
        for score_name, definition in definitions.items():
            components = []
            for axis, direction in definition:
                value = empirical_percentile(sorted_reference[axis], matrix[:, indices[axis]])
                components.append(value if direction == 1 else 1.0 - value)
            component_matrix = np.column_stack(components)
            valid = np.isfinite(component_matrix).all(axis=1)
            score = np.full(len(matrix), np.nan, dtype=np.float32)
            score[valid] = component_matrix[valid].mean(axis=1).astype(np.float32)
            task_scores[score_name] = score
        destination = score_root / task.corpus / task.suite / f"{task.task}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            episode_id=metadata["episode_id"],
            query=metadata["query"],
            score_names=np.asarray(tuple(task_scores)),
            scores=np.column_stack(tuple(task_scores.values())),
        )
        for event in ("loop", "static"):
            for lead in LEADS:
                pairs = matched_pairs(task, metadata, event, lead)
                for score_name, score in task_scores.items():
                    result = scalar_auc(score, pairs)
                    result.update(
                        {
                            "corpus": task.corpus,
                            "suite": task.suite,
                            "task": task.task,
                            "event": event,
                            "lead": lead,
                            "score": score_name,
                        }
                    )
                    composite_rows.append(result)
        print(f"[score {position}/{len(tasks)}] {task.corpus}/{task.suite}/{task.task}", flush=True)
        del matrix

    aggregate_composites = []
    for corpus in ("main16x32", "grid50x8"):
        for event in ("loop", "static"):
            for lead in LEADS:
                for score_name in definitions:
                    subset = [
                        row
                        for row in composite_rows
                        if row["corpus"] == corpus and row["event"] == event
                        and row["lead"] == lead and row["score"] == score_name
                    ]
                    pairs = sum(int(row["pairs"]) for row in subset)
                    weighted = sum(float(row["auc_event_high"]) * int(row["pairs"]) for row in subset if row["pairs"])
                    aggregate_composites.append(
                        {
                            "corpus": corpus,
                            "event": event,
                            "lead": lead,
                            "score": score_name,
                            "auc_event_high": weighted / pairs if pairs else float("nan"),
                            "events": sum(int(row["events"]) for row in subset),
                            "pairs": pairs,
                        }
                    )
    composite_csv = args.output / "composite_matched_auc.csv"
    with composite_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(aggregate_composites[0]))
        writer.writeheader()
        writer.writerows(aggregate_composites)

    source_breakdown: dict[str, int] = {}
    for axis in axis_names:
        family = axis.split("|", 1)[0]
        source_breakdown[family] = source_breakdown.get(family, 0) + 1
    summary = {
        "schema": "himoe.dynamic_axis_evaluation.v1",
        "raw_router_shape": [8, 10, 11, 32],
        "dynamic_axes": axis_count,
        "axis_family_counts": source_breakdown,
        "fitted_weights": False,
        "classifier_trained": False,
        "percentile_reference": "all main16x32 query rows; outcomes not used",
        "axis_direction_selection": "main16x32 q-2 labels only",
        "external_confirmation": "grid50x8; previously seen corpus, not a pristine holdout",
        "matched_control": "same task, scene, and absolute query; neither loop nor static",
        "predeclared_composites": {
            event: [{"axis": axis, "direction": direction} for axis, direction in axes]
            for event, axes in PREDECLARED.items()
        },
        "discovery_full_coverage_axes": selected_full,
        "discovery_history_axes": selected_history,
        "coverage_policy": (
            "full composites require the maximum main16x32 q-2 event and pair support; "
            "history composites allow axes with at least 50 main16x32 events and report reduced support"
        ),
        "top_cross_corpus_stable_axes_descriptive": stable,
        "composite_results": aggregate_composites,
        "static_timing_caution": "static onset requires seven consecutive overlapping width-2 stationary windows; q-2 is confirmation of an ongoing run, not a two-query fate forecast",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    headline = [
        row
        for row in aggregate_composites
        if row["corpus"] == "grid50x8" and row["lead"] == -2
        and row["score"] in {
            "predeclared_loop",
            "predeclared_static",
            "discovery_full_loop",
            "discovery_full_static",
            "discovery_history_loop",
            "discovery_history_static",
        }
        and row["event"] in row["score"]
    ]
    print(json.dumps({"dynamic_axes": axis_count, "external_q_minus_2": headline}, indent=2))


if __name__ == "__main__":
    main()
