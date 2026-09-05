#!/usr/bin/env python3
"""Scan every retained MoE axis and externally test discovery-locked composites."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "assurance"))

from evaluation import LOOP_BLIND_TASKS, load_task, profile_inventory, row_lookup  # noqa: E402
from profile_schema import empirical_percentile  # noqa: E402


OUTPUT = BUNDLE / "results/full_moe_axes"
LEADS = (-8, -6, -4, -2, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--composite-size", type=int, default=5)
    parser.add_argument("--correlation-cap", type=float, default=0.85)
    return parser.parse_args()


def axes(data) -> tuple[tuple[str, ...], np.ndarray]:
    profile = data.profile
    names = list(profile.feature_names)
    blocks = [profile.features]
    layer_names = ("L12", "L13", "L14", "L15")
    for layer, layer_name in enumerate(layer_names):
        names.extend(
            f"layer_{layer_name}_{feature}" for feature in profile.layer_feature_names
        )
        blocks.append(profile.layer_features[:, layer, :])
    names.extend(f"flow_transition_{index}" for index in range(profile.flow_profile.shape[1]))
    blocks.append(profile.flow_profile)
    names.extend(f"recurrence_lag_{index + 1}" for index in range(profile.recurrence_distance.shape[1]))
    blocks.append(profile.recurrence_distance)
    return tuple(names), np.column_stack(blocks).astype(np.float64)


def matched_pairs(data, event_type: str, lead: int):
    if event_type == "loop" and data.task in LOOP_BLIND_TASKS:
        return []
    onset_key = f"{event_type}_onset_q"
    lookup = row_lookup(data.profile)
    controls = {}
    for event in data.events:
        if int(event["loop_onset_q"]) >= 0 or int(event["static_onset_q"]) >= 0:
            continue
        episode = int(event["episode_id"])
        scene = int(event["scene"])
        rows = np.flatnonzero(data.profile.episode_id == episode)
        for row in rows:
            controls.setdefault((scene, int(data.profile.query[row])), []).append(row)
    pairs = []
    for event in data.events:
        onset = int(event[onset_key])
        query = onset + lead
        if onset < 0 or query < 0:
            continue
        positive = lookup.get((int(event["episode_id"]), query))
        negative = controls.get((int(event["scene"]), query), ())
        if positive is not None and negative:
            pairs.append((positive, np.asarray(negative, dtype=np.int64)))
    return pairs


def evaluate_axis(task_matrices, pair_map, column: int) -> dict[str, float | int]:
    wins = 0.0
    pairs = 0
    events = 0
    for key, matched in pair_map.items():
        values = task_matrices[key][:, column]
        for positive, negative_rows in matched:
            positive_value = values[positive]
            negative = values[negative_rows]
            negative = negative[np.isfinite(negative)]
            if not np.isfinite(positive_value) or not len(negative):
                continue
            wins += float(np.count_nonzero(positive_value > negative))
            wins += 0.5 * float(np.count_nonzero(positive_value == negative))
            pairs += len(negative)
            events += 1
    auc = wins / pairs if pairs else float("nan")
    return {"auc_event_high": auc, "events": events, "pairs": pairs}


def evaluate_score(task_scores, pair_map) -> dict[str, float | int]:
    wins = 0.0
    pairs = 0
    events = 0
    for key, matched in pair_map.items():
        values = task_scores[key]
        for positive, negative_rows in matched:
            positive_value = values[positive]
            negative = values[negative_rows]
            negative = negative[np.isfinite(negative)]
            if not np.isfinite(positive_value) or not len(negative):
                continue
            wins += float(np.count_nonzero(positive_value > negative))
            wins += 0.5 * float(np.count_nonzero(positive_value == negative))
            pairs += len(negative)
            events += 1
    return {
        "auc_event_high": wins / pairs if pairs else float("nan"),
        "events": events,
        "pairs": pairs,
    }


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def select_axes(rows, event_type, axis_names, main_matrix, count, cap):
    relevant = [row for row in rows if row["corpus"] == "main16x32" and row["event"] == event_type]
    relevant.sort(key=lambda row: row["auc_detection"], reverse=True)
    selected = []
    selected_vectors = []
    for row in relevant:
        index = axis_names.index(row["axis"])
        direction = 1 if row["auc_event_high"] >= 0.5 else -1
        vector = main_matrix[:, index] * direction
        if any(abs(finite_spearman(vector, prior)) >= cap for prior in selected_vectors):
            continue
        selected.append(
            {
                "axis": row["axis"],
                "direction": direction,
                "main_auc": row["auc_detection"],
            }
        )
        selected_vectors.append(vector)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(item) for item in profile_inventory()]
    task_matrices = {}
    axis_names = None
    for data in tasks:
        names, values = axes(data)
        if axis_names is None:
            axis_names = names
        elif names != axis_names:
            raise RuntimeError("axis schema drift")
        task_matrices[(data.corpus, data.suite, data.task)] = values

    pair_maps = {}
    for corpus in ("main16x32", "grid50x8"):
        for event_type in ("loop", "static"):
            for lead in LEADS:
                pair_maps[(corpus, event_type, lead)] = {
                    (data.corpus, data.suite, data.task): matched_pairs(data, event_type, lead)
                    for data in tasks
                    if data.corpus == corpus
                }

    rows = []
    for (corpus, event_type, lead), pairs in pair_maps.items():
        for column, name in enumerate(axis_names):
            result = evaluate_axis(task_matrices, pairs, column)
            auc = result["auc_event_high"]
            rows.append(
                {
                    "corpus": corpus,
                    "event": event_type,
                    "lead": lead,
                    "axis": name,
                    **result,
                    "auc_detection": max(auc, 1 - auc) if np.isfinite(auc) else float("nan"),
                }
            )
    with (args.output / "all_axis_onset_auc.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    main_keys = [key for key in task_matrices if key[0] == "main16x32"]
    main_matrix = np.concatenate([task_matrices[key] for key in main_keys], axis=0)
    composites = []
    for event_type in ("loop", "static"):
        selected = select_axes(
            [row for row in rows if row["lead"] == -2],
            event_type,
            axis_names,
            main_matrix,
            args.composite_size,
            args.correlation_cap,
        )
        references = {
            item["axis"]: main_matrix[:, axis_names.index(item["axis"])] for item in selected
        }
        task_scores = {}
        for key, values in task_matrices.items():
            parts = []
            for item in selected:
                value = values[:, axis_names.index(item["axis"])]
                percentile = empirical_percentile(references[item["axis"]], value)
                parts.append(percentile if item["direction"] > 0 else 1.0 - percentile)
            stacked = np.column_stack(parts)
            count = np.isfinite(stacked).sum(axis=1)
            task_scores[key] = np.divide(
                np.nansum(stacked, axis=1),
                count,
                out=np.full(len(stacked), np.nan, dtype=np.float64),
                where=count > 0,
            )
        evaluation = {}
        for corpus in ("main16x32", "grid50x8"):
            evaluation[corpus] = {
                str(lead): evaluate_score(
                    task_scores, pair_maps[(corpus, event_type, lead)]
                )
                for lead in LEADS
            }
        composites.append(
            {
                "event": event_type,
                "selection": "top main16x32 q-2 detection AUC, abs Spearman-pruned, equal percentile weights",
                "selected_axes": selected,
                "external_evaluation": evaluation,
            }
        )
        for data in tasks:
            key = (data.corpus, data.suite, data.task)
            destination = args.output / "composites" / data.corpus / data.suite / f"{data.task}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if destination.exists():
                with np.load(destination, allow_pickle=False) as archive:
                    existing = {name: np.asarray(archive[name]) for name in archive.files}
            existing.update(
                {
                    "episode_id": data.profile.episode_id,
                    "query": data.profile.query,
                    f"{event_type}_risk": task_scores[key].astype(np.float32),
                }
            )
            np.savez_compressed(destination, **existing)

    locked = []
    for event_type in ("loop", "static"):
        main_rows = {
            row["axis"]: row
            for row in rows
            if row["corpus"] == "main16x32" and row["event"] == event_type and row["lead"] == -2
        }
        grid_rows = {
            row["axis"]: row
            for row in rows
            if row["corpus"] == "grid50x8" and row["event"] == event_type and row["lead"] == -2
        }
        for name in axis_names:
            direction = 1 if main_rows[name]["auc_event_high"] >= 0.5 else -1
            external = grid_rows[name]["auc_event_high"]
            external_locked = external if direction > 0 else 1 - external
            locked.append(
                {
                    "event": event_type,
                    "axis": name,
                    "direction_from_main": direction,
                    "main_locked_auc": main_rows[name]["auc_detection"],
                    "grid_locked_auc": external_locked,
                    "stable_advantage": min(main_rows[name]["auc_detection"], external_locked) - 0.5,
                }
            )
    locked.sort(key=lambda row: (row["event"], -row["stable_advantage"]))
    with (args.output / "discovery_locked_external.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(locked[0]))
        writer.writeheader()
        writer.writerows(locked)

    summary = {
        "schema": "himoe.assurance.full_moe_axis_scan.v1",
        "axes": len(axis_names),
        "aggregate_axes": len(tasks[0].profile.feature_names),
        "layer_resolved_axes": 4 * len(tasks[0].profile.layer_feature_names),
        "flow_resolved_axes": tasks[0].profile.flow_profile.shape[1],
        "recurrence_axes": tasks[0].profile.recurrence_distance.shape[1],
        "label_use": "main16x32 direction/selection only; grid50x8 is untouched confirmation",
        "fitted_weights": False,
        "composites": composites,
        "top_stable_axes": {
            event: [row for row in locked if row["event"] == event][:15]
            for event in ("loop", "static")
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"axes": len(axis_names), "composites": composites, "top_stable_axes": summary["top_stable_axes"]}, indent=2))


if __name__ == "__main__":
    main()
