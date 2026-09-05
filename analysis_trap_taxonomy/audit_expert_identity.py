#!/usr/bin/env python3
"""Test whether loop/static correspond to fixed HB expert identities.

The audit compares each event at its physical onset with episodes having no
event of that type at the same task, initial scene, and absolute query.  It
tests both full-softmax probabilities and the model's recorded top-4 dispatch
sets.  Task-scene cells are the inferential units; no classifier is fitted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "VLA_MUI_HUB" / "cache_new" / "HiMoE-VLA"
RUNS = {
    "seed1000_1007": "right-50x8-20260903",
    "seed1008_1015": "right-50x8b-20260903",
}
EVENTS = (("loop", "loop_onset"), ("static", "static_onset"))
LAYERS = (12, 13, 14, 15)
N_EXPERTS = 32


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return math.nan
    tail = min(positive, negative)
    value = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n
    return min(value, 1.0)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.minimum(ranked, 1.0)
    return output


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(float(x) for x in np.quantile(values[index].mean(axis=1), (0.025, 0.975)))


def query_index(episodes: np.ndarray) -> np.ndarray:
    output = np.empty(len(episodes), np.int16)
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        if len(index) > 1 and np.any(np.diff(index) != 1):
            raise ValueError(f"non-contiguous route rows for episode {episode}")
        output[index] = np.arange(len(index))
    return output


def route_vectors(group: zarr.Group) -> dict[str, np.ndarray]:
    probability = np.asarray(
        group["hb_router_probs"][:, 4:8, 9, 1:11, :], np.float32
    )
    probability = np.maximum(probability, 0.0)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    probability = probability.mean(axis=2).reshape(len(probability), -1)

    expert_ids = np.asarray(
        group["hb_expert_ids"][:, 4:8, 9, 1:11, :], np.uint8
    )
    occupancy = np.empty((len(expert_ids), len(LAYERS), N_EXPERTS), np.float32)
    for expert in range(N_EXPERTS):
        occupancy[:, :, expert] = (expert_ids == expert).any(axis=-1).mean(axis=2)
    return {
        "soft_probability": probability,
        "top4_occupancy": occupancy.reshape(len(occupancy), -1),
    }


def comparison_effect(value: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return (
        np.mean(value[None, :] > controls, axis=0)
        + 0.5 * np.mean(value[None, :] == controls, axis=0)
        - 0.5
    )


def collect_cells(events: pd.DataFrame) -> dict[tuple, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    output: dict[tuple, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for label, run_id in RUNS.items():
        run_events = events[events["run"] == label]
        for task_index, (task, task_events) in enumerate(run_events.groupby("task_key"), 1):
            path = HUB / task / run_id
            group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
            episodes = np.asarray(group["episode_id"][:], np.int32)
            queries = query_index(episodes)
            vectors = route_vectors(group)
            for event, onset_column in EVENTS:
                selected = task_events[task_events[onset_column] >= 0]
                for row in selected.itertuples(index=False):
                    query = int(getattr(row, onset_column))
                    event_index = np.flatnonzero(
                        (episodes == int(row.episode)) & (queries == query)
                    )
                    control_episodes = task_events[
                        (task_events["scene"] == int(row.scene))
                        & (task_events[onset_column] < 0)
                    ]["episode"].to_numpy()
                    control_index = np.flatnonzero(
                        np.isin(episodes, control_episodes) & (queries == query)
                    )
                    if len(event_index) != 1 or not len(control_index):
                        continue
                    cell = (label, event, task, int(row.scene))
                    metrics = output.setdefault(cell, {})
                    for metric, values in vectors.items():
                        event_value = values[event_index[0]]
                        controls = values[control_index]
                        metrics.setdefault(metric, []).append(
                            (
                                comparison_effect(event_value, controls),
                                event_value - controls.mean(axis=0),
                            )
                        )
            print(f"[{label} expert {task_index:02d}/40] {task}", flush=True)
    return output


def summarize_cells(cells: dict, draws: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(20260904)
    rows = []
    task_vectors: dict[tuple, list[np.ndarray]] = {}
    for run in RUNS:
        for event, _ in EVENTS:
            for metric in ("soft_probability", "top4_occupancy"):
                selected = []
                raw_selected = []
                for (cell_run, cell_event, task, _scene), values in cells.items():
                    if (cell_run, cell_event) != (run, event) or metric not in values:
                        continue
                    auc = np.mean([item[0] for item in values[metric]], axis=0)
                    raw = np.mean([item[1] for item in values[metric]], axis=0)
                    selected.append(auc)
                    raw_selected.append(raw)
                    task_vectors.setdefault((run, event, metric, task), []).append(auc)
                matrix = np.stack(selected)
                raw_matrix = np.stack(raw_selected)
                p_values = np.asarray(
                    [
                        exact_sign_p(int((matrix[:, i] > 0).sum()), int((matrix[:, i] < 0).sum()))
                        for i in range(matrix.shape[1])
                    ]
                )
                adjusted = bh_adjust(p_values)
                for index in range(matrix.shape[1]):
                    low, high = bootstrap_ci(matrix[:, index], rng, draws)
                    rows.append(
                        {
                            "run": run,
                            "event": event,
                            "metric": metric,
                            "layer": LAYERS[index // N_EXPERTS],
                            "expert": index % N_EXPERTS,
                            "auc_minus_half": float(matrix[:, index].mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "mean_raw_difference": float(raw_matrix[:, index].mean()),
                            "cells": len(matrix),
                            "positive_cells": int((matrix[:, index] > 0).sum()),
                            "negative_cells": int((matrix[:, index] < 0).sum()),
                            "sign_p": float(p_values[index]),
                            "bh_q": float(adjusted[index]),
                        }
                    )

    frame = pd.DataFrame(rows)
    cross_run = {}
    for event, _ in EVENTS:
        cross_run[event] = {}
        for metric in ("soft_probability", "top4_occupancy"):
            a = frame[
                (frame.run == "seed1000_1007")
                & (frame.event == event)
                & (frame.metric == metric)
            ].sort_values(["layer", "expert"])
            b = frame[
                (frame.run == "seed1008_1015")
                & (frame.event == event)
                & (frame.metric == metric)
            ].sort_values(["layer", "expert"])
            x = a.auc_minus_half.to_numpy()
            y = b.auc_minus_half.to_numpy()
            same_sign = np.sign(x) == np.sign(y)
            replicated = (a.bh_q.to_numpy() < 0.05) & (b.bh_q.to_numpy() < 0.05) & same_sign
            top_a = set(np.argsort(np.abs(x))[-10:])
            top_b = set(np.argsort(np.abs(y))[-10:])
            task_cosines = []
            tasks = {
                key[3] for key in task_vectors if key[:3] == ("seed1000_1007", event, metric)
            } & {
                key[3] for key in task_vectors if key[:3] == ("seed1008_1015", event, metric)
            }
            for task in tasks:
                left = np.mean(task_vectors[("seed1000_1007", event, metric, task)], axis=0)
                right = np.mean(task_vectors[("seed1008_1015", event, metric, task)], axis=0)
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                if denominator:
                    task_cosines.append(float(left @ right / denominator))
            cross_run[event][metric] = {
                "auc_vector_correlation": float(np.corrcoef(x, y)[0, 1]),
                "auc_vector_cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
                "sign_agreement": float(same_sign.mean()),
                "replicated_bh_q05_same_sign": int(replicated.sum()),
                "top10_absolute_overlap": len(top_a & top_b),
                "task_vector_cosine_median": float(np.median(task_cosines)),
                "tasks_compared": len(task_cosines),
                "largest_single_cell_matched_auc_deviation_a": float(np.max(np.abs(x))),
                "largest_single_cell_matched_auc_deviation_b": float(np.max(np.abs(y))),
            }
    return frame, cross_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.output / "events.csv.gz")
    cells = collect_cells(events)
    effects, cross_run = summarize_cells(cells, args.bootstrap)
    effects.to_csv(args.output / "expert_identity_effects.csv", index=False)
    summary = {
        "schema": "himoe.trainfree_expert_identity_audit.v1",
        "training": False,
        "comparison": "event onset vs no-same-event controls at task+scene+absolute query",
        "inference_unit": "task-scene",
        "multiple_testing": "Benjamini-Hochberg across 4 layers x 32 experts per event/metric/run",
        "cross_run": cross_run,
        "interpretation": (
            "replicated mean reweighting may name a distributed signature; it does not "
            "make any one expert a deterministic trap switch"
        ),
        "limitations": [
            "the two runs share one trained checkpoint and differ in flow-noise seeds",
            "expert identities are layer-specific",
            "top-4 slots are unsorted, so occupancy ignores slot order",
            "event labels remain kinematic proxies",
        ],
    }
    (args.output / "expert_identity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(cross_run, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
