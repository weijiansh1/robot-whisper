#!/usr/bin/env python3
"""Audit HB expert identities around persistent phantom-grasp events."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_belief_mismatch as base  # noqa: E402


LAYERS = (12, 13, 14, 15)
RELATIVE_QUERIES = (2, 3)
N_EXPERTS = 32
METRICS = ("soft_probability", "top4_occupancy")


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return math.nan
    tail = min(positive, negative)
    value = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n
    return min(value, 1.0)


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.minimum(ranked, 1.0)
    return output


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(
        float(value)
        for value in np.quantile(values[index].mean(axis=1), (0.025, 0.975))
    )


def soft_vector(probability: np.ndarray) -> np.ndarray:
    values = base.normalize(probability[:, 9, 1:11, :])
    return values.mean(axis=1).reshape(-1)


def occupancy_vector(expert_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(expert_ids[:, 9, 1:11, :], dtype=np.uint8)
    output = np.empty((len(LAYERS), N_EXPERTS), dtype=np.float32)
    for expert in range(N_EXPERTS):
        output[:, expert] = (ids == expert).any(axis=-1).mean(axis=1)
    return output.reshape(-1)


def load_event_vectors(events: pd.DataFrame) -> pd.DataFrame:
    selected = base.matched_route_events(events)
    records: list[dict[str, Any]] = []
    for run_label, cfg in base.RUNS.items():
        run_events = selected[selected.run == run_label]
        for task, local_events in run_events.groupby("task", sort=True):
            run = base.HUB / str(task) / str(cfg["run_id"])
            store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
            episode_ids = np.asarray(store["episode_id"][:], dtype=np.int32)
            control_steps = np.asarray(store["control_step"][:], dtype=np.int64)
            episode_rows = {
                int(episode): rows[np.argsort(control_steps[rows])]
                for episode in local_events.episode.unique()
                for rows in [np.flatnonzero(episode_ids == int(episode))]
            }
            requests = []
            needed: set[int] = set()
            for event in local_events.to_dict("records"):
                rows = episode_rows[int(event["episode"])]
                anchor = int(event["post_closure_state_query"])
                for relative in RELATIVE_QUERIES:
                    query = anchor + relative
                    if query >= len(rows):
                        continue
                    row = int(rows[query])
                    needed.add(row)
                    requests.append((event, relative, row))
            ordered = np.asarray(sorted(needed), dtype=np.int64)
            probabilities = np.asarray(
                store["hb_router_probs"].oindex[ordered, 4:8, :, :, :],
                dtype=np.float32,
            )
            expert_ids = np.asarray(
                store["hb_expert_ids"].oindex[ordered, 4:8, :, :, :],
                dtype=np.uint8,
            )
            probability_by_row = {
                int(row): soft_vector(probabilities[index])
                for index, row in enumerate(ordered)
            }
            occupancy_by_row = {
                int(row): occupancy_vector(expert_ids[index])
                for index, row in enumerate(ordered)
            }
            for event, relative, row in requests:
                records.append(
                    {
                        "run": run_label,
                        "task": str(task),
                        "episode": int(event["episode"]),
                        "init_state_id": int(event["init_state_id"]),
                        "target": str(event["target"]),
                        "event_type": str(event["event_type"]),
                        "relative_query": int(relative),
                        "soft_probability": probability_by_row[row],
                        "top4_occupancy": occupancy_by_row[row],
                    }
                )
            print(f"[belief expert {run_label}] {task}", flush=True)
    return pd.DataFrame(records)


def comparison_effect(value: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return (
        np.mean(value[None, :] > controls, axis=0)
        + 0.5 * np.mean(value[None, :] == controls, axis=0)
        - 0.5
    )


def summarize_vectors(
    vectors: pd.DataFrame, draws: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(20260904)
    rows = []
    keys = ["run", "relative_query", "task", "init_state_id", "target"]
    cell_vectors: dict[tuple[str, int, str], list[np.ndarray]] = {}
    cell_raw: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for cell_key, cell in vectors.groupby(keys, sort=False):
        controls = cell[cell.event_type == "successful_coupled_grasp"]
        positive = cell[cell.event_type == "belief_mismatch"]
        if not len(controls) or not len(positive):
            continue
        run, relative = str(cell_key[0]), int(cell_key[1])
        for metric in METRICS:
            control_values = np.stack(controls[metric].to_numpy())
            positive_values = np.stack(positive[metric].to_numpy())
            effects = np.stack(
                [comparison_effect(value, control_values) for value in positive_values]
            ).mean(axis=0)
            raw = positive_values.mean(axis=0) - control_values.mean(axis=0)
            cell_vectors.setdefault((run, relative, metric), []).append(effects)
            cell_raw.setdefault((run, relative, metric), []).append(raw)

    for run in base.RUNS:
        for metric in METRICS:
            family_rows = []
            family_p = []
            for relative in RELATIVE_QUERIES:
                matrix = np.stack(cell_vectors[(run, relative, metric)])
                raw_matrix = np.stack(cell_raw[(run, relative, metric)])
                for index in range(matrix.shape[1]):
                    positive_cells = int((matrix[:, index] > 0).sum())
                    negative_cells = int((matrix[:, index] < 0).sum())
                    p_value = exact_sign_p(positive_cells, negative_cells)
                    low, high = bootstrap_ci(matrix[:, index], rng, draws)
                    family_rows.append(
                        {
                            "run": run,
                            "relative_query": relative,
                            "metric": metric,
                            "layer": LAYERS[index // N_EXPERTS],
                            "expert": index % N_EXPERTS,
                            "auc_minus_half": float(matrix[:, index].mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "mean_raw_difference": float(raw_matrix[:, index].mean()),
                            "cells": len(matrix),
                            "positive_cells": positive_cells,
                            "negative_cells": negative_cells,
                            "sign_p": p_value,
                        }
                    )
                    family_p.append(p_value)
            adjusted = bh_adjust(np.asarray(family_p))
            for record, q_value in zip(family_rows, adjusted):
                record["bh_q_across_2x4x32"] = float(q_value)
                rows.append(record)

    frame = pd.DataFrame(rows)
    cross_run: dict[str, Any] = {}
    for relative in RELATIVE_QUERIES:
        cross_run[str(relative)] = {}
        for metric in METRICS:
            a = frame[
                (frame.run == "seed1000_1007")
                & (frame.relative_query == relative)
                & (frame.metric == metric)
            ].sort_values(["layer", "expert"])
            b = frame[
                (frame.run == "seed1008_1015")
                & (frame.relative_query == relative)
                & (frame.metric == metric)
            ].sort_values(["layer", "expert"])
            left = a.auc_minus_half.to_numpy()
            right = b.auc_minus_half.to_numpy()
            same_sign = np.sign(left) == np.sign(right)
            replicated = (
                (a.bh_q_across_2x4x32.to_numpy() < 0.05)
                & (b.bh_q_across_2x4x32.to_numpy() < 0.05)
                & same_sign
            )
            top_index = int(np.argmax(np.abs(left)))
            direction = 1.0 if left[top_index] >= 0 else -1.0
            cross_run[str(relative)][metric] = {
                "auc_vector_correlation": float(np.corrcoef(left, right)[0, 1]),
                "auc_vector_cosine": float(
                    left @ right / (np.linalg.norm(left) * np.linalg.norm(right))
                ),
                "sign_agreement": float(same_sign.mean()),
                "replicated_bh_q05_same_sign": int(replicated.sum()),
                "best_A_fixed_cell": {
                    "layer": int(a.iloc[top_index].layer),
                    "expert": int(a.iloc[top_index].expert),
                    "direction": "high" if direction > 0 else "low",
                    "oriented_auc_A": float(0.5 + direction * left[top_index]),
                    "oriented_auc_B": float(0.5 + direction * right[top_index]),
                    "bh_q_A": float(a.iloc[top_index].bh_q_across_2x4x32),
                    "bh_q_B": float(b.iloc[top_index].bh_q_across_2x4x32),
                },
            }
    return frame, cross_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=base.RESULTS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output = args.output.resolve()
    events = pd.read_csv(output / "belief_mismatch_events.csv.gz")
    vectors = load_event_vectors(events)
    effects, cross_run = summarize_vectors(vectors, args.bootstrap)
    effects.to_csv(output / "belief_mismatch_expert_effects.csv", index=False)
    summary = {
        "schema": "himoe.belief_mismatch_expert_identity_audit.v1",
        "training": False,
        "relative_queries": list(RELATIVE_QUERIES),
        "comparison": (
            "persistent belief mismatch vs successful coupled grasp, matched "
            "within task + init_state + target"
        ),
        "multiple_testing": "BH across 2 relative queries x 4 layers x 32 experts",
        "cross_run": cross_run,
        "interpretation": (
            "A replicated distributed signature does not make any single expert "
            "a deterministic phantom-grasp switch."
        ),
    }
    (output / "belief_mismatch_expert_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(cross_run, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
