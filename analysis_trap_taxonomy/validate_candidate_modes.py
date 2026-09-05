#!/usr/bin/env python3
"""Holdout audit of candidate trap modes not covered by loop/static.

Physical targets and empirical thresholds are calibrated only from successful
episodes in run A (flow-noise seeds 1000--1007), then frozen before run B is
labelled.  Routing comparisons are restricted to horizon-censored failures
without a loop/static event and matched within task plus initial scene.  No
failure label is used to choose a threshold, feature, direction, or weight.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "VLA_MUI_HUB" / "cache_new" / "HiMoE-VLA"
sys.path.insert(0, str(ROOT / "himoe-vla_trap" / "code"))

import analyze_hub_phenotype_atlas as atlas  # noqa: E402


RUNS = {
    "seed1000_1007": {
        "run_id": "right-50x8-20260903",
        "manifest": ROOT
        / "himoe-vla_trap/results/moe_invariant_alarm_cache_new_recomputed_50x8/route_only_manifest.json",
    },
    "seed1008_1015": {
        "run_id": "right-50x8b-20260903",
        "manifest": ROOT
        / "himoe-vla_trap/results/moe_invariant_alarm_cache_new_50x8b/route_only_manifest.json",
    },
}
SIGNALS = ("late_flow_volatility", "route_acceleration", "recurrence_raw")


def load_task_tapes(run: Path) -> tuple[list[dict], dict, list[tuple[np.ndarray, ...]]]:
    summaries = atlas.load_summaries(run)
    client = run / "client"
    layout = json.loads((client / "sim_layout.json").read_text())
    tapes = []
    for summary in summaries:
        episode = int(summary["episode_index"])
        with np.load(atlas.episode_path(client, episode), allow_pickle=False) as data:
            state = np.asarray(data["state"], np.float32)
            actions = np.asarray(data["actions"], np.float32)
            sim = np.asarray(data["sim_state"], np.float32)
        expected = int(summary["inference_calls"])
        if state.shape != (expected, 8) or actions.shape != (expected, 10, 7):
            raise ValueError(f"state/action shape mismatch: {run}/{episode}")
        if len(sim) != expected:
            raise ValueError(f"sim-state shape mismatch: {run}/{episode}")
        tapes.append((state, actions, sim))
    return summaries, layout, tapes


def remap_targets(targets: list[dict], layout: dict) -> list[dict]:
    joints = {str(item["joint"]): item for item in layout["joints"]}
    result = []
    for target in targets:
        joint = joints.get(str(target["name"]))
        if joint is None:
            raise ValueError(f"target {target['name']} absent from confirmation layout")
        lo = int(joint["state_lo"])
        result.append({**target, "lo": lo, "hi": lo + 3})
    return result


def feature_rows(
    label: str,
    task: str,
    summaries: list[dict],
    tapes: list[tuple[np.ndarray, ...]],
    targets: list[dict],
) -> list[dict]:
    rows = []
    for summary, (state, actions, sim) in zip(summaries, tapes):
        features, _ = atlas.extract_episode_features(
            state, actions, sim, targets, stop_fraction=1.0
        )
        rows.append(
            {
                "run": label,
                "task": task,
                "episode": int(summary["episode_index"]),
                "init_state_id": int(summary["init_state_id"]),
                "flow_noise_seed": int(summary["flow_noise_seed"]),
                "episode_length": int(summary["inference_calls"]),
                "failure": not bool(summary["success"]),
                "action_midlate_change": atlas.action_change_feature(actions),
                **features,
            }
        )
    return rows


def build_frozen_physical_labels() -> tuple[pd.DataFrame, list[dict]]:
    by_run = {}
    for label, cfg in RUNS.items():
        paths = sorted(HUB.glob(f"libero_*/*/{cfg['run_id']}"))
        complete = []
        for path in paths:
            meta = json.loads((path / "meta.json").read_text())
            if meta.get("status") == "complete" and meta.get("sampling", {}).get("complete"):
                complete.append(path)
        if len(complete) != 40:
            raise RuntimeError(f"{label}: expected 40 complete tasks, found {len(complete)}")
        by_run[label] = {str(path.relative_to(HUB).parent): path for path in complete}

    rows = []
    target_audit = []
    development = "seed1000_1007"
    confirmation = "seed1008_1015"
    if set(by_run[development]) != set(by_run[confirmation]):
        raise RuntimeError("development/confirmation task sets differ")

    for index, task in enumerate(sorted(by_run[development]), 1):
        a_summaries, a_layout, a_tapes = load_task_tapes(by_run[development][task])
        a_targets = atlas.identify_targets(
            a_summaries, a_layout, [tape[2] for tape in a_tapes]
        )
        rows.extend(feature_rows(development, task, a_summaries, a_tapes, a_targets))

        b_summaries, b_layout, b_tapes = load_task_tapes(by_run[confirmation][task])
        b_targets = remap_targets(a_targets, b_layout)
        rows.extend(feature_rows(confirmation, task, b_summaries, b_tapes, b_targets))
        target_audit.append(
            {
                "task": task,
                "target_source": development,
                "target_names": [str(target["name"]) for target in a_targets],
                "target_goals": [np.asarray(target["goal"]).tolist() for target in a_targets],
            }
        )
        print(f"[physical {index:02d}/40] {task}", flush=True)

    frame = pd.DataFrame(rows).reset_index(drop=True)
    calibration = frame.index[
        (frame.run == development) & (~frame.failure)
    ].to_numpy()
    labels, thresholds = atlas.classify_behaviors(frame, success_index=calibration)
    return atlas.attach_labels(frame, labels, thresholds), target_audit


def route_cache_map(path: Path) -> dict[str, Path]:
    manifest = json.loads(path.read_text())
    if bool(manifest.get("endpoint_labels_loaded", True)):
        raise RuntimeError(f"route cache is not outcome-blind: {path}")
    result = {str(row["task"]): Path(row["cache"]) for row in manifest["tasks"]}
    if len(result) != 40:
        raise RuntimeError(f"expected 40 task caches in {path}")
    return result


def aggregate_routes(label: str, manifest: Path) -> pd.DataFrame:
    outputs = []
    for task, path in sorted(route_cache_map(manifest).items()):
        with np.load(path, allow_pickle=False) as data:
            frame = pd.DataFrame(
                {
                    "episode": np.asarray(data["episode"], np.int32),
                    "query": np.asarray(data["query"], np.int16),
                    **{
                        signal: np.asarray(data[signal], np.float32)
                        for signal in SIGNALS
                    },
                }
            )
        last_query = frame.groupby("episode")["query"].transform("max").clip(lower=1)
        phase = frame["query"] / last_query
        selected = frame[(phase >= 0.5) & (phase <= 0.9)]
        aggregate = selected.groupby("episode", as_index=False)[list(SIGNALS)].mean()
        aggregate.insert(0, "task", task)
        aggregate.insert(0, "run", label)
        outputs.append(aggregate)
    return pd.concat(outputs, ignore_index=True)


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(float(x) for x in np.quantile(values[index].mean(axis=1), (0.025, 0.975)))


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return math.nan
    tail = min(positive, negative)
    value = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n
    return min(value, 1.0)


def candidate_route_effects(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260904)
    records = []
    for run, run_frame in frame.groupby("run", sort=False):
        no_event = run_frame[run_frame.failure & (run_frame.first_event == "no_event")]
        for label in atlas.PHYSICAL_LABELS:
            for assignment, column in (
                ("overlapping", f"label_{label}"),
                ("primary", "primary_behavior"),
            ):
                positive_mask = (
                    no_event[column].astype(bool)
                    if assignment == "overlapping"
                    else no_event[column].astype(str).eq(label)
                )
                for signal in SIGNALS:
                    cell_values = []
                    positive_n = negative_n = 0
                    for _, cell in no_event.groupby(["task", "init_state_id"]):
                        local_positive = (
                            cell[column].astype(bool).to_numpy()
                            if assignment == "overlapping"
                            else cell[column].astype(str).eq(label).to_numpy()
                        )
                        values = cell[signal].to_numpy(np.float64)
                        good = np.isfinite(values)
                        positive = values[local_positive & good]
                        negative = values[(~local_positive) & good]
                        if len(positive) and len(negative):
                            cell_values.append(pair_auc(positive, negative) - 0.5)
                            positive_n += len(positive)
                            negative_n += len(negative)
                    if not cell_values:
                        continue
                    values = np.asarray(cell_values, np.float64)
                    low, high = bootstrap_ci(values, rng, draws)
                    n_pos = int((values > 0).sum())
                    n_neg = int((values < 0).sum())
                    records.append(
                        {
                            "run": run,
                            "label": label,
                            "assignment": assignment,
                            "signal": signal,
                            "effect_auc_minus_half": float(values.mean()),
                            "ci_low": low,
                            "ci_high": high,
                            "cells": len(values),
                            "positive_episodes_in_cells": positive_n,
                            "negative_episodes_in_cells": negative_n,
                            "positive_cells": n_pos,
                            "negative_cells": n_neg,
                            "sign_p": exact_sign_p(n_pos, n_neg),
                            "total_labelled_no_event_failures": int(positive_mask.sum()),
                        }
                    )
    return pd.DataFrame(records)


def label_inventory(frame: pd.DataFrame) -> dict:
    result = {}
    for run, group in frame.groupby("run", sort=False):
        no_event_failure = group.failure & group.first_event.eq("no_event")
        labels = {}
        for label in atlas.PHYSICAL_LABELS:
            mask = group[f"label_{label}"].astype(bool)
            labels[label] = {
                "all_failures": int((group.failure & mask).sum()),
                "no_event_failures": int((no_event_failure & mask).sum()),
                "successes": int((~group.failure & mask).sum()),
                "success_rate": float(mask[~group.failure].mean()),
                "no_event_primary": int(
                    (no_event_failure & group.primary_behavior.astype(str).eq(label)).sum()
                ),
            }
        result[run] = {
            "no_event_failures": int(no_event_failure.sum()),
            "with_any_candidate": int(
                (no_event_failure & group[[f"label_{x}" for x in atlas.PHYSICAL_LABELS]].any(axis=1)).sum()
            ),
            "with_no_candidate": int(
                (no_event_failure & ~group[[f"label_{x}" for x in atlas.PHYSICAL_LABELS]].any(axis=1)).sum()
            ),
            "labels": labels,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    physical, targets = build_frozen_physical_labels()
    events = pd.read_csv(args.output / "events.csv.gz")[
        ["run", "task_key", "episode", "first_event"]
    ].rename(columns={"task_key": "task"})
    routes = pd.concat(
        [aggregate_routes(label, cfg["manifest"]) for label, cfg in RUNS.items()],
        ignore_index=True,
    )
    frame = physical.merge(events, on=["run", "task", "episode"], validate="one_to_one")
    frame = frame.merge(routes, on=["run", "task", "episode"], validate="one_to_one")
    effects = candidate_route_effects(frame, args.bootstrap)

    frame.to_csv(args.output / "candidate_holdout_episodes.csv.gz", index=False, compression="gzip")
    effects.to_csv(args.output / "candidate_route_effects.csv", index=False)
    (args.output / "candidate_target_audit.json").write_text(json.dumps(targets, indent=2) + "\n")
    summary = {
        "schema": "himoe.trainfree_candidate_holdout.v1",
        "training": False,
        "physical_calibration": "run A successes only",
        "confirmation": "run B; thresholds and task targets frozen from run A",
        "analysis_population": "horizon-censored failures with no loop/static event",
        "route_matching": "task + initial scene; episode means over relative phase 0.5--0.9",
        "inventory": label_inventory(frame),
        "limitations": [
            "candidate physical labels overlap and are query-boundary kinematic proxies",
            "only V/A/recurrence route axes are tested in the holdout",
            "small within-cell counts limit inference for rare primary labels",
            "absence of a distinct signature is evidence against, not proof of absence",
        ],
    }
    (args.output / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["inventory"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
