#!/usr/bin/env python3
"""Cross-run audit of missed-grasp / premature-transport belief mismatch.

The event definition is kinematic and is applied identically to both runs:
the gripper closes near an unfinished target, stays closed while the end
effector departs, and the target remains stationary and separates from the
gripper.  Core distance thresholds predate this audit; target disambiguation
and sustained closure are semantic tightenings added here.  No classifier is
trained. Event-relative MoE features are compared with successful, physically
coupled grasps in the same task, scene, and target.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "VLA_MUI_HUB" / "cache_new" / "HiMoE-VLA"
RESULTS = ROOT / "analysis_trap_taxonomy" / "results"

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

# These thresholds predate this audit.  They come from the existing dense
# missed-grasp analysis and the physical taxonomy's 5 cm goal tolerance.
CLOSE_APERTURE = 0.05
NEAR_TARGET_M = 0.16
UNFINISHED_GOAL_M = 0.05
EEF_DEPARTURE_M = 0.10
TARGET_STATIONARY_M = 0.01
EEF_TARGET_SEPARATION_M = 0.15
COUPLED_TARGET_MOVEMENT_M = 0.03

ROUTE_SIGNALS = (
    "convergence_raw",
    "late_flow_volatility",
    "route_acceleration",
    "state_jump",
    "action_jump",
    "feedback_split_signed",
    "response_raw",
    "recurrence_raw",
)
STRUCTURAL_SIGNALS = (
    "layer5_state_action_gap",
    "front_state_action_gap",
    "back_state_action_gap",
    "front_chunk_jump",
    "back_chunk_jump",
    "front_action_distance_to_healthy",
)
RELATIVE_QUERIES = (-1, 0, 1, 2, 3, 4, 5)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_target_audit(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["task"]): row for row in rows}


def route_cache_map(path: Path) -> dict[str, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if bool(manifest.get("endpoint_labels_loaded", True)):
        raise RuntimeError(f"route cache is not outcome-blind: {path}")
    return {str(row["task"]): Path(row["cache"]) for row in manifest["tasks"]}


def episode_path(client: Path, episode: int) -> Path:
    return client / f"episode_{episode:02d}.npz"


def target_layout(
    audit: dict[str, Any], layout: dict[str, Any]
) -> list[dict[str, Any]]:
    joints = {str(item["joint"]): item for item in layout["joints"]}
    output = []
    for name, goal in zip(audit["target_names"], audit["target_goals"]):
        joint = joints.get(str(name))
        if joint is None:
            raise RuntimeError(f"target {name} is absent from sim layout")
        output.append(
            {
                "name": str(name),
                "lo": int(joint["state_lo"]),
                "goal": np.asarray(goal, dtype=np.float32),
            }
        )
    return output


def closure_events(
    state: np.ndarray, sim: np.ndarray, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aperture = np.abs(state[:, 6:8]).sum(axis=1)
    crossings = np.flatnonzero(
        (aperture[:-1] >= CLOSE_APERTURE) & (aperture[1:] < CLOSE_APERTURE)
    ) + 1
    events = []
    used_targets: set[str] = set()
    for state_query_value in crossings:
        state_query = int(state_query_value)
        nearby = []
        for target in targets:
            lo = int(target["lo"])
            target_xyz = sim[state_query, lo : lo + 3]
            eef_target_distance = float(
                np.linalg.norm(state[state_query, :3] - target_xyz)
            )
            goal_distance = float(np.linalg.norm(target_xyz - target["goal"]))
            if (
                eef_target_distance < NEAR_TARGET_M
                and goal_distance > UNFINISHED_GOAL_M
            ):
                nearby.append((eef_target_distance, target, target_xyz))
        if not nearby:
            continue

        # A single closure is assigned to the nearest unfinished target.  This
        # prevents a nearby already-placed object in a multi-object task from
        # being called a missed grasp.
        distance, target, target_xyz = min(nearby, key=lambda item: item[0])
        lo = int(target["lo"])
        future_eef = state[state_query:, :3]
        future_target = sim[state_query:, lo : lo + 3]
        eef_displacement = np.linalg.norm(
            future_eef - state[state_query, :3], axis=1
        )
        target_displacement = np.linalg.norm(
            future_target - target_xyz, axis=1
        )
        separation = np.linalg.norm(future_eef - future_target, axis=1)
        remained_closed = np.logical_and.accumulate(
            aperture[state_query:] < CLOSE_APERTURE
        )
        mismatch_mask = (
            (eef_displacement >= EEF_DEPARTURE_M)
            & (target_displacement < TARGET_STATIONARY_M)
            & (separation >= EEF_TARGET_SEPARATION_M)
            & remained_closed
        )
        mismatch_offsets = np.flatnonzero(mismatch_mask)
        missed = bool(
            len(mismatch_offsets)
            and float(target_displacement.max()) < TARGET_STATIONARY_M
        )
        coupled = bool(
            float(target_displacement.max()) >= COUPLED_TARGET_MOVEMENT_M
        )
        name = str(target["name"])
        # Repeated aperture crossings for the same target are retries.  Keep
        # them as physical events, then choose one primary event per episode.
        events.append(
            {
                "target": name,
                "closure_action_query": state_query - 1,
                "post_closure_state_query": state_query,
                "mismatch_query": (
                    state_query + int(mismatch_offsets[0])
                    if len(mismatch_offsets)
                    else None
                ),
                "eef_target_distance_at_closure_m": distance,
                "target_goal_distance_at_closure_m": float(
                    np.linalg.norm(target_xyz - target["goal"])
                ),
                "eef_max_displacement_after_m": float(eef_displacement.max()),
                "target_max_displacement_after_m": float(
                    target_displacement.max()
                ),
                "target_max_lift_after_m": float(
                    (future_target[:, 2] - target_xyz[2]).max()
                ),
                "eef_target_max_separation_after_m": float(separation.max()),
                "closed_mismatch_observed": bool(len(mismatch_offsets)),
                "closed_mismatch_within_5q": bool(
                    len(mismatch_offsets) and int(mismatch_offsets[0]) <= 5
                ),
                "missed_grasp_then_departure": missed,
                "coupled_target_motion": coupled,
                "repeated_target_closure": name in used_targets,
            }
        )
        used_targets.add(name)
    return events


def primary_events(events: list[dict[str, Any]], success: bool) -> list[dict[str, Any]]:
    if success:
        # At most one healthy event per target prevents long trajectories with
        # repeated aperture crossings from dominating the control distribution.
        selected = []
        seen: set[str] = set()
        for event in events:
            if event["coupled_target_motion"] and event["target"] not in seen:
                selected.append(event)
                seen.add(str(event["target"]))
        return selected
    missed = [event for event in events if event["missed_grasp_then_departure"]]
    return missed[:1]


def scan_physics(
    target_audit: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    episode_rows = []
    for run_label, cfg in RUNS.items():
        for task_index, (task, audit) in enumerate(sorted(target_audit.items()), 1):
            run = HUB / task / str(cfg["run_id"])
            client = run / "client"
            summaries = json.loads(
                (client / "summaries.json").read_text(encoding="utf-8")
            )
            layout = json.loads(
                (client / "sim_layout.json").read_text(encoding="utf-8")
            )
            targets = target_layout(audit, layout)
            for summary in summaries:
                episode = int(summary["episode_index"])
                with np.load(episode_path(client, episode), allow_pickle=False) as data:
                    state = np.asarray(data["state"], dtype=np.float32)
                    sim = np.asarray(data["sim_state"], dtype=np.float32)
                events = closure_events(state, sim, targets) if targets else []
                success = bool(summary["success"])
                selected = primary_events(events, success)
                for event in events:
                    event_rows.append(
                        {
                            "run": run_label,
                            "task": task,
                            "episode": episode,
                            "init_state_id": int(summary["init_state_id"]),
                            "flow_noise_seed": int(summary["flow_noise_seed"]),
                            "success": success,
                            "selected_for_route_control": event in selected,
                            **event,
                        }
                    )
                missed = [
                    event for event in events if event["missed_grasp_then_departure"]
                ]
                closed_mismatch = [
                    event for event in events if event["closed_mismatch_observed"]
                ]
                closed_mismatch_5q = [
                    event for event in events if event["closed_mismatch_within_5q"]
                ]
                episode_rows.append(
                    {
                        "run": run_label,
                        "task": task,
                        "episode": episode,
                        "init_state_id": int(summary["init_state_id"]),
                        "flow_noise_seed": int(summary["flow_noise_seed"]),
                        "success": success,
                        "failure": not success,
                        "near_unfinished_closure_count": len(events),
                        "belief_mismatch": bool(missed),
                        "belief_mismatch_count": len(missed),
                        "closed_mismatch_candidate": bool(closed_mismatch),
                        "closed_mismatch_within_5q": bool(closed_mismatch_5q),
                        "first_belief_mismatch_query": (
                            int(missed[0]["mismatch_query"]) if missed else -1
                        ),
                        "first_belief_mismatch_target": (
                            str(missed[0]["target"]) if missed else ""
                        ),
                    }
                )
            print(
                f"[physics {run_label} {task_index:02d}/40] {task}", flush=True
            )
    return pd.DataFrame(event_rows), pd.DataFrame(episode_rows)


def attach_macro_labels(episodes: pd.DataFrame, path: Path) -> pd.DataFrame:
    macro = pd.read_csv(path)
    columns = [
        "run",
        "task",
        "episode",
        "first_event",
        "primary_behavior",
        "behavior_signature",
    ]
    return episodes.merge(
        macro[columns], on=["run", "task", "episode"], how="left", validate="one_to_one"
    )


def load_cached_route_rows(selected: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for run_label, cfg in RUNS.items():
        caches = route_cache_map(Path(cfg["manifest"]))
        run_events = selected[selected["run"] == run_label]
        for task, local_events in run_events.groupby("task", sort=True):
            with np.load(caches[str(task)], allow_pickle=False) as data:
                route = pd.DataFrame(
                    {
                        "episode": np.asarray(data["episode"], dtype=np.int32),
                        "query": np.asarray(data["query"], dtype=np.int16),
                        **{
                            signal: np.asarray(data[signal], dtype=np.float32)
                            for signal in ROUTE_SIGNALS
                        },
                    }
                )
            lookup = route.set_index(["episode", "query"])
            for event in local_events.to_dict("records"):
                anchor = int(event["post_closure_state_query"])
                for relative in RELATIVE_QUERIES:
                    key = (int(event["episode"]), anchor + relative)
                    if key not in lookup.index:
                        continue
                    values = lookup.loc[key]
                    if isinstance(values, pd.DataFrame):
                        raise RuntimeError(f"duplicate route row: {task} {key}")
                    outputs.append(
                        {
                            "run": run_label,
                            "task": str(task),
                            "episode": int(event["episode"]),
                            "init_state_id": int(event["init_state_id"]),
                            "target": str(event["target"]),
                            "event_type": str(event["event_type"]),
                            "relative_query": relative,
                            **{signal: float(values[signal]) for signal in ROUTE_SIGNALS},
                        }
                    )
    return pd.DataFrame(outputs)


def load_route_rows(events: pd.DataFrame) -> pd.DataFrame:
    selected = events[
        events["selected_for_route_control"]
        & (
            events["missed_grasp_then_departure"]
            | (events["success"] & events["coupled_target_motion"])
        )
    ].copy()
    selected["event_type"] = np.where(
        selected.missed_grasp_then_departure,
        "belief_mismatch",
        "successful_coupled_grasp",
    )
    return load_cached_route_rows(selected)


def load_other_failure_route_rows(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["run", "task", "episode"]
    episode_has_mismatch = (
        events.groupby(keys).missed_grasp_then_departure.any().rename("episode_has_mismatch")
    )
    annotated = events.merge(
        episode_has_mismatch.reset_index(),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    positive = annotated[
        annotated.selected_for_route_control & annotated.missed_grasp_then_departure
    ].copy()
    positive["event_type"] = "belief_mismatch"
    other = (
        annotated[(~annotated.success) & (~annotated.episode_has_mismatch)]
        .sort_values([*keys, "post_closure_state_query"])
        .groupby(keys, as_index=False)
        .head(1)
        .copy()
    )
    other["event_type"] = "other_failure_closure"
    return load_cached_route_rows(pd.concat([positive, other], ignore_index=True))


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(probabilities, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = normalize(left)
    q = normalize(right)
    return np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum(axis=-1))


def state_action_gap(route: np.ndarray, layer_indices: tuple[int, ...]) -> float:
    p = normalize(route[list(layer_indices)])
    state = p[:, :, 0]
    action = normalize(p[:, :, 1:11].mean(axis=2))
    return float(hellinger(state, action).mean())


def chunk_jump(
    current: np.ndarray, previous: np.ndarray, layer_indices: tuple[int, ...]
) -> float:
    return float(
        hellinger(
            current[list(layer_indices), :, 1:11],
            previous[list(layer_indices), :, 1:11],
        ).mean()
    )


def front_action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(hellinger(left[:4, :, 1:11], right[:4, :, 1:11]).mean())


def matched_route_events(events: pd.DataFrame) -> pd.DataFrame:
    selected = events[
        events["selected_for_route_control"]
        & (
            events["missed_grasp_then_departure"]
            | (events["success"] & events["coupled_target_motion"])
        )
    ].copy()
    selected["event_type"] = np.where(
        selected.missed_grasp_then_departure,
        "belief_mismatch",
        "successful_coupled_grasp",
    )
    keys = ["run", "task", "init_state_id", "target"]
    mixed = (
        selected.groupby(keys).event_type.nunique().loc[lambda value: value == 2]
    )
    return selected.merge(
        mixed.rename("event_type_count").reset_index(),
        on=keys,
        how="inner",
        validate="many_to_one",
    )


def load_structural_route_rows(events: pd.DataFrame) -> pd.DataFrame:
    selected = matched_route_events(events)
    records: list[dict[str, Any]] = []
    route_values: list[np.ndarray] = []
    for run_label, cfg in RUNS.items():
        run_events = selected[selected.run == run_label]
        for task, local_events in run_events.groupby("task", sort=True):
            run = HUB / str(task) / str(cfg["run_id"])
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
                    if query <= 0 or query >= len(rows):
                        continue
                    current_row = int(rows[query])
                    previous_row = int(rows[query - 1])
                    needed.update((current_row, previous_row))
                    requests.append((event, relative, current_row, previous_row))
            ordered = np.asarray(sorted(needed), dtype=np.int64)
            loaded = np.asarray(
                store["hb_router_probs"].oindex[ordered, :, :, :, :],
                dtype=np.float32,
            )
            by_row = {int(row): normalize(loaded[index]) for index, row in enumerate(ordered)}
            for event, relative, current_row, previous_row in requests:
                current = by_row[current_row]
                previous = by_row[previous_row]
                records.append(
                    {
                        "run": run_label,
                        "task": str(task),
                        "episode": int(event["episode"]),
                        "init_state_id": int(event["init_state_id"]),
                        "target": str(event["target"]),
                        "event_type": str(event["event_type"]),
                        "relative_query": int(relative),
                        "layer5_state_action_gap": state_action_gap(current, (3,)),
                        "front_state_action_gap": state_action_gap(current, (0, 1, 2, 3)),
                        "back_state_action_gap": state_action_gap(current, (4, 5, 6, 7)),
                        "front_chunk_jump": chunk_jump(current, previous, (0, 1, 2, 3)),
                        "back_chunk_jump": chunk_jump(current, previous, (4, 5, 6, 7)),
                    }
                )
                route_values.append(current)
            print(f"[structural {run_label}] {task}", flush=True)

    # Compare action-side routing with a healthy event template constructed
    # only from successful controls in the same scene and target.  Successful
    # rows use leave-one-out templates, so they are not compared with themselves.
    keys = ["run", "task", "init_state_id", "target", "relative_query"]
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        key = tuple(record[name] for name in keys)
        grouped.setdefault(key, []).append(index)
    for indices in grouped.values():
        controls = [
            index
            for index in indices
            if records[index]["event_type"] == "successful_coupled_grasp"
        ]
        if not controls:
            continue
        control_stack = np.stack([route_values[index] for index in controls])
        center = control_stack.mean(axis=0)
        for index in indices:
            if index in controls and len(controls) > 1:
                local = [value for value in controls if value != index]
                reference = np.stack([route_values[value] for value in local]).mean(axis=0)
            elif index in controls:
                records[index]["front_action_distance_to_healthy"] = math.nan
                continue
            else:
                reference = center
            records[index]["front_action_distance_to_healthy"] = front_action_distance(
                route_values[index], reference
            )
    return pd.DataFrame(records)


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


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


def route_effects(
    rows: pd.DataFrame,
    draws: int,
    signals: tuple[str, ...] = ROUTE_SIGNALS,
    negative_type: str = "successful_coupled_grasp",
) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(20260904)
    for (run, relative), frame in rows.groupby(["run", "relative_query"]):
        cells = list(frame.groupby(["task", "init_state_id", "target"]))
        for signal in signals:
            aucs = []
            positive_n = negative_n = 0
            for _key, cell in cells:
                positive = cell.loc[
                    cell.event_type == "belief_mismatch", signal
                ].to_numpy(np.float64)
                negative = cell.loc[
                    cell.event_type == negative_type, signal
                ].to_numpy(np.float64)
                positive = positive[np.isfinite(positive)]
                negative = negative[np.isfinite(negative)]
                if not len(positive) or not len(negative):
                    continue
                aucs.append(pair_auc(positive, negative))
                positive_n += len(positive)
                negative_n += len(negative)
            if not aucs:
                continue
            values = np.asarray(aucs, dtype=np.float64)
            low, high = bootstrap_ci(values, rng, draws)
            records.append(
                {
                    "run": run,
                    "comparison": f"belief_mismatch_vs_{negative_type}",
                    "relative_query": int(relative),
                    "signal": signal,
                    "raw_auc": float(values.mean()),
                    "raw_auc_ci_low": low,
                    "raw_auc_ci_high": high,
                    "mixed_cells": len(values),
                    "positive_events_in_cells": positive_n,
                    "control_events_in_cells": negative_n,
                }
            )
    effects = pd.DataFrame(records)
    development = effects[effects.run == "seed1000_1007"].copy()
    direction = {
        (int(row.relative_query), str(row.signal)): (
            "high" if float(row.raw_auc) >= 0.5 else "low"
        )
        for row in development.itertuples()
    }
    effects["direction_frozen_from_A"] = [
        direction.get((int(row.relative_query), str(row.signal)), "")
        for row in effects.itertuples()
    ]
    effects["oriented_auc"] = np.where(
        effects.direction_frozen_from_A == "high",
        effects.raw_auc,
        1.0 - effects.raw_auc,
    )
    return effects


def summarize(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    effects: pd.DataFrame,
    structural_effects: pd.DataFrame,
    other_failure_effects: pd.DataFrame,
) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for run, frame in episodes.groupby("run", sort=False):
        failure = frame.failure
        mismatch = frame.belief_mismatch
        mismatch_failures = frame[failure & mismatch]
        counts[str(run)] = {
            "episodes": len(frame),
            "failures": int(failure.sum()),
            "belief_mismatch_failures": int((failure & mismatch).sum()),
            "fraction_of_failures": float((failure & mismatch).sum() / failure.sum()),
            "belief_mismatch_successes": int(((~failure) & mismatch).sum()),
            "closed_mismatch_candidates": {
                "failures": int((failure & frame.closed_mismatch_candidate).sum()),
                "successes": int(((~failure) & frame.closed_mismatch_candidate).sum()),
            },
            "closed_mismatch_within_5q": {
                "failures": int((failure & frame.closed_mismatch_within_5q).sum()),
                "successes": int(((~failure) & frame.closed_mismatch_within_5q).sum()),
            },
            "tasks_with_belief_mismatch": int(mismatch_failures.task.nunique()),
            "macro_overlap": {
                str(key): int(value)
                for key, value in Counter(mismatch_failures.first_event).items()
            },
        }

    best = []
    for relative in RELATIVE_QUERIES:
        a = effects[
            (effects.run == "seed1000_1007")
            & (effects.relative_query == relative)
        ]
        b = effects[
            (effects.run == "seed1008_1015")
            & (effects.relative_query == relative)
        ]
        joined = a.merge(b, on=["relative_query", "signal"], suffixes=("_A", "_B"))
        if len(joined):
            joined["replicated_min_auc"] = joined[
                ["oriented_auc_A", "oriented_auc_B"]
            ].min(axis=1)
            row = joined.sort_values("replicated_min_auc", ascending=False).iloc[0]
            best.append(
                {
                    "relative_query": relative,
                    "signal": str(row.signal),
                    "direction_frozen_from_A": str(row.direction_frozen_from_A_A),
                    "oriented_auc_A": float(row.oriented_auc_A),
                    "oriented_auc_B": float(row.oriented_auc_B),
                    "mixed_cells_A": int(row.mixed_cells_A),
                    "mixed_cells_B": int(row.mixed_cells_B),
                }
            )
    return {
        "schema": "himoe.belief_mismatch_cross_run_audit.v1",
        "training": False,
        "event_definition": {
            "close_aperture_below": CLOSE_APERTURE,
            "near_unfinished_target_m": NEAR_TARGET_M,
            "unfinished_goal_distance_above_m": UNFINISHED_GOAL_M,
            "eef_departure_at_least_m": EEF_DEPARTURE_M,
            "target_remains_below_m": TARGET_STATIONARY_M,
            "eef_target_separation_at_least_m": EEF_TARGET_SEPARATION_M,
            "target_assignment": "nearest unfinished target",
            "gripper_requirement": (
                "aperture remains below threshold from closure through departure"
            ),
            "contact_or_force_used": False,
        },
        "counts": counts,
        "selected_route_comparison": (
            "belief mismatch vs successful coupled grasp, matched within "
            "task + init_state + target"
        ),
        "best_single_cached_signal_by_relative_query": best,
        "structural_route_effects": [
            {
                "run": str(row.run),
                "relative_query": int(row.relative_query),
                "signal": str(row.signal),
                "direction_frozen_from_A": str(row.direction_frozen_from_A),
                "oriented_auc": float(row.oriented_auc),
                "raw_auc": float(row.raw_auc),
                "mixed_cells": int(row.mixed_cells),
            }
            for row in structural_effects.itertuples()
            if int(row.relative_query) in (2, 3)
        ],
        "other_failure_closure_comparison": [
            {
                "run": str(row.run),
                "relative_query": int(row.relative_query),
                "signal": str(row.signal),
                "direction_frozen_from_A": str(row.direction_frozen_from_A),
                "oriented_auc": float(row.oriented_auc),
                "raw_auc": float(row.raw_auc),
                "mixed_cells": int(row.mixed_cells),
            }
            for row in other_failure_effects.itertuples()
            if int(row.relative_query) in (2, 3)
            and str(row.signal) in ("route_acceleration", "late_flow_volatility")
        ],
        "interpretation": [
            "This confirms a semantic/physical trap beyond endpoint timeout.",
            "Overlap with loop/static is allowed: belief mismatch is an event cause, while loop/static are trajectory dynamics.",
            "Cached scalar routing features test transfer, but do not by themselves prove a latent belief variable.",
        ],
        "event_rows": len(events),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-audit",
        type=Path,
        default=RESULTS / "candidate_target_audit.json",
    )
    parser.add_argument(
        "--macro-labels",
        type=Path,
        default=RESULTS / "candidate_holdout_episodes.csv.gz",
    )
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_audit = load_target_audit(args.target_audit.resolve())
    events, episodes = scan_physics(target_audit)
    episodes = attach_macro_labels(episodes, args.macro_labels.resolve())
    route_rows = load_route_rows(events)
    effects = route_effects(route_rows, args.bootstrap)
    other_failure_rows = load_other_failure_route_rows(events)
    other_failure_effects = route_effects(
        other_failure_rows,
        args.bootstrap,
        negative_type="other_failure_closure",
    )
    structural_rows = load_structural_route_rows(events)
    structural_effects = route_effects(
        structural_rows, args.bootstrap, STRUCTURAL_SIGNALS
    )
    summary = summarize(
        events,
        episodes,
        effects,
        structural_effects,
        other_failure_effects,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "belief_mismatch_events.csv.gz", index=False)
    episodes.to_csv(output / "belief_mismatch_episodes.csv.gz", index=False)
    route_rows.to_csv(output / "belief_mismatch_route_rows.csv.gz", index=False)
    effects.to_csv(output / "belief_mismatch_route_effects.csv", index=False)
    other_failure_rows.to_csv(
        output / "belief_mismatch_other_failure_route_rows.csv.gz", index=False
    )
    other_failure_effects.to_csv(
        output / "belief_mismatch_other_failure_route_effects.csv", index=False
    )
    structural_rows.to_csv(
        output / "belief_mismatch_structural_route_rows.csv.gz", index=False
    )
    structural_effects.to_csv(
        output / "belief_mismatch_structural_route_effects.csv", index=False
    )
    (output / "belief_mismatch_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
