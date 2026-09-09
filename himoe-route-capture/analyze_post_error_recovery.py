#!/usr/bin/env python3
"""Inventory post-error recovery events before fitting a prognosis model.

The Hub records one simulator state immediately before each action chunk.  It
does not contain dense contact labels, so this script deliberately uses a
conservative kinematic proxy for a transport loss: an object first moves with a
closed gripper and the end effector, then is lower and separated at the next
query even though the preceding chunk still commanded gripper closure.

The primary output is a feasibility gate.  A physical -> route -> action ->
hidden prognosis model is not fit unless the event cohort has within-initial-
state outcome support and enough independent clusters.  Recurrence scalars are
still extracted descriptively so that a failed gate is informative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import numpy as np
import zarr
from scipy.stats import spearmanr


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/post-error-recovery-hub"

LONG_HOLD_DISTANCE_M = 0.115
OTHER_HOLD_DISTANCE_M = 0.085
LONG_GRASP_NEAR_M = 0.13
OTHER_GRASP_NEAR_M = 0.10

# Frozen from the first argument(s) of each transport predicate in the LIBERO
# BDDL goal.  Receptacles and articulated-only tasks are intentionally absent.
TASK_TRANSPORT_OBJECTS = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": (),
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": (
        "akita_black_bowl_1",
    ),
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (
        "moka_pot_1",
        "moka_pot_2",
    ),
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": (
        "akita_black_bowl_1",
    ),
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (
        "akita_black_bowl_1",
    ),
}


@dataclass(frozen=True)
class EventRule:
    hold_gap_max_m: float = 0.025
    continue_gap_max_m: float = 0.028
    min_object_step_m: float = 0.005
    max_motion_residual_m: float = 0.04
    min_displacement_m: float = 0.012
    min_drop_m: float = 0.012
    min_close_command: float = 0.5


MAIN_RULE = EventRule()


@dataclass(frozen=True)
class Target:
    name: str
    lo: int
    hi: int


@dataclass
class Event:
    task: str
    episode: int
    init_state_id: int
    flow_noise_seed: int
    target: str
    bout_start: int
    held_end: int
    drop_query: int
    failed_chunk_anchor: int
    grasp_anchor: int
    anchor_confidence: str
    inference_calls: int
    remaining_queries: int
    terminal_success: bool
    future_label: str
    later_rehold_query: int | None
    future_proxy_rehold: bool
    hold_distance_threshold_m: float
    object_drop_m: float
    separation_increase_m: float
    max_lift_in_bout_m: float
    object_distance_at_post_m: float
    eef_distance_to_grasp_anchor_m: float
    target_distance_to_grasp_anchor_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def episode_path(client: pathlib.Path, episode: int) -> pathlib.Path:
    return client / ("episode_%02d.npz" % episode)


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = []
    for path in sorted(cache_root.glob("libero_*/*/right-16x32/client/summaries.json")):
        run = path.parents[1]
        if (run / "server/routes.zarr").exists() and (
            run / "client/sim_layout.json"
        ).exists():
            runs.append(run)
    if not runs:
        raise RuntimeError("no complete right-16x32 LIBERO runs found")
    return runs


def task_key(run: pathlib.Path, cache_root: pathlib.Path) -> str:
    return str(run.relative_to(cache_root).parent)


def load_client_task(
    run: pathlib.Path,
) -> tuple[list[dict], dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if [int(row["episode_index"]) for row in summaries] != list(range(len(summaries))):
        raise ValueError("episode_index is not contiguous")
    layout = json.loads((run / "client/sim_layout.json").read_text())
    sims: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(episode_path(run / "client", episode), allow_pickle=False) as archive:
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
            state = np.asarray(archive["state"], dtype=np.float32)
            action = np.asarray(archive["actions"], dtype=np.float32)
        expected = int(row["inference_calls"])
        if len(sim) != expected or len(state) != expected or len(action) != expected:
            raise ValueError("client arrays do not align with summaries.json")
        if state.shape[1:] != (8,) or action.shape[1:] != (10, 7):
            raise ValueError("unexpected state/action shape")
        sims.append(sim)
        states.append(state)
        actions.append(action)
    return summaries, layout, sims, states, actions


def identify_targets(task: str, layout: dict) -> list[Target]:
    if task not in TASK_TRANSPORT_OBJECTS:
        raise ValueError(f"task has no frozen BDDL target mapping: {task}")
    wanted = set(TASK_TRANSPORT_OBJECTS[task])
    targets: list[Target] = []
    for joint in layout["joints"]:
        lo, hi = int(joint["state_lo"]), int(joint["state_hi"])
        if bool(joint["is_robot"]) or hi - lo != 7:
            continue
        name = str(joint["joint"])
        object_name = name.removesuffix("_joint0")
        if object_name in wanted:
            targets.append(Target(name, lo, lo + 3))
    found = {target.name.removesuffix("_joint0") for target in targets}
    if found != wanted:
        raise ValueError(f"BDDL target/layout mismatch for {task}: wanted={wanted}, found={found}")
    return targets


def evidence_tape(
    position: np.ndarray,
    eef: np.ndarray,
    gap: np.ndarray,
    hold_distance_m: float,
    rule: EventRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = np.linalg.norm(position - eef, axis=1)
    object_step = np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    motion_residual = np.r_[
        0.0,
        np.linalg.norm(np.diff(position, axis=0) - np.diff(eef, axis=0), axis=1),
    ]
    displacement = np.linalg.norm(position - position[:1], axis=1)
    evidence = (
        (distance < hold_distance_m)
        & (gap < rule.hold_gap_max_m)
        & (object_step > rule.min_object_step_m)
        & (motion_residual < rule.max_motion_residual_m)
        & (displacement > rule.min_displacement_m)
    )
    return evidence, distance, object_step, motion_residual


def detect_target_events(
    position: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
    hold_distance_m: float,
    rule: EventRule = MAIN_RULE,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    eef = np.asarray(state[:, :3], dtype=np.float32)
    gap = np.mean(np.abs(state[:, 6:8]), axis=1)
    close_command = np.mean(actions[:, :, 6], axis=1)
    evidence, distance, _object_step, _motion_residual = evidence_tape(
        position, eef, gap, hold_distance_m, rule
    )
    events: list[dict[str, Any]] = []
    active = False
    start = -1
    for query in range(len(position)):
        if not active:
            if evidence[query]:
                active, start = True, query
            continue
        if distance[query] < hold_distance_m and gap[query] < rule.continue_gap_max_m:
            continue
        held_end = query - 1
        post = held_end + 1
        if (
            post < len(position)
            and gap[post] < rule.hold_gap_max_m
            and close_command[held_end] > rule.min_close_command
            and position[post, 2] - position[held_end, 2] < -rule.min_drop_m
            and distance[post] > hold_distance_m
        ):
            events.append(
                {
                    "bout_start": int(start),
                    "held_end": int(held_end),
                    "drop_query": int(post),
                    "object_drop_m": float(position[held_end, 2] - position[post, 2]),
                    "separation_increase_m": float(distance[post] - distance[held_end]),
                    "max_lift_in_bout_m": float(
                        position[start : held_end + 1, 2].max() - position[0, 2]
                    ),
                    "object_distance_at_post_m": float(distance[post]),
                }
            )
        active = False
        if evidence[query]:
            active, start = True, query
    return events, evidence, distance


def find_grasp_anchor(
    position: np.ndarray,
    state: np.ndarray,
    bout_start: int,
    near_distance_m: float,
) -> tuple[int, str]:
    gap = np.mean(np.abs(state[:, 6:8]), axis=1)
    distance = np.linalg.norm(position - state[:, :3], axis=1)
    lo = max(0, bout_start - 4)
    candidates = np.flatnonzero(
        (np.arange(len(position)) >= lo)
        & (np.arange(len(position)) < bout_start)
        & (gap >= 0.030)
        & (distance < near_distance_m)
    )
    if len(candidates):
        return int(candidates[-1]), "high"
    return max(0, int(bout_start) - 1), "fallback"


def make_rule_variant(rule: EventRule, name: str) -> EventRule:
    if name == "main":
        return rule
    if name not in {"strict", "loose"}:
        raise ValueError(name)
    max_scale = 0.8 if name == "strict" else 1.2
    min_scale = 1.2 if name == "strict" else 0.8
    return replace(
        rule,
        hold_gap_max_m=rule.hold_gap_max_m * max_scale,
        continue_gap_max_m=rule.continue_gap_max_m * max_scale,
        min_object_step_m=rule.min_object_step_m * min_scale,
        max_motion_residual_m=rule.max_motion_residual_m * max_scale,
        min_displacement_m=rule.min_displacement_m * min_scale,
        min_drop_m=rule.min_drop_m * min_scale,
        min_close_command=rule.min_close_command * min_scale,
    )


def inventory_task(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    rule: EventRule = MAIN_RULE,
    distance_scale: float = 1.0,
) -> tuple[list[Event], dict[str, Any]]:
    summaries, layout, sims, states, actions = load_client_task(run)
    key = task_key(run, cache_root)
    targets = identify_targets(key, layout)
    is_long = key.startswith("libero_long/")
    hold_distance = (
        LONG_HOLD_DISTANCE_M if is_long else OTHER_HOLD_DISTANCE_M
    ) * distance_scale
    near_distance = (
        LONG_GRASP_NEAR_M if is_long else OTHER_GRASP_NEAR_M
    ) * distance_scale
    events: list[Event] = []
    for episode, (row, sim, state, action) in enumerate(
        zip(summaries, sims, states, actions)
    ):
        candidates: list[tuple[int, str, Target, dict, np.ndarray, np.ndarray]] = []
        for target in targets:
            position = sim[:, target.lo : target.hi]
            target_events, evidence, distance = detect_target_events(
                position, state, action, hold_distance, rule
            )
            for detected in target_events:
                candidates.append(
                    (
                        int(detected["drop_query"]),
                        target.name,
                        target,
                        detected,
                        evidence,
                        distance,
                    )
                )
        if not candidates:
            continue
        _post, _name, target, detected, evidence, distance = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        position = sim[:, target.lo : target.hi]
        post = int(detected["drop_query"])
        start = int(detected["bout_start"])
        grasp_anchor, confidence = find_grasp_anchor(
            position, state, start, near_distance
        )
        later = np.flatnonzero(evidence[post + 1 :])
        later_rehold = int(post + 1 + later[0]) if len(later) else None
        success = bool(row["success"])
        events.append(
            Event(
                task=key,
                episode=episode,
                init_state_id=int(row["init_state_id"]),
                flow_noise_seed=int(row["flow_noise_seed"]),
                target=target.name,
                bout_start=start,
                held_end=int(detected["held_end"]),
                drop_query=post,
                failed_chunk_anchor=post - 1,
                grasp_anchor=grasp_anchor,
                anchor_confidence=confidence,
                inference_calls=int(row["inference_calls"]),
                remaining_queries=int(row["inference_calls"]) - post - 1,
                terminal_success=success,
                future_label=(
                    "terminal_success_after_proxy"
                    if success
                    else "terminal_failure_after_proxy"
                ),
                later_rehold_query=later_rehold,
                future_proxy_rehold=later_rehold is not None,
                hold_distance_threshold_m=hold_distance,
                object_drop_m=float(detected["object_drop_m"]),
                separation_increase_m=float(detected["separation_increase_m"]),
                max_lift_in_bout_m=float(detected["max_lift_in_bout_m"]),
                object_distance_at_post_m=float(detected["object_distance_at_post_m"]),
                eef_distance_to_grasp_anchor_m=float(
                    np.linalg.norm(state[post, :3] - state[grasp_anchor, :3])
                ),
                target_distance_to_grasp_anchor_m=float(
                    np.linalg.norm(position[post] - position[grasp_anchor])
                ),
            )
        )
    return events, {
        "task": key,
        "episodes": len(summaries),
        "successes": int(sum(bool(row["success"]) for row in summaries)),
        "targets": [asdict(target) for target in targets],
        "target_selection": "frozen BDDL transport-predicate first argument",
        "events": len(events),
        "event_successes": int(sum(event.terminal_success for event in events)),
    }


def normalized_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("router probability vector has zero mass")
    return values / mass


def route_hellinger_similarity(left: np.ndarray, right: np.ndarray, tokens: slice) -> float:
    left = normalized_probabilities(left)[..., tokens, :]
    right = normalized_probabilities(right)[..., tokens, :]
    distance = np.sqrt(
        0.5 * np.sum(np.square(np.sqrt(left) - np.sqrt(right)), axis=-1)
    )
    return float(1.0 - distance.mean())


def top4_jaccard_similarity(left: np.ndarray, right: np.ndarray, tokens: slice) -> float:
    left = np.asarray(left)[..., tokens, :]
    right = np.asarray(right)[..., tokens, :]
    intersection = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(axis=-1)
    union = 8 - intersection
    return float(np.mean(intersection / np.maximum(union, 1)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(float(denominator), 1e-12))


def hidden_cosine_similarity(left: np.ndarray, right: np.ndarray, tokens: slice) -> float:
    left = np.asarray(left, dtype=np.float32)[..., tokens, :]
    right = np.asarray(right, dtype=np.float32)[..., tokens, :]
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def validate_server_alignment(run: pathlib.Path, summaries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    control_step = np.asarray(group["control_step"][:], dtype=np.int64)
    if len(episode_id) != int(lengths.sum()):
        raise ValueError("server rows do not match summary lengths")
    if not np.array_equal(control_step, np.arange(len(control_step))):
        raise ValueError("control_step is not globally contiguous")
    for episode, (offset, length) in enumerate(zip(offsets, lengths)):
        if not np.all(episode_id[offset : offset + length] == episode):
            raise ValueError("episode_id alignment failed")
    return lengths, offsets


def extract_recurrence(
    run: pathlib.Path,
    events: list[Event],
    skip_hidden: bool,
) -> list[dict[str, Any]]:
    if not events:
        return []
    summaries, _layout, sims, states, actions = load_client_task(run)
    _lengths, offsets = validate_server_alignment(run, summaries)
    routes = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden_path = run / "server/hidden.zarr"
    hidden = None if skip_hidden else zarr.open_group(str(hidden_path), mode="r")
    if hidden is not None:
        if not np.array_equal(routes["episode_id"][:], hidden["episode_id"][:]):
            raise ValueError("route and hidden episode axes differ")
        if not np.array_equal(routes["control_step"][:], hidden["control_step"][:]):
            raise ValueError("route and hidden control axes differ")

    rows = []
    for event in events:
        post = int(event.drop_query)
        for anchor_kind, anchor in (
            ("failed_chunk", int(event.failed_chunk_anchor)),
            ("grasp_basin", int(event.grasp_anchor)),
        ):
            flat = offsets[event.episode] + np.asarray([post, anchor], dtype=np.int64)
            probabilities = np.asarray(
                routes["hb_router_probs"].oindex[flat, :, :, :, :], dtype=np.float32
            )
            expert_ids = np.asarray(
                routes["hb_expert_ids"].oindex[flat, :, :, :, :], dtype=np.uint8
            )
            left_action = actions[event.episode][post]
            right_action = actions[event.episode][anchor]
            record: dict[str, Any] = {
                **asdict(event),
                "anchor_kind": anchor_kind,
                "anchor_query": anchor,
                "route_action_hellinger_similarity": route_hellinger_similarity(
                    probabilities[0], probabilities[1], slice(1, 11)
                ),
                "route_state_hellinger_similarity": route_hellinger_similarity(
                    probabilities[0], probabilities[1], slice(0, 1)
                ),
                "route_action_top4_jaccard": top4_jaccard_similarity(
                    expert_ids[0], expert_ids[1], slice(1, 11)
                ),
                "route_state_top4_jaccard": top4_jaccard_similarity(
                    expert_ids[0], expert_ids[1], slice(0, 1)
                ),
                "action_cosine_similarity": cosine_similarity(left_action, right_action),
                "action_raw_rms_distance": float(
                    np.sqrt(np.mean(np.square(left_action - right_action, dtype=np.float32)))
                ),
                "gripper_command_agreement": float(
                    np.mean(np.signbit(left_action[:, 6]) == np.signbit(right_action[:, 6]))
                ),
                "proprio_raw_rms_distance": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                states[event.episode][post] - states[event.episode][anchor],
                                dtype=np.float32,
                            )
                        )
                    )
                ),
                "sim_state_raw_rms_distance": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                sims[event.episode][post, 1:] - sims[event.episode][anchor, 1:],
                                dtype=np.float32,
                            )
                        )
                    )
                ),
                "hidden_action_cosine_similarity": None,
                "hidden_state_cosine_similarity": None,
            }
            if hidden is not None:
                hidden_values = np.asarray(
                    hidden["hb_hidden"].oindex[flat, :, :, :, :], dtype=np.float32
                )
                record["hidden_action_cosine_similarity"] = hidden_cosine_similarity(
                    hidden_values[0], hidden_values[1], slice(1, 11)
                )
                record["hidden_state_cosine_similarity"] = hidden_cosine_similarity(
                    hidden_values[0], hidden_values[1], slice(0, 1)
                )
            rows.append(record)
    return rows


def event_counts(events: Iterable[Event]) -> dict[str, Any]:
    rows = list(events)
    clusters: dict[tuple[str, int], list[Event]] = {}
    for event in rows:
        clusters.setdefault((event.task, event.init_state_id), []).append(event)
    mixed = {
        key: values
        for key, values in clusters.items()
        if {event.terminal_success for event in values} == {False, True}
    }
    tasks_with_both = {
        task
        for task in {event.task for event in rows}
        if {event.terminal_success for event in rows if event.task == task} == {False, True}
    }
    return {
        "events": len(rows),
        "terminal_success_after_proxy": int(sum(event.terminal_success for event in rows)),
        "terminal_failure_after_proxy": int(sum(not event.terminal_success for event in rows)),
        "future_proxy_rehold": int(sum(event.future_proxy_rehold for event in rows)),
        "tasks": len({event.task for event in rows}),
        "tasks_with_both_outcomes": len(tasks_with_both),
        "task_initial_state_clusters": len(clusters),
        "mixed_task_initial_state_clusters": len(mixed),
        "conditional_pairs": int(
            sum(
                sum(event.terminal_success for event in values)
                * sum(not event.terminal_success for event in values)
                for values in mixed.values()
            )
        ),
        "mixed_clusters": [
            {
                "task": key[0],
                "init_state_id": key[1],
                "terminal_success_after_proxy": int(
                    sum(event.terminal_success for event in values)
                ),
                "terminal_failure_after_proxy": int(
                    sum(not event.terminal_success for event in values)
                ),
            }
            for key, values in sorted(mixed.items())
        ],
    }


def double_holdout_feasibility(events: list[Event]) -> dict[str, Any]:
    if not events:
        return {
            "folds": 16,
            "evaluable_folds": 0,
            "all_folds_evaluable": False,
            "all_events_tested_once": False,
            "details": [],
        }
    cluster_keys = sorted({(event.task, event.init_state_id) for event in events})
    cluster_rank = {key: index for index, key in enumerate(cluster_keys)}
    seeds = sorted({event.flow_noise_seed for event in events})
    seed_rank = {seed: index for index, seed in enumerate(seeds)}
    state_fold = np.asarray(
        [cluster_rank[(event.task, event.init_state_id)] % 4 for event in events]
    )
    seed_fold = np.asarray([seed_rank[event.flow_noise_seed] % 4 for event in events])
    label = np.asarray([event.terminal_success for event in events])
    test_count = np.zeros(len(events), dtype=np.int64)
    details = []
    for held_state in range(4):
        for held_seed in range(4):
            test = (state_fold == held_state) & (seed_fold == held_seed)
            train = (state_fold != held_state) & (seed_fold != held_seed)
            test_count += test
            train_counts = event_counts(
                [event for event, keep in zip(events, train) if keep]
            )
            test_counts = event_counts(
                [event for event, keep in zip(events, test) if keep]
            )
            evaluable = (
                len(np.unique(label[train])) == 2
                and len(np.unique(label[test])) == 2
                and train_counts["conditional_pairs"] > 0
                and test_counts["conditional_pairs"] > 0
            )
            details.append(
                {
                    "state_fold": held_state,
                    "seed_fold": held_seed,
                    "train_n": int(train.sum()),
                    "test_n": int(test.sum()),
                    "train_classes": int(len(np.unique(label[train]))),
                    "test_classes": int(len(np.unique(label[test]))),
                    "train_mixed_clusters": train_counts[
                        "mixed_task_initial_state_clusters"
                    ],
                    "test_mixed_clusters": test_counts[
                        "mixed_task_initial_state_clusters"
                    ],
                    "train_conditional_pairs": train_counts["conditional_pairs"],
                    "test_conditional_pairs": test_counts["conditional_pairs"],
                    "evaluable": bool(evaluable),
                }
            )
    return {
        "folds": 16,
        "evaluable_folds": int(sum(row["evaluable"] for row in details)),
        "all_folds_evaluable": bool(all(row["evaluable"] for row in details)),
        "all_events_tested_once": bool(np.all(test_count == 1)),
        "details": details,
    }


def rank_separation_auc(events: list[Event], field: str) -> float | None:
    positive = np.asarray(
        [float(getattr(event, field)) for event in events if event.terminal_success]
    )
    negative = np.asarray(
        [float(getattr(event, field)) for event in events if not event.terminal_success]
    )
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    auc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))
    return max(auc, 1.0 - auc)


def phase_followup_diagnostics(events: list[Event]) -> dict[str, Any]:
    phase_clusters: dict[tuple[str, int, int], list[Event]] = {}
    for event in events:
        key = (event.task, event.init_state_id, event.drop_query // 5)
        phase_clusters.setdefault(key, []).append(event)
    mixed = [
        values
        for values in phase_clusters.values()
        if {event.terminal_success for event in values} == {False, True}
    ]
    followup = np.asarray([event.remaining_queries for event in events], dtype=np.int64)
    sentinels = {}
    for field in (
        "drop_query",
        "remaining_queries",
        "object_drop_m",
        "separation_increase_m",
        "max_lift_in_bout_m",
        "object_distance_at_post_m",
    ):
        sentinels[field] = rank_separation_auc(events, field)
    return {
        "phase_bin_width_queries": 5,
        "mixed_task_initial_state_phase_clusters": len(mixed),
        "conditional_pairs_within_phase": int(
            sum(
                sum(event.terminal_success for event in values)
                * sum(not event.terminal_success for event in values)
                for values in mixed
            )
        ),
        "minimum_remaining_queries": int(followup.min()) if len(followup) else None,
        "events_with_at_least_5_remaining_queries": int(np.sum(followup >= 5)),
        "all_events_have_at_least_5_remaining_queries": bool(
            len(followup) and np.all(followup >= 5)
        ),
        "univariate_absolute_auc_sentinels": sentinels,
    }


def descriptive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "route_action_hellinger_similarity",
        "route_state_hellinger_similarity",
        "route_action_top4_jaccard",
        "route_state_top4_jaccard",
        "action_cosine_similarity",
        "action_raw_rms_distance",
        "gripper_command_agreement",
        "proprio_raw_rms_distance",
        "sim_state_raw_rms_distance",
        "hidden_action_cosine_similarity",
        "hidden_state_cosine_similarity",
    )
    output: dict[str, Any] = {}
    for anchor in ("failed_chunk", "grasp_basin"):
        subset = [row for row in rows if row["anchor_kind"] == anchor]
        anchor_summary: dict[str, Any] = {}
        for label, positive in (
            ("terminal_success_after_proxy", True),
            ("terminal_failure_after_proxy", False),
        ):
            group = [row for row in subset if bool(row["terminal_success"]) is positive]
            values = {}
            for metric in metrics:
                array = np.asarray(
                    [row[metric] for row in group if row.get(metric) is not None],
                    dtype=np.float64,
                )
                values[metric] = {
                    "n": int(len(array)),
                    "mean": float(array.mean()) if len(array) else None,
                    "median": float(np.median(array)) if len(array) else None,
                }
            anchor_summary[label] = {"n": len(group), "metrics": values}

        correlations = {}
        for other in (
            "action_cosine_similarity",
            "proprio_raw_rms_distance",
            "sim_state_raw_rms_distance",
            "hidden_action_cosine_similarity",
        ):
            pairs = [
                (row["route_action_hellinger_similarity"], row[other])
                for row in subset
                if row.get(other) is not None
            ]
            if len(pairs) >= 3:
                statistic, _pvalue = spearmanr(
                    np.asarray([pair[0] for pair in pairs]),
                    np.asarray([pair[1] for pair in pairs]),
                )
                if np.isfinite(statistic):
                    correlations[other] = {
                        "n": len(pairs),
                        "spearman_r": float(statistic),
                    }
        anchor_summary["route_action_correlations"] = correlations
        output[anchor] = anchor_summary
    return output


def gate_result(
    counts: dict[str, Any], folds: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    requirements = {
        "tasks_with_both_outcomes": 3,
        "mixed_task_initial_state_clusters": 20,
        "mixed_task_initial_state_phase_clusters": 20,
        "events_per_outcome": 40,
        "minimum_followup_queries": 5,
        "all_double_holdout_folds_evaluable": True,
    }
    checks = {
        "tasks_with_both_outcomes": counts["tasks_with_both_outcomes"]
        >= requirements["tasks_with_both_outcomes"],
        "mixed_task_initial_state_clusters": counts["mixed_task_initial_state_clusters"]
        >= requirements["mixed_task_initial_state_clusters"],
        "mixed_task_initial_state_phase_clusters": diagnostics[
            "mixed_task_initial_state_phase_clusters"
        ]
        >= requirements["mixed_task_initial_state_phase_clusters"],
        "terminal_success_after_proxy_events": counts["terminal_success_after_proxy"]
        >= requirements["events_per_outcome"],
        "terminal_failure_after_proxy_events": counts["terminal_failure_after_proxy"]
        >= requirements["events_per_outcome"],
        "minimum_followup_queries": bool(
            diagnostics["all_events_have_at_least_5_remaining_queries"]
        ),
        "all_double_holdout_folds_evaluable": bool(folds["all_folds_evaluable"]),
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "requirements": requirements,
        "checks": checks,
        "model_status": "eligible_not_fitted" if passed else "skipped_gate0",
        "reason": None
        if passed
        else "The event cohort lacks independent within-initial-state outcome support.",
    }


def sensitivity_summary(main: list[Event], variant: list[Event]) -> dict[str, Any]:
    main_by_id = {(event.task, event.episode, event.target): event for event in main}
    variant_by_id = {
        (event.task, event.episode, event.target): event for event in variant
    }
    main_ids = set(main_by_id)
    variant_ids = set(variant_by_id)
    common = main_ids & variant_ids
    union = main_ids | variant_ids
    shifted = [
        {
            "task": key[0],
            "episode": key[1],
            "target": key[2],
            "main_drop_query": main_by_id[key].drop_query,
            "variant_drop_query": variant_by_id[key].drop_query,
        }
        for key in common
        if main_by_id[key].drop_query != variant_by_id[key].drop_query
    ]
    shifts = [
        abs(row["main_drop_query"] - row["variant_drop_query"]) for row in shifted
    ]
    main_success_ids = {
        key for key, event in main_by_id.items() if event.terminal_success
    }
    output = event_counts(variant)
    output["identity_stability"] = {
        "retained_main_events": len(common),
        "removed_main_events": len(main_ids - variant_ids),
        "added_events": len(variant_ids - main_ids),
        "event_identity_jaccard": float(len(common) / len(union)) if union else 1.0,
        "shifted_drop_queries": len(shifts),
        "maximum_absolute_query_shift": max(shifts, default=0),
        "retained_main_terminal_successes": len(main_success_ids & variant_ids),
        "main_terminal_successes": len(main_success_ids),
        "removed_main_event_ids": [
            {"task": key[0], "episode": key[1], "target": key[2]}
            for key in sorted(main_ids - variant_ids)
        ],
        "added_event_ids": [
            {"task": key[0], "episode": key[1], "target": key[2]}
            for key in sorted(variant_ids - main_ids)
        ],
        "shifted_event_ids": shifted,
    }
    return output


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    counts = summary["event_counts"]
    gate = summary["gate0"]
    diagnostics = summary["phase_followup_diagnostics"]
    lines = [
        "# Post-loss proxy eventual-outcome feasibility: Gate 0",
        "",
        "Transport targets are frozen from the BDDL goals before outcomes are read.",
        "The landmark is the first query after a conservative kinematic transport-loss",
        "proxy. The proxy is not a dense-contact slip annotation.",
        "",
        "## Event inventory",
        "",
        "| task | events | terminal success after proxy | terminal failure after proxy |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["coverage"]:
        lines.append(
            "| %s | %d | %d | %d |"
            % (
                row["task"],
                row["events"],
                row["event_successes"],
                row["events"] - row["event_successes"],
            )
        )
    lines.extend(
        [
            "",
            "Total: %d events (%d later terminal successes, %d later terminal failures), %d mixed task x initial-state clusters, %d conditional pairs."
            % (
                counts["events"],
                counts["terminal_success_after_proxy"],
                counts["terminal_failure_after_proxy"],
                counts["mixed_task_initial_state_clusters"],
                counts["conditional_pairs"],
            ),
            "",
            "## Gate decision",
            "",
            "**%s.** %s"
            % (gate["status"].upper(), gate["reason"] or "The cohort is eligible."),
            "",
            "| requirement | observed/pass |",
            "|---|---:|",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"| {name} | {str(passed).lower()} |")
    if gate["status"] == "fail":
        lines.extend(
            [
                "",
                "The physical -> +route -> +action -> +hidden classifier was not fit.",
                (
                    "With only %d later terminal successes, an AUC or p-value would be "
                    "dominated by task, initial state, and phase rather than post-error adaptation."
                )
                % counts["terminal_success_after_proxy"],
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The cohort passed Gate 0, but this inventory-only script does not fit the classifier.",
            ]
        )
    lines.extend(
        [
            "",
            "## Phase and follow-up diagnostics",
            "",
            "- Mixed task x initial-state x 5-query phase clusters: %d."
            % diagnostics["mixed_task_initial_state_phase_clusters"],
            "- Conditional pairs within those phase clusters: %d."
            % diagnostics["conditional_pairs_within_phase"],
            "- Minimum remaining queries after the landmark: %s."
            % diagnostics["minimum_remaining_queries"],
            "- Absolute univariate AUC for query phase alone: %s."
            % fmt(
                diagnostics["univariate_absolute_auc_sentinels"]["drop_query"]
            ),
            "",
            "## Descriptive recurrence",
            "",
            "Values below compare the post-error query with the chunk that caused",
            "the detected loss. They are descriptive and carry no inferential claim.",
            "",
            "| outcome | n | action-route Hellinger similarity | action top-4 Jaccard | action cosine | hidden-action cosine |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    desc = summary["descriptive_recurrence"]["failed_chunk"]
    for label in (
        "terminal_success_after_proxy",
        "terminal_failure_after_proxy",
    ):
        group = desc[label]
        metrics = group["metrics"]
        lines.append(
            "| %s | %d | %s | %s | %s | %s |"
            % (
                label,
                group["n"],
                fmt(metrics["route_action_hellinger_similarity"]["median"]),
                fmt(metrics["route_action_top4_jaccard"]["median"]),
                fmt(metrics["action_cosine_similarity"]["median"]),
                fmt(metrics["hidden_action_cosine_similarity"]["median"]),
            )
        )
    lines.extend(
        [
            "",
            "## Threshold sensitivity",
            "",
            "All strict/loose variants are joint 20% perturbations unless marked hold-distance-only.",
            "",
            "| rule | events | success | failure | mixed clusters | identity Jaccard | shifted query |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["threshold_sensitivity"].items():
        lines.append(
            "| %s | %d | %d | %d | %d | %s | %d |"
            % (
                name,
                row["events"],
                row["terminal_success_after_proxy"],
                row["terminal_failure_after_proxy"],
                row["mixed_task_initial_state_clusters"],
                fmt(row["identity_stability"]["event_identity_jaccard"]),
                row["identity_stability"]["shifted_drop_queries"],
            )
        )
    lines.extend(
        [
            "",
            "## Causal boundary",
            "",
            "- `drop_query` is after the first detected physical loss and before the next emitted chunk.",
            "- Recurrence at that query cannot predict the loss that already occurred.",
            "- It could prognose later recovery only after a larger, contact-validated cohort is collected.",
            "- The binary label is eventual task outcome, not verified regrasp, second slip, or trap.",
            "- `hb_hidden` is router-input contextual hidden state, not expert contribution.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    length = 8
    position = np.zeros((length, 3), np.float32)
    eef = np.zeros((length, 3), np.float32)
    position[:, 2] = 1.0
    eef[:, 2] = 1.0
    position[2] = [0.02, 0.0, 1.02]
    position[3] = [0.04, 0.0, 1.04]
    position[4] = [0.04, 0.0, 1.015]
    eef[2:4] = position[2:4]
    eef[4] = [0.20, 0.0, 1.04]
    state = np.zeros((length, 8), np.float32)
    state[:, :3] = eef
    state[:, 6:8] = 0.01
    state[1, 6:8] = 0.04
    action = np.zeros((length, 10, 7), np.float32)
    action[:, :, 6] = 1.0
    events, _evidence, _distance = detect_target_events(
        position, state, action, 0.085
    )
    assert len(events) == 1
    assert events[0]["bout_start"] == 2
    assert events[0]["drop_query"] == 4

    probability = np.full((8, 10, 11, 32), 1.0 / 32, np.float32)
    expert = np.tile(np.arange(4, dtype=np.uint8), (8, 10, 11, 1))
    assert math.isclose(route_hellinger_similarity(probability, probability, slice(1, 11)), 1.0)
    assert math.isclose(top4_jaccard_similarity(expert, expert, slice(1, 11)), 1.0)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.cache_root)
    events: list[Event] = []
    coverage = []
    events_by_run: dict[pathlib.Path, list[Event]] = {}
    for run in runs:
        task_events, task_coverage = inventory_task(run, args.cache_root)
        events.extend(task_events)
        events_by_run[run] = task_events
        coverage.append(task_coverage)
        print(
            "%s: %d events (%d later terminal successes)"
            % (
                task_coverage["task"],
                task_coverage["events"],
                task_coverage["event_successes"],
            ),
            flush=True,
        )

    recurrence_rows: list[dict[str, Any]] = []
    for run, task_events in events_by_run.items():
        recurrence_rows.extend(extract_recurrence(run, task_events, args.skip_hidden))
        if task_events:
            print(f"extracted recurrence for {task_key(run, args.cache_root)}", flush=True)

    sensitivity = {}
    sensitivity_specs = {
        "joint_strict": (make_rule_variant(MAIN_RULE, "strict"), 0.8),
        "main": (MAIN_RULE, 1.0),
        "joint_loose": (make_rule_variant(MAIN_RULE, "loose"), 1.2),
        "hold_distance_strict_only": (MAIN_RULE, 0.8),
        "hold_distance_loose_only": (MAIN_RULE, 1.2),
    }
    for name, (rule, distance_scale) in sensitivity_specs.items():
        if name == "main":
            variant_events = events
        else:
            variant_events = []
            for run in runs:
                rows, _coverage = inventory_task(
                    run, args.cache_root, rule, distance_scale
                )
                variant_events.extend(rows)
        sensitivity[name] = sensitivity_summary(events, variant_events)

    counts = event_counts(events)
    folds = double_holdout_feasibility(events)
    diagnostics = phase_followup_diagnostics(events)
    summary = {
        "schema": "himoe.post_error_recovery_gate0.v2",
        "causal_index": {
            "failed_chunk_anchor": "query whose emitted action chunk precedes the observed loss",
            "drop_query": "first policy query that observes the loss proxy",
            "grasp_anchor": "last nearby open-gripper query within four queries before held-bout onset",
        },
        "event_rule": asdict(MAIN_RULE),
        "task_hold_distance_m": {
            "libero_long": LONG_HOLD_DISTANCE_M,
            "other": OTHER_HOLD_DISTANCE_M,
        },
        "target_selection": {
            "source": "frozen first argument(s) of BDDL transport goal predicates",
            "mapping": {key: list(value) for key, value in TASK_TRANSPORT_OBJECTS.items()},
            "uses_rollout_outcome": False,
        },
        "outcome_contract": {
            "primary_inventory_label": "eventual episode task success after the proxy",
            "future_proxy_rehold": "later query-level kinematic hold evidence",
            "not_annotated": [
                "contact-validated regrasp for the full cohort",
                "second same-type physical loss",
                "repeated-policy trap",
            ],
        },
        "coverage": coverage,
        "event_counts": counts,
        "double_holdout_feasibility": folds,
        "phase_followup_diagnostics": diagnostics,
        "gate0": gate_result(counts, folds, diagnostics),
        "threshold_sensitivity": sensitivity,
        "descriptive_recurrence": descriptive_summary(recurrence_rows),
        "limitations": [
            "query-level kinematic proxy, not dense contact ground truth",
            "eventual task outcome only; recovery and second same-type error are not fully annotated",
            "no classifier or significance test is run after Gate 0 fails",
        ],
    }

    event_rows = [asdict(event) for event in events]
    write_csv(args.out_dir / "event_inventory.csv", event_rows)
    write_csv(args.out_dir / "recurrence_inventory.csv", recurrence_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
