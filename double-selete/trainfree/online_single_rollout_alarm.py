#!/usr/bin/env python3
"""Seal and replay the train-free single-rollout streaming alarm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import online_closed_loop_alarm_v2 as v2
import online_multihead_alarm as v1
from single_rollout_monitor import (
    DETECTORS,
    PRIMARY_DETECTOR,
    TYPED_STATES,
    WARMUP_QUERY,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_ROUTE_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
DEFAULT_PHYSICAL_CACHE = HERE / "results/online_closed_loop_v2/unlabeled_physical_features.npz"
DEFAULT_OUTPUT = HERE / "results/online_single_rollout"
PROTOCOL = HERE / "ONLINE_SINGLE_ROLLOUT_PROTOCOL.md"
OPERATING_QUANTILES = (0.90, 0.95, 0.975)
MIN_REFERENCE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-cache", type=Path, default=DEFAULT_ROUTE_CACHE)
    parser.add_argument("--physical-cache", type=Path, default=DEFAULT_PHYSICAL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def complete_mean(stack: np.ndarray) -> np.ndarray:
    output = np.full(stack.shape[:2], np.nan, dtype=np.float32)
    good = np.isfinite(stack).all(axis=-1)
    output[good] = stack[good].mean(axis=-1)
    return output


def available_max(stack: np.ndarray) -> np.ndarray:
    output = np.full(stack.shape[:2], np.nan, dtype=np.float32)
    good = np.isfinite(stack).any(axis=-1)
    output[good] = np.nanmax(stack[good], axis=-1)
    return output


def trailing_mean(values: np.ndarray, width: int = 3) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float32)
    for query in range(width - 1, values.shape[1]):
        window = values[:, query - width + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].mean(axis=1)
    return output


def adjacent_min(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float32)
    good = np.isfinite(values[:, 1:]) & np.isfinite(values[:, :-1])
    current = np.minimum(values[:, 1:], values[:, :-1])
    output[:, 1:][good] = current[good]
    return output


def geometric(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.full_like(left, np.nan, dtype=np.float32)
    good = np.isfinite(left) & np.isfinite(right)
    output[good] = np.sqrt(np.clip(left[good] * right[good], 0.0, 1.0))
    return output


def state_machine_scores(
    route_heads: dict[str, np.ndarray], physical_heads: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    static_route = np.maximum(
        route_heads["lock_in"], route_heads["flat_narrow_support"]
    )
    loop_physical = geometric(
        route_heads["instability"], physical_heads["physical_cycle"]
    )
    static_physical = geometric(static_route, physical_heads["motion_stall"])
    feedback_physical = geometric(
        route_heads["feedback_decoupling"], physical_heads["motion_stall"]
    )

    loop_switch = np.full_like(static_route, np.nan, dtype=np.float32)
    for query in range(3, static_route.shape[1]):
        past = route_heads["instability"][:, query - 3 : query]
        good = np.isfinite(past).all(axis=1) & np.isfinite(static_route[:, query])
        loop_switch[good, query] = np.sqrt(
            np.clip(past[good].max(axis=1) * static_route[good, query], 0.0, 1.0)
        )
    loop_hold = trailing_mean(loop_physical)
    loop_state = available_max(np.stack((loop_switch, loop_hold), axis=-1))
    static_state = trailing_mean(static_physical)
    feedback_state = trailing_mean(feedback_physical)
    typed = {
        "loop_state": loop_state,
        "static_state": static_state,
        "feedback_state": feedback_state,
    }
    instant_route = available_max(np.stack(tuple(route_heads.values()), axis=-1))
    phenotype = available_max(np.stack(tuple(typed.values()), axis=-1))
    scores = {
        **typed,
        "instant_route": instant_route,
        "persistent_route": adjacent_min(instant_route),
        "multi_state": phenotype,
        "single_rollout_multi": adjacent_min(phenotype),
    }
    typed_stack = np.stack([typed[name] for name in TYPED_STATES], axis=-1)
    winner = np.full(typed_stack.shape[:2], -1, dtype=np.int8)
    good = np.isfinite(typed_stack).any(axis=-1)
    winner[good] = np.nanargmax(typed_stack[good], axis=-1).astype(np.int8)
    return scores, winner


def normalize_heads(
    route_features: np.ndarray,
    physical_features: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    route_index = {name: index for index, name in enumerate(v1.FEATURES)}
    route_rank: dict[tuple[str, int], np.ndarray] = {}
    members = {member for head_members in v1.HEADS.values() for member in head_members}
    for name, direction in sorted(members):
        route_rank[(name, direction)] = v2.directed_rank(
            route_features[:, :, route_index[name]],
            direction,
            reference,
            valid,
            v1.RAW_SCORE_START_QUERY,
        )
    route_heads: dict[str, np.ndarray] = {}
    for head, head_members in v1.HEADS.items():
        instant = complete_mean(
            np.stack([route_rank[member] for member in head_members], axis=-1)
        )
        route_heads[head] = trailing_mean(instant)

    physical_index = {name: index for index, name in enumerate(v2.PHYSICAL_FEATURES)}
    specifications = {
        "low_eef_motion": ("eef_motion_w3", -1),
        "high_prior_command": ("prior_command_w3", 1),
        "high_eef_reversal": ("eef_reversal_w3", 1),
        "high_gripper_flip": ("gripper_flip_w4", 1),
        "high_state_return": ("state_return_w4", 1),
    }
    physical_rank = {
        label: v2.directed_rank(
            physical_features[:, :, physical_index[name]],
            direction,
            reference,
            valid,
            v2.WARMUP_QUERY,
        )
        for label, (name, direction) in specifications.items()
    }
    physical_heads = {
        "motion_stall": complete_mean(
            np.stack(
                (physical_rank["low_eef_motion"], physical_rank["high_prior_command"]),
                axis=-1,
            )
        ),
        "physical_cycle": complete_mean(
            np.stack(
                (
                    physical_rank["high_eef_reversal"],
                    physical_rank["high_gripper_flip"],
                    physical_rank["high_state_return"],
                ),
                axis=-1,
            )
        ),
    }
    return route_heads, physical_heads


def score_against_reference(
    route_features: np.ndarray,
    physical_features: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    if int(reference.sum()) < MIN_REFERENCE:
        raise ValueError("too few historical reference trajectories")
    route_heads, physical_heads = normalize_heads(
        route_features, physical_features, valid, reference
    )
    scores, winner = state_machine_scores(route_heads, physical_heads)
    eligible = valid.copy()
    eligible[:, :WARMUP_QUERY] = False
    for collection in (route_heads, physical_heads, scores):
        for name in collection:
            collection[name][~eligible] = np.nan
    winner[~eligible] = -1
    return scores, route_heads, physical_heads, winner


def calibrate_thresholds(
    scores: dict[str, np.ndarray], reference: np.ndarray
) -> np.ndarray:
    threshold = np.full(
        (len(DETECTORS), len(OPERATING_QUANTILES)), np.nan, dtype=np.float32
    )
    for detector_position, detector in enumerate(DETECTORS):
        maxima = v2.nan_row_max(scores[detector][reference])
        for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
            threshold[detector_position, quantile_position] = v2.quantile_higher(
                maxima, quantile
            )
    if not np.isfinite(threshold).all():
        raise ValueError("non-finite detector threshold")
    return threshold


def calibrate_and_replay(
    route_cache: dict[str, np.ndarray],
    physical_cache: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    route_features = np.asarray(route_cache["features"], dtype=np.float32)
    physical_features = np.asarray(physical_cache["features"], dtype=np.float32)
    valid = np.asarray(route_cache["valid"], dtype=bool)
    task_names = np.asarray(route_cache["task_names"]).astype(str)
    task_index = np.asarray(route_cache["task_index"], dtype=np.int16)
    episode = np.asarray(route_cache["episode"], dtype=np.int16)
    init_state = np.asarray(route_cache["init_state_id"], dtype=np.int16)
    flow_seed = np.asarray(route_cache["flow_noise_seed"], dtype=np.int16)
    lengths = np.asarray(route_cache["length"], dtype=np.int16)
    n_episode, n_query = valid.shape

    detector_scores = np.full(
        (n_episode, n_query, len(DETECTORS)), np.nan, dtype=np.float32
    )
    route_scores = np.full(
        (n_episode, n_query, len(v1.HEADS)), np.nan, dtype=np.float32
    )
    physical_scores = np.full(
        (n_episode, n_query, len(v2.PHYSICAL_HEADS)), np.nan, dtype=np.float32
    )
    threshold_by_episode = np.full(
        (n_episode, len(DETECTORS), len(OPERATING_QUANTILES)), np.nan, dtype=np.float32
    )
    alarms = np.zeros(
        (n_episode, n_query, len(DETECTORS), len(OPERATING_QUANTILES)), dtype=bool
    )
    winning_state = np.full((n_episode, n_query), -1, dtype=np.int8)
    deployment_thresholds = np.full(
        (len(task_names), len(DETECTORS), len(OPERATING_QUANTILES)),
        np.nan,
        dtype=np.float32,
    )
    threshold_rows: list[dict[str, Any]] = []

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"expected 400 trajectories for {task}")
        task_valid = valid[take]
        task_init = init_state[take]
        for held_out in np.unique(task_init):
            reference = task_init != held_out
            test = ~reference
            scores, route_heads, physical_heads, winner = score_against_reference(
                route_features[take],
                physical_features[take],
                task_valid,
                reference,
            )
            threshold = calibrate_thresholds(scores, reference)
            global_test = take[test]
            detector_scores[global_test] = np.stack(
                [scores[name][test] for name in DETECTORS], axis=-1
            )
            route_scores[global_test] = np.stack(
                [route_heads[name][test] for name in v1.HEADS], axis=-1
            )
            physical_scores[global_test] = np.stack(
                [physical_heads[name][test] for name in v2.PHYSICAL_HEADS], axis=-1
            )
            winning_state[global_test] = winner[test]
            threshold_by_episode[global_test] = threshold
            for detector_position, detector in enumerate(DETECTORS):
                for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                    trigger = np.isfinite(scores[detector][test]) & (
                        scores[detector][test]
                        > threshold[detector_position, quantile_position]
                    )
                    alarms[
                        global_test, :, detector_position, quantile_position
                    ] = np.maximum.accumulate(trigger, axis=1) & task_valid[test]
                    threshold_rows.append(
                        {
                            "task": task,
                            "held_out_init_state_id": int(held_out),
                            "detector": detector,
                            "quantile": quantile,
                            "threshold": float(
                                threshold[detector_position, quantile_position]
                            ),
                            "outcome_blind_reference_episodes": int(reference.sum()),
                        }
                    )

        full_reference = np.ones(len(take), dtype=bool)
        full_scores, _, _, _ = score_against_reference(
            route_features[take], physical_features[take], task_valid, full_reference
        )
        deployment_thresholds[task_position] = calibrate_thresholds(
            full_scores, full_reference
        )
        print(f"[streaming calibration {task_position + 1}/{len(task_names)}] {task}", flush=True)

    if not np.isfinite(threshold_by_episode).all():
        raise ValueError("non-finite held-out threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sealed_online_scores.npz",
        schema=np.asarray("himoe.single_rollout.sealed.v1"),
        task_names=task_names,
        task_index=task_index,
        episode=episode,
        init_state_id=init_state,
        flow_noise_seed=flow_seed,
        length=lengths,
        valid=valid,
        detector_names=np.asarray(DETECTORS),
        route_head_names=np.asarray(tuple(v1.HEADS)),
        physical_head_names=np.asarray(v2.PHYSICAL_HEADS),
        typed_state_names=np.asarray(TYPED_STATES),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        scores=detector_scores,
        route_head_scores=route_scores,
        physical_head_scores=physical_scores,
        thresholds=threshold_by_episode,
        alarms=alarms,
        winning_state=winning_state,
    )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "unlabeled_thresholds.csv", index=False
    )
    np.savez_compressed(
        output_dir / "deployment_profiles.npz",
        schema=np.asarray("himoe.single_rollout.profile.v1"),
        task_names=task_names,
        reference_task_index=task_index,
        reference_valid=valid,
        route_feature_names=np.asarray(v1.FEATURES),
        physical_feature_names=np.asarray(v2.PHYSICAL_FEATURES),
        route_reference=route_features,
        physical_reference=physical_features,
        detector_names=np.asarray(DETECTORS),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        thresholds=deployment_thresholds,
    )

    rows: list[dict[str, Any]] = []
    primary_position = DETECTORS.index(PRIMARY_DETECTOR)
    for row in range(n_episode):
        item: dict[str, Any] = {
            "task": task_names[task_index[row]],
            "episode": int(episode[row]),
            "init_state_id": int(init_state[row]),
            "flow_noise_seed": int(flow_seed[row]),
            "length": int(lengths[row]),
        }
        for detector_position, detector in enumerate(DETECTORS):
            for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                query = np.flatnonzero(
                    alarms[row, :, detector_position, quantile_position]
                )
                suffix = str(quantile).replace("0.", "q")
                item[f"first_{detector}_{suffix}"] = int(query[0]) if len(query) else -1
        primary = np.flatnonzero(alarms[row, :, primary_position, 1])
        winner = int(winning_state[row, primary[0]]) if len(primary) else -1
        item["primary_phenotype"] = TYPED_STATES[winner] if winner >= 0 else "none"
        rows.append(item)
    pd.DataFrame(rows).to_csv(output_dir / "sealed_episode_alarms.csv", index=False)


def self_test() -> None:
    rng = np.random.default_rng(9)
    shape = (5, 15)
    route = {name: rng.uniform(size=shape).astype(np.float32) for name in v1.HEADS}
    physical = {
        name: rng.uniform(size=shape).astype(np.float32) for name in v2.PHYSICAL_HEADS
    }
    scores, winner = state_machine_scores(route, physical)
    assert set(scores) == set(DETECTORS)
    assert np.isnan(scores[PRIMARY_DETECTOR][:, :3]).all()
    assert winner.shape == shape
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (args.output / "sealed_manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed result: {args.output}")
    route_cache = v1.load_feature_cache(args.route_cache)
    physical_cache = v2.load_physical_cache(args.physical_cache, route_cache)
    args.output.mkdir(parents=True, exist_ok=True)
    calibrate_and_replay(route_cache, physical_cache, args.output)

    artifacts = {
        "protocol_sha256": sha256(PROTOCOL),
        "scorer_sha256": sha256(Path(__file__)),
        "runtime_monitor_sha256": sha256(HERE / "single_rollout_monitor.py"),
        "route_feature_cache_sha256": sha256(args.route_cache),
        "physical_feature_cache_sha256": sha256(args.physical_cache),
        "sealed_scores_sha256": sha256(args.output / "sealed_online_scores.npz"),
        "thresholds_sha256": sha256(args.output / "unlabeled_thresholds.csv"),
        "episode_alarms_sha256": sha256(args.output / "sealed_episode_alarms.csv"),
        "deployment_profiles_sha256": sha256(args.output / "deployment_profiles.npz"),
    }
    manifest = {
        "schema": "himoe.single_rollout.manifest.v1",
        "cohort": "main",
        "run_id": v1.RUN_ID,
        "tasks": int(len(route_cache["task_names"])),
        "episodes": int(len(route_cache["episode"])),
        "runtime_api": "SingleRolloutMonitor.update(router_prob, expert_ids, state, actions)",
        "runtime_inputs": ["current_router_prob", "current_expert_ids", "current_state", "current_actions"],
        "runtime_cross_rollout_access": False,
        "runtime_future_access": False,
        "runtime_final_length_access": False,
        "duration_detector_feature": False,
        "termination_or_censoring_filter": False,
        "historical_reference_policy": "all 392 other-init trajectories retained outcome-blind",
        "query_phase_normalization": True,
        "route_features": list(v1.FEATURES),
        "physical_features": list(v2.PHYSICAL_FEATURES),
        "route_heads": plain(v1.HEADS),
        "physical_heads": list(v2.PHYSICAL_HEADS),
        "typed_states": list(TYPED_STATES),
        "detectors": list(DETECTORS),
        "primary_detector": PRIMARY_DETECTOR,
        "primary_quantile": 0.95,
        "warmup_query": WARMUP_QUERY,
        "operating_quantiles": list(OPERATING_QUANTILES),
        "query_causal": True,
        "labels_used": [],
        "sim_state_used": False,
        "artifacts": artifacts,
    }
    (args.output / "sealed_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"sealed strict streaming alarms for {manifest['episodes']} rollouts at {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
