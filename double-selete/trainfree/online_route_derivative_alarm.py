#!/usr/bin/env python3
"""Seal strict single-rollout alarms from route level and query derivatives."""

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
from route_derivative_monitor import (
    BRANCHES,
    DERIVATIVE_DECIMALS,
    DETECTORS,
    PRIMARY_DETECTOR,
    WARMUP_QUERY,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ROUTE_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
DEFAULT_OUTPUT = HERE / "results/online_route_derivative"
PROTOCOL = HERE / "ONLINE_ROUTE_DERIVATIVE_PROTOCOL.md"
OPERATING_QUANTILES = (0.90, 0.95, 0.975)
MIN_REFERENCE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-cache", type=Path, default=DEFAULT_ROUTE_CACHE)
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


def complete_max(stack: np.ndarray) -> np.ndarray:
    output = np.full(stack.shape[:2], np.nan, dtype=np.float32)
    good = np.isfinite(stack).all(axis=-1)
    output[good] = stack[good].max(axis=-1)
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


def normalize_route_heads(
    features: np.ndarray, valid: np.ndarray, reference: np.ndarray
) -> dict[str, np.ndarray]:
    feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
    rank: dict[tuple[str, int], np.ndarray] = {}
    members = {member for head_members in v1.HEADS.values() for member in head_members}
    for name, direction in sorted(members):
        rank[(name, direction)] = v2.directed_rank(
            features[:, :, feature_index[name]],
            direction,
            reference,
            valid,
            v1.RAW_SCORE_START_QUERY,
        )
    heads: dict[str, np.ndarray] = {}
    for head, head_members in v1.HEADS.items():
        instant = complete_mean(
            np.stack([rank[member] for member in head_members], axis=-1)
        )
        heads[head] = trailing_mean(instant)
    return heads


def raw_derivatives(
    heads: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.stack([heads[name] for name in v1.HEADS], axis=-1)
    velocity = np.full(vector.shape[:2], np.nan, dtype=np.float32)
    acceleration = np.full(vector.shape[:2], np.nan, dtype=np.float32)
    velocity_good = np.isfinite(vector[:, 1:]).all(axis=-1) & np.isfinite(
        vector[:, :-1]
    ).all(axis=-1)
    difference = np.round(
        np.linalg.norm(vector[:, 1:] - vector[:, :-1], axis=-1) / 2.0,
        DERIVATIVE_DECIMALS,
    )
    velocity[:, 1:][velocity_good] = difference[velocity_good]
    acceleration_good = (
        np.isfinite(vector[:, 2:]).all(axis=-1)
        & np.isfinite(vector[:, 1:-1]).all(axis=-1)
        & np.isfinite(vector[:, :-2]).all(axis=-1)
    )
    curvature = np.round(
        np.linalg.norm(
            vector[:, 2:] - 2.0 * vector[:, 1:-1] + vector[:, :-2], axis=-1
        )
        / 4.0,
        DERIVATIVE_DECIMALS,
    )
    acceleration[:, 2:][acceleration_good] = curvature[acceleration_good]
    return vector, velocity, acceleration


def derivative_scores(
    heads: dict[str, np.ndarray], valid: np.ndarray, reference: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vector, velocity_raw, acceleration_raw = raw_derivatives(heads)
    velocity = v2.directed_rank(
        velocity_raw, 1, reference, valid, v1.RAW_SCORE_START_QUERY + 3
    )
    acceleration = v2.directed_rank(
        acceleration_raw, 1, reference, valid, v1.RAW_SCORE_START_QUERY + 4
    )
    level = complete_max(vector)
    persistent = adjacent_min(level)
    level_velocity = geometric(level, velocity)
    level_acceleration = geometric(level, acceleration)
    mean = complete_mean(np.stack((level, velocity, acceleration), axis=-1))
    branch_stack = np.stack(
        (persistent, level_velocity, level_acceleration), axis=-1
    )
    fusion = complete_max(branch_stack)
    scores = {
        "instant_level": level,
        "persistent_level": persistent,
        "level_velocity": level_velocity,
        "level_acceleration": level_acceleration,
        "derivative_mean": mean,
        "derivative_fusion": fusion,
    }
    winner_branch = np.full(level.shape, -1, dtype=np.int8)
    branch_good = np.isfinite(branch_stack).all(axis=-1)
    winner_branch[branch_good] = np.argmax(
        branch_stack[branch_good], axis=-1
    ).astype(np.int8)
    winner_head = np.full(level.shape, -1, dtype=np.int8)
    head_good = np.isfinite(vector).all(axis=-1)
    winner_head[head_good] = np.argmax(vector[head_good], axis=-1).astype(np.int8)

    eligible = valid.copy()
    eligible[:, :WARMUP_QUERY] = False
    for collection in (heads, scores):
        for name in collection:
            collection[name][~eligible] = np.nan
    winner_branch[~eligible] = -1
    winner_head[~eligible] = -1
    return scores, velocity_raw, acceleration_raw, winner_branch, winner_head


def score_against_reference(
    features: np.ndarray, valid: np.ndarray, reference: np.ndarray
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if int(reference.sum()) < MIN_REFERENCE:
        raise ValueError("too few outcome-blind references")
    heads = normalize_route_heads(features, valid, reference)
    scores, velocity, acceleration, branch, head = derivative_scores(
        heads, valid, reference
    )
    return scores, heads, velocity, acceleration, branch, head


def calibrate_thresholds(
    scores: dict[str, np.ndarray], reference: np.ndarray
) -> np.ndarray:
    output = np.full(
        (len(DETECTORS), len(OPERATING_QUANTILES)), np.nan, dtype=np.float32
    )
    for detector_position, detector in enumerate(DETECTORS):
        maxima = v2.nan_row_max(scores[detector][reference])
        for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
            output[detector_position, quantile_position] = v2.quantile_higher(
                maxima, quantile
            )
    if not np.isfinite(output).all():
        raise ValueError("non-finite derivative threshold")
    return output


def calibrate_and_replay(cache: dict[str, np.ndarray], output_dir: Path) -> None:
    features = np.asarray(cache["features"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    task_names = np.asarray(cache["task_names"]).astype(str)
    task_index = np.asarray(cache["task_index"], dtype=np.int16)
    episode = np.asarray(cache["episode"], dtype=np.int16)
    init_state = np.asarray(cache["init_state_id"], dtype=np.int16)
    flow_seed = np.asarray(cache["flow_noise_seed"], dtype=np.int16)
    lengths = np.asarray(cache["length"], dtype=np.int16)
    n_episode, n_query = valid.shape

    detector_scores = np.full(
        (n_episode, n_query, len(DETECTORS)), np.nan, dtype=np.float32
    )
    head_scores = np.full(
        (n_episode, n_query, len(v1.HEADS)), np.nan, dtype=np.float32
    )
    thresholds = np.full(
        (n_episode, len(DETECTORS), len(OPERATING_QUANTILES)), np.nan, dtype=np.float32
    )
    alarms = np.zeros(
        (n_episode, n_query, len(DETECTORS), len(OPERATING_QUANTILES)), dtype=bool
    )
    winning_branch = np.full((n_episode, n_query), -1, dtype=np.int8)
    winning_head = np.full((n_episode, n_query), -1, dtype=np.int8)
    deployment_velocity = np.full(valid.shape, np.nan, dtype=np.float32)
    deployment_acceleration = np.full(valid.shape, np.nan, dtype=np.float32)
    deployment_threshold = np.full(
        (len(task_names), len(DETECTORS), len(OPERATING_QUANTILES)),
        np.nan,
        dtype=np.float32,
    )
    threshold_rows: list[dict[str, Any]] = []

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        task_valid = valid[take]
        task_init = init_state[take]
        for held_out in np.unique(task_init):
            reference = task_init != held_out
            test = ~reference
            result = score_against_reference(features[take], task_valid, reference)
            scores, heads, _, _, branch, head = result
            threshold = calibrate_thresholds(scores, reference)
            global_test = take[test]
            detector_scores[global_test] = np.stack(
                [scores[name][test] for name in DETECTORS], axis=-1
            )
            head_scores[global_test] = np.stack(
                [heads[name][test] for name in v1.HEADS], axis=-1
            )
            thresholds[global_test] = threshold
            winning_branch[global_test] = branch[test]
            winning_head[global_test] = head[test]
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
        full_result = score_against_reference(
            features[take], task_valid, full_reference
        )
        full_scores, _, velocity, acceleration, _, _ = full_result
        deployment_velocity[take] = velocity
        deployment_acceleration[take] = acceleration
        deployment_threshold[task_position] = calibrate_thresholds(
            full_scores, full_reference
        )
        print(f"[derivative calibration {task_position + 1}/{len(task_names)}] {task}", flush=True)

    if not np.isfinite(thresholds).all():
        raise ValueError("non-finite held-out threshold")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sealed_online_scores.npz",
        schema=np.asarray("himoe.route_derivative.sealed.v1"),
        task_names=task_names,
        task_index=task_index,
        episode=episode,
        init_state_id=init_state,
        flow_noise_seed=flow_seed,
        length=lengths,
        valid=valid,
        detector_names=np.asarray(DETECTORS),
        head_names=np.asarray(tuple(v1.HEADS)),
        branch_names=np.asarray(BRANCHES),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        scores=detector_scores,
        route_head_scores=head_scores,
        thresholds=thresholds,
        alarms=alarms,
        winning_branch=winning_branch,
        winning_head=winning_head,
    )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "unlabeled_thresholds.csv", index=False
    )
    np.savez_compressed(
        output_dir / "deployment_profiles.npz",
        schema=np.asarray("himoe.route_derivative.profile.v1"),
        task_names=task_names,
        reference_task_index=task_index,
        reference_valid=valid,
        route_feature_names=np.asarray(v1.FEATURES),
        route_reference=features,
        velocity_reference=deployment_velocity,
        acceleration_reference=deployment_acceleration,
        detector_names=np.asarray(DETECTORS),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        thresholds=deployment_threshold,
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
        if len(primary):
            query = int(primary[0])
            branch = int(winning_branch[row, query])
            head = int(winning_head[row, query])
            item["primary_branch"] = BRANCHES[branch] if branch >= 0 else "none"
            item["primary_head"] = tuple(v1.HEADS)[head] if head >= 0 else "none"
        else:
            item["primary_branch"] = "none"
            item["primary_head"] = "none"
        rows.append(item)
    pd.DataFrame(rows).to_csv(output_dir / "sealed_episode_alarms.csv", index=False)


def self_test() -> None:
    head = {
        name: np.asarray([[0.2, 0.4, 0.7]], dtype=np.float32)
        for name in v1.HEADS
    }
    _, velocity, acceleration = raw_derivatives(head)
    np.testing.assert_allclose(velocity[0, 1:], [0.2, 0.3], atol=2e-8)
    np.testing.assert_allclose(acceleration[0, 2], 0.05, atol=2e-8)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (args.output / "sealed_manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed result: {args.output}")
    cache = v1.load_feature_cache(args.route_cache)
    args.output.mkdir(parents=True, exist_ok=True)
    calibrate_and_replay(cache, args.output)
    manifest = {
        "schema": "himoe.route_derivative.manifest.v1",
        "cohort": "main",
        "run_id": v1.RUN_ID,
        "tasks": int(len(cache["task_names"])),
        "episodes": int(len(cache["episode"])),
        "runtime_api": "RouteDerivativeMonitor.update(router_prob, expert_ids)",
        "runtime_inputs": ["current_router_prob", "current_expert_ids"],
        "runtime_cross_rollout_access": False,
        "runtime_future_access": False,
        "runtime_final_length_access": False,
        "duration_detector_feature": False,
        "termination_or_censoring_filter": False,
        "state_or_action_used": False,
        "sim_state_used": False,
        "historical_reference_policy": "all 392 other-init trajectories retained outcome-blind",
        "features": list(v1.FEATURES),
        "heads": plain(v1.HEADS),
        "detectors": list(DETECTORS),
        "branches": list(BRANCHES),
        "primary_detector": PRIMARY_DETECTOR,
        "primary_quantile": 0.95,
        "warmup_query": WARMUP_QUERY,
        "derivative_round_decimals": DERIVATIVE_DECIMALS,
        "operating_quantiles": list(OPERATING_QUANTILES),
        "labels_used": [],
        "query_causal": True,
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "scorer_sha256": sha256(Path(__file__)),
            "runtime_monitor_sha256": sha256(HERE / "route_derivative_monitor.py"),
            "route_feature_cache_sha256": sha256(args.route_cache),
            "sealed_scores_sha256": sha256(args.output / "sealed_online_scores.npz"),
            "thresholds_sha256": sha256(args.output / "unlabeled_thresholds.csv"),
            "episode_alarms_sha256": sha256(args.output / "sealed_episode_alarms.csv"),
            "deployment_profiles_sha256": sha256(args.output / "deployment_profiles.npz"),
        },
    }
    (args.output / "sealed_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"sealed route-derivative alarms for {manifest['episodes']} rollouts at {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
