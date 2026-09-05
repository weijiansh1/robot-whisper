#!/usr/bin/env python3
"""Build and seal the censor-aware train-free closed-loop Trap alarm v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import online_multihead_alarm as v1


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_ROUTE_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
DEFAULT_OUTPUT = HERE / "results/online_closed_loop_v2"
PROTOCOL = HERE / "ONLINE_CLOSED_LOOP_V2_PROTOCOL.md"
RUN_ID = v1.RUN_ID

WARMUP_QUERY = 4
SMOOTHING_WIDTH = 3
MIN_REFERENCE = 32
OPERATING_QUANTILES = (0.90, 0.95, 0.975)

PHYSICAL_FEATURES = (
    "eef_motion_w3",
    "gripper_motion_w3",
    "prior_command_w3",
    "eef_reversal_w3",
    "gripper_flip_w4",
    "state_return_w4",
)
PHYSICAL_HEADS = ("motion_stall", "physical_cycle")
CONFIRMATIONS = ("loop_confirm", "static_confirm", "feedback_confirm")
DETECTORS = (
    "completion_delay",
    "route_only",
    "mechanism_only",
    "delay_route",
    "delay_confirmed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--route-cache", type=Path, default=DEFAULT_ROUTE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-physical", action="store_true")
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


def empirical_midrank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirical percentile with ties placed at the middle of their mass."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if len(reference) < MIN_REFERENCE:
        return np.full(values.shape, np.nan, dtype=np.float32)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return ((left + right) / (2.0 * len(reference))).astype(np.float32)


def trailing_mean(values: np.ndarray, width: int = SMOOTHING_WIDTH) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, np.nan)
    for query in range(width - 1, values.shape[1]):
        window = values[:, query - width + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].mean(axis=1)
    return output


def nan_row_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmax(values[good], axis=1)
    return output


def max_with_fallback(base: np.ndarray, addition: np.ndarray) -> np.ndarray:
    output = np.asarray(base, dtype=np.float32).copy()
    good = np.isfinite(output) & np.isfinite(addition)
    output[good] = np.maximum(output[good], addition[good])
    return output


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_REFERENCE:
        return float("nan")
    return float(np.quantile(finite, quantile, method="higher"))


def extract_physical_features(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Extract causal client-side signals for one episode."""
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 8:
        raise ValueError(f"expected state [query,8], got {state.shape}")
    if actions.ndim != 3 or actions.shape[0] != len(state) or actions.shape[2] != 7:
        raise ValueError(f"expected actions [query,chunk,7], got {actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite client state/action value")

    length = len(state)
    output = np.full((length, len(PHYSICAL_FEATURES)), np.nan, dtype=np.float32)
    feature_index = {name: index for index, name in enumerate(PHYSICAL_FEATURES)}

    eef_delta = np.diff(state[:, :3], axis=0)
    eef_motion = np.linalg.norm(eef_delta, axis=1)
    gripper_motion = np.linalg.norm(np.diff(state[:, 6:8], axis=0), axis=1)
    command = np.linalg.norm(actions[:, :, :3], axis=-1).mean(axis=1)
    gripper_command = actions[:, :, 6].mean(axis=1)

    # Transition index q-1 is the realized change into state q; align it with
    # the already-issued action chunk q-1.
    for query in range(3, length):
        transitions = slice(query - 3, query)
        output[query, feature_index["eef_motion_w3"]] = float(
            eef_motion[transitions].mean()
        )
        output[query, feature_index["gripper_motion_w3"]] = float(
            gripper_motion[transitions].mean()
        )
        output[query, feature_index["prior_command_w3"]] = float(
            command[transitions].mean()
        )

    scaled_state = np.concatenate(
        (state[:, :3] / 0.05, state[:, 6:8] / 0.04), axis=1
    )
    for query in range(WARMUP_QUERY, length):
        recent_delta = eef_delta[query - 4 : query]
        left = recent_delta[:-1]
        right = recent_delta[1:]
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        eligible = (left_norm >= 0.001) & (right_norm >= 0.001)
        reversals = np.zeros(len(left), dtype=bool)
        if eligible.any():
            cosine = np.sum(left[eligible] * right[eligible], axis=1) / (
                left_norm[eligible] * right_norm[eligible]
            )
            reversals[eligible] = cosine < -0.25
        output[query, feature_index["eef_reversal_w3"]] = (
            float(reversals[eligible].mean()) if eligible.any() else 0.0
        )

        signs = np.where(gripper_command[query - 4 : query] >= 0.0, 1, -1)
        output[query, feature_index["gripper_flip_w4"]] = float(
            np.mean(signs[1:] != signs[:-1])
        )

        immediate = float(
            np.linalg.norm(scaled_state[query] - scaled_state[query - 1])
        )
        older = [
            float(np.linalg.norm(scaled_state[query] - scaled_state[query - lag]))
            for lag in range(2, 5)
        ]
        output[query, feature_index["state_return_w4"]] = max(
            immediate - distance for distance in older
        )

    if length > WARMUP_QUERY and not np.isfinite(output[WARMUP_QUERY:]).all():
        raise ValueError("non-finite eligible physical feature")
    return output


def build_physical_cache(
    route_cache: dict[str, np.ndarray], cache_root: Path, output_path: Path
) -> dict[str, np.ndarray]:
    task_names = np.asarray(route_cache["task_names"]).astype(str)
    task_index = np.asarray(route_cache["task_index"], dtype=np.int16)
    episodes = np.asarray(route_cache["episode"], dtype=np.int16)
    lengths = np.asarray(route_cache["length"], dtype=np.int16)
    valid = np.asarray(route_cache["valid"], dtype=bool)
    feature = np.full(
        (len(episodes), valid.shape[1], len(PHYSICAL_FEATURES)),
        np.nan,
        dtype=np.float32,
    )
    source_digest = hashlib.sha256()

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"expected 400 episodes for {task}, found {len(take)}")
        client = cache_root / task / RUN_ID / "client"
        for row in take:
            episode = int(episodes[row])
            path = client / f"episode_{episode:02d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                if "state" not in archive.files or "actions" not in archive.files:
                    raise ValueError(f"missing deployable arrays in {path}")
                state = np.asarray(archive["state"], dtype=np.float32)
                actions = np.asarray(archive["actions"], dtype=np.float32)
            if len(state) != int(lengths[row]):
                raise ValueError(f"client/route length mismatch in {path}")
            values = extract_physical_features(state, actions)
            feature[row, : len(values)] = values
            source_digest.update(task.encode("utf-8"))
            source_digest.update(np.asarray(episode, dtype="<i4").tobytes())
            source_digest.update(np.asarray(state, dtype="<f4").tobytes())
            source_digest.update(np.asarray(actions, dtype="<f4").tobytes())
        print(f"[physical {task_position + 1}/{len(task_names)}] {task}", flush=True)

    cache = {
        "schema": np.asarray("himoe.online_closed_loop.physical.v2"),
        "feature_names": np.asarray(PHYSICAL_FEATURES),
        "task_names": task_names,
        "task_index": task_index,
        "episode": episodes,
        "init_state_id": np.asarray(route_cache["init_state_id"], dtype=np.int16),
        "flow_noise_seed": np.asarray(route_cache["flow_noise_seed"], dtype=np.int16),
        "length": lengths,
        "valid": valid,
        "features": feature,
        "logical_source_sha256": np.asarray(source_digest.hexdigest()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)
    return cache


def load_physical_cache(path: Path, route_cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    if str(cache["schema"]) != "himoe.online_closed_loop.physical.v2":
        raise ValueError(f"unexpected physical cache schema in {path}")
    if tuple(cache["feature_names"].tolist()) != PHYSICAL_FEATURES:
        raise ValueError("physical feature order mismatch")
    for name in (
        "task_names",
        "task_index",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "length",
        "valid",
    ):
        if not np.array_equal(cache[name], route_cache[name]):
            raise ValueError(f"physical/route cache mismatch: {name}")
    return cache


def directed_rank(
    values: np.ndarray,
    direction: int,
    reference_episode: np.ndarray,
    valid: np.ndarray,
    start_query: int,
) -> np.ndarray:
    directed = direction * np.asarray(values, dtype=np.float32)
    output = np.full_like(directed, np.nan)
    for query in range(start_query, directed.shape[1]):
        reference = directed[reference_episode & valid[:, query], query]
        reference = reference[np.isfinite(reference)]
        take = valid[:, query] & np.isfinite(directed[:, query])
        output[take, query] = empirical_midrank(reference, directed[take, query])
    return output


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


def score_fold(
    route_features: np.ndarray,
    physical_features: np.ndarray,
    valid: np.ndarray,
    init_state: np.ndarray,
    lengths: np.ndarray,
    held_out: int,
    query_limit: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    train = init_state != held_out
    if int(train.sum()) != 392:
        raise ValueError(f"held-out fold must have 392 references, got {train.sum()}")
    completed_reference = train & (lengths < query_limit)
    if int(completed_reference.sum()) < MIN_REFERENCE:
        raise ValueError("too few completed references")

    route_index = {name: index for index, name in enumerate(v1.FEATURES)}
    route_ranks: dict[tuple[str, int], np.ndarray] = {}
    for member in sorted({member for members in v1.HEADS.values() for member in members}):
        name, direction = member
        route_ranks[member] = directed_rank(
            route_features[:, :, route_index[name]],
            direction,
            completed_reference,
            valid,
            v1.RAW_SCORE_START_QUERY,
        )

    route_heads: dict[str, np.ndarray] = {}
    for head, members in v1.HEADS.items():
        instant = complete_mean(np.stack([route_ranks[member] for member in members], axis=-1))
        route_heads[head] = trailing_mean(instant)

    physical_index = {name: index for index, name in enumerate(PHYSICAL_FEATURES)}
    physical_ranks = {
        "low_eef_motion": directed_rank(
            physical_features[:, :, physical_index["eef_motion_w3"]],
            -1,
            completed_reference,
            valid,
            WARMUP_QUERY,
        ),
        "high_prior_command": directed_rank(
            physical_features[:, :, physical_index["prior_command_w3"]],
            1,
            completed_reference,
            valid,
            WARMUP_QUERY,
        ),
        "high_eef_reversal": directed_rank(
            physical_features[:, :, physical_index["eef_reversal_w3"]],
            1,
            completed_reference,
            valid,
            WARMUP_QUERY,
        ),
        "high_gripper_flip": directed_rank(
            physical_features[:, :, physical_index["gripper_flip_w4"]],
            1,
            completed_reference,
            valid,
            WARMUP_QUERY,
        ),
        "high_state_return": directed_rank(
            physical_features[:, :, physical_index["state_return_w4"]],
            1,
            completed_reference,
            valid,
            WARMUP_QUERY,
        ),
    }
    physical_heads = {
        "motion_stall": complete_mean(
            np.stack(
                (physical_ranks["low_eef_motion"], physical_ranks["high_prior_command"]),
                axis=-1,
            )
        ),
        "physical_cycle": complete_mean(
            np.stack(
                (
                    physical_ranks["high_eef_reversal"],
                    physical_ranks["high_gripper_flip"],
                    physical_ranks["high_state_return"],
                ),
                axis=-1,
            )
        ),
    }

    static_route = np.maximum(
        route_heads["lock_in"], route_heads["flat_narrow_support"]
    )
    confirmations = {
        "loop_confirm": np.sqrt(
            np.clip(route_heads["instability"] * physical_heads["physical_cycle"], 0.0, 1.0)
        ),
        "static_confirm": np.sqrt(
            np.clip(static_route * physical_heads["motion_stall"], 0.0, 1.0)
        ),
        "feedback_confirm": np.sqrt(
            np.clip(
                route_heads["feedback_decoupling"] * physical_heads["motion_stall"],
                0.0,
                1.0,
            )
        ),
    }

    route_any = complete_max(np.stack(tuple(route_heads.values()), axis=-1))
    mechanism_any = complete_max(np.stack(tuple(confirmations.values()), axis=-1))

    completed_lengths = lengths[completed_reference]
    elapsed = np.arange(1, valid.shape[1] + 1, dtype=np.float32)
    completion_curve = empirical_midrank(completed_lengths, elapsed)
    completion_delay = np.broadcast_to(completion_curve, valid.shape).copy()
    completion_delay[~valid] = np.nan

    route_boost = np.sqrt(np.clip(completion_delay * route_any, 0.0, 1.0))
    mechanism_boost = np.sqrt(np.clip(completion_delay * mechanism_any, 0.0, 1.0))
    scores = {
        "completion_delay": completion_delay,
        "route_only": route_any,
        "mechanism_only": mechanism_any,
        "delay_route": max_with_fallback(completion_delay, route_boost),
        "delay_confirmed": max_with_fallback(completion_delay, mechanism_boost),
    }

    eligible = valid.copy()
    eligible[:, :WARMUP_QUERY] = False
    for collection in (route_heads, physical_heads, confirmations, scores):
        for name in collection:
            collection[name][~eligible] = np.nan
    return scores, route_heads, physical_heads, confirmations, train, completed_reference


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
    episodes = np.asarray(route_cache["episode"], dtype=np.int16)
    init_state = np.asarray(route_cache["init_state_id"], dtype=np.int16)
    flow_seed = np.asarray(route_cache["flow_noise_seed"], dtype=np.int16)
    lengths = np.asarray(route_cache["length"], dtype=np.int16)

    n_episode, n_query = valid.shape
    detector_scores = np.full((n_episode, n_query, len(DETECTORS)), np.nan, dtype=np.float32)
    route_scores = np.full((n_episode, n_query, len(v1.HEADS)), np.nan, dtype=np.float32)
    physical_scores = np.full((n_episode, n_query, len(PHYSICAL_HEADS)), np.nan, dtype=np.float32)
    confirmation_scores = np.full((n_episode, n_query, len(CONFIRMATIONS)), np.nan, dtype=np.float32)
    thresholds_by_episode = np.full(
        (n_episode, len(DETECTORS), len(OPERATING_QUANTILES)), np.nan, dtype=np.float32
    )
    alarms = np.zeros(
        (n_episode, n_query, len(DETECTORS), len(OPERATING_QUANTILES)), dtype=bool
    )
    winning_confirmation = np.full((n_episode, n_query), -1, dtype=np.int8)
    threshold_rows: list[dict[str, Any]] = []

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        suite = task.split("/", 1)[0]
        query_limit = v1.TASK_QUERY_LIMITS[suite]
        for held_out in np.unique(init_state[take]):
            result = score_fold(
                route_features[take],
                physical_features[take],
                valid[take],
                init_state[take],
                lengths[take],
                int(held_out),
                query_limit,
            )
            scores, route_heads, physical_heads, confirmations, train, completed = result
            test = ~train
            global_test = take[test]
            route_scores[global_test] = np.stack(
                [route_heads[name][test] for name in v1.HEADS], axis=-1
            )
            physical_scores[global_test] = np.stack(
                [physical_heads[name][test] for name in PHYSICAL_HEADS], axis=-1
            )
            confirmation_scores[global_test] = np.stack(
                [confirmations[name][test] for name in CONFIRMATIONS], axis=-1
            )
            typed = np.stack([confirmations[name] for name in CONFIRMATIONS], axis=-1)
            typed_good = np.isfinite(typed).all(axis=-1)
            local_winner = np.full(valid[take].shape, -1, dtype=np.int8)
            local_winner[typed_good] = np.argmax(typed[typed_good], axis=-1).astype(np.int8)
            winning_confirmation[global_test] = local_winner[test]

            for detector_position, detector in enumerate(DETECTORS):
                values = scores[detector]
                detector_scores[global_test, :, detector_position] = values[test]
                reference_max = nan_row_max(values[completed])
                for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                    threshold = quantile_higher(reference_max, quantile)
                    thresholds_by_episode[
                        global_test, detector_position, quantile_position
                    ] = threshold
                    trigger = np.isfinite(values[test]) & (values[test] > threshold)
                    alarms[global_test, :, detector_position, quantile_position] = (
                        np.maximum.accumulate(trigger, axis=1) & valid[take][test]
                    )
                    threshold_rows.append(
                        {
                            "task": task,
                            "held_out_init_state_id": int(held_out),
                            "detector": detector,
                            "quantile": quantile,
                            "threshold": threshold,
                            "historical_episodes": int(train.sum()),
                            "completed_reference_episodes": int(completed.sum()),
                            "censored_reference_episodes": int((train & ~completed).sum()),
                            "finite_reference_maxima": int(np.isfinite(reference_max).sum()),
                        }
                    )
        print(f"[calibration {task_position + 1}/{len(task_names)}] {task}", flush=True)

    if not np.isfinite(thresholds_by_episode).all():
        raise ValueError("non-finite detector threshold")
    if not np.isfinite(detector_scores[:, WARMUP_QUERY:]).any(axis=(1, 2)).all():
        raise ValueError("at least one episode has no eligible detector score")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sealed_online_scores.npz",
        schema=np.asarray("himoe.online_closed_loop.sealed.v2"),
        task_names=task_names,
        task_index=task_index,
        episode=episodes,
        init_state_id=init_state,
        flow_noise_seed=flow_seed,
        length=lengths,
        valid=valid,
        detector_names=np.asarray(DETECTORS),
        route_head_names=np.asarray(tuple(v1.HEADS)),
        physical_head_names=np.asarray(PHYSICAL_HEADS),
        confirmation_names=np.asarray(CONFIRMATIONS),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        scores=detector_scores,
        route_head_scores=route_scores,
        physical_head_scores=physical_scores,
        confirmation_scores=confirmation_scores,
        thresholds=thresholds_by_episode,
        alarms=alarms,
        winning_confirmation=winning_confirmation,
    )
    pd.DataFrame(threshold_rows).to_csv(output_dir / "unlabeled_thresholds.csv", index=False)

    rows: list[dict[str, Any]] = []
    for row in range(n_episode):
        item: dict[str, Any] = {
            "task": task_names[task_index[row]],
            "episode": int(episodes[row]),
            "init_state_id": int(init_state[row]),
            "flow_noise_seed": int(flow_seed[row]),
            "length": int(lengths[row]),
        }
        for detector_position, detector in enumerate(DETECTORS):
            for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                indices = np.flatnonzero(alarms[row, :, detector_position, quantile_position])
                suffix = str(quantile).replace("0.", "q")
                item[f"first_{detector}_{suffix}"] = int(indices[0]) if len(indices) else -1
        primary = np.flatnonzero(
            alarms[row, :, DETECTORS.index("delay_confirmed"), 1]
        )
        winner = int(winning_confirmation[row, primary[0]]) if len(primary) else -1
        item["primary_winning_confirmation"] = (
            CONFIRMATIONS[winner] if winner >= 0 else "completion_delay_only"
        )
        rows.append(item)
    pd.DataFrame(rows).to_csv(output_dir / "sealed_episode_alarms.csv", index=False)


def self_test() -> None:
    reference = np.asarray([0.0] * 20 + [1.0] * 20)
    observed = empirical_midrank(reference, np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(observed, [0.0, 0.25, 0.5, 0.75, 1.0])

    state = np.zeros((6, 8), dtype=np.float32)
    state[:, 0] = [0.0, 0.01, 0.02, 0.01, 0.0, 0.01]
    state[:, 6] = 0.04
    state[:, 7] = -0.04
    actions = np.zeros((6, 10, 7), dtype=np.float32)
    actions[:, :, 0] = 0.5
    actions[:, :, 6] = np.asarray([-1, -1, 1, 1, -1, -1])[:, None]
    feature = extract_physical_features(state, actions)
    index = {name: i for i, name in enumerate(PHYSICAL_FEATURES)}
    np.testing.assert_allclose(feature[4, index["eef_motion_w3"]], 0.01)
    np.testing.assert_allclose(feature[4, index["prior_command_w3"]], 0.5)
    np.testing.assert_allclose(feature[4, index["eef_reversal_w3"]], 1 / 3)
    np.testing.assert_allclose(feature[4, index["gripper_flip_w4"]], 1 / 3)
    assert feature[4, index["state_return_w4"]] > 0.0
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (args.output / "sealed_manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed result: {args.output}")
    route_cache = v1.load_feature_cache(args.route_cache)
    if len(route_cache["episode"]) != v1.EXPECTED["main"][1]:
        raise ValueError("route cache is not the frozen 14,800-episode cohort")

    args.output.mkdir(parents=True, exist_ok=True)
    physical_path = args.output / "unlabeled_physical_features.npz"
    if args.reuse_physical and physical_path.exists():
        physical_cache = load_physical_cache(physical_path, route_cache)
        print(f"reused {physical_path}", flush=True)
    else:
        physical_cache = build_physical_cache(route_cache, args.cache_root, physical_path)
    calibrate_and_replay(route_cache, physical_cache, args.output)

    artifacts = {
        "protocol_sha256": sha256(PROTOCOL),
        "scorer_sha256": sha256(Path(__file__)),
        "v1_feature_code_sha256": sha256(HERE / "online_multihead_alarm.py"),
        "route_feature_cache_sha256": sha256(args.route_cache),
        "physical_feature_cache_sha256": sha256(physical_path),
        "sealed_scores_sha256": sha256(args.output / "sealed_online_scores.npz"),
        "thresholds_sha256": sha256(args.output / "unlabeled_thresholds.csv"),
        "episode_alarms_sha256": sha256(args.output / "sealed_episode_alarms.csv"),
    }
    manifest = {
        "schema": "himoe.online_closed_loop.manifest.v2",
        "cohort": "main",
        "run_id": RUN_ID,
        "tasks": int(len(route_cache["task_names"])),
        "episodes": int(len(route_cache["episode"])),
        "route_features": list(v1.FEATURES),
        "physical_features": list(PHYSICAL_FEATURES),
        "route_heads": plain(v1.HEADS),
        "physical_heads": list(PHYSICAL_HEADS),
        "confirmations": list(CONFIRMATIONS),
        "detectors": list(DETECTORS),
        "primary_detector": "delay_confirmed",
        "primary_quantile": 0.95,
        "operating_quantiles": list(OPERATING_QUANTILES),
        "minimum_reference_episodes": MIN_REFERENCE,
        "warmup_query": WARMUP_QUERY,
        "calibration": "leave-one-init-state-out within task; completed historical references only",
        "censoring_rule": "historical inference_calls < configured suite query limit",
        "censoring_uses_reference_termination_time": True,
        "held_out_eventual_length_used_by_online_update": False,
        "query_causal": True,
        "future_queries_used_by_online_update": False,
        "labels_used": [],
        "summary_fields_used": [
            "episode_index",
            "init_state_id",
            "flow_noise_seed",
            "inference_calls",
        ],
        "client_arrays_used": ["state", "actions"],
        "client_arrays_explicitly_excluded": ["sim_state"],
        "logical_client_source_sha256": str(physical_cache["logical_source_sha256"]),
        "artifacts": artifacts,
    }
    (args.output / "sealed_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"sealed {manifest['episodes']} episodes across {manifest['tasks']} tasks at {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
