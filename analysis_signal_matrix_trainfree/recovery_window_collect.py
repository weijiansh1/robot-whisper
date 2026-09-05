#!/usr/bin/env python3
"""Retrospective snapshot-fork test around a physical loop onset.

This is deliberately not an online detector experiment.  A deterministic trunk
is run to completion while every query boundary is snapshotted.  Its first loop
onset is then defined only from the physical trajectory, and fresh-noise
continuations are launched from onset-4, onset-2, and onset.  The same candidate
noise streams are reused at every fork time, making the recovery-window
comparison paired.

The server must run ``serve_with_recorder.py --return-full-probs``.  The returned
full 32-way probabilities let this client record the exact two train-free MoE
signals evaluated by ``run.py`` without reading a live Zarr store.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUTE_CAPTURE = ROOT / "himoe-route-capture"
sys.path.insert(0, str(ROOT / "himoe-libero-wrist-fix/src"))
sys.path.insert(0, str(ROUTE_CAPTURE))

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    validate_action_response,
)

from branch_snapshot import restore_full_state, save_full_state  # noqa: E402
from rollout_with_routes import sim_joint_layout  # noqa: E402
from rolling_star_collect import (  # noqa: E402
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _run_branch,
    _save_snapshot_state,
    _write_video,
    trunk_noise,
)
from triggered_fork_collect import episode_id_for  # noqa: E402


SCHEMA = "himoe.recovery_window.v1"
FULL_PROBS_KEY = "recorder/hb_router_probs"
DEEP_LAYERS = slice(4, 8)
ACTION_TOKENS = slice(1, 11)
LATE_FLOW_START = 6
TRAILING_WIDTH = 4
DEFAULT_OFFSETS = (-4, -2, 0)
# Kept fixed across fork times so candidate k receives the same relative noise
# stream at every intervention time.
PAIRED_NOISE_STREAM_KEY = 9_731


def normalize_probability(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.minimum(left, right).sum(axis=-1)
    denominator = np.maximum(left, right).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def route_scalars(probabilities: np.ndarray) -> Tuple[float, float]:
    """Match ``run.py`` exactly for one query's two within-flow signals."""

    expected = (8, 10, 11, 32)
    if probabilities.shape != expected:
        raise RuntimeError("full router probabilities must be %s, got %s" % (
            expected, probabilities.shape
        ))
    p = normalize_probability(probabilities)
    action = p[DEEP_LAYERS, :, ACTION_TOKENS, :]
    late = action[:, LATE_FLOW_START:, :, :]
    volatility = float(
        (1.0 - weighted_jaccard(late[:, 1:], late[:, :-1])).mean()
    )
    root = np.sqrt(np.maximum(action, 0.0))
    acceleration = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
    acceleration_value = float(
        np.linalg.norm(acceleration, axis=-1).mean() / np.sqrt(2.0)
    )
    return volatility, acceleration_value


def trailing_mean(values: Sequence[float], width: int = TRAILING_WIDTH) -> List[float]:
    output: List[float] = []
    for right in range(len(values)):
        if right + 1 < width:
            output.append(float("nan"))
        else:
            output.append(float(np.mean(values[right - width + 1 : right + 1])))
    return output


def route_mobility(probabilities: Sequence[np.ndarray]) -> List[float]:
    """d9 Hellinger mobility, retained as the explicit MoE baseline."""

    raw = [float("nan")]
    for previous, current in zip(probabilities[:-1], probabilities[1:]):
        left = normalize_probability(previous)[DEEP_LAYERS, 9, ACTION_TOKENS]
        right = normalize_probability(current)[DEEP_LAYERS, 9, ACTION_TOKENS]
        coefficient = np.sqrt(np.maximum(left, 0.0) * np.maximum(right, 0.0)).sum(-1)
        raw.append(float(np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0)).mean()))
    return trailing_mean(raw)


def goal_distance(objects: np.ndarray, references: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(
        objects[:, :, None, :] - references[None, None, :, :], axis=-1
    ).min(axis=-1)
    return distance.max(axis=1)


def loop_onset_query(
    eef: np.ndarray,
    objects: np.ndarray,
    gripper: np.ndarray,
    references: np.ndarray,
) -> Tuple[int, int]:
    """Frozen query-resolution rule from ``run.py::physical_onsets``."""

    goal = goal_distance(objects, references)
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    for right in range(3, len(eef)):
        for left in range(0, right - 2):
            if (
                np.linalg.norm(eef[right] - eef[left]) <= 0.045
                and np.linalg.norm(objects[right] - objects[left], axis=1).max()
                <= 0.030
                and abs(gripper[right] - gripper[left]) <= 0.012
                and cumulative[right] - cumulative[left] >= 0.120
                and goal[left] - goal[right] <= 0.035
            ):
                return right, left
    return -1, -1


def request_with_full_routes(
    client: PolicyClient,
    policy_observation: Mapping[str, Any],
    noise: np.ndarray,
    episode_id: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    request = dict(policy_observation)
    request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request["episode_id"] = int(episode_id)
    response = validate_action_response(client.infer(request))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    if FULL_PROBS_KEY not in response:
        raise RuntimeError(
            "policy response has no %s; launch serve_with_recorder.py with "
            "--return-full-probs" % FULL_PROBS_KEY
        )
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    probabilities = np.asarray(response[FULL_PROBS_KEY], dtype=np.float32)
    if actions.shape != (10, 7):
        raise RuntimeError("unexpected action shape %s" % (actions.shape,))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("router probabilities contain NaN or infinity")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=2e-3, rtol=2e-3):
        raise RuntimeError("router probabilities do not sum to one")
    return actions, probabilities, float(response.get("server/inference_ms", np.nan))


def parse_offsets(value: str) -> Tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets or len(set(offsets)) != len(offsets) or any(item > 0 for item in offsets):
        raise argparse.ArgumentTypeError("offsets must be distinct comma-separated integers <= 0")
    return offsets


def load_references(path: pathlib.Path) -> np.ndarray:
    definitions = json.loads(path.read_text(encoding="utf-8"))
    references = np.asarray(
        definitions["success_terminal_references"]["moka_pot_1_joint0"],
        dtype=np.float64,
    )
    if references.ndim != 2 or references.shape[1] != 3:
        raise RuntimeError("terminal references have the wrong shape")
    return references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-id", type=int, default=7)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--max-trunk-queries", type=int, default=52)
    parser.add_argument("--fork-offsets", type=parse_offsets, default=DEFAULT_OFFSETS)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--physical-definitions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("k must be positive")
    if args.replan_steps != 10:
        raise ValueError("this experiment freezes replan_steps=10")

    out = pathlib.Path(args.out)
    if (out / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite completed run %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    references = load_references(pathlib.Path(args.physical_definitions))
    started = time.time()
    config = {
        "schema": SCHEMA,
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "worker_id": args.worker_id,
        "k": args.k,
        "seed": args.seed,
        "environment_seed": args.environment_seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "fork_offsets": list(args.fork_offsets),
        "paired_noise_stream_key": PAIRED_NOISE_STREAM_KEY,
        "onset_source": "physical query trajectory only",
        "training": False,
    }
    _atomic_json(out / "experiment_config.json", config)

    environment, observation, task, prompt = _load_task(EpisodeConfig(
        task_suite=args.benchmark,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.environment_seed,
        host=args.host,
        port=args.port,
        libero_root=args.libero_root,
        output_root=str(out),
        settle_steps=args.settle_steps,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        inference_timeout=args.inference_timeout,
    ))

    snapshots: List[Mapping[str, Any]] = []
    policy_observations: List[Mapping[str, Any]] = []
    action_steps_at_query: List[int] = []
    probabilities: List[np.ndarray] = []
    action_chunks: List[np.ndarray] = []
    policy_states: List[np.ndarray] = []
    sim_states: List[np.ndarray] = []
    inference_ms: List[float] = []
    frames: List[np.ndarray] = []
    trunk_success = False
    action_steps = 0

    try:
        for _ in range(args.settle_steps):
            observation, _reward, _done, _info = environment.step(
                LIBERO_DUMMY_ACTION.tolist()
            )
        _atomic_json(out / "sim_layout.json", sim_joint_layout(environment))

        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, args.benchmark)
            if client.metadata.get("full_router_probs_response_key") != FULL_PROBS_KEY:
                raise RuntimeError("server does not advertise the full-probability response")
            _atomic_json(out / "server_metadata.json", dict(client.metadata))

            print("trunk", flush=True)
            for query in range(args.max_trunk_queries):
                if action_steps >= args.max_steps or trunk_success:
                    break
                policy_observation = build_policy_observation(observation, prompt)
                snapshot = save_full_state(environment)
                snapshots.append(snapshot)
                policy_observations.append(dict(policy_observation))
                action_steps_at_query.append(action_steps)
                policy_states.append(np.asarray(
                    policy_observation["observation/state"], dtype=np.float32
                ))
                sim_states.append(np.asarray(environment.get_sim_state(), dtype=np.float32))

                noise = trunk_noise(args.seed, args.task_id, args.init_state_id, query)
                actions, route, server_ms = request_with_full_routes(
                    client,
                    policy_observation,
                    noise,
                    episode_id_for(args.worker_id, 0, 0),
                )
                probabilities.append(route.astype(np.float16))
                action_chunks.append(actions)
                inference_ms.append(server_ms)
                volatility, acceleration = route_scalars(route)
                print(
                    "  q%02d  vol=%.6f  acc=%.6f" % (query, volatility, acceleration),
                    flush=True,
                )

                take = min(args.replan_steps, len(actions), args.max_steps - action_steps)
                for index in range(take):
                    observation, _reward, _done, _info = environment.step(
                        actions[index].tolist()
                    )
                    action_steps += 1
                    frames.append(np.ascontiguousarray(
                        np.asarray(observation["agentview_image"], dtype=np.uint8)[::-1, ::-1]
                    ))
                    trunk_success = bool(environment.check_success())
                    if trunk_success:
                        break

            route_array = np.stack(probabilities).astype(np.float16)
            state_array = np.stack(policy_states)
            sim_array = np.stack(sim_states)
            action_array = np.stack(action_chunks)
            raw_volatility = []
            raw_acceleration = []
            for route in route_array:
                volatility, acceleration = route_scalars(route.astype(np.float32))
                raw_volatility.append(volatility)
                raw_acceleration.append(acceleration)
            volatility_w4 = trailing_mean(raw_volatility)
            acceleration_w4 = trailing_mean(raw_acceleration)
            mobility_w4 = route_mobility([row.astype(np.float32) for row in route_array])

            eef = state_array[:, :3].astype(np.float64)
            objects = np.stack((sim_array[:, 10:13], sim_array[:, 17:20]), axis=1).astype(
                np.float64
            )
            gripper = state_array[:, 6:8].mean(axis=1).astype(np.float64)
            onset, partner = loop_onset_query(eef, objects, gripper, references)

            _atomic_npz(
                out / "trunk.npz",
                {
                    "hb_router_probs": route_array,
                    "policy_state": state_array,
                    "sim_state": sim_array,
                    "action_chunks": action_array,
                    "action_steps_at_query": np.asarray(action_steps_at_query, dtype=np.int32),
                    "late_flow_volatility": np.asarray(raw_volatility, dtype=np.float32),
                    "late_flow_volatility_w4": np.asarray(volatility_w4, dtype=np.float32),
                    "route_acceleration": np.asarray(raw_acceleration, dtype=np.float32),
                    "route_acceleration_w4": np.asarray(acceleration_w4, dtype=np.float32),
                    "route_mobility_w4": np.asarray(mobility_w4, dtype=np.float32),
                    "server_inference_ms": np.asarray(inference_ms, dtype=np.float32),
                },
            )
            if frames:
                _write_video(
                    out / ("trunk_%s.mp4" % ("success" if trunk_success else "failure")),
                    frames,
                )

            trunk_record = {
                "success": bool(trunk_success),
                "queries": len(probabilities),
                "action_steps": action_steps,
                "loop_onset_query": onset,
                "loop_partner_query": partner,
                "late_flow_volatility_w4_at_onset_minus_2": (
                    None if onset < 2 else volatility_w4[onset - 2]
                ),
                "route_acceleration_w4_at_onset_minus_2": (
                    None if onset < 2 else acceleration_w4[onset - 2]
                ),
                "route_mobility_w4_at_onset_minus_2": (
                    None if onset < 2 else mobility_w4[onset - 2]
                ),
            }
            _atomic_json(out / "trunk.json", trunk_record)
            print(
                "trunk: success=%s queries=%d loop_onset=%d partner=%d"
                % (trunk_success, len(probabilities), onset, partner),
                flush=True,
            )

            if trunk_success or onset < 0:
                status = "trunk_succeeded" if trunk_success else "no_physical_loop"
                _atomic_json(out / "manifest.json", {
                    "schema": SCHEMA,
                    "status": status,
                    "task": {"name": str(task.name), "prompt": prompt},
                    "trunk": trunk_record,
                    "forks": {},
                    "wall_s": round(time.time() - started, 1),
                })
                return 0

            fork_results: Dict[str, Any] = {}
            for offset_index, offset in enumerate(args.fork_offsets):
                fork_query = onset + offset
                if fork_query < 0 or fork_query >= len(snapshots):
                    continue
                directory = out / ("fork_%+d" % offset)
                directory.mkdir(parents=True, exist_ok=True)
                _save_snapshot_state(
                    directory,
                    snapshots[fork_query],
                    policy_observations[fork_query],
                )
                candidate_summaries = []
                print("fork offset=%+d q=%d" % (offset, fork_query), flush=True)
                for candidate in range(args.k):
                    restored = restore_full_state(environment, snapshots[fork_query])
                    first_observation = build_policy_observation(restored, prompt)
                    summary, arrays, branch_frames = _run_branch(
                        environment,
                        first_observation,
                        prompt,
                        client,
                        args.seed,
                        args.task_id,
                        args.init_state_id,
                        PAIRED_NOISE_STREAM_KEY,
                        candidate,
                        episode_id_for(args.worker_id, offset_index + 1, candidate),
                        action_steps_at_query[fork_query],
                        args.max_steps,
                        args.replan_steps,
                        candidate < 2,
                    )
                    summary.update({
                        "fork_query": fork_query,
                        "fork_offset": offset,
                        "noise_stream_key": PAIRED_NOISE_STREAM_KEY,
                    })
                    _atomic_npz(directory / ("candidate_%02d.npz" % candidate), arrays)
                    _atomic_json(directory / ("candidate_%02d.json" % candidate), summary)
                    if branch_frames:
                        outcome = "success" if summary["success"] else "failure"
                        _write_video(
                            directory / ("c%02d_%s.mp4" % (candidate, outcome)),
                            branch_frames,
                        )
                    candidate_summaries.append(summary)
                    print(
                        "  c%02d %s queries=%d" % (
                            candidate,
                            "SUCCESS" if summary["success"] else "failure",
                            summary["inference_calls"],
                        ),
                        flush=True,
                    )
                successes = int(sum(bool(item["success"]) for item in candidate_summaries))
                fork_results[str(offset)] = {
                    "offset": offset,
                    "query": fork_query,
                    "n": len(candidate_summaries),
                    "successes": successes,
                    "candidate_success": [
                        bool(item["success"]) for item in candidate_summaries
                    ],
                }

            _atomic_json(out / "manifest.json", {
                "schema": SCHEMA,
                "status": "complete",
                "task": {
                    "benchmark": args.benchmark,
                    "task_id": args.task_id,
                    "init_state_id": args.init_state_id,
                    "name": str(task.name),
                    "prompt": prompt,
                },
                "design": {
                    "training": False,
                    "onset_source": "physical query trajectory only",
                    "fork_offsets": list(args.fork_offsets),
                    "paired_candidate_noise_across_offsets": True,
                    "paired_noise_stream_key": PAIRED_NOISE_STREAM_KEY,
                    "trunk_continuation_is_control": True,
                },
                "trunk": trunk_record,
                "forks": fork_results,
                "wall_s": round(time.time() - started, 1),
            })
            print("recovery window complete: %s" % {
                key: "%d/%d" % (item["successes"], item["n"])
                for key, item in fork_results.items()
            }, flush=True)
    finally:
        try:
            environment.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
