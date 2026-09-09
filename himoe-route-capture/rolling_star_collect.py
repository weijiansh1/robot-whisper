"""Collect terminal K-way counterfactual rollouts at every trunk query.

The model server owns the route recorder.  This client owns one LIBERO
environment and writes one atomic artifact per terminal branch.  A full
simulator/controller snapshot is restored before every branch, so branches at
one trunk query differ only through their explicitly recorded flow-noise stream.

Run this file in the Python 3.8 LIBERO environment.  Multiple instances may
share one model server when each has a distinct ``--worker-id`` and output
directory; the synchronous policy server serializes recorder access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import imageio
import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

from branch_snapshot import decode_full_state, encode_full_state, restore_full_state, save_full_state
from rollout_with_routes import sim_joint_layout


SCHEMA = "himoe.rolling_star.v1"
EPISODE_ID_KEY = "episode_id"
TRUNK_ROLE = 0x5452554E  # TRUN
BRANCH_ROLE = 0x4252414E  # BRAN
ORDER_ROLE = 0x4F524445  # ORDE


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _atomic_npz(path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _noise(seed_words: Sequence[int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(value) for value in seed_words]))
    return rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)


def trunk_noise(seed: int, task_id: int, init_state: int, query: int) -> np.ndarray:
    return _noise((seed, TRUNK_ROLE, task_id, init_state, query))


def branch_noise(
    seed: int,
    task_id: int,
    init_state: int,
    snapshot: int,
    candidate: int,
    query: int,
) -> np.ndarray:
    return _noise((seed, BRANCH_ROLE, task_id, init_state, snapshot, candidate, query))


def candidate_order(seed: int, task_id: int, init_state: int, snapshot: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, ORDER_ROLE, task_id, init_state, snapshot))
    )
    return np.asarray(rng.permutation(k), dtype=np.int32)


def branch_episode_id(worker_id: int, snapshot: int, candidate: int) -> int:
    value = (worker_id + 1) * 100_000_000 + snapshot * 100 + candidate
    if not 0 <= value <= np.iinfo(np.int32).max:
        raise ValueError("branch episode id does not fit int32")
    return value


def trunk_episode_id(worker_id: int, snapshot: int) -> int:
    value = (worker_id + 1) * 100_000_000 + 90_000_000 + snapshot
    if not 0 <= value <= np.iinfo(np.int32).max:
        raise ValueError("trunk episode id does not fit int32")
    return value


def _frame(observation: Mapping[str, Any]) -> np.ndarray:
    image = np.asarray(observation["agentview_image"], dtype=np.uint8)
    return np.ascontiguousarray(image[::-1, ::-1])


def _write_video(path: pathlib.Path, frames: Sequence[np.ndarray], fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    imageio.mimwrite(str(temporary), [np.asarray(frame) for frame in frames], fps=fps)
    os.replace(str(temporary), str(path))


def _save_snapshot_state(
    directory: pathlib.Path,
    snapshot: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
) -> Dict[str, str]:
    metadata, arrays = encode_full_state(snapshot)
    meta_path = directory / "full_state.json"
    arrays_path = directory / "full_state.npz"
    input_path = directory / "policy_input.npz"
    if not meta_path.exists():
        _atomic_npz(arrays_path, arrays)
        _atomic_json(meta_path, metadata)
        _atomic_npz(
            input_path,
            {
                "image": np.asarray(policy_observation["observation/image"], dtype=np.uint8),
                "wrist_image": np.asarray(
                    policy_observation["observation/wrist_image"], dtype=np.uint8
                ),
                "state": np.asarray(policy_observation["observation/state"], dtype=np.float32),
            },
        )
    return {
        "full_state_json_sha256": _sha256_file(meta_path),
        "full_state_npz_sha256": _sha256_file(arrays_path),
        "policy_input_sha256": _sha256_file(input_path),
    }


def _load_snapshot_state(directory: pathlib.Path) -> Dict[str, Any]:
    metadata = json.loads((directory / "full_state.json").read_text(encoding="utf-8"))
    with np.load(str(directory / "full_state.npz"), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return decode_full_state(metadata, arrays)


def _request(
    client: PolicyClient,
    policy_observation: Mapping[str, Any],
    noise: np.ndarray,
    episode_id: int,
) -> Tuple[np.ndarray, float]:
    request = dict(policy_observation)
    request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request[EPISODE_ID_KEY] = int(episode_id)
    response = validate_action_response(client.infer(request))
    expected = _array_sha256(noise)
    if response.get(FLOW_NOISE_SHA256_KEY) != expected:
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError("unexpected action shape %s" % (actions.shape,))
    return actions, float(response.get("server/inference_ms", np.nan))


def _run_branch(
    environment: Any,
    first_policy_observation: Mapping[str, Any],
    prompt: str,
    client: PolicyClient,
    seed: int,
    task_id: int,
    init_state: int,
    snapshot_index: int,
    candidate: int,
    episode_id: int,
    start_action_steps: int,
    max_steps: int,
    replan_steps: int,
    capture_frames: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[np.ndarray]]:
    policy_observation = dict(first_policy_observation)
    action_steps = int(start_action_steps)
    success = bool(environment.check_success())
    query = 0
    flow_noises: List[np.ndarray] = []
    policy_states: List[np.ndarray] = []
    sim_states: List[np.ndarray] = []
    action_chunks: List[np.ndarray] = []
    executed_masks: List[np.ndarray] = []
    success_after_query: List[bool] = []
    inference_ms: List[float] = []
    frames: List[np.ndarray] = []
    initial_state = np.asarray(
        policy_observation["observation/state"], dtype=np.float32
    )
    control_sim_states: List[np.ndarray] = [
        np.asarray(environment.get_sim_state(), dtype=np.float32)
    ]
    control_eef_positions: List[np.ndarray] = [initial_state[:3].copy()]
    control_gripper_qpos: List[np.ndarray] = [initial_state[6:8].copy()]
    control_actions: List[np.ndarray] = []
    control_query_indices: List[int] = []
    control_success: List[bool] = []

    while action_steps < max_steps and not success:
        flow_noise = branch_noise(seed, task_id, init_state, snapshot_index, candidate, query)
        actions, server_ms = _request(
            client,
            policy_observation,
            flow_noise,
            episode_id,
        )
        flow_noises.append(flow_noise)
        policy_states.append(
            np.asarray(policy_observation["observation/state"], dtype=np.float32)
        )
        sim_states.append(np.asarray(environment.get_sim_state(), dtype=np.float32))
        chunk = np.zeros((replan_steps, 7), dtype=np.float32)
        mask = np.zeros((replan_steps,), dtype=np.bool_)
        take = min(replan_steps, len(actions), max_steps - action_steps)
        chunk[:take] = actions[:take]
        observation = None
        for action_index in range(take):
            observation, _reward, _done, _info = environment.step(actions[action_index].tolist())
            mask[action_index] = True
            action_steps += 1
            success = bool(environment.check_success())
            control_actions.append(np.asarray(actions[action_index], dtype=np.float32))
            control_query_indices.append(query)
            control_success.append(success)
            control_sim_states.append(
                np.asarray(environment.get_sim_state(), dtype=np.float32)
            )
            control_eef_positions.append(
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
            )
            control_gripper_qpos.append(
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
            )
            if capture_frames:
                frames.append(_frame(observation))
            if success:
                break
        action_chunks.append(chunk)
        executed_masks.append(mask)
        success_after_query.append(success)
        inference_ms.append(server_ms)
        query += 1
        if not success and action_steps < max_steps:
            if observation is None:
                raise RuntimeError("branch executed no environment action")
            policy_observation = build_policy_observation(observation, prompt)

    arrays = {
        "flow_noise": np.stack(flow_noises),
        "policy_state": np.stack(policy_states),
        "sim_state": np.stack(sim_states),
        "action_chunks": np.stack(action_chunks),
        "action_executed": np.stack(executed_masks),
        "success_after_query": np.asarray(success_after_query, dtype=np.bool_),
        "server_inference_ms": np.asarray(inference_ms, dtype=np.float32),
        # Dense physical controls are deliberately small and contain no model
        # hidden state.  Their A+1 state samples preserve chunk-internal drops,
        # returns, and the exact terminal state needed for outcome taxonomy.
        "control_sim_state": np.stack(control_sim_states),
        "control_eef_position": np.stack(control_eef_positions),
        "control_gripper_qpos": np.stack(control_gripper_qpos),
        "control_action": np.stack(control_actions),
        "control_query_index": np.asarray(control_query_indices, dtype=np.int32),
        "control_success_after_step": np.asarray(control_success, dtype=np.bool_),
    }
    summary = {
        "schema": SCHEMA,
        "snapshot_index": snapshot_index,
        "candidate": candidate,
        "episode_id": int(episode_id),
        "success": bool(success),
        "start_action_steps": int(start_action_steps),
        "terminal_action_steps": int(action_steps),
        "branch_action_steps": int(action_steps - start_action_steps),
        "inference_calls": int(query),
        "failure_type": None if success else "pending_physical_taxonomy",
    }
    return summary, arrays, frames


def _execute_recorded_trunk(environment: Any, path: pathlib.Path) -> Tuple[Any, bool, int]:
    with np.load(str(path), allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
        mask = np.asarray(archive["action_executed"], dtype=np.bool_)
    observation = None
    success = bool(environment.check_success())
    executed = 0
    for index, action in enumerate(actions):
        if not mask[index]:
            continue
        observation, _reward, _done, _info = environment.step(action.tolist())
        executed += 1
        success = bool(environment.check_success())
        if success:
            break
    return observation, success, executed


def _existing_snapshot_count(out: pathlib.Path) -> int:
    count = 0
    while (out / ("snapshot_%03d" % count) / "manifest.json").is_file():
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-id", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--max-trunk-queries", type=int, default=52)
    parser.add_argument("--stop-before-unix", type=float, default=0.0)
    parser.add_argument("--videos-per-outcome", type=int, default=2)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.k != 16:
        raise ValueError("this experiment freezes K=16")
    if not 0 <= args.worker_id <= 19:
        raise ValueError("worker-id must be in 0..19")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_value = {
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
        "routing": "server-side full probabilities; no hidden state",
    }
    config_path = out / "experiment_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config_value:
            raise RuntimeError("existing experiment config differs")
    else:
        _atomic_json(config_path, config_value)

    episode_config = EpisodeConfig(
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
    )
    environment, observation, task, prompt = _load_task(episode_config)
    action_steps = 0
    trunk_success = False
    try:
        for _ in range(args.settle_steps):
            observation, _reward, _done, _info = environment.step(LIBERO_DUMMY_ACTION.tolist())
        if not (out / "sim_layout.json").exists():
            _atomic_json(out / "sim_layout.json", sim_joint_layout(environment))

        completed = _existing_snapshot_count(out)
        for snapshot_index in range(completed):
            trunk_path = out / ("snapshot_%03d" % snapshot_index) / "trunk.npz"
            observation, trunk_success, executed = _execute_recorded_trunk(environment, trunk_path)
            action_steps += executed
            if trunk_success:
                break

        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, args.benchmark)
            if client.metadata.get("episode_id_key") != EPISODE_ID_KEY:
                raise RuntimeError("rolling-star capture requires the recorder episode-id contract")
            if not bool(client.metadata.get("store_full_probs", True)):
                raise RuntimeError("server is not configured for full route probabilities")
            _atomic_json(out / "server_metadata.json", dict(client.metadata))

            video_counts = {
                "success": len(list((out / "videos").glob("*success*.mp4"))),
                "failure": len(list((out / "videos").glob("*failure*.mp4"))),
            }
            snapshot_index = completed
            while (
                snapshot_index < args.max_trunk_queries
                and action_steps < args.max_steps
                and not trunk_success
            ):
                if args.stop_before_unix and time.time() >= args.stop_before_unix:
                    break
                started = time.time()
                directory = out / ("snapshot_%03d" % snapshot_index)
                directory.mkdir(parents=True, exist_ok=True)
                snapshot = save_full_state(environment)
                policy_observation = build_policy_observation(observation, prompt)
                snapshot_hashes = _save_snapshot_state(directory, snapshot, policy_observation)
                order = candidate_order(
                    args.seed, args.task_id, args.init_state_id, snapshot_index, args.k
                )
                candidate_summaries = []
                for candidate_value in order:
                    candidate = int(candidate_value)
                    json_path = directory / ("candidate_%02d.json" % candidate)
                    npz_path = directory / ("candidate_%02d.npz" % candidate)
                    if json_path.is_file() and npz_path.is_file():
                        candidate_summaries.append(
                            json.loads(json_path.read_text(encoding="utf-8"))
                        )
                        continue
                    restore_full_state(environment, snapshot)
                    episode_id = branch_episode_id(
                        args.worker_id, snapshot_index, candidate
                    )
                    want_frames = any(
                        video_counts[key] < args.videos_per_outcome
                        for key in ("success", "failure")
                    )
                    summary, arrays, frames = _run_branch(
                        environment,
                        policy_observation,
                        prompt,
                        client,
                        args.seed,
                        args.task_id,
                        args.init_state_id,
                        snapshot_index,
                        candidate,
                        episode_id,
                        action_steps,
                        args.max_steps,
                        args.replan_steps,
                        want_frames,
                    )
                    _atomic_npz(npz_path, arrays)
                    summary["npz_sha256"] = _sha256_file(npz_path)
                    _atomic_json(json_path, summary)
                    candidate_summaries.append(summary)
                    outcome = "success" if summary["success"] else "failure"
                    if frames and video_counts[outcome] < args.videos_per_outcome:
                        video_path = out / "videos" / (
                            "s%03d_c%02d_%s.mp4" % (snapshot_index, candidate, outcome)
                        )
                        _write_video(video_path, frames)
                        video_counts[outcome] += 1
                    print(
                        "worker=%d snapshot=%d candidate=%d success=%s queries=%d %.1fs"
                        % (
                            args.worker_id,
                            snapshot_index,
                            candidate,
                            summary["success"],
                            summary["inference_calls"],
                            float(np.nansum(arrays["server_inference_ms"])) / 1000.0,
                        ),
                        flush=True,
                    )

                restore_full_state(environment, snapshot)
                trunk_flow = trunk_noise(
                    args.seed, args.task_id, args.init_state_id, snapshot_index
                )
                trunk_actions, trunk_server_ms = _request(
                    client,
                    policy_observation,
                    trunk_flow,
                    trunk_episode_id(args.worker_id, snapshot_index),
                )
                take = min(args.replan_steps, len(trunk_actions), args.max_steps - action_steps)
                mask = np.zeros((args.replan_steps,), dtype=np.bool_)
                saved_actions = np.zeros((args.replan_steps, 7), dtype=np.float32)
                saved_actions[:take] = trunk_actions[:take]
                for action_index in range(take):
                    observation, _reward, _done, _info = environment.step(
                        trunk_actions[action_index].tolist()
                    )
                    mask[action_index] = True
                    action_steps += 1
                    trunk_success = bool(environment.check_success())
                    if trunk_success:
                        break
                trunk_path = directory / "trunk.npz"
                _atomic_npz(
                    trunk_path,
                    {
                        "flow_noise": trunk_flow,
                        "actions": saved_actions,
                        "action_executed": mask,
                        "server_inference_ms": np.asarray(trunk_server_ms, dtype=np.float32),
                    },
                )
                candidate_summaries.sort(key=lambda item: int(item["candidate"]))
                manifest = {
                    "schema": SCHEMA,
                    "status": "complete",
                    "snapshot_index": snapshot_index,
                    "task_name": str(task.name),
                    "prompt": prompt,
                    "trunk_action_steps_before": action_steps - int(mask.sum()),
                    "trunk_action_steps_after": action_steps,
                    "trunk_success_after": trunk_success,
                    "candidate_execution_order": order.tolist(),
                    "candidate_successes": int(
                        sum(bool(item["success"]) for item in candidate_summaries)
                    ),
                    "candidate_failures": int(
                        sum(not bool(item["success"]) for item in candidate_summaries)
                    ),
                    "candidate_inference_calls": int(
                        sum(int(item["inference_calls"]) for item in candidate_summaries)
                    ),
                    "candidate_summaries": candidate_summaries,
                    "trunk_npz_sha256": _sha256_file(trunk_path),
                    "wall_s": time.time() - started,
                    **snapshot_hashes,
                }
                _atomic_json(directory / "manifest.json", manifest)
                _atomic_json(
                    out / "progress.json",
                    {
                        "schema": SCHEMA,
                        "completed_snapshots": snapshot_index + 1,
                        "trunk_action_steps": action_steps,
                        "trunk_success": trunk_success,
                        "last_snapshot_wall_s": manifest["wall_s"],
                        "updated_unix": time.time(),
                    },
                )
                snapshot_index += 1
                print(
                    "COMMITTED worker=%d snapshot=%d mixed=%d/%d wall=%.1fs"
                    % (
                        args.worker_id,
                        snapshot_index - 1,
                        manifest["candidate_successes"],
                        args.k,
                        manifest["wall_s"],
                    ),
                    flush=True,
                )
    finally:
        try:
            environment.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
