#!/usr/bin/env python3
"""Collect paired full-horizon and short-burst recovery continuations."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "himoe-route-capture"
BRIDGE_SRC = Path("/home/jovyan/work/himoe-libero-wrist-fix/src")
for source in (CAPTURE_ROOT, BRIDGE_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from branch_snapshot import save_full_state  # noqa: E402
from capture_behavior_forks import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_npz,
    _full_state_sha256,
    _hard_restore,
)
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
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    validate_action_response,
)


SCHEMA = "himoe.moe_horizon_recovery.v1"
CHECKPOINT_SHA256 = "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256"
TASK_NAME = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
PROMPT = "put both moka pots on the stove"
LEAD = -4
ALARM_THRESHOLD = 0.2945967216491699
NOISE_SEED = 20260905
ACTION_BUDGET = 200
SETTLE_STEPS = 10
MAX_STEPS = 900
ARM_NAMES = ("h10", "h2_burst10")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _memory_events() -> Dict[str, int]:
    result = {}
    for line in (
        Path("/sys/fs/cgroup/memory.events").read_text(encoding="ascii").splitlines()
    ):
        key, value = line.split()
        result[key] = int(value)
    return result


def _power_limit(gpu: int) -> Dict[str, float]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=power.limit,power.max_limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current, maximum = (float(part.strip()) for part in output.split(","))
    if current != maximum:
        subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--power-limit", str(maximum)],
            check=True,
        )
        return _power_limit(gpu)
    return {"power_limit_w": current, "power_max_limit_w": maximum}


def horizon_at(arm: str, physical_offset: int) -> int:
    if arm == "h10":
        return 10
    if arm == "h2_burst10":
        return 2 if physical_offset < 10 else 10
    raise ValueError("unknown arm %r" % arm)


def future_noise(pair_id: int, physical_offset: int) -> Tuple[np.ndarray, List[int]]:
    words = [NOISE_SEED, int(pair_id), int(physical_offset)]
    rng = np.random.default_rng(np.random.SeedSequence(words))
    return rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32), words


def split_pairs(pair_ids: Sequence[int], number_of_shards: int) -> List[List[int]]:
    return [
        [int(value) for value in shard]
        for shard in np.array_split(
            np.asarray(sorted(pair_ids), dtype=np.int16), number_of_shards
        )
    ]


def _policy_hashes(observation: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "image_sha256": _sha256_array(np.asarray(observation[IMAGE_KEY], np.uint8)),
        "wrist_image_sha256": _sha256_array(
            np.asarray(observation[WRIST_IMAGE_KEY], np.uint8)
        ),
        "state_sha256": _sha256_array(np.asarray(observation[STATE_KEY], np.float32)),
    }


def _physical_point(
    environment: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    base = getattr(environment, "env", environment)
    robot = list(base.robots)[0]
    data = base.sim.data
    sim_state = np.asarray(environment.get_sim_state(), np.float64).copy()
    position = np.asarray(data.site_xpos[int(robot.eef_site_id)], np.float64).copy()
    quaternion_wxyz = np.asarray(
        data.get_body_xquat(robot.robot_model.eef_name), np.float64
    ).copy()
    quaternion = quaternion_wxyz[[1, 2, 3, 0]]
    gripper_indices = np.asarray(robot._ref_gripper_joint_pos_indexes, np.int64)
    gripper = np.asarray(data.qpos[gripper_indices], np.float64).copy()
    values = (sim_state, position, quaternion, gripper)
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("non-finite physical state")
    return sim_state, position, quaternion, gripper, bool(environment.check_success())


def _image_observables(environment: Any) -> List[Any]:
    base = getattr(environment, "env", environment)
    images = [
        observable
        for observable in base._observables.values()
        if observable.modality == "image"
    ]
    if {observable.name for observable in images} != {
        "agentview_image",
        "robot0_eye_in_hand_image",
    }:
        raise RuntimeError("unexpected image observables")
    return images


def _set_images_enabled(observables: Sequence[Any], enabled: bool) -> None:
    for observable in observables:
        observable.set_enabled(enabled)


def _refresh_observation(environment: Any) -> Mapping[str, Any]:
    state = np.asarray(environment.get_sim_state(), np.float64).copy()
    return environment.regenerate_obs_from_state(state)


def _load_inputs(
    plan_path: Path,
    capture_records_path: Path,
    feature_path: Path,
    rich_inputs_path: Path,
    input_audit_path: Path,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, Dict[str, Any]],
    Dict[int, float],
    Dict[int, np.ndarray],
]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "himoe.rich_event_capture_plan.v1":
        raise RuntimeError("unexpected rich event plan schema")
    capture_records = {
        int(record["row_id"]): record
        for record in (
            json.loads(line)
            for line in capture_records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    audit_rows = {int(row["row_id"]): row for row in audit["rows"]}
    with np.load(feature_path, allow_pickle=False) as source:
        feature_rows = {
            int(row_id): float(score)
            for row_id, score in zip(source["row_id"], source["routed_shared_cosine"])
        }
    with np.load(rich_inputs_path, allow_pickle=False) as source:
        original_noise = {
            int(row_id): np.asarray(noise, np.float32)
            for row_id, noise in zip(source["row_id"], source["flow_noise"])
        }

    lead_rows = [row for row in plan["rows"] if int(row["relative_query"]) == LEAD]
    by_pair: Dict[int, List[Dict[str, Any]]] = {}
    for row in lead_rows:
        by_pair.setdefault(int(row["pair_id"]), []).append(dict(row))
    eligible_pairs = []
    for pair_id, rows in sorted(by_pair.items()):
        if {row["role"] for row in rows} != {"event", "control"} or len(rows) != 2:
            raise RuntimeError(
                "pair %d is not a complete event/control match" % pair_id
            )
        if all(
            bool(capture_records[int(row["row_id"])]["restore_pass"]) for row in rows
        ):
            eligible_pairs.append(pair_id)
    if len(eligible_pairs) != 23 or 4 in eligible_pairs:
        raise RuntimeError("frozen eligibility changed: %s" % eligible_pairs)
    rows = [row for row in lead_rows if int(row["pair_id"]) in eligible_pairs]
    rows.sort(key=lambda row: (int(row["pair_id"]), 0 if row["role"] == "event" else 1))
    for row in rows:
        row_id = int(row["row_id"])
        if (
            row_id not in audit_rows
            or row_id not in feature_rows
            or row_id not in original_noise
        ):
            raise RuntimeError("missing rich input for row %d" % row_id)
    return rows, audit_rows, feature_rows, original_noise


def _reconstruct_snapshot(
    environment: Any,
    row: Mapping[str, Any],
    audit_row: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Dict[str, Any]]:
    episode_path = ROOT / str(row["episode_path"])
    query = int(row["query_index"])
    with np.load(episode_path, allow_pickle=False) as source:
        target_sim = np.asarray(source["sim_state"][query], np.float64)
        saved_policy_state = np.asarray(source["state"][query], np.float32)
        replay_actions = np.asarray(source["actions"][:query], np.float64)

    images = _image_observables(environment)
    early_success = False
    _set_images_enabled(images, False)
    try:
        for _ in range(SETTLE_STEPS):
            _observation, _reward, _done, _info = environment.step(
                LIBERO_DUMMY_ACTION.tolist()
            )
        for chunk in replay_actions:
            for action in chunk:
                _observation, _reward, _done, _info = environment.step(action.tolist())
                early_success = early_success or bool(environment.check_success())
    finally:
        _set_images_enabled(images, True)
    replay_sim = np.asarray(environment.get_sim_state(), np.float64)
    replay_error = float(np.max(np.abs(replay_sim - target_sim)))

    observation = environment.regenerate_obs_from_state(target_sim)
    pinned_error = float(
        np.max(np.abs(np.asarray(environment.get_sim_state(), np.float64) - target_sim))
    )
    if pinned_error > 1e-9:
        raise RuntimeError("row %d failed exact MuJoCo pin" % int(row["row_id"]))
    policy_observation = build_policy_observation(observation, PROMPT)
    hashes = _policy_hashes(policy_observation)
    for key in ("image_sha256", "wrist_image_sha256"):
        if hashes[key] != audit_row[key]:
            raise RuntimeError("row %d %s changed" % (int(row["row_id"]), key))
    policy_state_error = float(
        np.max(
            np.abs(
                np.asarray(policy_observation[STATE_KEY], np.float32)
                - saved_policy_state
            )
        )
    )
    if policy_state_error > 0.01:
        raise RuntimeError(
            "row %d policy-state restore exceeded 0.01" % int(row["row_id"])
        )
    if bool(environment.check_success()):
        raise RuntimeError("row %d is already successful" % int(row["row_id"]))
    snapshot = save_full_state(environment)
    metadata = {
        "source_episode_sha256": _sha256_file(episode_path),
        "replay_sim_max_abs_error_before_pin": replay_error,
        "pinned_sim_max_abs_error": pinned_error,
        "policy_state_max_abs_error": policy_state_error,
        "historical_success_seen_during_replay": early_success,
        "full_state_sha256": _full_state_sha256(snapshot),
        **hashes,
    }
    return snapshot, policy_observation, metadata


def _run_arm(
    environment: Any,
    snapshot: Mapping[str, Any],
    prompt: str,
    client: PolicyClient,
    row: Mapping[str, Any],
    arm: str,
    original_noise: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    observation = _hard_restore(environment, snapshot)
    restored_snapshot_sha256 = _full_state_sha256(save_full_state(environment))
    expected_snapshot_sha256 = _full_state_sha256(snapshot)
    if restored_snapshot_sha256 != expected_snapshot_sha256:
        raise RuntimeError("full-state restore hash changed before %s" % arm)
    initial_policy = build_policy_observation(observation, prompt)
    initial_hashes = _policy_hashes(initial_policy)

    physical = [_physical_point(environment)]
    if physical[0][-1]:
        raise RuntimeError("arm started in a successful state")
    query_offsets: List[int] = []
    query_horizons: List[int] = []
    query_noises: List[np.ndarray] = []
    query_seed_words: List[List[int]] = []
    query_noise_hashes: List[str] = []
    action_chunks: List[np.ndarray] = []
    action_counts: List[int] = []
    inference_seconds: List[float] = []
    action_steps = 0
    success = False

    while action_steps < ACTION_BUDGET and not success:
        policy_observation = build_policy_observation(observation, prompt)
        offset = action_steps
        if offset == 0:
            noise = np.ascontiguousarray(original_noise, np.float32)
            seed_words = [-1, int(row["row_id"]), 0]
        else:
            noise, seed_words = future_noise(int(row["pair_id"]), offset)
        request = dict(policy_observation)
        request[FLOW_NOISE_KEY] = noise
        started = time.perf_counter()
        response = validate_action_response(client.infer(request))
        inference_seconds.append(time.perf_counter() - started)
        noise_hash = _sha256_array(noise)
        if response.get(FLOW_NOISE_SHA256_KEY) != noise_hash:
            raise RuntimeError("server did not acknowledge exact flow noise")
        actions = np.asarray(response[ACTION_KEY], np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError("invalid action chunk %s" % (actions.shape,))
        take = min(horizon_at(arm, offset), ACTION_BUDGET - action_steps)
        query_offsets.append(offset)
        query_horizons.append(horizon_at(arm, offset))
        query_noises.append(noise)
        query_seed_words.append(seed_words)
        query_noise_hashes.append(noise_hash)
        action_chunks.append(actions)
        executed = 0
        images = _image_observables(environment)
        _set_images_enabled(images, False)
        try:
            for action in actions[:take]:
                observation, _reward, _done, _info = environment.step(
                    np.asarray(action, np.float64).tolist()
                )
                action_steps += 1
                executed += 1
                point = _physical_point(environment)
                physical.append(point)
                success = bool(point[-1])
                if success:
                    break
        finally:
            _set_images_enabled(images, True)
        action_counts.append(executed)
        if not success and action_steps < ACTION_BUDGET:
            observation = _refresh_observation(environment)

    arrays = {
        "sim_state": np.stack([point[0] for point in physical]),
        "eef_position": np.stack([point[1] for point in physical]),
        "eef_quaternion": np.stack([point[2] for point in physical]),
        "gripper_qpos": np.stack([point[3] for point in physical]),
        "success_by_step": np.asarray([point[4] for point in physical], bool),
        "query_physical_offset": np.asarray(query_offsets, np.int16),
        "query_horizon": np.asarray(query_horizons, np.int8),
        "query_noise": np.stack(query_noises).astype(np.float32),
        "query_seed_words": np.asarray(query_seed_words, np.int64),
        "query_noise_sha256": np.asarray(query_noise_hashes, dtype="<U64"),
        "action_chunks": np.stack(action_chunks).astype(np.float32),
        "executed_action_count": np.asarray(action_counts, np.int8),
        "inference_seconds": np.asarray(inference_seconds, np.float64),
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "pair_id": int(row["pair_id"]),
        "row_id": int(row["row_id"]),
        "role": str(row["role"]),
        "arm": arm,
        "success": bool(success),
        "actions_executed": int(action_steps),
        "queries": len(query_offsets),
        "success_action": int(np.argmax(arrays["success_by_step"]))
        if success
        else None,
        "inference_seconds": float(np.sum(inference_seconds)),
        "initial_full_state_sha256": expected_snapshot_sha256,
        "restored_full_state_sha256": restored_snapshot_sha256,
        "initial_policy_hashes": initial_hashes,
        "first_noise_sha256": query_noise_hashes[0],
        "first_action_sha256": _sha256_array(action_chunks[0]),
    }
    return arrays, summary


def _arm_paths(out: Path, row: Mapping[str, Any], arm: str) -> Tuple[Path, Path]:
    stem = "pair_%03d_%s_%s" % (int(row["pair_id"]), row["role"], arm)
    return out / "arms" / (stem + ".npz"), out / "arms" / (stem + ".json")


def _load_complete_arm(
    npz_path: Path, json_path: Path
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("schema") != SCHEMA or summary.get("status") != "complete":
        raise RuntimeError("invalid completed arm %s" % json_path)
    if summary.get("npz_sha256") != _sha256_file(npz_path):
        raise RuntimeError("completed arm checksum changed: %s" % npz_path)
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    return arrays, summary


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    if _sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise RuntimeError("recovery protocol hash does not match the frozen value")
    rows, audit_rows, feature_rows, original_noise = _load_inputs(
        args.plan,
        args.capture_records,
        args.features,
        args.rich_inputs,
        args.input_audit,
    )
    pair_ids = sorted({int(row["pair_id"]) for row in rows})
    shards = split_pairs(pair_ids, args.num_shards)
    selected_pairs = shards[args.shard_index]
    if args.limit_pairs is not None:
        selected_pairs = selected_pairs[: args.limit_pairs]
    selected_rows = [row for row in rows if int(row["pair_id"]) in selected_pairs]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arms").mkdir(exist_ok=True)
    manifest_path = args.out / "manifest.json"
    immutable = {
        "schema": SCHEMA,
        "protocol_sha256": args.expected_protocol_sha256,
        "plan_sha256": _sha256_file(args.plan),
        "features_sha256": _sha256_file(args.features),
        "rich_inputs_sha256": _sha256_file(args.rich_inputs),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pair_ids": selected_pairs,
        "port": args.port,
        "physical_gpu": args.physical_gpu,
        "action_budget": ACTION_BUDGET,
        "noise_seed": NOISE_SEED,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in immutable.items():
            if existing.get(key) != value:
                raise RuntimeError("resume manifest mismatch at %s" % key)
    elif any(args.out.iterdir()):
        entries = [path.name for path in args.out.iterdir() if path.name != "arms"]
        if entries or any((args.out / "arms").iterdir()):
            raise RuntimeError("output directory is non-empty without a manifest")

    pre_memory = _memory_events()
    pre_power = _power_limit(args.physical_gpu)
    started = time.time()
    base_manifest = {
        **immutable,
        "status": "collecting",
        "started_unix": started,
        "preflight_memory_events": pre_memory,
        "preflight_power": pre_power,
    }
    _atomic_write_json(manifest_path, base_manifest)

    config = EpisodeConfig(
        libero_root=str(args.libero_root),
        output_root=str(args.out),
        task_suite="libero_10",
        task_id=8,
        init_state_id=0,
        seed=7,
        settle_steps=SETTLE_STEPS,
        max_steps=MAX_STEPS,
        replan_steps=10,
        render_size=224,
        inference_timeout=300.0,
    )
    state_records = []
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=300.0,
    ) as client:
        validate_policy_suite(client.metadata, "libero_10")
        if client.metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint identity mismatch")
        if client.metadata.get("libero_wrist_layout") != "paper-right":
            raise RuntimeError("wrist layout mismatch")
        _atomic_write_json(args.out / "server_metadata.json", dict(client.metadata))

        for row_index, row in enumerate(selected_rows):
            arm_data = {}
            complete = True
            for arm in ARM_NAMES:
                npz_path, json_path = _arm_paths(args.out, row, arm)
                if npz_path.exists() and json_path.exists():
                    arm_data[arm] = _load_complete_arm(npz_path, json_path)
                else:
                    complete = False
            reconstruction = None
            if not complete:
                row_config = dataclasses.replace(
                    config, init_state_id=int(row["init_state_id"])
                )
                environment, _observation, task, prompt = _load_task(row_config)
                if str(task.name) != TASK_NAME or prompt != PROMPT:
                    environment.close()
                    raise RuntimeError("task identity mismatch")
                try:
                    snapshot, _policy_observation, reconstruction = (
                        _reconstruct_snapshot(
                            environment,
                            row,
                            audit_rows[int(row["row_id"])],
                        )
                    )
                    order = list(ARM_NAMES)
                    if (int(row["pair_id"]) + (row["role"] == "control")) % 2:
                        order.reverse()
                    for arm in order:
                        npz_path, json_path = _arm_paths(args.out, row, arm)
                        if npz_path.exists() and json_path.exists():
                            arm_data[arm] = _load_complete_arm(npz_path, json_path)
                            continue
                        arrays, summary = _run_arm(
                            environment,
                            snapshot,
                            prompt,
                            client,
                            row,
                            arm,
                            original_noise[int(row["row_id"])],
                        )
                        _atomic_write_npz(npz_path, arrays)
                        summary["npz"] = npz_path.name
                        summary["npz_sha256"] = _sha256_file(npz_path)
                        _atomic_write_json(json_path, summary)
                        arm_data[arm] = arrays, summary
                        print(
                            "shard%d pair%02d %-7s %-11s %s steps=%d q=%d"
                            % (
                                args.shard_index,
                                int(row["pair_id"]),
                                row["role"],
                                arm,
                                "SUCCESS" if summary["success"] else "failure",
                                summary["actions_executed"],
                                summary["queries"],
                            ),
                            flush=True,
                        )
                finally:
                    environment.close()

            h10_arrays, h10_summary = arm_data["h10"]
            h2_arrays, h2_summary = arm_data["h2_burst10"]
            hard_checks = {
                "initial_full_state_equal": (
                    h10_summary["initial_full_state_sha256"]
                    == h2_summary["initial_full_state_sha256"]
                ),
                "initial_policy_equal": (
                    h10_summary["initial_policy_hashes"]
                    == h2_summary["initial_policy_hashes"]
                ),
                "first_noise_equal": (
                    h10_summary["first_noise_sha256"]
                    == h2_summary["first_noise_sha256"]
                ),
                "first_action_bitwise_equal": bool(
                    np.array_equal(
                        h10_arrays["action_chunks"][0], h2_arrays["action_chunks"][0]
                    )
                ),
            }
            if not all(hard_checks.values()):
                raise RuntimeError(
                    "pair %d %s failed paired validity: %s"
                    % (int(row["pair_id"]), row["role"], hard_checks)
                )
            score = feature_rows[int(row["row_id"])]
            state_path = args.out / (
                "pair_%03d_%s.json" % (int(row["pair_id"]), row["role"])
            )
            if reconstruction is None and state_path.exists():
                reconstruction = _load_json(state_path).get("reconstruction")
            state_record = {
                "pair_id": int(row["pair_id"]),
                "row_id": int(row["row_id"]),
                "role": row["role"],
                "score": score,
                "alarm": bool(score > ALARM_THRESHOLD),
                "hard_checks": hard_checks,
                "reconstruction": reconstruction,
                "h10": h10_summary,
                "h2_burst10": h2_summary,
            }
            _atomic_write_json(state_path, state_record)
            state_records.append(state_record)
            print(
                "shard%d completed state %d/%d (pair%02d %s alarm=%s)"
                % (
                    args.shard_index,
                    row_index + 1,
                    len(selected_rows),
                    int(row["pair_id"]),
                    row["role"],
                    state_record["alarm"],
                ),
                flush=True,
            )

    post_memory = _memory_events()
    post_power = _power_limit(args.physical_gpu)
    completed = {
        **base_manifest,
        "status": "complete",
        "server_checkpoint_sha256": CHECKPOINT_SHA256,
        "states": len(state_records),
        "arms": 2 * len(state_records),
        "alarm_threshold": ALARM_THRESHOLD,
        "postflight_memory_events": post_memory,
        "postflight_power": post_power,
        "oom_kill_unchanged": post_memory.get("oom_kill") == pre_memory.get("oom_kill"),
        "wall_seconds": time.time() - started,
    }
    _atomic_write_json(manifest_path, completed)
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path("/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/rich_event_plan.json",
    )
    parser.add_argument(
        "--capture-records",
        type=Path,
        default=ROOT
        / "analysis_moe_execution_signals/rich_event_functional_32d/records.jsonl",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/rich_event_row_features.npz",
    )
    parser.add_argument(
        "--rich-inputs",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/rich_event_inputs.npz",
    )
    parser.add_argument(
        "--input-audit",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/rich_event_inputs_audit.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/RECOVERY_HORIZON_PROTOCOL.md",
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must be inside 0..num-shards-1")
    if args.limit_pairs is not None and args.limit_pairs <= 0:
        parser.error("limit-pairs must be positive")
    return args


def main() -> int:
    result = collect(build_parser())
    return 0 if result["status"] == "complete" and result["oom_kill_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
