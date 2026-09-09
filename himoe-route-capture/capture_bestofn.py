#!/usr/bin/env python3
"""Capture new candidate proposals and terminal CRN continuations for Best-of-N."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_forks_v2 import (
    EVENT_LABELS,
    ArtifactError,
    atomic_json,
    atomic_npz,
    commit_artifact,
    dense_event_tape,
    load_npz,
    read_full_state_bundle,
    verify_artifact,
    write_full_state_bundle,
)
from bestofn_artifacts import validate_candidate, validate_continuation
from bestofn_protocol import (
    CANDIDATE_SCHEMA,
    CONTINUATION_SCHEMA,
    BestOfNError,
    candidate_dir,
    candidate_query_id,
    continuation_query_id,
    continuation_shard_dir,
    flow_noise,
    load_config,
    load_json,
    seed_words,
    sha256_file,
    validate_plan,
)
from branch_snapshot import save_full_state
from capture_behavior_forks import (
    _contact_tape,
    _enhanced_sim_layout,
    _hard_restore,
    _physical_point,
    _trajectory_pair_spread,
)
from capture_behavior_study import (
    GatedQueryLedger,
    _episode_config,
    _reach_snapshot,
    _require_gated_metadata,
    _server_identity,
    _settle,
    _source_actions,
)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import _load_task, validate_policy_suite
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)
from himoe_candidate_capture_protocol import VLM_FEATURE_KEY


def _stage(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ArtifactError(f"uncommitted artifact directory already exists: {target}")
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))


def _publish(stage: Path, target: Path) -> None:
    if not (stage / "artifact.json").is_file():
        raise ArtifactError("staged artifact has no descriptor")
    os.replace(stage, target)


def _load_protocol(config_path: Path, plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    plan = load_json(plan_path)
    validate_plan(plan, config)
    if str(plan.get("config_sha256")) != sha256_file(config_path):
        raise BestOfNError("plan was built from another config")
    return config, plan


def _state(plan: Mapping[str, Any], state_id: str) -> dict[str, Any]:
    rows = [row for row in plan["states"] if row["state_id"] == state_id]
    if len(rows) != 1:
        raise BestOfNError(f"plan contains {len(rows)} rows for {state_id}")
    return dict(rows[0])


def _existing(path: Path, schema: str, validator) -> bool:
    descriptor = path / "artifact.json"
    if not descriptor.is_file():
        return False
    metadata = verify_artifact(descriptor, schema)
    arrays = load_npz(path / metadata["files"]["data"]["file"])
    validator(arrays, metadata)
    return True


def capture_candidates(args: argparse.Namespace) -> int:
    config, plan = _load_protocol(args.config, args.plan)
    state = _state(plan, args.state_id)
    target = candidate_dir(args.out.resolve(), args.state_id)
    if _existing(target, CANDIDATE_SCHEMA, validate_candidate):
        print(f"candidate artifact already complete: {target}")
        return 0
    k = int(state["candidate_count"])
    h = int(config["proposal"]["action_chunk_horizon"])
    source_actions = _source_actions(Path(state["source_episode_file"]))
    episode_config = _episode_config(
        state,
        str(config["benchmark"]),
        args.libero_root,
        args.out,
        1200,
        int(state["settle_steps"]),
    )
    environment, _initial, task, prompt = _load_task(episode_config)
    try:
        observation = _reach_snapshot(
            environment, source_actions, int(state["fork_step"]), episode_config.settle_steps
        )
        snapshot = save_full_state(environment)
        observation = _hard_restore(environment, snapshot)
        policy_observation = build_policy_observation(observation, prompt)
        point = _physical_point(environment, observation)
        state_dim, gripper_dim = len(point[0]), len(point[3])
        layout = _enhanced_sim_layout(environment, state_dim)
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, str(config["benchmark"]))
            advertised_run_id = _require_gated_metadata(
                client.metadata, args.route_store_id or None
            )
            ledger = GatedQueryLedger(
                client,
                expected_capture_run_id=args.route_store_id or None,
                expected_recorder_rows=args.expected_recorder_rows,
            )
            noises = np.empty((k, *FLOW_NOISE_SHAPE), dtype=np.float32)
            actions = np.empty((k, h, 7), dtype=np.float32)
            query_ids = np.empty(k, dtype=np.int32)
            rows = np.empty(k, dtype=np.int64)
            store_ids = np.empty(k, dtype="<U64")
            feature_dim = int(client.metadata.get("candidate_vlm_feature_dim", 0))
            if client.metadata.get("candidate_vlm_feature_key") != VLM_FEATURE_KEY:
                raise ArtifactError("server does not advertise frozen VLM feature capture")
            if feature_dim <= 0:
                raise ArtifactError("server advertises an invalid frozen VLM feature width")
            vlm_features = np.empty((k, feature_dim), dtype=np.float16)
            for candidate in range(k):
                words = seed_words(
                    int(config["master_seed"]),
                    "bestofn/candidate",
                    int(state["task_id"]),
                    int(state["episode"]),
                    int(state["fork_step"]),
                    (candidate,),
                )
                noise = flow_noise(words)
                query_id = candidate_query_id(int(state["snapshot_index"]), candidate)
                response, row, acknowledgement = ledger.infer(
                    policy_observation,
                    noise,
                    query_id=query_id,
                    capture=True,
                    snapshot_index=int(state["snapshot_index"]),
                    candidate_id=candidate,
                    flush_after=candidate == k - 1,
                    coordinates={"state_id": args.state_id, "candidate": candidate},
                    seed_entropy=words,
                )
                noises[candidate] = noise
                actions[candidate] = np.asarray(response[ACTION_KEY], dtype=np.float32)[:h]
                feature = np.asarray(response.get(VLM_FEATURE_KEY), dtype=np.float16)
                if feature.shape != (feature_dim,) or not np.all(np.isfinite(feature)):
                    raise ArtifactError("captured response has an invalid frozen VLM feature")
                vlm_features[candidate] = feature
                query_ids[candidate], rows[candidate] = query_id, row
                store_ids[candidate] = str(acknowledgement["capture_run_id"])
            if ledger.capture_run_id != advertised_run_id:
                raise ArtifactError("server metadata and capture acknowledgements disagree")
            server_metadata = dict(client.metadata)

        order_words = seed_words(
            int(config["master_seed"]),
            "bestofn/candidate-execution-order",
            int(state["task_id"]),
            int(state["episode"]),
            int(state["fork_step"]),
        )
        execution_order = np.asarray(
            np.random.default_rng(np.random.SeedSequence(order_words)).permutation(k),
            dtype=np.int32,
        )
        sim = np.empty((k, h + 1, state_dim), dtype=np.float64)
        eef = np.empty((k, h + 1, 3), dtype=np.float64)
        quat = np.empty((k, h + 1, 4), dtype=np.float64)
        gripper = np.empty((k, h + 1, gripper_dim), dtype=np.float64)
        success = np.empty((k, h + 1), dtype=np.bool_)
        executed = np.zeros((k, h), dtype=np.bool_)
        contact_sets: list[list[set[tuple[str, str]]]] = [[] for _ in range(k)]
        post_states: list[Mapping[str, Any] | None] = [None] * k
        for candidate in execution_order:
            observation = _hard_restore(environment, snapshot)
            points = [_physical_point(environment, observation, gripper_dim)]
            succeeded = bool(environment.check_success())
            for action_index, action in enumerate(actions[candidate]):
                if succeeded:
                    break
                observation, _reward, _done, _info = environment.step(
                    np.asarray(action, dtype=np.float64).tolist()
                )
                executed[candidate, action_index] = True
                points.append(_physical_point(environment, observation, gripper_dim))
                succeeded = bool(environment.check_success())
            post_states[candidate] = save_full_state(environment)
            while len(points) < h + 1:
                points.append(points[-1])
            sim[candidate] = np.stack([value[0] for value in points])
            eef[candidate] = np.stack([value[1] for value in points])
            quat[candidate] = np.stack([value[2] for value in points])
            gripper[candidate] = np.stack([value[3] for value in points])
            success[candidate] = np.asarray([value[4] for value in points])
            contact_sets[candidate] = [value[5] for value in points]

        contact, contact_names, _pairs = _contact_tape(contact_sets)
        events = dense_event_tape(
            contact,
            gripper,
            sim,
            success,
            int(layout["nq"]),
            layout["object_qpos_indices"],
        )
        reference = int(execution_order[0])
        observation = _hard_restore(environment, snapshot)
        rerun = [np.asarray(environment.get_sim_state(), dtype=np.float64).copy()]
        for action_index, action in enumerate(actions[reference]):
            if not executed[reference, action_index]:
                break
            observation, _reward, _done, _info = environment.step(
                np.asarray(action, dtype=np.float64).tolist()
            )
            rerun.append(np.asarray(environment.get_sim_state(), dtype=np.float64).copy())
        original = sim[reference, : len(rerun)]
        rerun_tape = np.stack(rerun)
        drift = float(np.max(np.abs(rerun_tape - original)))
        spread_mean, spread_max = _trajectory_pair_spread(sim)
        threshold = max(float(args.fidelity_abs_tol), float(args.fidelity_ratio) * spread_mean)
        fidelity_passed = bool(drift <= threshold)
        vlm_deviation = float(
            np.max(
                np.abs(
                    vlm_features.astype(np.float32)
                    - vlm_features[:1].astype(np.float32)
                )
            )
        )
        vlm_tolerance = 0.01
    finally:
        environment.close()
    if not fidelity_passed:
        raise ArtifactError(f"candidate fidelity failed: {drift} > {threshold}")
    if vlm_deviation > vlm_tolerance:
        raise ArtifactError(
            f"candidate-invariant frozen VLM feature changed: {vlm_deviation}"
        )
    if any(value is None for value in post_states):
        raise ArtifactError("candidate post-state bundle is incomplete")
    arrays = {
        "candidate_ids": np.arange(k, dtype=np.int32),
        "actions": actions,
        "candidate_flow_noise": noises,
        "query_ids": query_ids,
        "server_trace_rows": rows,
        "store_ids": store_ids,
        "execution_order": execution_order,
        "candidate_action_executed": executed,
        "sim_states": sim,
        "eef_positions": eef,
        "eef_quaternions": quat,
        "gripper_qpos": gripper,
        "chunk_success": success,
        "chunk_terminal_success": success.any(axis=1),
        "contact_active": contact,
        "contact_pair_names": contact_names,
        "event_flags": events,
        "event_labels": np.asarray(EVENT_LABELS),
        "observation_image": np.asarray(policy_observation[IMAGE_KEY], dtype=np.uint8),
        "observation_wrist_image": np.asarray(
            policy_observation[WRIST_IMAGE_KEY], dtype=np.uint8
        ),
        "observation_state": np.asarray(policy_observation[STATE_KEY], dtype=np.float32),
        "frozen_vlm_feature": vlm_features,
        "fidelity_reference_candidate": np.asarray(reference, dtype=np.int32),
        "fidelity_rerun_sim_states": rerun_tape,
        "fidelity_drift": np.asarray(drift),
        "fidelity_candidate_spread_mean": np.asarray(spread_mean),
        "fidelity_candidate_spread_max": np.asarray(spread_max),
        "fidelity_threshold": np.asarray(threshold),
        "fidelity_passed": np.asarray(fidelity_passed),
    }
    metadata = {
        "state_id": args.state_id,
        "snapshot_index": int(state["snapshot_index"]),
        "task_id": int(state["task_id"]),
        "episode": int(state["episode"]),
        "fork_step": int(state["fork_step"]),
        "phase": state["phase"],
        "split": state["split"],
        "candidate_count": k,
        "action_chunk_horizon": h,
        "init_state_id": int(state["init_state_id"]),
        "environment_seed": int(state["environment_seed"]),
        "settle_steps": int(state["settle_steps"]),
        "benchmark": config["benchmark"],
        "task_name": str(task.name),
        "prompt": prompt,
        "config_file": str(args.config),
        "config_sha256": sha256_file(args.config),
        "plan_file": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "source_episode_file": state["source_episode_file"],
        "source_episode_sha256": state["source_episode_sha256"],
        "route_store_id": ledger.capture_run_id,
        "server_row_start": int(rows[0]),
        "server_row_stop": int(rows[-1]) + 1,
        "server_durable_through_row": int(ledger.records[-1]["durable_through_row"]),
        "server_identity": _server_identity(server_metadata),
        "seed_domain": "bestofn/candidate",
        "master_seed": int(config["master_seed"]),
        "terminal_semantics": "stop candidate execution on first success",
        "frozen_vlm_feature_dim": feature_dim,
        "frozen_vlm_feature_pooling": server_metadata[
            "candidate_vlm_feature_pooling"
        ],
        "frozen_vlm_feature_max_deviation": vlm_deviation,
        "frozen_vlm_feature_tolerance": vlm_tolerance,
    }
    validate_candidate(arrays, metadata)
    stage = _stage(target)
    atomic_npz(stage / "candidates.npz", arrays)
    atomic_json(stage / "layout.json", layout)
    atomic_json(stage / "queries.json", {"queries": ledger.records})
    atomic_json(stage / "server_metadata.json", server_metadata)
    state_files = write_full_state_bundle(stage / "post_states", post_states)  # type: ignore[arg-type]
    commit_artifact(
        stage,
        CANDIDATE_SCHEMA,
        metadata,
        {
            "data": stage / "candidates.npz",
            "layout": stage / "layout.json",
            "queries": stage / "queries.json",
            "server_metadata": stage / "server_metadata.json",
            **state_files,
        },
    )
    _publish(stage, target)
    print(f"captured {k} new Best-of-N candidates: {target}")
    return 0


def capture_continuations(args: argparse.Namespace) -> int:
    config, plan = _load_protocol(args.config, args.plan)
    state = _state(plan, args.state_id)
    start, stop = int(args.repeat_start), int(args.repeat_stop)
    if start < 0 or stop <= start or stop > int(config["continuation"]["topup_repeats"]):
        raise BestOfNError("continuation repeat interval is outside the frozen panel")
    target = continuation_shard_dir(args.out.resolve(), args.state_id, start, stop)
    if _existing(target, CONTINUATION_SCHEMA, validate_continuation):
        print(f"terminal continuation shard already complete: {target}")
        return 0
    candidate_artifact = candidate_dir(args.out.resolve(), args.state_id) / "artifact.json"
    candidate_meta = verify_artifact(candidate_artifact, CANDIDATE_SCHEMA)
    if str(candidate_meta.get("state_id")) != args.state_id:
        raise ArtifactError("candidate artifact belongs to another planned state")
    if candidate_meta.get("config_sha256") != sha256_file(args.config):
        raise ArtifactError("candidate artifact was captured under another config")
    if candidate_meta.get("plan_sha256") != sha256_file(args.plan):
        raise ArtifactError("candidate artifact was captured under another plan")
    candidate_arrays = load_npz(
        candidate_artifact.parent / candidate_meta["files"]["data"]["file"]
    )
    validate_candidate(candidate_arrays, candidate_meta)
    post_states = read_full_state_bundle(
        candidate_artifact.parent / candidate_meta["files"]["full_states_meta"]["file"],
        candidate_artifact.parent / candidate_meta["files"]["full_states_arrays"]["file"],
    )
    k = int(candidate_meta["candidate_count"])
    h = int(candidate_meta["action_chunk_horizon"])
    repeats = stop - start
    terminal_budget = int(config["continuation"]["terminal_environment_steps"])
    source_steps = int(candidate_meta["fork_step"]) * h
    maximum_future_queries = max(0, (terminal_budget - source_steps + h - 1) // h)
    noise = np.empty((repeats, maximum_future_queries, *FLOW_NOISE_SHAPE), dtype=np.float32)
    for local_repeat, repeat in enumerate(range(start, stop)):
        for future_query in range(maximum_future_queries):
            noise[local_repeat, future_query] = flow_noise(
                seed_words(
                    int(config["master_seed"]),
                    "bestofn/terminal-continuation",
                    int(state["task_id"]),
                    int(state["episode"]),
                    int(state["fork_step"]),
                    (repeat, future_query),
                )
            )
    episode_config = _episode_config(
        state,
        str(config["benchmark"]),
        args.libero_root,
        args.out,
        1200,
        int(state["settle_steps"]),
    )
    environment, _initial, _task, prompt = _load_task(episode_config)
    try:
        _settle(environment, episode_config.settle_steps)
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, str(config["benchmark"]))
            _require_gated_metadata(client.metadata, str(candidate_meta["route_store_id"]))
            current_identity = _server_identity(client.metadata)
            for key in (
                "checkpoint_sha256",
                "normalization_stats_sha256",
                "normalization_action_std",
                "libero_wrist_layout",
                "himoe_upstream_commit",
                "himoe_patch_sha256",
                "himoe_working_tree_diff_sha256",
                "himoe_patches_verified_applied",
            ):
                if current_identity.get(key) != candidate_meta["server_identity"].get(key):
                    raise ArtifactError(f"continuation server differs from candidate server: {key}")
            ledger = GatedQueryLedger(
                client,
                expected_capture_run_id=str(candidate_meta["route_store_id"]),
                expected_recorder_rows=args.expected_recorder_rows,
            )
            terminal_success = np.zeros((k, repeats), dtype=np.bool_)
            censored = np.zeros((k, repeats), dtype=np.bool_)
            continuation_steps = np.zeros((k, repeats), dtype=np.int32)
            total_steps = np.zeros((k, repeats), dtype=np.int32)
            final_states = np.empty(
                (k, repeats, candidate_arrays["sim_states"].shape[-1]), dtype=np.float64
            )
            query_ids = np.full((k, repeats, maximum_future_queries), -1, dtype=np.int32)
            rows = np.full((k, repeats, maximum_future_queries), -1, dtype=np.int64)
            row_counts = np.full((k, repeats, maximum_future_queries), -1, dtype=np.int64)
            durable_rows = np.full((k, repeats, maximum_future_queries), -1, dtype=np.int64)
            query_executed = np.zeros((k, repeats, maximum_future_queries), dtype=np.bool_)
            continuation_actions = np.zeros(
                (k, repeats, maximum_future_queries, h, 7), dtype=np.float32
            )
            action_executed = np.zeros(
                (k, repeats, maximum_future_queries, h), dtype=np.bool_
            )
            candidate_steps = np.asarray(
                candidate_arrays["candidate_action_executed"], dtype=np.bool_
            ).sum(axis=1)
            candidate_success = np.asarray(
                candidate_arrays["chunk_terminal_success"], dtype=np.bool_
            )
            for candidate in range(k):
                for local_repeat, repeat in enumerate(range(start, stop)):
                    observation = _hard_restore(environment, post_states[candidate])
                    succeeded = bool(candidate_success[candidate])
                    used = source_steps + int(candidate_steps[candidate])
                    for future_query in range(maximum_future_queries):
                        if succeeded or used >= terminal_budget:
                            break
                        words = seed_words(
                            int(config["master_seed"]),
                            "bestofn/terminal-continuation",
                            int(state["task_id"]),
                            int(state["episode"]),
                            int(state["fork_step"]),
                            (repeat, future_query),
                        )
                        query_id = continuation_query_id(
                            int(state["snapshot_index"]), candidate, repeat, future_query
                        )
                        response, row, acknowledgement = ledger.infer(
                            build_policy_observation(observation, prompt),
                            noise[local_repeat, future_query],
                            query_id=query_id,
                            capture=False,
                            coordinates={
                                "state_id": args.state_id,
                                "candidate": candidate,
                                "repeat": repeat,
                                "future_query": future_query,
                            },
                            seed_entropy=words,
                        )
                        query_ids[candidate, local_repeat, future_query] = query_id
                        rows[candidate, local_repeat, future_query] = row
                        row_counts[candidate, local_repeat, future_query] = int(
                            acknowledgement["row_count"]
                        )
                        durable_rows[candidate, local_repeat, future_query] = int(
                            acknowledgement["durable_through_row"]
                        )
                        query_executed[candidate, local_repeat, future_query] = True
                        actions = np.asarray(response[ACTION_KEY], dtype=np.float32)[:h]
                        continuation_actions[candidate, local_repeat, future_query] = actions
                        for action_index, action in enumerate(actions):
                            if used >= terminal_budget or succeeded:
                                break
                            observation, _reward, _done, _info = environment.step(
                                np.asarray(action, dtype=np.float64).tolist()
                            )
                            action_executed[
                                candidate, local_repeat, future_query, action_index
                            ] = True
                            continuation_steps[candidate, local_repeat] += 1
                            used += 1
                            succeeded = bool(environment.check_success())
                    total_steps[candidate, local_repeat] = used
                    terminal_success[candidate, local_repeat] = succeeded
                    censored[candidate, local_repeat] = not succeeded and used >= terminal_budget
                    final_states[candidate, local_repeat] = np.asarray(
                        environment.get_sim_state(), dtype=np.float64
                    )
    finally:
        environment.close()
    arrays = {
        "terminal_success": terminal_success,
        "censored_at_budget": censored,
        "continuation_action_steps": continuation_steps,
        "total_environment_steps": total_steps,
        "terminal_final_sim_states": final_states,
        "continuation_flow_noise": noise,
        "query_ids": query_ids,
        "server_trace_rows": rows,
        "server_row_counts": row_counts,
        "server_durable_through_rows": durable_rows,
        "query_executed": query_executed,
        "continuation_actions": continuation_actions,
        "continuation_action_executed": action_executed,
    }
    metadata = {
        "state_id": args.state_id,
        "snapshot_index": int(state["snapshot_index"]),
        "task_id": int(state["task_id"]),
        "episode": int(state["episode"]),
        "fork_step": int(state["fork_step"]),
        "candidate_count": k,
        "action_chunk_horizon": h,
        "repeat_start": start,
        "repeat_stop": stop,
        "maximum_future_queries": maximum_future_queries,
        "terminal_environment_steps": terminal_budget,
        "continuation_policy": "frozen_base_himoe_vla",
        "common_random_numbers": True,
        "seed_excludes_candidate": True,
        "seed_domain": "bestofn/terminal-continuation",
        "master_seed": int(config["master_seed"]),
        "candidate_artifact": str(candidate_artifact),
        "candidate_artifact_sha256": sha256_file(candidate_artifact),
        "route_capture": False,
        "capture_run_id": str(candidate_meta["route_store_id"]),
        "recorder_boundary": (
            [int(args.expected_recorder_rows), int(args.expected_recorder_rows) - 1]
            if ledger.uncaptured_boundary is None
            else list(ledger.uncaptured_boundary)
        ),
        "recorder_row_count": int(args.expected_recorder_rows),
        "recorder_durable_through_row": int(args.expected_recorder_rows) - 1,
    }
    validate_continuation(arrays, metadata)
    stage = _stage(target)
    atomic_npz(stage / "terminal_continuation.npz", arrays)
    atomic_json(stage / "queries.json", {"queries": ledger.records})
    commit_artifact(
        stage,
        CONTINUATION_SCHEMA,
        metadata,
        {"data": stage / "terminal_continuation.npz", "queries": stage / "queries.json"},
    )
    _publish(stage, target)
    print(f"captured terminal continuation repeats [{start},{stop}): {target}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidate")
    candidate.add_argument("--state-id", required=True)
    candidate.add_argument("--host", default="127.0.0.1")
    candidate.add_argument("--port", type=int, required=True)
    candidate.add_argument("--libero-root", required=True)
    candidate.add_argument("--route-store-id", default="")
    candidate.add_argument("--expected-recorder-rows", type=int, required=True)
    candidate.add_argument("--inference-timeout", type=float, default=600.0)
    candidate.add_argument("--fidelity-abs-tol", type=float, default=1e-7)
    candidate.add_argument("--fidelity-ratio", type=float, default=0.01)
    continuation = sub.add_parser("continuation-shard")
    continuation.add_argument("--state-id", required=True)
    continuation.add_argument("--repeat-start", type=int, required=True)
    continuation.add_argument("--repeat-stop", type=int, required=True)
    continuation.add_argument("--host", default="127.0.0.1")
    continuation.add_argument("--port", type=int, required=True)
    continuation.add_argument("--libero-root", required=True)
    continuation.add_argument("--expected-recorder-rows", type=int, required=True)
    continuation.add_argument("--inference-timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.command == "candidate":
        return capture_candidates(args)
    if args.command == "continuation-shard":
        return capture_continuations(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
