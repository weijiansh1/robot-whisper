"""Replay source events and capture staged v2 behavior-fork artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_forks_v2 import (
    CANDIDATE_SCHEMA,
    LABEL_COHORTS,
    EVENT_LABELS,
    EVENT_SCHEMA,
    PLAN_SCHEMA,
    ROOT_SEED,
    SEED_DOMAINS,
    SHARD_REPEATS,
    SHARD_SCHEMA,
    ArtifactError,
    atomic_json,
    atomic_npz,
    candidate_query_id,
    candidate_dir,
    cohort_candidate_pool,
    cohort_seed_pool,
    commit_artifact,
    continuation_query_id,
    dense_event_tape,
    flow_noise,
    load_npz,
    pool_domain,
    read_full_state_bundle,
    rng_from_words,
    seed_words,
    shard_dir,
    sha256_file,
    validate_candidate_arrays,
    validate_shard_arrays,
    verify_artifact,
    write_full_state_bundle,
)
from branch_snapshot import save_full_state
from capture_behavior_forks import (
    _contact_tape,
    _enhanced_sim_layout,
    _hard_restore,
    _physical_point,
    _trajectory_pair_spread,
)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import ACTION_KEY, FLOW_NOISE_SHAPE
from himoe_libero_bridge.protocol import (
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    validate_action_response,
)
from himoe_candidate_capture_protocol import (
    ACK_KEY,
    CANDIDATE_ID_KEY,
    CAPTURE_KEY,
    FLUSH_AFTER_KEY,
    QUERY_ID_KEY,
    SCHEMA as GATED_CAPTURE_SCHEMA,
    SNAPSHOT_INDEX_KEY,
)


def _parse_ints(raw: str | None, available: Sequence[int] | None = None) -> list[int]:
    if raw is None:
        if available is None:
            raise ValueError("an explicit integer list is required")
        return list(available)
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("integer list must be non-empty, unique, and non-negative")
    return values


def _source_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
        sim_states = np.asarray(archive["sim_state"], dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise ArtifactError(f"source actions have invalid shape in {path}: {actions.shape}")
    if sim_states.ndim != 2 or sim_states.shape[0] != actions.shape[0]:
        raise ArtifactError(f"source sim_state does not align with actions in {path}")
    return actions, sim_states


def _source_actions(path: Path) -> np.ndarray:
    return _source_episode(path)[0]


def _stage(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ArtifactError(f"uncommitted final artifact directory exists: {target}")
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))


def _publish(stage: Path, target: Path) -> None:
    if not (stage / "artifact.json").is_file():
        raise ArtifactError(f"staging directory has no committed descriptor: {stage}")
    os.replace(stage, target)


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or len(plan.get("states", [])) != 18:
        raise ArtifactError(f"invalid behavior-study plan: {path}")
    return plan


class GatedQueryLedger:
    """Client-side verifier for the request-gated recorder acknowledgement."""

    def __init__(
        self,
        client: PolicyClient,
        *,
        route_off_smoke: bool = False,
        expected_capture_run_id: str | None = None,
        expected_recorder_rows: int | None = None,
    ) -> None:
        self.client = client
        self.route_off_smoke = bool(route_off_smoke)
        self.expected_capture_run_id = expected_capture_run_id or None
        self.expected_recorder_rows = (
            None if expected_recorder_rows is None else int(expected_recorder_rows)
        )
        if self.expected_recorder_rows is not None and self.expected_recorder_rows < 0:
            raise ValueError("expected recorder rows must be non-negative")
        self.capture_run_id: str | None = None
        self.last_capture_row: int | None = None
        self.uncaptured_boundary: tuple[int, int] | None = None
        self.records: list[dict[str, Any]] = []

    def infer(
        self,
        observation: Mapping[str, Any],
        noise: np.ndarray,
        *,
        query_id: int,
        capture: bool,
        snapshot_index: int = -1,
        candidate_id: int = -1,
        flush_after: bool = False,
        coordinates: Mapping[str, Any],
        seed_entropy: Sequence[int],
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        request = dict(observation)
        request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
        if self.route_off_smoke:
            if capture:
                raise ArtifactError("route-off smoke cannot issue a captured request")
            response = validate_action_response(self.client.infer(request))
            acknowledgement = {
                "schema": GATED_CAPTURE_SCHEMA,
                "capture_run_id": "",
                "query_id": query_id,
                "captured": False,
                "row": -1,
                "row_count": -1,
                "durable_through_row": -1,
            }
        else:
            request[CAPTURE_KEY] = bool(capture)
            request[QUERY_ID_KEY] = int(query_id)
            if capture:
                request[SNAPSHOT_INDEX_KEY] = int(snapshot_index)
                request[CANDIDATE_ID_KEY] = int(candidate_id)
                request[FLUSH_AFTER_KEY] = bool(flush_after)
            response = validate_action_response(self.client.infer(request))
            acknowledgement = response.get(ACK_KEY)
            if not isinstance(acknowledgement, Mapping):
                raise ArtifactError("request-gated server returned no recorder acknowledgement")
            acknowledgement = dict(acknowledgement)
            if acknowledgement.get("schema") != GATED_CAPTURE_SCHEMA:
                raise ArtifactError("recorder acknowledgement schema mismatch")
            if int(acknowledgement.get("query_id", -1)) != query_id:
                raise ArtifactError("recorder acknowledged a different query id")
            if bool(acknowledgement.get("captured")) is not capture:
                raise ArtifactError("recorder capture acknowledgement disagrees with request")
            run_id = acknowledgement.get("capture_run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ArtifactError("recorder acknowledgement has no capture_run_id")
            if (
                self.expected_capture_run_id is not None
                and run_id != self.expected_capture_run_id
            ):
                raise ArtifactError("recorder acknowledgement uses another capture_run_id")
            if self.capture_run_id is None:
                self.capture_run_id = run_id
            elif run_id != self.capture_run_id:
                raise ArtifactError("capture_run_id changed within one artifact")
            row = int(acknowledgement.get("row", -2))
            row_count = int(acknowledgement.get("row_count", -2))
            durable = int(acknowledgement.get("durable_through_row", -2))
            if capture:
                if row < 0 or row_count != row + 1:
                    raise ArtifactError("captured acknowledgement has inconsistent row counters")
                if (
                    self.last_capture_row is None
                    and self.expected_recorder_rows is not None
                    and row != self.expected_recorder_rows
                ):
                    raise ArtifactError(
                        "first captured row differs from the artifact-confirmed prefix"
                    )
                if self.last_capture_row is not None and row != self.last_capture_row + 1:
                    raise ArtifactError("captured server rows are not contiguous")
                self.last_capture_row = row
                if flush_after and durable != row_count - 1:
                    raise ArtifactError(
                        "flush_after did not make the exact final candidate row durable"
                    )
            else:
                if row != -1:
                    raise ArtifactError("uncaptured request unexpectedly received a server row")
                boundary = (row_count, durable)
                if durable != row_count - 1:
                    raise ArtifactError("uncaptured request observed a pending recorder pool")
                if (
                    self.expected_recorder_rows is not None
                    and boundary
                    != (self.expected_recorder_rows, self.expected_recorder_rows - 1)
                ):
                    raise ArtifactError(
                        "uncaptured request boundary differs from the artifact-confirmed prefix"
                    )
                if self.uncaptured_boundary is None:
                    self.uncaptured_boundary = boundary
                elif boundary != self.uncaptured_boundary:
                    raise ArtifactError("uncaptured continuation advanced recorder storage")
        expected_noise = __import__("hashlib").sha256(
            np.ascontiguousarray(noise, dtype=np.float32).tobytes()
        ).hexdigest()
        if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise:
            raise ArtifactError("policy did not acknowledge the requested flow noise")
        record = {
            "query_id": int(query_id),
            "captured": bool(capture),
            "server_row": int(acknowledgement["row"]),
            "capture_run_id": str(acknowledgement["capture_run_id"]),
            "row_count": int(acknowledgement["row_count"]),
            "durable_through_row": int(acknowledgement["durable_through_row"]),
            "flush_after": bool(flush_after),
            "coordinates": dict(coordinates),
            "seed_words_uint32": [int(value) for value in seed_entropy],
            "flow_noise_sha256": expected_noise,
        }
        self.records.append(record)
        return response, int(acknowledgement["row"]), acknowledgement


def _state_from_plan(plan: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    matches = [row for row in plan["states"] if row["state_id"] == identifier]
    if len(matches) != 1:
        raise ArtifactError(f"plan contains {len(matches)} rows for state {identifier!r}")
    return dict(matches[0])


def _episode_config(
    state: Mapping[str, Any],
    benchmark: str,
    libero_root: str,
    output_root: Path,
    max_steps: int,
    settle_steps: int = 10,
) -> EpisodeConfig:
    return EpisodeConfig(
        task_suite=benchmark,
        task_id=int(state["task_id"]),
        init_state_id=int(state["init_state_id"]),
        seed=int(state["environment_seed"]),
        host="127.0.0.1",
        port=1,
        libero_root=libero_root,
        output_root=str(output_root),
        settle_steps=settle_steps,
        max_steps=max_steps,
        replan_steps=10,
        inference_timeout=300.0,
    )


def _settle(environment: Any, count: int) -> Mapping[str, Any]:
    observation = None
    for _ in range(count):
        observation, _reward, _done, _info = environment.step(LIBERO_DUMMY_ACTION.tolist())
    if observation is None:
        raise ArtifactError("settle_steps must be positive")
    return observation


def capture_source_events(args: argparse.Namespace) -> int:
    source = args.source_dir.expanduser().resolve()
    summaries_path = source / "summaries.json"
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    by_episode = {int(row["episode_index"]): row for row in summaries}
    episodes = _parse_ints(args.episodes, sorted(by_episode))
    for episode in episodes:
        summary = by_episode[episode]
        if int(summary["task_id"]) != args.task_id:
            raise ArtifactError(f"episode {episode} is not task {args.task_id}")
        target = args.out.resolve() / f"task_{args.task_id:02d}" / f"episode_{episode:04d}"
        descriptor_path = target / "artifact.json"
        if descriptor_path.exists():
            verify_artifact(descriptor_path, EVENT_SCHEMA)
            print(f"source events already complete: task={args.task_id} episode={episode}")
            continue
        actions, source_sim = _source_episode(source / f"episode_{episode:02d}.npz")
        total_actions = int(summary["action_steps"])
        state = {
            "task_id": args.task_id,
            "init_state_id": int(summary["init_state_id"]),
            "environment_seed": int(summary["seed"]),
        }
        config = _episode_config(
            state,
            args.benchmark,
            args.libero_root,
            args.out,
            max(1000, total_actions + args.settle_steps + 20),
            args.settle_steps,
        )
        environment, observation, task, prompt = _load_task(config)
        try:
            observation = _settle(environment, args.settle_steps)
            first = _physical_point(environment, observation)
            state_dim, gripper_dim = len(first[0]), len(first[3])
            layout = _enhanced_sim_layout(environment, state_dim)
            t_count, h = len(actions), actions.shape[1]
            sim = np.empty((t_count, h + 1, state_dim), dtype=np.float64)
            eef = np.empty((t_count, h + 1, 3), dtype=np.float64)
            quat = np.empty((t_count, h + 1, 4), dtype=np.float64)
            gripper = np.empty((t_count, h + 1, gripper_dim), dtype=np.float64)
            success = np.empty((t_count, h + 1), dtype=np.bool_)
            valid = np.zeros((t_count, h + 1), dtype=np.bool_)
            contact_sets: list[list[set[tuple[str, str]]]] = []
            replay_drift = np.empty(t_count, dtype=np.float64)
            executed = 0
            for control_step in range(t_count):
                current_sim = np.asarray(environment.get_sim_state(), dtype=np.float64)
                if current_sim.shape != source_sim[control_step].shape:
                    raise ArtifactError("source replay sim-state dimension changed")
                replay_drift[control_step] = float(
                    np.max(np.abs(current_sim - source_sim[control_step]))
                )
                point = _physical_point(environment, observation, gripper_dim)
                points = [point]
                valid[control_step, 0] = executed < total_actions
                step_count = min(h, max(0, total_actions - executed))
                for action in actions[control_step, :step_count]:
                    observation, _reward, _done, _info = environment.step(
                        np.asarray(action, dtype=np.float64).tolist()
                    )
                    executed += 1
                    points.append(_physical_point(environment, observation, gripper_dim))
                    valid[control_step, len(points) - 1] = True
                while len(points) < h + 1:
                    points.append(points[-1])
                sim[control_step] = np.stack([point[0] for point in points])
                eef[control_step] = np.stack([point[1] for point in points])
                quat[control_step] = np.stack([point[2] for point in points])
                gripper[control_step] = np.stack([point[3] for point in points])
                success[control_step] = np.asarray([point[4] for point in points])
                contact_sets.append([point[5] for point in points])
            if executed != total_actions:
                raise ArtifactError(
                    f"source replay executed {executed} actions, expected {total_actions}"
                )
            if bool(success[valid][-1]) != bool(summary["success"]):
                raise ArtifactError("source replay terminal success disagrees with summaries.json")
            replay_max = float(replay_drift.max())
            if replay_max > args.source_fidelity_tol:
                raise ArtifactError(
                    f"source replay fidelity failed: {replay_max:.3e} > "
                    f"{args.source_fidelity_tol:.3e}"
                )
            contact, names, _pairs = _contact_tape(contact_sets)
            events = dense_event_tape(
                contact,
                gripper,
                sim,
                success,
                int(layout["nq"]),
                layout["object_qpos_indices"],
            )
        finally:
            environment.close()
        stage = _stage(target)
        arrays = {
            "sim_states": sim,
            "eef_positions": eef,
            "eef_quaternions": quat,
            "gripper_qpos": gripper,
            "success": success,
            "valid_steps": valid,
            "contact_active": contact,
            "contact_pair_names": names,
            "event_flags": events,
            "event_labels": np.asarray(EVENT_LABELS),
            "source_replay_drift": replay_drift,
        }
        atomic_npz(stage / "events.npz", arrays)
        atomic_json(stage / "layout.json", layout)
        commit_artifact(
            stage,
            EVENT_SCHEMA,
            {
                "task_id": args.task_id,
                "episode": episode,
                "source_success": bool(summary["success"]),
                "source_episode_file": str(source / f"episode_{episode:02d}.npz"),
                "source_episode_sha256": sha256_file(source / f"episode_{episode:02d}.npz"),
                "summaries_sha256": sha256_file(summaries_path),
                "task_name": str(task.name),
                "prompt": prompt,
                "selection_contract": "physical replay only; no candidate or route data",
                "replay_fidelity_metric": "max_abs_flat_sim_at_each_control_query",
                "replay_fidelity_max_abs": replay_max,
                "replay_fidelity_tolerance": args.source_fidelity_tol,
                "replay_fidelity_passed": True,
                "settle_steps": args.settle_steps,
            },
            {"events": stage / "events.npz", "layout": stage / "layout.json"},
        )
        _publish(stage, target)
        print(f"captured source events: task={args.task_id} episode={episode}")
    return 0


def _reach_snapshot(environment: Any, actions: np.ndarray, fork_step: int, settle_steps: int) -> Mapping[str, Any]:
    observation = _settle(environment, settle_steps)
    if fork_step > len(actions):
        raise ArtifactError("fork_step exceeds source inference calls")
    for chunk in actions[:fork_step]:
        for action in chunk:
            observation, _reward, _done, _info = environment.step(
                np.asarray(action, dtype=np.float64).tolist()
            )
    return observation


def _server_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol",
        "protocol_version",
        "checkpoint_sha256",
        "checkpoint_path",
        "normalization_stats_sha256",
        "normalization_action_std",
        "libero_wrist_layout",
        "himoe_upstream_commit",
        "himoe_patch_sha256",
        "himoe_working_tree_dirty",
        "himoe_working_tree_diff_sha256",
        "himoe_patches_verified_applied",
        "route_recorder",
        "route_axis_n_denoise",
        "route_axis_n_suffix",
        "n_action_steps",
        "num_steps",
        "candidate_capture_supported",
        "candidate_capture_schema",
        "candidate_capture_run_id",
    )
    return {key: metadata.get(key) for key in keys}


def _require_gated_metadata(metadata: Mapping[str, Any], expected_run_id: str | None = None) -> str:
    if metadata.get("candidate_capture_supported") is not True:
        raise ArtifactError("server does not advertise request-gated candidate capture")
    if metadata.get("candidate_capture_schema") != GATED_CAPTURE_SCHEMA:
        raise ArtifactError("server candidate-capture schema mismatch")
    expected_request_keys = {
        "capture": CAPTURE_KEY,
        "query_id": QUERY_ID_KEY,
        "snapshot_index": SNAPSHOT_INDEX_KEY,
        "candidate_id": CANDIDATE_ID_KEY,
        "flush_after": FLUSH_AFTER_KEY,
    }
    if metadata.get("candidate_capture_request_keys") != expected_request_keys:
        raise ArtifactError("server candidate-capture request keys mismatch")
    if metadata.get("candidate_capture_ack_key") != ACK_KEY:
        raise ArtifactError("server candidate-capture acknowledgement key mismatch")
    run_id = metadata.get("candidate_capture_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactError("server metadata has no candidate_capture_run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ArtifactError("server metadata capture_run_id differs from the staged artifact")
    return run_id


def _existing_candidate(target: Path) -> bool:
    descriptor = target / "artifact.json"
    if not descriptor.exists():
        return False
    metadata = verify_artifact(descriptor, CANDIDATE_SCHEMA)
    arrays = load_npz(target / metadata["files"]["data"]["file"])
    validate_candidate_arrays(arrays, metadata)
    return True


def capture_candidate(args: argparse.Namespace) -> int:
    plan = _load_plan(args.plan)
    state = _state_from_plan(plan, args.state_id)
    pool_spec = plan["pools"][args.pool]
    k = int(pool_spec["candidate_count"])
    h = int(plan["action_chunk_steps"])
    target = candidate_dir(args.out.resolve(), args.pool, args.state_id)
    if _existing_candidate(target):
        print(f"candidate artifact already complete: {target}")
        return 0
    source_actions = _source_actions(Path(state["source_episode_file"]))
    config = _episode_config(
        state,
        args.benchmark,
        args.libero_root,
        args.out,
        max(1000, (int(state["fork_step"]) + int(plan["continuation_control_steps"]) + 4) * h),
        int(state["settle_steps"]),
    )
    environment, _observation, task, prompt = _load_task(config)
    try:
        observation = _reach_snapshot(
            environment, source_actions, int(state["fork_step"]), config.settle_steps
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
            validate_policy_suite(client.metadata, args.benchmark)
            if not args.route_off_smoke and not client.metadata.get("route_recorder"):
                raise ArtifactError("candidate capture requires the request-gated recorder server")
            if not args.route_off_smoke:
                advertised_run_id = _require_gated_metadata(
                    client.metadata, args.route_store_id or None
                )
            else:
                advertised_run_id = ""
            ledger = GatedQueryLedger(
                client,
                route_off_smoke=args.route_off_smoke,
                expected_capture_run_id=args.route_store_id or None,
                expected_recorder_rows=args.expected_recorder_rows,
            )
            candidate_noise = np.empty((k,) + FLOW_NOISE_SHAPE, dtype=np.float32)
            actions = np.empty((k, h, 7), dtype=np.float32)
            query_ids = np.empty(k, dtype=np.int32)
            rows = np.empty(k, dtype=np.int64)
            store_ids = np.empty(k, dtype="<U64")
            candidate_domain = pool_domain(args.pool, "candidate")
            for candidate in range(k):
                words = seed_words(
                    candidate_domain,
                    int(state["task_id"]),
                    int(state["episode"]),
                    int(state["fork_step"]),
                    (candidate,),
                )
                noise = flow_noise(words)
                query_id = candidate_query_id(
                    int(state["snapshot_index"]), args.pool, candidate
                )
                response, row, acknowledgement = ledger.infer(
                    policy_observation,
                    noise,
                    query_id=query_id,
                    capture=not args.route_off_smoke,
                    snapshot_index=int(state["snapshot_index"]),
                    candidate_id=candidate,
                    flush_after=(candidate == k - 1 and not args.route_off_smoke),
                    coordinates={
                        "pool": args.pool,
                        "state_id": args.state_id,
                        "candidate": candidate,
                    },
                    seed_entropy=words,
                )
                candidate_noise[candidate] = noise
                actions[candidate] = np.asarray(response[ACTION_KEY], dtype=np.float32)[:h]
                rows[candidate], query_ids[candidate] = row, query_id
                store_ids[candidate] = str(acknowledgement["capture_run_id"])
            if args.route_store_id and ledger.capture_run_id != args.route_store_id:
                raise ArtifactError("server capture_run_id differs from --route-store-id")
            if not args.route_off_smoke and ledger.capture_run_id != advertised_run_id:
                raise ArtifactError("server metadata and acknowledgements disagree on capture_run_id")
            server_metadata = dict(client.metadata)
        order_words = seed_words(
            SEED_DOMAINS["execution"],
            int(state["task_id"]),
            int(state["episode"]),
            int(state["fork_step"]),
            (0 if args.pool == "screen" else 1,),
        )
        execution_order = np.asarray(rng_from_words(order_words).permutation(k), dtype=np.int32)
        sim = np.empty((k, h + 1, state_dim), dtype=np.float64)
        eef = np.empty((k, h + 1, 3), dtype=np.float64)
        quat = np.empty((k, h + 1, 4), dtype=np.float64)
        gripper = np.empty((k, h + 1, gripper_dim), dtype=np.float64)
        success = np.empty((k, h + 1), dtype=np.bool_)
        contact_sets: list[list[set[tuple[str, str]]]] = [[] for _ in range(k)]
        post_states: list[Mapping[str, Any] | None] = [None] * k
        for candidate in execution_order:
            observation = _hard_restore(environment, snapshot)
            points = [_physical_point(environment, observation, gripper_dim)]
            for action in actions[candidate]:
                observation, _reward, _done, _info = environment.step(
                    np.asarray(action, dtype=np.float64).tolist()
                )
                points.append(_physical_point(environment, observation, gripper_dim))
            sim[candidate] = np.stack([value[0] for value in points])
            eef[candidate] = np.stack([value[1] for value in points])
            quat[candidate] = np.stack([value[2] for value in points])
            gripper[candidate] = np.stack([value[3] for value in points])
            success[candidate] = np.asarray([value[4] for value in points])
            contact_sets[candidate] = [value[5] for value in points]
            post_states[candidate] = save_full_state(environment)
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
        for action in actions[reference]:
            observation, _reward, _done, _info = environment.step(
                np.asarray(action, dtype=np.float64).tolist()
            )
            rerun.append(np.asarray(environment.get_sim_state(), dtype=np.float64).copy())
        rerun_tape = np.stack(rerun)
        drift = float(np.max(np.abs(rerun_tape - sim[reference])))
        spread_mean, spread_max = _trajectory_pair_spread(sim)
        threshold = max(args.fidelity_abs_tol, args.fidelity_ratio * spread_mean)
        fidelity_passed = bool(drift <= threshold)
    finally:
        environment.close()
    if not fidelity_passed:
        raise ArtifactError(f"candidate fidelity failed: {drift} > {threshold}")
    arrays = {
        "candidate_ids": np.arange(k, dtype=np.int32),
        "actions": actions,
        "candidate_flow_noise": candidate_noise,
        "query_ids": query_ids,
        "server_trace_rows": rows,
        "store_ids": store_ids,
        "execution_order": execution_order,
        "sim_states": sim,
        "eef_positions": eef,
        "eef_quaternions": quat,
        "gripper_qpos": gripper,
        "chunk_success": success,
        "contact_active": contact,
        "contact_pair_names": contact_names,
        "event_flags": events,
        "event_labels": np.asarray(EVENT_LABELS),
        "robot_qpos_indices": np.asarray(layout["robot_qpos_indices"], dtype=np.int32),
        "object_qpos_indices": np.asarray(layout["object_qpos_indices"], dtype=np.int32),
        "nq": np.asarray(layout["nq"], dtype=np.int32),
        "fidelity_reference_candidate": np.asarray(reference, dtype=np.int32),
        "fidelity_rerun_sim_states": rerun_tape,
        "fidelity_drift": np.asarray(drift),
        "fidelity_candidate_spread_mean": np.asarray(spread_mean),
        "fidelity_candidate_spread_max": np.asarray(spread_max),
        "fidelity_threshold": np.asarray(threshold),
        "fidelity_passed": np.asarray(fidelity_passed),
    }
    metadata = {
        "pool": args.pool,
        "state_id": args.state_id,
        "snapshot_index": int(state["snapshot_index"]),
        "task_id": int(state["task_id"]),
        "episode": int(state["episode"]),
        "fork_step": int(state["fork_step"]),
        "init_state_id": int(state["init_state_id"]),
        "environment_seed": int(state["environment_seed"]),
        "settle_steps": int(state["settle_steps"]),
        "benchmark": args.benchmark,
        "libero_root": str(Path(args.libero_root).expanduser().resolve()),
        "task_name": str(task.name),
        "prompt": prompt,
        "candidate_count": k,
        "action_chunk_steps": h,
        "continuation_control_steps": int(plan["continuation_control_steps"]),
        "plan_file": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "source_episode_file": state["source_episode_file"],
        "source_episode_sha256": state["source_episode_sha256"],
        "route_capture": not args.route_off_smoke,
        "route_off_smoke": bool(args.route_off_smoke),
        "route_store_id": ledger.capture_run_id,
        "server_row_start": None if args.route_off_smoke else int(rows[0]),
        "server_row_stop": None if args.route_off_smoke else int(rows[-1]) + 1,
        "server_durable_through_row": int(ledger.records[-1]["durable_through_row"]),
        "server_identity": _server_identity(server_metadata),
        "seed_domain": pool_domain(args.pool, "candidate"),
        "root_seed": ROOT_SEED,
    }
    validate_candidate_arrays(arrays, metadata)
    if any(value is None for value in post_states):
        raise ArtifactError("candidate post-state bundle is incomplete")
    stage = _stage(target)
    atomic_npz(stage / "candidate.npz", arrays)
    atomic_json(stage / "layout.json", layout)
    atomic_json(stage / "queries.json", {"queries": ledger.records})
    atomic_json(stage / "server_metadata.json", server_metadata)
    state_files = write_full_state_bundle(stage / "post_states", post_states)  # type: ignore[arg-type]
    commit_artifact(
        stage,
        CANDIDATE_SCHEMA,
        metadata,
        {
            "data": stage / "candidate.npz",
            "layout": stage / "layout.json",
            "queries": stage / "queries.json",
            "server_metadata": stage / "server_metadata.json",
            **state_files,
        },
    )
    _publish(stage, target)
    print(f"captured {args.pool} candidate pool: {target}")
    return 0


def capture_continuation_shard(args: argparse.Namespace) -> int:
    candidate_path = args.candidate_artifact.resolve()
    candidate_meta = verify_artifact(candidate_path, CANDIDATE_SCHEMA)
    pool = str(candidate_meta["pool"])
    cohort = args.cohort
    if pool != cohort_candidate_pool(cohort):
        raise ArtifactError(
            f"cohort {cohort!r} requires {cohort_candidate_pool(cohort)!r} candidates, got {pool!r}"
        )
    identifier = str(candidate_meta["state_id"])
    target = shard_dir(args.out.resolve(), pool, identifier, cohort, args.shard_index)
    descriptor_path = target / "artifact.json"
    if descriptor_path.exists():
        shard_meta = verify_artifact(descriptor_path, SHARD_SCHEMA)
        candidate_arrays = load_npz(
            candidate_path.parent / candidate_meta["files"]["data"]["file"]
        )
        shard_arrays = load_npz(target / shard_meta["files"]["data"]["file"])
        validate_shard_arrays(shard_arrays, shard_meta, len(candidate_arrays["candidate_ids"]))
        print(f"continuation shard already complete: {target}")
        return 0
    candidate_arrays = load_npz(
        candidate_path.parent / candidate_meta["files"]["data"]["file"]
    )
    validate_candidate_arrays(candidate_arrays, candidate_meta)
    post_states = read_full_state_bundle(
        candidate_path.parent / candidate_meta["files"]["full_states_meta"]["file"],
        candidate_path.parent / candidate_meta["files"]["full_states_arrays"]["file"],
    )
    k = len(candidate_arrays["candidate_ids"])
    if len(post_states) != k:
        raise ArtifactError("post-state count disagrees with candidate count")
    h = int(candidate_meta["action_chunk_steps"])
    b = int(candidate_meta["continuation_control_steps"])
    repeat_start = args.shard_index * SHARD_REPEATS
    seed_pool = cohort_seed_pool(cohort)
    domain = pool_domain(seed_pool, "continuation")
    noise = np.empty((SHARD_REPEATS, b) + FLOW_NOISE_SHAPE, dtype=np.float32)
    for local_repeat in range(SHARD_REPEATS):
        repeat = repeat_start + local_repeat
        for future_step in range(b):
            noise[local_repeat, future_step] = flow_noise(
                seed_words(
                    domain,
                    int(candidate_meta["task_id"]),
                    int(candidate_meta["episode"]),
                    int(candidate_meta["fork_step"]),
                    (repeat, future_step),
                )
            )
    state = {
        "task_id": candidate_meta["task_id"],
        "init_state_id": candidate_meta["init_state_id"],
        "environment_seed": candidate_meta["environment_seed"],
    }
    config = _episode_config(
        state,
        str(candidate_meta["benchmark"]),
        args.libero_root,
        args.out,
        max(1000, (int(candidate_meta["fork_step"]) + b + 4) * h),
        int(candidate_meta.get("settle_steps", 10)),
    )
    environment, _observation, _task, prompt = _load_task(config)
    try:
        _settle(environment, config.settle_steps)
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, str(candidate_meta["benchmark"]))
            candidate_smoke = bool(candidate_meta.get("route_off_smoke", False))
            if bool(args.route_off_smoke) is not candidate_smoke:
                raise ArtifactError("candidate and continuation route-off-smoke modes disagree")
            if not args.route_off_smoke and not client.metadata.get("route_recorder"):
                raise ArtifactError("continuation requires the same request-gated recorder server")
            if not args.route_off_smoke:
                _require_gated_metadata(
                    client.metadata, str(candidate_meta["route_store_id"])
                )
            current_identity = _server_identity(client.metadata)
            expected_identity = dict(candidate_meta["server_identity"])
            for key in ("checkpoint_sha256", "normalization_action_std", "libero_wrist_layout"):
                if current_identity.get(key) != expected_identity.get(key):
                    raise ArtifactError(f"continuation server differs from candidate server: {key}")
            ledger = GatedQueryLedger(
                client,
                route_off_smoke=args.route_off_smoke,
                expected_capture_run_id=(
                    None
                    if args.route_off_smoke
                    else str(candidate_meta["route_store_id"])
                ),
                expected_recorder_rows=args.expected_recorder_rows,
            )
            outcomes = np.zeros((k, SHARD_REPEATS), dtype=np.bool_)
            final_states = np.empty(
                (k, SHARD_REPEATS, candidate_arrays["sim_states"].shape[-1]), dtype=np.float64
            )
            action_steps = np.zeros((k, SHARD_REPEATS), dtype=np.int32)
            query_ids = np.full((k, SHARD_REPEATS, b), -1, dtype=np.int32)
            rows = np.full((k, SHARD_REPEATS, b), -1, dtype=np.int64)
            row_counts = np.full((k, SHARD_REPEATS, b), -1, dtype=np.int64)
            durable_rows = np.full((k, SHARD_REPEATS, b), -1, dtype=np.int64)
            query_executed = np.zeros((k, SHARD_REPEATS, b), dtype=np.bool_)
            continuation_actions = np.zeros((k, SHARD_REPEATS, b, h, 7), dtype=np.float32)
            action_executed = np.zeros((k, SHARD_REPEATS, b, h), dtype=np.bool_)
            for candidate in range(k):
                for local_repeat in range(SHARD_REPEATS):
                    observation = _hard_restore(environment, post_states[candidate])
                    succeeded = bool(environment.check_success())
                    for future_step in range(b):
                        if succeeded:
                            break
                        repeat = repeat_start + local_repeat
                        words = seed_words(
                            domain,
                            int(candidate_meta["task_id"]),
                            int(candidate_meta["episode"]),
                            int(candidate_meta["fork_step"]),
                            (repeat, future_step),
                        )
                        query_id = continuation_query_id(
                            int(candidate_meta["snapshot_index"]),
                            cohort,
                            candidate,
                            repeat,
                            future_step,
                            b,
                        )
                        response, row, acknowledgement = ledger.infer(
                            build_policy_observation(observation, prompt),
                            noise[local_repeat, future_step],
                            query_id=query_id,
                            capture=False,
                            coordinates={
                                "pool": pool,
                                "label_cohort": cohort,
                                "state_id": identifier,
                                "candidate": candidate,
                                "repeat": repeat,
                                "future_step": future_step,
                            },
                            seed_entropy=words,
                        )
                        if not args.route_off_smoke:
                            if acknowledgement["capture_run_id"] != candidate_meta.get(
                                "route_store_id"
                            ):
                                raise ArtifactError(
                                    "continuation acknowledgement uses another capture_run_id"
                                )
                            if int(acknowledgement["durable_through_row"]) < int(
                                candidate_meta["server_durable_through_row"]
                            ):
                                raise ArtifactError(
                                    "recorder durable boundary regressed after candidate capture"
                                )
                        query_ids[candidate, local_repeat, future_step] = query_id
                        rows[candidate, local_repeat, future_step] = row
                        row_counts[candidate, local_repeat, future_step] = int(
                            acknowledgement["row_count"]
                        )
                        durable_rows[candidate, local_repeat, future_step] = int(
                            acknowledgement["durable_through_row"]
                        )
                        query_executed[candidate, local_repeat, future_step] = True
                        actions = np.asarray(response[ACTION_KEY], dtype=np.float32)[:h]
                        continuation_actions[candidate, local_repeat, future_step] = actions
                        for action_index, action in enumerate(actions):
                            observation, _reward, _done, _info = environment.step(
                                np.asarray(action, dtype=np.float64).tolist()
                            )
                            action_executed[
                                candidate, local_repeat, future_step, action_index
                            ] = True
                            action_steps[candidate, local_repeat] += 1
                            succeeded = bool(environment.check_success())
                            if succeeded:
                                break
                    outcomes[candidate, local_repeat] = succeeded
                    final_states[candidate, local_repeat] = np.asarray(
                        environment.get_sim_state(), dtype=np.float64
                    )
    finally:
        environment.close()
    arrays = {
        "continuation_success": outcomes,
        "continuation_final_sim_states": final_states,
        "continuation_action_steps": action_steps,
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
        "pool": pool,
        "label_cohort": cohort,
        "seed_pool": seed_pool,
        "state_id": identifier,
        "task_id": int(candidate_meta["task_id"]),
        "episode": int(candidate_meta["episode"]),
        "fork_step": int(candidate_meta["fork_step"]),
        "shard_index": args.shard_index,
        "repeat_start": repeat_start,
        "repeat_stop": repeat_start + SHARD_REPEATS,
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256_file(candidate_path),
        "route_capture": False,
        "route_off_smoke": bool(args.route_off_smoke),
        "capture_run_id": ledger.capture_run_id,
        "recorder_boundary": None
        if ledger.uncaptured_boundary is None
        else list(ledger.uncaptured_boundary),
        "seed_domain": domain,
        "root_seed": ROOT_SEED,
    }
    validate_shard_arrays(arrays, metadata, k)
    stage = _stage(target)
    atomic_npz(stage / "continuation.npz", arrays)
    atomic_json(stage / "queries.json", {"queries": ledger.records})
    commit_artifact(
        stage,
        SHARD_SCHEMA,
        metadata,
        {"data": stage / "continuation.npz", "queries": stage / "queries.json"},
    )
    _publish(stage, target)
    print(f"captured continuation repeats [{repeat_start},{repeat_start + SHARD_REPEATS}): {target}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    events = sub.add_parser("source-events")
    events.add_argument("--source-dir", type=Path, required=True)
    events.add_argument("--task-id", type=int, required=True)
    events.add_argument("--episodes")
    events.add_argument("--benchmark", default="libero_goal")
    events.add_argument("--libero-root", required=True)
    events.add_argument("--settle-steps", type=int, default=10)
    events.add_argument("--source-fidelity-tol", type=float, default=1e-5)
    events.add_argument("--out", type=Path, required=True)

    candidate = sub.add_parser("candidate")
    candidate.add_argument("--plan", type=Path, required=True)
    candidate.add_argument("--state-id", required=True)
    candidate.add_argument("--pool", choices=("screen", "formal"), required=True)
    candidate.add_argument("--host", default="127.0.0.1")
    candidate.add_argument("--port", type=int, required=True)
    candidate.add_argument("--benchmark", default="libero_goal")
    candidate.add_argument("--libero-root", required=True)
    candidate.add_argument("--out", type=Path, required=True)
    candidate.add_argument("--route-store-id", default="",
                           help="optional expected server capture_run_id")
    candidate.add_argument(
        "--expected-recorder-rows",
        type=int,
        help="artifact-confirmed shared store prefix before this candidate pool",
    )
    candidate.add_argument("--route-off-smoke", action="store_true",
                           help="explicit smoke-only mode against an ordinary non-recorder server")
    candidate.add_argument("--fidelity-ratio", type=float, default=0.05)
    candidate.add_argument("--fidelity-abs-tol", type=float, default=1e-10)
    candidate.add_argument("--inference-timeout", type=float, default=300.0)

    shard = sub.add_parser("continuation-shard")
    shard.add_argument("--candidate-artifact", type=Path, required=True)
    shard.add_argument("--cohort", choices=LABEL_COHORTS, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--host", default="127.0.0.1")
    shard.add_argument("--port", type=int, required=True)
    shard.add_argument("--libero-root", required=True)
    shard.add_argument("--out", type=Path, required=True)
    shard.add_argument("--route-off-smoke", action="store_true")
    shard.add_argument(
        "--expected-recorder-rows",
        type=int,
        help="artifact-confirmed fully durable candidate-store row count",
    )
    shard.add_argument("--inference-timeout", type=float, default=300.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "source-events":
        if args.settle_steps < 0:
            raise SystemExit("--settle-steps must be non-negative")
        if not np.isfinite(args.source_fidelity_tol) or args.source_fidelity_tol <= 0.0:
            raise SystemExit("--source-fidelity-tol must be finite and positive")
        return capture_source_events(args)
    if args.command == "candidate":
        if args.expected_recorder_rows is not None and args.expected_recorder_rows < 0:
            raise SystemExit("--expected-recorder-rows must be non-negative")
        return capture_candidate(args)
    if args.command == "continuation-shard":
        if args.shard_index < 0:
            raise SystemExit("--shard-index must be non-negative")
        if args.expected_recorder_rows is not None and args.expected_recorder_rows < 0:
            raise SystemExit("--expected-recorder-rows must be non-negative")
        return capture_continuation_shard(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
