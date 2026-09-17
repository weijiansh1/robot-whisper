"""Closed-loop single-arm and recoverable paired-block RAD evaluation."""

from __future__ import annotations

import dataclasses
import datetime
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import sha256_file
from himoe_libero_bridge.libero_runtime import (
    EventLog,
    LIBERO_DUMMY_ACTION,
    _jsonable,
    _load_task,
    _render_stream_metadata,
    _sha256_array,
    _sim_state,
    _stable_policy_identity,
    _write_video,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation, frame_from_observation
from himoe_libero_bridge.policies import (
    HIMOE_UPSTREAM_COMMIT,
    LIBERO_UPSTREAM_COMMIT,
    himoe_patch_hashes,
)
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    IMAGE_KEY,
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    ROUTING_EXPERT_WEIGHTS_KEY,
    ROUTING_LAYER_INDICES_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    validate_action_response,
)
from himoe_libero_bridge.rad_experiment import (
    PAIRED_SELECTORS,
    RAD_EXPERIMENT_SCHEMA,
    SELECTOR_ACTION_MEDOID,
    SELECTOR_K1,
    SELECTOR_RANDOM,
    SELECTOR_ROUTING_MEDOID,
    generate_noise_schedule,
    generate_random_selector_schedule,
    select_candidate,
)
from himoe_libero_bridge.rad_trace import (
    RAD_TRACE_MANIFEST_FILE,
    load_rad_trace,
    save_rad_trace,
)
from himoe_libero_bridge.routing import routing_metric_contract

RAD_POOL_SIZE = 8
RAD_BLOCK_SCHEMA = "himoe-libero-rad-block-v3"
RAD_SOURCE_IDENTITY_SCHEMA = "himoe-libero-rad-source-identity-v1"
RAD_BLOCK_CONFIG_FILE = "rad-block-config.json"
RAD_BLOCK_MANIFEST_FILE = "rad-block-manifest.json"
RAD_BLOCK_SUMMARY_FILE = "rad-block-summary.json"
_ARM_ORDER_STREAM_TAG = 0x41524D4F


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_tree(root: pathlib.Path, directories: Sequence[pathlib.Path]) -> Dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError("Missing source/asset directory: %s" % directory)
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            payload = path.read_bytes()
            digest.update(str(len(relative)).encode("ascii") + b"\0" + relative)
            digest.update(str(len(payload)).encode("ascii") + b"\0" + payload)
            file_count += 1
            byte_count += len(payload)
    return {
        "algorithm": "sha256-path-length-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "byte_count": byte_count,
    }


def _local_bridge_source_identity() -> Dict[str, Any]:
    root = pathlib.Path(__file__).resolve().parents[2]
    identity = _hash_tree(root, (root / "src" / "himoe_libero_bridge",))
    identity.update({"kind": "runtime-source-tree", "root_name": root.name})
    return identity


def _git_commit(root: pathlib.Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not inspect LIBERO source %s: %s" % (root, error))
    if completed.returncode != 0:
        raise RuntimeError("Could not inspect LIBERO source commit: %s" % root)
    return completed.stdout.decode("utf-8").strip()


def collect_rad_experiment_identity(
    config: Any, bridge_source_identity: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    libero_root = pathlib.Path(config.libero_root).expanduser().resolve()
    actual_libero_commit = _git_commit(libero_root)
    if actual_libero_commit != LIBERO_UPSTREAM_COMMIT:
        raise RuntimeError(
            "LIBERO upstream commit mismatch: expected %s, got %s"
            % (LIBERO_UPSTREAM_COMMIT, actual_libero_commit)
        )
    benchmark_root = libero_root / "libero" / "libero"
    suite_assets = _hash_tree(
        libero_root,
        (
            benchmark_root / "bddl_files" / config.task_suite,
            benchmark_root / "init_files" / config.task_suite,
        ),
    )
    bridge_identity = dict(
        bridge_source_identity
        if bridge_source_identity is not None
        else _local_bridge_source_identity()
    )
    identity = {
        "schema": RAD_SOURCE_IDENTITY_SCHEMA,
        "bridge_source_identity": bridge_identity,
        "bridge_source_identity_sha256": _canonical_json_sha256(bridge_identity),
        "himoe": {
            "upstream_commit": HIMOE_UPSTREAM_COMMIT,
            "patch_sha256": himoe_patch_hashes(),
        },
        "libero": {
            "upstream_commit": actual_libero_commit,
            "suite_asset_identity": suite_assets,
        },
        "case": {
            "task_suite": config.task_suite,
            "task_id": int(config.task_id),
            "init_state_id": int(config.init_state_id),
        },
    }
    identity["identity_sha256"] = _canonical_json_sha256(identity)
    return identity


def _validate_experiment_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    canonical = json.loads(
        json.dumps(_jsonable(identity), allow_nan=False, sort_keys=True)
    )
    declared = canonical.pop("identity_sha256", None)
    if canonical.get("schema") != RAD_SOURCE_IDENTITY_SCHEMA:
        raise RuntimeError("RAD experiment source identity schema is invalid")
    if declared != _canonical_json_sha256(canonical):
        raise RuntimeError("RAD experiment source identity SHA-256 mismatch")
    canonical["identity_sha256"] = declared
    return canonical


@dataclasses.dataclass(frozen=True)
class RadEpisodeConfig:
    libero_root: str
    output_root: str = "artifacts"
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite: str = "libero_goal"
    task_id: int = 0
    init_state_id: int = 0
    seed: int = 7
    settle_steps: int = 10
    max_steps: int = 300
    replan_steps: int = 10
    render_size: int = 256
    fps: int = 20
    inference_timeout: float = 180.0
    selector: str = SELECTOR_K1
    noise_seed: int = 42
    random_selector_seed: int = 1042
    experiment_identity: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass(frozen=True)
class RadBlockConfig:
    block_dir: str
    libero_root: str
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite: str = "libero_goal"
    task_id: int = 0
    init_state_id: int = 0
    seed: int = 7
    settle_steps: int = 10
    max_steps: int = 300
    replan_steps: int = 10
    render_size: int = 256
    fps: int = 20
    inference_timeout: float = 180.0
    noise_seed: int = 42
    random_selector_seed: int = 1042
    arm_order_seed: int = 2042


class RadEpisodeRunError(RuntimeError):
    def __init__(self, artifact_dir: pathlib.Path, cause: BaseException) -> None:
        self.artifact_dir = artifact_dir
        self.cause = cause
        super().__init__("RAD episode failed in %s: %s" % (artifact_dir, cause))


def _validate_rad_episode_config(config: RadEpisodeConfig) -> None:
    if config.selector not in PAIRED_SELECTORS:
        raise ValueError("selector must be one of %s" % (PAIRED_SELECTORS,))
    if config.noise_seed < 0 or config.random_selector_seed < 0:
        raise ValueError("RAD seeds must be non-negative")
    if config.max_steps < 1 or not 1 <= config.replan_steps <= 10:
        raise ValueError("RAD step limits are invalid")
    if config.settle_steps < 0 or config.render_size < 1 or config.fps < 1:
        raise ValueError("RAD environment/render configuration is invalid")


def _normalization_action_std(metadata: Mapping[str, Any]) -> np.ndarray:
    raw = metadata.get("normalization_action_std")
    values = np.asarray(raw)
    if values.shape != (7,) or not np.issubdtype(values.dtype, np.number):
        raise RuntimeError("Policy metadata normalization_action_std must have shape (7,)")
    canonical = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(canonical)) or np.any(canonical <= 0.0):
        raise RuntimeError(
            "Policy metadata normalization_action_std must be finite and positive"
        )
    return np.ascontiguousarray(canonical)


def _rad_policy_identity(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    identity = _stable_policy_identity(metadata)
    try:
        routing_layers = [int(value) for value in metadata["routing_hb_layer_indices"]]
        routing_experts = int(metadata["routing_experts"])
        routing_top_k = int(metadata["routing_top_k"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Policy metadata is missing the RAD routing identity") from error
    if (
        len(routing_layers) != 8
        or any(right <= left for left, right in zip(routing_layers, routing_layers[1:]))
    ):
        raise RuntimeError("Policy metadata routing_hb_layer_indices is invalid")
    identity.update(
        {
            "routing_hb_layer_indices": routing_layers,
            "routing_experts": routing_experts,
            "routing_top_k": routing_top_k,
            "normalization_action_std": _normalization_action_std(metadata).tolist(),
        }
    )
    backend = identity["backend"]
    if isinstance(backend, str) and backend.startswith("himoe-vla-"):
        required = (
            "himoe_upstream_commit",
            "himoe_patch_sha256",
            "himoe_patches_verified_applied",
            "himoe_working_tree_dirty",
            "himoe_working_tree_diff_sha256",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise RuntimeError(
                "Policy metadata is missing HiMoE source identity fields: %s"
                % ", ".join(missing)
            )
        if metadata["himoe_upstream_commit"] != HIMOE_UPSTREAM_COMMIT:
            raise RuntimeError("Policy server reports an unexpected HiMoE commit")
        if metadata["himoe_patch_sha256"] != himoe_patch_hashes():
            raise RuntimeError("Policy server reports unexpected HiMoE patch hashes")
        if metadata["himoe_patches_verified_applied"] is not True:
            raise RuntimeError("Policy server did not verify the HiMoE runtime patches")
        identity.update({key: metadata[key] for key in required})
    return identity


def _new_rad_artifact_dir(config: RadEpisodeConfig) -> pathlib.Path:
    root = pathlib.Path(config.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suite = config.task_suite.replace("_", "-")
    path = root / (
        "rad-%s-%s-task%02d-init%02d-seed%d-%s"
        % (
            config.selector.replace("_", "-"),
            suite,
            config.task_id,
            config.init_state_id,
            config.seed,
            timestamp,
        )
    )
    path.mkdir(parents=False, exist_ok=False)
    return path


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _stack(records: Sequence[np.ndarray], dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(np.stack(records, axis=0), dtype=dtype)


def _rad_trace_arrays(
    records: Mapping[str, Any],
    noise_schedule: np.ndarray,
    layer_indices: np.ndarray,
    final_observation: Mapping[str, Any],
    final_sim_state: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "images": _stack(records["images"], np.uint8),
        "wrist_images": _stack(records["wrist_images"], np.uint8),
        "states": _stack(records["states"], np.float32),
        "sim_states_before": _stack(records["sim_states_before"], np.float64),
        "noise_schedule": np.ascontiguousarray(noise_schedule, dtype=np.float32),
        "candidate_actions": _stack(records["candidate_actions"], np.float32),
        "expert_ids": _stack(records["expert_ids"], np.int16),
        "expert_weights": _stack(records["expert_weights"], np.float32),
        "layer_indices": np.ascontiguousarray(layer_indices, dtype=np.int16),
        "selected_indices": np.asarray(records["selected_indices"], dtype=np.int16),
        "selector_scores": _stack(records["selector_scores"], np.float64),
        "selector_score_mask": _stack(records["selector_score_mask"], np.bool_),
        "selected_actions": _stack(records["selected_actions"], np.float32),
        "candidate_inference_ms": _stack(records["candidate_inference_ms"], np.float64),
        "executed_lengths": np.asarray(records["executed_lengths"], dtype=np.int16),
        "replan_start_steps": np.asarray(records["replan_start_steps"], dtype=np.int32),
        "executed_actions": _stack(records["executed_actions"], np.float32),
        "rewards": np.asarray(records["rewards"], dtype=np.float64),
        "dones": np.asarray(records["dones"], dtype=np.bool_),
        "successes": np.asarray(records["successes"], dtype=np.bool_),
        "sim_states_after": _stack(records["sim_states_after"], np.float64),
        "final_image": np.ascontiguousarray(final_observation[IMAGE_KEY], dtype=np.uint8),
        "final_wrist_image": np.ascontiguousarray(
            final_observation[WRIST_IMAGE_KEY], dtype=np.uint8
        ),
        "final_state": np.ascontiguousarray(final_observation[STATE_KEY], dtype=np.float32),
        "final_sim_state": np.ascontiguousarray(final_sim_state, dtype=np.float64),
    }


def run_rad_episode(
    config: RadEpisodeConfig,
    *,
    task_loader: Callable[[Any], Tuple[Any, Any, Any, str]] = _load_task,
    client_factory: Callable[..., Any] = PolicyClient,
    video_writer: Callable[[pathlib.Path, List[np.ndarray], int], None] = _write_video,
    bridge_source_identity: Optional[Mapping[str, Any]] = None,
) -> pathlib.Path:
    _validate_rad_episode_config(config)
    experiment_identity = _validate_experiment_identity(
        config.experiment_identity
        if config.experiment_identity is not None
        else collect_rad_experiment_identity(config, bridge_source_identity)
    )
    actual_candidate_count = 1 if config.selector == SELECTOR_K1 else RAD_POOL_SIZE
    max_replans = (config.max_steps + config.replan_steps - 1) // config.replan_steps
    noise_schedule = generate_noise_schedule(max_replans, RAD_POOL_SIZE, config.noise_seed)
    random_schedule = generate_random_selector_schedule(
        max_replans, RAD_POOL_SIZE, config.random_selector_seed
    )
    artifact_dir = _new_rad_artifact_dir(config)
    event_log = EventLog(artifact_dir / "events.jsonl")
    frames = []  # type: List[np.ndarray]
    environment = None
    client = None
    observation = None
    error = None  # type: Optional[BaseException]
    secondary_errors = []  # type: List[Tuple[str, BaseException]]
    started = time.perf_counter()
    summary = {
        "status": "failed",
        "success": False,
        "selector": config.selector,
        "candidate_count": actual_candidate_count,
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "noise_seed": config.noise_seed,
        "random_selector_seed": config.random_selector_seed,
        "action_steps": 0,
        "replan_count": 0,
        "model_inference_calls": 0,
    }  # type: Dict[str, Any]
    records = {
        key: []
        for key in (
            "images",
            "wrist_images",
            "states",
            "sim_states_before",
            "candidate_actions",
            "expert_ids",
            "expert_weights",
            "selected_indices",
            "selector_scores",
            "selector_score_mask",
            "selected_actions",
            "candidate_inference_ms",
            "executed_lengths",
            "replan_start_steps",
            "executed_actions",
            "rewards",
            "dones",
            "successes",
            "sim_states_after",
        )
    }
    layer_indices = None  # type: Optional[np.ndarray]
    policy_identity = None  # type: Optional[Dict[str, Any]]
    action_std = None  # type: Optional[np.ndarray]
    prompt = None  # type: Optional[str]
    task_name = None  # type: Optional[str]
    initial_runtime_sim_state_sha256 = None  # type: Optional[str]
    try:
        environment, observation, task, prompt = task_loader(config)
        task_name = str(task.name)
        initial_runtime_sim_state_sha256 = _sha256_array(_sim_state(environment))
        summary.update(
            {
                "prompt": prompt,
                "task_name": task_name,
                "experiment_identity": experiment_identity,
                "initial_runtime_sim_state_sha256": initial_runtime_sim_state_sha256,
            }
        )
        frames.append(frame_from_observation(observation))
        event_log.write("environment_ready", prompt=prompt, task_name=task_name)
        for settle_index in range(config.settle_steps):
            observation, reward, done, info = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frames.append(frame_from_observation(observation))
            event_log.write("settle_step", index=settle_index, reward=reward, done=done)

        client = client_factory(
            host=config.host,
            port=config.port,
            connect_timeout=120.0,
            inference_timeout=config.inference_timeout,
        )
        validate_policy_suite(client.metadata, config.task_suite)
        if not client.metadata.get("routing_capture_supported", False):
            raise RuntimeError("RAD requires routing capture support")
        action_std = _normalization_action_std(client.metadata)
        policy_identity = _rad_policy_identity(client.metadata)
        summary["server_metadata"] = client.metadata
        event_log.write("policy_connected", metadata=client.metadata)

        while summary["action_steps"] < config.max_steps and not summary["success"]:
            replan_index = summary["replan_count"]
            policy_observation = build_policy_observation(observation, prompt)
            sim_state_before = _sim_state(environment)
            candidate_actions = []
            candidate_expert_ids = []
            candidate_expert_weights = []
            candidate_inference_ms = []
            for candidate_index in range(actual_candidate_count):
                flow_noise = noise_schedule[replan_index, candidate_index]
                request = dict(policy_observation)
                request[FLOW_NOISE_KEY] = flow_noise
                request[ROUTING_CAPTURE_KEY] = True
                request_started = time.perf_counter()
                response = validate_action_response(client.infer(request))
                inference_ms = (time.perf_counter() - request_started) * 1000.0
                expected_noise_sha = _sha256_array(flow_noise)
                if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise_sha:
                    raise RuntimeError(
                        "Server did not acknowledge RAD noise at replan %d candidate %d"
                        % (replan_index, candidate_index)
                    )
                response_layers = response[ROUTING_LAYER_INDICES_KEY]
                if layer_indices is None:
                    layer_indices = np.array(response_layers, dtype=np.int16, copy=True)
                    if layer_indices.tolist() != policy_identity[
                        "routing_hb_layer_indices"
                    ]:
                        raise RuntimeError(
                            "Captured routing layers disagree with policy metadata"
                        )
                elif not np.array_equal(layer_indices, response_layers):
                    raise RuntimeError("RAD routing layer indices changed during the episode")
                candidate_actions.append(response[ACTION_KEY])
                candidate_expert_ids.append(response[ROUTING_EXPERT_IDS_KEY])
                candidate_expert_weights.append(response[ROUTING_EXPERT_WEIGHTS_KEY])
                candidate_inference_ms.append(inference_ms)
                summary["model_inference_calls"] += 1

            action_pool = _stack(candidate_actions, np.float32)
            expert_id_pool = _stack(candidate_expert_ids, np.int16)
            expert_weight_pool = _stack(candidate_expert_weights, np.float32)
            selection = select_candidate(
                config.selector,
                action_pool,
                action_std=action_std,
                expert_ids=expert_id_pool,
                expert_weights=expert_weight_pool,
                random_index=int(random_schedule[replan_index]),
            )
            score_values = np.zeros((actual_candidate_count,), dtype=np.float64)
            score_mask = np.zeros((actual_candidate_count,), dtype=np.bool_)
            if selection.scores is not None:
                score_values[:] = selection.scores
                score_mask[:] = True
            selected_actions = action_pool[selection.index]

            records["images"].append(policy_observation[IMAGE_KEY])
            records["wrist_images"].append(policy_observation[WRIST_IMAGE_KEY])
            records["states"].append(policy_observation[STATE_KEY])
            records["sim_states_before"].append(sim_state_before)
            records["candidate_actions"].append(action_pool)
            records["expert_ids"].append(expert_id_pool)
            records["expert_weights"].append(expert_weight_pool)
            records["selected_indices"].append(selection.index)
            records["selector_scores"].append(score_values)
            records["selector_score_mask"].append(score_mask)
            records["selected_actions"].append(selected_actions)
            records["candidate_inference_ms"].append(
                np.asarray(candidate_inference_ms, dtype=np.float64)
            )
            records["replan_start_steps"].append(summary["action_steps"])
            summary["replan_count"] += 1
            event_log.write(
                "rad_selection",
                replan_index=replan_index,
                selector=config.selector,
                selected_index=selection.index,
                metric=selection.metric,
                scores=selection.scores,
                observation={
                    "image_sha256": _sha256_array(policy_observation[IMAGE_KEY]),
                    "wrist_image_sha256": _sha256_array(
                        policy_observation[WRIST_IMAGE_KEY]
                    ),
                    "state_sha256": _sha256_array(policy_observation[STATE_KEY]),
                    "sim_state_sha256": _sha256_array(sim_state_before),
                },
                noise_bank_sha256=_sha256_array(
                    noise_schedule[replan_index, :actual_candidate_count]
                ),
                candidate_actions_sha256=_sha256_array(action_pool),
                expert_ids_sha256=_sha256_array(expert_id_pool),
                expert_weights_sha256=_sha256_array(expert_weight_pool),
            )

            executed_length = 0
            for chunk_index, action in enumerate(selected_actions[: config.replan_steps]):
                if summary["action_steps"] >= config.max_steps:
                    break
                observation, reward, done, info = environment.step(action.tolist())
                frames.append(frame_from_observation(observation))
                success = bool(environment.check_success())
                summary["action_steps"] += 1
                summary["success"] = success
                executed_length += 1
                records["executed_actions"].append(action)
                records["rewards"].append(float(reward))
                records["dones"].append(bool(done))
                records["successes"].append(success)
                records["sim_states_after"].append(_sim_state(environment))
                event_log.write(
                    "action_step",
                    step=summary["action_steps"],
                    replan_index=replan_index,
                    chunk_index=chunk_index,
                    selected_index=selection.index,
                    reward=reward,
                    done=done,
                    success=success,
                )
                if success:
                    break
            if executed_length < 1:
                raise RuntimeError("RAD replan executed no actions")
            records["executed_lengths"].append(executed_length)

        final_policy_observation = build_policy_observation(observation, prompt)
        final_sim_state = _sim_state(environment)
        if layer_indices is None or action_std is None or policy_identity is None:
            raise RuntimeError("RAD episode completed without routing/policy identity")
        trace_arrays = _rad_trace_arrays(
            records,
            noise_schedule,
            layer_indices,
            final_policy_observation,
            final_sim_state,
        )
        summary["status"] = "completed"
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
        try:
            event_log.write("rad_episode_error", error_type=type(caught).__name__, error=str(caught))
        except BaseException as log_error:
            secondary_errors.append(("error_log", log_error))
        trace_arrays = None
        final_policy_observation = None
        final_sim_state = None
    finally:
        for label, resource in (("client_close", client), ("environment_close", environment)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as cleanup_error:
                secondary_errors.append((label, cleanup_error))

        source_render = None
        if frames:
            try:
                source_render = _render_stream_metadata(frames)
            except BaseException as render_error:
                secondary_errors.append(("render_hash", render_error))
        video_path = artifact_dir / "episode.mp4"
        source_video = {"file": video_path.name, "encoded": False, "sha256": None}
        if frames:
            try:
                video_writer(video_path, frames, config.fps)
                if not video_path.is_file() or video_path.stat().st_size < 1:
                    raise RuntimeError("RAD video writer produced no file")
                source_video.update({"encoded": True, "sha256": sha256_file(video_path)})
                summary["video"] = str(video_path)
            except BaseException as video_error:
                secondary_errors.append(("video", video_error))

        if trace_arrays is not None:
            try:
                if source_render is None or not source_video["encoded"]:
                    raise RuntimeError("A complete RAD trace requires render/video evidence")
                trace_metadata = {
                    "experiment_schema": RAD_EXPERIMENT_SCHEMA,
                    "config": {
                        "selector": config.selector,
                        "candidate_count": actual_candidate_count,
                        "master_candidate_count": RAD_POOL_SIZE,
                        "task_suite": config.task_suite,
                        "task_id": config.task_id,
                        "init_state_id": config.init_state_id,
                        "seed": config.seed,
                        "settle_steps": config.settle_steps,
                        "max_steps": config.max_steps,
                        "replan_steps": config.replan_steps,
                        "max_replans": max_replans,
                        "render_size": config.render_size,
                        "fps": config.fps,
                        "noise_seed": config.noise_seed,
                        "random_selector_seed": config.random_selector_seed,
                    },
                    "metric_schema": routing_metric_contract(
                        posthoc_selection_allowed=False
                    ),
                    "normalization_action_std": action_std.tolist(),
                    "policy_identity": policy_identity,
                    "experiment_identity": experiment_identity,
                    "experiment_identity_sha256": experiment_identity[
                        "identity_sha256"
                    ],
                    "routing_layer_indices": layer_indices.tolist(),
                    "source_server": {
                        "instance_id": summary["server_metadata"].get("server_instance_id"),
                        "pid": summary["server_metadata"].get("server_pid"),
                        "started_unix_ns": summary["server_metadata"].get(
                            "server_started_unix_ns"
                        ),
                    },
                    "prompt": prompt,
                    "task_name": task_name,
                    "initial_runtime_sim_state_sha256": initial_runtime_sim_state_sha256,
                    "result": {
                        "status": "completed",
                        "success": summary["success"],
                        "action_steps": summary["action_steps"],
                        "replan_count": summary["replan_count"],
                        "model_inference_calls": summary["model_inference_calls"],
                    },
                    "timing": {
                        "candidate_inference_ms_total": float(
                            np.sum(trace_arrays["candidate_inference_ms"])
                        ),
                        "candidate_inference_ms_mean": float(
                            np.mean(trace_arrays["candidate_inference_ms"])
                        ),
                    },
                    "source_render": source_render,
                    "source_video": source_video,
                }
                manifest = save_rad_trace(artifact_dir, trace_metadata, trace_arrays)
                summary["rad_trace"] = str(artifact_dir / RAD_TRACE_MANIFEST_FILE)
                summary["rad_trace_array_sha256"] = manifest["array_file_sha256"]
                summary["selected_indices"] = trace_arrays["selected_indices"]
                summary["executed_lengths"] = trace_arrays["executed_lengths"]
                summary["fairness_identity"] = _fairness_identity_from_trace(
                    trace_metadata, trace_arrays
                )
            except BaseException as trace_error:
                secondary_errors.append(("trace", trace_error))

        try:
            event_log.close()
        except BaseException as close_error:
            secondary_errors.append(("event_log_close", close_error))
        if secondary_errors:
            summary["secondary_errors"] = [
                {"stage": label, "error_type": type(value).__name__, "error": str(value)}
                for label, value in secondary_errors
            ]
            if error is None:
                error = secondary_errors[0][1]
        if error is not None:
            summary["status"] = "failed"
            summary.setdefault("error_type", type(error).__name__)
            summary.setdefault("error", str(error))
        summary["duration_seconds"] = time.perf_counter() - started
        _atomic_json(artifact_dir / "summary.json", summary)

    if error is not None:
        raise RadEpisodeRunError(artifact_dir, error) from error
    logging.info("RAD episode artifacts: %s", artifact_dir)
    return artifact_dir


def _rad_arm_order(seed: int) -> Tuple[str, ...]:
    if seed < 0:
        raise ValueError("arm_order_seed must be non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _ARM_ORDER_STREAM_TAG]))
    order = list(PAIRED_SELECTORS)
    rng.shuffle(order)
    return tuple(order)


def _block_config_dict(config: RadBlockConfig) -> Dict[str, Any]:
    return dataclasses.asdict(config)


def _first_mismatch(expected: Any, actual: Any, path: str = "root") -> Optional[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return "%s.keys expected=%s actual=%s" % (
                path,
                sorted(expected_keys),
                sorted(actual_keys),
            )
        for key in sorted(expected_keys):
            mismatch = _first_mismatch(expected[key], actual[key], "%s.%s" % (path, key))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return "%s.length expected=%d actual=%d" % (path, len(expected), len(actual))
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatch = _first_mismatch(
                expected_item, actual_item, "%s[%d]" % (path, index)
            )
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return "%s expected=%r actual=%r" % (path, expected, actual)
    return None


def _require_equal(expected: Any, actual: Any, label: str) -> None:
    mismatch = _first_mismatch(expected, actual, label)
    if mismatch is not None:
        raise RuntimeError("RAD immutable identity mismatch: %s" % mismatch)


@contextlib.contextmanager
def _rad_block_lock(block_dir: pathlib.Path) -> Iterable[None]:
    lock_path = block_dir / ".rad-block.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("RAD block is already active: %s" % block_dir) from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_utc": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _block_immutable(
    config: RadBlockConfig,
    order: Sequence[str],
    experiment_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    config_value = _block_config_dict(config)
    value = {
        "schema": RAD_BLOCK_SCHEMA,
        "config": config_value,
        "config_sha256": _canonical_json_sha256(config_value),
        "arm_order": list(order),
        "experiment_identity": dict(experiment_identity),
        "experiment_identity_sha256": experiment_identity["identity_sha256"],
    }
    value["immutable_sha256"] = _canonical_json_sha256(value)
    return value


def _expected_trace_config(config: RadBlockConfig, selector: str) -> Dict[str, Any]:
    return {
        "selector": selector,
        "candidate_count": 1 if selector == SELECTOR_K1 else RAD_POOL_SIZE,
        "master_candidate_count": RAD_POOL_SIZE,
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "settle_steps": config.settle_steps,
        "max_steps": config.max_steps,
        "replan_steps": config.replan_steps,
        "max_replans": (config.max_steps + config.replan_steps - 1)
        // config.replan_steps,
        "render_size": config.render_size,
        "fps": config.fps,
        "noise_seed": config.noise_seed,
        "random_selector_seed": config.random_selector_seed,
    }


def _fairness_identity_from_trace(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Dict[str, Any]:
    shared = {
        "initial_image_sha256": _sha256_array(arrays["images"][0]),
        "initial_wrist_image_sha256": _sha256_array(arrays["wrist_images"][0]),
        "initial_state_sha256": _sha256_array(arrays["states"][0]),
        "initial_sim_state_sha256": _sha256_array(arrays["sim_states_before"][0]),
        "initial_runtime_sim_state_sha256": metadata[
            "initial_runtime_sim_state_sha256"
        ],
        "master_noise_schedule_sha256": _sha256_array(arrays["noise_schedule"]),
        "first_candidate0_actions_sha256": _sha256_array(
            arrays["candidate_actions"][0, 0]
        ),
        "first_candidate0_expert_ids_sha256": _sha256_array(
            arrays["expert_ids"][0, 0]
        ),
        "first_candidate0_expert_weights_sha256": _sha256_array(
            arrays["expert_weights"][0, 0]
        ),
        "normalization_action_std": metadata["normalization_action_std"],
        "policy_identity": metadata["policy_identity"],
        "experiment_identity": metadata["experiment_identity"],
        "experiment_identity_sha256": metadata["experiment_identity_sha256"],
        "routing_layer_indices": metadata["routing_layer_indices"],
        "metric_schema": metadata["metric_schema"],
    }
    selector = metadata["config"]["selector"]
    initial_k8_candidate_pool = None
    if selector != SELECTOR_K1:
        if arrays["candidate_actions"].shape[1] != RAD_POOL_SIZE:
            raise RuntimeError("K=8 RAD arm is missing its complete candidate pool")
        initial_k8_candidate_pool = {
            "candidate_count": RAD_POOL_SIZE,
            "actions_sha256": _sha256_array(arrays["candidate_actions"][0]),
            "expert_ids_sha256": _sha256_array(arrays["expert_ids"][0]),
            "expert_weights_sha256": _sha256_array(arrays["expert_weights"][0]),
        }
    return {
        "shared_all_arms": shared,
        "initial_k8_candidate_pool": initial_k8_candidate_pool,
    }


def _validate_completed_arm(
    config: RadBlockConfig,
    selector: str,
    artifact_dir: pathlib.Path,
    expected_identity: Mapping[str, Any],
    trace_loader: Callable[
        [pathlib.Path], Tuple[Dict[str, Any], Dict[str, np.ndarray]]
    ],
) -> Dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("selector") != selector:
        raise RuntimeError("RAD arm did not produce a valid completed summary")
    trace_path = pathlib.Path(str(summary.get("rad_trace", "")))
    trace_manifest, trace_arrays = trace_loader(trace_path)
    metadata = trace_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("RAD arm trace metadata is missing")
    expected_trace_config = _expected_trace_config(config, selector)
    actual_trace_config = metadata.get("config")
    if not isinstance(actual_trace_config, Mapping):
        raise RuntimeError("RAD arm trace config is missing")
    for key, expected in expected_trace_config.items():
        _require_equal(expected, actual_trace_config.get(key), "trace.config.%s" % key)
    _require_equal(
        expected_identity,
        metadata.get("experiment_identity"),
        "trace.experiment_identity",
    )
    _require_equal(
        expected_identity["identity_sha256"],
        metadata.get("experiment_identity_sha256"),
        "trace.experiment_identity_sha256",
    )
    policy_identity = metadata.get("policy_identity")
    if not isinstance(policy_identity, Mapping):
        raise RuntimeError("RAD arm trace policy identity is missing")
    routing_layers = metadata.get("routing_layer_indices")
    _require_equal(
        policy_identity.get("routing_hb_layer_indices"),
        routing_layers,
        "trace.routing_layer_indices",
    )
    _require_equal(
        routing_layers,
        trace_arrays["layer_indices"].astype(int).tolist(),
        "trace.array.layer_indices",
    )
    if str(policy_identity.get("backend", "")).startswith("himoe-vla-"):
        _require_equal(
            expected_identity["himoe"]["upstream_commit"],
            policy_identity.get("himoe_upstream_commit"),
            "trace.policy.himoe_upstream_commit",
        )
        _require_equal(
            expected_identity["himoe"]["patch_sha256"],
            policy_identity.get("himoe_patch_sha256"),
            "trace.policy.himoe_patch_sha256",
        )
    if not isinstance(metadata.get("result", {}).get("success"), bool):
        raise RuntimeError("RAD arm trace success outcome must be boolean")
    _require_equal(
        summary.get("success"),
        metadata["result"]["success"],
        "trace.result.success",
    )
    fairness_identity = _fairness_identity_from_trace(metadata, trace_arrays)
    _require_equal(
        fairness_identity,
        summary.get("fairness_identity"),
        "summary.fairness_identity",
    )
    return {
        "summary": summary,
        "fairness_identity": fairness_identity,
        "verification": {
            "summary_sha256": sha256_file(summary_path),
            "trace_manifest_sha256": sha256_file(
                trace_path / RAD_TRACE_MANIFEST_FILE
                if trace_path.is_dir()
                else trace_path
            ),
            "trace_array_sha256": trace_manifest["array_file_sha256"],
            "trace_metadata_sha256": trace_manifest["metadata_sha256"],
            "video_sha256": metadata["source_video"]["sha256"],
        },
    }


def _episode_from_block(
    config: RadBlockConfig, selector: str, experiment_identity: Mapping[str, Any]
) -> RadEpisodeConfig:
    return RadEpisodeConfig(
        libero_root=config.libero_root,
        output_root=str(pathlib.Path(config.block_dir).expanduser().resolve() / "arms"),
        host=config.host,
        port=config.port,
        task_suite=config.task_suite,
        task_id=config.task_id,
        init_state_id=config.init_state_id,
        seed=config.seed,
        settle_steps=config.settle_steps,
        max_steps=config.max_steps,
        replan_steps=config.replan_steps,
        render_size=config.render_size,
        fps=config.fps,
        inference_timeout=config.inference_timeout,
        selector=selector,
        noise_seed=config.noise_seed,
        random_selector_seed=config.random_selector_seed,
        experiment_identity=experiment_identity,
    )


def _run_rad_block_locked(
    config: RadBlockConfig,
    *,
    episode_runner: Callable[..., pathlib.Path] = run_rad_episode,
    trace_loader: Callable[[pathlib.Path], Tuple[Dict[str, Any], Dict[str, np.ndarray]]] = load_rad_trace,
    experiment_identity_provider: Callable[[Any], Mapping[str, Any]],
) -> pathlib.Path:
    root = pathlib.Path(config.block_dir).expanduser().resolve()
    (root / "arms").mkdir(parents=True, exist_ok=True)
    order = _rad_arm_order(config.arm_order_seed)
    experiment_identity = _validate_experiment_identity(
        experiment_identity_provider(config)
    )

    def require_current_experiment_identity() -> None:
        current_identity = _validate_experiment_identity(
            experiment_identity_provider(config)
        )
        _require_equal(
            experiment_identity, current_identity, "current_experiment_identity"
        )

    immutable = _block_immutable(config, order, experiment_identity)
    config_path = root / RAD_BLOCK_CONFIG_FILE
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        _require_equal(immutable, existing, "block_config")
    else:
        _atomic_json(config_path, immutable)

    manifest_path = root / RAD_BLOCK_MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_equal(RAD_BLOCK_SCHEMA, manifest.get("schema"), "manifest.schema")
        _require_equal(list(order), manifest.get("arm_order"), "manifest.arm_order")
        _require_equal(
            immutable["immutable_sha256"],
            manifest.get("immutable_sha256"),
            "manifest.immutable_sha256",
        )
    else:
        manifest = {
            "schema": RAD_BLOCK_SCHEMA,
            "status": "running",
            "arm_order": list(order),
            "immutable_sha256": immutable["immutable_sha256"],
            "config_sha256": immutable["config_sha256"],
            "experiment_identity_sha256": immutable[
                "experiment_identity_sha256"
            ],
            "arms": {selector: {"status": "pending"} for selector in PAIRED_SELECTORS},
        }
        _atomic_json(manifest_path, manifest)

    for selector in order:
        require_current_experiment_identity()
        arm = manifest["arms"][selector]
        if arm.get("status") == "completed":
            artifact = pathlib.Path(str(arm.get("artifact_dir", "")))
            validation = _validate_completed_arm(
                config, selector, artifact, experiment_identity, trace_loader
            )
            _require_equal(
                arm.get("verification"),
                validation["verification"],
                "manifest.arms.%s.verification" % selector,
            )
            _require_equal(
                arm.get("fairness_identity"),
                validation["fairness_identity"],
                "manifest.arms.%s.fairness_identity" % selector,
            )
            require_current_experiment_identity()
            continue
        for stale_field in (
            "artifact_dir",
            "error_type",
            "error",
            "success",
            "trace",
            "fairness_identity",
            "verification",
            "completed_utc",
        ):
            arm.pop(stale_field, None)
        arm.update(
            {
                "status": "running",
                "started_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        try:
            artifact_dir = episode_runner(
                _episode_from_block(config, selector, experiment_identity)
            )
            validation = _validate_completed_arm(
                config, selector, artifact_dir, experiment_identity, trace_loader
            )
        except RadEpisodeRunError as error:
            arm.update(
                {
                    "status": "failed",
                    "artifact_dir": str(error.artifact_dir),
                    "error_type": type(error.cause).__name__,
                    "error": str(error.cause),
                }
            )
            manifest["status"] = "failed"
            _atomic_json(manifest_path, manifest)
            raise
        except BaseException as error:
            arm.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            manifest["status"] = "failed"
            _atomic_json(manifest_path, manifest)
            raise
        summary = validation["summary"]
        arm.update(
            {
                "status": "completed",
                "artifact_dir": str(artifact_dir),
                "success": summary["success"],
                "trace": summary.get("rad_trace"),
                "fairness_identity": validation["fairness_identity"],
                "verification": validation["verification"],
                "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        require_current_experiment_identity()

    require_current_experiment_identity()
    k1_fairness = manifest["arms"][SELECTOR_K1].get("fairness_identity")
    if not isinstance(k1_fairness, Mapping):
        raise RuntimeError("RAD K=1 arm fairness identity is missing")
    shared_reference = k1_fairness.get("shared_all_arms")
    shared_mismatches = [
        selector
        for selector in PAIRED_SELECTORS[1:]
        if manifest["arms"][selector]
        .get("fairness_identity", {})
        .get("shared_all_arms")
        != shared_reference
    ]
    k8_reference = manifest["arms"][SELECTOR_RANDOM].get("fairness_identity", {}).get(
        "initial_k8_candidate_pool"
    )
    k8_mismatches = [
        selector
        for selector in (SELECTOR_ACTION_MEDOID, SELECTOR_ROUTING_MEDOID)
        if manifest["arms"][selector]
        .get("fairness_identity", {})
        .get("initial_k8_candidate_pool")
        != k8_reference
    ]
    if k1_fairness.get("initial_k8_candidate_pool") is not None:
        shared_mismatches.append(SELECTOR_K1)
    if not isinstance(k8_reference, Mapping):
        k8_mismatches.append(SELECTOR_RANDOM)
    fairness_mismatches = {
        "shared_all_arms": shared_mismatches,
        "initial_k8_candidate_pool": k8_mismatches,
    }
    if shared_mismatches or k8_mismatches:
        manifest["status"] = "failed"
        manifest["fairness_mismatches"] = fairness_mismatches
        _atomic_json(manifest_path, manifest)
        raise RuntimeError(
            "RAD paired arms failed exact fairness checks: shared=%s k8_pool=%s"
            % (shared_mismatches, k8_mismatches)
        )

    fairness_reference = {
        "shared_all_arms": shared_reference,
        "initial_k8_candidate_pool": k8_reference,
    }

    outcomes = {
        selector: manifest["arms"][selector]["success"]
        for selector in PAIRED_SELECTORS
    }
    summary = {
        "schema": RAD_BLOCK_SCHEMA,
        "status": "completed",
        "scope": "single_case_diagnostic",
        "diagnostic_only": True,
        "inferential_statistics": None,
        "case_id": "%s/task%d/init%d/env%d/noise%d"
        % (
            config.task_suite,
            config.task_id,
            config.init_state_id,
            config.seed,
            config.noise_seed,
        ),
        "arm_order": list(order),
        "fairness_identity": fairness_reference,
        "fairness_exact": True,
        "raw_outcomes": outcomes,
        "immutable_sha256": immutable["immutable_sha256"],
        "experiment_identity": experiment_identity,
        "experiment_identity_sha256": experiment_identity["identity_sha256"],
        "arms": manifest["arms"],
    }
    _atomic_json(root / RAD_BLOCK_SUMMARY_FILE, summary)
    manifest["status"] = "completed"
    manifest["summary"] = str(root / RAD_BLOCK_SUMMARY_FILE)
    _atomic_json(manifest_path, manifest)
    require_current_experiment_identity()
    return root


def run_rad_block(
    config: RadBlockConfig,
    *,
    episode_runner: Callable[..., pathlib.Path] = run_rad_episode,
    trace_loader: Callable[
        [pathlib.Path], Tuple[Dict[str, Any], Dict[str, np.ndarray]]
    ] = load_rad_trace,
    experiment_identity_provider: Optional[
        Callable[[Any], Mapping[str, Any]]
    ] = None,
) -> pathlib.Path:
    root = pathlib.Path(config.block_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    provider = experiment_identity_provider or collect_rad_experiment_identity
    with _rad_block_lock(root):
        return _run_rad_block_locked(
            config,
            episode_runner=episode_runner,
            trace_loader=trace_loader,
            experiment_identity_provider=provider,
        )
