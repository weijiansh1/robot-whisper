"""Single-task LIBERO runtime and artifact recording."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import (
    EpisodeTraceRecorder,
    load_episode_trace,
    save_episode_trace,
    sha256_file,
)
from himoe_libero_bridge.preprocess import build_policy_observation, frame_from_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
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
from himoe_libero_bridge.routing import analyze_candidate_pool

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


@dataclasses.dataclass
class EpisodeConfig:
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
    render_size: int = 512
    fps: int = 20
    inference_timeout: float = 180.0
    flow_noise_seed: Optional[int] = None


class EventLog:
    def __init__(self, path: pathlib.Path) -> None:
        self._stream = path.open("w", encoding="utf-8")

    def write(self, event: str, **values: Any) -> None:
        record = {"event": event, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record.update(_jsonable(values))
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _configure_libero(libero_root: pathlib.Path) -> None:
    benchmark_root = libero_root / "libero" / "libero"
    config_root = pathlib.Path(
        os.environ.get("LIBERO_CONFIG_PATH", "/home/jovyan/.cache/himoe-libero-bridge/libero-config")
    )
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(libero_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    # Written atomically and only when it changes: many clients start concurrently and LIBERO reads this
    # file at import time, so a plain truncating write lets a neighbour read an empty file.
    target = config_root / "config.yaml"
    text = json.dumps(config, indent=2)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        tmp = target.with_name("config.yaml.%d.tmp" % os.getpid())
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _load_task(config: EpisodeConfig) -> Tuple[Any, Any, Any, str]:
    libero_root = pathlib.Path(config.libero_root).expanduser().resolve()
    if not (libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError("Invalid LIBERO root: %s" % libero_root)
    _configure_libero(libero_root)
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if config.task_suite not in benchmark_dict:
        raise ValueError("Unknown task suite %s; available: %s" % (config.task_suite, sorted(benchmark_dict)))
    suite = benchmark_dict[config.task_suite]()
    if not 0 <= config.task_id < suite.n_tasks:
        raise ValueError("task_id %d is outside [0, %d)" % (config.task_id, suite.n_tasks))
    task = suite.get_task(config.task_id)
    initial_states = suite.get_task_init_states(config.task_id)
    if not 0 <= config.init_state_id < len(initial_states):
        raise ValueError("init_state_id %d is outside [0, %d)" % (config.init_state_id, len(initial_states)))

    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=config.render_size,
        camera_widths=config.render_size,
        horizon=config.max_steps + config.settle_steps + 1,
    )
    environment.seed(config.seed)
    environment.reset()
    observation = environment.set_init_state(initial_states[config.init_state_id])
    return environment, observation, task, str(task.language)


def _new_artifact_dir(root: str, prefix: str, suite: str, task_id: int, seed: int) -> pathlib.Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_slug = suite.replace("_", "-")
    directory = pathlib.Path(root).expanduser().resolve() / (
        "%s-%s-task%02d-seed%d-%s" % (prefix, suite_slug, task_id, seed, timestamp)
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_video(path: pathlib.Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames were recorded")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Video writer did not produce a non-empty file: %s" % path)


def _action_stats(actions: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min": float(np.min(actions)),
        "max": float(np.max(actions)),
        "mean": float(np.mean(actions)),
        "std": float(np.std(actions)),
    }


def validate_policy_suite(metadata: Dict[str, Any], expected_benchmark: str) -> None:
    actual_benchmark = metadata.get("benchmark")
    # Pro perturbations use the weights and normalization of their base suite.
    pro_base_suites = {
        "%s_%s" % (base, perturbation): base
        for base in ("libero_10", "libero_goal", "libero_object", "libero_spatial")
        for perturbation in ("swap", "object", "lan", "task", "env")
    }
    compatible_benchmarks = (
        expected_benchmark,
        pro_base_suites.get(expected_benchmark, expected_benchmark),
    )
    if actual_benchmark not in compatible_benchmarks:
        raise RuntimeError(
            "Policy/environment suite mismatch: environment=%s, server=%r"
            % (expected_benchmark, actual_benchmark)
        )


_COMMON_POLICY_IDENTITY_KEYS = (
    "protocol",
    "protocol_version",
    "backend",
    "benchmark",
    "flow_steps",
    "predicted_action_steps",
    "internal_action_dim",
)
_HIMOE_POLICY_IDENTITY_KEYS = (
    "checkpoint_sha256",
    "normalization_stats_sha256",
    "train_config",
    "dataset_config",
    "normalization_asset",
)


def _stable_policy_identity(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    backend = metadata.get("backend")
    keys = list(_COMMON_POLICY_IDENTITY_KEYS)
    if isinstance(backend, str) and backend.startswith("himoe-vla-"):
        keys.extend(_HIMOE_POLICY_IDENTITY_KEYS)
    elif backend != "mock":
        raise RuntimeError("Unsupported policy backend identity: %r" % backend)
    missing = [key for key in keys if metadata.get(key) is None]
    if missing:
        raise RuntimeError(
            "Policy metadata is missing stable identity fields: %s" % ", ".join(missing)
        )
    return {key: metadata[key] for key in keys}


def _render_stream_metadata(frames: List[np.ndarray]) -> Dict[str, Any]:
    if not frames:
        raise RuntimeError("Cannot describe an empty render stream")
    canonical = [np.ascontiguousarray(frame, dtype=np.uint8) for frame in frames]
    shape = canonical[0].shape
    if any(frame.shape != shape for frame in canonical):
        raise RuntimeError("Render stream contains inconsistent frame shapes")
    stream_digest = hashlib.sha256()
    frame_digests = []
    for frame in canonical:
        payload = frame.tobytes()
        stream_digest.update(payload)
        frame_digests.append(hashlib.sha256(payload).hexdigest())
    return {
        "frame_count": len(canonical),
        "frame_shape": list(shape),
        "frame_dtype": "uint8",
        "frame_sha256": frame_digests,
        "stream_sha256": stream_digest.hexdigest(),
    }


def _validate_replay_manifest(
    manifest: Mapping[str, Any], trace: Mapping[str, np.ndarray]
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    config = manifest.get("config")
    result = manifest.get("result")
    source_render = manifest.get("source_render")
    if not isinstance(config, Mapping) or not isinstance(result, Mapping):
        raise RuntimeError("Episode trace manifest is missing config/result metadata")
    if not isinstance(source_render, Mapping):
        raise RuntimeError("Episode trace manifest is missing source_render metadata")
    required_config = (
        "task_suite",
        "task_id",
        "init_state_id",
        "seed",
        "settle_steps",
        "max_steps",
        "replan_steps",
        "render_size",
        "fps",
        "flow_noise_seed",
    )
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise RuntimeError(
            "Episode trace config is missing: %s" % ", ".join(missing_config)
        )
    if int(config["flow_noise_seed"]) < 0:
        raise RuntimeError("Episode trace flow_noise_seed must be non-negative")
    if int(config["settle_steps"]) < 0 or int(config["max_steps"]) < 1:
        raise RuntimeError("Episode trace contains invalid step limits")
    if int(config["render_size"]) < 1 or int(config["fps"]) < 1:
        raise RuntimeError("Episode trace contains invalid render settings")

    action_steps = int(np.sum(trace["executed_lengths"], dtype=np.int64))
    replan_count = int(trace["executed_lengths"].shape[0])
    if result.get("status") != "completed" or result.get("trace_complete") is not True:
        raise RuntimeError("Only a completed episode trace can be replayed")
    if int(result.get("action_steps", -1)) != action_steps:
        raise RuntimeError("Trace result action_steps does not match the trace arrays")
    if int(result.get("inference_calls", -1)) != replan_count:
        raise RuntimeError("Trace result inference_calls does not match the trace arrays")
    if bool(result.get("success")) != bool(trace["successes"][-1]):
        raise RuntimeError("Trace result success does not match the trace arrays")
    if not isinstance(manifest.get("prompt"), str) or not manifest["prompt"].strip():
        raise RuntimeError("Episode trace prompt is missing")
    if not isinstance(manifest.get("task_name"), str) or not manifest["task_name"].strip():
        raise RuntimeError("Episode trace task_name is missing")

    expected_frame_count = 1 + int(config["settle_steps"]) + action_steps
    frame_hashes = source_render.get("frame_sha256")
    expected_frame_shape = [int(config["render_size"]), int(config["render_size"]), 3]
    if (
        source_render.get("frame_count") != expected_frame_count
        or source_render.get("frame_shape") != expected_frame_shape
        or source_render.get("frame_dtype") != "uint8"
        or not isinstance(frame_hashes, list)
        or len(frame_hashes) != expected_frame_count
    ):
        raise RuntimeError("Episode trace source_render metadata is inconsistent")
    digests = list(frame_hashes) + [source_render.get("stream_sha256")]
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise RuntimeError("Episode trace source_render contains an invalid SHA-256")
    policy_identity = manifest.get("policy_identity")
    if not isinstance(policy_identity, Mapping):
        raise RuntimeError("Episode trace policy identity is invalid")
    _stable_policy_identity(policy_identity)
    source_server = manifest.get("source_server")
    if source_server is not None:
        if not isinstance(source_server, Mapping):
            raise RuntimeError("Episode trace source_server metadata is invalid")
        instance_id = source_server.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or len(instance_id) != 32
            or any(character not in "0123456789abcdef" for character in instance_id)
            or not isinstance(source_server.get("pid"), int)
            or source_server["pid"] < 1
            or not isinstance(source_server.get("started_unix_ns"), int)
            or source_server["started_unix_ns"] < 1
        ):
            raise RuntimeError("Episode trace source_server identity is invalid")
    source_video = manifest.get("source_video")
    if not isinstance(source_video, Mapping):
        raise RuntimeError("Episode trace source_video metadata is invalid")
    if source_video.get("frames") != expected_frame_count:
        raise RuntimeError("Episode trace source_video frame count is inconsistent")
    if source_video.get("encoded") is True:
        video_digest = source_video.get("sha256")
        if (
            not isinstance(video_digest, str)
            or len(video_digest) != 64
            or any(character not in "0123456789abcdef" for character in video_digest)
        ):
            raise RuntimeError("Episode trace source_video SHA-256 is invalid")
    return config, result, source_render


def _sim_state(environment: Any) -> np.ndarray:
    return np.ascontiguousarray(environment.get_sim_state(), dtype=np.float64)


def _assert_exact_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if np.array_equal(actual_array, expected_array):
        return
    detail = ""
    if (
        actual_array.shape == expected_array.shape
        and np.issubdtype(actual_array.dtype, np.number)
        and np.issubdtype(expected_array.dtype, np.number)
    ):
        detail = "; max_abs_diff=%.9g" % float(
            np.max(np.abs(actual_array.astype(np.float64) - expected_array.astype(np.float64)))
        )
    raise RuntimeError(
        "%s differs: actual shape/dtype=%s/%s, expected=%s/%s%s"
        % (
            label,
            actual_array.shape,
            actual_array.dtype,
            expected_array.shape,
            expected_array.dtype,
            detail,
        )
    )


def run_episode(config: EpisodeConfig) -> pathlib.Path:
    if not 1 <= config.replan_steps <= FLOW_NOISE_SHAPE[0]:
        raise ValueError("replan_steps must be between 1 and %d" % FLOW_NOISE_SHAPE[0])
    if config.flow_noise_seed is not None and config.flow_noise_seed < 0:
        raise ValueError("flow_noise_seed must be non-negative")
    trace_recorder = (
        EpisodeTraceRecorder(config.replan_steps)
        if config.flow_noise_seed is not None
        else None
    )
    trace_rng = (
        np.random.default_rng(config.flow_noise_seed)
        if config.flow_noise_seed is not None
        else None
    )
    artifact_dir = _new_artifact_dir(
        config.output_root, "episode", config.task_suite, config.task_id, config.seed
    )
    event_log = EventLog(artifact_dir / "events.jsonl")
    frames = []  # type: List[np.ndarray]
    environment = None
    observation = None
    client = None
    trace_arrays = None  # type: Optional[Dict[str, np.ndarray]]
    trace_complete = False
    summary = {
        "status": "failed",
        "success": False,
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "action_steps": 0,
        "inference_calls": 0,
        "executed_action_steps_per_replan": config.replan_steps,
        "flow_noise_seed": config.flow_noise_seed,
        "episode_trace_recording": trace_recorder is not None,
    }  # type: Dict[str, Any]
    error = None  # type: Optional[BaseException]
    secondary_errors = []  # type: List[Tuple[str, BaseException]]
    started = time.perf_counter()
    try:
        environment, observation, task, prompt = _load_task(config)
        summary["task_name"] = str(task.name)
        summary["prompt"] = prompt
        frames.append(frame_from_observation(observation))
        event_log.write("environment_ready", observation_keys=sorted(observation), prompt=prompt)

        for settle_index in range(config.settle_steps):
            observation, reward, done, info = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frames.append(frame_from_observation(observation))
            event_log.write("settle_step", index=settle_index, reward=reward, done=done)

        client = PolicyClient(
            host=config.host,
            port=config.port,
            connect_timeout=120.0,
            inference_timeout=config.inference_timeout,
        )
        summary["server_metadata"] = client.metadata
        validate_policy_suite(client.metadata, config.task_suite)
        summary["flow_steps"] = client.metadata.get("flow_steps")
        summary["predicted_action_steps"] = client.metadata.get("predicted_action_steps")
        event_log.write("policy_connected", metadata=client.metadata)

        while summary["action_steps"] < config.max_steps and not summary["success"]:
            policy_observation = build_policy_observation(observation, prompt)
            request = dict(policy_observation)
            flow_noise = None  # type: Optional[np.ndarray]
            if trace_rng is not None:
                flow_noise = trace_rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
                request[FLOW_NOISE_KEY] = flow_noise
            request_started = time.perf_counter()
            response = validate_action_response(client.infer(request))
            latency_ms = (time.perf_counter() - request_started) * 1000.0
            actions = response[ACTION_KEY]
            if trace_recorder is not None:
                if flow_noise is None:
                    raise RuntimeError("Trace recording requires explicit flow noise")
                expected_noise_sha256 = _sha256_array(flow_noise)
                if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise_sha256:
                    raise RuntimeError(
                        "Server did not acknowledge episode flow noise at replan %d"
                        % summary["inference_calls"]
                    )
                trace_recorder.start_replan(
                    policy_observation,
                    flow_noise,
                    actions,
                    _sim_state(environment),
                )
            summary["inference_calls"] += 1
            event_log.write(
                "inference",
                index=summary["inference_calls"] - 1,
                latency_ms=latency_ms,
                image_shape=list(policy_observation["observation/image"].shape),
                wrist_image_shape=list(policy_observation["observation/wrist_image"].shape),
                state=policy_observation["observation/state"],
                action_stats=_action_stats(actions),
                image_sha256=_sha256_array(policy_observation[IMAGE_KEY]),
                wrist_image_sha256=_sha256_array(policy_observation[WRIST_IMAGE_KEY]),
                state_sha256=_sha256_array(policy_observation[STATE_KEY]),
                flow_noise_sha256=(
                    _sha256_array(flow_noise) if flow_noise is not None else None
                ),
                action_chunk_sha256=_sha256_array(actions),
            )

            for chunk_index, action in enumerate(actions[: config.replan_steps]):
                if summary["action_steps"] >= config.max_steps:
                    break
                observation, reward, done, info = environment.step(action.tolist())
                frames.append(frame_from_observation(observation))
                summary["action_steps"] += 1
                success = bool(environment.check_success())
                summary["success"] = success
                if trace_recorder is not None:
                    trace_recorder.record_action(
                        action,
                        reward,
                        done,
                        success,
                        _sim_state(environment),
                    )
                event_log.write(
                    "action_step",
                    step=summary["action_steps"],
                    chunk_index=chunk_index,
                    action=action,
                    reward=reward,
                    done=done,
                    success=success,
                )
                if success:
                    break

        if trace_recorder is not None:
            final_policy_observation = build_policy_observation(observation, prompt)
            trace_recorder.finish(final_policy_observation, _sim_state(environment))
            trace_arrays = trace_recorder.arrays()
            trace_complete = True
        summary["status"] = "completed"
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
        try:
            event_log.write(
                "episode_error", error_type=type(caught).__name__, error=str(caught)
            )
        except BaseException as log_error:
            secondary_errors.append(("episode_error_log", log_error))
    finally:
        if (
            trace_recorder is not None
            and trace_arrays is None
            and trace_recorder.action_count > 0
            and observation is not None
            and environment is not None
        ):
            try:
                final_policy_observation = build_policy_observation(
                    observation, summary["prompt"]
                )
                trace_recorder.finish(final_policy_observation, _sim_state(environment))
                trace_arrays = trace_recorder.arrays()
            except BaseException as partial_trace_error:
                secondary_errors.append(("partial_trace_finalize", partial_trace_error))
        for cleanup_label, resource in (("client_close", client), ("environment_close", environment)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as cleanup_error:
                secondary_errors.append((cleanup_label, cleanup_error))

        source_render = None
        if frames:
            try:
                source_render = _render_stream_metadata(frames)
                summary["render_stream_sha256"] = source_render["stream_sha256"]
            except BaseException as render_error:
                secondary_errors.append(("render_stream_hash", render_error))

        source_video = {
            "file": "episode.mp4",
            "frames": len(frames),
            "encoded": False,
            "sha256": None,
        }  # type: Dict[str, Any]
        try:
            _write_video(artifact_dir / "episode.mp4", frames, config.fps)
            summary["video"] = str(artifact_dir / "episode.mp4")
            summary["video_frames"] = len(frames)
            source_video.update(
                {
                    "encoded": True,
                    "sha256": sha256_file(artifact_dir / "episode.mp4"),
                }
            )
        except BaseException as video_error:
            summary["video_error"] = "%s: %s" % (type(video_error).__name__, video_error)
            secondary_errors.append(("video", video_error))
        if trace_arrays is not None:
            try:
                if source_render is None:
                    raise RuntimeError("Cannot save an episode trace without render hashes")
                policy_identity = _stable_policy_identity(summary["server_metadata"])
                trace_manifest = save_episode_trace(
                    artifact_dir,
                    {
                        "config": {
                            "task_suite": config.task_suite,
                            "task_id": config.task_id,
                            "init_state_id": config.init_state_id,
                            "seed": config.seed,
                            "settle_steps": config.settle_steps,
                            "max_steps": config.max_steps,
                            "replan_steps": config.replan_steps,
                            "render_size": config.render_size,
                            "fps": config.fps,
                            "flow_noise_seed": config.flow_noise_seed,
                        },
                        "prompt": summary["prompt"],
                        "task_name": summary["task_name"],
                        "policy_identity": policy_identity,
                        "source_server": {
                            "instance_id": summary["server_metadata"].get(
                                "server_instance_id"
                            ),
                            "pid": summary["server_metadata"].get("server_pid"),
                            "started_unix_ns": summary["server_metadata"].get(
                                "server_started_unix_ns"
                            ),
                        },
                        "result": {
                            "status": "completed" if trace_complete else "failed",
                            "trace_complete": trace_complete,
                            "success": summary["success"],
                            "action_steps": summary["action_steps"],
                            "inference_calls": summary["inference_calls"],
                        },
                        "source_render": source_render,
                        "source_video": source_video,
                    },
                    trace_arrays,
                )
                summary["episode_trace"] = str(artifact_dir / "episode-trace.json")
                summary["episode_trace_array_sha256"] = trace_manifest[
                    "array_file_sha256"
                ]
                summary["executed_lengths"] = trace_arrays["executed_lengths"]
                try:
                    event_log.write(
                        "episode_trace_saved",
                        manifest=summary["episode_trace"],
                        array_file_sha256=summary["episode_trace_array_sha256"],
                        executed_lengths=trace_arrays["executed_lengths"],
                        trace_complete=trace_complete,
                    )
                except BaseException as log_error:
                    secondary_errors.append(("trace_event_log", log_error))
            except BaseException as trace_error:
                summary["trace_error"] = "%s: %s" % (
                    type(trace_error).__name__,
                    trace_error,
                )
                secondary_errors.append(("trace", trace_error))

        try:
            event_log.close()
        except BaseException as close_error:
            secondary_errors.append(("event_log_close", close_error))
        if secondary_errors:
            summary["secondary_errors"] = [
                {
                    "stage": label,
                    "error_type": type(caught).__name__,
                    "error": str(caught),
                }
                for label, caught in secondary_errors
            ]
            if error is None:
                error = secondary_errors[0][1]
        if error is not None:
            summary["status"] = "failed"
            summary.setdefault("error_type", type(error).__name__)
            summary.setdefault("error", str(error))
        summary["duration_seconds"] = time.perf_counter() - started
        try:
            (artifact_dir / "summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except BaseException as summary_error:
            if error is None:
                error = summary_error
            else:
                logging.error("Could not write episode summary: %s", summary_error)

    logging.info("Episode artifacts: %s", artifact_dir)
    if error is not None:
        raise error
    return artifact_dir


def run_episode_replay(
    trace_path: str,
    libero_root: str,
    output_root: str = "artifacts",
    host: str = "127.0.0.1",
    port: int = 8000,
    inference_timeout: float = 180.0,
    require_server_restart: bool = False,
) -> pathlib.Path:
    manifest, trace = load_episode_trace(pathlib.Path(trace_path))
    trace_config, trace_result, source_render = _validate_replay_manifest(manifest, trace)
    config = EpisodeConfig(
        libero_root=libero_root,
        output_root=output_root,
        host=host,
        port=port,
        task_suite=str(trace_config["task_suite"]),
        task_id=int(trace_config["task_id"]),
        init_state_id=int(trace_config["init_state_id"]),
        seed=int(trace_config["seed"]),
        settle_steps=int(trace_config["settle_steps"]),
        max_steps=int(trace_config["max_steps"]),
        replan_steps=int(trace_config["replan_steps"]),
        render_size=int(trace_config["render_size"]),
        fps=int(trace_config["fps"]),
        inference_timeout=inference_timeout,
        flow_noise_seed=int(trace_config["flow_noise_seed"]),
    )
    artifact_dir = _new_artifact_dir(
        output_root, "episode-replay", config.task_suite, config.task_id, config.seed
    )
    event_log = EventLog(artifact_dir / "events.jsonl")
    environment = None
    client = None
    frames = []  # type: List[np.ndarray]
    error = None  # type: Optional[BaseException]
    secondary_errors = []  # type: List[Tuple[str, BaseException]]
    started = time.perf_counter()
    summary = {
        "status": "failed",
        "exact_replay": False,
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "source_trace": str(pathlib.Path(trace_path).expanduser().resolve()),
        "source_trace_array_sha256": manifest["array_file_sha256"],
        "action_steps": 0,
        "inference_calls": 0,
        "model_actions_exact": False,
        "environment_states_exact": False,
        "require_server_restart": require_server_restart,
        "source_server_instance_id": manifest.get("source_server", {}).get(
            "instance_id"
        ),
    }  # type: Dict[str, Any]
    try:
        environment, observation, task, prompt = _load_task(config)
        if prompt != manifest.get("prompt") or str(task.name) != manifest.get("task_name"):
            raise RuntimeError("Trace task identity does not match the LIBERO task")
        frames.append(frame_from_observation(observation))
        for settle_index in range(config.settle_steps):
            observation, reward, done, info = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frames.append(frame_from_observation(observation))
            event_log.write("settle_step", index=settle_index, reward=reward, done=done)

        client = PolicyClient(
            host=host,
            port=port,
            connect_timeout=120.0,
            inference_timeout=inference_timeout,
        )
        validate_policy_suite(client.metadata, config.task_suite)
        expected_identity = manifest["policy_identity"]
        for key, expected_value in expected_identity.items():
            if client.metadata.get(key) != expected_value:
                raise RuntimeError(
                    "Replay policy identity mismatch for %s: expected %r, got %r"
                    % (key, expected_value, client.metadata.get(key))
                )
        source_server_instance_id = summary["source_server_instance_id"]
        replay_server_instance_id = client.metadata.get("server_instance_id")
        summary["replay_server_instance_id"] = replay_server_instance_id
        summary["server_instance_changed"] = (
            isinstance(source_server_instance_id, str)
            and isinstance(replay_server_instance_id, str)
            and source_server_instance_id != replay_server_instance_id
        )
        if require_server_restart and not summary["server_instance_changed"]:
            raise RuntimeError(
                "Replay requires a different model-server instance from the capture"
            )
        summary["server_metadata"] = client.metadata
        action_cursor = 0
        for replan_index, executed_length in enumerate(trace["executed_lengths"]):
            if int(trace["replan_start_steps"][replan_index]) != action_cursor:
                raise RuntimeError("Replay action cursor mismatch at replan %d" % replan_index)
            policy_observation = build_policy_observation(observation, prompt)
            _assert_exact_array(
                policy_observation[IMAGE_KEY],
                trace["images"][replan_index],
                "replan %d agent image" % replan_index,
            )
            _assert_exact_array(
                policy_observation[WRIST_IMAGE_KEY],
                trace["wrist_images"][replan_index],
                "replan %d wrist image" % replan_index,
            )
            _assert_exact_array(
                policy_observation[STATE_KEY],
                trace["states"][replan_index],
                "replan %d robot state" % replan_index,
            )
            _assert_exact_array(
                _sim_state(environment),
                trace["sim_states_before"][replan_index],
                "replan %d simulator state" % replan_index,
            )
            request = dict(policy_observation)
            flow_noise = trace["flow_noises"][replan_index]
            request[FLOW_NOISE_KEY] = flow_noise
            response = validate_action_response(client.infer(request))
            expected_noise_sha256 = _sha256_array(flow_noise)
            if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise_sha256:
                raise RuntimeError(
                    "Server did not acknowledge replay flow noise at replan %d"
                    % replan_index
                )
            _assert_exact_array(
                response[ACTION_KEY],
                trace["predicted_actions"][replan_index],
                "replan %d predicted action chunk" % replan_index,
            )
            summary["inference_calls"] += 1
            event_log.write(
                "inference_replayed",
                index=replan_index,
                flow_noise_sha256=expected_noise_sha256,
                action_chunk_sha256=_sha256_array(response[ACTION_KEY]),
                exact=True,
            )

            for chunk_index, action in enumerate(
                response[ACTION_KEY][: int(executed_length)]
            ):
                observation, reward, done, info = environment.step(action.tolist())
                frames.append(frame_from_observation(observation))
                success = bool(environment.check_success())
                expected_reward = float(trace["rewards"][action_cursor])
                expected_done = bool(trace["dones"][action_cursor])
                expected_success = bool(trace["successes"][action_cursor])
                if (
                    float(reward) != expected_reward
                    or bool(done) != expected_done
                    or success != expected_success
                ):
                    raise RuntimeError(
                        "Environment outcome mismatch at action %d: actual=%r, expected=%r"
                        % (
                            action_cursor + 1,
                            (float(reward), bool(done), success),
                            (expected_reward, expected_done, expected_success),
                        )
                    )
                _assert_exact_array(
                    _sim_state(environment),
                    trace["sim_states_after"][action_cursor],
                    "action %d simulator state" % (action_cursor + 1),
                )
                action_cursor += 1
                summary["action_steps"] = action_cursor
                event_log.write(
                    "action_replayed",
                    step=action_cursor,
                    replan_index=replan_index,
                    chunk_index=chunk_index,
                    reward=reward,
                    done=done,
                    success=success,
                )

        final_policy_observation = build_policy_observation(observation, prompt)
        _assert_exact_array(
            final_policy_observation[IMAGE_KEY], trace["final_image"], "final agent image"
        )
        _assert_exact_array(
            final_policy_observation[WRIST_IMAGE_KEY],
            trace["final_wrist_image"],
            "final wrist image",
        )
        _assert_exact_array(
            final_policy_observation[STATE_KEY], trace["final_state"], "final robot state"
        )
        _assert_exact_array(
            _sim_state(environment), trace["final_sim_state"], "final simulator state"
        )
        actual_success = bool(environment.check_success())
        if action_cursor != int(trace_result["action_steps"]):
            raise RuntimeError("Replay action count does not match the trace result")
        if actual_success != bool(trace_result["success"]):
            raise RuntimeError("Replay final success does not match the trace result")
        replay_render = _render_stream_metadata(frames)
        if replay_render["frame_sha256"] != source_render["frame_sha256"]:
            mismatch_index = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(replay_render["frame_sha256"], source_render["frame_sha256"])
                )
                if actual != expected
            )
            raise RuntimeError(
                "Replay render frame %d does not match the source pixels" % mismatch_index
            )
        if replay_render["stream_sha256"] != source_render["stream_sha256"]:
            raise RuntimeError("Replay render stream SHA-256 does not match the source")
        summary.update(
            {
                "status": "completed",
                "exact_replay": True,
                "success": actual_success,
                "prompt": prompt,
                "task_name": str(task.name),
                "model_actions_exact": True,
                "environment_states_exact": True,
                "render_frames_exact": True,
                "render_stream_sha256": replay_render["stream_sha256"],
                "executed_lengths": trace["executed_lengths"],
            }
        )
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
        try:
            event_log.write(
                "replay_error", error_type=type(caught).__name__, error=str(caught)
            )
        except BaseException as log_error:
            secondary_errors.append(("replay_error_log", log_error))
    finally:
        for cleanup_label, resource in (("client_close", client), ("environment_close", environment)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as cleanup_error:
                secondary_errors.append((cleanup_label, cleanup_error))
        if frames:
            try:
                video_path = artifact_dir / "episode.mp4"
                _write_video(video_path, frames, config.fps)
                video_sha256 = sha256_file(video_path)
                source_video_sha256 = manifest.get("source_video", {}).get("sha256")
                summary["video"] = str(video_path)
                summary["video_frames"] = len(frames)
                summary["video_sha256"] = video_sha256
                summary["source_video_sha256"] = source_video_sha256
                summary["video_sha256_equal"] = video_sha256 == source_video_sha256
            except BaseException as video_error:
                summary["video_error"] = "%s: %s" % (
                    type(video_error).__name__,
                    video_error,
                )
                secondary_errors.append(("video", video_error))
        try:
            event_log.close()
        except BaseException as close_error:
            secondary_errors.append(("event_log_close", close_error))
        if secondary_errors:
            summary["secondary_errors"] = [
                {
                    "stage": label,
                    "error_type": type(caught).__name__,
                    "error": str(caught),
                }
                for label, caught in secondary_errors
            ]
            if error is None:
                error = secondary_errors[0][1]
        if error is not None:
            summary["status"] = "failed"
            summary["exact_replay"] = False
            summary.setdefault("error_type", type(error).__name__)
            summary.setdefault("error", str(error))
        summary["duration_seconds"] = time.perf_counter() - started
        try:
            (artifact_dir / "summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except BaseException as summary_error:
            if error is None:
                error = summary_error
            else:
                logging.error("Could not write replay summary: %s", summary_error)
    if error is not None:
        raise error
    logging.info("Episode replay artifacts: %s", artifact_dir)
    return artifact_dir


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run_determinism_check(config: EpisodeConfig, noise_seed: int = 42) -> pathlib.Path:
    artifact_dir = _new_artifact_dir(
        config.output_root, "determinism", config.task_suite, config.task_id, config.seed
    )
    environment = None
    client = None
    error = None  # type: Optional[BaseException]
    summary = {
        "status": "failed",
        "exact_equal": False,
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "noise_seed": noise_seed,
    }  # type: Dict[str, Any]
    try:
        environment, observation, task, prompt = _load_task(config)
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
        policy_observation = build_policy_observation(observation, prompt)
        flow_noise = np.random.default_rng(noise_seed).standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
        policy_observation[FLOW_NOISE_KEY] = flow_noise

        client = PolicyClient(
            host=config.host,
            port=config.port,
            connect_timeout=120.0,
            inference_timeout=config.inference_timeout,
        )
        validate_policy_suite(client.metadata, config.task_suite)
        first = validate_action_response(client.infer(policy_observation))
        second = validate_action_response(client.infer(policy_observation))
        first_actions = first[ACTION_KEY]
        second_actions = second[ACTION_KEY]
        exact_equal = bool(np.array_equal(first_actions, second_actions))
        max_abs_diff = float(np.max(np.abs(first_actions - second_actions)))
        noise_sha256 = _sha256_array(flow_noise)
        if first.get(FLOW_NOISE_SHA256_KEY) != noise_sha256:
            raise RuntimeError("Server did not acknowledge the first fixed flow noise")
        if second.get(FLOW_NOISE_SHA256_KEY) != noise_sha256:
            raise RuntimeError("Server did not acknowledge the second fixed flow noise")

        np.savez_compressed(
            artifact_dir / "replay.npz",
            image=policy_observation["observation/image"],
            wrist_image=policy_observation["observation/wrist_image"],
            state=policy_observation["observation/state"],
            flow_noise=flow_noise,
            first_actions=first_actions,
            second_actions=second_actions,
        )
        summary.update(
            {
                "status": "completed" if exact_equal else "mismatch",
                "exact_equal": exact_equal,
                "max_abs_diff": max_abs_diff,
                "prompt": prompt,
                "observation_image_sha256": _sha256_array(policy_observation["observation/image"]),
                "observation_wrist_image_sha256": _sha256_array(
                    policy_observation["observation/wrist_image"]
                ),
                "observation_state_sha256": _sha256_array(policy_observation["observation/state"]),
                "flow_noise_shape": list(flow_noise.shape),
                "flow_noise_sha256": noise_sha256,
                "actions_shape": list(first_actions.shape),
                "first_actions_sha256": _sha256_array(first_actions),
                "second_actions_sha256": _sha256_array(second_actions),
                "server_metadata": client.metadata,
            }
        )
        if not exact_equal:
            raise RuntimeError("Fixed observation/noise actions differ; max abs diff %.9g" % max_abs_diff)
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
    finally:
        if client is not None:
            client.close()
        if environment is not None:
            environment.close()
        (artifact_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if error is not None:
        raise error
    logging.info("Determinism artifacts: %s", artifact_dir)
    return artifact_dir


def run_routing_probe(
    config: EpisodeConfig,
    candidate_count: int = 8,
    noise_seed: int = 42,
    rad_neighbors: int = 2,
) -> pathlib.Path:
    if not 2 <= candidate_count <= 64:
        raise ValueError("candidate_count must be between 2 and 64")
    if not 1 <= rad_neighbors < candidate_count:
        raise ValueError("rad_neighbors must be in [1, candidate_count - 1]")
    artifact_dir = _new_artifact_dir(
        config.output_root, "routing-probe", config.task_suite, config.task_id, config.seed
    )
    environment = None
    client = None
    error = None  # type: Optional[BaseException]
    started = time.perf_counter()
    summary = {
        "status": "failed",
        "task_suite": config.task_suite,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "noise_seed": noise_seed,
        "candidate_count": candidate_count,
        "rad_neighbors": rad_neighbors,
    }  # type: Dict[str, Any]
    try:
        environment, observation, task, prompt = _load_task(config)
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
        policy_observation = build_policy_observation(observation, prompt)
        imageio.imwrite(artifact_dir / "agentview.png", policy_observation["observation/image"])
        imageio.imwrite(artifact_dir / "wrist.png", policy_observation["observation/wrist_image"])

        noise_bank = np.random.default_rng(noise_seed).standard_normal(
            (candidate_count,) + FLOW_NOISE_SHAPE
        ).astype(np.float32)
        actions = []  # type: List[np.ndarray]
        expert_ids = []  # type: List[np.ndarray]
        expert_weights = []  # type: List[np.ndarray]
        inference_ms = []  # type: List[float]
        layer_indices = None  # type: Optional[np.ndarray]

        client = PolicyClient(
            host=config.host,
            port=config.port,
            connect_timeout=120.0,
            inference_timeout=config.inference_timeout,
        )
        validate_policy_suite(client.metadata, config.task_suite)
        if not client.metadata.get("routing_capture_supported", False):
            raise RuntimeError("Policy server does not advertise routing capture support")

        for candidate_index in range(candidate_count):
            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = noise_bank[candidate_index]
            request[ROUTING_CAPTURE_KEY] = True
            request_started = time.perf_counter()
            response = validate_action_response(client.infer(request))
            inference_ms.append((time.perf_counter() - request_started) * 1000.0)
            expected_noise_sha256 = _sha256_array(noise_bank[candidate_index])
            if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise_sha256:
                raise RuntimeError(
                    "Server did not acknowledge candidate %d flow noise" % candidate_index
                )
            actions.append(response[ACTION_KEY])
            expert_ids.append(response[ROUTING_EXPERT_IDS_KEY])
            expert_weights.append(response[ROUTING_EXPERT_WEIGHTS_KEY])
            candidate_layers = response[ROUTING_LAYER_INDICES_KEY]
            if layer_indices is None:
                layer_indices = candidate_layers
            elif not np.array_equal(layer_indices, candidate_layers):
                raise RuntimeError("HB-MoE layer indices changed between candidates")

        action_pool = np.stack(actions, axis=0)
        expert_id_pool = np.stack(expert_ids, axis=0)
        expert_weight_pool = np.stack(expert_weights, axis=0)

        repeat_request = dict(policy_observation)
        repeat_request[FLOW_NOISE_KEY] = noise_bank[0]
        repeat_request[ROUTING_CAPTURE_KEY] = True
        repeat_response = validate_action_response(client.infer(repeat_request))
        no_capture_request = dict(policy_observation)
        no_capture_request[FLOW_NOISE_KEY] = noise_bank[0]
        no_capture_response = validate_action_response(client.infer(no_capture_request))
        candidate_zero_noise_sha256 = _sha256_array(noise_bank[0])
        if repeat_response.get(FLOW_NOISE_SHA256_KEY) != candidate_zero_noise_sha256:
            raise RuntimeError("Server did not acknowledge repeated candidate flow noise")
        if no_capture_response.get(FLOW_NOISE_SHA256_KEY) != candidate_zero_noise_sha256:
            raise RuntimeError("Server did not acknowledge no-capture control flow noise")
        if any(
            key in no_capture_response
            for key in (
                ROUTING_EXPERT_IDS_KEY,
                ROUTING_EXPERT_WEIGHTS_KEY,
                ROUTING_LAYER_INDICES_KEY,
            )
        ):
            raise RuntimeError("Server returned routing data when capture was disabled")
        route_ids_reproducible = bool(
            np.array_equal(expert_id_pool[0], repeat_response[ROUTING_EXPERT_IDS_KEY])
        )
        route_weights_reproducible = bool(
            np.array_equal(expert_weight_pool[0], repeat_response[ROUTING_EXPERT_WEIGHTS_KEY])
        )
        actions_reproducible = bool(
            np.array_equal(action_pool[0], repeat_response[ACTION_KEY])
        )
        capture_is_non_intervening = bool(
            np.array_equal(action_pool[0], no_capture_response[ACTION_KEY])
        )
        if not all(
            (
                route_ids_reproducible,
                route_weights_reproducible,
                actions_reproducible,
                capture_is_non_intervening,
            )
        ):
            raise RuntimeError("Routing capture reproducibility/non-intervention check failed")

        analysis = analyze_candidate_pool(
            action_pool,
            expert_id_pool,
            expert_weight_pool,
            action_std=np.asarray(
                client.metadata.get("normalization_action_std"), dtype=np.float64
            ),
            rad_neighbors=rad_neighbors,
        )
        analysis["selectors"]["random"] = int(
            np.random.default_rng(noise_seed + 1_000_003).integers(candidate_count)
        )
        route_distance_max = analysis["routing"]["aligned_mean_tv_off_diagonal"]["max"]
        route_signal = bool(
            analysis["routing"]["candidate_dependent"] and route_distance_max > 1e-6
        )
        action_signal = bool(
            analysis["actions"]["candidate_dependent"]
            and analysis["actions"]["pairwise_raw_max_abs"]["max"] > 1e-6
        )
        go_decision = route_signal and action_signal

        np.savez_compressed(
            artifact_dir / "routing-probe.npz",
            image=policy_observation["observation/image"],
            wrist_image=policy_observation["observation/wrist_image"],
            state=policy_observation["observation/state"],
            noise_bank=noise_bank,
            actions=action_pool,
            expert_ids=expert_id_pool,
            expert_weights=expert_weight_pool,
            layer_indices=layer_indices,
            repeat_actions=repeat_response[ACTION_KEY],
            repeat_expert_ids=repeat_response[ROUTING_EXPERT_IDS_KEY],
            repeat_expert_weights=repeat_response[ROUTING_EXPERT_WEIGHTS_KEY],
            no_capture_actions=no_capture_response[ACTION_KEY],
        )
        summary.update(
            {
                "status": "completed",
                "prompt": prompt,
                "task_name": str(task.name),
                "server_metadata": client.metadata,
                "observation": {
                    "image_sha256": _sha256_array(policy_observation["observation/image"]),
                    "wrist_image_sha256": _sha256_array(
                        policy_observation["observation/wrist_image"]
                    ),
                    "state_sha256": _sha256_array(policy_observation["observation/state"]),
                },
                "noise_bank_shape": list(noise_bank.shape),
                "noise_bank_sha256": _sha256_array(noise_bank),
                "actions_shape": list(action_pool.shape),
                "routing_shape": list(expert_id_pool.shape),
                "routing_layer_indices": layer_indices,
                "inference_ms": {
                    "per_candidate": inference_ms,
                    "mean": float(np.mean(inference_ms)),
                    "total": float(np.sum(inference_ms)),
                },
                "analysis": analysis,
                "capture_invariants": {
                    "same_noise_actions_exact": actions_reproducible,
                    "same_noise_route_ids_exact": route_ids_reproducible,
                    "same_noise_route_weights_exact": route_weights_reproducible,
                    "capture_does_not_change_actions": capture_is_non_intervening,
                    "candidate0_actions_sha256": _sha256_array(action_pool[0]),
                    "repeat_actions_sha256": _sha256_array(repeat_response[ACTION_KEY]),
                    "no_capture_actions_sha256": _sha256_array(no_capture_response[ACTION_KEY]),
                },
                "go_no_go": {
                    "route_candidate_signal": route_signal,
                    "action_candidate_signal": action_signal,
                    "continue_to_outcome_evaluation": go_decision,
                    "scope": "signal check only; selector quality is not established",
                },
                "artifact_npz": str(artifact_dir / "routing-probe.npz"),
            }
        )
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
    finally:
        if client is not None:
            client.close()
        if environment is not None:
            environment.close()
        summary["duration_seconds"] = time.perf_counter() - started
        (artifact_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if error is not None:
        raise error
    logging.info("Routing probe artifacts: %s", artifact_dir)
    return artifact_dir


def run_environment_check(config: EpisodeConfig, action_steps: int = 10) -> pathlib.Path:
    artifact_dir = _new_artifact_dir(
        config.output_root, "env-check", config.task_suite, config.task_id, config.seed
    )
    environment = None
    frames = []  # type: List[np.ndarray]
    started = time.perf_counter()
    summary = {"status": "failed", "action_steps": 0}  # type: Dict[str, Any]
    error = None  # type: Optional[BaseException]
    try:
        environment, observation, task, prompt = _load_task(config)
        frames.append(frame_from_observation(observation))
        for step in range(action_steps):
            observation, reward, done, info = environment.step(LIBERO_DUMMY_ACTION.tolist())
            frames.append(frame_from_observation(observation))
            summary["action_steps"] = step + 1
        summary.update(
            {
                "status": "completed",
                "task_suite": config.task_suite,
                "task_id": config.task_id,
                "prompt": prompt,
                "observation_keys": sorted(observation),
                "frame_shape": list(frames[-1].shape),
            }
        )
    except BaseException as caught:
        error = caught
        summary["error_type"] = type(caught).__name__
        summary["error"] = str(caught)
    finally:
        if environment is not None:
            environment.close()
        if frames:
            try:
                _write_video(artifact_dir / "env-check.mp4", frames, config.fps)
                summary["video"] = str(artifact_dir / "env-check.mp4")
            except BaseException as video_error:
                summary["video_error"] = "%s: %s" % (type(video_error).__name__, video_error)
                if error is None:
                    error = video_error
        summary["duration_seconds"] = time.perf_counter() - started
        (artifact_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if error is not None:
        raise error
    logging.info("Environment check artifacts: %s", artifact_dir)
    return artifact_dir
