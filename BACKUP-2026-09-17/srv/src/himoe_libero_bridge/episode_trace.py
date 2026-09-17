"""Versioned, hash-checked episode traces for exact local replay."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from himoe_libero_bridge.protocol import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)

TRACE_SCHEMA = "himoe-libero-episode-trace-v1"
TRACE_ARRAY_FILE = "episode-trace.npz"
TRACE_MANIFEST_FILE = "episode-trace.json"

TRACE_ARRAY_KEYS = (
    "images",
    "wrist_images",
    "states",
    "flow_noises",
    "predicted_actions",
    "executed_lengths",
    "replan_start_steps",
    "sim_states_before",
    "rewards",
    "dones",
    "successes",
    "sim_states_after",
    "final_image",
    "final_wrist_image",
    "final_state",
    "final_sim_state",
)


class EpisodeTraceError(ValueError):
    """Raised when an episode trace is incomplete or internally inconsistent."""


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EpisodeTraceError("Trace metadata is not canonical JSON: %s" % error)
    return hashlib.sha256(payload).hexdigest()


def _require_array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    dtype: np.dtype,
    shape: Optional[Tuple[Optional[int], ...]] = None,
) -> np.ndarray:
    if key not in arrays:
        raise EpisodeTraceError("Episode trace is missing array %s" % key)
    value = np.asarray(arrays[key])
    if value.dtype != np.dtype(dtype):
        raise EpisodeTraceError(
            "Episode trace %s dtype must be %s, got %s" % (key, np.dtype(dtype), value.dtype)
        )
    if shape is not None:
        if value.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)
        ):
            raise EpisodeTraceError(
                "Episode trace %s shape must match %s, got %s" % (key, shape, value.shape)
            )
    if value.dtype.kind in ("f", "c") and not np.all(np.isfinite(value)):
        raise EpisodeTraceError("Episode trace %s contains NaN or infinity" % key)
    return np.ascontiguousarray(value)


def validate_episode_trace_arrays(
    arrays: Mapping[str, np.ndarray], replan_steps: int
) -> Dict[str, np.ndarray]:
    if not 1 <= int(replan_steps) <= ACTION_CHUNK_STEPS:
        raise EpisodeTraceError("replan_steps must be in [1, %d]" % ACTION_CHUNK_STEPS)
    missing = sorted(set(TRACE_ARRAY_KEYS) - set(arrays))
    extra = sorted(set(arrays) - set(TRACE_ARRAY_KEYS))
    if missing:
        raise EpisodeTraceError("Episode trace is missing arrays: %s" % ", ".join(missing))
    if extra:
        raise EpisodeTraceError("Episode trace has unexpected arrays: %s" % ", ".join(extra))

    images = _require_array(arrays, "images", np.uint8, (None,) + IMAGE_SHAPE)
    replan_count = images.shape[0]
    if replan_count < 1:
        raise EpisodeTraceError("Episode trace must contain at least one replan")
    canonical = {
        "images": images,
        "wrist_images": _require_array(
            arrays, "wrist_images", np.uint8, (replan_count,) + IMAGE_SHAPE
        ),
        "states": _require_array(
            arrays, "states", np.float32, (replan_count, STATE_DIM)
        ),
        "flow_noises": _require_array(
            arrays, "flow_noises", np.float32, (replan_count,) + FLOW_NOISE_SHAPE
        ),
        "predicted_actions": _require_array(
            arrays,
            "predicted_actions",
            np.float32,
            (replan_count, ACTION_CHUNK_STEPS, ACTION_DIM),
        ),
        "executed_lengths": _require_array(
            arrays, "executed_lengths", np.int16, (replan_count,)
        ),
        "replan_start_steps": _require_array(
            arrays, "replan_start_steps", np.int32, (replan_count,)
        ),
        "sim_states_before": _require_array(
            arrays, "sim_states_before", np.float64, (replan_count, None)
        ),
    }
    executed_lengths = canonical["executed_lengths"]
    if np.any(executed_lengths < 1) or np.any(executed_lengths > replan_steps):
        raise EpisodeTraceError("executed_lengths must be in [1, replan_steps]")
    if replan_count > 1 and np.any(executed_lengths[:-1] != replan_steps):
        raise EpisodeTraceError("Only the final replan may execute a partial action chunk")
    expected_starts = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(executed_lengths[:-1], dtype=np.int64),
        )
    )
    if not np.array_equal(canonical["replan_start_steps"], expected_starts):
        raise EpisodeTraceError("replan_start_steps does not match executed_lengths")

    action_steps = int(np.sum(executed_lengths, dtype=np.int64))
    sim_state_dim = canonical["sim_states_before"].shape[1]
    if sim_state_dim < 1:
        raise EpisodeTraceError("Simulator state vectors must not be empty")
    canonical.update(
        {
            "rewards": _require_array(arrays, "rewards", np.float64, (action_steps,)),
            "dones": _require_array(arrays, "dones", np.bool_, (action_steps,)),
            "successes": _require_array(arrays, "successes", np.bool_, (action_steps,)),
            "sim_states_after": _require_array(
                arrays, "sim_states_after", np.float64, (action_steps, sim_state_dim)
            ),
            "final_image": _require_array(arrays, "final_image", np.uint8, IMAGE_SHAPE),
            "final_wrist_image": _require_array(
                arrays, "final_wrist_image", np.uint8, IMAGE_SHAPE
            ),
            "final_state": _require_array(arrays, "final_state", np.float32, (STATE_DIM,)),
            "final_sim_state": _require_array(
                arrays, "final_sim_state", np.float64, (sim_state_dim,)
            ),
        }
    )
    if action_steps and not np.array_equal(
        canonical["final_sim_state"], canonical["sim_states_after"][-1]
    ):
        raise EpisodeTraceError("final_sim_state does not match the last action state")
    if np.any(canonical["successes"][:-1]):
        raise EpisodeTraceError("A trace must stop immediately after its first successful action")
    return canonical


class EpisodeTraceRecorder:
    def __init__(self, replan_steps: int) -> None:
        if not 1 <= int(replan_steps) <= ACTION_CHUNK_STEPS:
            raise ValueError("replan_steps must be in [1, %d]" % ACTION_CHUNK_STEPS)
        self.replan_steps = int(replan_steps)
        self._replans = []
        self._rewards = []
        self._dones = []
        self._successes = []
        self._sim_states_after = []
        self._final_observation = None
        self._final_sim_state = None

    @property
    def action_count(self) -> int:
        return len(self._rewards)

    def start_replan(
        self,
        observation: Mapping[str, Any],
        flow_noise: np.ndarray,
        predicted_actions: np.ndarray,
        sim_state: np.ndarray,
    ) -> None:
        if self._replans and self._replans[-1]["executed_length"] != self.replan_steps:
            raise EpisodeTraceError("Cannot start a replan after a partial action chunk")
        self._replans.append(
            {
                "image": np.array(observation[IMAGE_KEY], dtype=np.uint8, copy=True, order="C"),
                "wrist_image": np.array(
                    observation[WRIST_IMAGE_KEY], dtype=np.uint8, copy=True, order="C"
                ),
                "state": np.array(observation[STATE_KEY], dtype=np.float32, copy=True, order="C"),
                "flow_noise": np.array(flow_noise, dtype=np.float32, copy=True, order="C"),
                "predicted_actions": np.array(
                    predicted_actions, dtype=np.float32, copy=True, order="C"
                ),
                "executed_length": 0,
                "start_step": len(self._rewards),
                "sim_state": np.array(sim_state, dtype=np.float64, copy=True, order="C"),
            }
        )

    def record_action(
        self,
        action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        sim_state: np.ndarray,
    ) -> None:
        if not self._replans:
            raise EpisodeTraceError("Cannot record an action before the first replan")
        replan = self._replans[-1]
        chunk_index = int(replan["executed_length"])
        if chunk_index >= self.replan_steps:
            raise EpisodeTraceError("The current action chunk is already complete")
        expected_action = replan["predicted_actions"][chunk_index]
        actual_action = np.asarray(action, dtype=np.float32)
        if not np.array_equal(actual_action, expected_action):
            raise EpisodeTraceError("Executed action does not match the predicted chunk prefix")
        replan["executed_length"] = chunk_index + 1
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._successes.append(bool(success))
        self._sim_states_after.append(
            np.array(sim_state, dtype=np.float64, copy=True, order="C")
        )

    def finish(self, observation: Mapping[str, Any], sim_state: np.ndarray) -> None:
        if not self._replans or self._replans[-1]["executed_length"] < 1:
            raise EpisodeTraceError("A complete trace must contain an executed action")
        self._final_observation = {
            IMAGE_KEY: np.array(observation[IMAGE_KEY], dtype=np.uint8, copy=True, order="C"),
            WRIST_IMAGE_KEY: np.array(
                observation[WRIST_IMAGE_KEY], dtype=np.uint8, copy=True, order="C"
            ),
            STATE_KEY: np.array(observation[STATE_KEY], dtype=np.float32, copy=True, order="C"),
        }
        self._final_sim_state = np.array(sim_state, dtype=np.float64, copy=True, order="C")

    def arrays(self) -> Dict[str, np.ndarray]:
        if self._final_observation is None or self._final_sim_state is None:
            raise EpisodeTraceError("Episode trace recorder has not been finished")
        arrays = {
            "images": np.stack([item["image"] for item in self._replans]),
            "wrist_images": np.stack([item["wrist_image"] for item in self._replans]),
            "states": np.stack([item["state"] for item in self._replans]),
            "flow_noises": np.stack([item["flow_noise"] for item in self._replans]),
            "predicted_actions": np.stack(
                [item["predicted_actions"] for item in self._replans]
            ),
            "executed_lengths": np.asarray(
                [item["executed_length"] for item in self._replans], dtype=np.int16
            ),
            "replan_start_steps": np.asarray(
                [item["start_step"] for item in self._replans], dtype=np.int32
            ),
            "sim_states_before": np.stack([item["sim_state"] for item in self._replans]),
            "rewards": np.asarray(self._rewards, dtype=np.float64),
            "dones": np.asarray(self._dones, dtype=np.bool_),
            "successes": np.asarray(self._successes, dtype=np.bool_),
            "sim_states_after": np.stack(self._sim_states_after),
            "final_image": self._final_observation[IMAGE_KEY],
            "final_wrist_image": self._final_observation[WRIST_IMAGE_KEY],
            "final_state": self._final_observation[STATE_KEY],
            "final_sim_state": self._final_sim_state,
        }
        return validate_episode_trace_arrays(arrays, self.replan_steps)


def _atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_episode_trace(
    directory: pathlib.Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    directory = pathlib.Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = metadata.get("config")
    if not isinstance(config, Mapping) or "replan_steps" not in config:
        raise EpisodeTraceError("Trace metadata must contain config.replan_steps")
    canonical = validate_episode_trace_arrays(arrays, int(config["replan_steps"]))
    array_path = directory / TRACE_ARRAY_FILE
    temporary = array_path.with_name(".%s.tmp-%d" % (array_path.name, os.getpid()))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **canonical)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(array_path))
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = dict(metadata)
    metadata_sha256 = _sha256_json(manifest)
    manifest.update(
        {
            "schema": TRACE_SCHEMA,
            "metadata_sha256": metadata_sha256,
            "array_file": TRACE_ARRAY_FILE,
            "array_file_sha256": sha256_file(array_path),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": sha256_array(value),
                }
                for key, value in canonical.items()
            },
        }
    )
    _atomic_write_json(directory / TRACE_MANIFEST_FILE, manifest)
    return manifest


def load_episode_trace(path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    source = pathlib.Path(path).expanduser().resolve()
    if source.is_dir():
        manifest_path = source / TRACE_MANIFEST_FILE
    elif source.name == TRACE_MANIFEST_FILE or source.suffix == ".json":
        manifest_path = source
    elif source.name == TRACE_ARRAY_FILE or source.suffix == ".npz":
        manifest_path = source.with_name(TRACE_MANIFEST_FILE)
    else:
        raise EpisodeTraceError("Trace path must be an artifact directory, JSON, or NPZ file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EpisodeTraceError("Could not read trace manifest %s: %s" % (manifest_path, error))
    if manifest.get("schema") != TRACE_SCHEMA:
        raise EpisodeTraceError("Unsupported episode trace schema: %r" % manifest.get("schema"))
    generated_keys = {
        "schema",
        "metadata_sha256",
        "array_file",
        "array_file_sha256",
        "arrays",
    }
    metadata_payload = {
        key: value for key, value in manifest.items() if key not in generated_keys
    }
    if _sha256_json(metadata_payload) != manifest.get("metadata_sha256"):
        raise EpisodeTraceError("Episode trace metadata SHA-256 mismatch")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or "replan_steps" not in config:
        raise EpisodeTraceError("Trace manifest is missing config.replan_steps")
    array_path = manifest_path.parent / str(manifest.get("array_file", TRACE_ARRAY_FILE))
    expected_file_sha256 = manifest.get("array_file_sha256")
    if not array_path.is_file():
        raise EpisodeTraceError("Episode trace array file is missing: %s" % array_path)
    actual_file_sha256 = sha256_file(array_path)
    if actual_file_sha256 != expected_file_sha256:
        raise EpisodeTraceError(
            "Episode trace file SHA-256 mismatch: expected %s, got %s"
            % (expected_file_sha256, actual_file_sha256)
        )
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != set(TRACE_ARRAY_KEYS):
                raise EpisodeTraceError("Episode trace archive keys do not match the schema")
            arrays = {key: np.array(archive[key], copy=True, order="C") for key in archive.files}
    except EpisodeTraceError:
        raise
    except Exception as error:
        raise EpisodeTraceError("Could not read episode trace arrays: %s" % error)
    canonical = validate_episode_trace_arrays(arrays, int(config["replan_steps"]))
    array_metadata = manifest.get("arrays")
    if not isinstance(array_metadata, Mapping):
        raise EpisodeTraceError("Trace manifest is missing per-array metadata")
    if set(array_metadata) != set(TRACE_ARRAY_KEYS):
        raise EpisodeTraceError("Trace manifest array metadata keys do not match the schema")
    for key, value in canonical.items():
        expected = array_metadata.get(key)
        if not isinstance(expected, Mapping):
            raise EpisodeTraceError("Trace manifest is missing metadata for %s" % key)
        if list(value.shape) != expected.get("shape") or str(value.dtype) != expected.get("dtype"):
            raise EpisodeTraceError("Trace manifest shape/dtype mismatch for %s" % key)
        if sha256_array(value) != expected.get("sha256"):
            raise EpisodeTraceError("Trace array SHA-256 mismatch for %s" % key)
    return manifest, canonical
