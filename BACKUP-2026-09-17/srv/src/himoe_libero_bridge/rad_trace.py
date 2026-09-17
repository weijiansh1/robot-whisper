"""Hash-checked trace artifacts for closed-loop routing-selection episodes."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from himoe_libero_bridge.episode_trace import sha256_array, sha256_file
from himoe_libero_bridge.protocol import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    FLOW_NOISE_SHAPE,
    HB_MOE_EXPERTS,
    HB_MOE_LAYERS,
    IMAGE_SHAPE,
    ROUTING_TRACE_SHAPE,
    STATE_DIM,
)
from himoe_libero_bridge.rad_experiment import (
    PAIRED_SELECTORS,
    RAD_EXPERIMENT_SCHEMA,
    SELECTOR_ACTION_MEDOID,
    SELECTOR_K1,
    SELECTOR_ROUTING_MEDOID,
    generate_noise_schedule,
    generate_random_selector_schedule,
    select_candidate,
)
from himoe_libero_bridge.routing import routing_metric_contract

RAD_TRACE_SCHEMA = "himoe-libero-rad-trace-v2"
RAD_SOURCE_IDENTITY_SCHEMA = "himoe-libero-rad-source-identity-v1"
RAD_TRACE_ARRAY_FILE = "rad-trace.npz"
RAD_TRACE_MANIFEST_FILE = "rad-trace.json"

RAD_TRACE_ARRAY_KEYS = (
    "images",
    "wrist_images",
    "states",
    "sim_states_before",
    "noise_schedule",
    "candidate_actions",
    "expert_ids",
    "expert_weights",
    "layer_indices",
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
    "final_image",
    "final_wrist_image",
    "final_state",
    "final_sim_state",
)


class RadTraceError(ValueError):
    pass


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    payload = dict(identity)
    payload.pop("identity_sha256", None)
    return _canonical_json_sha256(payload)


def _array(
    arrays: Mapping[str, np.ndarray],
    key: str,
    dtype: np.dtype,
    shape: Tuple[Any, ...],
) -> np.ndarray:
    if key not in arrays:
        raise RadTraceError("RAD trace is missing array %s" % key)
    value = np.asarray(arrays[key])
    if value.dtype != np.dtype(dtype):
        raise RadTraceError(
            "RAD trace %s dtype must be %s, got %s" % (key, np.dtype(dtype), value.dtype)
        )
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise RadTraceError(
            "RAD trace %s shape must match %s, got %s" % (key, shape, value.shape)
        )
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise RadTraceError("RAD trace %s contains NaN or infinity" % key)
    return np.ascontiguousarray(value)


def _trace_config(metadata: Mapping[str, Any]) -> Tuple[str, int, int, int]:
    if metadata.get("experiment_schema") != RAD_EXPERIMENT_SCHEMA:
        raise RadTraceError("RAD trace has an unexpected experiment schema")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise RadTraceError("RAD trace metadata is missing config")
    selector = config.get("selector")
    if selector not in PAIRED_SELECTORS:
        raise RadTraceError("RAD trace contains an invalid selector")
    try:
        candidate_count = int(config["candidate_count"])
        replan_steps = int(config["replan_steps"])
        max_replans = int(config["max_replans"])
    except (KeyError, TypeError, ValueError) as error:
        raise RadTraceError("RAD trace config has invalid dimensions") from error
    expected_candidates = 1 if selector == SELECTOR_K1 else 8
    if candidate_count != expected_candidates:
        raise RadTraceError(
            "RAD selector %s requires candidate_count=%d" % (selector, expected_candidates)
        )
    if not 1 <= replan_steps <= ACTION_CHUNK_STEPS or max_replans < 1:
        raise RadTraceError("RAD trace config has invalid replan limits")
    metric = metadata.get("metric_schema")
    expected_metric = routing_metric_contract(posthoc_selection_allowed=False)
    if metric != expected_metric:
        raise RadTraceError("RAD trace metric schema is not the frozen primary schema")
    identity = metadata.get("policy_identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("backend"), str):
        raise RadTraceError("RAD trace policy identity is missing")
    if identity["backend"].startswith("himoe-vla-") and (
        not isinstance(identity.get("checkpoint_sha256"), str)
        or not isinstance(identity.get("normalization_stats_sha256"), str)
        or identity.get("himoe_upstream_commit") is None
        or not isinstance(identity.get("himoe_patch_sha256"), Mapping)
        or identity.get("himoe_patches_verified_applied") is not True
    ):
        raise RadTraceError("RAD trace HiMoE identity is missing checkpoint/stat hashes")
    routing_layers = metadata.get("routing_layer_indices")
    if (
        not isinstance(routing_layers, list)
        or len(routing_layers) != HB_MOE_LAYERS
        or any(not isinstance(value, int) for value in routing_layers)
        or any(right <= left for left, right in zip(routing_layers, routing_layers[1:]))
        or identity.get("routing_hb_layer_indices") != routing_layers
    ):
        raise RadTraceError("RAD trace routing layer identity is invalid")
    experiment_identity = metadata.get("experiment_identity")
    if (
        not isinstance(experiment_identity, Mapping)
        or experiment_identity.get("schema") != RAD_SOURCE_IDENTITY_SCHEMA
        or not _is_sha256(experiment_identity.get("identity_sha256"))
        or experiment_identity["identity_sha256"]
        != _identity_sha256(experiment_identity)
        or metadata.get("experiment_identity_sha256")
        != experiment_identity["identity_sha256"]
    ):
        raise RadTraceError("RAD trace experiment source identity is invalid")
    if not _is_sha256(metadata.get("initial_runtime_sim_state_sha256")):
        raise RadTraceError("RAD trace initial runtime simulator identity is invalid")
    action_std = np.asarray(metadata.get("normalization_action_std"))
    if (
        action_std.shape != (ACTION_DIM,)
        or not np.issubdtype(action_std.dtype, np.number)
        or not np.all(np.isfinite(action_std))
        or np.any(action_std <= 0.0)
    ):
        raise RadTraceError("RAD trace normalization_action_std is invalid")
    return selector, candidate_count, replan_steps, max_replans


def validate_rad_trace_arrays(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    missing = sorted(set(RAD_TRACE_ARRAY_KEYS) - set(arrays))
    extra = sorted(set(arrays) - set(RAD_TRACE_ARRAY_KEYS))
    if missing or extra:
        raise RadTraceError("RAD trace array keys mismatch; missing=%s extra=%s" % (missing, extra))
    selector, candidate_count, replan_steps, max_replans = _trace_config(metadata)

    images = _array(arrays, "images", np.uint8, (None,) + IMAGE_SHAPE)
    replan_count = images.shape[0]
    if not 1 <= replan_count <= max_replans:
        raise RadTraceError("RAD trace contains an invalid number of replans")
    sim_states_before = _array(
        arrays, "sim_states_before", np.float64, (replan_count, None)
    )
    sim_state_dim = sim_states_before.shape[1]
    if sim_state_dim < 1:
        raise RadTraceError("RAD trace simulator states must not be empty")

    canonical = {
        "images": images,
        "wrist_images": _array(
            arrays, "wrist_images", np.uint8, (replan_count,) + IMAGE_SHAPE
        ),
        "states": _array(arrays, "states", np.float32, (replan_count, STATE_DIM)),
        "sim_states_before": sim_states_before,
        "noise_schedule": _array(
            arrays,
            "noise_schedule",
            np.float32,
            (max_replans, 8) + FLOW_NOISE_SHAPE,
        ),
        "candidate_actions": _array(
            arrays,
            "candidate_actions",
            np.float32,
            (replan_count, candidate_count, ACTION_CHUNK_STEPS, ACTION_DIM),
        ),
        "expert_ids": _array(
            arrays,
            "expert_ids",
            np.int16,
            (replan_count, candidate_count) + ROUTING_TRACE_SHAPE,
        ),
        "expert_weights": _array(
            arrays,
            "expert_weights",
            np.float32,
            (replan_count, candidate_count) + ROUTING_TRACE_SHAPE,
        ),
        "layer_indices": _array(
            arrays, "layer_indices", np.int16, (HB_MOE_LAYERS,)
        ),
        "selected_indices": _array(
            arrays, "selected_indices", np.int16, (replan_count,)
        ),
        "selector_scores": _array(
            arrays, "selector_scores", np.float64, (replan_count, candidate_count)
        ),
        "selector_score_mask": _array(
            arrays, "selector_score_mask", np.bool_, (replan_count, candidate_count)
        ),
        "selected_actions": _array(
            arrays,
            "selected_actions",
            np.float32,
            (replan_count, ACTION_CHUNK_STEPS, ACTION_DIM),
        ),
        "candidate_inference_ms": _array(
            arrays, "candidate_inference_ms", np.float64, (replan_count, candidate_count)
        ),
        "executed_lengths": _array(
            arrays, "executed_lengths", np.int16, (replan_count,)
        ),
        "replan_start_steps": _array(
            arrays, "replan_start_steps", np.int32, (replan_count,)
        ),
    }

    if np.any(canonical["expert_ids"] < 0) or np.any(
        canonical["expert_ids"] >= HB_MOE_EXPERTS
    ):
        raise RadTraceError("RAD trace contains an out-of-range expert id")
    if np.any(
        np.diff(np.sort(canonical["expert_ids"], axis=-1), axis=-1) == 0
    ):
        raise RadTraceError("RAD trace top-k expert ids must be unique at every site")
    if np.any(canonical["expert_weights"] < 0.0) or not np.allclose(
        canonical["expert_weights"].sum(axis=-1), 1.0, atol=1e-4, rtol=1e-4
    ):
        raise RadTraceError("RAD trace expert weights must be non-negative and sum to one")
    if np.any(np.diff(canonical["layer_indices"]) <= 0):
        raise RadTraceError("RAD trace layer indices must be strictly increasing")
    if canonical["layer_indices"].astype(int).tolist() != metadata[
        "routing_layer_indices"
    ]:
        raise RadTraceError("RAD trace routing layer array disagrees with metadata")
    selected = canonical["selected_indices"]
    if np.any(selected < 0) or np.any(selected >= candidate_count):
        raise RadTraceError("RAD trace contains an invalid selected candidate index")
    expected_selected_actions = canonical["candidate_actions"][
        np.arange(replan_count), selected.astype(np.int64)
    ]
    if not np.array_equal(canonical["selected_actions"], expected_selected_actions):
        raise RadTraceError("RAD trace selected actions do not match candidate_actions")

    config = metadata["config"]
    try:
        noise_seed = int(config["noise_seed"])
        random_selector_seed = int(config["random_selector_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RadTraceError("RAD trace selector seeds are invalid") from error
    if noise_seed < 0 or random_selector_seed < 0:
        raise RadTraceError("RAD trace selector seeds must be non-negative")
    expected_noise_schedule = generate_noise_schedule(max_replans, 8, noise_seed)
    if not np.array_equal(canonical["noise_schedule"], expected_noise_schedule):
        raise RadTraceError("RAD trace noise schedule does not match its declared seed")
    random_schedule = generate_random_selector_schedule(
        max_replans, 8, random_selector_seed
    )
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    for replan_index in range(replan_count):
        expected_selection = select_candidate(
            selector,
            canonical["candidate_actions"][replan_index],
            action_std=action_std,
            expert_ids=canonical["expert_ids"][replan_index],
            expert_weights=canonical["expert_weights"][replan_index],
            random_index=int(random_schedule[replan_index]),
        )
        if int(selected[replan_index]) != expected_selection.index:
            raise RadTraceError(
                "RAD trace selected index does not match %s at replan %d"
                % (selector, replan_index)
            )
        expected_has_scores = selector in (
            SELECTOR_ACTION_MEDOID,
            SELECTOR_ROUTING_MEDOID,
        )
        if not np.all(canonical["selector_score_mask"][replan_index] == expected_has_scores):
            raise RadTraceError("RAD trace selector score mask is inconsistent")
        if expected_has_scores and not np.array_equal(
            canonical["selector_scores"][replan_index], expected_selection.scores
        ):
            raise RadTraceError("RAD trace selector scores are inconsistent")
        if not expected_has_scores and np.any(
            canonical["selector_scores"][replan_index] != 0.0
        ):
            raise RadTraceError("RAD trace scoreless selector must store zero scores")

    executed_lengths = canonical["executed_lengths"]
    if np.any(executed_lengths < 1) or np.any(executed_lengths > replan_steps):
        raise RadTraceError("RAD trace executed_lengths are invalid")
    if replan_count > 1 and np.any(executed_lengths[:-1] != replan_steps):
        raise RadTraceError("Only the final RAD action chunk may be partial")
    expected_starts = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(executed_lengths[:-1], dtype=np.int64),
        )
    )
    if not np.array_equal(canonical["replan_start_steps"], expected_starts):
        raise RadTraceError("RAD trace replan_start_steps are inconsistent")

    action_steps = int(np.sum(executed_lengths, dtype=np.int64))
    canonical.update(
        {
            "executed_actions": _array(
                arrays, "executed_actions", np.float32, (action_steps, ACTION_DIM)
            ),
            "rewards": _array(arrays, "rewards", np.float64, (action_steps,)),
            "dones": _array(arrays, "dones", np.bool_, (action_steps,)),
            "successes": _array(arrays, "successes", np.bool_, (action_steps,)),
            "sim_states_after": _array(
                arrays, "sim_states_after", np.float64, (action_steps, sim_state_dim)
            ),
            "final_image": _array(arrays, "final_image", np.uint8, IMAGE_SHAPE),
            "final_wrist_image": _array(
                arrays, "final_wrist_image", np.uint8, IMAGE_SHAPE
            ),
            "final_state": _array(arrays, "final_state", np.float32, (STATE_DIM,)),
            "final_sim_state": _array(
                arrays, "final_sim_state", np.float64, (sim_state_dim,)
            ),
        }
    )
    expected_executed = np.concatenate(
        [
            canonical["selected_actions"][index, : int(length)]
            for index, length in enumerate(executed_lengths)
        ],
        axis=0,
    )
    if not np.array_equal(canonical["executed_actions"], expected_executed):
        raise RadTraceError("RAD trace executed actions are not selected chunk prefixes")
    if not np.array_equal(canonical["final_sim_state"], canonical["sim_states_after"][-1]):
        raise RadTraceError("RAD trace final simulator state is inconsistent")
    action_offsets = np.cumsum(executed_lengths, dtype=np.int64)
    for replan_index in range(1, replan_count):
        previous_final = canonical["sim_states_after"][action_offsets[replan_index - 1] - 1]
        if not np.array_equal(
            canonical["sim_states_before"][replan_index], previous_final
        ):
            raise RadTraceError(
                "RAD trace simulator state is discontinuous before replan %d"
                % replan_index
            )
    if np.any(canonical["successes"][:-1]):
        raise RadTraceError("RAD trace must stop after its first success")

    result = metadata.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        raise RadTraceError("RAD trace result is not completed")
    expected_inferences = replan_count * candidate_count
    if (
        int(result.get("action_steps", -1)) != action_steps
        or int(result.get("replan_count", -1)) != replan_count
        or int(result.get("model_inference_calls", -1)) != expected_inferences
        or bool(result.get("success")) != bool(canonical["successes"][-1])
    ):
        raise RadTraceError("RAD trace result counts do not match its arrays")
    if np.any(canonical["candidate_inference_ms"] < 0.0):
        raise RadTraceError("RAD trace candidate inference timing must be non-negative")

    source_render = metadata.get("source_render")
    expected_frames = 1 + int(config.get("settle_steps", -1)) + action_steps
    if not isinstance(source_render, Mapping) or source_render.get("frame_count") != expected_frames:
        raise RadTraceError("RAD trace render frame count is invalid")
    frame_hashes = source_render.get("frame_sha256")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != expected_frames:
        raise RadTraceError("RAD trace render frame hashes are invalid")
    source_video = metadata.get("source_video")
    if (
        not isinstance(source_video, Mapping)
        or source_video.get("encoded") is not True
        or source_video.get("file") != "episode.mp4"
        or not _is_sha256(source_video.get("sha256"))
    ):
        raise RadTraceError("RAD trace video evidence is invalid")
    return canonical


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def save_rad_trace(
    artifact_dir: pathlib.Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    root = pathlib.Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    canonical = validate_rad_trace_arrays(metadata, arrays)
    array_path = root / RAD_TRACE_ARRAY_FILE
    temporary_array = array_path.with_name(array_path.name + ".tmp")
    with temporary_array.open("wb") as stream:
        np.savez_compressed(stream, **canonical)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary_array), str(array_path))

    array_manifest = {
        key: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": sha256_array(value),
        }
        for key, value in canonical.items()
    }
    manifest = {
        "schema": RAD_TRACE_SCHEMA,
        "array_file": RAD_TRACE_ARRAY_FILE,
        "array_file_sha256": sha256_file(array_path),
        "arrays": array_manifest,
        "metadata": dict(metadata),
        "metadata_sha256": _canonical_json_sha256(metadata),
    }
    _atomic_json(root / RAD_TRACE_MANIFEST_FILE, manifest)
    return manifest


def load_rad_trace(path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    candidate = pathlib.Path(path).expanduser().resolve()
    manifest_path = candidate / RAD_TRACE_MANIFEST_FILE if candidate.is_dir() else candidate
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RAD_TRACE_SCHEMA:
        raise RadTraceError("Unexpected RAD trace schema")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RadTraceError("RAD trace manifest metadata is invalid")
    if manifest.get("metadata_sha256") != _canonical_json_sha256(metadata):
        raise RadTraceError("RAD trace metadata SHA-256 mismatch")
    if manifest.get("array_file") != RAD_TRACE_ARRAY_FILE:
        raise RadTraceError("RAD trace array path is invalid")
    array_path = (manifest_path.parent / RAD_TRACE_ARRAY_FILE).resolve()
    try:
        array_path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise RadTraceError("RAD trace array path escapes its artifact directory") from error
    try:
        actual_array_sha256 = sha256_file(array_path)
    except OSError as error:
        raise RadTraceError("RAD trace array file is missing or unreadable") from error
    if manifest.get("array_file_sha256") != actual_array_sha256:
        raise RadTraceError("RAD trace array file SHA-256 mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        loaded = {key: np.array(archive[key], copy=True) for key in archive.files}
    canonical = validate_rad_trace_arrays(metadata, loaded)
    array_manifest = manifest.get("arrays")
    if not isinstance(array_manifest, Mapping) or set(array_manifest) != set(canonical):
        raise RadTraceError("RAD trace per-array manifest is invalid")
    for key, value in canonical.items():
        record = array_manifest[key]
        if (
            not isinstance(record, Mapping)
            or record.get("shape") != list(value.shape)
            or record.get("dtype") != value.dtype.str
            or record.get("sha256") != sha256_array(value)
        ):
            raise RadTraceError("RAD trace array %s manifest mismatch" % key)
    source_video = metadata["source_video"]
    video_relative = pathlib.Path(source_video["file"])
    if video_relative.is_absolute() or video_relative.parts != ("episode.mp4",):
        raise RadTraceError("RAD trace video path must be relative episode.mp4")
    video_path = (manifest_path.parent / video_relative).resolve()
    try:
        video_path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise RadTraceError("RAD trace video path escapes its artifact directory") from error
    try:
        if not video_path.is_file() or video_path.stat().st_size < 1:
            raise RadTraceError("RAD trace video file is missing or empty")
        actual_video_sha256 = sha256_file(video_path)
    except OSError as error:
        raise RadTraceError("RAD trace video file is missing or unreadable") from error
    if actual_video_sha256 != source_video["sha256"]:
        raise RadTraceError("RAD trace video SHA-256 mismatch")
    return dict(manifest), canonical
