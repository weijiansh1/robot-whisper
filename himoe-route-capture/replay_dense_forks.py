"""Replay saved HiMoE fork chunks into dense LIBERO physics tapes.

This tool performs no policy inference.  It reconstructs one historical fork
snapshot by replaying the recorded source-rollout prefix, then executes saved
candidate chunks from the legacy fork pilot.  The output is a physics-only
audit artifact: it contains no continuation rollout, Q estimate, value label,
or claim about behavioral equivalence.

Run this file in the LIBERO Python 3.8 environment.  No policy server is needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)

from branch_snapshot import save_full_state
from capture_behavior_forks import (
    _array_metadata,
    _atomic_write_json,
    _atomic_write_npz,
    _contact_tape,
    _enhanced_sim_layout,
    _full_state_sha256,
    _hard_restore,
    _physical_point,
    _sha256_array,
    _sha256_file,
    _utc_now,
)


SCHEMA_NAME = "himoe.dense_fork_replay"
SCHEMA_VERSION = 1
SCHEMA = "%s.v%d" % (SCHEMA_NAME, SCHEMA_VERSION)
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"

ARRAY_SCHEMA = {
    "schema_version": "int32 scalar",
    "episode": "int32 scalar",
    "fork_step": "int32 scalar",
    "candidate_ids": "int32[K], legacy candidate ids in array-axis order",
    "execution_order": "int32[K], legacy candidate ids in physical execution order",
    "actions": "float32[K,H,7], saved real HiMoE action commands",
    "sim_states": "float64[K,H+1,D]",
    "eef_positions": "float64[K,H+1,3]",
    "eef_quaternions": "float64[K,H+1,4], raw LIBERO xyzw convention",
    "gripper_qpos": "float64[K,H+1,G]",
    "chunk_success": "bool[K,H+1]",
    "contact_active": "bool[K,H+1,P], raw MuJoCo contact-pair presence",
    "contact_pair_names": "unicode[P], canonical unordered geom-name pairs",
    "snapshot_sim_state": "float64[D], reconstructed full snapshot sim component",
    "legacy_snapshot_sim_state": "float64[D], historical snap_ep*_t*.npy",
    "snapshot_sim_max_abs_error": "float64 scalar",
    "legacy_qpos_after_chunk": "float64[K,nq]",
    "legacy_qvel_after_chunk": "float64[K,nv]",
    "legacy_eef_after_chunk": "float64[K,3]",
    "legacy_gripper_after_chunk": "float64[K,G]",
    "endpoint_qpos_max_abs_error": "float64[K]",
    "endpoint_qvel_max_abs_error": "float64[K]",
    "endpoint_eef_max_abs_error": "float64[K]",
    "endpoint_gripper_max_abs_error": "float64[K]",
    "endpoint_passed": "bool[K], hard legacy agreement using qpos and qvel only",
    "fidelity_reference_candidate_id": "int32 scalar",
    "fidelity_rerun_sim_states": "float64[H+1,D]",
    "fidelity_rerun_eef_positions": "float64[H+1,3]",
    "fidelity_rerun_eef_quaternions": "float64[H+1,4]",
    "fidelity_rerun_gripper_qpos": "float64[H+1,G]",
    "fidelity_rerun_chunk_success": "bool[H+1]",
    "fidelity_rerun_contact_active": "bool[H+1,P]",
    "fidelity_sim_max_abs_error": "float64 scalar",
    "fidelity_eef_max_abs_error": "float64 scalar",
    "fidelity_quaternion_max_abs_error": "float64 scalar",
    "fidelity_gripper_max_abs_error": "float64 scalar",
    "fidelity_success_mismatch_count": "int32 scalar",
    "fidelity_contact_mismatch_count": "int32 scalar",
    "fidelity_passed": "bool scalar",
}


def _parse_candidate_ids(value: str) -> List[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("--candidates must be a comma-separated integer list") from error
    if not result:
        raise ValueError("--candidates must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("--candidates contains duplicates")
    if any(candidate < 0 for candidate in result):
        raise ValueError("--candidates must contain non-negative integers")
    return result


def _selected_candidate_ids(
    total: int,
    explicit: Optional[Sequence[int]],
    count: Optional[int],
    seed: int,
) -> np.ndarray:
    """Return a stable candidate axis, sampling without replacement if requested."""

    if total < 1:
        raise ValueError("legacy candidate pool is empty")
    if explicit is not None and count is not None:
        raise ValueError("use either --candidates or --candidate-count, not both")
    if explicit is not None:
        values = np.asarray(list(explicit), dtype=np.int64)
        if np.any(values >= total):
            raise ValueError("candidate id is outside the saved pool of size %d" % total)
        return values.astype(np.int32)
    if count is None:
        return np.arange(total, dtype=np.int32)
    if count < 1 or count > total:
        raise ValueError("--candidate-count must be in 1..%d" % total)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454C45]))
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int32)


def _execution_order(candidate_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4F524445]))
    return np.asarray(rng.permutation(candidate_ids), dtype=np.int32)


def _load_json(path: pathlib.Path, label: str) -> Any:
    if not path.is_file():
        raise RuntimeError("%s is missing: %s" % (label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("could not read %s: %s" % (label, path)) from error


def _summary_for_episode(source: pathlib.Path, episode: int) -> Tuple[Dict[str, Any], pathlib.Path]:
    path = source / "summaries.json"
    raw = _load_json(path, "source summaries")
    if not isinstance(raw, list):
        raise RuntimeError("source summaries must be a JSON list")
    matches = [item for item in raw if int(item.get("episode_index", -1)) == episode]
    if len(matches) != 1:
        raise RuntimeError("source summaries contain %d records for episode %d" % (len(matches), episode))
    summary = dict(matches[0])
    for key in ("task_id", "init_state_id", "seed", "inference_calls"):
        if key not in summary:
            raise RuntimeError("source episode summary is missing %s" % key)
    return summary, path


def _legacy_records_for_snapshot(
    path: pathlib.Path,
    episode: int,
    fork_step: int,
    pool_size: int,
) -> Dict[int, Dict[str, Any]]:
    raw = _load_json(path, "legacy fork records")
    if not isinstance(raw, list):
        raise RuntimeError("legacy fork_records.json must be a JSON list")
    selected = [
        dict(item)
        for item in raw
        if int(item.get("episode", -1)) == episode
        and int(item.get("fork_step", -1)) == fork_step
    ]
    by_id: Dict[int, Dict[str, Any]] = {}
    required = (
        "qpos_after_chunk",
        "qvel_after_chunk",
        "eef_after_chunk",
        "gripper_after_chunk",
    )
    for item in selected:
        candidate = int(item.get("candidate", -1))
        if candidate in by_id:
            raise RuntimeError("duplicate legacy endpoint for candidate %d" % candidate)
        missing = [key for key in required if key not in item]
        if missing:
            raise RuntimeError(
                "legacy endpoint for candidate %d is missing %s"
                % (candidate, ", ".join(missing))
            )
        by_id[candidate] = item
    expected = set(range(pool_size))
    actual = set(by_id)
    if actual != expected:
        raise RuntimeError(
            "legacy endpoint ids disagree with chunk axis: missing=%s extra=%s"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    return by_id


def _stack_legacy_endpoints(
    records: Mapping[int, Mapping[str, Any]], candidate_ids: np.ndarray
) -> Dict[str, np.ndarray]:
    fields = {
        "legacy_qpos_after_chunk": "qpos_after_chunk",
        "legacy_qvel_after_chunk": "qvel_after_chunk",
        "legacy_eef_after_chunk": "eef_after_chunk",
        "legacy_gripper_after_chunk": "gripper_after_chunk",
    }
    result: Dict[str, np.ndarray] = {}
    for output_key, record_key in fields.items():
        values = [np.asarray(records[int(candidate)][record_key], dtype=np.float64) for candidate in candidate_ids]
        try:
            array = np.stack(values)
        except ValueError as error:
            raise RuntimeError("legacy %s values have inconsistent shapes" % record_key) from error
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise RuntimeError("legacy %s values must be finite vectors" % record_key)
        result[output_key] = array
    return result


def _endpoint_errors(
    sim_states: np.ndarray,
    eef_positions: np.ndarray,
    gripper_qpos: np.ndarray,
    endpoints: Mapping[str, np.ndarray],
    nq: int,
    nv: int,
    atol: float,
) -> Dict[str, np.ndarray]:
    final = sim_states[:, -1]
    if final.shape[1] < 1 + nq + nv:
        raise RuntimeError("sim state is too short for qpos/qvel endpoint comparison")
    current = {
        "qpos": final[:, 1 : 1 + nq],
        "qvel": final[:, 1 + nq : 1 + nq + nv],
        "eef": eef_positions[:, -1],
        "gripper": gripper_qpos[:, -1],
    }
    legacy = {
        "qpos": endpoints["legacy_qpos_after_chunk"],
        "qvel": endpoints["legacy_qvel_after_chunk"],
        "eef": endpoints["legacy_eef_after_chunk"],
        "gripper": endpoints["legacy_gripper_after_chunk"],
    }
    errors: Dict[str, np.ndarray] = {}
    for key in current:
        if current[key].shape != legacy[key].shape:
            raise RuntimeError(
                "current and legacy %s endpoints differ in shape: %s vs %s"
                % (key, current[key].shape, legacy[key].shape)
            )
        errors["endpoint_%s_max_abs_error" % key] = np.max(
            np.abs(current[key] - legacy[key]), axis=1
        ).astype(np.float64)
    # The historical EEF/gripper values came from robosuite's sampled observable
    # cache.  They are retained as diagnostics, but only the simulator-native
    # qpos/qvel readings are a valid legacy replay gate.
    physics = np.stack(
        [errors["endpoint_qpos_max_abs_error"], errors["endpoint_qvel_max_abs_error"]]
    )
    errors["endpoint_passed"] = np.all(physics <= float(atol), axis=0).astype(np.bool_)
    return errors


def _capture_tape(
    environment: Any,
    snapshot: Mapping[str, Any],
    actions: np.ndarray,
    gripper_dim: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[bool], List[Any]]:
    observation = _hard_restore(environment, snapshot)
    initial = _physical_point(environment, observation, gripper_dim)
    sim, eef, quat, gripper, success, contacts = [[value] for value in initial]
    for action in actions:
        observation, _reward, _done, _info = environment.step(
            np.asarray(action, dtype=np.float64).tolist()
        )
        point = _physical_point(environment, observation, len(initial[3]))
        for tape, value in zip((sim, eef, quat, gripper, success, contacts), point):
            tape.append(value)
    return sim, eef, quat, gripper, success, contacts


def _stack_tapes(tapes: Sequence[Tuple[List[Any], ...]]) -> Dict[str, np.ndarray]:
    keys = ("sim_states", "eef_positions", "eef_quaternions", "gripper_qpos", "chunk_success")
    dtypes = (np.float64, np.float64, np.float64, np.float64, np.bool_)
    result: Dict[str, np.ndarray] = {}
    for index, (key, dtype) in enumerate(zip(keys, dtypes)):
        result[key] = np.asarray([tape[index] for tape in tapes], dtype=dtype)
    return result


def _validate_arrays(arrays: Mapping[str, np.ndarray], k: int, h: int) -> None:
    sim = arrays["sim_states"]
    d, g, p = sim.shape[-1], arrays["gripper_qpos"].shape[-1], len(arrays["contact_pair_names"])
    expected = {
        "candidate_ids": (k,),
        "execution_order": (k,),
        "actions": (k, h, 7),
        "sim_states": (k, h + 1, d),
        "eef_positions": (k, h + 1, 3),
        "eef_quaternions": (k, h + 1, 4),
        "gripper_qpos": (k, h + 1, g),
        "chunk_success": (k, h + 1),
        "contact_active": (k, h + 1, p),
        "fidelity_rerun_sim_states": (h + 1, d),
        "fidelity_rerun_eef_positions": (h + 1, 3),
        "fidelity_rerun_eef_quaternions": (h + 1, 4),
        "fidelity_rerun_gripper_qpos": (h + 1, g),
        "fidelity_rerun_chunk_success": (h + 1,),
        "fidelity_rerun_contact_active": (h + 1, p),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError("%s has shape %s, expected %s" % (key, arrays[key].shape, shape))
    if len(set(int(value) for value in arrays["candidate_ids"])) != k:
        raise RuntimeError("candidate_ids must be unique")
    if set(arrays["execution_order"].tolist()) != set(arrays["candidate_ids"].tolist()):
        raise RuntimeError("execution_order must be a permutation of candidate_ids")
    if not np.all(sim[:, 0] == sim[0, 0]):
        raise RuntimeError("candidate tapes do not start at one identical sim state")
    finite = (
        "actions",
        "sim_states",
        "eef_positions",
        "eef_quaternions",
        "gripper_qpos",
        "fidelity_rerun_sim_states",
        "fidelity_rerun_eef_positions",
        "fidelity_rerun_eef_quaternions",
        "fidelity_rerun_gripper_qpos",
    )
    for key in finite:
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError("%s contains NaN or infinity" % key)
    if np.any(np.linalg.norm(arrays["eef_quaternions"], axis=-1) <= 1e-12):
        raise RuntimeError("EEF tape contains a zero-norm quaternion")


def _identity_checks(source_meta: Mapping[str, Any], legacy_meta: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("protocol", "checkpoint_sha256", "libero_wrist_layout", "normalization_action_std")
    values: Dict[str, Any] = {}
    mismatched: List[str] = []
    for key in keys:
        source_value, legacy_value = source_meta.get(key), legacy_meta.get(key)
        match = source_value is None or legacy_value is None or source_value == legacy_value
        values[key] = {"source": source_value, "legacy": legacy_value, "match": match}
        if not match:
            mismatched.append(key)
    values["passed"] = not mismatched
    values["mismatched_fields"] = mismatched
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rollout-dir", required=True)
    parser.add_argument("--legacy-fork-dir", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--fork-step", required=True, type=int)
    parser.add_argument("--candidates", help="comma-separated legacy candidate ids")
    parser.add_argument("--candidate-count", type=int, help="seeded subset size; default is all")
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--snapshot-atol", type=float, default=1e-9)
    parser.add_argument("--endpoint-atol", type=float, default=1e-8)
    parser.add_argument("--fidelity-atol", type=float, default=1e-10)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.episode < 0 or args.fork_step < 0:
        raise SystemExit("--episode and --fork-step must be non-negative")
    if args.settle_steps < 0 or args.replan_steps < 1:
        raise SystemExit("--settle-steps must be non-negative and --replan-steps positive")
    if args.seed < 0 or args.seed > 0xFFFFFFFF:
        raise SystemExit("--seed must fit uint32")
    for key in ("snapshot_atol", "endpoint_atol", "fidelity_atol"):
        value = float(getattr(args, key))
        if not np.isfinite(value) or value < 0.0:
            raise SystemExit("--%s must be finite and non-negative" % key.replace("_", "-"))

    source = pathlib.Path(args.source_rollout_dir).expanduser().resolve()
    legacy = pathlib.Path(args.legacy_fork_dir).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    summary, summaries_path = _summary_for_episode(source, args.episode)
    if args.fork_step >= int(summary["inference_calls"]):
        raise SystemExit(
            "episode %d has only %d source control steps; cannot fork at %d"
            % (args.episode, int(summary["inference_calls"]), args.fork_step)
        )

    source_episode_path = source / ("episode_%02d.npz" % args.episode)
    if not source_episode_path.is_file():
        raise SystemExit("source episode is missing: %s" % source_episode_path)
    with np.load(source_episode_path, allow_pickle=False) as archive:
        source_actions = np.asarray(archive["actions"], dtype=np.float32)
    if source_actions.ndim != 3 or source_actions.shape[-1] != 7:
        raise SystemExit("source actions have invalid shape %s" % (source_actions.shape,))
    if args.fork_step > source_actions.shape[0]:
        raise SystemExit(
            "source episode stores only %d action chunks; cannot reconstruct step %d"
            % (source_actions.shape[0], args.fork_step)
        )
    if source_actions.shape[1] < args.replan_steps:
        raise SystemExit("source action chunks are shorter than --replan-steps")

    chunk_path = legacy / ("chunks_ep%02d_t%02d.npy" % (args.episode, args.fork_step))
    snapshot_path = legacy / ("snap_ep%02d_t%02d.npy" % (args.episode, args.fork_step))
    records_path = legacy / "fork_records.json"
    legacy_meta_path = legacy / "server_metadata.json"
    source_meta_path = source / "server_metadata.json"
    for path, label in (
        (chunk_path, "legacy chunks"),
        (snapshot_path, "legacy snapshot"),
        (records_path, "legacy records"),
        (legacy_meta_path, "legacy server metadata"),
        (source_meta_path, "source server metadata"),
    ):
        if not path.is_file():
            raise SystemExit("%s is missing: %s" % (label, path))
    chunks = np.asarray(np.load(chunk_path, allow_pickle=False), dtype=np.float32)
    legacy_snapshot = np.asarray(np.load(snapshot_path, allow_pickle=False), dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[-1] != 7 or not np.all(np.isfinite(chunks)):
        raise SystemExit("legacy chunks have invalid shape or values: %s" % (chunks.shape,))
    if args.replan_steps > chunks.shape[1]:
        raise SystemExit("legacy chunks are shorter than --replan-steps")
    if legacy_snapshot.ndim != 1 or not np.all(np.isfinite(legacy_snapshot)):
        raise SystemExit("legacy snapshot must be one finite vector")
    explicit = _parse_candidate_ids(args.candidates) if args.candidates is not None else None
    try:
        candidate_ids = _selected_candidate_ids(
            len(chunks), explicit, args.candidate_count, args.seed
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    actions = np.asarray(chunks[candidate_ids, : args.replan_steps], dtype=np.float32)
    legacy_records = _legacy_records_for_snapshot(
        records_path, args.episode, args.fork_step, len(chunks)
    )
    endpoints = _stack_legacy_endpoints(legacy_records, candidate_ids)
    source_meta = _load_json(source_meta_path, "source server metadata")
    legacy_meta = _load_json(legacy_meta_path, "legacy server metadata")
    identity = _identity_checks(source_meta, legacy_meta)
    if not identity["passed"]:
        raise SystemExit(
            "source rollout and legacy fork policy identities differ on %s"
            % ", ".join(identity["mismatched_fields"])
        )

    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise SystemExit("output directory must be empty: %s" % out)
    config = {
        "source_rollout_dir": str(source),
        "legacy_fork_dir": str(legacy),
        "episode": args.episode,
        "fork_step": args.fork_step,
        "candidate_ids": candidate_ids.tolist(),
        "candidate_count": len(candidate_ids),
        "benchmark": args.benchmark,
        "libero_root": str(pathlib.Path(args.libero_root).expanduser().resolve()),
        "settle_steps": args.settle_steps,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "snapshot_atol": args.snapshot_atol,
        "endpoint_atol": args.endpoint_atol,
        "fidelity_atol": args.fidelity_atol,
    }
    provenance_files = {
        "source_summaries": summaries_path,
        "source_episode": source_episode_path,
        "source_server_metadata": source_meta_path,
        "legacy_chunks": chunk_path,
        "legacy_snapshot": snapshot_path,
        "legacy_records": records_path,
        "legacy_server_metadata": legacy_meta_path,
    }
    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "complete": False,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "config": config,
        "scope": {
            "kind": "saved-action physical replay",
            "policy_server_used": False,
            "policy_inference_calls": 0,
            "continuation_executed": False,
            "q_or_value_label_present": False,
            "claim": "physics/contact capture validation only; no continuation or Q claim",
        },
        "array_schema": ARRAY_SCHEMA,
        "source_policy_identity": identity,
        "provenance": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in provenance_files.items()
        },
    }
    _atomic_write_json(out / MANIFEST_FILE, manifest)

    environment = None
    try:
        episode_config = EpisodeConfig(
            task_suite=args.benchmark,
            task_id=int(summary["task_id"]),
            init_state_id=int(summary["init_state_id"]),
            seed=int(summary["seed"]),
            libero_root=str(pathlib.Path(args.libero_root).expanduser().resolve()),
            output_root=str(out),
            settle_steps=args.settle_steps,
            max_steps=max(1000, (args.fork_step + 2) * args.replan_steps),
            replan_steps=args.replan_steps,
        )
        environment, observation, task, prompt = _load_task(episode_config)
        for _ in range(args.settle_steps):
            observation, _reward, _done, _info = environment.step(LIBERO_DUMMY_ACTION.tolist())
        for prefix_step in range(args.fork_step):
            for action in source_actions[prefix_step, : args.replan_steps]:
                observation, _reward, _done, _info = environment.step(
                    np.asarray(action, dtype=np.float64).tolist()
                )
        snapshot = save_full_state(environment)
        snapshot_sim = np.asarray(snapshot["sim"], dtype=np.float64)
        if snapshot_sim.shape != legacy_snapshot.shape:
            raise RuntimeError(
                "reconstructed and legacy snapshot shapes differ: %s vs %s"
                % (snapshot_sim.shape, legacy_snapshot.shape)
            )
        snapshot_error = float(np.max(np.abs(snapshot_sim - legacy_snapshot)))
        snapshot_exact = bool(np.array_equal(snapshot_sim, legacy_snapshot))
        snapshot_passed = bool(snapshot_error <= args.snapshot_atol)
        if not snapshot_passed:
            raise RuntimeError(
                "source prefix did not reconstruct the legacy snapshot: %.3e > %.3e"
                % (snapshot_error, args.snapshot_atol)
            )

        initial_observation = _hard_restore(environment, snapshot)
        initial_point = _physical_point(environment, initial_observation)
        layout = _enhanced_sim_layout(environment, len(initial_point[0]))
        execution_order = _execution_order(candidate_ids, args.seed)
        tape_by_id: Dict[int, Tuple[List[Any], ...]] = {}
        for candidate in execution_order.tolist():
            axis = int(np.flatnonzero(candidate_ids == candidate)[0])
            tape_by_id[candidate] = _capture_tape(
                environment, snapshot, actions[axis], len(initial_point[3])
            )
        tapes = [tape_by_id[int(candidate)] for candidate in candidate_ids]
        tape_arrays = _stack_tapes(tapes)

        reference_candidate = int(execution_order[0])
        reference_axis = int(np.flatnonzero(candidate_ids == reference_candidate)[0])
        rerun = _capture_tape(
            environment, snapshot, actions[reference_axis], len(initial_point[3])
        )
        combined_contacts, contact_names, raw_pairs = _contact_tape(
            [tape[5] for tape in tapes] + [rerun[5]]
        )
        tape_arrays["contact_active"] = combined_contacts[:-1]

        fidelity_values = {
            "sim": np.asarray(rerun[0], dtype=np.float64),
            "eef": np.asarray(rerun[1], dtype=np.float64),
            "quaternion": np.asarray(rerun[2], dtype=np.float64),
            "gripper": np.asarray(rerun[3], dtype=np.float64),
            "success": np.asarray(rerun[4], dtype=np.bool_),
            "contact": combined_contacts[-1],
        }
        reference_values = {
            "sim": tape_arrays["sim_states"][reference_axis],
            "eef": tape_arrays["eef_positions"][reference_axis],
            "quaternion": tape_arrays["eef_quaternions"][reference_axis],
            "gripper": tape_arrays["gripper_qpos"][reference_axis],
            "success": tape_arrays["chunk_success"][reference_axis],
            "contact": tape_arrays["contact_active"][reference_axis],
        }
        fidelity_errors = {
            key: float(np.max(np.abs(fidelity_values[key] - reference_values[key])))
            for key in ("sim", "eef", "quaternion", "gripper")
        }
        success_mismatch = int(np.count_nonzero(fidelity_values["success"] != reference_values["success"]))
        contact_mismatch = int(np.count_nonzero(fidelity_values["contact"] != reference_values["contact"]))
        fidelity_passed = bool(
            all(error <= args.fidelity_atol for error in fidelity_errors.values())
            and success_mismatch == 0
            and contact_mismatch == 0
        )

        endpoint_errors = _endpoint_errors(
            tape_arrays["sim_states"],
            tape_arrays["eef_positions"],
            tape_arrays["gripper_qpos"],
            endpoints,
            int(layout["nq"]),
            int(layout["nv"]),
            args.endpoint_atol,
        )
        arrays: Dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "episode": np.asarray(args.episode, dtype=np.int32),
            "fork_step": np.asarray(args.fork_step, dtype=np.int32),
            "candidate_ids": candidate_ids,
            "execution_order": execution_order,
            "actions": actions,
            **tape_arrays,
            "contact_pair_names": contact_names,
            "snapshot_sim_state": snapshot_sim,
            "legacy_snapshot_sim_state": legacy_snapshot,
            "snapshot_sim_max_abs_error": np.asarray(snapshot_error, dtype=np.float64),
            **endpoints,
            **endpoint_errors,
            "fidelity_reference_candidate_id": np.asarray(reference_candidate, dtype=np.int32),
            "fidelity_rerun_sim_states": fidelity_values["sim"],
            "fidelity_rerun_eef_positions": fidelity_values["eef"],
            "fidelity_rerun_eef_quaternions": fidelity_values["quaternion"],
            "fidelity_rerun_gripper_qpos": fidelity_values["gripper"],
            "fidelity_rerun_chunk_success": fidelity_values["success"],
            "fidelity_rerun_contact_active": fidelity_values["contact"],
            "fidelity_sim_max_abs_error": np.asarray(fidelity_errors["sim"], dtype=np.float64),
            "fidelity_eef_max_abs_error": np.asarray(fidelity_errors["eef"], dtype=np.float64),
            "fidelity_quaternion_max_abs_error": np.asarray(fidelity_errors["quaternion"], dtype=np.float64),
            "fidelity_gripper_max_abs_error": np.asarray(fidelity_errors["gripper"], dtype=np.float64),
            "fidelity_success_mismatch_count": np.asarray(success_mismatch, dtype=np.int32),
            "fidelity_contact_mismatch_count": np.asarray(contact_mismatch, dtype=np.int32),
            "fidelity_passed": np.asarray(fidelity_passed, dtype=np.bool_),
        }
        _validate_arrays(arrays, len(candidate_ids), args.replan_steps)

        layout.update(
            {
                "schema": SCHEMA,
                "episode": args.episode,
                "fork_step": args.fork_step,
                "task_id": int(summary["task_id"]),
                "init_state_id": int(summary["init_state_id"]),
                "task_name": str(task.name),
                "prompt": prompt,
                "candidate_axis_semantics": "candidate_ids supplies the legacy id for every K-axis row",
                "contact_pair_encoding": {
                    "kind": "canonical unordered raw MuJoCo geom-name pair set",
                    "delimiter": " <-> ",
                    "multiplicity": "collapsed to presence/absence at each time point",
                    "pair_names": [[left, right] for left, right in raw_pairs],
                },
                "restore": "hard: sim.reset then restore_full_state for every branch and fidelity rerun",
                "physical_readout_audit": {
                    "implementation": "capture_behavior_forks._physical_point",
                    "canonical_sources": "physical_tape_sources in this layout",
                    "legacy_endpoint_note": (
                        "historical eef_after_chunk and gripper_after_chunk came from "
                        "env.step observable dictionaries; their errors are diagnostic only"
                    ),
                },
                "fidelity": {
                    "kind": "full H+1 physical tape rerun",
                    "reference_candidate_id": reference_candidate,
                    "continuous_atol": args.fidelity_atol,
                    "continuous_max_abs_errors": fidelity_errors,
                    "success_mismatch_count": success_mismatch,
                    "contact_mismatch_count": contact_mismatch,
                    "passed": fidelity_passed,
                },
                "arrays": _array_metadata(arrays),
            }
        )
        endpoint_maxima = {
            key: float(np.max(value))
            for key, value in endpoint_errors.items()
            if key.endswith("max_abs_error")
        }
        endpoint_core_passed = bool(np.all(endpoint_errors["endpoint_passed"]))
        hard_validation_passed = bool(endpoint_core_passed and fidelity_passed)
        result_summary = {
            "schema": SCHEMA,
            "status": "complete" if hard_validation_passed else "validation_failed",
            "episode": args.episode,
            "fork_step": args.fork_step,
            "task_id": int(summary["task_id"]),
            "init_state_id": int(summary["init_state_id"]),
            "task_name": str(task.name),
            "candidate_ids": candidate_ids.tolist(),
            "candidate_count": len(candidate_ids),
            "horizon": args.replan_steps,
            "snapshot_reconstruction": {
                "array_equal": snapshot_exact,
                "max_abs_error": snapshot_error,
                "atol": args.snapshot_atol,
                "passed": snapshot_passed,
                "reconstructed_full_state_sha256": _full_state_sha256(snapshot),
                "reconstructed_sim_sha256": _sha256_array(snapshot_sim),
                "legacy_sim_sha256": _sha256_array(legacy_snapshot),
            },
            "legacy_endpoint_comparison": {
                "atol": args.endpoint_atol,
                "passed_candidates": int(np.count_nonzero(endpoint_errors["endpoint_passed"])),
                "total_candidates": len(candidate_ids),
                "all_passed": endpoint_core_passed,
                "hard_gate_fields": ["qpos_after_chunk", "qvel_after_chunk"],
                "diagnostic_only_fields": [
                    "eef_after_chunk (historical cached observable)",
                    "gripper_after_chunk (historical cached observable)",
                ],
                "max_abs_errors": endpoint_maxima,
            },
            "fidelity_rerun": {
                "reference_candidate_id": reference_candidate,
                "continuous_atol": args.fidelity_atol,
                "continuous_max_abs_errors": fidelity_errors,
                "success_mismatch_count": success_mismatch,
                "contact_mismatch_count": contact_mismatch,
                "passed": fidelity_passed,
            },
            "contacts": {
                "pair_count": len(contact_names),
                "active_cells": int(np.count_nonzero(tape_arrays["contact_active"])),
                "candidates_with_any_contact": int(
                    np.count_nonzero(np.any(tape_arrays["contact_active"], axis=(1, 2)))
                ),
            },
            "scope": manifest["scope"],
        }
        stem = "dense_replay_ep%04d_t%04d" % (args.episode, args.fork_step)
        npz_path = out / (stem + ".npz")
        layout_path = out / (stem + ".layout.json")
        summary_path = out / SUMMARY_FILE
        _atomic_write_npz(npz_path, arrays)
        _atomic_write_json(layout_path, layout)
        _atomic_write_json(summary_path, result_summary)
        manifest.update(
            {
                "status": result_summary["status"],
                "complete": hard_validation_passed,
                "updated_utc": _utc_now(),
                "completed_utc": _utc_now(),
                "artifact": {
                    "npz_file": npz_path.name,
                    "npz_sha256": _sha256_file(npz_path),
                    "layout_file": layout_path.name,
                    "layout_sha256": _sha256_file(layout_path),
                    "summary_file": summary_path.name,
                    "summary_sha256": _sha256_file(summary_path),
                    "arrays": _array_metadata(arrays),
                },
                "validation": {
                    "snapshot_reconstruction_passed": snapshot_passed,
                    "legacy_endpoint_qpos_qvel_all_passed": endpoint_core_passed,
                    "full_tape_fidelity_passed": fidelity_passed,
                },
            }
        )
        _atomic_write_json(out / MANIFEST_FILE, manifest)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["complete"] = False
        manifest["updated_utc"] = _utc_now()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
        _atomic_write_json(out / MANIFEST_FILE, manifest)
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except BaseException:
                pass

    print(
        "replayed %d saved candidates at ep %d step %d | contacts %d | endpoint %d/%d | fidelity %s | %s"
        % (
            len(candidate_ids),
            args.episode,
            args.fork_step,
            result_summary["contacts"]["active_cells"],
            result_summary["legacy_endpoint_comparison"]["passed_candidates"],
            len(candidate_ids),
            fidelity_passed,
            out,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
