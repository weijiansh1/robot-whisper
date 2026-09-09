"""Versioned, staged artifacts for the HiMoE behavior-geometry study."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from branch_snapshot import decode_full_state, encode_full_state


ROOT_SEED = 20260824
SCHEMA_VERSION = 2
PLAN_SCHEMA = "himoe.behavior_study.plan.v2"
EVENT_SCHEMA = "himoe.behavior_study.source_events.v2"
CANDIDATE_SCHEMA = "himoe.behavior_forks.candidates.v2"
SHARD_SCHEMA = "himoe.behavior_forks.continuation_shard.v2"
ASSEMBLY_SCHEMA = "himoe.behavior_forks.assembly.v2"
POOL_KINDS = ("screen", "formal")
LABEL_COHORTS = ("screen", "main", "enriched")
SHARD_REPEATS = 8
FORMAL_TARGETS = (48, 96)

SEED_DOMAINS = {
    "plan_episode": "plan/episode-selection",
    "audit_inclusion": "audit/inclusion",
    "screen_candidate": "screen/candidate",
    "screen_continuation": "screen/continuation",
    "formal_candidate": "formal/candidate",
    "formal_continuation": "formal/continuation",
    "execution": "execution",
}

EVENT_LABELS = (
    "raw_contact",
    "contact_onset",
    "contact_release",
    "gripper_closing",
    "gripper_opening",
    "object_motion",
    "success",
    "success_onset",
)


class ArtifactError(RuntimeError):
    """A staged artifact is incomplete, corrupt, or violates the v2 contract."""


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype.hasobject:
            raise ArtifactError(f"object array is forbidden: {name}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def domain_low32(domain: str) -> int:
    if not isinstance(domain, str) or not domain:
        raise ValueError("seed domain must be a non-empty string")
    return int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[-4:], "big")


def seed_words(
    domain: str,
    task: int,
    episode: int,
    fork_step: int,
    coordinates: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return the preregistered SeedSequence entropy vector."""

    words = (ROOT_SEED, domain_low32(domain), task, episode, fork_step, *coordinates)
    if any(not isinstance(value, (int, np.integer)) for value in words):
        raise ValueError("seed coordinates must be integers")
    result = tuple(int(value) for value in words)
    if any(value < 0 or value > 0xFFFFFFFF for value in result):
        raise ValueError(f"seed coordinate does not fit uint32: {result}")
    return result


def rng_from_words(words: Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(value) for value in words]))


def flow_noise(words: Sequence[int], shape: tuple[int, ...] = (10, 24)) -> np.ndarray:
    return rng_from_words(words).standard_normal(shape).astype(np.float32)


def pool_domain(pool: str, role: str) -> str:
    if pool not in POOL_KINDS:
        raise ValueError(f"unknown pool {pool!r}")
    key = f"{pool}_{role}"
    try:
        return SEED_DOMAINS[key]
    except KeyError as error:
        raise ValueError(f"unknown seed role {role!r}") from error


def state_id(task: int, episode: int, fork_step: int) -> str:
    return f"task{task:02d}-episode{episode:04d}-step{fork_step:04d}"


def candidate_query_id(snapshot_index: int, pool: str, candidate: int) -> int:
    if pool not in POOL_KINDS:
        raise ValueError(f"unknown pool {pool!r}")
    if not 0 <= snapshot_index < 18 or not 0 <= candidate < 32:
        raise ValueError("candidate query coordinates are outside the micro-pilot frame")
    return (0 if pool == "screen" else 18 * 32) + snapshot_index * 32 + candidate


def continuation_query_id(
    snapshot_index: int,
    cohort: str,
    candidate: int,
    repeat: int,
    future_step: int,
    continuation_steps: int,
) -> int:
    cohort_code = {"screen": 0, "main": 1, "enriched": 2}.get(cohort)
    if cohort_code is None:
        raise ValueError(f"unknown label cohort {cohort!r}")
    if not 0 <= snapshot_index < 18 or not 0 <= candidate < 32:
        raise ValueError("continuation query coordinates are outside the micro-pilot frame")
    if repeat < 0 or future_step < 0 or future_step >= continuation_steps:
        raise ValueError("invalid continuation repeat/future_step")
    value = (
        10_000
        + cohort_code * 10_000_000
        + snapshot_index * 500_000
        + candidate * 10_000
        + repeat * continuation_steps
        + future_step
    )
    if value > np.iinfo(np.int32).max:
        raise ValueError("continuation query id does not fit int32")
    return value


def candidate_dir(root: Path, pool: str, identifier: str) -> Path:
    if pool not in POOL_KINDS:
        raise ValueError(f"unknown pool {pool!r}")
    return root / pool / identifier / "candidate"


def cohort_candidate_pool(cohort: str) -> str:
    if cohort == "main":
        return "formal"
    if cohort in {"screen", "enriched"}:
        return "screen"
    raise ValueError(f"unknown label cohort {cohort!r}")


def cohort_seed_pool(cohort: str) -> str:
    if cohort == "screen":
        return "screen"
    if cohort in {"main", "enriched"}:
        return "formal"
    raise ValueError(f"unknown label cohort {cohort!r}")


def shard_dir(root: Path, pool: str, identifier: str, cohort: str, shard_index: int) -> Path:
    if pool != cohort_candidate_pool(cohort):
        raise ValueError(f"cohort {cohort!r} cannot label candidate pool {pool!r}")
    return root / pool / identifier / "continuations" / cohort / f"shard_{shard_index:03d}"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def commit_artifact(
    directory: Path,
    schema: str,
    metadata: Mapping[str, Any],
    files: Mapping[str, Path],
) -> Path:
    """Atomically publish an artifact only after all payload files are durable."""

    directory.mkdir(parents=True, exist_ok=True)
    descriptor = directory / "artifact.json"
    if descriptor.exists():
        raise ArtifactError(f"artifact is already committed: {descriptor}")
    payload_files = {}
    for role, path in files.items():
        resolved = path.resolve()
        if resolved.parent != directory.resolve() or not resolved.is_file():
            raise ArtifactError(f"artifact payload must exist inside its directory: {path}")
        payload_files[role] = {"file": path.name, "sha256": sha256_file(path)}
    payload = {
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": utc_now(),
        **json_safe(metadata),
        "files": payload_files,
    }
    atomic_json(descriptor, payload)
    return descriptor


def verify_artifact(path: Path, expected_schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise ArtifactError(f"unexpected artifact schema in {path}: {value.get('schema')!r}")
    if value.get("status") != "complete" or not isinstance(value.get("files"), dict):
        raise ArtifactError(f"artifact is not complete: {path}")
    for record in value["files"].values():
        payload = path.parent / str(record["file"])
        if not payload.is_file() or sha256_file(payload) != record.get("sha256"):
            raise ArtifactError(f"artifact payload checksum mismatch: {payload}")
    return value


def write_full_state_bundle(path_prefix: Path, snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    metadata, arrays = encode_full_state({"snapshots": list(snapshots)})
    json_path = path_prefix.with_suffix(".json")
    npz_path = path_prefix.with_suffix(".npz")
    atomic_npz(npz_path, arrays)
    atomic_json(json_path, metadata)
    return {"full_states_meta": json_path, "full_states_arrays": npz_path}


def read_full_state_bundle(meta_path: Path, arrays_path: Path) -> list[dict[str, Any]]:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    result = decode_full_state(metadata, load_npz(arrays_path)).get("snapshots")
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise ArtifactError("full-state bundle root must be a list of snapshots")
    return result


def dense_event_tape(
    contact_active: np.ndarray,
    gripper_qpos: np.ndarray,
    sim_states: np.ndarray,
    success: np.ndarray,
    nq: int,
    object_qpos_indices: Sequence[int],
    motion_threshold: float = 1e-4,
) -> np.ndarray:
    """Derive conservative dense primitive events without claiming task semantics."""

    contact = np.asarray(contact_active, dtype=np.bool_)
    gripper = np.asarray(gripper_qpos, dtype=np.float64)
    sim = np.asarray(sim_states, dtype=np.float64)
    succeeded = np.asarray(success, dtype=np.bool_)
    if contact.ndim != 3:
        raise ValueError("contact_active must have shape [K,T,P]")
    k, steps = contact.shape[:2]
    if gripper.ndim != 3 or gripper.shape[:2] != (k, steps):
        raise ValueError("gripper_qpos must have shape [K,T,G]")
    if sim.ndim != 3 or sim.shape[:2] != (k, steps) or sim.shape[-1] < 1 + nq:
        raise ValueError("sim_states must have shape [K,T,D]")
    if succeeded.shape != (k, steps):
        raise ValueError("success must have shape [K,T]")
    active = contact.any(axis=-1)
    onset = np.zeros_like(active)
    release = np.zeros_like(active)
    onset[:, 1:] = ~active[:, :-1] & active[:, 1:]
    release[:, 1:] = active[:, :-1] & ~active[:, 1:]
    aperture = gripper.mean(axis=-1)
    closing = np.zeros_like(active)
    opening = np.zeros_like(active)
    closing[:, 1:] = np.diff(aperture, axis=1) < -1e-6
    opening[:, 1:] = np.diff(aperture, axis=1) > 1e-6
    object_motion = np.zeros_like(active)
    indices = np.asarray(object_qpos_indices, dtype=np.int64)
    if len(indices):
        if np.any(indices < 0) or np.any(indices >= nq):
            raise ValueError("object_qpos_indices fall outside qpos")
        qpos = sim[..., 1 : 1 + nq][..., indices]
        object_motion[:, 1:] = np.linalg.norm(np.diff(qpos, axis=1), axis=-1) >= motion_threshold
    success_onset = np.zeros_like(succeeded)
    success_onset[:, 0] = succeeded[:, 0]
    success_onset[:, 1:] = ~succeeded[:, :-1] & succeeded[:, 1:]
    return np.stack(
        (active, onset, release, closing, opening, object_motion, succeeded, success_onset),
        axis=-1,
    )


def validate_candidate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    actions = np.asarray(arrays["actions"])
    if actions.ndim != 3 or actions.shape[-1] != 7 or actions.shape[0] < 2:
        raise ArtifactError("actions must have shape [K,H,7], K>=2")
    k, h = actions.shape[:2]
    sim = np.asarray(arrays["sim_states"])
    if sim.ndim != 3 or sim.shape[:2] != (k, h + 1):
        raise ArtifactError("sim_states must have shape [K,H+1,D]")
    d = sim.shape[-1]
    expected = {
        "candidate_ids": (k,),
        "candidate_flow_noise": (k, 10, 24),
        "query_ids": (k,),
        "server_trace_rows": (k,),
        "store_ids": (k,),
        "eef_positions": (k, h + 1, 3),
        "eef_quaternions": (k, h + 1, 4),
        "chunk_success": (k, h + 1),
        "event_flags": (k, h + 1, len(EVENT_LABELS)),
        "fidelity_rerun_sim_states": (h + 1, d),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ArtifactError(f"{name} has shape {np.asarray(arrays[name]).shape}, expected {shape}")
    contacts = np.asarray(arrays["contact_active"])
    if contacts.ndim != 3 or contacts.shape[:2] != (k, h + 1):
        raise ArtifactError("contact_active must have shape [K,H+1,P]")
    if np.asarray(arrays["contact_pair_names"]).shape != (contacts.shape[-1],):
        raise ArtifactError("contact_pair_names is not aligned with contact_active")
    if list(np.asarray(arrays["event_labels"]).astype(str)) != list(EVENT_LABELS):
        raise ArtifactError("event_labels do not match the v2 primitive event schema")
    if not np.array_equal(np.asarray(arrays["candidate_ids"]), np.arange(k)):
        raise ArtifactError("candidate_ids must match the candidate axis")
    if not bool(np.asarray(arrays["fidelity_passed"]).item()):
        raise ArtifactError("candidate artifact failed fidelity")
    if metadata.get("pool") not in POOL_KINDS:
        raise ArtifactError("candidate artifact has no explicit screen/formal pool")
    _require_finite(arrays)


def validate_shard_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], candidate_count: int
) -> None:
    success = np.asarray(arrays["continuation_success"])
    if success.shape != (candidate_count, SHARD_REPEATS):
        raise ArtifactError(
            f"continuation_success must have shape [{candidate_count},{SHARD_REPEATS}]"
        )
    k, repeats = success.shape
    noise = np.asarray(arrays["continuation_flow_noise"])
    if noise.ndim != 4 or noise.shape[0] != repeats or noise.shape[-2:] != (10, 24):
        raise ArtifactError("continuation_flow_noise must have shape [8,B,10,24]")
    b = noise.shape[1]
    steps = np.asarray(arrays["continuation_action_steps"])
    final = np.asarray(arrays["continuation_final_sim_states"])
    query_ids = np.asarray(arrays["query_ids"])
    rows = np.asarray(arrays["server_trace_rows"])
    executed = np.asarray(arrays["query_executed"], dtype=np.bool_)
    if steps.shape != (k, repeats) or final.shape[:2] != (k, repeats):
        raise ArtifactError("continuation result arrays have inconsistent K/R axes")
    if query_ids.shape != (k, repeats, b) or rows.shape != query_ids.shape or executed.shape != query_ids.shape:
        raise ArtifactError("continuation query arrays must have shape [K,8,B]")
    if np.any(query_ids[executed] < 0) or np.any(query_ids[~executed] != -1):
        raise ArtifactError("continuation query id sentinels are inconsistent")
    if np.any(rows != -1):
        raise ArtifactError("continuation shards must not capture route rows")
    first_repeat = int(metadata.get("repeat_start", -1))
    if first_repeat < 0 or first_repeat % SHARD_REPEATS:
        raise ArtifactError("continuation shard repeat_start must be 8-aligned")
    if metadata.get("pool") not in POOL_KINDS:
        raise ArtifactError("continuation shard has no explicit pool")
    cohort = str(metadata.get("label_cohort", ""))
    if cohort not in LABEL_COHORTS or cohort_candidate_pool(cohort) != metadata.get("pool"):
        raise ArtifactError("continuation shard has an invalid label cohort/pool mapping")
    if metadata.get("seed_pool") != cohort_seed_pool(cohort):
        raise ArtifactError("continuation shard seed_pool disagrees with its label cohort")
    _require_finite(arrays)


def _require_finite(arrays: Mapping[str, np.ndarray]) -> None:
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ArtifactError(f"{name} contains NaN or infinity")


def seed_records_from_candidate(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> list[tuple[int, ...]]:
    task = int(metadata["task_id"])
    episode = int(metadata["episode"])
    fork_step = int(metadata["fork_step"])
    domain = pool_domain(str(metadata["pool"]), "candidate")
    return [seed_words(domain, task, episode, fork_step, (int(candidate),)) for candidate in arrays["candidate_ids"]]


def seed_records_from_shard(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> list[tuple[int, ...]]:
    task = int(metadata["task_id"])
    episode = int(metadata["episode"])
    fork_step = int(metadata["fork_step"])
    domain = pool_domain(str(metadata["seed_pool"]), "continuation")
    repeat_start = int(metadata["repeat_start"])
    b = np.asarray(arrays["continuation_flow_noise"]).shape[1]
    return [
        seed_words(domain, task, episode, fork_step, (repeat, future_step))
        for repeat in range(repeat_start, repeat_start + SHARD_REPEATS)
        for future_step in range(b)
    ]


def assert_no_seed_overlap(records: Sequence[tuple[int, ...]]) -> None:
    seen: set[tuple[int, ...]] = set()
    for words in records:
        key = tuple(int(value) for value in words)
        if key in seen:
            raise ArtifactError(f"duplicate SeedSequence entropy vector: {key}")
        seen.add(key)
