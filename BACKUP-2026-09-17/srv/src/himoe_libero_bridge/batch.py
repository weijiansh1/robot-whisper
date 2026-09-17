"""Recoverable, auditable orchestration for sequential LIBERO episodes."""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime
import fcntl
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from himoe_libero_bridge.libero_runtime import EpisodeConfig, run_episode
from himoe_libero_bridge.suites import get_suite


BATCH_SCHEMA = "himoe-libero-batch-v1"
BATCH_CONFIG_FILE = "batch-config.json"
BATCH_MANIFEST_FILE = "manifest.json"
BATCH_SUMMARY_FILE = "summary.json"
BATCH_CSV_FILE = "episodes.csv"
BATCH_EVENTS_FILE = "events.jsonl"
VIDEO_POLICIES = ("all", "success", "failure", "none")
SUITE_MAX_STEPS = {"goal": 300, "spatial": 220, "object": 280, "long": 520}
LIBERO_TASKS_PER_SUITE = 10
LIBERO_INIT_STATES_PER_TASK = 50
WILSON_95_Z = 1.959963984540054
SUCCESS_RATE_DENOMINATOR_DEFINITION = (
    "completed episodes (succeeded + task_failed); infra_failed excluded"
)
_SOURCE_EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
}


class BatchError(RuntimeError):
    """Raised when a batch definition or persisted batch is invalid."""


@dataclasses.dataclass(frozen=True)
class BatchConfig:
    libero_root: str
    output_root: str = "artifacts"
    host: str = "127.0.0.1"
    port: int = 8000
    suite: str = "goal"
    task_ids: Tuple[int, ...] = (0,)
    init_state_ids: Tuple[int, ...] = (0,)
    episodes_per_task: int = 1
    start_seed: int = 7
    flow_noise_seed_start: Optional[int] = None
    settle_steps: int = 10
    max_steps: int = 300
    replan_steps: int = 10
    render_size: int = 256
    fps: int = 20
    inference_timeout: float = 180.0
    video_policy: str = "all"


EpisodeRunner = Callable[[EpisodeConfig], pathlib.Path]


def parse_id_spec(value: str, *, allow_all: bool = False) -> Tuple[int, ...]:
    """Parse a comma-separated list of integer IDs and inclusive ranges."""

    text = value.strip().lower()
    if allow_all and text == "all":
        return tuple(range(10))
    if not text:
        raise ValueError("ID selection must not be empty")
    result = []  # type: List[int]
    seen = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError("ID selection contains an empty item: %r" % value)
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise ValueError("Invalid inclusive ID range: %r" % token)
            start, end = (int(piece) for piece in pieces)
            if end < start:
                raise ValueError("ID range ends before it starts: %r" % token)
            values = range(start, end + 1)
        else:
            if not token.isdigit():
                raise ValueError("Invalid ID: %r" % token)
            values = (int(token),)
        for item in values:
            if item in seen:
                raise ValueError("Duplicate ID %d in selection %r" % (item, value))
            seen.add(item)
            result.append(item)
    return tuple(result)


def validate_batch_config(config: BatchConfig) -> None:
    get_suite(config.suite)
    if not config.task_ids:
        raise ValueError("task_ids must not be empty")
    if not config.init_state_ids:
        raise ValueError("init_state_ids must not be empty")
    if len(set(config.task_ids)) != len(config.task_ids):
        raise ValueError("task_ids must be unique")
    if len(set(config.init_state_ids)) != len(config.init_state_ids):
        raise ValueError("init_state_ids must be unique")
    if any(not 0 <= task_id < LIBERO_TASKS_PER_SUITE for task_id in config.task_ids):
        raise ValueError("task_ids must be between 0 and 9")
    if any(
        not 0 <= init_state_id < LIBERO_INIT_STATES_PER_TASK
        for init_state_id in config.init_state_ids
    ):
        raise ValueError("init_state_ids must be between 0 and 49")
    if config.episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if config.start_seed < 0:
        raise ValueError("start_seed must be non-negative")
    if config.flow_noise_seed_start is not None and config.flow_noise_seed_start < 0:
        raise ValueError("flow_noise_seed_start must be non-negative")
    if config.settle_steps < 0 or config.max_steps < 1:
        raise ValueError("Invalid settle/max step limits")
    if not 1 <= config.replan_steps <= 10:
        raise ValueError("replan_steps must be between 1 and 10")
    if config.render_size < 1 or config.fps < 1 or config.inference_timeout <= 0:
        raise ValueError("Invalid render, FPS, or inference timeout setting")
    if not 1 <= config.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if config.video_policy not in VIDEO_POLICIES:
        raise ValueError("video_policy must be one of %s" % (", ".join(VIDEO_POLICIES)))


def build_job_plan(config: BatchConfig) -> List[Dict[str, Any]]:
    """Build the deterministic task-local seed and initial-state schedule."""

    validate_batch_config(config)
    jobs = []  # type: List[Dict[str, Any]]
    order = 0
    for task_id in config.task_ids:
        for episode_index in range(config.episodes_per_task):
            init_state_id = config.init_state_ids[
                episode_index % len(config.init_state_ids)
            ]
            seed = config.start_seed
            replica_index = episode_index // len(config.init_state_ids)
            flow_noise_seed = (
                None
                if config.flow_noise_seed_start is None
                else config.flow_noise_seed_start
                + LIBERO_INIT_STATES_PER_TASK * task_id
                + init_state_id
                + (
                    LIBERO_TASKS_PER_SUITE
                    * LIBERO_INIT_STATES_PER_TASK
                    * replica_index
                )
            )
            jobs.append(
                {
                    "order": order,
                    "job_id": "task%02d-episode%03d-init%03d-seed%d"
                    % (task_id, episode_index, init_state_id, seed),
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "init_state_id": init_state_id,
                    "seed": seed,
                    "flow_noise_seed": flow_noise_seed,
                }
            )
            order += 1
    return jobs


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def wilson_interval(successes: int, denominator: int) -> Dict[str, Any]:
    """Return the two-sided 95% Wilson score interval for a binomial rate."""

    if denominator < 0 or successes < 0 or successes > denominator:
        raise ValueError("Invalid Wilson interval counts")
    if denominator == 0:
        lower = None
        upper = None
    else:
        rate = successes / denominator
        z_squared = WILSON_95_Z * WILSON_95_Z
        scale = 1.0 + z_squared / denominator
        center = (rate + z_squared / (2.0 * denominator)) / scale
        radius = (
            WILSON_95_Z
            * math.sqrt(
                rate * (1.0 - rate) / denominator
                + z_squared / (4.0 * denominator * denominator)
            )
            / scale
        )
        lower = max(0.0, center - radius)
        upper = min(1.0, center + radius)
    return {
        "method": "wilson-score",
        "confidence_level": 0.95,
        "successes": successes,
        "denominator": denominator,
        "lower": lower,
        "upper": upper,
    }


def _git_output(root: pathlib.Path, arguments: Sequence[str]) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root)] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _fallback_source_files(root: pathlib.Path) -> List[pathlib.Path]:
    files = []
    for directory, names, filenames in os.walk(str(root)):
        names[:] = sorted(
            name for name in names if name not in _SOURCE_EXCLUDED_DIRECTORIES
        )
        directory_path = pathlib.Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.is_file() or path.is_symlink():
                files.append(path)
    return files


def _source_tree_digest(root: pathlib.Path) -> Tuple[str, int]:
    listed = _git_output(
        root, ("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    )
    if listed is None:
        paths = _fallback_source_files(root)
    else:
        paths = [
            root / os.fsdecode(item)
            for item in listed.split(b"\0")
            if item
        ]
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            marker = b"symlink"
            payload = os.readlink(str(path)).encode("utf-8")
        elif path.is_file():
            marker = b"file"
            payload = path.read_bytes()
        else:
            continue
        digest.update(marker + b"\0")
        digest.update(str(len(relative)).encode("ascii") + b"\0" + relative)
        digest.update(str(len(payload)).encode("ascii") + b"\0" + payload)
        file_count += 1
    return digest.hexdigest(), file_count


def collect_source_identity(project_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Identify the exact bridge source without depending on an existing commit."""

    root = (
        pathlib.Path(__file__).resolve().parents[2]
        if project_root is None
        else pathlib.Path(project_root).expanduser().resolve()
    )
    git_root_output = _git_output(root, ("rev-parse", "--show-toplevel"))
    if git_root_output:
        root = pathlib.Path(os.fsdecode(git_root_output).strip()).resolve()
    commit_output = _git_output(root, ("rev-parse", "--verify", "HEAD"))
    tree_sha256, file_count = _source_tree_digest(root)
    if commit_output:
        status_output = _git_output(
            root, ("status", "--porcelain=v1", "--untracked-files=all")
        )
        dirty = bool(status_output and status_output.strip())
        identity = {
            "kind": "git",
            "commit": os.fsdecode(commit_output).strip(),
            "dirty": dirty,
        }  # type: Dict[str, Any]
        if dirty:
            identity.update(
                {
                    "source_tree_sha256": tree_sha256,
                    "source_file_count": file_count,
                    "tree_hash_algorithm": "sha256-path-length-content-v1",
                }
            )
        return identity
    return {
        "kind": "source-tree",
        "commit": None,
        "dirty": None,
        "source_tree_sha256": tree_sha256,
        "source_file_count": file_count,
        "tree_hash_algorithm": "sha256-path-length-content-v1",
    }


def _config_dict(config: BatchConfig) -> Dict[str, Any]:
    value = dataclasses.asdict(config)
    value["task_ids"] = list(config.task_ids)
    value["init_state_ids"] = list(config.init_state_ids)
    return value


def _config_from_dict(value: Mapping[str, Any]) -> BatchConfig:
    expected = {field.name for field in dataclasses.fields(BatchConfig)}
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise BatchError("Invalid batch config keys; missing=%s extra=%s" % (missing, extra))
    config_value = dict(value)
    config_value["task_ids"] = tuple(config_value["task_ids"])
    config_value["init_state_ids"] = tuple(config_value["init_state_ids"])
    config = BatchConfig(**config_value)
    validate_batch_config(config)
    return config


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _append_event(path: pathlib.Path, event: str, **values: Any) -> None:
    record = {"event": event, "time_utc": _utc_now()}
    record.update(values)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _batch_lock(batch_dir: pathlib.Path) -> Iterable[None]:
    lock_path = batch_dir / ".batch.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchError("Batch is already active: %s" % batch_dir) from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_utc": _utc_now()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _new_batch_dir(output_root: str, suite: str) -> pathlib.Path:
    root = pathlib.Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / ("batch-libero-%s-%s" % (suite, timestamp))
    path.mkdir(parents=False, exist_ok=False)
    return path


def _resolve_batch_dir(path: str) -> pathlib.Path:
    candidate = pathlib.Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if not (candidate / BATCH_CONFIG_FILE).is_file():
        raise BatchError("Not a batch artifact directory: %s" % candidate)
    return candidate


def _within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _episode_config(config: BatchConfig, job: Mapping[str, Any], job_dir: pathlib.Path) -> EpisodeConfig:
    return EpisodeConfig(
        libero_root=config.libero_root,
        output_root=str(job_dir),
        host=config.host,
        port=config.port,
        task_suite=get_suite(config.suite).benchmark,
        task_id=int(job["task_id"]),
        init_state_id=int(job["init_state_id"]),
        seed=int(job["seed"]),
        settle_steps=config.settle_steps,
        max_steps=config.max_steps,
        replan_steps=config.replan_steps,
        render_size=config.render_size,
        fps=config.fps,
        inference_timeout=config.inference_timeout,
        flow_noise_seed=job["flow_noise_seed"],
    )


def _read_episode_summary(artifact_dir: pathlib.Path) -> Dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise BatchError("Episode runner did not produce summary.json: %s" % artifact_dir)
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError("Could not read episode summary %s: %s" % (summary_path, error)) from error
    if not isinstance(value, dict):
        raise BatchError("Episode summary is not a JSON object: %s" % summary_path)
    return value


def _summary_paths(job_dir: pathlib.Path) -> List[pathlib.Path]:
    return sorted(
        (path for path in job_dir.glob("*/summary.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def _episode_result(artifact_dir: pathlib.Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    video = summary.get("video")
    success = summary.get("success")
    if summary.get("status") == "completed" and not isinstance(success, bool):
        raise BatchError("Completed episode summary must contain a boolean success value")
    return {
        "episode_status": summary.get("status"),
        "success": success if isinstance(success, bool) else None,
        "task_name": summary.get("task_name"),
        "prompt": summary.get("prompt"),
        "action_steps": summary.get("action_steps"),
        "inference_calls": summary.get("inference_calls"),
        "duration_seconds": summary.get("duration_seconds"),
        "artifact_dir": str(artifact_dir),
        "video_path": video if isinstance(video, str) else None,
        "video_retained": bool(isinstance(video, str) and pathlib.Path(video).is_file()),
        "error_type": summary.get("error_type"),
        "error": summary.get("error"),
    }


def _should_keep_video(policy: str, state: str, success: Optional[bool]) -> bool:
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "success":
        return state == "completed" and success is True
    return state != "completed" or success is not True


def _apply_video_policy(
    batch_dir: pathlib.Path,
    result: Dict[str, Any],
    state: str,
    policy: str,
) -> Optional[str]:
    video_value = result.get("video_path")
    if not isinstance(video_value, str):
        result["video_retained"] = False
        return None
    video_path = pathlib.Path(video_value).expanduser().resolve()
    artifact_value = result.get("artifact_dir")
    if not isinstance(artifact_value, str):
        return "Episode result has a video but no artifact directory"
    artifact_dir = pathlib.Path(artifact_value).resolve()
    if not _within(artifact_dir, batch_dir) or not _within(video_path, artifact_dir):
        return "Refusing to manage video outside this batch: %s" % video_path
    keep = _should_keep_video(policy, state, result.get("success"))
    try:
        if not keep and video_path.is_file():
            video_path.unlink()
        result["video_retained"] = video_path.is_file()
    except OSError as error:
        return "Could not apply video policy to %s: %s" % (video_path, error)
    return None


def _job_outcome(job: Mapping[str, Any]) -> str:
    state = job.get("state")
    if state in ("pending", "running"):
        return str(state)
    if state == "failed":
        return "infra_failed"
    if state == "completed":
        success = (job.get("result") or {}).get("success")
        if success is True:
            return "succeeded"
        if success is False:
            return "task_failed"
    return "infra_failed"


def _aggregate(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    jobs = manifest["jobs"]
    outcomes = [_job_outcome(job) for job in jobs]
    outcome_counts = {
        outcome: outcomes.count(outcome)
        for outcome in (
            "succeeded",
            "task_failed",
            "infra_failed",
            "pending",
            "running",
        )
    }
    completed = outcome_counts["succeeded"] + outcome_counts["task_failed"]
    per_task = {}  # type: Dict[str, Dict[str, Any]]
    for job, outcome in zip(jobs, outcomes):
        key = str(job["task_id"])
        task = per_task.setdefault(
            key,
            {
                "planned": 0,
                "completed": 0,
                "succeeded": 0,
                "task_failed": 0,
                "infra_failed": 0,
                "pending": 0,
                "running": 0,
            },
        )
        task["planned"] += 1
        task[outcome] += 1
        if outcome in ("succeeded", "task_failed"):
            task["completed"] += 1
    for task in per_task.values():
        task["success_rate_numerator"] = task["succeeded"]
        task["success_rate_denominator"] = task["completed"]
        task["success_rate_denominator_definition"] = (
            SUCCESS_RATE_DENOMINATOR_DEFINITION
        )
        task["success_rate"] = (
            task["succeeded"] / task["completed"] if task["completed"] else None
        )
        task["success_rate_wilson_95"] = wilson_interval(
            task["succeeded"], task["completed"]
        )
        task["successes"] = task["succeeded"]
        task["failed"] = task["infra_failed"]
    unfinished = outcome_counts["pending"] + outcome_counts["running"]
    status = (
        "paused"
        if unfinished
        else "completed_with_errors"
        if outcome_counts["infra_failed"]
        else "completed"
    )
    summary = {
        "schema": BATCH_SCHEMA,
        "batch_id": manifest["batch_id"],
        "suite": manifest["config"]["suite"],
        "config_sha256": manifest["config_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "status": status,
        "planned_episodes": len(jobs),
        "completed_episodes": completed,
        "succeeded_episodes": outcome_counts["succeeded"],
        "task_failed_episodes": outcome_counts["task_failed"],
        "infra_failed_episodes": outcome_counts["infra_failed"],
        "pending_episodes": outcome_counts["pending"],
        "running_episodes": outcome_counts["running"],
        "unfinished_episodes": unfinished,
        "episode_counts": {
            "planned": len(jobs),
            "completed": completed,
            **outcome_counts,
        },
        "success_rate_numerator": outcome_counts["succeeded"],
        "success_rate_denominator": completed,
        "success_rate_denominator_definition": SUCCESS_RATE_DENOMINATOR_DEFINITION,
        "success_rate": (
            outcome_counts["succeeded"] / completed if completed else None
        ),
        "success_rate_wilson_95": wilson_interval(
            outcome_counts["succeeded"], completed
        ),
        "failed_episodes": outcome_counts["infra_failed"],
        "task_successes": outcome_counts["succeeded"],
        "task_success_rate": (
            outcome_counts["succeeded"] / completed if completed else None
        ),
        "per_task": per_task,
        "updated_utc": _utc_now(),
    }
    if "source_identity" in manifest:
        summary["source_identity"] = manifest["source_identity"]
        summary["source_identity_sha256"] = manifest.get("source_identity_sha256")
    return summary


_CSV_FIELDS = (
    "order",
    "job_id",
    "suite",
    "task_id",
    "episode_index",
    "init_state_id",
    "seed",
    "flow_noise_seed",
    "state",
    "outcome",
    "is_completed",
    "is_succeeded",
    "is_task_failed",
    "is_infra_failed",
    "is_pending",
    "is_running",
    "attempt_count",
    "episode_status",
    "success",
    "action_steps",
    "inference_calls",
    "duration_seconds",
    "artifact_dir",
    "video_path",
    "video_retained",
    "error_type",
    "error",
)


def _write_csv(path: pathlib.Path, suite: str, jobs: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for job in jobs:
            result = job.get("result") or {}
            outcome = _job_outcome(job)
            writer.writerow(
                {
                    "order": job["order"],
                    "job_id": job["job_id"],
                    "suite": suite,
                    "task_id": job["task_id"],
                    "episode_index": job["episode_index"],
                    "init_state_id": job["init_state_id"],
                    "seed": job["seed"],
                    "flow_noise_seed": job["flow_noise_seed"],
                    "state": job["state"],
                    "outcome": outcome,
                    "is_completed": int(outcome in ("succeeded", "task_failed")),
                    "is_succeeded": int(outcome == "succeeded"),
                    "is_task_failed": int(outcome == "task_failed"),
                    "is_infra_failed": int(outcome == "infra_failed"),
                    "is_pending": int(outcome == "pending"),
                    "is_running": int(outcome == "running"),
                    "attempt_count": len(job["attempts"]),
                    **{field: result.get(field) for field in _CSV_FIELDS if field in result},
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _persist(batch_dir: pathlib.Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    manifest["updated_utc"] = _utc_now()
    summary = _aggregate(manifest)
    _atomic_json(batch_dir / BATCH_MANIFEST_FILE, manifest)
    _atomic_json(batch_dir / BATCH_SUMMARY_FILE, summary)
    _write_csv(batch_dir / BATCH_CSV_FILE, manifest["config"]["suite"], manifest["jobs"])
    return summary


def _load_batch(batch_dir: pathlib.Path) -> Tuple[BatchConfig, Dict[str, Any]]:
    try:
        immutable = json.loads((batch_dir / BATCH_CONFIG_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((batch_dir / BATCH_MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError("Could not load batch metadata: %s" % error) from error
    if immutable.get("schema") != BATCH_SCHEMA or manifest.get("schema") != BATCH_SCHEMA:
        raise BatchError("Unsupported batch schema")
    for key in ("batch_id", "config", "config_sha256", "plan_sha256"):
        if immutable.get(key) != manifest.get(key):
            raise BatchError("Batch manifest disagrees with immutable metadata for %s" % key)
    immutable_source = immutable.get("source_identity")
    manifest_source = manifest.get("source_identity")
    if immutable_source is not None or manifest_source is not None:
        if immutable_source != manifest_source:
            raise BatchError(
                "Batch manifest disagrees with immutable metadata for source_identity"
            )
        source_sha256 = immutable.get("source_identity_sha256")
        if (
            not isinstance(immutable_source, Mapping)
            or not isinstance(source_sha256, str)
            or source_sha256 != _sha256_json(immutable_source)
            or manifest.get("source_identity_sha256") != source_sha256
        ):
            raise BatchError("Batch source identity hash mismatch")
    if immutable["config_sha256"] != _sha256_json(immutable["config"]):
        raise BatchError("Batch config hash mismatch")
    plan = immutable.get("plan")
    if not isinstance(plan, list) or immutable["plan_sha256"] != _sha256_json(plan):
        raise BatchError("Batch plan hash mismatch")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise BatchError("Batch manifest jobs are invalid")
    persisted_plan = [
        {
            key: job.get(key)
            for key in (
                "order",
                "job_id",
                "task_id",
                "episode_index",
                "init_state_id",
                "seed",
                "flow_noise_seed",
            )
        }
        for job in jobs
    ]
    if persisted_plan != plan:
        raise BatchError("Batch job plan was modified")
    config = _config_from_dict(immutable["config"])
    return config, manifest


def _recover_jobs(batch_dir: pathlib.Path, manifest: Dict[str, Any]) -> None:
    for job in manifest["jobs"]:
        if job["state"] not in ("pending", "running") or not job["attempts"]:
            continue
        job_dir = batch_dir / "episodes" / job["job_id"]
        summaries = _summary_paths(job_dir) if job_dir.is_dir() else []
        attempt = job["attempts"][-1]
        known_summary_files = set(attempt.get("known_summary_files", []))
        new_summaries = [
            summary
            for summary in summaries
            if str(summary.relative_to(job_dir)) not in known_summary_files
        ]
        if new_summaries:
            artifact_dir = new_summaries[-1].parent
            episode_summary = _read_episode_summary(artifact_dir)
            result = _episode_result(artifact_dir, episode_summary)
            state = "completed" if episode_summary.get("status") == "completed" else "failed"
            job["state"] = state
            job["result"] = result
            attempt.update(
                {
                    "state": state,
                    "artifact_dir": str(artifact_dir),
                    "finished_utc": attempt.get("finished_utc") or _utc_now(),
                    "recovered": True,
                }
            )
        elif job["state"] == "running":
            attempt.update(
                {
                    "state": "interrupted",
                    "finished_utc": _utc_now(),
                    "error_type": "InterruptedAttempt",
                    "error": "No terminal episode summary was found during resume",
                    "recovered": True,
                }
            )
            job["state"] = "pending"
            job["result"] = None


def _create_batch(config: BatchConfig) -> Tuple[pathlib.Path, Dict[str, Any]]:
    normalized = dataclasses.replace(
        config,
        libero_root=str(pathlib.Path(config.libero_root).expanduser().resolve()),
        output_root=str(pathlib.Path(config.output_root).expanduser().resolve()),
    )
    validate_batch_config(normalized)
    plan = build_job_plan(normalized)
    source_identity = collect_source_identity()
    source_identity_sha256 = _sha256_json(source_identity)
    batch_dir = _new_batch_dir(normalized.output_root, normalized.suite)
    created = _utc_now()
    config_value = _config_dict(normalized)
    immutable = {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_dir.name,
        "created_utc": created,
        "config": config_value,
        "config_sha256": _sha256_json(config_value),
        "plan": plan,
        "plan_sha256": _sha256_json(plan),
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
    }
    jobs = []
    for planned in plan:
        job = dict(planned)
        job.update({"state": "pending", "attempts": [], "result": None})
        jobs.append(job)
    manifest = {
        key: immutable[key]
        for key in (
            "schema",
            "batch_id",
            "created_utc",
            "config",
            "config_sha256",
            "plan_sha256",
            "source_identity",
            "source_identity_sha256",
        )
    }
    manifest.update({"updated_utc": created, "invocations": [], "jobs": jobs})
    _atomic_json(batch_dir / BATCH_CONFIG_FILE, immutable)
    (batch_dir / "episodes").mkdir()
    _persist(batch_dir, manifest)
    return batch_dir, manifest


def _run_jobs(
    batch_dir: pathlib.Path,
    config: BatchConfig,
    manifest: Dict[str, Any],
    *,
    episode_runner: EpisodeRunner,
    retry_failed: bool,
    max_new_episodes: Optional[int],
) -> pathlib.Path:
    if max_new_episodes is not None and max_new_episodes < 1:
        raise ValueError("max_new_episodes must be positive")
    _recover_jobs(batch_dir, manifest)
    for job in manifest["jobs"]:
        if job["state"] in ("completed", "failed") and isinstance(job.get("result"), dict):
            retention_error = _apply_video_policy(
                batch_dir, job["result"], job["state"], config.video_policy
            )
            if retention_error:
                job["result"]["video_retention_error"] = retention_error
    if retry_failed:
        for job in manifest["jobs"]:
            if job["state"] == "failed":
                job["state"] = "pending"

    invocation = {
        "started_utc": _utc_now(),
        "pid": os.getpid(),
        "host": config.host,
        "port": config.port,
        "inference_timeout": config.inference_timeout,
        "retry_failed": retry_failed,
        "max_new_episodes": max_new_episodes,
    }
    manifest["invocations"].append(invocation)
    _append_event(batch_dir / BATCH_EVENTS_FILE, "batch_invocation_started", **invocation)
    _persist(batch_dir, manifest)

    jobs_started = 0
    for job in manifest["jobs"]:
        if job["state"] != "pending":
            continue
        if max_new_episodes is not None and jobs_started >= max_new_episodes:
            break
        jobs_started += 1
        job_dir = batch_dir / "episodes" / job["job_id"]
        job_dir.mkdir(parents=False, exist_ok=True)
        _atomic_json(
            job_dir / "job.json",
            {
                "schema": BATCH_SCHEMA,
                "batch_id": manifest["batch_id"],
                "suite": config.suite,
                "source_identity_sha256": manifest.get("source_identity_sha256"),
                **{
                    key: job[key]
                    for key in (
                        "order",
                        "job_id",
                        "task_id",
                        "episode_index",
                        "init_state_id",
                        "seed",
                        "flow_noise_seed",
                    )
                },
            },
        )
        known_summaries = set(_summary_paths(job_dir))
        attempt = {
            "attempt": len(job["attempts"]) + 1,
            "state": "running",
            "started_utc": _utc_now(),
            "finished_utc": None,
            "artifact_dir": None,
            "known_summary_files": [
                str(path.relative_to(job_dir)) for path in sorted(known_summaries)
            ],
        }
        job["attempts"].append(attempt)
        job["state"] = "running"
        job["result"] = None
        _append_event(
            batch_dir / BATCH_EVENTS_FILE,
            "episode_started",
            job_id=job["job_id"],
            attempt=attempt["attempt"],
        )
        _persist(batch_dir, manifest)
        started = time.perf_counter()
        artifact_dir = None  # type: Optional[pathlib.Path]
        try:
            returned = pathlib.Path(
                episode_runner(_episode_config(config, job, job_dir))
            ).expanduser().resolve()
            if not _within(returned, job_dir):
                raise BatchError("Episode runner returned an artifact outside its job directory")
            artifact_dir = returned
            episode_summary = _read_episode_summary(artifact_dir)
            if episode_summary.get("status") != "completed":
                raise BatchError(
                    "Episode runtime status is %r" % episode_summary.get("status")
                )
            result = _episode_result(artifact_dir, episode_summary)
            job["state"] = "completed"
            job["result"] = result
            attempt.update(
                {
                    "state": "completed",
                    "finished_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "artifact_dir": str(artifact_dir),
                }
            )
            retention_error = _apply_video_policy(
                batch_dir, result, job["state"], config.video_policy
            )
            if retention_error:
                result["video_retention_error"] = retention_error
            _append_event(
                batch_dir / BATCH_EVENTS_FILE,
                "episode_completed",
                job_id=job["job_id"],
                attempt=attempt["attempt"],
                success=result["success"],
                artifact_dir=str(artifact_dir),
                video_retained=result["video_retained"],
            )
        except Exception as error:
            if artifact_dir is None:
                new_summaries = [
                    path for path in _summary_paths(job_dir) if path not in known_summaries
                ]
                if new_summaries:
                    artifact_dir = new_summaries[-1].parent
            if artifact_dir is not None:
                try:
                    episode_summary = _read_episode_summary(artifact_dir)
                    result = _episode_result(artifact_dir, episode_summary)
                except Exception:
                    result = {"artifact_dir": str(artifact_dir)}
            else:
                result = {"artifact_dir": str(job_dir)}
            result.update(
                {
                    "episode_status": result.get("episode_status") or "failed",
                    "success": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            job["state"] = "failed"
            job["result"] = result
            attempt.update(
                {
                    "state": "failed",
                    "finished_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "artifact_dir": str(artifact_dir) if artifact_dir else None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            retention_error = _apply_video_policy(
                batch_dir, result, job["state"], config.video_policy
            )
            if retention_error:
                result["video_retention_error"] = retention_error
            _atomic_json(
                job_dir / ("attempt-%03d-error.json" % attempt["attempt"]),
                {"job_id": job["job_id"], **attempt},
            )
            _append_event(
                batch_dir / BATCH_EVENTS_FILE,
                "episode_failed",
                job_id=job["job_id"],
                attempt=attempt["attempt"],
                error_type=type(error).__name__,
                error=str(error),
                artifact_dir=str(artifact_dir) if artifact_dir else None,
            )
        _persist(batch_dir, manifest)

    invocation["finished_utc"] = _utc_now()
    invocation["jobs_started"] = jobs_started
    summary = _persist(batch_dir, manifest)
    _append_event(
        batch_dir / BATCH_EVENTS_FILE,
        "batch_invocation_finished",
        status=summary["status"],
        jobs_started=jobs_started,
        completed_episodes=summary["completed_episodes"],
        failed_episodes=summary["failed_episodes"],
        pending_episodes=summary["pending_episodes"],
    )
    return batch_dir


def run_batch(
    config: BatchConfig,
    *,
    episode_runner: EpisodeRunner = run_episode,
    max_new_episodes: Optional[int] = None,
) -> pathlib.Path:
    """Create and execute a new deterministic batch."""

    batch_dir, manifest = _create_batch(config)
    normalized_config = _config_from_dict(manifest["config"])
    with _batch_lock(batch_dir):
        return _run_jobs(
            batch_dir,
            normalized_config,
            manifest,
            episode_runner=episode_runner,
            retry_failed=False,
            max_new_episodes=max_new_episodes,
        )


def resume_batch(
    path: str,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    inference_timeout: Optional[float] = None,
    retry_failed: bool = False,
    max_new_episodes: Optional[int] = None,
    episode_runner: EpisodeRunner = run_episode,
) -> pathlib.Path:
    """Resume pending work, optionally retrying prior infrastructure failures."""

    batch_dir = _resolve_batch_dir(path)
    with _batch_lock(batch_dir):
        config, manifest = _load_batch(batch_dir)
        effective = dataclasses.replace(
            config,
            host=config.host if host is None else host,
            port=config.port if port is None else port,
            inference_timeout=(
                config.inference_timeout if inference_timeout is None else inference_timeout
            ),
        )
        validate_batch_config(effective)
        return _run_jobs(
            batch_dir,
            effective,
            manifest,
            episode_runner=episode_runner,
            retry_failed=retry_failed,
            max_new_episodes=max_new_episodes,
        )


def load_batch_summary(path: str) -> Dict[str, Any]:
    batch_dir = _resolve_batch_dir(path)
    try:
        value = json.loads((batch_dir / BATCH_SUMMARY_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError("Could not load batch summary: %s" % error) from error
    if not isinstance(value, dict):
        raise BatchError("Batch summary is not a JSON object")
    return value
