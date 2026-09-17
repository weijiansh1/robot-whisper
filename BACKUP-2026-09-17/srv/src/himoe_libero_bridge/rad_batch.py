"""Recoverable multi-case orchestration for paired four-arm RAD blocks."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import fcntl
import hashlib
import json
import math
import os
import pathlib
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from himoe_libero_bridge.batch import collect_source_identity
from himoe_libero_bridge.rad_experiment import (
    PAIRED_SELECTORS,
    summarize_paired_success,
)
from himoe_libero_bridge.rad_runtime import (
    RAD_BLOCK_MANIFEST_FILE,
    RAD_BLOCK_SCHEMA,
    RAD_BLOCK_SUMMARY_FILE,
    RadBlockConfig,
    collect_rad_experiment_identity,
    run_rad_block,
)


RAD_BATCH_SCHEMA = "himoe-libero-rad-batch-v2"
RAD_BATCH_CONFIG_FILE = "rad-batch-config.json"
RAD_BATCH_MANIFEST_FILE = "rad-batch-manifest.json"
RAD_BATCH_EVENTS_FILE = "rad-batch-events.jsonl"
RAD_BATCH_SUMMARY_FILE = "rad-batch-summary.json"
RAD_BATCH_TASK_COUNT = 10
RAD_BATCH_INIT_STATE_COUNT = 50
RAD_BATCH_SEED_DERIVATION = "sha256-namespace-task-init-u63-v1"

_NOISE_NAMESPACE = "flow-noise"
_RANDOM_SELECTOR_NAMESPACE = "random-selector"
_ARM_ORDER_NAMESPACE = "arm-order"
_PAIRED_BOOTSTRAP_NAMESPACE = "paired-bootstrap"


class RadBatchError(RuntimeError):
    """Raised when a RAD batch definition or persisted artifact is invalid."""


class RadBatchIncompleteError(RadBatchError):
    """Raised when a returned block is not eligible for paired statistics."""


@dataclasses.dataclass(frozen=True)
class RadBatchConfig:
    batch_dir: str
    libero_root: str
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite: str = "libero_goal"
    task_ids: Tuple[int, ...] = (0,)
    init_state_ids: Tuple[int, ...] = (0,)
    env_seed: int = 7
    master_seed: int = 42
    settle_steps: int = 10
    max_steps: int = 300
    replan_steps: int = 10
    render_size: int = 256
    fps: int = 20
    inference_timeout: float = 180.0
    bootstrap_samples: int = 10_000


RadBlockRunner = Callable[[RadBlockConfig], pathlib.Path]


def _default_rad_block_runner(config: RadBlockConfig) -> pathlib.Path:
    return run_rad_block(config)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def derive_rad_case_seed(
    master_seed: int,
    namespace: str,
    task_id: int,
    init_state_id: int,
) -> int:
    """Derive an order- and batch-size-invariant non-negative seed."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if not 0 <= task_id < RAD_BATCH_TASK_COUNT:
        raise ValueError("task_id must be between 0 and 9")
    if not 0 <= init_state_id < RAD_BATCH_INIT_STATE_COUNT:
        raise ValueError("init_state_id must be between 0 and 49")
    payload = {
        "derivation": RAD_BATCH_SEED_DERIVATION,
        "master_seed": master_seed,
        "namespace": namespace,
        "task_id": task_id,
        "init_state_id": init_state_id,
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(payload)).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _derive_global_seed(master_seed: int, namespace: str) -> int:
    payload = {
        "derivation": RAD_BATCH_SEED_DERIVATION,
        "master_seed": master_seed,
        "namespace": namespace,
        "scope": "batch",
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(payload)).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def validate_rad_batch_config(config: RadBatchConfig) -> None:
    if config.task_suite not in ("libero_goal", "libero_spatial", "libero_object"):
        raise ValueError("task_suite must be a supported LIBERO benchmark")
    if not config.task_ids or len(set(config.task_ids)) != len(config.task_ids):
        raise ValueError("task_ids must be non-empty and unique")
    if not config.init_state_ids or len(set(config.init_state_ids)) != len(
        config.init_state_ids
    ):
        raise ValueError("init_state_ids must be non-empty and unique")
    if any(not 0 <= value < RAD_BATCH_TASK_COUNT for value in config.task_ids):
        raise ValueError("task_ids must be between 0 and 9")
    if any(
        not 0 <= value < RAD_BATCH_INIT_STATE_COUNT
        for value in config.init_state_ids
    ):
        raise ValueError("init_state_ids must be between 0 and 49")
    for value, name in (
        (config.env_seed, "env_seed"),
        (config.master_seed, "master_seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("%s must be a non-negative integer" % name)
    if not 1 <= config.port <= 65535 or not config.host:
        raise ValueError("host/port are invalid")
    if config.settle_steps < 0 or config.max_steps < 1:
        raise ValueError("RAD batch step limits are invalid")
    if not 1 <= config.replan_steps <= 10:
        raise ValueError("replan_steps must be between 1 and 10")
    if config.render_size < 1 or config.fps < 1 or config.inference_timeout <= 0:
        raise ValueError("RAD batch render/timing settings are invalid")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")


def build_rad_case_plan(config: RadBatchConfig) -> List[Dict[str, Any]]:
    """Build the deterministic Cartesian task-by-initial-state plan."""

    validate_rad_batch_config(config)
    plan = []  # type: List[Dict[str, Any]]
    order = 0
    for task_id in config.task_ids:
        for init_state_id in config.init_state_ids:
            case_id = "%s/task%02d/init%03d/env%d" % (
                config.task_suite,
                task_id,
                init_state_id,
                config.env_seed,
            )
            plan.append(
                {
                    "order": order,
                    "case_id": case_id,
                    "case_slug": "task%02d-init%03d" % (task_id, init_state_id),
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "env_seed": config.env_seed,
                    "noise_seed": derive_rad_case_seed(
                        config.master_seed,
                        _NOISE_NAMESPACE,
                        task_id,
                        init_state_id,
                    ),
                    "random_selector_seed": derive_rad_case_seed(
                        config.master_seed,
                        _RANDOM_SELECTOR_NAMESPACE,
                        task_id,
                        init_state_id,
                    ),
                    "arm_order_seed": derive_rad_case_seed(
                        config.master_seed,
                        _ARM_ORDER_NAMESPACE,
                        task_id,
                        init_state_id,
                    ),
                }
            )
            order += 1
    return plan


def _config_dict(config: RadBatchConfig) -> Dict[str, Any]:
    value = dataclasses.asdict(config)
    value["task_ids"] = list(config.task_ids)
    value["init_state_ids"] = list(config.init_state_ids)
    return value


def _config_from_dict(value: Mapping[str, Any]) -> RadBatchConfig:
    expected = {field.name for field in dataclasses.fields(RadBatchConfig)}
    if set(value) != expected:
        raise RadBatchError("RAD batch config fields do not match this implementation")
    canonical = dict(value)
    canonical["task_ids"] = tuple(canonical["task_ids"])
    canonical["init_state_ids"] = tuple(canonical["init_state_ids"])
    config = RadBatchConfig(**canonical)
    validate_rad_batch_config(config)
    return config


def _atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _append_event(path: pathlib.Path, event: str, **values: Any) -> None:
    record = {"event": event, "time_utc": _utc_now()}
    record.update(values)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _rad_batch_lock(batch_dir: pathlib.Path) -> Iterable[None]:
    with (batch_dir / ".rad-batch.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RadBatchError("RAD batch is already active: %s" % batch_dir) from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_utc": _utc_now()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RadBatchError("Could not read %s: %s" % (path, error)) from error
    if not isinstance(value, dict):
        raise RadBatchError("Expected a JSON object in %s" % path)
    return value


def _arm_evidence(block_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    mappings = []  # type: List[Mapping[str, Any]]
    for filename in (RAD_BLOCK_SUMMARY_FILE, RAD_BLOCK_MANIFEST_FILE):
        path = block_dir / filename
        if not path.is_file():
            continue
        value = _read_json(path)
        arms = value.get("arms")
        if isinstance(arms, Mapping):
            mappings.append(arms)
    records = {}  # type: Dict[str, Dict[str, Any]]
    for selector in PAIRED_SELECTORS:
        arm = None
        for mapping in mappings:
            candidate = mapping.get(selector)
            if isinstance(candidate, Mapping) and candidate.get("artifact_dir"):
                arm = candidate
                break
        if arm is None:
            continue
        artifact_dir = pathlib.Path(str(arm["artifact_dir"])).expanduser().resolve()
        if not _within(artifact_dir, block_dir):
            continue
        summary_path = artifact_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        calls = summary.get("model_inference_calls")
        duration = summary.get("duration_seconds")
        if (
            summary.get("selector") != selector
            or isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            continue
        records[selector] = {
            "selector": selector,
            "artifact_dir": str(artifact_dir),
            "episode_status": summary.get("status"),
            "success": summary.get("success")
            if isinstance(summary.get("success"), bool)
            else None,
            "model_inference_calls": calls,
            "duration_seconds": float(duration),
        }
    return records


def _validate_completed_block(block_dir: pathlib.Path) -> Dict[str, Any]:
    summary_path = block_dir / RAD_BLOCK_SUMMARY_FILE
    if not summary_path.is_file():
        raise RadBatchIncompleteError("RAD block did not produce its summary")
    summary = _read_json(summary_path)
    if summary.get("schema") != RAD_BLOCK_SCHEMA or summary.get("status") != "completed":
        raise RadBatchIncompleteError("RAD block summary is not completed")
    if summary.get("fairness_exact") is not True:
        raise RadBatchIncompleteError("RAD block fairness_exact is not true")
    fairness_identity = summary.get("fairness_identity")
    if not isinstance(fairness_identity, Mapping):
        raise RadBatchIncompleteError("RAD block fairness identity is missing")
    if not isinstance(fairness_identity.get("shared_all_arms"), Mapping):
        raise RadBatchIncompleteError("RAD block shared-arm fairness identity is missing")
    if not isinstance(fairness_identity.get("initial_k8_candidate_pool"), Mapping):
        raise RadBatchIncompleteError("RAD block K=8 pool fairness identity is missing")
    arms = summary.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(PAIRED_SELECTORS):
        raise RadBatchIncompleteError("RAD block does not contain exactly four arms")
    evidence = _arm_evidence(block_dir)
    if set(evidence) != set(PAIRED_SELECTORS):
        raise RadBatchIncompleteError("RAD block arm cost summaries are incomplete")
    for selector in PAIRED_SELECTORS:
        arm = arms[selector]
        if (
            not isinstance(arm, Mapping)
            or arm.get("status") != "completed"
            or not isinstance(arm.get("success"), bool)
            or evidence[selector]["episode_status"] != "completed"
            or evidence[selector]["success"] != arm["success"]
        ):
            raise RadBatchIncompleteError("RAD block arm %s is incomplete" % selector)
    return {
        "block_dir": str(block_dir),
        "fairness_exact": True,
        "fairness_identity": dict(fairness_identity),
        "arms": evidence,
    }


def _has_explicit_fairness_failure(block_dir: pathlib.Path) -> bool:
    summary_path = block_dir / RAD_BLOCK_SUMMARY_FILE
    if summary_path.is_file():
        try:
            summary = _read_json(summary_path)
        except RadBatchError:
            summary = {}
        if summary.get("status") == "completed" and summary.get("fairness_exact") is not True:
            return True
    manifest_path = block_dir / RAD_BLOCK_MANIFEST_FILE
    if manifest_path.is_file():
        try:
            manifest = _read_json(manifest_path)
        except RadBatchError:
            manifest = {}
        mismatches = manifest.get("fairness_mismatches")
        if (isinstance(mismatches, list) and mismatches) or (
            isinstance(mismatches, Mapping)
            and any(bool(value) for value in mismatches.values())
        ):
            return True
    return False


def _case_plan_fields(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "order",
            "case_id",
            "case_slug",
            "task_id",
            "init_state_id",
            "env_seed",
            "noise_seed",
            "random_selector_seed",
            "arm_order_seed",
        )
    }


def _block_config(
    config: RadBatchConfig, case: Mapping[str, Any], block_dir: pathlib.Path
) -> RadBlockConfig:
    return RadBlockConfig(
        block_dir=str(block_dir),
        libero_root=config.libero_root,
        host=config.host,
        port=config.port,
        task_suite=config.task_suite,
        task_id=int(case["task_id"]),
        init_state_id=int(case["init_state_id"]),
        seed=config.env_seed,
        settle_steps=config.settle_steps,
        max_steps=config.max_steps,
        replan_steps=config.replan_steps,
        render_size=config.render_size,
        fps=config.fps,
        inference_timeout=config.inference_timeout,
        noise_seed=int(case["noise_seed"]),
        random_selector_seed=int(case["random_selector_seed"]),
        arm_order_seed=int(case["arm_order_seed"]),
    )


def _recover_stale_cases(batch_dir: pathlib.Path, manifest: Dict[str, Any]) -> None:
    for case in manifest["cases"]:
        if case["state"] == "completed":
            try:
                case["result"] = _validate_completed_block(
                    batch_dir / "cases" / case["case_slug"]
                )
            except (OSError, ValueError, RadBatchError) as error:
                case["state"] = "incomplete"
                case["error_type"] = type(error).__name__
                case["error"] = str(error)
            continue
        if case["state"] != "running":
            continue
        block_dir = batch_dir / "cases" / case["case_slug"]
        try:
            result = _validate_completed_block(block_dir)
        except (OSError, ValueError, RadBatchError):
            attempt = case["attempts"][-1]
            attempt.update(
                {
                    "state": "interrupted",
                    "finished_utc": _utc_now(),
                    "arm_evidence": list(_arm_evidence(block_dir).values()),
                }
            )
            case["state"] = "pending"
            case["result"] = None
        else:
            case["state"] = "completed"
            case["result"] = result
            case["attempts"][-1].update(
                {
                    "state": "completed",
                    "finished_utc": _utc_now(),
                    "recovered": True,
                    "arm_evidence": list(result["arms"].values()),
                }
            )


def _cost_summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    unique = {}  # type: Dict[Tuple[str, str], Mapping[str, Any]]
    for case in cases:
        for attempt in case.get("attempts", []):
            for arm in attempt.get("arm_evidence", []):
                if isinstance(arm, Mapping):
                    key = (str(arm.get("selector")), str(arm.get("artifact_dir")))
                    unique[key] = arm
        result = case.get("result")
        if isinstance(result, Mapping):
            for arm in result.get("arms", {}).values():
                if isinstance(arm, Mapping):
                    key = (str(arm.get("selector")), str(arm.get("artifact_dir")))
                    unique[key] = arm
    costs = {}
    for selector in PAIRED_SELECTORS:
        records = [
            value for (name, _), value in unique.items() if name == selector
        ]
        calls = sum(int(record["model_inference_calls"]) for record in records)
        duration = sum(float(record["duration_seconds"]) for record in records)
        costs[selector] = {
            "arm_episode_attempts": len(records),
            "completed_arm_episodes": sum(
                record.get("episode_status") == "completed" for record in records
            ),
            "failed_arm_episodes": sum(
                record.get("episode_status") != "completed" for record in records
            ),
            "model_inference_calls_total": calls,
            "duration_seconds_total": duration,
            "model_inference_calls_mean": calls / len(records) if records else None,
            "duration_seconds_mean": duration / len(records) if records else None,
        }
    return costs


def _aggregate(
    config: RadBatchConfig,
    manifest: Mapping[str, Any],
    *,
    include_paired_statistics: bool,
) -> Dict[str, Any]:
    cases = manifest["cases"]
    counts = {
        state: sum(case["state"] == state for case in cases)
        for state in ("pending", "running", "completed", "incomplete", "failed")
    }
    eligible = [case for case in cases if case["state"] == "completed"]
    paired = None
    if eligible and include_paired_statistics:
        outcomes = {
            selector: [bool(case["result"]["arms"][selector]["success"]) for case in eligible]
            for selector in PAIRED_SELECTORS
        }
        paired = summarize_paired_success(
            outcomes,
            task_ids=[case["task_id"] for case in eligible],
            case_ids=[case["case_id"] for case in eligible],
            bootstrap_seed=int(manifest["paired_bootstrap_seed"]),
            bootstrap_samples=config.bootstrap_samples,
        )
    unfinished = counts["pending"] + counts["running"]
    status = (
        "paused"
        if unfinished
        else "completed_with_errors"
        if counts["incomplete"] or counts["failed"]
        else "completed"
    )
    return {
        "schema": RAD_BATCH_SCHEMA,
        "batch_id": manifest["batch_id"],
        "status": status,
        "task_suite": config.task_suite,
        "config_sha256": manifest["config_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "source_identity": manifest["source_identity"],
        "source_identity_sha256": manifest["source_identity_sha256"],
        "paired_bootstrap_seed": manifest["paired_bootstrap_seed"],
        "bootstrap_samples": config.bootstrap_samples,
        "planned_cases": len(cases),
        "paired_eligible_cases": counts["completed"],
        "incomplete_cases": counts["incomplete"],
        "infra_failed_cases": counts["failed"],
        "pending_cases": counts["pending"],
        "running_cases": counts["running"],
        "paired_denominator_definition": (
            "only cases with four completed arms and fairness_exact=true"
        ),
        "eligible_case_ids": [case["case_id"] for case in eligible],
        "excluded_cases": [
            {
                "case_id": case["case_id"],
                "state": case["state"],
                "error_type": case.get("error_type"),
                "error": case.get("error"),
            }
            for case in cases
            if case["state"] != "completed"
        ],
        "paired_success": paired,
        "paired_statistics_current": include_paired_statistics,
        "cost_by_arm": _cost_summary(cases),
        "updated_utc": _utc_now(),
    }


def _persist(
    batch_dir: pathlib.Path,
    config: RadBatchConfig,
    manifest: Dict[str, Any],
    *,
    include_paired_statistics: bool = False,
) -> Dict[str, Any]:
    manifest["updated_utc"] = _utc_now()
    summary = _aggregate(
        config,
        manifest,
        include_paired_statistics=include_paired_statistics,
    )
    _atomic_json(batch_dir / RAD_BATCH_MANIFEST_FILE, manifest)
    _atomic_json(batch_dir / RAD_BATCH_SUMMARY_FILE, summary)
    return summary


def _create(config: RadBatchConfig) -> Tuple[pathlib.Path, RadBatchConfig, Dict[str, Any]]:
    normalized = dataclasses.replace(
        config,
        batch_dir=str(pathlib.Path(config.batch_dir).expanduser().resolve()),
        libero_root=str(pathlib.Path(config.libero_root).expanduser().resolve()),
    )
    validate_rad_batch_config(normalized)
    batch_dir = pathlib.Path(normalized.batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "cases").mkdir()
    plan = build_rad_case_plan(normalized)
    config_value = _config_dict(normalized)
    source_identity = collect_source_identity()
    immutable = {
        "schema": RAD_BATCH_SCHEMA,
        "batch_id": batch_dir.name,
        "created_utc": _utc_now(),
        "config": config_value,
        "config_sha256": _sha256_json(config_value),
        "plan": plan,
        "plan_sha256": _sha256_json(plan),
        "source_identity": source_identity,
        "source_identity_sha256": _sha256_json(source_identity),
        "seed_derivation": RAD_BATCH_SEED_DERIVATION,
        "paired_bootstrap_seed": _derive_global_seed(
            normalized.master_seed, _PAIRED_BOOTSTRAP_NAMESPACE
        ),
    }
    cases = []
    for planned in plan:
        case = dict(planned)
        case.update({"state": "pending", "attempts": [], "result": None})
        cases.append(case)
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
            "seed_derivation",
            "paired_bootstrap_seed",
        )
    }
    manifest.update({"updated_utc": immutable["created_utc"], "invocations": [], "cases": cases})
    _atomic_json(batch_dir / RAD_BATCH_CONFIG_FILE, immutable)
    _persist(batch_dir, normalized, manifest)
    return batch_dir, normalized, manifest


def _load(batch_dir: pathlib.Path) -> Tuple[RadBatchConfig, Dict[str, Any]]:
    immutable = _read_json(batch_dir / RAD_BATCH_CONFIG_FILE)
    manifest = _read_json(batch_dir / RAD_BATCH_MANIFEST_FILE)
    if immutable.get("schema") != RAD_BATCH_SCHEMA or manifest.get("schema") != RAD_BATCH_SCHEMA:
        raise RadBatchError("Unsupported RAD batch schema")
    for key in (
        "batch_id",
        "config",
        "config_sha256",
        "plan_sha256",
        "source_identity",
        "source_identity_sha256",
        "seed_derivation",
        "paired_bootstrap_seed",
    ):
        if immutable.get(key) != manifest.get(key):
            raise RadBatchError("RAD batch manifest disagrees with immutable %s" % key)
    if immutable["config_sha256"] != _sha256_json(immutable["config"]):
        raise RadBatchError("RAD batch config hash mismatch")
    if immutable["source_identity_sha256"] != _sha256_json(
        immutable["source_identity"]
    ):
        raise RadBatchError("RAD batch source identity hash mismatch")
    current_source_identity = collect_source_identity()
    if current_source_identity != immutable["source_identity"]:
        raise RadBatchError(
            "Current source identity differs from the frozen RAD batch source"
        )
    plan = immutable.get("plan")
    if not isinstance(plan, list) or immutable["plan_sha256"] != _sha256_json(plan):
        raise RadBatchError("RAD batch plan hash mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or [_case_plan_fields(case) for case in cases] != plan:
        raise RadBatchError("RAD batch case plan was modified")
    return _config_from_dict(immutable["config"]), manifest


def _run_cases(
    batch_dir: pathlib.Path,
    config: RadBatchConfig,
    manifest: Dict[str, Any],
    *,
    block_runner: RadBlockRunner,
    retry_failed: bool,
    max_new_cases: Optional[int],
) -> pathlib.Path:
    if max_new_cases is not None and max_new_cases < 1:
        raise ValueError("max_new_cases must be positive")
    _recover_stale_cases(batch_dir, manifest)
    if retry_failed:
        for case in manifest["cases"]:
            if case["state"] in ("failed", "incomplete"):
                case["state"] = "pending"
                case["result"] = None
    invocation = {
        "started_utc": _utc_now(),
        "pid": os.getpid(),
        "retry_failed": retry_failed,
        "max_new_cases": max_new_cases,
    }
    manifest["invocations"].append(invocation)
    _append_event(batch_dir / RAD_BATCH_EVENTS_FILE, "batch_started", **invocation)
    _persist(batch_dir, config, manifest)

    started_count = 0
    frozen_source_identity = manifest["source_identity"]

    def require_frozen_source_identity() -> Mapping[str, Any]:
        current_source_identity = collect_source_identity()
        if current_source_identity != frozen_source_identity:
            raise RadBatchError(
                "Current source identity differs from the frozen RAD batch source"
            )
        return current_source_identity

    def experiment_identity_provider(block_config: RadBlockConfig) -> Mapping[str, Any]:
        current_source_identity = require_frozen_source_identity()
        return collect_rad_experiment_identity(
            block_config, current_source_identity
        )

    for case in manifest["cases"]:
        if case["state"] != "pending":
            continue
        if max_new_cases is not None and started_count >= max_new_cases:
            break
        started_count += 1
        block_dir = batch_dir / "cases" / case["case_slug"]
        attempt = {
            "attempt": len(case["attempts"]) + 1,
            "state": "running",
            "started_utc": _utc_now(),
            "finished_utc": None,
            "arm_evidence": [],
        }
        case["attempts"].append(attempt)
        case["state"] = "running"
        case["result"] = None
        case.pop("error_type", None)
        case.pop("error", None)
        _append_event(
            batch_dir / RAD_BATCH_EVENTS_FILE,
            "case_started",
            case_id=case["case_id"],
            attempt=attempt["attempt"],
        )
        _persist(batch_dir, config, manifest)
        started = time.perf_counter()
        try:
            block_config = _block_config(config, case, block_dir)
            returned = pathlib.Path(
                block_runner(block_config)
                if block_runner is not _default_rad_block_runner
                and block_runner is not run_rad_block
                else run_rad_block(
                    block_config,
                    experiment_identity_provider=experiment_identity_provider,
                )
            ).expanduser().resolve()
            require_frozen_source_identity()
            if returned != block_dir.resolve():
                raise RadBatchIncompleteError(
                    "RAD block runner returned an unexpected directory"
                )
            result = _validate_completed_block(block_dir)
        except RadBatchIncompleteError as error:
            evidence = list(_arm_evidence(block_dir).values())
            case["state"] = "incomplete"
            case["result"] = None
            case["error_type"] = type(error).__name__
            case["error"] = str(error)
            attempt.update(
                {
                    "state": "incomplete",
                    "finished_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "arm_evidence": evidence,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _append_event(
                batch_dir / RAD_BATCH_EVENTS_FILE,
                "case_incomplete",
                case_id=case["case_id"],
                error=str(error),
            )
        except Exception as error:
            evidence = list(_arm_evidence(block_dir).values())
            fairness_failure = _has_explicit_fairness_failure(block_dir)
            terminal_state = "incomplete" if fairness_failure else "failed"
            case["state"] = terminal_state
            case["result"] = None
            case["error_type"] = type(error).__name__
            case["error"] = str(error)
            attempt.update(
                {
                    "state": terminal_state,
                    "finished_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "arm_evidence": evidence,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _append_event(
                batch_dir / RAD_BATCH_EVENTS_FILE,
                "case_incomplete" if fairness_failure else "case_failed",
                case_id=case["case_id"],
                error_type=type(error).__name__,
                error=str(error),
            )
        else:
            case["state"] = "completed"
            case["result"] = result
            attempt.update(
                {
                    "state": "completed",
                    "finished_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "arm_evidence": list(result["arms"].values()),
                }
            )
            _append_event(
                batch_dir / RAD_BATCH_EVENTS_FILE,
                "case_completed",
                case_id=case["case_id"],
                outcomes={
                    selector: result["arms"][selector]["success"]
                    for selector in PAIRED_SELECTORS
                },
            )
        _persist(batch_dir, config, manifest)

    invocation["finished_utc"] = _utc_now()
    invocation["cases_started"] = started_count
    summary = _persist(
        batch_dir,
        config,
        manifest,
        include_paired_statistics=True,
    )
    _append_event(
        batch_dir / RAD_BATCH_EVENTS_FILE,
        "batch_finished",
        status=summary["status"],
        cases_started=started_count,
        paired_eligible_cases=summary["paired_eligible_cases"],
        incomplete_cases=summary["incomplete_cases"],
        infra_failed_cases=summary["infra_failed_cases"],
    )
    return batch_dir


def run_rad_batch(
    config: RadBatchConfig,
    *,
    block_runner: RadBlockRunner = _default_rad_block_runner,
    max_new_cases: Optional[int] = None,
) -> pathlib.Path:
    """Create and execute a new deterministic multi-case RAD batch."""

    batch_dir, normalized, manifest = _create(config)
    with _rad_batch_lock(batch_dir):
        return _run_cases(
            batch_dir,
            normalized,
            manifest,
            block_runner=block_runner,
            retry_failed=False,
            max_new_cases=max_new_cases,
        )


def resume_rad_batch(
    path: str,
    *,
    retry_failed: bool = False,
    max_new_cases: Optional[int] = None,
    block_runner: RadBlockRunner = _default_rad_block_runner,
) -> pathlib.Path:
    """Resume pending RAD cases and optionally retry terminal failures."""

    batch_dir = pathlib.Path(path).expanduser().resolve()
    if batch_dir.is_file():
        batch_dir = batch_dir.parent
    if not (batch_dir / RAD_BATCH_CONFIG_FILE).is_file():
        raise RadBatchError("Not a RAD batch directory: %s" % batch_dir)
    with _rad_batch_lock(batch_dir):
        config, manifest = _load(batch_dir)
        return _run_cases(
            batch_dir,
            config,
            manifest,
            block_runner=block_runner,
            retry_failed=retry_failed,
            max_new_cases=max_new_cases,
        )


def load_rad_batch_summary(path: str) -> Dict[str, Any]:
    batch_dir = pathlib.Path(path).expanduser().resolve()
    if batch_dir.is_file():
        batch_dir = batch_dir.parent
    return _read_json(batch_dir / RAD_BATCH_SUMMARY_FILE)
