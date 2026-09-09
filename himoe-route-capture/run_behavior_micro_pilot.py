"""Run and audit the frozen 18-state behavior-geometry micro-pilot.

The runner is intentionally strict.  The JSON config is the preregistration:
evaluation thresholds, candidate counts, continuation batch sizes, and gates are
validated before any expensive process starts.  Every mutating stage is
resumable through the v2 artifact manifests; this file never infers completion
from a filename alone.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "behavior_micro_pilot_config.json"
CONFIG_SCHEMA = "himoe.behavior_micro_pilot.config.v1"
RUN_SCHEMA = "himoe.behavior_micro_pilot.run.v1"
PREPARE_IMPLEMENTATION_FILES = (
    "branch_snapshot.py",
    "behavior_forks_v2.py",
    "behavior_study_plan.py",
    "capture_behavior_study.py",
)
COLLECTION_IMPLEMENTATION_FILES = (
    "audit_candidate_capture_noop.py",
    "branch_snapshot.py",
    "behavior_forks_v2.py",
    "capture_behavior_study.py",
    "assemble_behavior_study.py",
    "himoe_candidate_capture_protocol.py",
    "himoe_route_store.py",
    "himoe_hidden_store.py",
    "himoe_flow_trajectory_store.py",
    "himoe_router_recorder.py",
    "serve_flow_trace.py",
    "serve_with_recorder.py",
    "run_behavior_micro_pilot.py",
)
ANALYSIS_IMPLEMENTATION_FILES = (
    "analyze_behavior_geometry.py",
    "analyze_route_outcome_geometry.py",
    "behavior_geometry.py",
    "behavior_micro_analysis.py",
)
STORE_NAMES = ("routes.zarr", "hidden.zarr", "flow_trajectory.zarr")
EXPECTED_CANDIDATE_ROWS = 18 * 32 * 2


class PilotConfigError(ValueError):
    """Raised before collection when the frozen design is internally invalid."""


class ResourceUnavailable(PilotConfigError):
    """The frozen run is valid, but deployment compute is currently occupied."""


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotConfigError("cannot read JSON %s: %s" % (path, error)) from error


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s.tmp-" % path.name, dir=str(path.parent)
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if "temporary_name" in locals() and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _implementation_digests(names: Sequence[str]) -> dict[str, str]:
    result = {}
    for name in names:
        path = HERE / name
        _require(path.is_file(), "implementation file is missing: %s" % path)
        result[name] = _sha256_file(path)
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotConfigError(message)


def _resolve(config_path: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen scientific and engineering invariants."""

    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected config schema")
    _require(config.get("confirmatory") is False, "micro-pilot must be non-confirmatory")
    _require(int(config.get("master_seed", -1)) == 20260824, "master seed changed")

    tasks = config.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 3, "exactly three tasks are required")
    task_ids = [int(item["task_id"]) for item in tasks]
    _require(task_ids == [0, 1, 3], "task order and ids must remain [0, 1, 3]")
    roles = {int(item["task_id"]): str(item["role"]) for item in tasks}
    _require(roles == {0: "calibration", 1: "evaluation", 3: "evaluation"},
             "task roles changed")

    sampling = config["snapshot_sampling"]
    per_task = int(sampling["snapshots_per_task"])
    _require(per_task == 6, "six snapshots per task are required")
    _require(bool(sampling["distinct_source_episodes"]), "source episodes must be distinct")
    _require(sampling["source_outcomes_per_task"] == {"success": 3, "failure": 3},
             "each task needs three success and three failure episodes")
    _require(int(sampling["event_critical"]["count_per_task"]) == 4,
             "four event-critical snapshots per task are required")
    _require(int(sampling["uniform_audit"]["count_per_task"]) == 2,
             "two uniform audit snapshots per task are required")

    seed_domains = list(config["seed_derivation"]["domains"])
    _require(len(seed_domains) == len(set(seed_domains)), "seed domains are not unique")
    required_domains = {
        "screen/candidate", "screen/continuation", "formal/candidate",
        "formal/continuation", "execution", "audit/inclusion",
    }
    _require(set(seed_domains) == required_domains, "seed domains changed")
    _require(bool(config["seed_derivation"]["continuation_seed_excludes_candidate"]),
             "continuation CRN key must exclude candidate id")

    capture = config["capture"]
    k = int(capture["candidates_per_pool"])
    snapshots = len(tasks) * per_task
    _require(k == 32, "candidate pool must remain K=32")
    _require(int(capture["action_chunk_horizon"]) == 10, "action horizon must remain H=10")
    _require(int(capture["continuation_control_steps"]) == 5,
             "continuation horizon must remain B=5")
    shard = int(capture["continuation_shard_repeats"])
    screen_r = int(capture["screen_repeats"])
    formal_r = int(capture["formal_initial_repeats"])
    topup_r = int(capture["formal_topup_repeats"])
    _require(shard == 8 and screen_r == 8, "screening must be one R=8 shard")
    _require(formal_r == 48 and topup_r == 96, "formal schedule must remain R48 -> R96")
    _require(formal_r % shard == 0 and topup_r % shard == 0,
             "continuation targets must align to shard boundaries")
    _require(int(capture["candidate_rows_expected"]) == snapshots * k * 2,
             "candidate row gate must count screen and formal pools")
    _require(int(capture["continuation_rows_expected"]) == 0,
             "continuations must never be recorded")
    _require(set(capture["capture_candidate_signals"]) == {
        "hb_router_probabilities", "router_hidden", "normalized_flow_trajectory"
    }, "candidate signal set changed")

    labels = config["formal_labels"]
    _require(float(labels["equivalence_margin"]) == 0.1,
             "formal equivalence margin must remain 0.1")
    _require(float(labels["confidence"]) == 0.95,
             "formal pair interval must remain 95%")
    _require(bool(labels["common_random_numbers"]), "formal labels require paired CRN")

    analysis = config["analysis"]
    _require(analysis["calibration_task_ids"] == [0], "only task 0 may calibrate")
    _require(analysis["evaluation_task_ids"] == [1, 3], "tasks 1 and 3 must evaluate")
    _require(float(analysis["action_threshold_quantiles"]["near"]) == 0.2,
             "near quantile changed")
    _require(float(analysis["action_threshold_quantiles"]["far"]) == 0.8,
             "far quantile changed")
    _require(bool(analysis["far_requires_physical_far"]),
             "far-action main test must require physically distinct trajectories")

    topup = config["global_topup_rule"]
    _require(int(topup["from_repeats"]) == formal_r, "top-up origin disagrees with capture")
    _require(int(topup["to_repeats"]) == topup_r, "top-up target disagrees with capture")
    _require(topup["apply_to"] == "all-formal-pools",
             "selective continuation top-up is forbidden")

    gates = config["gates"]
    _require(gates["effect_direction_is_gate"] is False,
             "route effect direction cannot gate this micro-pilot")
    _require(int(gates["engineering"]["planned_snapshots_complete"]) == snapshots,
             "engineering snapshot gate disagrees with plan")
    _require(int(gates["engineering"]["candidate_rows_exact"]) == snapshots * k * 2,
             "engineering row gate disagrees with capture")

    restrictions = config["restrictions"]
    _require(restrictions["evaluation_threshold_retuning"] is False,
             "evaluation retuning must stay disabled")
    _require(restrictions["scientific_failure_rerun_same_plan"] is False,
             "scientific failure cannot be rerun on the same evaluation plan")


def preflight(config_path: pathlib.Path, *, min_free_gib: float) -> dict[str, Any]:
    config = _load_json(config_path)
    if not isinstance(config, Mapping):
        raise PilotConfigError("config must be one JSON object")
    validate_config(config)

    checkpoint = _resolve(config_path, str(config["checkpoint"]["path"]))
    _require(checkpoint.is_dir(), "checkpoint directory does not exist: %s" % checkpoint)

    task_audits = []
    expected_checkpoint_sha = str(config["checkpoint"]["sha256"])
    expected_wrist = str(config["checkpoint"]["libero_wrist_layout"])
    for task in config["tasks"]:
        source = _resolve(config_path, str(task["source_rollout_dir"]))
        summaries_path = source / "summaries.json"
        metadata_path = source / "server_metadata.json"
        _require(summaries_path.is_file(), "missing source summaries: %s" % summaries_path)
        _require(metadata_path.is_file(), "missing source metadata: %s" % metadata_path)
        summaries = _load_json(summaries_path)
        metadata = _load_json(metadata_path)
        task_id = int(task["task_id"])
        matching = [row for row in summaries if int(row["task_id"]) == task_id]
        success = sum(bool(row["success"]) for row in matching)
        failure = len(matching) - success
        _require(success >= 3 and failure >= 3,
                 "task %d lacks three success and failure source episodes" % task_id)
        _require(metadata.get("checkpoint_sha256") == expected_checkpoint_sha,
                 "task %d source checkpoint differs from frozen checkpoint" % task_id)
        _require(metadata.get("libero_wrist_layout") == expected_wrist,
                 "task %d source wrist layout differs from frozen layout" % task_id)
        task_audits.append({
            "task_id": task_id,
            "source": str(source),
            "summaries_sha256": _sha256_file(summaries_path),
            "server_metadata_sha256": _sha256_file(metadata_path),
            "episodes": len(matching),
            "success": success,
            "failure": failure,
        })

    disk = shutil.disk_usage(str(config_path.parent))
    free_gib = disk.free / float(1024 ** 3)
    _require(free_gib >= min_free_gib,
             "%.2f GiB free is below the %.2f GiB preflight floor" % (free_gib, min_free_gib))
    return {
        "schema": "himoe.behavior_micro_pilot.preflight.v1",
        "ok": True,
        "confirmatory": False,
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "libero_wrist_layout": expected_wrist,
        "task_sources": task_audits,
        "disk_free_gib": free_gib,
        "minimum_disk_free_gib": min_free_gib,
    }


def _run_checked(
    argv: Sequence[str], *, cwd: pathlib.Path, environment: Optional[Mapping[str, str]] = None
) -> None:
    rendered = " ".join(argv)
    print("+ %s" % rendered, flush=True)
    subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=None if environment is None else dict(environment),
        check=True,
    )


def _runtime_paths(config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    config = _load_json(config_path)
    checkpoint = _resolve(config_path, str(config["checkpoint"]["path"]))
    cache_root = checkpoint.parent.parent
    workspace = HERE.parents[1]
    return {
        "cache_root": cache_root,
        "checkpoint": checkpoint,
        "libero_python": cache_root / "envs" / "libero" / "bin" / "python",
        "model_python": cache_root / "envs" / "model" / "bin" / "python",
        "libero_root": cache_root / "upstream" / "LIBERO",
        "upstream_root": cache_root / "upstream" / "HiMoE-VLA",
        "bridge_source": workspace / "himoe-libero-wrist-fix" / "src",
        "rs141_overlay": workspace / ".rs141-audit",
        "paper_overlay": workspace / ".paper-eval-overlay",
        "system_libs": cache_root / "system-libs" / "usr" / "lib" / "x86_64-linux-gnu",
    }


def _libero_environment(paths: Mapping[str, pathlib.Path]) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [
        paths["bridge_source"],
        paths["rs141_overlay"],
        paths["paper_overlay"],
        paths["libero_root"],
        paths["upstream_root"] / "packages" / "openpi-client" / "src",
    ]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(pathlib.Path(existing_python_path))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(
                    None,
                    [str(paths["system_libs"]), "/usr/lib/x86_64-linux-gnu",
                     environment.get("LD_LIBRARY_PATH", "")],
                )
            ),
        }
    )
    return environment


def _initialize_run(
    output: pathlib.Path, config_path: pathlib.Path, audit: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output / "pilot_manifest.json"
    config_digest = str(audit["config_sha256"])
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("schema") != RUN_SCHEMA:
            raise PilotConfigError("output has an incompatible pilot manifest")
        if manifest.get("config_sha256") != config_digest:
            raise PilotConfigError("output was initialized with a different frozen config")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise PilotConfigError("non-empty output has no pilot_manifest.json: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": RUN_SCHEMA,
        "confirmatory": False,
        "config_file": str(config_path),
        "config_sha256": config_digest,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "libero_wrist_layout": audit["libero_wrist_layout"],
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "initialized",
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def prepare_source_plan(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    min_free_gib: float,
) -> dict[str, Any]:
    """Replay source-only physical events and freeze the 18-state plan."""

    audit = preflight(config_path, min_free_gib=min_free_gib)
    config = _load_json(config_path)
    paths = _runtime_paths(config_path)
    required_paths = (
        "libero_python", "libero_root", "upstream_root", "bridge_source",
        "rs141_overlay", "paper_overlay", "system_libs",
    )
    for name in required_paths:
        _require(paths[name].exists(), "runtime path does not exist: %s" % paths[name])
    output = output.expanduser().resolve()
    manifest = _initialize_run(output, config_path, audit)
    manifest_path = output / "pilot_manifest.json"
    plan_stage = manifest.get("stages", {}).get("snapshot_plan", {})
    if plan_stage.get("complete"):
        plan_path = pathlib.Path(str(plan_stage.get("file", "")))
        _require(plan_path.is_file(), "completed plan stage has no plan file")
        _require(_sha256_file(plan_path) == plan_stage.get("sha256"),
                 "completed plan checksum changed")
        return manifest
    implementation = _implementation_digests(PREPARE_IMPLEMENTATION_FILES)
    prior_implementation = manifest.get("prepare_implementation_sha256")
    if prior_implementation is not None and prior_implementation != implementation:
        raise PilotConfigError(
            "prepare implementation changed during an interrupted source replay; "
            "start a new output to avoid mixed event extraction"
        )
    manifest["prepare_implementation_sha256"] = implementation
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    event_root = output / "source_events"
    environment = _libero_environment(paths)
    try:
        for task in config["tasks"]:
            if _implementation_digests(PREPARE_IMPLEMENTATION_FILES) != implementation:
                raise PilotConfigError(
                    "prepare implementation changed while source replay was running"
                )
            task_id = int(task["task_id"])
            source = _resolve(config_path, str(task["source_rollout_dir"]))
            _run_checked(
                [
                    str(paths["libero_python"]),
                    str(HERE / "capture_behavior_study.py"),
                    "source-events",
                    "--source-dir", str(source),
                    "--task-id", str(task_id),
                    "--benchmark", str(config["benchmark"]),
                    "--libero-root", str(paths["libero_root"]),
                    "--out", str(event_root),
                ],
                cwd=HERE,
                environment=environment,
            )
            manifest.setdefault("stages", {})["source_events_task_%d" % task_id] = {
                "complete": True,
                "completed_utc": _utc_now(),
            }
            manifest["status"] = "preparing"
            manifest["updated_utc"] = _utc_now()
            _atomic_json(manifest_path, manifest)

        plan_path = output / "plan.json"
        if not plan_path.exists():
            argv = [
                sys.executable,
                str(HERE / "behavior_study_plan.py"),
                "build",
            ]
            for task in config["tasks"]:
                source = _resolve(config_path, str(task["source_rollout_dir"]))
                argv.extend(["--task-source", "%d=%s" % (int(task["task_id"]), source)])
            argv.extend(["--event-root", str(event_root), "--out", str(plan_path)])
            _run_checked(argv, cwd=HERE)
        plan = _load_json(plan_path)
        _require(plan.get("schema") == "himoe.behavior_study.plan.v2",
                 "prepared plan has an unexpected schema")
        _require(len(plan.get("states", [])) == 18, "prepared plan does not contain 18 states")
        _require({int(row["task_id"]) for row in plan["states"]} == {0, 1, 3},
                 "prepared plan has unexpected tasks")
        manifest["stages"]["snapshot_plan"] = {
            "complete": True,
            "completed_utc": _utc_now(),
            "file": str(plan_path),
            "sha256": _sha256_file(plan_path),
            "states": 18,
        }
        manifest["status"] = "plan_ready"
        manifest["updated_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        return manifest
    except BaseException:
        manifest["status"] = "prepare_interrupted"
        manifest["updated_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        raise


def _freeze_implementation(
    manifest: dict[str, Any], manifest_path: pathlib.Path, key: str, names: Sequence[str]
) -> None:
    current = _implementation_digests(names)
    previous = manifest.get(key)
    if previous is not None and previous != current:
        raise PilotConfigError(
            "%s changed after this run was initialized; use a new output directory" % key
        )
    manifest[key] = current
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)


def _save_run_stage(
    manifest: dict[str, Any],
    manifest_path: pathlib.Path,
    name: str,
    **values: Any,
) -> None:
    manifest.setdefault("stages", {})[name] = {
        **manifest.get("stages", {}).get(name, {}),
        **values,
        "updated_utc": _utc_now(),
    }
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)


def _plan_for_run(output: pathlib.Path, manifest: Mapping[str, Any]) -> tuple[pathlib.Path, dict[str, Any]]:
    stage = manifest.get("stages", {}).get("snapshot_plan", {})
    if not stage.get("complete"):
        raise PilotConfigError("the source replay and 18-state plan are not complete")
    path = pathlib.Path(str(stage.get("file", output / "plan.json"))).resolve()
    _require(path.is_file(), "snapshot plan is missing: %s" % path)
    _require(_sha256_file(path) == stage.get("sha256"), "snapshot plan checksum changed")
    value = _load_json(path)
    _require(
        value.get("schema") == "himoe.behavior_study.plan.v2"
        and len(value.get("states", [])) == 18,
        "snapshot plan is invalid",
    )
    states = sorted(value["states"], key=lambda row: int(row["snapshot_index"]))
    _require(
        [int(row["snapshot_index"]) for row in states] == list(range(18)),
        "snapshot plan index is not contiguous",
    )
    value["states"] = states
    return path, value


def _candidate_prefix_audit(
    capture_root: pathlib.Path, plan: Mapping[str, Any], plan_path: pathlib.Path
) -> dict[str, Any]:
    """Return the checksum-verified, gap-free published candidate prefix."""

    from behavior_forks_v2 import (
        CANDIDATE_SCHEMA,
        candidate_dir,
        load_npz,
        validate_candidate_arrays,
        verify_artifact,
    )

    prefix = 0
    run_id: str | None = None
    artifacts = []
    missing_seen = False
    for pool in ("screen", "formal"):
        for state in plan["states"]:
            identifier = str(state["state_id"])
            descriptor = candidate_dir(capture_root, pool, identifier) / "artifact.json"
            if not descriptor.is_file():
                missing_seen = True
                continue
            if missing_seen:
                raise PilotConfigError(
                    "published candidate artifacts have a gap before %s/%s"
                    % (pool, identifier)
                )
            metadata = verify_artifact(descriptor, CANDIDATE_SCHEMA)
            if metadata.get("pool") != pool or metadata.get("state_id") != identifier:
                raise PilotConfigError("candidate artifact identity differs from the plan")
            if metadata.get("plan_sha256") != _sha256_file(plan_path):
                raise PilotConfigError("candidate artifact was generated from another plan")
            arrays = load_npz(descriptor.parent / metadata["files"]["data"]["file"])
            validate_candidate_arrays(arrays, metadata)
            rows = np.asarray(arrays["server_trace_rows"], dtype=np.int64)
            expected = np.arange(prefix, prefix + 32, dtype=np.int64)
            if not np.array_equal(rows, expected):
                raise PilotConfigError(
                    "candidate artifact rows do not extend the confirmed prefix at %s" % descriptor
                )
            ids = np.asarray(arrays["store_ids"]).astype(str)
            artifact_run_id = str(metadata.get("route_store_id", ""))
            if (
                not artifact_run_id
                or len(set(ids.tolist())) != 1
                or ids[0] != artifact_run_id
            ):
                raise PilotConfigError("candidate artifact has inconsistent store identity")
            if run_id is None:
                run_id = artifact_run_id
            elif artifact_run_id != run_id:
                raise PilotConfigError("candidate artifacts span multiple capture_run_ids")
            prefix += 32
            artifacts.append(str(descriptor))
    return {
        "confirmed_rows": prefix,
        "artifacts": len(artifacts),
        "capture_run_id": run_id,
        "complete": prefix == EXPECTED_CANDIDATE_ROWS,
        "artifact_files": artifacts,
    }


def _store_paths(store_root: pathlib.Path) -> list[pathlib.Path]:
    return [store_root / name for name in STORE_NAMES]


def _candidate_store_audit(
    store_root: pathlib.Path, *, expected_rows: int | None = None
) -> dict[str, Any]:
    paths = _store_paths(store_root)
    presence = [path.is_dir() for path in paths]
    if not any(presence):
        if expected_rows not in (None, 0):
            raise PilotConfigError("candidate stores are absent but published artifacts exist")
        return {"present": False, "rows": 0, "capture_run_id": None}
    if not all(presence):
        raise PilotConfigError("candidate route/hidden/flow stores are only partially present")
    import zarr

    groups = [zarr.open_group(str(path), mode="r") for path in paths]
    ids = [group.attrs.get("capture_run_id") for group in groups]
    if len(set(ids)) != 1 or not isinstance(ids[0], str) or not ids[0]:
        raise PilotConfigError("candidate stores do not share one capture_run_id")
    lengths = [
        int(groups[0]["episode_id"].shape[0]),
        int(groups[1]["episode_id"].shape[0]),
        int(groups[2]["query_id"].shape[0]),
    ]
    if len(set(lengths)) != 1:
        raise PilotConfigError("candidate stores have different row counts: %s" % lengths)
    declared = [int(group.attrs.get("common_durable_rows", -1)) for group in groups]
    if declared != [lengths[0]] * 3:
        raise PilotConfigError(
            "candidate stores disagree on their shared durable prefix: %s" % declared
        )
    if expected_rows is not None and lengths[0] != expected_rows:
        raise PilotConfigError(
            "candidate stores contain %d rows, expected %d" % (lengths[0], expected_rows)
        )
    return {
        "present": True,
        "rows": lengths[0],
        "capture_run_id": ids[0],
        "common_durable_rows": declared[0],
    }


def _gpu_free_gib(gpu: str) -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = float(result.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        raise PilotConfigError("cannot query free memory for GPU %s: %s" % (gpu, error)) from error
    return value


def _cgroup_free_gib() -> float | None:
    root = pathlib.Path("/sys/fs/cgroup")
    maximum_path = root / "memory.max"
    current_path = root / "memory.current"
    if not maximum_path.is_file() or not current_path.is_file():
        return None
    maximum_raw = maximum_path.read_text(encoding="ascii").strip()
    if maximum_raw == "max":
        return None
    try:
        maximum = int(maximum_raw)
        current = int(current_path.read_text(encoding="ascii").strip())
    except ValueError as error:
        raise PilotConfigError("cannot parse cgroup memory limits") from error
    # Linux can reclaim inactive file cache under pressure. Treating
    # memory.max-memory.current as the only headroom would make a large Zarr read
    # look like anonymous-memory exhaustion and spuriously block model startup.
    reclaimable = 0
    stat_path = root / "memory.stat"
    if stat_path.is_file():
        for line in stat_path.read_text(encoding="ascii").splitlines():
            name, separator, raw = line.partition(" ")
            if name == "inactive_file" and separator:
                try:
                    reclaimable = int(raw)
                except ValueError:
                    reclaimable = 0
                break
    available = maximum - current + reclaimable
    return min(maximum, max(0, available)) / float(1024 ** 3)


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind((host, 0))
        return int(stream.getsockname()[1])


def _validate_noop_audit(
    path: pathlib.Path, config: Mapping[str, Any], gpu: str
) -> dict[str, Any]:
    value = _load_json(path)
    checks = value.get("checks", {})
    if (
        value.get("schema") != "himoe.candidate_capture.noop_audit.v1"
        or value.get("passed") is not True
        or checks.get("actions_bitwise_equal") is not True
        or checks.get("flow_trajectories_bitwise_equal") is not True
        or checks.get("full_route_and_hidden_shapes") is not True
    ):
        raise PilotConfigError("candidate instrumentation no-op audit did not pass")
    if value.get("implementation_sha256") != _sha256_file(
        HERE / "audit_candidate_capture_noop.py"
    ):
        raise PilotConfigError("no-op audit implementation changed after execution")
    if value.get("checkpoint_sha256") != config["checkpoint"]["sha256"]:
        raise PilotConfigError("no-op audit used another checkpoint")
    if value.get("libero_wrist_layout") != config["checkpoint"]["libero_wrist_layout"]:
        raise PilotConfigError("no-op audit used another wrist layout")
    if str(value.get("gpu")) != str(gpu):
        raise PilotConfigError("no-op audit was run on another deployment GPU selector")
    return value


def _ensure_noop_audit(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    gpu: str,
    min_gpu_free_gib: float,
    min_cgroup_free_gib: float,
) -> tuple[pathlib.Path, dict[str, Any]]:
    if gpu == "cpu":
        raise PilotConfigError("the instrumentation audit requires the deployment CUDA path")
    config = _load_json(config_path)
    target = output / "audits" / "candidate_capture_noop.json"
    if target.is_file():
        return target, _validate_noop_audit(target, config, gpu)
    free_gib = _gpu_free_gib(gpu)
    if free_gib < min_gpu_free_gib:
        raise ResourceUnavailable(
            "GPU %s has %.2f GiB free; %.2f GiB is required for the no-op audit"
            % (gpu, free_gib, min_gpu_free_gib)
        )
    cgroup_free = _cgroup_free_gib()
    if cgroup_free is not None and cgroup_free < min_cgroup_free_gib:
        raise ResourceUnavailable(
            "cgroup has %.2f GiB free; %.2f GiB is required for the no-op audit"
            % (cgroup_free, min_cgroup_free_gib)
        )
    paths = _runtime_paths(config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".candidate_capture_noop.stage-", suffix=".json", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    temporary.unlink()
    try:
        _run_checked(
            [
                str(paths["model_python"]),
                str(HERE / "audit_candidate_capture_noop.py"),
                "--gpu",
                gpu,
                "--suite",
                str(config["benchmark"]).removeprefix("libero_"),
                "--checkpoint-dir",
                str(paths["checkpoint"]),
                "--upstream-root",
                str(paths["upstream_root"]),
                "--libero-wrist-layout",
                str(config["checkpoint"]["libero_wrist_layout"]),
                "--seed",
                str(config["master_seed"]),
                "--out",
                str(temporary),
            ],
            cwd=HERE,
            environment=_model_environment(paths),
        )
        value = _validate_noop_audit(temporary, config, gpu)
        os.replace(temporary, target)
        return target, value
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_environment(paths: Mapping[str, pathlib.Path]) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [
        HERE,
        paths["bridge_source"],
        paths["upstream_root"] / "packages" / "openpi-client" / "src",
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(pathlib.Path(environment["PYTHONPATH"]))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _tail(path: pathlib.Path, limit: int = 12_000) -> str:
    if not path.is_file():
        return ""
    value = path.read_text(encoding="utf-8", errors="replace")
    return value[-limit:]


@dataclass
class ManagedServer:
    process: subprocess.Popen[Any]
    log_stream: Any
    log_path: pathlib.Path
    host: str
    port: int
    capture_run_id: str
    store_root: pathlib.Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        self.log_stream.close()


def _start_server(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    confirmed_rows: int,
    confirmed_run_id: str | None,
    gpu: str,
    min_gpu_free_gib: float,
    min_cgroup_free_gib: float,
    host: str,
    port: int,
    startup_timeout: float,
) -> ManagedServer:
    if gpu == "cpu":
        raise PilotConfigError("the formal micro-pilot requires the deployed CUDA path")
    free_gib = _gpu_free_gib(gpu)
    if free_gib < min_gpu_free_gib:
        raise ResourceUnavailable(
            "GPU %s has %.2f GiB free; %.2f GiB is required before loading HiMoE"
            % (gpu, free_gib, min_gpu_free_gib)
        )
    cgroup_free = _cgroup_free_gib()
    if cgroup_free is not None and cgroup_free < min_cgroup_free_gib:
        raise ResourceUnavailable(
            "cgroup has %.2f GiB free; %.2f GiB host-memory headroom is required"
            % (cgroup_free, min_cgroup_free_gib)
        )
    paths = _runtime_paths(config_path)
    config = _load_json(config_path)
    store_root = output / "candidate_stores"
    present = [path.is_dir() for path in _store_paths(store_root)]
    if any(present) and not all(present):
        if confirmed_rows:
            raise PilotConfigError("partial stores cannot recover published candidate artifacts")
        recovery = output / "recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        destination = recovery / (
            "candidate_stores.partial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        os.replace(store_root, destination)
        present = [False, False, False]
    resume = all(present)
    if confirmed_rows and not resume:
        raise PilotConfigError("published candidate artifacts have no resumable stores")
    if resume:
        before = _candidate_store_audit(store_root)
        if confirmed_run_id is not None and before["capture_run_id"] != confirmed_run_id:
            raise PilotConfigError("published artifacts and candidate stores use different run ids")
        if int(before["rows"]) < confirmed_rows:
            raise PilotConfigError("candidate stores are shorter than the published prefix")
    store_root.mkdir(parents=True, exist_ok=True)
    logs = output / "server_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (
        "session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".log"
    )
    log_stream = log_path.open("a", encoding="utf-8")
    actual_port = port if port > 0 else _free_tcp_port(host)
    argv = [
        str(paths["model_python"]),
        "-u",
        str(HERE / "serve_with_recorder.py"),
        "--host",
        host,
        "--port",
        str(actual_port),
        "--gpu",
        gpu,
        "--suite",
        str(config["benchmark"]).removeprefix("libero_"),
        "--checkpoint-dir",
        str(paths["checkpoint"]),
        "--upstream-root",
        str(paths["upstream_root"]),
        "--libero-wrist-layout",
        str(config["checkpoint"]["libero_wrist_layout"]),
        "--out",
        str(store_root),
        "--request-gated-capture",
        "--parent-pid",
        str(os.getpid()),
        "--chunk-steps",
        "32",
    ]
    if resume:
        argv.extend(
            ["--resume-capture", "--resume-capture-rows", str(confirmed_rows)]
        )
    process = subprocess.Popen(
        argv,
        cwd=str(HERE),
        env=_model_environment(paths),
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_stream.close()
            raise PilotConfigError(
                "HiMoE server exited during startup (code %s):\n%s"
                % (process.returncode, _tail(log_path))
            )
        try:
            with socket.create_connection((host, actual_port), timeout=1.0):
                break
        except OSError:
            time.sleep(1.0)
    else:
        process.terminate()
        process.wait(timeout=30)
        log_stream.close()
        raise PilotConfigError("HiMoE server did not become ready:\n%s" % _tail(log_path))
    audit = _candidate_store_audit(store_root, expected_rows=confirmed_rows)
    run_id = str(audit["capture_run_id"])
    if confirmed_run_id is not None and run_id != confirmed_run_id:
        process.terminate()
        process.wait(timeout=30)
        log_stream.close()
        raise PilotConfigError("resumed server changed the candidate capture_run_id")
    return ManagedServer(
        process=process,
        log_stream=log_stream,
        log_path=log_path,
        host=host,
        port=actual_port,
        capture_run_id=run_id,
        store_root=store_root,
    )


@contextlib.contextmanager
def _server_session(*args: Any, **kwargs: Any) -> Iterator[ManagedServer]:
    server = _start_server(*args, **kwargs)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate_runner(_signum: int, _frame: Any) -> None:
        server.stop()
        raise KeyboardInterrupt("runner received SIGTERM")

    signal.signal(signal.SIGTERM, terminate_runner)
    try:
        yield server
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.stop()


def _capture_command_environment(config_path: pathlib.Path) -> dict[str, str]:
    return _libero_environment(_runtime_paths(config_path))


def _candidate_artifact_path(
    capture_root: pathlib.Path, pool: str, state_id: str
) -> pathlib.Path:
    from behavior_forks_v2 import candidate_dir

    return candidate_dir(capture_root, pool, state_id) / "artifact.json"


def _shard_artifact_path(
    capture_root: pathlib.Path,
    pool: str,
    state_id: str,
    cohort: str,
    shard_index: int,
) -> pathlib.Path:
    from behavior_forks_v2 import shard_dir

    return shard_dir(capture_root, pool, state_id, cohort, shard_index) / "artifact.json"


def _collect_candidates(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    plan_path: pathlib.Path,
    plan: Mapping[str, Any],
    server: ManagedServer,
    manifest: dict[str, Any],
    manifest_path: pathlib.Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    paths = _runtime_paths(config_path)
    capture_root = output / "capture"
    environment = _capture_command_environment(config_path)
    audit = _candidate_prefix_audit(capture_root, plan, plan_path)
    if audit["capture_run_id"] not in (None, server.capture_run_id):
        raise PilotConfigError("published candidate artifacts use another server run id")
    for pool in ("screen", "formal"):
        for state in plan["states"]:
            identifier = str(state["state_id"])
            descriptor = _candidate_artifact_path(capture_root, pool, identifier)
            if descriptor.is_file():
                continue
            expected_rows = int(audit["confirmed_rows"])
            try:
                _run_checked(
                    [
                        str(paths["libero_python"]),
                        str(HERE / "capture_behavior_study.py"),
                        "candidate",
                        "--plan",
                        str(plan_path),
                        "--state-id",
                        identifier,
                        "--pool",
                        pool,
                        "--host",
                        server.host,
                        "--port",
                        str(server.port),
                        "--benchmark",
                        str(config["benchmark"]),
                        "--libero-root",
                        str(paths["libero_root"]),
                        "--out",
                        str(capture_root),
                        "--route-store-id",
                        server.capture_run_id,
                        "--expected-recorder-rows",
                        str(expected_rows),
                    ],
                    cwd=HERE,
                    environment=environment,
                )
            except BaseException:
                # The surrounding server context is deliberately abandoned on any
                # pool error. Its SIGTERM rolls back an in-memory half-pool; the next
                # invocation truncates a durable orphan to the published prefix.
                raise
            audit = _candidate_prefix_audit(capture_root, plan, plan_path)
            if int(audit["confirmed_rows"]) != expected_rows + 32:
                raise PilotConfigError("candidate process did not publish exactly one K=32 pool")
            store = _candidate_store_audit(
                server.store_root, expected_rows=int(audit["confirmed_rows"])
            )
            if store["capture_run_id"] != server.capture_run_id:
                raise PilotConfigError("candidate store identity changed during collection")
            _save_run_stage(
                manifest,
                manifest_path,
                "candidates",
                complete=bool(audit["complete"]),
                confirmed_rows=int(audit["confirmed_rows"]),
                artifacts=int(audit["artifacts"]),
                capture_run_id=server.capture_run_id,
                last_pool={"pool": pool, "state_id": identifier},
            )
    audit = _candidate_prefix_audit(capture_root, plan, plan_path)
    if int(audit["confirmed_rows"]) != EXPECTED_CANDIDATE_ROWS:
        raise PilotConfigError("candidate collection ended without exactly 1152 rows")
    _candidate_store_audit(server.store_root, expected_rows=EXPECTED_CANDIDATE_ROWS)
    return audit


def _validate_existing_shard(path: pathlib.Path, cohort: str, shard_index: int) -> None:
    from behavior_forks_v2 import SHARD_SCHEMA, verify_artifact

    metadata = verify_artifact(path, SHARD_SCHEMA)
    if metadata.get("label_cohort") != cohort or int(metadata["shard_index"]) != shard_index:
        raise PilotConfigError("continuation shard identity mismatch: %s" % path)


def _missing_shards(
    capture_root: pathlib.Path,
    plan: Mapping[str, Any],
    cohort: str,
    shard_start: int,
    shard_stop: int,
    state_ids: set[str] | None = None,
) -> int:
    pool = "formal" if cohort == "main" else "screen"
    missing = 0
    for state in plan["states"]:
        identifier = str(state["state_id"])
        if state_ids is not None and identifier not in state_ids:
            continue
        for shard_index in range(shard_start, shard_stop):
            path = _shard_artifact_path(
                capture_root, pool, identifier, cohort, shard_index
            )
            if path.is_file():
                _validate_existing_shard(path, cohort, shard_index)
            else:
                missing += 1
    return missing


def _collect_shards(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    plan: Mapping[str, Any],
    cohort: str,
    shard_start: int,
    shard_stop: int,
    server: ManagedServer,
    manifest: dict[str, Any],
    manifest_path: pathlib.Path,
    state_ids: set[str] | None = None,
) -> None:
    from behavior_forks_v2 import CANDIDATE_SCHEMA, verify_artifact

    paths = _runtime_paths(config_path)
    capture_root = output / "capture"
    environment = _capture_command_environment(config_path)
    pool = "formal" if cohort == "main" else "screen"
    _candidate_store_audit(server.store_root, expected_rows=EXPECTED_CANDIDATE_ROWS)
    completed = 0
    total = 0
    for state in plan["states"]:
        identifier = str(state["state_id"])
        if state_ids is not None and identifier not in state_ids:
            continue
        candidate = _candidate_artifact_path(capture_root, pool, identifier)
        verify_artifact(candidate, CANDIDATE_SCHEMA)
        for shard_index in range(shard_start, shard_stop):
            total += 1
            descriptor = _shard_artifact_path(
                capture_root, pool, identifier, cohort, shard_index
            )
            if descriptor.is_file():
                _validate_existing_shard(descriptor, cohort, shard_index)
                completed += 1
                continue
            _run_checked(
                [
                    str(paths["libero_python"]),
                    str(HERE / "capture_behavior_study.py"),
                    "continuation-shard",
                    "--candidate-artifact",
                    str(candidate),
                    "--cohort",
                    cohort,
                    "--shard-index",
                    str(shard_index),
                    "--host",
                    server.host,
                    "--port",
                    str(server.port),
                    "--libero-root",
                    str(paths["libero_root"]),
                    "--out",
                    str(capture_root),
                    "--expected-recorder-rows",
                    str(EXPECTED_CANDIDATE_ROWS),
                ],
                cwd=HERE,
                environment=environment,
            )
            _validate_existing_shard(descriptor, cohort, shard_index)
            _candidate_store_audit(server.store_root, expected_rows=EXPECTED_CANDIDATE_ROWS)
            completed += 1
            _save_run_stage(
                manifest,
                manifest_path,
                "continuations_%s" % cohort,
                complete=False,
                completed_shards=completed,
                requested_shards=total,
                last={"state_id": identifier, "shard_index": shard_index},
                candidate_rows_unchanged=EXPECTED_CANDIDATE_ROWS,
            )
    _save_run_stage(
        manifest,
        manifest_path,
        "continuations_%s" % cohort,
        complete=True,
        shard_start=shard_start,
        shard_stop=shard_stop,
        selected_states=(18 if state_ids is None else len(state_ids)),
        candidate_rows_unchanged=EXPECTED_CANDIDATE_ROWS,
    )


def _assembly_path(output: pathlib.Path, cohort: str, repeats: int) -> pathlib.Path:
    return output / "assemblies" / ("%s-r%d" % (cohort, repeats))


def _assemble(
    *,
    output: pathlib.Path,
    plan_path: pathlib.Path,
    cohort: str,
    repeats: int,
    state_ids_file: pathlib.Path | None = None,
) -> pathlib.Path:
    target = _assembly_path(output, cohort, repeats)
    argv = [
        sys.executable,
        str(HERE / "assemble_behavior_study.py"),
        "--plan",
        str(plan_path),
        "--capture-root",
        str(output / "capture"),
        "--cohort",
        cohort,
        "--target-r",
        str(repeats),
        "--out",
        str(target),
    ]
    if state_ids_file is not None:
        argv.extend(["--state-ids-file", str(state_ids_file)])
    _run_checked(argv, cwd=HERE)
    manifest = _load_json(target / "assembly_manifest.v2.json")
    expected_states = 18
    if state_ids_file is not None:
        expected_states = len(_load_json(state_ids_file))
    if (
        manifest.get("schema") != "himoe.behavior_forks.assembly.v2"
        or manifest.get("complete") is not True
        or int(manifest.get("target_repeats", -1)) != repeats
        or int(manifest.get("state_count", -1)) != expected_states
    ):
        raise PilotConfigError("assembled cohort failed its manifest gate: %s" % target)
    return target


def _write_frozen_json(path: pathlib.Path, value: Mapping[str, Any] | list[Any]) -> None:
    if path.is_file():
        existing = _load_json(path)
        if existing != value:
            raise PilotConfigError("frozen JSON artifact would change on resume: %s" % path)
        return
    _atomic_json(path, value)


def _derive_screen_selection(
    *,
    output: pathlib.Path,
    screen_capture: pathlib.Path,
    main_capture: pathlib.Path,
) -> tuple[pathlib.Path, set[str]]:
    """Freeze enriched states without opening route, hidden, or flow stores."""

    from analyze_route_outcome_geometry import _load_formal_capture
    from behavior_micro_analysis import (
        apply_physical_scales,
        fit_formal_calibration,
        screen_snapshot_flags,
    )

    _screen_provenance, screen_pools, screen_rows = _load_formal_capture(
        screen_capture, 0.25, 0.95
    )
    _main_provenance, _main_pools, main_rows = _load_formal_capture(
        main_capture, 0.25, 0.95
    )
    # Screening uses task-0 action/physical thresholds. The shared calibration
    # helper also fits a route residual, which is irrelevant here; a constant
    # placeholder keeps this path demonstrably independent of all MoE stores.
    calibration_rows = [dict(row, d_route=0.0) for row in main_rows]
    calibration = fit_formal_calibration(
        calibration_rows,
        calibration_task_ids=(0,),
        near_quantile=0.2,
        far_quantile=0.8,
        caliper_multiplier=0.2,
    )
    apply_physical_scales(
        screen_rows, calibration.physical_schema, calibration.physical_scales
    )
    flags = screen_snapshot_flags(screen_rows, calibration, min_pairs=2, min_candidates=3)
    uid_to_state = {
        "task%d/%s" % (int(pool["task_id"]), pool["snapshot_state_sha256"]): str(
            pool["snapshot_id"]
        )
        for pool in screen_pools
    }
    positives = {
        uid_to_state[uid]
        for uid, strata in flags.items()
        if uid in uid_to_state
        and any(
            bool(strata.get(stratum, {}).get("screen_positive", False))
            for stratum in ("near", "far")
        )
    }
    ordered = [
        str(pool["snapshot_id"])
        for pool in sorted(screen_pools, key=lambda row: int(row["snapshot_index"]))
        if str(pool["snapshot_id"]) in positives
    ]
    selection = {
        "schema": "himoe.behavior_screen_selection.v1",
        "confirmatory": False,
        "allowed_inputs": [
            "action",
            "physical_trajectory",
            "event_tape",
            "R8_outcome",
        ],
        "forbidden_inputs_opened": [],
        "screen_capture": str(screen_capture),
        "screen_capture_manifest_sha256": _sha256_file(
            screen_capture / "assembly_manifest.v2.json"
        ),
        "main_capture": str(main_capture),
        "main_capture_manifest_sha256": _sha256_file(
            main_capture / "assembly_manifest.v2.json"
        ),
        "thresholds": {
            "action_near_q20": calibration.action_near_q20,
            "action_far_q80": calibration.action_far_q80,
            "physics_far_q80": calibration.physics_far_q80,
            "same_events": "exact",
            "same_abs_delta_q_max": 0.125,
            "different_event_or_abs_delta_q_min": 0.25,
            "same_pairs_min": 2,
            "different_pairs_min": 2,
            "unique_candidates_per_class_min": 3,
        },
        "flags": flags,
        "state_ids": ordered,
    }
    selection_path = output / "screen_selection.json"
    _write_frozen_json(selection_path, selection)
    ids_path = output / "enriched_state_ids.json"
    _write_frozen_json(ids_path, ordered)
    return ids_path, set(ordered)


def _formal_spec(
    *,
    output: pathlib.Path,
    plan_path: pathlib.Path,
    repeats: int,
    screen_capture: pathlib.Path,
    main_capture: pathlib.Path,
    enriched_capture: pathlib.Path | None,
    instrumentation_audit: pathlib.Path,
) -> pathlib.Path:
    bundle: dict[str, Any] = {
        "id": "behavior-micro-pilot-20260824",
        "screen_capture": str(screen_capture),
        "main_capture": str(main_capture),
    }
    if enriched_capture is not None:
        bundle["enriched_capture"] = str(enriched_capture)
    value = {
        "schema": "himoe.route_outcome_formal_input.v1",
        "confirmatory": False,
        "calibration_task_ids": [0],
        "evaluation_task_ids": [1, 3],
        "hidden_projection_seed": 20260824,
        "min_matches": 5,
        "study_plan": str(plan_path),
        "instrumentation_audit": str(instrumentation_audit),
        "stores": {
            "routes": str(output / "candidate_stores" / "routes.zarr"),
            "hidden": str(output / "candidate_stores" / "hidden.zarr"),
            "flow_trajectory": str(
                output / "candidate_stores" / "flow_trajectory.zarr"
            ),
        },
        "bundles": [bundle],
    }
    path = output / "analysis_inputs" / ("formal-r%d.json" % repeats)
    _write_frozen_json(path, value)
    return path


def _run_formal_analysis(
    *, output: pathlib.Path, formal_spec: pathlib.Path, repeats: int, bootstrap: int
) -> dict[str, Any]:
    target = output / "analysis" / ("formal-r%d" % repeats)
    summary_path = target / "summary.json"
    report_path = target / "REPORT.md"
    if summary_path.is_file():
        summary = _load_json(summary_path)
        if summary.get("formal_spec", {}).get("sha256") != _sha256_file(formal_spec):
            raise PilotConfigError("existing analysis was generated from another formal spec")
    if not summary_path.is_file() or not report_path.is_file():
        _run_checked(
            [
                sys.executable,
                str(HERE / "analyze_route_outcome_geometry.py"),
                "--mode",
                "formal",
                "--formal-spec",
                str(formal_spec),
                "--out-dir",
                str(target),
                "--bootstrap",
                str(bootstrap),
                "--confidence",
                "0.95",
                "--seed",
                "20260824",
            ],
            cwd=HERE,
        )
        summary = _load_json(summary_path)
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        raise PilotConfigError("formal analysis did not publish a complete report")
    if summary.get("confirmatory") is not False:
        raise PilotConfigError("formal micro-pilot report is not marked non-confirmatory")
    gates = summary.get("gates", {})
    if gates.get("effect_direction_used") is not False:
        raise PilotConfigError("analysis incorrectly used route-effect direction as a gate")
    return summary


def _server_kwargs(args: argparse.Namespace, config_path: pathlib.Path, output: pathlib.Path,
                   prefix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_path": config_path,
        "output": output,
        "confirmed_rows": int(prefix["confirmed_rows"]),
        "confirmed_run_id": prefix.get("capture_run_id"),
        "gpu": str(args.gpu),
        "min_gpu_free_gib": float(args.min_gpu_free_gib),
        "min_cgroup_free_gib": float(args.min_cgroup_free_gib),
        "host": str(args.host),
        "port": int(args.port),
        "startup_timeout": float(args.server_startup_timeout),
    }


def _initial_collection_missing(
    output: pathlib.Path, plan: Mapping[str, Any], plan_path: pathlib.Path
) -> tuple[dict[str, Any], bool]:
    prefix = _candidate_prefix_audit(output / "capture", plan, plan_path)
    missing = not bool(prefix["complete"])
    if not missing:
        missing = bool(
            _missing_shards(output / "capture", plan, "screen", 0, 1)
            or _missing_shards(output / "capture", plan, "main", 0, 6)
        )
    try:
        _candidate_store_audit(
            output / "candidate_stores", expected_rows=int(prefix["confirmed_rows"])
        )
    except PilotConfigError:
        missing = True
    return prefix, missing


def _execute_micro_pilot_impl(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the frozen pilot through collection, global top-up, and reporting."""

    prepare_source_plan(
        config_path, output=output, min_free_gib=float(args.min_free_gib)
    )
    manifest_path = output / "pilot_manifest.json"
    manifest = _load_json(manifest_path)
    plan_path, plan = _plan_for_run(output, manifest)
    _freeze_implementation(
        manifest,
        manifest_path,
        "collection_implementation_sha256",
        COLLECTION_IMPLEMENTATION_FILES,
    )
    manifest["status"] = "collecting"
    _atomic_json(manifest_path, manifest)
    noop_audit_path, noop_audit = _ensure_noop_audit(
        config_path,
        output=output,
        gpu=str(args.gpu),
        min_gpu_free_gib=float(args.min_gpu_free_gib),
        min_cgroup_free_gib=float(args.min_cgroup_free_gib),
    )
    _save_run_stage(
        manifest,
        manifest_path,
        "instrumentation_noop_audit",
        complete=True,
        passed=True,
        file=str(noop_audit_path),
        sha256=_sha256_file(noop_audit_path),
        gpu=str(args.gpu),
    )
    prefix, initial_missing = _initial_collection_missing(output, plan, plan_path)
    if initial_missing:
        try:
            with _server_session(**_server_kwargs(args, config_path, output, prefix)) as server:
                if manifest.get("capture_run_id") not in (None, server.capture_run_id):
                    raise PilotConfigError("pilot manifest capture_run_id changed")
                manifest["capture_run_id"] = server.capture_run_id
                _atomic_json(manifest_path, manifest)
                prefix = _collect_candidates(
                    config_path,
                    output=output,
                    plan_path=plan_path,
                    plan=plan,
                    server=server,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                _collect_shards(
                    config_path,
                    output=output,
                    plan=plan,
                    cohort="screen",
                    shard_start=0,
                    shard_stop=1,
                    server=server,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                _collect_shards(
                    config_path,
                    output=output,
                    plan=plan,
                    cohort="main",
                    shard_start=0,
                    shard_stop=6,
                    server=server,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
        except BaseException as error:
            manifest["status"] = "collection_interrupted"
            _save_run_stage(
                manifest,
                manifest_path,
                "last_error",
                complete=False,
                error_type=type(error).__name__,
                message=str(error),
            )
            raise
    prefix = _candidate_prefix_audit(output / "capture", plan, plan_path)
    if not prefix["complete"]:
        raise PilotConfigError("initial collection did not complete the candidate stores")
    _candidate_store_audit(
        output / "candidate_stores", expected_rows=EXPECTED_CANDIDATE_ROWS
    )

    screen_capture = _assemble(
        output=output, plan_path=plan_path, cohort="screen", repeats=8
    )
    main_r48 = _assemble(
        output=output, plan_path=plan_path, cohort="main", repeats=48
    )
    _freeze_implementation(
        manifest,
        manifest_path,
        "analysis_implementation_sha256",
        ANALYSIS_IMPLEMENTATION_FILES,
    )
    enriched_ids_file, enriched_ids = _derive_screen_selection(
        output=output, screen_capture=screen_capture, main_capture=main_r48
    )
    enriched_r48 = None
    if enriched_ids:
        if _missing_shards(
            output / "capture", plan, "enriched", 0, 6, enriched_ids
        ):
            with _server_session(**_server_kwargs(args, config_path, output, prefix)) as server:
                _collect_shards(
                    config_path,
                    output=output,
                    plan=plan,
                    cohort="enriched",
                    shard_start=0,
                    shard_stop=6,
                    server=server,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    state_ids=enriched_ids,
                )
        enriched_r48 = _assemble(
            output=output,
            plan_path=plan_path,
            cohort="enriched",
            repeats=48,
            state_ids_file=enriched_ids_file,
        )
    spec_r48 = _formal_spec(
        output=output,
        plan_path=plan_path,
        repeats=48,
        screen_capture=screen_capture,
        main_capture=main_r48,
        enriched_capture=enriched_r48,
        instrumentation_audit=noop_audit_path,
    )
    summary_r48 = _run_formal_analysis(
        output=output,
        formal_spec=spec_r48,
        repeats=48,
        bootstrap=int(args.bootstrap),
    )
    topup = bool(summary_r48["global_topup"]["global_topup_required"])
    final_repeats = 48
    final_summary = summary_r48
    if topup:
        need_main = _missing_shards(output / "capture", plan, "main", 6, 12)
        need_enriched = (
            _missing_shards(
                output / "capture", plan, "enriched", 6, 12, enriched_ids
            )
            if enriched_ids
            else 0
        )
        if need_main or need_enriched:
            with _server_session(**_server_kwargs(args, config_path, output, prefix)) as server:
                _collect_shards(
                    config_path,
                    output=output,
                    plan=plan,
                    cohort="main",
                    shard_start=6,
                    shard_stop=12,
                    server=server,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                if enriched_ids:
                    _collect_shards(
                        config_path,
                        output=output,
                        plan=plan,
                        cohort="enriched",
                        shard_start=6,
                        shard_stop=12,
                        server=server,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        state_ids=enriched_ids,
                    )
        main_r96 = _assemble(
            output=output, plan_path=plan_path, cohort="main", repeats=96
        )
        enriched_r96 = None
        if enriched_ids:
            enriched_r96 = _assemble(
                output=output,
                plan_path=plan_path,
                cohort="enriched",
                repeats=96,
                state_ids_file=enriched_ids_file,
            )
        spec_r96 = _formal_spec(
            output=output,
            plan_path=plan_path,
            repeats=96,
            screen_capture=screen_capture,
            main_capture=main_r96,
            enriched_capture=enriched_r96,
            instrumentation_audit=noop_audit_path,
        )
        final_summary = _run_formal_analysis(
            output=output,
            formal_spec=spec_r96,
            repeats=96,
            bootstrap=int(args.bootstrap),
        )
        final_repeats = 96
    runner_gate_passed = bool(final_summary["gates"]["passed"] and noop_audit["passed"])
    decision = {
        "schema": "himoe.behavior_micro_pilot.decision.v1",
        "confirmatory": False,
        "formal_repeats": final_repeats,
        "statistics_gate_passed": bool(final_summary["gates"]["passed"]),
        "instrumentation_noop_gate_passed": bool(noop_audit["passed"]),
        "effect_direction_used": False,
        "passed": runner_gate_passed,
        "summary": str(
            output / "analysis" / ("formal-r%d" % final_repeats) / "summary.json"
        ),
    }
    _atomic_json(output / "final_decision.json", decision)
    manifest["status"] = "completed"
    _save_run_stage(
        manifest,
        manifest_path,
        "final_analysis",
        complete=True,
        confirmatory=False,
        repeats=final_repeats,
        gate_passed=runner_gate_passed,
        effect_direction_used=False,
        summary=str(output / "analysis" / ("formal-r%d" % final_repeats) / "summary.json"),
        decision=str(output / "final_decision.json"),
    )
    manifest["status"] = "completed"
    manifest["updated_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    return {**final_summary, "runner_decision": decision}


def execute_micro_pilot(
    config_path: pathlib.Path,
    *,
    output: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        return _execute_micro_pilot_impl(
            config_path, output=output, args=args
        )
    except BaseException as error:
        manifest_path = output / "pilot_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = _load_json(manifest_path)
                if manifest.get("status") != "completed":
                    resource_wait = isinstance(error, ResourceUnavailable)
                    manifest["status"] = (
                        "waiting_for_resources" if resource_wait else "interrupted"
                    )
                    _save_run_stage(
                        manifest,
                        manifest_path,
                        "resource_gate" if resource_wait else "last_error",
                        complete=False,
                        error_type=type(error).__name__,
                        message=str(error),
                    )
            except Exception:
                # Preserve the original collection/analysis failure. A corrupt
                # run manifest will be diagnosed explicitly on the next resume.
                pass
        raise


def pilot_status(
    config_path: pathlib.Path, *, output: pathlib.Path, gpu: str | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "himoe.behavior_micro_pilot.status.v1",
        "confirmatory": False,
        "output": str(output),
        "initialized": (output / "pilot_manifest.json").is_file(),
    }
    result["cgroup_free_gib"] = _cgroup_free_gib()
    if gpu is not None and gpu != "cpu":
        try:
            result["gpu"] = {"id": gpu, "free_gib": _gpu_free_gib(gpu)}
        except PilotConfigError as error:
            result["gpu"] = {"id": gpu, "error": str(error)}
    if not result["initialized"]:
        return result
    manifest = _load_json(output / "pilot_manifest.json")
    result["run_status"] = manifest.get("status")
    result["source_event_artifacts"] = len(
        list((output / "source_events").glob("task_*/episode_*/artifact.json"))
    )
    audit_path = output / "audits" / "candidate_capture_noop.json"
    if audit_path.is_file():
        try:
            audit = _validate_noop_audit(
                audit_path, _load_json(config_path), "0" if gpu is None else gpu
            )
            result["instrumentation_noop_audit"] = {
                "passed": True,
                "file": str(audit_path),
                "sha256": _sha256_file(audit_path),
                "gpu": audit["gpu"],
            }
        except PilotConfigError as error:
            result["instrumentation_noop_audit"] = {
                "passed": False,
                "error": str(error),
            }
    result["plan_ready"] = bool(
        manifest.get("stages", {}).get("snapshot_plan", {}).get("complete")
    )
    if result["plan_ready"]:
        plan_path, plan = _plan_for_run(output, manifest)
        try:
            result["candidates"] = _candidate_prefix_audit(
                output / "capture", plan, plan_path
            )
        except (PilotConfigError, ValueError) as error:
            result["candidates"] = {"error": str(error)}
        try:
            result["stores"] = _candidate_store_audit(output / "candidate_stores")
        except (PilotConfigError, ValueError) as error:
            result["stores"] = {"error": str(error)}
    final_stage = manifest.get("stages", {}).get("final_analysis")
    if final_stage:
        result["final_analysis"] = final_stage
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight", help="validate frozen inputs without mutation")
    check.add_argument("--min-free-gib", type=float, default=5.0)
    check.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    prepare = subparsers.add_parser(
        "prepare", help="replay source events and freeze the 18-state plan"
    )
    prepare.add_argument("--out", type=pathlib.Path, required=True)
    prepare.add_argument("--min-free-gib", type=float, default=5.0)
    status = subparsers.add_parser("status", help="audit resumable artifacts and GPU capacity")
    status.add_argument("--out", type=pathlib.Path, required=True)
    status.add_argument("--gpu", default="0")
    status.add_argument("--json", action="store_true")
    run = subparsers.add_parser(
        "run", help="execute the frozen pilot through R48/R96 analysis"
    )
    run.add_argument("--out", type=pathlib.Path, required=True)
    run.add_argument("--min-free-gib", type=float, default=5.0)
    run.add_argument("--gpu", default="0")
    run.add_argument("--min-gpu-free-gib", type=float, default=24.0)
    run.add_argument("--min-cgroup-free-gib", type=float, default=1.5)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--server-startup-timeout", type=float, default=900.0)
    run.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        if args.command == "preflight":
            audit = preflight(config_path, min_free_gib=float(args.min_free_gib))
            if args.json:
                print(json.dumps(audit, indent=2, sort_keys=True))
            else:
                print(
                    "preflight passed: 3 tasks, 18 planned snapshots, "
                    "%.2f GiB free, confirmatory=false" % audit["disk_free_gib"]
                )
            return 0
        if args.command == "prepare":
            manifest = prepare_source_plan(
                config_path,
                output=args.out,
                min_free_gib=float(args.min_free_gib),
            )
            print(
                "source replay and 18-state plan complete: %s"
                % manifest["stages"]["snapshot_plan"]["file"]
            )
            return 0
        if args.command == "status":
            status = pilot_status(
                config_path,
                output=args.out.expanduser().resolve(),
                gpu=str(args.gpu),
            )
            if args.json:
                print(json.dumps(status, indent=2, sort_keys=True))
            else:
                candidates = status.get("candidates", {})
                gpu = status.get("gpu", {})
                print(
                    "status=%s plan_ready=%s candidate_rows=%s gpu_free_gib=%s "
                    "confirmatory=false"
                    % (
                        status.get("run_status", "uninitialized"),
                        status.get("plan_ready", False),
                        candidates.get("confirmed_rows", 0),
                        gpu.get("free_gib", "unknown"),
                    )
                )
            return 0
        if args.command == "run":
            if args.port < 0 or args.port > 65535:
                raise PilotConfigError("--port must be 0 or a valid TCP port")
            if args.bootstrap < 100:
                raise PilotConfigError("--bootstrap must be at least 100")
            summary = execute_micro_pilot(
                config_path,
                output=args.out.expanduser().resolve(),
                args=args,
            )
            print(
                "micro-pilot complete: confirmatory=false gate_passed=%s summary=%s"
                % (
                    bool(summary["runner_decision"]["passed"]),
                    summary["runner_decision"]["summary"],
                )
            )
            return 0
        raise AssertionError("unhandled command %s" % args.command)
    except ResourceUnavailable as error:
        print("micro-pilot waiting for resources: %s" % error, file=sys.stderr)
        return 3
    except (PilotConfigError, ValueError, subprocess.CalledProcessError) as error:
        print("micro-pilot failed: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
