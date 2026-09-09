#!/usr/bin/env python3
"""Run the frozen counterfactual terminal-Q Gate 1 experiment.

The runner publishes candidates strictly in plan order so every candidate has
one recoverable, contiguous row in the route/hidden/flow stores.  Continuation
queries never advance those stores.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_forks_v2 import atomic_json, load_npz, verify_artifact
from bestofn_artifacts import validate_candidate, validate_continuation
from bestofn_protocol import (
    CANDIDATE_SCHEMA,
    CONTINUATION_SCHEMA,
    candidate_dir,
    candidate_query_id,
    continuation_shard_dir,
    load_config,
    load_json,
    sha256_file,
    validate_plan,
)
from run_behavior_micro_pilot import (
    ResourceUnavailable,
    _candidate_store_audit,
    _cgroup_free_gib,
    _gpu_free_gib,
    _libero_environment,
    _runtime_paths,
    _server_session,
)


HERE = Path(__file__).resolve().parent
RUN_SCHEMA = "himoe.bestofn.run.v1"
IMPLEMENTATION_FILES = (
    "bestofn_protocol.py",
    "bestofn_artifacts.py",
    "build_bestofn_plan.py",
    "capture_bestofn.py",
    "analyze_bestofn_oracle.py",
    "run_bestofn_experiment.py",
    "serve_with_recorder.py",
    "serve_flow_trace.py",
    "himoe_candidate_capture_protocol.py",
    "himoe_router_recorder.py",
    "himoe_route_store.py",
    "himoe_hidden_store.py",
    "himoe_flow_trajectory_store.py",
    "himoe_vlm_feature_tracer.py",
    "bestofn_critic_features.py",
    "build_bestofn_critic_dataset.py",
    "bestofn_critic.py",
    "train_bestofn_critics.py",
    "branch_snapshot.py",
    "behavior_forks_v2.py",
    "capture_behavior_forks.py",
    "capture_behavior_study.py",
    "BESTOFN_COUNTERFACTUAL_Q_EXPERIMENT.md",
)


class RunError(RuntimeError):
    """The frozen run or its published prefix is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_protocol(config_path: Path, plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    plan = load_json(plan_path)
    validate_plan(plan, config)
    if plan.get("config_sha256") != sha256_file(config_path):
        raise RunError("plan was generated from another frozen config")
    return config, plan


def _implementation_digests() -> dict[str, str]:
    result = {}
    for name in IMPLEMENTATION_FILES:
        path = HERE / name
        if not path.is_file():
            raise RunError(f"implementation file is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _preflight(config_path: Path, plan_path: Path, *, verify_sources: bool) -> dict[str, Any]:
    config, plan = _load_protocol(config_path, plan_path)
    paths = _runtime_paths(config_path)
    required = (
        paths["model_python"],
        paths["libero_python"],
        paths["checkpoint"] / "pytorch_model.pth",
        paths["libero_root"],
        paths["upstream_root"],
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RunError("runtime dependency is missing: " + ", ".join(missing))
    checkpoint_sha = sha256_file(paths["checkpoint"] / "pytorch_model.pth")
    if checkpoint_sha != config["checkpoint"]["sha256"]:
        raise RunError("deployed checkpoint differs from the frozen config")
    source_files = {}
    for state in plan["states"]:
        source_files.setdefault(
            str(state["source_episode_file"]), str(state["source_episode_sha256"])
        )
    if verify_sources:
        for raw, expected in source_files.items():
            path = Path(raw)
            if not path.is_file() or sha256_file(path) != expected:
                raise RunError(f"source replay changed after snapshot planning: {path}")
    free_bytes = shutil.disk_usage(plan_path.parent).free
    if free_bytes < 12 * 1024**3:
        raise ResourceUnavailable(
            f"only {free_bytes / 1024**3:.2f} GiB disk is free; 12 GiB is required"
        )
    horizon = int(config["proposal"]["action_chunk_horizon"])
    repeats = int(config["continuation"]["initial_repeats"])
    terminal_steps = int(config["continuation"]["terminal_environment_steps"])
    continuation_query_upper_bound = sum(
        int(state["candidate_count"])
        * repeats
        * max(
            0,
            (terminal_steps - (int(state["fork_step"]) + 1) * horizon + horizon - 1)
            // horizon,
        )
        for state in plan["states"]
    )
    planned_candidates = sum(int(row["candidate_count"]) for row in plan["states"])
    return {
        "config_sha256": sha256_file(config_path),
        "plan_sha256": sha256_file(plan_path),
        "checkpoint_sha256": checkpoint_sha,
        "planned_snapshots": len(plan["states"]),
        "planned_candidates": planned_candidates,
        "r4_continuation_query_upper_bound": continuation_query_upper_bound,
        "gate1_model_query_upper_bound": planned_candidates
        + continuation_query_upper_bound,
        "gate1_mujoco_action_upper_bound": continuation_query_upper_bound * horizon,
        "unique_source_episodes": len(source_files),
        "source_hashes_verified": bool(verify_sources),
        "disk_free_gib": free_bytes / 1024**3,
    }


def _manifest(output: Path, config_path: Path, plan_path: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    path = output / "run_manifest.json"
    if path.is_file():
        value = load_json(path)
        if value.get("schema") != RUN_SCHEMA:
            raise RunError("output has an incompatible run manifest")
        if value.get("config_sha256") != audit["config_sha256"]:
            raise RunError("output was initialized with another config")
        if value.get("plan_sha256") != audit["plan_sha256"]:
            raise RunError("output was initialized with another snapshot plan")
        if value.get("implementation_sha256") != _implementation_digests():
            raise RunError(
                "collection implementation changed after this run was initialized"
            )
        return value
    output.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": RUN_SCHEMA,
        "confirmatory": False,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "initialized",
        "config_file": str(config_path),
        "config_sha256": audit["config_sha256"],
        "plan_file": str(plan_path),
        "plan_sha256": audit["plan_sha256"],
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "implementation_sha256": _implementation_digests(),
        "audit": dict(audit),
        "progress": {},
    }
    atomic_json(path, value)
    return value


def _save_manifest(output: Path, manifest: dict[str, Any], status: str, **progress: Any) -> None:
    manifest["status"] = status
    manifest["updated_utc"] = _utc_now()
    manifest.setdefault("progress", {}).update(progress)
    atomic_json(output / "run_manifest.json", manifest)


def _candidate_prefix(output: Path, plan: Mapping[str, Any], config_path: Path, plan_path: Path) -> dict[str, Any]:
    prefix = 0
    artifacts = 0
    run_id = None
    server_identity = None
    gap_seen = False
    config_sha = sha256_file(config_path)
    plan_sha = sha256_file(plan_path)
    for state in plan["states"]:
        descriptor = candidate_dir(output, str(state["state_id"])) / "artifact.json"
        if not descriptor.is_file():
            gap_seen = True
            continue
        if gap_seen:
            raise RunError("candidate artifacts are not one contiguous plan prefix")
        metadata = verify_artifact(descriptor, CANDIDATE_SCHEMA)
        arrays = load_npz(descriptor.parent / metadata["files"]["data"]["file"])
        validate_candidate(arrays, metadata)
        k = int(state["candidate_count"])
        if int(metadata["candidate_count"]) != k:
            raise RunError(f"candidate count differs from plan: {descriptor}")
        if metadata.get("config_sha256") != config_sha or metadata.get("plan_sha256") != plan_sha:
            raise RunError(f"candidate artifact uses another protocol: {descriptor}")
        rows = np.asarray(arrays["server_trace_rows"], dtype=np.int64)
        if not np.array_equal(rows, np.arange(prefix, prefix + k)):
            raise RunError(f"candidate route rows do not extend the published prefix: {descriptor}")
        expected_queries = np.asarray(
            [candidate_query_id(int(state["snapshot_index"]), candidate) for candidate in range(k)],
            dtype=np.int32,
        )
        if not np.array_equal(arrays["query_ids"], expected_queries):
            raise RunError(f"candidate query ids differ from the frozen coordinates: {descriptor}")
        artifact_run_id = str(metadata["route_store_id"])
        if run_id is None:
            run_id = artifact_run_id
        elif run_id != artifact_run_id:
            raise RunError("published candidates span multiple route stores")
        identity = dict(metadata["server_identity"])
        if server_identity is None:
            server_identity = identity
        elif server_identity != identity:
            raise RunError("published candidates span multiple policy implementations")
        prefix += k
        artifacts += 1
    return {
        "confirmed_rows": prefix,
        "candidate_artifacts": artifacts,
        "capture_run_id": run_id,
        "server_identity": server_identity,
        "complete": artifacts == len(plan["states"]),
    }


def _continuation_valid(path: Path) -> bool:
    descriptor = path / "artifact.json"
    if not descriptor.is_file():
        return False
    metadata = verify_artifact(descriptor, CONTINUATION_SCHEMA)
    arrays = load_npz(descriptor.parent / metadata["files"]["data"]["file"])
    validate_continuation(arrays, metadata)
    return True


def _recover_incomplete(path: Path, output: Path) -> None:
    if not path.exists() or (path / "artifact.json").is_file():
        return
    recovery = output / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = recovery / f"{path.name}.incomplete-{stamp}"
    os.replace(path, destination)


def _status(config_path: Path, plan_path: Path, output: Path) -> dict[str, Any]:
    config, plan = _load_protocol(config_path, plan_path)
    prefix = _candidate_prefix(output, plan, config_path, plan_path)
    initial = int(config["continuation"]["initial_repeats"])
    topup = int(config["continuation"]["topup_repeats"])
    r4 = 0
    r8 = 0
    for state in plan["states"]:
        state_id = str(state["state_id"])
        if _continuation_valid(continuation_shard_dir(output, state_id, 0, initial)):
            r4 += 1
        if topup > initial and _continuation_valid(
            continuation_shard_dir(output, state_id, initial, topup)
        ):
            r8 += 1
    store = _candidate_store_audit(output / "candidate_stores")
    if store["present"] and int(store["rows"]) < int(prefix["confirmed_rows"]):
        raise RunError("candidate stores are shorter than published artifacts")
    gpu = []
    for index in ("0", "1"):
        try:
            gpu.append({"gpu": index, "free_gib": _gpu_free_gib(index)})
        except Exception as error:  # status remains useful without nvidia-smi
            gpu.append({"gpu": index, "error": str(error)})
    return {
        **prefix,
        "initial_continuation_artifacts": r4,
        "topup_continuation_artifacts": r8,
        "planned_snapshots": len(plan["states"]),
        "route_store": store,
        "gpu": gpu,
        "cgroup_free_gib": _cgroup_free_gib(),
    }


def _run_checked(argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    print("+ " + " ".join(str(value) for value in argv), flush=True)
    subprocess.run(list(argv), cwd=str(cwd), env=dict(environment), check=True)


def _capture_command(
    python: Path,
    config_path: Path,
    plan_path: Path,
    output: Path,
    state_id: str,
    server: Any,
    expected_rows: int,
    libero_root: Path,
    *,
    repeat_interval: tuple[int, int] | None = None,
) -> list[str]:
    common = [
        str(python),
        str(HERE / "capture_bestofn.py"),
        "--config",
        str(config_path),
        "--plan",
        str(plan_path),
        "--out",
        str(output),
    ]
    if repeat_interval is None:
        return common + [
            "candidate",
            "--state-id",
            state_id,
            "--host",
            server.host,
            "--port",
            str(server.port),
            "--libero-root",
            str(libero_root),
            "--route-store-id",
            server.capture_run_id,
            "--expected-recorder-rows",
            str(expected_rows),
        ]
    start, stop = repeat_interval
    return common + [
        "continuation-shard",
        "--state-id",
        state_id,
        "--repeat-start",
        str(start),
        "--repeat-stop",
        str(stop),
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--libero-root",
        str(libero_root),
        "--expected-recorder-rows",
        str(expected_rows),
    ]


def collect_gate1(args: argparse.Namespace) -> int:
    config, plan = _load_protocol(args.config, args.plan)
    audit = _preflight(args.config, args.plan, verify_sources=True)
    manifest = _manifest(args.out, args.config, args.plan, audit)
    prefix = _candidate_prefix(args.out, plan, args.config, args.plan)
    initial = int(config["continuation"]["initial_repeats"])
    missing = []
    for state in plan["states"]:
        state_id = str(state["state_id"])
        candidate_missing = not (candidate_dir(args.out, state_id) / "artifact.json").is_file()
        continuation_missing = not (
            continuation_shard_dir(args.out, state_id, 0, initial) / "artifact.json"
        ).is_file()
        if candidate_missing or continuation_missing:
            missing.append(state_id)
    if not missing:
        return analyze_gate1(args, manifest=manifest)
    gpu = args.gpu
    if gpu == "auto":
        gpu = max(("0", "1"), key=_gpu_free_gib)
    processed = 0
    try:
        with _server_session(
            args.config,
            output=args.out,
            confirmed_rows=int(prefix["confirmed_rows"]),
            confirmed_run_id=prefix["capture_run_id"],
            gpu=gpu,
            min_gpu_free_gib=args.min_gpu_free_gib,
            min_cgroup_free_gib=args.min_cgroup_free_gib,
            host=args.host,
            port=args.port,
            startup_timeout=args.startup_timeout,
        ) as server:
            paths = _runtime_paths(args.config)
            environment = _libero_environment(paths)
            for state in plan["states"]:
                state_id = str(state["state_id"])
                candidate_target = candidate_dir(args.out, state_id)
                continuation_target = continuation_shard_dir(args.out, state_id, 0, initial)
                changed = False
                if not (candidate_target / "artifact.json").is_file():
                    _recover_incomplete(candidate_target, args.out)
                    before = _candidate_prefix(args.out, plan, args.config, args.plan)
                    _run_checked(
                        _capture_command(
                            paths["libero_python"], args.config, args.plan, args.out,
                            state_id, server, int(before["confirmed_rows"]), paths["libero_root"]
                        ),
                        cwd=HERE,
                        environment=environment,
                    )
                    after = _candidate_prefix(args.out, plan, args.config, args.plan)
                    expected = int(before["confirmed_rows"]) + int(state["candidate_count"])
                    if int(after["confirmed_rows"]) != expected:
                        raise RunError("candidate process did not publish exactly one planned pool")
                    _candidate_store_audit(server.store_root, expected_rows=expected)
                    changed = True
                current = _candidate_prefix(args.out, plan, args.config, args.plan)
                if not (continuation_target / "artifact.json").is_file():
                    _recover_incomplete(continuation_target, args.out)
                    _run_checked(
                        _capture_command(
                            paths["libero_python"], args.config, args.plan, args.out,
                            state_id, server, int(current["confirmed_rows"]), paths["libero_root"],
                            repeat_interval=(0, initial),
                        ),
                        cwd=HERE,
                        environment=environment,
                    )
                    if not _continuation_valid(continuation_target):
                        raise RunError("continuation process did not publish a valid R=4 shard")
                    _candidate_store_audit(
                        server.store_root, expected_rows=int(current["confirmed_rows"])
                    )
                    changed = True
                if changed:
                    processed += 1
                    status = _status(args.config, args.plan, args.out)
                    _save_manifest(args.out, manifest, "collecting_gate1", last_state=state_id, **status)
                    if args.max_new_states and processed >= args.max_new_states:
                        print(f"stopped after {processed} newly completed state(s); run is resumable")
                        return 0
    except ResourceUnavailable:
        _save_manifest(args.out, manifest, "waiting_for_gpu", **_status(args.config, args.plan, args.out))
        raise
    return analyze_gate1(args, manifest=manifest)


def analyze_gate1(args: argparse.Namespace, *, manifest: dict[str, Any] | None = None) -> int:
    config, plan = _load_protocol(args.config, args.plan)
    status = _status(args.config, args.plan, args.out)
    if status["candidate_artifacts"] != len(plan["states"]):
        raise RunError("Gate 1 analysis requires candidates for all planned snapshots")
    if status["initial_continuation_artifacts"] != len(plan["states"]):
        raise RunError("Gate 1 analysis requires R=4 outcomes for all planned snapshots")
    paths = _runtime_paths(args.config)
    out = args.out / "analysis" / "gate1_r4"
    _run_checked(
        [
            sys.executable,
            str(HERE / "analyze_bestofn_oracle.py"),
            "--config", str(args.config),
            "--plan", str(args.plan),
            "--run-root", str(args.out),
            "--out-dir", str(out),
            "--bootstrap", str(args.bootstrap),
        ],
        cwd=HERE,
        environment=dict(os.environ),
    )
    summary = load_json(out / "summary.json")
    if manifest is None:
        audit = _preflight(args.config, args.plan, verify_sources=False)
        manifest = _manifest(args.out, args.config, args.plan, audit)
    decision = "gate1_passed" if summary["gate_1"]["passed"] else "gate1_stopped"
    _save_manifest(
        args.out,
        manifest,
        decision,
        gate1_summary=str(out / "summary.json"),
        gate1_passed=bool(summary["gate_1"]["passed"]),
        **status,
    )
    return 0


def collect_topup(args: argparse.Namespace) -> int:
    config, plan = _load_protocol(args.config, args.plan)
    audit = _preflight(args.config, args.plan, verify_sources=True)
    manifest = _manifest(args.out, args.config, args.plan, audit)
    gate_dir = args.out / "analysis" / "gate1_r4"
    gate_path = gate_dir / "summary.json"
    topup_path = gate_dir / "topup_plan.json"
    if not gate_path.is_file() or not topup_path.is_file():
        raise RunError("adaptive R=8 labels require the completed R=4 Gate 1 analysis")
    gate = load_json(gate_path)
    if gate.get("gate_1", {}).get("passed") is not True:
        raise RunError("Gate 1 did not pass; critic-label top-up is not authorized")
    topup_plan = load_json(topup_path)
    state_ids = set(str(value) for value in topup_plan["states"])
    known = {str(row["state_id"]) for row in plan["states"]}
    if not state_ids <= known:
        raise RunError("top-up plan names a state outside the frozen snapshot panel")
    initial = int(config["continuation"]["initial_repeats"])
    target_repeats = int(config["continuation"]["topup_repeats"])
    missing = [
        state_id
        for state_id in sorted(state_ids)
        if not (
            continuation_shard_dir(args.out, state_id, initial, target_repeats)
            / "artifact.json"
        ).is_file()
    ]
    prefix = _candidate_prefix(args.out, plan, args.config, args.plan)
    if not prefix["complete"]:
        raise RunError("R=8 top-up requires the complete candidate prefix")
    processed = 0
    if missing:
        gpu = args.gpu
        if gpu == "auto":
            gpu = max(("0", "1"), key=_gpu_free_gib)
        try:
            with _server_session(
                args.config,
                output=args.out,
                confirmed_rows=int(prefix["confirmed_rows"]),
                confirmed_run_id=prefix["capture_run_id"],
                gpu=gpu,
                min_gpu_free_gib=args.min_gpu_free_gib,
                min_cgroup_free_gib=args.min_cgroup_free_gib,
                host=args.host,
                port=args.port,
                startup_timeout=args.startup_timeout,
            ) as server:
                paths = _runtime_paths(args.config)
                environment = _libero_environment(paths)
                for state_id in missing:
                    target = continuation_shard_dir(
                        args.out, state_id, initial, target_repeats
                    )
                    _recover_incomplete(target, args.out)
                    _run_checked(
                        _capture_command(
                            paths["libero_python"],
                            args.config,
                            args.plan,
                            args.out,
                            state_id,
                            server,
                            int(prefix["confirmed_rows"]),
                            paths["libero_root"],
                            repeat_interval=(initial, target_repeats),
                        ),
                        cwd=HERE,
                        environment=environment,
                    )
                    if not _continuation_valid(target):
                        raise RunError("top-up process did not publish a valid R=8 shard")
                    _candidate_store_audit(
                        server.store_root,
                        expected_rows=int(prefix["confirmed_rows"]),
                    )
                    processed += 1
                    _save_manifest(
                        args.out,
                        manifest,
                        "collecting_critic_labels",
                        last_topup_state=state_id,
                        completed_topup_states=processed,
                        planned_topup_states=len(state_ids),
                    )
                    if args.max_new_states and processed >= args.max_new_states:
                        print(
                            f"stopped after {processed} new R=8 shard(s); run is resumable"
                        )
                        return 0
        except ResourceUnavailable:
            _save_manifest(
                args.out,
                manifest,
                "waiting_for_gpu_topup",
                **_status(args.config, args.plan, args.out),
            )
            raise
    label_dir = args.out / "analysis" / "adaptive_labels"
    _run_checked(
        [
            sys.executable,
            str(HERE / "analyze_bestofn_oracle.py"),
            "--config",
            str(args.config),
            "--plan",
            str(args.plan),
            "--run-root",
            str(args.out),
            "--out-dir",
            str(label_dir),
            "--bootstrap",
            str(args.bootstrap),
        ],
        cwd=HERE,
        environment=dict(os.environ),
    )
    _save_manifest(
        args.out,
        manifest,
        "critic_labels_ready",
        gate1_passed=True,
        gate1_summary=str(gate_path),
        adaptive_label_summary=str(label_dir / "summary.json"),
        planned_topup_states=len(state_ids),
        **_status(args.config, args.plan, args.out),
    )
    return 0


def train_critics(args: argparse.Namespace) -> int:
    _config, _plan = _load_protocol(args.config, args.plan)
    audit = _preflight(args.config, args.plan, verify_sources=False)
    manifest = _manifest(args.out, args.config, args.plan, audit)
    gate_path = args.out / "analysis" / "gate1_r4" / "summary.json"
    if not gate_path.is_file() or load_json(gate_path).get("gate_1", {}).get("passed") is not True:
        raise RunError("critic training is unauthorized until the R=4 Gate 1 passes")
    dataset = args.out / "critic_dataset"
    results = args.out / "analysis" / "critics"
    _run_checked(
        [
            sys.executable,
            str(HERE / "build_bestofn_critic_dataset.py"),
            "--config",
            str(args.config),
            "--plan",
            str(args.plan),
            "--run-root",
            str(args.out),
            "--out",
            str(dataset),
        ],
        cwd=HERE,
        environment=dict(os.environ),
    )
    _save_manifest(
        args.out,
        manifest,
        "critic_dataset_ready",
        critic_dataset=str(dataset / "dataset_manifest.json"),
    )
    _run_checked(
        [
            sys.executable,
            str(HERE / "train_bestofn_critics.py"),
            "--config",
            str(args.config),
            "--dataset",
            str(dataset),
            "--out",
            str(results),
            "--device",
            args.device,
        ],
        cwd=HERE,
        environment=dict(os.environ),
    )
    result = load_json(results / "results.json")
    _save_manifest(
        args.out,
        manifest,
        "gate2_passed" if result["gate_2"]["passed"] else "gate2_stopped",
        critic_results=str(results / "results.json"),
        gate2_passed=bool(result["gate_2"]["passed"]),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "bestofn_experiment_config.json")
    parser.add_argument("--plan", type=Path, default=HERE / "runs/bestofn-critic-20260825/plan.json")
    parser.add_argument("--out", type=Path, default=HERE / "runs/bestofn-critic-20260825")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--skip-source-hashes", action="store_true")
    run = sub.add_parser("gate1")
    run.add_argument("--gpu", default="auto")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--min-gpu-free-gib", type=float, default=24.0)
    run.add_argument("--min-cgroup-free-gib", type=float, default=3.0)
    run.add_argument("--startup-timeout", type=float, default=900.0)
    run.add_argument("--max-new-states", type=int, default=0)
    run.add_argument("--bootstrap", type=int, default=10000)
    analysis = sub.add_parser("analyze-gate1")
    analysis.add_argument("--bootstrap", type=int, default=10000)
    topup = sub.add_parser("topup")
    topup.add_argument("--gpu", default="auto")
    topup.add_argument("--host", default="127.0.0.1")
    topup.add_argument("--port", type=int, default=0)
    topup.add_argument("--min-gpu-free-gib", type=float, default=24.0)
    topup.add_argument("--min-cgroup-free-gib", type=float, default=3.0)
    topup.add_argument("--startup-timeout", type=float, default=900.0)
    topup.add_argument("--max-new-states", type=int, default=0)
    topup.add_argument("--bootstrap", type=int, default=10000)
    critics = sub.add_parser("critics")
    critics.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    try:
        if args.command == "status":
            print(json.dumps(_status(args.config, args.plan, args.out), indent=2, sort_keys=True))
            return 0
        if args.command == "preflight":
            print(
                json.dumps(
                    _preflight(
                        args.config, args.plan, verify_sources=not args.skip_source_hashes
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "gate1":
            return collect_gate1(args)
        if args.command == "analyze-gate1":
            return analyze_gate1(args)
        if args.command == "topup":
            return collect_topup(args)
        if args.command == "critics":
            return train_critics(args)
        raise AssertionError(args.command)
    except ResourceUnavailable as error:
        print(f"RESOURCE_UNAVAILABLE: {error}", file=sys.stderr)
        return 3
    except (RunError, subprocess.CalledProcessError) as error:
        print(f"RUN_FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
