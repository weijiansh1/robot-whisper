#!/usr/bin/env python3
"""Capture frozen event inputs with one model migrated across physical GPUs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "himoe-route-capture"
SCHEMA = "himoe.rich_event_functional_capture.v1"
MIN_ANON_HEADROOM_BYTES = 3 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_snapshot(*, enforce_headroom: bool) -> dict[str, int]:
    root = Path("/sys/fs/cgroup")
    maximum_text = (root / "memory.max").read_text().strip()
    if maximum_text == "max":
        raise RuntimeError("capture requires a finite cgroup memory limit")
    maximum = int(maximum_text)
    current = int((root / "memory.current").read_text().strip())
    stats = {}
    for line in (root / "memory.stat").read_text().splitlines():
        key, value = line.split()
        stats[key] = int(value)
    anon = stats["anon"]
    anon_headroom = maximum - anon
    if enforce_headroom and anon_headroom < MIN_ANON_HEADROOM_BYTES:
        raise RuntimeError(
            "insufficient anonymous-memory headroom: %.2f GiB < %.2f GiB"
            % (anon_headroom / 1024**3, MIN_ANON_HEADROOM_BYTES / 1024**3)
        )
    events = {}
    for line in (root / "memory.events").read_text().splitlines():
        key, value = line.split()
        events[key] = int(value)
    return {
        "maximum": maximum,
        "current": current,
        "anon": anon,
        "anon_headroom": anon_headroom,
        "oom": events.get("oom", 0),
        "oom_kill": events.get("oom_kill", 0),
    }


def _power_limits(gpus: tuple[int, ...]) -> dict[str, dict[str, float]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit,power.max_limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available = {}
    for line in completed.stdout.splitlines():
        index, limit, maximum = (part.strip() for part in line.split(","))
        available[int(index)] = {
            "power_limit_w": float(limit),
            "power_max_limit_w": float(maximum),
        }
    result = {str(gpu): available[gpu] for gpu in gpus}
    wrong = {
        gpu: values
        for gpu, values in result.items()
        if values["power_limit_w"] != values["power_max_limit_w"]
    }
    if wrong:
        raise RuntimeError("GPU power limit is below hardware maximum: %s" % wrong)
    return result


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("gpus must be comma-separated integers") from error
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("gpus must be a non-empty unique list")
    return result


def _action(policy: Any, request: dict[str, Any]) -> tuple[np.ndarray, float]:
    from himoe_libero_bridge.protocol import ACTION_KEY, validate_action_response

    started = time.perf_counter()
    response = validate_action_response(policy.infer(dict(request)))
    return np.asarray(response[ACTION_KEY], np.float32), time.perf_counter() - started


def main() -> int:
    args = parse_args()
    preflight = _memory_snapshot(enforce_headroom=True)
    power = _power_limits(args.gpus)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("output directory is not empty: %s" % args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.out_dir / "snapshots"
    snapshots_dir.mkdir()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    for source in (
        CAPTURE_ROOT,
        Path("/home/jovyan/work/himoe-libero-wrist-fix/src"),
    ):
        sys.path.insert(0, str(source))

    import torch

    from audit_functional_snapshot_noop import _load_policy
    from himoe_functional_recorder import (
        HBFunctionalSnapshotRecorder,
        save_functional_record,
    )
    from himoe_libero_bridge.protocol import (
        FLOW_NOISE_KEY,
        IMAGE_KEY,
        PROMPT_KEY,
        STATE_KEY,
        WRIST_IMAGE_KEY,
    )

    if torch.cuda.device_count() != len(args.gpus):
        raise RuntimeError(
            "expected %d visible CUDA devices, found %d"
            % (len(args.gpus), torch.cuda.device_count())
        )
    with np.load(args.inputs, allow_pickle=False) as source:
        inputs = {key: np.asarray(source[key]) for key in source.files}
    count = len(inputs["row_id"])
    if count == 0 or any(len(value) != count for value in inputs.values()):
        raise RuntimeError("input arrays are empty or row-misaligned")
    input_audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    if int(input_audit["rendered_rows"]) != count:
        raise RuntimeError("input audit row count mismatch")
    restore_by_row = {int(row["row_id"]): row for row in input_audit["rows"]}

    torch.cuda.set_device(0)
    load_args = argparse.Namespace(
        checkpoint_root=args.checkpoint_root,
        upstream_root=args.upstream_root,
        suite="long",
        libero_wrist_layout="paper-right",
    )
    policy, load_audit = _load_policy(load_args)
    checkpoint_sha256 = str(policy.metadata["checkpoint_sha256"])
    rows_by_gpu = np.array_split(np.arange(count), len(args.gpus))
    records = []
    previous_device = 0
    journal_path = args.out_dir / "records.jsonl"
    try:
        for logical_gpu, (physical_gpu, row_indices) in enumerate(
            zip(args.gpus, rows_by_gpu, strict=True)
        ):
            if logical_gpu:
                policy._policy.model.to(torch.device("cuda:%d" % logical_gpu))
                torch.cuda.synchronize(logical_gpu)
                with torch.cuda.device(previous_device):
                    torch.cuda.empty_cache()
            torch.cuda.set_device(logical_gpu)
            recorder = HBFunctionalSnapshotRecorder(
                policy._policy.model,
                expected_denoise=10,
                sketch_dim=args.sketch_dim,
            ).attach()
            try:
                for index in row_indices:
                    row = int(index)
                    row_id = int(inputs["row_id"][row])
                    request = {
                        IMAGE_KEY: inputs["image"][row],
                        WRIST_IMAGE_KEY: inputs["wrist_image"][row],
                        STATE_KEY: inputs["state"][row],
                        PROMPT_KEY: "put both moka pots on the stove",
                        FLOW_NOISE_KEY: inputs["flow_noise"][row],
                    }
                    disabled_action, disabled_seconds = _action(policy, request)
                    recorder.begin(
                        episode_id=int(inputs["global_episode"][row]),
                        control_step=int(inputs["query_index"][row]),
                    )
                    captured_action, captured_seconds = _action(policy, request)
                    functional = recorder.end()
                    transparent = bool(np.array_equal(disabled_action, captured_action))
                    if not transparent:
                        raise RuntimeError(
                            "functional capture changed action at row %d (max abs %.9g)"
                            % (
                                row_id,
                                np.max(np.abs(disabled_action - captured_action)),
                            )
                        )
                    expected = inputs["expected_actions"][row]
                    snapshot_path = snapshots_dir / ("row_%04d.npz" % row_id)
                    save_functional_record(functional, snapshot_path)
                    restore = restore_by_row[row_id]
                    restore_pass = bool(
                        restore["position_restore_max_abs_error_m"]
                        <= input_audit["restore_tolerances"]["position_m"]
                        and restore["rotation_restore_max_abs_error_rad"]
                        <= input_audit["restore_tolerances"]["rotation_rad"]
                        and restore["gripper_restore_max_abs_error"]
                        <= input_audit["restore_tolerances"]["gripper"]
                        and restore["sim_restore_max_abs_error"]
                        <= input_audit["restore_tolerances"]["sim_state"]
                    )
                    record = {
                        "row_id": row_id,
                        "physical_gpu": int(physical_gpu),
                        "logical_gpu": int(logical_gpu),
                        "pair_id": int(inputs["pair_id"][row]),
                        "event": bool(inputs["event"][row]),
                        "relative_query": int(inputs["relative_query"][row]),
                        "global_episode": int(inputs["global_episode"][row]),
                        "query_index": int(inputs["query_index"][row]),
                        "restore_pass": restore_pass,
                        "capture_transparent_bitwise": transparent,
                        "original_action_bitwise": bool(
                            np.array_equal(captured_action, expected)
                        ),
                        "original_action_max_abs_error": float(
                            np.max(np.abs(captured_action - expected))
                        ),
                        "disabled_seconds": disabled_seconds,
                        "captured_seconds": captured_seconds,
                        "snapshot": str(snapshot_path.resolve()),
                        "snapshot_sha256": _sha256(snapshot_path),
                        "snapshot_file_bytes": snapshot_path.stat().st_size,
                        "snapshot_array_bytes": functional.array_bytes,
                    }
                    records.append(record)
                    with journal_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "gpu%d: captured %d rows (%d/%d total)"
                    % (physical_gpu, len(row_indices), len(records), count),
                    flush=True,
                )
            finally:
                recorder.close()
                del recorder
            previous_device = logical_gpu
    finally:
        del policy
        gc.collect()
        for logical_gpu in range(torch.cuda.device_count()):
            with torch.cuda.device(logical_gpu):
                torch.cuda.empty_cache()

    postflight = _memory_snapshot(enforce_headroom=False)
    summary = {
        "schema": SCHEMA,
        "passed": bool(
            len(records) == count
            and all(row["capture_transparent_bitwise"] for row in records)
            and postflight["oom_kill"] == preflight["oom_kill"]
        ),
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": _sha256(args.inputs),
        "input_audit": str(args.input_audit.resolve()),
        "input_audit_sha256": _sha256(args.input_audit),
        "gpus": list(args.gpus),
        "gpu_power_limits": power,
        "sketch_dim": int(args.sketch_dim),
        "checkpoint_sha256": checkpoint_sha256,
        "load_audit": load_audit,
        "preflight_memory": preflight,
        "postflight_memory": postflight,
        "rows": len(records),
        "restore_pass_rows": int(sum(row["restore_pass"] for row in records)),
        "capture_transparent_rows": int(
            sum(row["capture_transparent_bitwise"] for row in records)
        ),
        "original_action_bitwise_rows": int(
            sum(row["original_action_bitwise"] for row in records)
        ),
        "original_action_max_abs_error": float(
            max(row["original_action_max_abs_error"] for row in records)
        ),
        "mean_disabled_seconds": float(
            np.mean([row["disabled_seconds"] for row in records])
        ),
        "mean_captured_seconds": float(
            np.mean([row["captured_seconds"] for row in records])
        ),
        "records_jsonl": str(journal_path.resolve()),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/home/jovyan/.cache/himoe-libero-bridge/checkpoints"),
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(
            "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
