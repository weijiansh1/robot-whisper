#!/usr/bin/env python3
"""Verify original checkpoint endpoints and absence of collection-owned processes."""

import argparse
import contextlib
import json
from pathlib import Path
import shutil

import probe_preloaded_models as probe
from collection_protocol import HERE, MANIFEST_SHA256, PARAMETERS_SHA256
from collection_storage import atomic_json, digest


def run(args):
    owned = set()
    runs = []
    for directory in args.runs:
        summary = json.loads((directory / "summary.json").read_text())
        owned.update(summary.get("temporary_model_pids", {}).values())
        owned.update(summary.get("temporary_replica_pids", []))
        owned.update(task["pid"] for task in summary["tasks"] if task.get("pid"))
        owned.update(row["controller_pid"] for row in summary.get("mps_start", {}).values())
        for row in summary.get("mps_clients", {}).values():
            owned.update(row["server_pids"])
        runs.append(dict(directory=str(directory), status=summary["status"], mps_cleanup=summary.get("mps_cleanup")))
    live = [pid for pid in sorted(owned) if Path("/proc/%d" % pid).exists()]
    endpoints = []
    gpus = getattr(args, "gpus", probe.ALLOWED_GPUS)
    for gpu in gpus:
        with contextlib.ExitStack() as stack:
            for model in probe.MODELS:
                connection, metadata = probe.open_endpoint(stack, "127.0.0.1", gpu, model)
                probe.check_checkpoint(metadata, model)
                row = dict(gpu=gpu, model=model, port=probe.port_for(gpu, model), pid=metadata["bundle_process_pid"],
                    checkpoint_sha256=metadata["checkpoint_sha256"], normalization_sha256=metadata["normalization_stats_sha256"])
                if args.infer:
                    response, latency = probe.infer(connection, probe.observation(model, 71000 + gpu, capture=True), model)
                    row.update(inference_seconds=latency, action_shape=list(response["actions"].shape))
                endpoints.append(row)
    manifest_sha = digest(HERE / "design/collection_manifest.csv")
    parameters_sha = digest(HERE / "design/frozen_alarm_comparison_20260908/profiles/parameters.json")
    passed = not live and manifest_sha == MANIFEST_SHA256 and parameters_sha == PARAMETERS_SHA256
    report = dict(status="passed" if passed else "failed", utc=probe.now(), endpoints=endpoints,
        original_pids=sorted({row["pid"] for row in endpoints}), original_endpoint_count=len(endpoints),
        live_collection_pids=live, checked_collection_pids=sorted(owned), runs=runs,
        formal_manifest_sha256=manifest_sha, frozen_parameters_sha256=parameters_sha,
        disk_free_bytes=shutil.disk_usage(HERE).free, memory_current_bytes=int(Path("/sys/fs/cgroup/memory.current").read_text()),
        allowed_gpus=list(gpus), excluded_gpu=6, gpu_sample=probe.telemetry(gpus))
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "original_endpoint_count", "original_pids", "live_collection_pids", "disk_free_bytes")}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--gpus", type=probe.validate_gpus, default=probe.ALLOWED_GPUS)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
