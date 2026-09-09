#!/usr/bin/env python3
"""Verify the existing model matrix and run a bounded inference load probe.

The client creates no CUDA context, never connects to physical GPU 6 endpoints,
and never starts, stops, reloads, or changes power settings on a model server.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime
import hashlib
import io
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = pathlib.Path(__file__).resolve().parent
ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)
MODELS = ("goal", "spatial", "object", "long", "calvin")
for source in reversed((
    HERE.parent / "online-servers",
    pathlib.Path("/home/jovyan/work/himoe-libero-wrist-fix/src"),
    pathlib.Path("/home/jovyan/work/himoe-calvin-alignment/src"),
    pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src"),
)):
    sys.path.insert(0, str(source))

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
from himoe_libero_bridge.suites import get_suite
from himoe_calvin_alignment import constants as calvin
from verify_model_bundles import _verify_bundle_identity


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_gpus(values):
    gpus = tuple(int(value) for value in values.split(","))
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in ALLOWED_GPUS for g in gpus):
        raise ValueError("Only physical GPUs 0,1,2,3,4,5,7 are allowed; GPU 6 is excluded")
    return gpus


def port_for(gpu, model):
    if gpu not in ALLOWED_GPUS:
        raise ValueError("GPU is outside the allowlist")
    return 8800 + gpu * 10 + MODELS.index(model)


def telemetry(gpus):
    result = subprocess.run([
        "nvidia-smi", "-i", ",".join(map(str, gpus)),
        "--query-gpu=index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=10)
    rows = []
    for fields in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        rows.append(dict(zip(
            ("gpu", "utilization_percent", "memory_mib", "power_w", "power_limit_w", "temperature_c"),
            (int(fields[0]), *[float(value) for value in fields[1:]]),
        )))
    return {"time": now(), "monotonic": time.monotonic(), "gpus": rows}


def wait_ports(gpus, seconds):
    required = {port_for(gpu, model) for gpu in gpus for model in MODELS}
    deadline = time.monotonic() + seconds
    previous = None
    while True:
        result = subprocess.run(["ss", "-H", "-ltn"], capture_output=True, text=True, check=True, timeout=10)
        ports = {int(line.split()[3].rsplit(":", 1)[1]) for line in result.stdout.splitlines()}
        ready = len(required & ports)
        if ready != previous:
            print(f"{now()} listening endpoints {ready}/{len(required)}", flush=True)
            previous = ready
        if ready == len(required):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Model endpoints are not ready: {sorted(required - ports)}")
        time.sleep(2)


def open_endpoint(stack, host, gpu, model):
    port = port_for(gpu, model)
    connection = stack.enter_context(connect(
        f"ws://{host}:{port}", compression=None, max_size=None,
        open_timeout=10, close_timeout=5, ping_interval=None,
    ))
    frame = connection.recv(timeout=15)
    if isinstance(frame, str):
        raise RuntimeError(frame)
    return connection, msgpack_numpy.unpackb(frame)


def observation(model, seed, capture=False):
    rng = np.random.default_rng(seed)
    state_dim = 15 if model == "calvin" else 8
    request = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(state_dim, dtype=np.float32),
        "prompt": "pick up the object and place it on the table",
        "flow/noise": rng.standard_normal((10, 24)).astype(np.float32),
    }
    if model != "calvin":
        request["routing/capture"] = capture
    return request


def infer(connection, request, model):
    started = time.monotonic()
    connection.send(msgpack_numpy.Packer().pack(request))
    frame = connection.recv(timeout=180)
    if isinstance(frame, str):
        raise RuntimeError(frame)
    response = msgpack_numpy.unpackb(frame)
    actions = np.asarray(response["actions"])
    expected = (10, 8 if model == "calvin" else 7)
    if actions.shape != expected or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid {model} actions: {actions.shape}")
    return response, time.monotonic() - started


def check_checkpoint(metadata, model):
    expected_hash = calvin.CHECKPOINT_SHA256 if model == "calvin" else get_suite(model).weights_sha256
    expected_bytes = calvin.CHECKPOINT_BYTES if model == "calvin" else get_suite(model).weights_bytes
    if metadata.get("checkpoint_sha256") != expected_hash or metadata.get("checkpoint_bytes") != expected_bytes:
        raise RuntimeError(f"{model}: checkpoint identity mismatch")
    audit = metadata["checkpoint_load_audit"]
    checkpoint = pathlib.Path(audit["checkpoint_path"])
    if checkpoint.stat().st_size != expected_bytes or not audit["strict"]:
        raise RuntimeError(f"{model}: checkpoint load audit is incomplete")
    if audit["state_key_count"] <= 0 or audit["aliased_tensor_count"] != audit["state_key_count"]:
        raise RuntimeError(f"{model}: checkpoint parameters are incomplete")
    asset = calvin.NORMALIZATION_ASSET if model == "calvin" else get_suite(model).normalization_asset
    stats = checkpoint.parent / asset / "meta/stats.json"
    raw = stats.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["normalization_stats_sha256"]:
        raise RuntimeError(f"{model}: normalization identity mismatch")
    parsed = json.loads(raw)
    if not np.isfinite(np.asarray(parsed["actions"]["std"], dtype=float)).all():
        raise RuntimeError(f"{model}: invalid normalization")


def verify_gpu(host, gpu, logical):
    checked = []
    with contextlib.ExitStack() as stack:
        for model in MODELS:
            connection, metadata = open_endpoint(stack, host, gpu, model)
            _verify_bundle_identity(metadata, port_for(gpu, model), gpu, logical, model)
            check_checkpoint(metadata, model)
            response, latency = infer(connection, observation(model, 71000 + gpu, capture=True), model)
            shapes = {key: list(value.shape) for key, value in response.items() if isinstance(value, np.ndarray)}
            if model != "calvin":
                ids = np.asarray(response["routing/expert_ids"])
                weights = np.asarray(response["routing/expert_weights"])
                if ids.shape != (10, 8, 10, 4) or weights.shape != ids.shape:
                    raise RuntimeError("Unexpected routing shape")
                if not np.isfinite(weights).all() or not np.allclose(weights.sum(-1), 1, atol=1e-3):
                    raise RuntimeError("Invalid routing combine weights")
            checked.append({
                "gpu": gpu, "logical_gpu": logical, "model": model,
                "port": port_for(gpu, model), "metadata": metadata,
                "response_array_shapes": shapes, "health_query_seconds": latency,
                "full_router_probabilities_returned": "recorder/hb_router_probs" in response,
            })
            print(f"verified gpu{gpu} {model} port={port_for(gpu, model)} {latency:.3f}s", flush=True)
    return checked


def load_probe(args, clients_per_gpu):
    stop = threading.Event()
    total_workers = len(args.gpus) * clients_per_gpu
    barrier = threading.Barrier(total_workers + 1)
    samples = []

    def worker(gpu, client):
        models = MODELS[:4]
        latencies = []
        with contextlib.ExitStack() as stack:
            try:
                connections = {model: open_endpoint(stack, args.host, gpu, model)[0] for model in models}
                requests = {model: observation(model, 72000 + gpu * 10 + client, args.capture_routing) for model in models}
                barrier.wait(timeout=30)
                while not stop.is_set():
                    model = models[(len(latencies) + client) % len(models)]
                    _response, latency = infer(connections[model], requests[model], model)
                    latencies.append(latency)
            except BaseException:
                barrier.abort()
                stop.set()
                raise
        return {"gpu": gpu, "client": client, "model_cycle": list(models), "latencies_seconds": latencies}

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = [executor.submit(worker, gpu, client) for gpu in args.gpus for client in range(clients_per_gpu)]
        barrier.wait(timeout=40)
        started = time.monotonic()
        try:
            while time.monotonic() - started < args.load_seconds and not stop.is_set():
                samples.append(telemetry(args.gpus))
                stop.wait(1)
        finally:
            stop.set()
        workers = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    summary = []
    for gpu in args.gpus:
        gpu_samples = [g for sample in samples for g in sample["gpus"] if g["gpu"] == gpu]
        latencies = [latency for worker_row in workers if worker_row["gpu"] == gpu for latency in worker_row["latencies_seconds"]]
        if not latencies:
            raise RuntimeError(f"No completed load requests on GPU {gpu}")
        summary.append({
            "gpu": gpu, "queries": len(latencies), "query_per_second": len(latencies) / elapsed,
            "latency_mean_seconds": statistics.mean(latencies),
            "latency_p95_seconds": float(np.percentile(latencies, 95)),
            "utilization_mean_percent": statistics.mean(g["utilization_percent"] for g in gpu_samples),
            "utilization_max_percent": max(g["utilization_percent"] for g in gpu_samples),
            "power_mean_w": statistics.mean(g["power_w"] for g in gpu_samples),
            "power_max_w": max(g["power_w"] for g in gpu_samples),
            "memory_max_mib": max(g["memory_mib"] for g in gpu_samples),
            "temperature_max_c": max(g["temperature_c"] for g in gpu_samples),
        })
    return {
        "clients_per_gpu": clients_per_gpu, "requested_seconds": args.load_seconds,
        "elapsed_including_drain_seconds": elapsed,
        "capture_top4_routes": args.capture_routing,
        "aggregate_query_per_second": sum(row["query_per_second"] for row in summary),
        "gpus": summary, "samples": samples, "workers": workers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=validate_gpus, default=ALLOWED_GPUS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--wait-seconds", type=float, default=600)
    parser.add_argument("--load-seconds", type=float, default=45)
    parser.add_argument("--clients-per-gpu", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--capture-routing", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "design/preload_probe.json")
    args = parser.parse_args()
    if args.load_seconds <= 0 or args.wait_seconds < 0 or any(n < 1 or n > 4 for n in args.clients_per_gpu):
        parser.error("positive load duration and 1..4 clients per GPU are required")
    matrix = tuple(int(value) for value in (HERE.parent / "online-servers/logs/matrix.gpus").read_text().split())
    if any(gpu not in matrix for gpu in args.gpus) or 6 in matrix:
        parser.error("Expected an existing matrix that excludes physical GPU 6")
    report = {
        "started_utc": now(), "gpus": list(args.gpus), "excluded_physical_gpu": 6,
        "workload": "real policy inference on synthetic observations; no simulator, no hidden capture, no intervention",
        "checkpoint_verification": "server constructor hashes entire checkpoint and strictly assigns all keys; client verifies metadata, stats hash and actual inference",
        "phases": [], "status": "waiting_for_preload",
    }
    write_json(args.out, report)
    wait_ports(args.gpus, args.wait_seconds)
    report["ready_utc"] = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(verify_gpu, args.host, gpu, matrix.index(gpu)) for gpu in args.gpus]
        report["endpoints"] = [row for future in futures for row in future.result()]
    report["status"] = "health_verified"
    write_json(args.out, report)
    for clients in args.clients_per_gpu:
        print(f"{now()} load phase: {clients} clients per GPU for {args.load_seconds}s", flush=True)
        phase = load_probe(args, clients)
        report["phases"].append(phase)
        write_json(args.out, report)
        print(json.dumps({"clients_per_gpu": clients, "aggregate_query_per_second": phase["aggregate_query_per_second"], "gpus": phase["gpus"]}), flush=True)
    report["status"] = "complete"
    report["completed_utc"] = now()
    write_json(args.out, report)


if __name__ == "__main__":
    main()
