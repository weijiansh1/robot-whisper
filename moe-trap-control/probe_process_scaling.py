#!/usr/bin/env python3
"""Measure isolated GPU processes against the running matrix without replacing it."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time

import probe_preloaded_models as probe
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PYTHON = Path("/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python")


def open_connection(stack, gpu, isolated, port_base):
    port = (port_base + 10 * gpu) if isolated else probe.port_for(gpu, "goal")
    connection = stack.enter_context(probe.connect(f"ws://127.0.0.1:{port}", compression=None,
        max_size=None, open_timeout=10, close_timeout=5, ping_interval=None))
    frame = connection.recv(timeout=15)
    if isinstance(frame, str):
        raise RuntimeError(frame)
    metadata = probe.msgpack_numpy.unpackb(frame)
    logical = 0 if isolated else probe.ALLOWED_GPUS.index(gpu)
    probe._verify_bundle_identity(metadata, port, gpu, logical, "goal")
    if isolated and not metadata.get("isolated_process"):
        raise RuntimeError("Expected an isolated worker")
    return connection, metadata


def load_phase(gpus, isolated, port_base, seconds, clients):
    barrier = threading.Barrier(len(gpus) * clients + 1)
    stop = threading.Event()
    samples = []

    def worker(gpu, client):
        latencies = []
        try:
            with contextlib.ExitStack() as stack:
                connection, metadata = open_connection(stack, gpu, isolated, port_base)
                request = probe.observation("goal", 93000 + gpu * 10 + client, capture=True)
                barrier.wait(timeout=45)
                while not stop.is_set():
                    response, latency = probe.infer(connection, request, "goal")
                    if "routing/expert_ids" not in response:
                        raise RuntimeError("Routing capture was not returned")
                    latencies.append(latency)
            return dict(gpu=gpu, client=client, latencies=latencies,
                        server_pid=metadata["bundle_process_pid"])
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus) * clients) as pool:
        futures = [pool.submit(worker, gpu, client) for gpu in gpus for client in range(clients)]
        barrier.wait(timeout=45)
        started = time.monotonic()
        try:
            while time.monotonic() - started < seconds and not stop.is_set():
                samples.append(probe.telemetry(gpus))
                stop.wait(1)
        finally:
            stop.set()
        workers = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    summary = []
    for gpu in gpus:
        telemetry = [r for s in samples for r in s["gpus"] if r["gpu"] == gpu]
        latencies = [v for w in workers if w["gpu"] == gpu for v in w["latencies"]]
        summary.append(dict(gpu=gpu, queries=len(latencies), query_per_second=len(latencies) / elapsed,
            latency_mean_seconds=statistics.mean(latencies),
            latency_p95_seconds=float(np.percentile(latencies, 95)),
            utilization_mean_percent=statistics.mean(r["utilization_percent"] for r in telemetry),
            power_mean_w=statistics.mean(r["power_w"] for r in telemetry),
            memory_max_mib=max(r["memory_mib"] for r in telemetry)))
    return dict(mode="isolated_processes" if isolated else "shared_matrix", selected_gpus=gpus,
        clients_per_gpu=clients, model="goal", capture_top4_routes=True, requested_seconds=seconds,
        elapsed_including_drain_seconds=elapsed, gpus=summary, samples=samples, workers=workers,
        aggregate_query_per_second=sum(r["query_per_second"] for r in summary),
        mean_gpu_utilization_percent=statistics.mean(r["utilization_mean_percent"] for r in summary))


def check_equivalence(gpus, port_base):
    def check(gpu):
        records = []
        with contextlib.ExitStack() as stack:
            baseline, old_metadata = open_connection(stack, gpu, False, port_base)
            isolated, new_metadata = open_connection(stack, gpu, True, port_base)
            probe.check_checkpoint(new_metadata, "goal")
            for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                        "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
                if old_metadata[key] != new_metadata[key]:
                    raise RuntimeError("Policy identity changed: " + key)
            for seed in (94000 + gpu, 95000 + gpu):
                request = probe.observation("goal", seed, capture=True)
                expected, _ = probe.infer(baseline, request, "goal")
                actual, _ = probe.infer(isolated, request, "goal")
                fields = ("actions", "routing/expert_ids", "routing/expert_weights", "routing/layer_indices")
                errors = {}
                for key in fields:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"GPU {gpu} {key}")
                    errors[key] = float(np.max(np.abs(np.asarray(actual[key], np.float64) - expected[key])))
                if actual["flow/noise_sha256"] != expected["flow/noise_sha256"]:
                    raise RuntimeError("Flow noise acknowledgement changed")
                records.append(dict(gpu=gpu, seed=seed, max_abs_errors=errors,
                    original_pid=old_metadata["bundle_process_pid"], isolated_pid=new_metadata["bundle_process_pid"],
                    isolated_gpu_uuid=new_metadata["isolated_gpu_uuid"]))
        print(f"EQUIVALENCE GPU {gpu}: two fixed inputs, exact actions and routes", flush=True)
        return records
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(check, gpu) for gpu in gpus]
        return [record for future in futures for record in future.result()]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="starting", started_utc=probe.now(), allowed_gpus=args.gpus, excluded_gpu=6,
        original_servers_replaced=False, workload="same Goal checkpoint and batch=1 inference with top-4 capture",
        source_code_sha256={}, phases=[], hidden_capture=False, simulator=False, full_hb_capture=False)
    for path in (Path(__file__), HERE / "serve_isolated_model.py", HERE / "probe_preloaded_models.py"):
        report["source_code_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    processes, logfiles = {}, []
    try:
        for gpu in args.gpus:
            free = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
            if int(free) < 26000:
                raise RuntimeError(f"GPU {gpu} needs 26000 MiB free for an isolated Goal worker")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", args.port_base + 10 * gpu))
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            log = (args.output / f"gpu{gpu}.log").open("w")
            logfiles.append(log)
            command = [str(MODEL_PYTHON), "-u", str(HERE / "serve_isolated_model.py"),
                       "--gpu", str(gpu), "--base-port", str(args.port_base + 10 * gpu), "--models", "goal"]
            processes[gpu] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["temporary_pids"] = {gpu: p.pid for gpu, p in processes.items()}
        probe.write_json(args.output / "results.json", report)
        deadline = time.monotonic() + 600
        pending = set(args.gpus)
        while pending:
            for gpu in list(pending):
                if processes[gpu].poll() is not None:
                    raise RuntimeError(f"GPU {gpu} worker exited; see gpu{gpu}.log")
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", args.port_base + 10 * gpu)) == 0:
                        pending.remove(gpu)
                        print(f"READY isolated GPU {gpu}, PID {processes[gpu].pid}", flush=True)
            if time.monotonic() > deadline:
                raise TimeoutError(f"Isolated workers did not become ready: {sorted(pending)}")
            if pending:
                time.sleep(1)
        report["equivalence_before"] = check_equivalence(args.gpus, args.port_base)
        report["status"] = "measuring"
        probe.write_json(args.output / "results.json", report)
        configs = [(False, args.gpus), (True, (args.gpus[0],))]
        if len(args.gpus) > 2:
            configs.append((True, args.gpus[:2]))
        configs.append((True, args.gpus))
        for isolated, gpus in configs:
            print(f"MEASURE isolated={isolated}, GPUs={gpus}, seconds={args.seconds}", flush=True)
            phase = load_phase(gpus, isolated, args.port_base, args.seconds, args.clients)
            report["phases"].append(phase)
            probe.write_json(args.output / "results.json", report)
            print(json.dumps({k: phase[k] for k in ("mode", "selected_gpus", "aggregate_query_per_second",
                                                    "mean_gpu_utilization_percent")}), flush=True)
        report["equivalence_after"] = check_equivalence(args.gpus, args.port_base)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logfiles:
            log.close()
        report["temporary_workers_stopped"] = all(p.poll() is not None for p in processes.values())
        report["completed_utc"] = probe.now()
        probe.write_json(args.output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=probe.validate_gpus, default=probe.ALLOWED_GPUS)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=8900)
    parser.add_argument("--output", type=Path, default=HERE / "design/process_scaling_20260908")
    args = parser.parse_args()
    if args.seconds <= 0 or args.clients not in (1, 2, 4):
        parser.error("Positive duration and 1, 2 or 4 clients per GPU are required")
    run(args)
