#!/usr/bin/env python3
"""Compare one and two independent batch-1 model processes on an allowed GPU."""

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

import numpy as np
import probe_process_scaling as scaling

probe = scaling.probe
HERE = Path(__file__).resolve().parent
FIELDS = ("actions", "routing/expert_ids", "routing/expert_weights", "routing/layer_indices")


def assert_equal(expected, actual):
    for key in FIELDS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    if expected["flow/noise_sha256"] != actual["flow/noise_sha256"]:
        raise RuntimeError("Explicit flow noise identity changed")


def measure(gpu, port_base, seconds, modes, requests, references):
    barrier = threading.Barrier(len(modes) + 1)
    stop = threading.Event()
    samples = []

    def worker(isolated):
        durations, checked = [], 0
        try:
            with contextlib.ExitStack() as stack:
                connection, metadata = scaling.open_connection(stack, gpu, isolated, port_base)
                request = requests[int(isolated)]
                expected = references[int(isolated)]
                warm, _ = probe.infer(connection, request, "goal")
                assert_equal(expected, warm)
                barrier.wait(timeout=30)
                while not stop.is_set():
                    response, latency = probe.infer(connection, request, "goal")
                    assert_equal(expected, response)
                    durations.append(latency)
                    checked += 1
            return dict(isolated=isolated, server_pid=metadata["bundle_process_pid"],
                        latencies_seconds=durations, exact_checked_queries=checked)
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(modes)) as pool:
        futures = [pool.submit(worker, mode) for mode in modes]
        barrier.wait(timeout=30)
        started = time.monotonic()
        try:
            while time.monotonic() - started < seconds and not stop.is_set():
                samples.append(probe.telemetry((gpu,)))
                stop.wait(.5)
        finally:
            stop.set()
        workers = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    rows = [row for sample in samples for row in sample["gpus"]]
    queries = sum(worker["exact_checked_queries"] for worker in workers)
    return dict(process_count=len(modes), gpu=gpu, modes=modes, batch_size=1,
        queries=queries, seconds_including_drain=elapsed, query_per_second=queries / elapsed,
        utilization_mean_percent=statistics.mean(row["utilization_percent"] for row in rows),
        power_mean_w=statistics.mean(row["power_w"] for row in rows),
        total_memory_peak_mib=max(row["memory_mib"] for row in rows),
        workers=workers, gpu_samples=samples, every_response_exact=True)


def run(args):
    if args.gpu not in probe.ALLOWED_GPUS or args.seconds <= 0:
        raise ValueError("An allowed GPU and positive duration are required")
    free = int(subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip())
    if free < 26000:
        raise RuntimeError("Temporary Goal worker requires 26000 MiB spare memory")
    port = args.port_base + 10 * args.gpu
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="loading", started_utc=probe.now(), gpu=args.gpu, excluded_gpu=6,
        batch_size=1, hidden_capture=False, full_hb_capture=False, simulator=False,
        original_servers_replaced=False, alarm_parameters_modified=False, phases=[],
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), HERE / "serve_isolated_model.py", HERE / "probe_process_scaling.py",
                         HERE / "probe_preloaded_models.py")})
    process = None
    with (args.output / "temporary_worker.log").open("w") as log:
        try:
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            process = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_isolated_model.py"),
                "--gpu", str(args.gpu), "--base-port", str(port), "--models", "goal"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            report["temporary_pid"] = process.pid
            probe.write_json(args.output / "results.json", report)
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Temporary worker exited; see temporary_worker.log")
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError("Temporary worker was not ready in 600 seconds")
                time.sleep(1)
            report["equivalence_before"] = scaling.check_equivalence((args.gpu,), args.port_base)
            requests = [probe.observation("goal", 98000 + i, capture=True) for i in range(2)]
            with contextlib.ExitStack() as stack:
                connection, metadata = scaling.open_connection(stack, args.gpu, True, args.port_base)
                references = [probe.infer(connection, request, "goal")[0] for request in requests]
                report["temporary_metadata"] = metadata
            report["status"] = "measuring"
            # A/B/B/A reduces the influence of stage order on this short comparison.
            for modes in ((True,), (False, True), (False, True), (True,)):
                print(f"MEASURE GPU {args.gpu}: {len(modes)} processes, batch=1 each", flush=True)
                phase = measure(args.gpu, args.port_base, args.seconds, modes, requests, references)
                report["phases"].append(phase)
                probe.write_json(args.output / "results.json", report)
                print("RESULT " + json.dumps({k: v for k, v in phase.items()
                    if k not in ("workers", "gpu_samples")}), flush=True)
            report["equivalence_after"] = scaling.check_equivalence((args.gpu,), args.port_base)
            report["status"] = "complete"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            report["temporary_worker_stopped"] = process is None or process.poll() is not None
            report["finished_utc"] = probe.now()
            probe.write_json(args.output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=probe.ALLOWED_GPUS, default=0)
    parser.add_argument("--port-base", type=int, default=8900)
    parser.add_argument("--seconds", type=float, default=25)
    parser.add_argument("--output", type=Path, default=HERE / "design/overlap_efficiency_20260908")
    run(parser.parse_args())
