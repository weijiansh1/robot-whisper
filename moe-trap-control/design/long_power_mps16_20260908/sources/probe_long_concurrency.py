#!/usr/bin/env python3
"""Measure independent batch-1 processes, optionally with private CUDA MPS."""

import argparse
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
from run_collection_preflight import open_replica, replica_port, verify_server, scaling, probe
from collection_routes import CAPTURE_KEY, FIELDS
from collection_storage import atomic_json, digest, load_snapshot, records
from himoe_libero_bridge.preprocess import build_policy_observation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "himoe-route-capture"))


def saved_inputs(root):
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "completed" or summary["model"] != "long":
        raise ValueError("A completed Long collection run is required")
    inputs = []
    for task in summary["tasks"][:4]:
        directory = Path(task["output"])
        saved_rows = {int(row["query"]): row for row in records(directory / "main") if int(row["query"]) in (0, 5)}
        for q in (0, 5):
            saved = load_snapshot(directory / ("preflight_q%03d" % q))
            row = saved_rows[q]
            request = build_policy_observation(saved["observation"], saved["prompt"])
            request.update({"flow/noise": row["noise"], "routing/capture": True, CAPTURE_KEY: True})
            inputs.append((request, row))
    return inputs


def check_response(response, expected):
    for field in ("actions", *FIELDS):
        np.testing.assert_array_equal(response[field], expected[field], err_msg=field)
    import hashlib
    if response["flow/noise_sha256"] != hashlib.sha256(expected["noise"].tobytes()).hexdigest():
        raise ValueError("Saved flow noise mismatch")


def measure(args, count, inputs):
    stop = threading.Event()
    barrier = threading.Barrier(count + 1)
    samples = []

    def worker(replica):
        latencies = []
        try:
            with contextlib.ExitStack() as stack:
                connection, metadata = open_replica(stack, args.gpu, args.port_base, "long", replica)
                warm_input = inputs[replica % len(inputs)]
                warm, _ = probe.infer(connection, warm_input[0], "long")
                check_response(warm, warm_input[1])
                barrier.wait(timeout=60)
                while not stop.is_set():
                    request, expected = inputs[(len(latencies) + replica) % len(inputs)]
                    response, latency = probe.infer(connection, request, "long")
                    check_response(response, expected)
                    latencies.append(latency)
            return dict(replica=replica, pid=metadata["bundle_process_pid"], queries=len(latencies),
                        latencies_seconds=latencies)
        except BaseException:
            stop.set()
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(worker, replica) for replica in range(count)]
        try:
            barrier.wait(timeout=60)
            started = time.monotonic()
            while time.monotonic() - started < args.seconds and not stop.is_set():
                samples.append(probe.telemetry((args.gpu,)))
                stop.wait(.5)
        finally:
            stop.set()
            barrier.abort()
        workers = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    rows = [row for sample in samples for row in sample["gpus"]]
    queries = sum(worker["queries"] for worker in workers)
    return dict(active_processes=count, resident_processes=args.replicas, mps=args.mps, queries=queries,
        elapsed_including_drain_seconds=elapsed, queries_per_second=queries / elapsed,
        mean_power_w=statistics.mean(row["power_w"] for row in rows),
        max_power_w=max(row["power_w"] for row in rows),
        samples_at_least_300w=sum(row["power_w"] >= 300 for row in rows),
        mean_gpu_utilization_percent=statistics.mean(row["utilization_percent"] for row in rows),
        peak_memory_mib=max(row["memory_mib"] for row in rows),
        every_response_exact=True, workers=workers, samples=samples)


def run(args):
    if args.gpu not in probe.ALLOWED_GPUS or args.seconds <= 0:
        raise ValueError("An allowed physical GPU and positive duration are required")
    if any(count not in (1, 2, 4, 8, 16) or count > args.replicas for count in args.counts):
        raise ValueError("Active process counts must be supported and no larger than the resident count")
    free = int(subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip())
    if free < 39000:
        raise ValueError("Concurrency probe requires 39000 MiB free")
    for replica in range(args.replicas):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", replica_port(args.gpu, args.port_base, "long", replica)))
    inputs = saved_inputs(args.collection)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="loading", started_utc=probe.now(), gpu=args.gpu, excluded_gpu=6,
        model="long", batch_size=1, hidden_capture=False, full_hb_capture=True,
        mps=args.mps, resident_processes=args.replicas, phase_order=args.counts,
        input_source=str(args.collection.resolve()), independent_saved_observations=len(inputs),
        simulator=False, new_trajectories=0, alarm_parameters_modified=False,
        original_servers_replaced=False, phases=[], source_sha256={str(path): digest(path)
            for path in (Path(__file__), HERE / "serve_isolated_model.py", HERE / "run_collection_preflight.py",
                         HERE / "collection_mps.py", HERE / "collection_routes.py")})
    (args.output / "sources").mkdir()
    for source in report["source_sha256"]:
        shutil.copy2(source, args.output / "sources" / Path(source).name)
    process, mps = None, None

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Concurrency probe interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (args.output / "model.log").open("w") as log:
        try:
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            if args.mps:
                from collection_mps import PrivateMPS
                mps = PrivateMPS(args.gpu, args.output)
                report["mps_controller"] = mps.start()
                env.update({key: value for key, value in mps.env.items() if key.startswith("CUDA_")})
            process = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_isolated_model.py"),
                "--gpu", str(args.gpu), "--base-port", str(args.port_base + 10 * args.gpu),
                "--models", "long", "--replicas", str(args.replicas), "--full-hb-capture"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            report["temporary_owner_pid"] = process.pid
            atomic_json(args.output / "results.json", report)
            waiting = set(range(args.replicas))
            deadline = time.monotonic() + 600
            while waiting:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Shared model startup failed or timed out")
                for replica in list(waiting):
                    with socket.socket() as sock:
                        sock.settimeout(.2)
                        if sock.connect_ex(("127.0.0.1", replica_port(args.gpu, args.port_base, "long", replica))) == 0:
                            waiting.remove(replica)
                time.sleep(.5)
            report["equivalence_before"] = verify_server(args.gpu, args.port_base, "long", args.replicas)
            if mps is not None:
                report["mps_clients"] = mps.verify_clients(report["equivalence_before"]["replica_pids"])
            report["status"] = "measuring"
            for count in args.counts:
                phase = measure(args, count, inputs)
                report["phases"].append(phase)
                atomic_json(args.output / "results.json", report)
                print(json.dumps({key: value for key, value in phase.items() if key not in ("workers", "samples")}), flush=True)
            report["equivalence_after"] = verify_server(args.gpu, args.port_base, "long", args.replicas)
            report["status"] = "completed"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            report["temporary_owner_stopped"] = process is None or process.poll() is not None
            report["live_replica_pids_after_cleanup"] = [pid for pid in report.get("equivalence_before", {}).get("replica_pids", [])
                if Path("/proc/%d" % pid).exists()]
            if mps is not None:
                report["mps_cleanup"] = mps.close()
            report["finished_utc"] = probe.now()
            atomic_json(args.output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=probe.ALLOWED_GPUS, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--counts", type=int, nargs="+", default=(1, 2, 4, 4, 2, 1))
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--port-base", type=int, default=10100)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
