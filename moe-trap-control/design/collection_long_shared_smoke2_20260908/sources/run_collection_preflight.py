#!/usr/bin/env python3
"""Use allowed GPUs for real, bounded Pro/Plus collection and C0 preflight."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time

import numpy as np
import probe_process_scaling as scaling
from benchmarks.run_benchmarks import BASE, ROOTS, environment, variants
from collection_routes import CAPTURE_KEY, FIELDS
from collection_storage import atomic_json, digest
from collection_worker_pool import PersistentWorker

probe = scaling.probe
HERE = Path(__file__).resolve().parent


SUITES = dict(goal="libero_goal", spatial="libero_spatial", object="libero_object", long="libero_10")


def select_jobs(per_category, model="goal"):
    groups = {}
    for benchmark in ("pro", "plus"):
        for row in variants(benchmark):
            if row["suite"] == SUITES[model]:
                groups.setdefault((benchmark, row["category"]), []).append(row)
    groups = {key: sorted(rows, key=lambda row: hashlib.sha256(row["variant_id"].encode()).hexdigest())
              for key, rows in groups.items()}
    return [groups[key][index] for index in range(per_category) for key in sorted(groups)
            if index < len(groups[key])]


def storage_used(directory):
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A writer may atomically rename its temporary file during the scan.
    return total


def replica_port(gpu, port_base, model, replica=0):
    return port_base + gpu * 10 + probe.MODELS.index(model) + replica * 5


def open_replica(stack, gpu, port_base, model, replica=0):
    port = replica_port(gpu, port_base, model, replica)
    connection = stack.enter_context(probe.connect("ws://127.0.0.1:%d" % port, compression=None,
        max_size=None, open_timeout=10, close_timeout=5, ping_interval=None))
    frame = connection.recv(timeout=15)
    if isinstance(frame, str):
        raise RuntimeError(frame)
    metadata = probe.msgpack_numpy.unpackb(frame)
    probe._verify_bundle_identity(metadata, port, gpu, 0, model)
    if not metadata.get("isolated_process") or metadata.get("replica_index") != replica:
        raise ValueError("Isolated replica identity mismatch")
    return connection, metadata


def verify_server(gpu, port_base, model="goal", replicas=1):
    with contextlib.ExitStack() as stack:
        old, original = probe.open_endpoint(stack, "127.0.0.1", gpu, model)
        connections = [open_replica(stack, gpu, port_base, model, replica) for replica in range(replicas)]
        for _, current in connections:
            probe.check_checkpoint(current, model)
            if not current.get("collection_full_hb_supported") or current.get("collection_hidden_capture"):
                raise ValueError("Collection capture metadata mismatch")
            for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                        "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
                if original[key] != current[key]:
                    raise ValueError("Original policy identity differs: " + key)
        for seed in (102000 + gpu, 103000 + gpu):
            request = probe.observation(model, seed, capture=True)
            expected, _ = probe.infer(old, request, model)
            request[CAPTURE_KEY] = True
            with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as pool:
                outputs = list(pool.map(lambda pair: probe.infer(pair[0], request, model)[0], connections))
            for actual in outputs:
                for key in ("actions", "routing/expert_ids", "routing/expert_weights", "routing/layer_indices"):
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
                if actual["flow/noise_sha256"] != expected["flow/noise_sha256"]:
                    raise ValueError("Flow noise identity changed")
                for field in FIELDS:
                    np.testing.assert_array_equal(actual[field], outputs[0][field], err_msg=field)
        return dict(gpu=gpu, original_pid=original["bundle_process_pid"],
                    replica_pids=[metadata["bundle_process_pid"] for _, metadata in connections],
                    actions_and_top4_exact=True, concurrent_full_hb_exact=True, checked_seeds=2,
                    full_hb_shapes={field: list(actual[field].shape) for field in FIELDS})


def run(args):
    args.output = args.output.resolve()
    if args.workers_per_gpu not in (1, 2, 4) or not 1 <= args.per_category <= 10:
        raise ValueError("Use 1/2/4 environment workers and 1..10 cases per category")
    if min(args.job_timeout, args.storage_quota_gib, args.disk_floor_gib) <= 0:
        raise ValueError("Timeout and storage bounds must be positive")
    if args.workers_per_gpu < args.replicas:
        raise ValueError("Each inference replica needs at least one environment worker")
    jobs = select_jobs(args.per_category, args.model)
    if args.variants is not None:
        inventory = {row["variant_id"]: row for benchmark in ("pro", "plus") for row in variants(benchmark)}
        if len(set(args.variants)) != len(args.variants):
            raise ValueError("Explicit variants must be unique")
        jobs = [inventory[variant_id] for variant_id in args.variants]
        if any(row["suite"] != SUITES[args.model] for row in jobs):
            raise ValueError("Explicit variants must match the selected model")
    if args.retry_from is not None:
        previous = json.loads((args.retry_from / "summary.json").read_text())
        inventory = {row["variant_id"]: row for benchmark in ("pro", "plus") for row in variants(benchmark)}
        retry_ids = [task["variant"]["variant_id"] for task in previous["tasks"]
                     if task["status"] != "completed" or (task.get("c0") or {}).get("status") != "passed"]
        if len(set(retry_ids)) != len(retry_ids):
            raise ValueError("Retry manifest has duplicate variants")
        jobs = [inventory[variant_id] for variant_id in retry_ids]
        if not jobs or any(row["suite"] != SUITES[args.model] for row in jobs):
            raise ValueError("Retry requires unfinished tasks for the selected model")
    if len(jobs) < len(args.gpus):
        raise ValueError("There must be at least one real task per GPU")
    render_gpus = args.render_gpus or tuple(gpu for gpu in args.gpus if gpu not in (1, 2))
    if not render_gpus:
        raise ValueError("EGL replay failed on GPUs 1/2; select --render-gpus from 0,3,4,5,7")
    render_map = {gpu: render_gpus[index % len(render_gpus)] for index, gpu in enumerate(args.gpus)}
    if shutil.disk_usage(HERE).free < args.disk_floor_gib * 1024**3:
        raise RuntimeError("Disk free space is below the configured floor")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    sources = [HERE / name for name in ("run_collection_preflight.py", "collect_preflight_worker.py",
               "collection_routes.py", "collection_storage.py", "serve_isolated_model.py", "probe_collection_render.py",
               "collection_noise.py", "collection_worker_pool.py")]
    (args.output / "sources").mkdir()
    for path in sources:
        shutil.copy2(path, args.output / "sources" / path.name)
    report = dict(status="loading", started_utc=probe.now(), gpus=args.gpus, excluded_gpu=6,
        workers_per_gpu=args.workers_per_gpu, render_gpu_map=render_map, exact_noise_fastpath=args.exact_noise_fastpath,
        batch_size=1, model=args.model, hidden_capture=False, replicas_per_gpu=args.replicas,
        persistent_environments=True,
        intervention=False, formal_manifest_modified=False, original_servers_replaced=False,
        storage_quota_gib=args.storage_quota_gib, disk_floor_gib=args.disk_floor_gib,
        source_sha256={str(path): digest(path) for path in sources},
        frozen_alarm_sha256=digest(HERE / "design/frozen_alarm_comparison_20260908/profiles/parameters.json"),
        tasks=[dict(variant=row, status="queued", output=str(args.output / "tasks" / row["variant_id"])) for row in jobs])
    if args.retry_from is not None:
        report["retry_from"] = str(args.retry_from.resolve())
        report["retry_manifest_sha256"] = digest(args.retry_from / "summary.json")
    atomic_json(args.output / "summary.json", report)
    servers, logs, active_envs = {}, [], {}
    stop = threading.Event()
    lock = threading.Lock()
    pool = None
    samples = []
    pending = queue.Queue()
    for index in range(len(jobs)):
        pending.put(index)

    def stop_signal(_signal, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def env_worker(gpu, worker_index):
        workers = {}
        replica = worker_index % args.replicas
        try:
            while not stop.is_set():
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                row = jobs[index]
                record = report["tasks"][index]
                seed = 20260908 + int(row["variant_id"][:6], 16)
                benchmark = row["benchmark"]
                if benchmark not in workers:
                    command = [str(BASE / "envs/libero/bin/python"), "-u", str(HERE / "collect_preflight_worker.py"),
                        "--benchmark", benchmark, "--worker-loop", "--model", args.model, "--gpu", str(gpu),
                        "--render-gpu", str(render_map[gpu]),
                        "--port", str(replica_port(gpu, args.port_base, args.model, replica))]
                    if args.exact_noise_fastpath:
                        command.append("--exact-noise-fastpath")
                    log_path = args.output / "logs" / ("env_gpu%d_slot%d_%s.log" % (gpu, worker_index, benchmark))
                    worker = PersistentWorker(command, environment(benchmark, render_map[gpu], "egl"),
                        str(ROOTS[benchmark]), log_path.open("w"))
                    workers[benchmark] = worker
                    with lock:
                        active_envs[worker.process.pid] = worker.process
                worker = workers[benchmark]
                with lock:
                    record.update(status="running", gpu=gpu, render_gpu=render_map[gpu], worker=worker_index,
                        replica=replica, pid=worker.process.pid, seed=seed, worker_log=str(worker.log.name))
                started = time.monotonic()
                try:
                    outcome = worker.execute(dict(variant=row["variant_id"], seed=seed, output=record["output"]),
                        args.job_timeout, stop)
                except Exception as error:
                    outcome = dict(exit_code=-1, error=repr(error))
                    worker.close()
                    workers.pop(benchmark)
                    with lock:
                        active_envs.pop(worker.process.pid, None)
                path = Path(record["output"]) / "result.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                with lock:
                    completed = outcome["exit_code"] == 0 and result.get("status") == "completed"
                    status = "completed" if completed else "failed"
                    if completed and (result.get("c0") or {}).get("status") != "passed":
                        status = "fidelity_failed"
                    record.update(status=status, exit_code=outcome["exit_code"],
                                  error=outcome.get("error", result.get("error")),
                                  worker_job_index=outcome.get("worker_job_index"),
                                  elapsed_seconds=time.monotonic() - started,
                                  queries=result.get("queries", 0), action_steps=result.get("action_steps", 0),
                                  success=result.get("success", False), first_alarm_query=result.get("first_alarm_query"),
                                  c0=result.get("c0"), main_complete=result.get("main_complete", False))
                print("TASK " + json.dumps({key: record.get(key) for key in
                    ("gpu", "status", "queries", "action_steps", "success", "first_alarm_query")}), flush=True)
                pending.task_done()
        finally:
            for worker in workers.values():
                worker.close()
                with lock:
                    active_envs.pop(worker.process.pid, None)

    try:
        for gpu in args.gpus:
            free = int(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip())
            required_free = 39000 if args.replicas > 1 else 30000
            if free < required_free:
                raise RuntimeError("GPU %d needs %d MiB free including rendering headroom" % (gpu, required_free))
            for replica in range(args.replicas):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", replica_port(gpu, args.port_base, args.model, replica)))
            log = (args.output / "logs" / ("model_gpu%d.log" % gpu)).open("w")
            logs.append(log)
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            servers[gpu] = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_isolated_model.py"),
                "--gpu", str(gpu), "--base-port", str(args.port_base + gpu * 10), "--models", args.model,
                "--replicas", str(args.replicas), "--full-hb-capture"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["temporary_model_pids"] = {gpu: process.pid for gpu, process in servers.items()}
        atomic_json(args.output / "summary.json", report)
        waiting = {(gpu, replica) for gpu in args.gpus for replica in range(args.replicas)}
        deadline = time.monotonic() + 600
        while waiting:
            if stop.is_set() or time.monotonic() > deadline:
                raise RuntimeError("Model startup interrupted or timed out")
            for gpu, replica in list(waiting):
                if servers[gpu].poll() is not None:
                    raise RuntimeError("Model worker exited on GPU %d" % gpu)
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", replica_port(gpu, args.port_base, args.model, replica))) == 0:
                        waiting.remove((gpu, replica))
            stop.wait(.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as checks:
            report["model_equivalence"] = list(checks.map(
                lambda gpu: verify_server(gpu, args.port_base, args.model, args.replicas), args.gpus))
        report["status"] = "collecting"
        report["collection_started_utc"] = probe.now()
        collection_started = time.monotonic()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu)
        futures = [pool.submit(env_worker, gpu, worker) for worker in range(args.workers_per_gpu) for gpu in args.gpus]
        iteration = 0
        while not all(future.done() for future in futures):
            for future in futures:
                if future.done():
                    future.result()
            if any(process.poll() is not None for process in servers.values()):
                raise RuntimeError("A model service exited during collection")
            used = storage_used(args.output)
            if shutil.disk_usage(args.output).free < args.disk_floor_gib * 1024**3 or used > args.storage_quota_gib * 1024**3:
                report["stop_reason"] = "storage_limit"
                stop.set()
            sample = probe.telemetry(args.gpus)
            with lock:
                sample["active_tasks_by_gpu"] = {gpu: sum(task["status"] == "running" and task["gpu"] == gpu
                    for task in report["tasks"]) for gpu in args.gpus}
                sample["queued_tasks"] = sum(task["status"] == "queued" for task in report["tasks"])
                samples.append(sample)
                for task in report["tasks"]:
                    path = Path(task["output"]) / "result.json"
                    if task["status"] == "running" and path.exists():
                        progress = json.loads(path.read_text())
                        task.update(queries=progress["queries"], action_steps=progress["action_steps"],
                                    main_complete=progress["main_complete"])
                report["storage_bytes"] = used
                report["completed_tasks"] = sum(task["status"] == "completed" for task in report["tasks"])
                report["latest_gpu_sample"] = sample
                atomic_json(args.output / "summary.json", report)
            if iteration % 5 == 0:
                print("PROGRESS " + json.dumps(dict(completed=report["completed_tasks"], total=len(jobs),
                    main_queries=sum(task.get("queries", 0) for task in report["tasks"]),
                    gpu_util={row["gpu"]: row["utilization_percent"] for row in sample["gpus"]})), flush=True)
            iteration += 1
            stop.wait(2)
        for future in futures:
            future.result()
        report["collection_elapsed_seconds"] = time.monotonic() - collection_started
        report["main_queries"] = sum(task.get("queries", 0) for task in report["tasks"])
        report["c0_queries"] = sum((task.get("c0") or {}).get("compared_queries", 0) for task in report["tasks"])
        report["queries_per_second"] = (report["main_queries"] + report["c0_queries"]) / report["collection_elapsed_seconds"]
        report["c0_passed"] = sum((task.get("c0") or {}).get("status") == "passed" for task in report["tasks"])
        report["c0_fidelity_failed"] = sum((task.get("c0") or {}).get("status") == "fidelity_failed" for task in report["tasks"])
        report["gpu_summary"] = [dict(gpu=gpu,
            mean_utilization_percent=float(np.mean([r["utilization_percent"] for s in samples for r in s["gpus"] if r["gpu"] == gpu])),
            max_memory_mib=max(r["memory_mib"] for s in samples for r in s["gpus"] if r["gpu"] == gpu)) for gpu in args.gpus]
        busy_samples = [sample for sample in samples if all(
            sample["active_tasks_by_gpu"][gpu] >= args.replicas for gpu in args.gpus)]
        report["all_replicas_supplied_samples"] = len(busy_samples)
        report["all_replicas_supplied_mean_utilization_percent"] = float(np.mean([
            row["utilization_percent"] for sample in busy_samples for row in sample["gpus"]])) if busy_samples else None
        report["reused_environment_tasks"] = sum((task.get("worker_job_index") or 0) > 1 for task in report["tasks"])
        report["status"] = "completed" if all(task["status"] == "completed" for task in report["tasks"]) else "incomplete"
        atomic_json(args.output / "gpu_samples.json", samples)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        stop.set()
        with lock:
            processes = list(active_envs.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        if pool is not None:
            pool.shutdown(wait=True)
        for process in servers.values():
            if process.poll() is None:
                process.terminate()
        for process in servers.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()
        report["temporary_models_stopped"] = all(process.poll() is not None for process in servers.values())
        report["environment_workers_stopped"] = all(process.poll() is not None for process in processes)
        report["finished_utc"] = probe.now()
        atomic_json(args.output / "summary.json", report)
    print("SUMMARY " + json.dumps({key: report.get(key) for key in
        ("status", "main_queries", "c0_queries", "queries_per_second", "c0_passed", "c0_fidelity_failed")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=probe.validate_gpus, default=probe.ALLOWED_GPUS)
    parser.add_argument("--render-gpus", type=probe.validate_gpus)
    parser.add_argument("--model", choices=tuple(SUITES), default="goal")
    parser.add_argument("--replicas", type=int, choices=(1, 2), default=1)
    parser.add_argument("--exact-noise-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=2)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--retry-from", type=Path)
    selection.add_argument("--variants", nargs="+")
    parser.add_argument("--port-base", type=int, default=8900)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--storage-quota-gib", type=float, default=4)
    parser.add_argument("--disk-floor-gib", type=float, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "design/collection_preflight_20260908")
    run(parser.parse_args())
