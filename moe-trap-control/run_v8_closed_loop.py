#!/usr/bin/env python3
"""Bounded v8 experiment scheduling on private batch-one model replicas, GPUs 0-3."""

import argparse
import concurrent.futures
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

from collection_storage import atomic_json, digest
from v8_closed_loop import PROTOCOL, ARMS, SETTINGS
from run_collection_preflight import (HERE, BASE, ROOTS, environment, scaling, probe, PrivateMPS,
    PersistentWorker, replica_port, verify_server, storage_used)


def run(args):
    plan = json.loads(args.plan.read_text())
    if (plan["protocol"] != PROTOCOL or plan["settings"] != SETTINGS or plan["arms"] != list(ARMS)
            or plan["allowed_gpus"] != [0, 1, 2, 3]):
        raise ValueError("Closed-loop plan differs from runtime")
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: "+path)
    output = args.output.resolve()
    if shutil.disk_usage(output.parent).free-plan["maximum_output_bytes"] < plan["disk_floor_gib"]*1024**3:
        raise ValueError("Insufficient output disk headroom")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "sources").mkdir()
    for name in plan["runtime_sources"]:
        shutil.copy2(HERE / name, output / "sources" / name)
    shutil.copy2(args.plan, output / "plan.json")
    tasks = {t["main_id"]: t for t in plan["tasks"]}
    jobs, children, ready = [], {}, queue.PriorityQueue()
    for task in plan["tasks"]:
        index = len(jobs)
        jobs.append(dict(kind="fixed_replay", main_id=task["main_id"], status="queued",
                         output=str(output / "replays" / task["main_id"])))
        ready.put((-task["parent_queries"], index))
    for task in plan["tasks"]:
        children[task["main_id"]] = []
        for event in task["events"]:
            for arm in ARMS:
                children[task["main_id"]].append(len(jobs))
                jobs.append(dict(kind="fixed_branches", main_id=task["main_id"], event=event, arm=arm,
                    status="waiting_for_c0", output=str(output / "events" / event["event_id"] / arm)))
    report = dict(status="starting", protocol=PROTOCOL, stage=plan["stage"], started_utc=probe.now(),
        plan_sha256=digest(args.plan), sources={name: digest(output / "sources" / name) for name in plan["runtime_sources"]}, tasks=jobs)
    stop, lock = threading.Event(), threading.Lock()
    servers, controllers, active, logs, samples = {}, {}, {}, [], []
    pool = None
    replicas, stride = 8, 40
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    def work(gpu, slot):
        workers = {}
        try:
            while not stop.is_set():
                try:
                    _, index = ready.get(timeout=.5)
                except queue.Empty:
                    with lock:
                        if all(j["status"] == "completed" for j in jobs):
                            return
                    continue
                job, task = jobs[index], tasks[jobs[index]["main_id"]]
                benchmark, render = task["benchmark"], 0 if gpu < 2 else 3
                if benchmark not in workers:
                    log = (output / "logs" / ("env_%d_%d_%s.log" % (gpu, slot, benchmark))).open("w")
                    command = [str(BASE / "envs/libero/bin/python"), "-u", str(HERE / "collect_v8_closed_loop.py"),
                        "--benchmark", benchmark, "--gpu", str(gpu), "--render-gpu", str(render),
                        "--port", str(replica_port(gpu, args.port_base, "long", slot, stride)), "--exact-noise-fastpath"]
                    worker = PersistentWorker(command, environment(benchmark, render, "egl"), str(ROOTS[benchmark]), log)
                    workers[benchmark] = worker
                    with lock:
                        active[worker.process.pid] = worker.process
                worker = workers[benchmark]
                with lock:
                    job.update(status="running", gpu=gpu, replica=slot, pid=worker.process.pid)
                control = dict(kind=job["kind"], parent=task, contract=SETTINGS)
                if job["kind"] == "fixed_branches":
                    control.update(event=job["event"], arm=job["arm"], replay_directory=str(output / "replays" / task["main_id"]))
                response = worker.execute(dict(variant=task["variant_id"], seed=task["noise_seed"], output=job["output"],
                    main_id=task["main_id"], init_index=task["init_index"], control=control), args.job_timeout, stop)
                final = json.loads((Path(job["output"]) / "result.json").read_text())
                passed = response["exit_code"] == 0 and final["status"] == "completed" and not final["invalid_pair"]
                if job["kind"] == "fixed_replay":
                    passed = passed and final.get("online_triggers_exact", False)
                elif job["arm"] == "native":
                    passed = passed and final["all_native_suffixes_exact"]
                with lock:
                    job.update(status="completed" if passed else "failed", error=final.get("error"),
                        actual_model_queries=final["actual_model_queries"], branches=final["branches"])
                    if passed and job["kind"] == "fixed_replay":
                        for child in children[task["main_id"]]:
                            jobs[child]["status"] = "queued"
                            weight = 1 if jobs[child]["arm"] == "native" else 4
                            ready.put((-weight*(52-jobs[child]["event"]["start_query"]), child))
                print("JOB "+json.dumps({k: job.get(k) for k in ("kind", "main_id", "arm", "status", "gpu")}), flush=True)
                ready.task_done()
                if not passed:
                    stop.set()
                    raise RuntimeError(final.get("error", "Closed-loop task failed"))
        finally:
            for worker in workers.values():
                worker.close()
                with lock:
                    active.pop(worker.process.pid, None)

    try:
        for gpu in (0, 1, 2, 3):
            free = int(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True).stdout.strip())
            if free < 39000:
                raise ValueError("Insufficient model headroom on GPU %d" % gpu)
            for replica in range(replicas):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride)))
            controllers[gpu] = PrivateMPS(gpu, output / ("gpu%d" % gpu))
            report.setdefault("mps_start", {})[gpu] = controllers[gpu].start()
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            env.update({k: v for k, v in controllers[gpu].env.items() if k.startswith("CUDA_")})
            log = (output / "logs" / ("model_gpu%d.log" % gpu)).open("w")
            logs.append(log)
            servers[gpu] = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_v8_feature_model.py"),
                "--gpu", str(gpu), "--base-port", str(args.port_base+gpu*stride), "--replicas", str(replicas)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["temporary_model_pids"] = {g: p.pid for g, p in servers.items()}
        atomic_json(output / "summary.json", report)
        waiting = {(g, r) for g in (0, 1, 2, 3) for r in range(replicas)}
        deadline = time.monotonic()+600
        while waiting:
            if stop.is_set() or time.monotonic() > deadline or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Model startup failed")
            for gpu, replica in list(waiting):
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride))) == 0:
                        waiting.remove((gpu, replica))
            stop.wait(.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as checks:
            report["model_equivalence"] = list(checks.map(lambda g: verify_server(g, args.port_base, "long", replicas, stride), (0, 1, 2, 3)))
        report["temporary_replica_pids"] = [p for row in report["model_equivalence"] for p in row["replica_pids"]]
        report["mps_clients"] = {row["gpu"]: controllers[row["gpu"]].verify_clients(row["replica_pids"]) for row in report["model_equivalence"]}
        report.update(status="collecting", collection_started_utc=probe.now())
        started = time.monotonic()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        futures = [pool.submit(work, gpu, slot) for slot in range(8) for gpu in (0, 1, 2, 3)]
        tick = 0
        while not all(f.done() for f in futures):
            for future in futures:
                if future.done():
                    future.result()
            if stop.is_set() or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Collection interrupted")
            used = storage_used(output)
            if used > plan["storage_quota_gib"]*1024**3 or shutil.disk_usage(output).free < plan["disk_floor_gib"]*1024**3:
                raise RuntimeError("Storage bound reached")
            sample = probe.telemetry([0, 1, 2, 3])
            with lock:
                sample["active_tasks_by_gpu"] = {g: sum(j["status"] == "running" and j["gpu"] == g for j in jobs) for g in (0, 1, 2, 3)}
                samples.append(sample)
                report.update(storage_bytes=used, latest_gpu_sample=sample, completed_jobs=sum(j["status"] == "completed" for j in jobs))
                atomic_json(output / "summary.json", report)
            if tick % 10 == 0:
                print("PROGRESS "+json.dumps(dict(completed=report["completed_jobs"], total=len(jobs),
                    util={r["gpu"]: r["utilization_percent"] for r in sample["gpus"]})), flush=True)
            tick += 1
            stop.wait(3)
        for future in futures:
            future.result()
        report.update(status="completed", completed_jobs=len(jobs), collection_elapsed_seconds=time.monotonic()-started,
                      actual_model_queries=sum(j["actual_model_queries"] for j in jobs))
        report["queries_per_second"] = report["actual_model_queries"]/report["collection_elapsed_seconds"]
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        with lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        if pool:
            pool.shutdown(wait=True)
        for process in servers.values():
            if process.poll() is None:
                process.terminate()
        for process in servers.values():
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()
        report["mps_cleanup"] = {}
        for gpu, controller in controllers.items():
            try:
                report["mps_cleanup"][gpu] = controller.close()
            except Exception as error:
                report["mps_cleanup"][gpu] = dict(error=repr(error))
                report["status"] = "cleanup_failed"
        report["temporary_models_stopped"] = all(p.poll() is not None for p in servers.values())
        report["live_replica_pids_after_cleanup"] = [p for p in report.get("temporary_replica_pids", []) if Path("/proc/%d" % p).exists()]
        report["environment_workers_stopped"] = all(p.poll() is not None for p in processes)
        report.update(storage_bytes=storage_used(output), finished_utc=probe.now())
        atomic_json(output / "gpu_samples.json", samples)
        atomic_json(output / "summary.json", report)
    print("SUMMARY "+json.dumps({k: report.get(k) for k in ("status", "actual_model_queries", "queries_per_second")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=12600)
    parser.add_argument("--job-timeout", type=float, default=3600)
    run(parser.parse_args())
