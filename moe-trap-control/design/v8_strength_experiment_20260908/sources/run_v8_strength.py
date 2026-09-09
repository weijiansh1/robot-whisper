#!/usr/bin/env python3
"""Schedule state/repeat jobs over isolated batch-1 replicas with bounded storage."""

import argparse
import concurrent.futures
import contextlib
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

from run_collection_preflight import (HERE, BASE, ROOTS, environment, scaling, probe, PrivateMPS,
    PersistentWorker, replica_port, open_replica, verify_server, storage_used)
from collection_storage import atomic_json, digest
from collection_routes import CAPTURE_KEY, PROBS_KEY, ALL_FIELDS, EFFECTIVE_WEIGHTS_KEY
from v8_strength_control import PROTOCOL, ARMS, BIAS_KEY, strength_bias


def preflight(gpu, port_base, replicas, stride):
    result = verify_server(gpu, port_base, "long", replicas, stride)
    with contextlib.ExitStack() as stack:
        clients = [open_replica(stack, gpu, port_base, "long", r, stride) for r in range(replicas)]
        if any(metadata.get("v8_strength_protocol") != PROTOCOL for _, metadata in clients):
            raise ValueError("Strength policy factory missing from a spawned replica")
        connection = clients[0][0]
        request = probe.observation("long", 118000 + gpu, capture=True)
        request[CAPTURE_KEY] = True
        native, _ = probe.infer(connection, request, "long")
        request[BIAS_KEY] = np.zeros((8, 10, 11, 32), np.float32)
        zero, _ = probe.infer(connection, request, "long")
        for key in ("actions", *ALL_FIELDS):
            np.testing.assert_array_equal(zero[key], native[key], err_msg="Zero strength fidelity")
        request[BIAS_KEY] = strength_bias(native[PROBS_KEY], native[PROBS_KEY], ARMS["combined16_5"], 6)
        strong, _ = probe.infer(connection, request, "long")
        request.pop(BIAS_KEY)
        released, _ = probe.infer(connection, request, "long")
        for key in ("actions", *ALL_FIELDS):
            np.testing.assert_array_equal(released[key], native[key], err_msg="Strength hook cleanup")
        result.update(zero_strength_exact=True, release_exact=True, strong_dtypes=strong["v8_strength/dtypes"],
            action_rms=float(np.sqrt(np.square(strong["actions"] - native["actions"]).mean())),
            mean_max_combine_weight=float(strong[EFFECTIVE_WEIGHTS_KEY][4:, :, 1:].max(-1).mean()))
    return result


def run(args):
    plan = json.loads(args.plan.read_text())
    if plan["protocol"] != PROTOCOL or plan["arms_registry"] != ARMS or plan["allowed_gpus"] != [0, 1, 2, 3]:
        raise ValueError("Strength plan mismatch")
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    if shutil.disk_usage(HERE).free - plan["maximum_output_bytes"] < plan["disk_floor_gib"] * 1024**3:
        raise ValueError("Insufficient disk headroom")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "sources").mkdir()
    sources = ("v8_strength_control.py", "collect_v8_strength.py", "serve_v8_strength_model.py", "run_v8_strength.py",
        "v8_feature_control.py", "serve_v8_feature_model.py", "collect_v8_feature_control.py", "collect_adaptive_control.py",
        "collection_routes.py", "collection_storage.py", "collection_protocol.py", "collect_preflight_worker.py",
        "collection_noise.py", "collection_mps.py", "collection_worker_pool.py", "adaptive_control.py",
        "serve_isolated_model.py", "run_collection_preflight.py")
    for name in sources:
        shutil.copy2(HERE / name, output / "sources" / name)
    shutil.copy2(args.plan, output / "plan.json")
    report = dict(status="starting", protocol=PROTOCOL, started_utc=probe.now(), plan_sha256=digest(args.plan),
        sources={name:digest(output / "sources" / name) for name in sources},
        tasks=[dict(job_id=j["job_id"], main_id=j["main_id"], replicate=j["replicate"], status="queued",
                    output=str(output / "jobs" / j["job_id"])) for j in plan["jobs"]])
    tasks = {t["main_id"]:t for t in plan["tasks"]}
    pending = queue.Queue()
    for index in range(len(plan["jobs"])):
        pending.put(index)
    replicas, stride = plan["replicas_per_gpu"], max(10, 5 * plan["replicas_per_gpu"])
    stop, lock = threading.Event(), threading.Lock()
    servers, controllers, active, logs, samples = {}, {}, {}, [], []
    pool = None
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_:stop.set())

    def work(gpu, slot):
        workers = {}
        try:
            while not stop.is_set():
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                job, record = plan["jobs"][index], report["tasks"][index]
                task = tasks[job["main_id"]]
                benchmark, render = task["benchmark"], 0 if gpu < 2 else 3
                if benchmark not in workers:
                    log = (output / "logs" / ("env_%d_%d_%s.log" % (gpu, slot, benchmark))).open("w")
                    command = [str(BASE / "envs/libero/bin/python"), "-u", str(HERE / "collect_v8_strength.py"),
                        "--benchmark", benchmark, "--gpu", str(gpu), "--render-gpu", str(render),
                        "--port", str(replica_port(gpu, args.port_base, "long", slot, stride)), "--exact-noise-fastpath"]
                    worker = PersistentWorker(command, environment(benchmark, render, "egl"), str(ROOTS[benchmark]), log)
                    workers[benchmark] = worker
                    with lock:
                        active[worker.process.pid] = worker.process
                worker = workers[benchmark]
                with lock:
                    record.update(status="running", gpu=gpu, replica=slot, pid=worker.process.pid)
                contract = {k:plan[k] for k in ("timing", "random_stream", "references", "control", "logit_evidence",
                                                "horizon", "chunk", "hidden_capture", "batch_size", "threshold_fitting")}
                message = dict(variant=task["variant_id"], seed=task["noise_seed"], output=record["output"],
                    main_id=task["main_id"], init_index=task["init_index"], control=dict(kind="strength_branches",
                    parent=task, replicate=job["replicate"], arms=plan["arms"], contract=contract))
                result = worker.execute(message, args.job_timeout, stop)
                final = json.loads((Path(record["output"]) / "result.json").read_text())
                passed = (result["exit_code"] == 0 and final["status"] == "completed" and
                          final["weak_full_suffix_exact"] and final["baseline_first_query_exact"])
                with lock:
                    record.update(status="completed" if passed else "failed", error=final.get("error"),
                        branches=final["branches"], actual_model_queries=final["actual_model_queries"])
                print("JOB " + json.dumps({k:record[k] for k in ("main_id", "replicate", "status", "gpu")}), flush=True)
                if not passed:
                    stop.set()
                    raise RuntimeError(final.get("error", "Strength job failed"))
                pending.task_done()
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
                raise ValueError("GPU %d lacks memory headroom" % gpu)
            for replica in range(replicas):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride)))
            controller = PrivateMPS(gpu, output / ("gpu%d" % gpu))
            controllers[gpu] = controller
            report.setdefault("mps_start", {})[gpu] = controller.start()
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            env.update({k:v for k,v in controller.env.items() if k.startswith("CUDA_")})
            log = (output / "logs" / ("model_gpu%d.log" % gpu)).open("w")
            logs.append(log)
            servers[gpu] = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_v8_strength_model.py"),
                "--gpu", str(gpu), "--base-port", str(args.port_base + gpu * stride), "--replicas", str(replicas)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["temporary_model_pids"] = {g:p.pid for g,p in servers.items()}
        atomic_json(output / "summary.json", report)
        waiting = {(g,r) for g in (0,1,2,3) for r in range(replicas)}
        deadline = time.monotonic() + 600
        while waiting:
            if stop.is_set() or time.monotonic() > deadline or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Strength model startup failed")
            for gpu, replica in list(waiting):
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride))) == 0:
                        waiting.remove((gpu, replica))
            stop.wait(.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as checks:
            report["model_equivalence"] = list(checks.map(lambda g:preflight(g, args.port_base, replicas, stride), (0,1,2,3)))
        report["temporary_replica_pids"] = [p for row in report["model_equivalence"] for p in row["replica_pids"]]
        report["mps_clients"] = {r["gpu"]:controllers[r["gpu"]].verify_clients(r["replica_pids"]) for r in report["model_equivalence"]}
        report.update(status="collecting", collection_started_utc=probe.now())
        started = time.monotonic()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4*replicas)
        futures = [pool.submit(work, gpu, slot) for slot in range(replicas) for gpu in (0,1,2,3)]
        tick = 0
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done():
                    f.result()
            if stop.is_set() or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Strength experiment interrupted")
            used = storage_used(output)
            if used > plan["storage_quota_gib"]*1024**3 or shutil.disk_usage(output).free < plan["disk_floor_gib"]*1024**3:
                raise RuntimeError("Strength storage bound reached")
            sample = probe.telemetry([0,1,2,3])
            with lock:
                sample["active_tasks_by_gpu"] = {g:sum(t["status"]=="running" and t["gpu"]==g for t in report["tasks"]) for g in (0,1,2,3)}
                samples.append(sample)
                report.update(storage_bytes=used, latest_gpu_sample=sample,
                              completed_jobs=sum(t["status"]=="completed" for t in report["tasks"]))
                atomic_json(output / "summary.json", report)
            if tick % 10 == 0:
                print("PROGRESS " + json.dumps(dict(completed=report["completed_jobs"], total=len(plan["jobs"]),
                    util={r["gpu"]:r["utilization_percent"] for r in sample["gpus"]})), flush=True)
            tick += 1
            stop.wait(3)
        for f in futures:
            f.result()
        report.update(status="completed", collection_elapsed_seconds=time.monotonic()-started)
        report["actual_model_queries"] = sum(t["actual_model_queries"] for t in report["tasks"])
        report["queries_per_second"] = report["actual_model_queries"]/report["collection_elapsed_seconds"]
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        with lock:
            processes = list(active.values())
        for p in processes:
            if p.poll() is None:
                p.terminate()
        if pool:
            pool.shutdown(wait=True)
        for p in servers.values():
            if p.poll() is None:
                p.terminate()
        for p in servers.values():
            try:
                p.wait(timeout=45)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
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
        report["live_replica_pids_after_cleanup"] = [p for p in report.get("temporary_replica_pids",[]) if Path("/proc/%d"%p).exists()]
        report["environment_workers_stopped"] = all(p.poll() is not None for p in processes)
        report["finished_utc"] = probe.now()
        atomic_json(output / "gpu_samples.json", samples)
        atomic_json(output / "summary.json", report)
    print("SUMMARY " + json.dumps({k:report.get(k) for k in ("status", "actual_model_queries", "queries_per_second")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=11200)
    parser.add_argument("--job-timeout", type=float, default=2400)
    run(parser.parse_args())
