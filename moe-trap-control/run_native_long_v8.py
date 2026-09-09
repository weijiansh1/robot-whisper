#!/usr/bin/env python3
"""Run original Long mains and paired frozen-selector branches on GPUs 0-3."""

import argparse
import concurrent.futures
import copy
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

from collection_protocol import stable_id
from collection_storage import atomic_json, digest
from native_long_runtime import ROOT, environment, verify_source
from run_collection_preflight import (HERE, BASE, scaling, probe, PrivateMPS, PersistentWorker,
                                      replica_port, verify_server, storage_used)
from v8_closed_loop import PROTOCOL, ARMS, SETTINGS


def resolve_parent(task, directory):
    original = json.loads((directory / "result.json").read_text())
    first = original["first_alarms"]
    q0 = first["v8_frozen"]+1
    events = []
    if first["v8_frozen"] >= 0 and q0 < original["queries"]:
        events.append(dict(event_id=stable_id(PROTOCOL, task["main_id"], q0), start_query=q0,
            alarm_query=q0-1, methods=["v8_frozen"], deployable=True))
    result = dict(task)
    result.update(parent_directory=str(directory), parent_queries=original["queries"],
        parent_commit_sha256=digest(directory / "main_complete.json"),
        parent_manifest_sha256=digest(directory / "main/manifest.json"), first_alarm=first["knn20"],
        first_alarms=first, native_success=original["success"], events=events)
    return result


def run(args):
    plan = json.loads(args.plan.read_text())
    if (plan["protocol"] != PROTOCOL or plan["settings"] != SETTINGS or plan["arms"] != list(ARMS) or
            plan["allowed_gpus"] != [0, 1, 2, 3] or plan["benchmark"] != "native_long"):
        raise ValueError("Frozen native cohort differs from runtime")
    if verify_source() != plan["original_source"]:
        raise ValueError("Original LIBERO identity changed")
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: "+path)
    environment(0, initialize=True)
    output = args.output.resolve()
    if shutil.disk_usage(output.parent).free-plan["maximum_output_bytes"] < plan["disk_floor_gib"]*1024**3:
        raise ValueError("Insufficient disk headroom")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "sources").mkdir()
    for name in plan["runtime_sources"]:
        shutil.copy2(HERE / name, output / "sources" / name)
    shutil.copy2(args.plan, output / "cohort_plan.json")
    tasks = {t["main_id"]: t for t in plan["tasks"]}
    jobs, children, ready = [], {}, queue.PriorityQueue()
    for task in plan["tasks"]:
        index = len(jobs)
        jobs.append(dict(kind="native_main", main_id=task["main_id"], status="queued",
            output=str(output / "mains" / task["main_id"])))
        ready.put((0, index))
    report = dict(status="starting", protocol=PROTOCOL, stage=plan["stage"], started_utc=probe.now(),
        cohort_plan_sha256=digest(args.plan), tasks=jobs,
        sources={name: digest(output / "sources" / name) for name in plan["runtime_sources"]})
    stop, lock = threading.Event(), threading.Lock()
    servers, controllers, active, logs, samples = {}, {}, {}, [], []
    pool, replicas, stride = None, 8, 40
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    def work(gpu, slot, phase):
        worker = None
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
                render = 0 if gpu < 2 else 3
                if worker is None:
                    log = (output / "logs" / ("env_%s_%d_%d.log" % (phase, gpu, slot))).open("w")
                    command = [str(BASE / "envs/libero/bin/python"), "-u", str(HERE / "collect_native_long_v8.py"),
                        "--gpu", str(gpu), "--render-gpu", str(render), "--port",
                        str(replica_port(gpu, args.port_base, "long", slot, stride))]
                    worker = PersistentWorker(command, environment(render), str(ROOT), log)
                    with lock:
                        active[worker.process.pid] = worker.process
                with lock:
                    job.update(status="running", gpu=gpu, replica=slot, pid=worker.process.pid)
                control = dict(kind=job["kind"], parent=task, contract=SETTINGS)
                if job["kind"] == "fixed_branches":
                    control.update(event=job["event"], arm=job["arm"], replay_directory=str(output / "replays" / task["main_id"]))
                response = worker.execute(dict(variant=task["variant_id"], seed=task["noise_seed"], output=job["output"],
                    main_id=task["main_id"], init_index=task["init_index"], control=control), args.job_timeout, stop)
                final = json.loads((Path(job["output"]) / "result.json").read_text())
                passed = response["exit_code"] == 0 and final["status"] == "completed" and not final["invalid_pair"]
                if job["kind"] == "native_main":
                    passed = passed and final["main_complete"] and not final["main_intervention"]
                elif job["kind"] == "fixed_replay":
                    passed = passed and final.get("online_triggers_exact", False)
                elif job["arm"] == "native":
                    passed = passed and final["all_native_suffixes_exact"]
                with lock:
                    job.update(status="completed" if passed else "failed", error=final.get("error"),
                        actual_model_queries=final["actual_model_queries"], success=final["success"], branches=final["branches"])
                    if passed and job["kind"] == "fixed_replay":
                        for child in children[task["main_id"]]:
                            jobs[child]["status"] = "queued"
                            weight = 1 if jobs[child]["arm"] == "native" else 4
                            ready.put((-weight*(52-jobs[child]["event"]["start_query"]), child))
                print("JOB "+json.dumps({k: job.get(k) for k in ("kind", "main_id", "arm", "status", "gpu", "success")}), flush=True)
                ready.task_done()
                if not passed:
                    raise RuntimeError(final.get("error", "Native collection task failed"))
        except BaseException:
            stop.set()
            raise
        finally:
            if worker is not None:
                worker.close()
                with lock:
                    active.pop(worker.process.pid, None)

    def drain(phase):
        nonlocal pool
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        futures = [pool.submit(work, gpu, slot, phase) for slot in range(8) for gpu in (0, 1, 2, 3)]
        tick = 0
        while not all(f.done() for f in futures):
            for future in futures:
                if future.done():
                    future.result()
            if stop.is_set() or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Native collection interrupted")
            used = storage_used(output)
            if used > plan["storage_quota_gib"]*1024**3 or shutil.disk_usage(output).free < plan["disk_floor_gib"]*1024**3:
                raise RuntimeError("Storage bound reached")
            sample = probe.telemetry([0, 1, 2, 3])
            with lock:
                sample["phase"] = phase
                sample["active_tasks_by_gpu"] = {g: sum(j["status"] == "running" and j["gpu"] == g for j in jobs) for g in (0, 1, 2, 3)}
                samples.append(sample)
                report.update(storage_bytes=used, latest_gpu_sample=sample, completed_jobs=sum(j["status"] == "completed" for j in jobs))
                atomic_json(output / "summary.json", report)
            if tick % 10 == 0:
                print("PROGRESS "+json.dumps(dict(phase=phase, completed=report["completed_jobs"], total=len(jobs),
                    util={r["gpu"]: r["utilization_percent"] for r in sample["gpus"]})), flush=True)
            tick += 1
            stop.wait(3)
        for future in futures:
            future.result()
        pool.shutdown(wait=True)
        pool = None

    try:
        for gpu in (0, 1, 2, 3):
            free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
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
        waiting, deadline = {(g, r) for g in (0, 1, 2, 3) for r in range(replicas)}, time.monotonic()+600
        while waiting:
            if stop.is_set() or time.monotonic() > deadline or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Native model startup failed")
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
        drain("mains")
        resolved = [resolve_parent(task, output / "mains" / task["main_id"]) for task in plan["tasks"]]
        tasks = {t["main_id"]: t for t in resolved}
        branch_plan = copy.deepcopy(plan)
        branch_plan.update(cohort=resolved, tasks=[t for t in resolved if t["events"]],
            cohort_plan_sha256=digest(output / "cohort_plan.json"), new_main_coverage=len(resolved))
        atomic_json(output / "plan.json", branch_plan)
        report.update(plan_sha256=digest(output / "plan.json"),
            native_mains=len(resolved), native_successes=sum(t["native_success"] for t in resolved),
            alarmed_parents=len(branch_plan["tasks"]))
        print("NATIVE_COHORT "+json.dumps({key: report[key] for key in ("native_mains", "native_successes", "alarmed_parents")}), flush=True)
        for task in resolved:
            index = len(jobs)
            jobs.append(dict(kind="fixed_replay", main_id=task["main_id"], status="queued",
                output=str(output / "replays" / task["main_id"])))
            ready.put((-task["parent_queries"], index))
            children[task["main_id"]] = []
            for event in task["events"]:
                for arm in ARMS:
                    children[task["main_id"]].append(len(jobs))
                    jobs.append(dict(kind="fixed_branches", main_id=task["main_id"], event=event, arm=arm,
                        status="waiting_for_c0", output=str(output / "events" / event["event_id"] / arm)))
        drain("recovery")
        report.update(status="completed", completed_jobs=len(jobs), collection_elapsed_seconds=time.monotonic()-started,
            actual_model_queries=sum(j["actual_model_queries"] for j in jobs))
        report["queries_per_second"] = report["actual_model_queries"]/report["collection_elapsed_seconds"]
        if verify_source() != plan["original_source"]:
            raise ValueError("Original source changed during collection")
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
    print("SUMMARY "+json.dumps({key: report.get(key) for key in ("status", "actual_model_queries", "queries_per_second")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=12800)
    parser.add_argument("--job-timeout", type=float, default=3600)
    run(parser.parse_args())
