#!/usr/bin/env python3
"""Bounded, isolated v8 feature experiment with original-service fidelity checks."""

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

from run_collection_preflight import (HERE, BASE, ROOTS, environment, scaling, probe,
    PrivateMPS, PersistentWorker, replica_port, open_replica, verify_server, storage_used)
from collection_storage import atomic_json, digest
from v8_feature_control import PROTOCOL, ARMS, BIAS_KEY, make_bias, flow_features
from collection_routes import CAPTURE_KEY, PROBS_KEY, ALL_FIELDS


def verify_intervention(gpu, port_base, stride):
    with contextlib.ExitStack() as stack:
        client, _ = open_replica(stack, gpu, port_base, "long", 0, stride)
        request = probe.observation("long", 117000 + gpu, capture=True)
        request[CAPTURE_KEY] = True
        native, _ = probe.infer(client, request, "long")
        request[BIAS_KEY] = np.zeros((8, 10, 11, 32), np.float32)
        zero, _ = probe.infer(client, request, "long")
        for key in ("actions", *ALL_FIELDS):
            np.testing.assert_array_equal(native[key], zero[key], err_msg="zero bias fidelity " + key)
        request[BIAS_KEY] = make_bias(native[PROBS_KEY], native[PROBS_KEY], "combined", 1, 3)
        controlled, _ = probe.infer(client, request, "long")
        request.pop(BIAS_KEY)
        released, _ = probe.infer(client, request, "long")
        for key in ("actions", *ALL_FIELDS):
            np.testing.assert_array_equal(native[key], released[key], err_msg="hook cleanup " + key)
        return dict(gpu=gpu, zero_bias_exact=True, release_exact=True,
            action_rms=float(np.sqrt(np.square(controlled["actions"] - native["actions"]).mean())),
            native_features=flow_features(native[PROBS_KEY])[0].tolist(),
            controlled_features=flow_features(controlled["v8_control/effective_probs_fp32"])[0].tolist())


def run(args):
    plan = json.loads(args.plan.read_text())
    if plan["protocol"] != PROTOCOL or plan["arms_registry"] != ARMS or plan["allowed_gpus"] != [0, 1, 2, 3]:
        raise ValueError("Frozen feature plan mismatch")
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Plan source changed: " + path)
    if shutil.disk_usage(HERE).free - plan["maximum_output_bytes"] < plan["disk_floor_gib"] * 1024**3:
        raise ValueError("Insufficient conservative disk headroom")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "sources").mkdir()
    sources = ("v8_feature_control.py", "collect_v8_feature_control.py", "serve_v8_feature_model.py",
        "run_v8_feature_experiment.py", "prepare_v8_feature_experiment.py", "collection_routes.py",
        "collection_storage.py", "collection_protocol.py", "collect_adaptive_control.py", "adaptive_control.py",
        "collect_preflight_worker.py", "collection_noise.py", "collection_mps.py", "collection_worker_pool.py",
        "serve_isolated_model.py", "run_collection_preflight.py")
    for name in sources:
        shutil.copy2(HERE / name, output / "sources" / name)
    shutil.copy2(args.plan, output / "plan.json")
    report = dict(status="starting", protocol=PROTOCOL, plan_sha256=digest(args.plan), started_utc=probe.now(),
        sources={name: digest(output / "sources" / name) for name in sources},
        tasks=[dict(main_id=t["main_id"], benchmark=t["benchmark"], status="queued",
                    output=str(output / "tasks" / t["main_id"])) for t in plan["tasks"]])
    stop, lock = threading.Event(), threading.Lock()
    pending = queue.Queue()
    for i in range(len(plan["tasks"])):
        pending.put(i)
    replicas, stride = plan["replicas_per_gpu"], max(10, 5 * plan["replicas_per_gpu"])
    servers, controllers, logs, active, samples = {}, {}, [], {}, []
    pool = None
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    def worker(gpu, slot):
        workers = {}
        try:
            while not stop.is_set():
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                task, record = plan["tasks"][index], report["tasks"][index]
                benchmark, render = task["benchmark"], 0 if gpu < 2 else 3
                if benchmark not in workers:
                    log = (output / "logs" / ("env_%d_%d_%s.log" % (gpu, slot, benchmark))).open("w")
                    command = [str(BASE / "envs/libero/bin/python"), "-u", str(HERE / "collect_v8_feature_control.py"),
                        "--benchmark", benchmark, "--gpu", str(gpu), "--render-gpu", str(render),
                        "--port", str(replica_port(gpu, args.port_base, "long", slot, stride)), "--exact-noise-fastpath"]
                    process = PersistentWorker(command, environment(benchmark, render, "egl"), str(ROOTS[benchmark]), log)
                    workers[benchmark] = process
                    with lock:
                        active[process.process.pid] = process.process
                process = workers[benchmark]
                with lock:
                    record.update(status="running", gpu=gpu, replica=slot, pid=process.process.pid)
                contract = {key: plan[key] for key in ("timing", "control", "random_null", "recovery", "measurements",
                                                      "paired_random_streams", "hidden_capture", "threshold_fitting", "batch_size")}
                message = dict(variant=task["variant_id"], seed=task["noise_seed"], output=record["output"],
                    main_id=task["main_id"], init_index=task["init_index"],
                    control=dict(kind="feature_experiment", parent=task, arms=plan["arms"],
                                 replicates=plan["replicates"], contract=contract))
                result = process.execute(message, args.job_timeout, stop)
                final = json.loads((Path(record["output"]) / "result.json").read_text())
                passed = result["exit_code"] == 0 and final["status"] == "completed" and final["c0"]["status"] == "passed"
                with lock:
                    record.update(status="completed" if passed else "failed", error=final.get("error"),
                        branches=final["branches"], actual_model_queries=final["actual_model_queries"])
                print("TASK " + json.dumps(dict(main_id=task["main_id"], status=record["status"], gpu=gpu)), flush=True)
                if not passed:
                    stop.set()
                    raise RuntimeError(final.get("error", "Feature task failed"))
                pending.task_done()
        finally:
            for process in workers.values():
                process.close()
                with lock:
                    active.pop(process.process.pid, None)

    try:
        for gpu in plan["allowed_gpus"]:
            free = int(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True).stdout.strip())
            if free < 39000:
                raise ValueError("Insufficient GPU headroom on %d" % gpu)
            for replica in range(replicas):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride)))
            controller = PrivateMPS(gpu, output / ("gpu%d" % gpu))
            controllers[gpu] = controller
            report.setdefault("mps_start", {})[gpu] = controller.start()
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072")
            env.update({key: value for key, value in controller.env.items() if key.startswith("CUDA_")})
            log = (output / "logs" / ("model_gpu%d.log" % gpu)).open("w")
            logs.append(log)
            servers[gpu] = subprocess.Popen([str(scaling.MODEL_PYTHON), "-u", str(HERE / "serve_v8_feature_model.py"),
                "--gpu", str(gpu), "--base-port", str(args.port_base + gpu * stride), "--replicas", str(replicas)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        report["temporary_model_pids"] = {g: p.pid for g, p in servers.items()}
        atomic_json(output / "summary.json", report)
        waiting = {(gpu, r) for gpu in plan["allowed_gpus"] for r in range(replicas)}
        deadline = time.monotonic() + 600
        while waiting:
            if stop.is_set() or time.monotonic() > deadline or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Feature server startup failed or interrupted")
            for gpu, replica in list(waiting):
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1", replica_port(gpu, args.port_base, "long", replica, stride))) == 0:
                        waiting.remove((gpu, replica))
            stop.wait(.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as checks:
            report["model_equivalence"] = list(checks.map(lambda g: verify_server(g, args.port_base, "long", replicas, stride), [0, 1, 2, 3]))
            report["feature_hook_preflight"] = list(checks.map(lambda g: verify_intervention(g, args.port_base, stride), [0, 1, 2, 3]))
        report["temporary_replica_pids"] = [pid for row in report["model_equivalence"] for pid in row["replica_pids"]]
        report["mps_clients"] = {r["gpu"]: controllers[r["gpu"]].verify_clients(r["replica_pids"]) for r in report["model_equivalence"]}
        report.update(status="collecting", collection_started_utc=probe.now())
        started = time.monotonic()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4 * replicas)
        futures = [pool.submit(worker, gpu, slot) for slot in range(replicas) for gpu in [0, 1, 2, 3]]
        iteration = 0
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done():
                    f.result()
            if stop.is_set() or any(p.poll() is not None for p in servers.values()):
                raise RuntimeError("Feature experiment interrupted")
            used = storage_used(output)
            if used > plan["storage_quota_gib"] * 1024**3 or shutil.disk_usage(output).free < plan["disk_floor_gib"] * 1024**3:
                raise RuntimeError("Feature experiment storage bound reached")
            sample = probe.telemetry([0, 1, 2, 3])
            with lock:
                sample["active_tasks_by_gpu"] = {gpu: sum(t["status"] == "running" and t["gpu"] == gpu for t in report["tasks"]) for gpu in [0, 1, 2, 3]}
                samples.append(sample)
                report.update(storage_bytes=used, latest_gpu_sample=sample,
                    completed_tasks=sum(t["status"] == "completed" for t in report["tasks"]))
                atomic_json(output / "summary.json", report)
            if iteration % 10 == 0:
                print("PROGRESS " + json.dumps(dict(completed=report["completed_tasks"], total=len(plan["tasks"]),
                    gpu_util={r["gpu"]: r["utilization_percent"] for r in sample["gpus"]},
                    power={r["gpu"]: r["power_w"] for r in sample["gpus"]})), flush=True)
            iteration += 1
            stop.wait(3)
        for f in futures:
            f.result()
        report.update(status="completed", collection_elapsed_seconds=time.monotonic() - started)
        report["actual_model_queries"] = sum(t["actual_model_queries"] for t in report["tasks"])
        report["queries_per_second"] = report["actual_model_queries"] / report["collection_elapsed_seconds"]
        report["gpu_summary"] = [dict(gpu=gpu,
            mean_utilization_percent=float(np.mean([r["utilization_percent"] for s in samples for r in s["gpus"] if r["gpu"] == gpu])),
            mean_power_w=float(np.mean([r["power_w"] for s in samples for r in s["gpus"] if r["gpu"] == gpu])),
            max_power_w=max(r["power_w"] for s in samples for r in s["gpus"] if r["gpu"] == gpu)) for gpu in [0, 1, 2, 3]]
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
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        for log in logs:
            log.close()
        report["mps_cleanup"] = {g: c.close() for g, c in controllers.items()}
        report["temporary_models_stopped"] = all(p.poll() is not None for p in servers.values())
        report["live_replica_pids_after_cleanup"] = [p for p in report.get("temporary_replica_pids", []) if Path("/proc/%d" % p).exists()]
        report["environment_workers_stopped"] = all(p.poll() is not None for p in processes)
        report["finished_utc"] = probe.now()
        atomic_json(output / "gpu_samples.json", samples)
        atomic_json(output / "summary.json", report)
    print("SUMMARY " + json.dumps({k: report.get(k) for k in ("status", "actual_model_queries", "queries_per_second", "gpu_summary")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=11000)
    parser.add_argument("--job-timeout", type=float, default=3600)
    run(parser.parse_args())
