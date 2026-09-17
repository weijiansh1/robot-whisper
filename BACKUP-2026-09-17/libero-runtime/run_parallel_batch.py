"""Large-scale LIBERO evaluation driver: fan episodes out over several policy servers.

Each episode is one run_benchmark.py process; a pool of worker slots is assigned
round-robin to the given server ports. Every episode gets its own output root so
same-second artifact names cannot collide. Results are aggregated to summary.csv.
"""
import argparse, csv, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_LOCK = threading.Lock()
_ACTIVE = {}          # port -> number of clients currently attached
_LAST_START = [0.0]
STAGGER_SECONDS = 0.5  # LIBERO rewrites its config.yaml on import; staggering avoids the read/write race


def acquire_port(ports):
    """Pick the server with the fewest attached clients (dynamic load balancing) and stagger launches."""
    with _LOCK:
        port = min(ports, key=lambda p: (_ACTIVE.get(p, 0), ports.index(p)))
        _ACTIVE[port] = _ACTIVE.get(port, 0) + 1
        wait = max(0.0, _LAST_START[0] + STAGGER_SECONDS - time.time())
        _LAST_START[0] = time.time() + wait
    time.sleep(wait)
    return port


def release_port(port):
    with _LOCK:
        _ACTIVE[port] -= 1

ROOT = Path(__file__).resolve().parent
PY = ROOT / "envs/libero/bin/python"


def run_one(job, out_root, benchmark, perturbation, runner="run_benchmark.py", episode_id_base=None, ports=None):
    task_id, init_id = job["task_id"], job["init_state_id"]
    benchmark, perturbation = job.get("benchmark", benchmark), job.get("perturbation", perturbation)
    port = job["port"] = acquire_port(ports) if ports else job["port"]
    label = job.get("label") or ("task%02d" % task_id)
    out = out_root / ("%s-init%02d" % (label, init_id))
    out.mkdir(parents=True, exist_ok=True)
    # every client is one process; stop OpenMP/ImageMagick/llvmpipe from spawning a thread per core each
    # the few image-post-processing categories are the critical path of a batch: give them a few threads
    heavy = job.get("category") in ("Sensor Noise", "Light Conditions")
    threads = os.environ.get("HIMOE_HEAVY_CLIENT_THREADS", "4") if heavy else "1"
    thread_env = {var: threads for var in ("OMP_NUM_THREADS", "MAGICK_THREAD_LIMIT", "LP_NUM_THREADS", "MKL_NUM_THREADS")}
    if os.environ.get("HIMOE_RENDER", "osmesa") == "egl":   # EGL is ~37x faster but not pixel-identical to OSMesa/remote
        env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                   MUJOCO_EGL_DEVICE_ID=str(os.environ.get("HIMOE_EGL_DEVICE", (ports.index(port) % 8) if ports else 0)),
                   __EGL_VENDOR_LIBRARY_FILENAMES=os.path.expanduser("~/.local/syslib/glvnd/egl_vendor.d/10_nvidia.json"),
                   **thread_env)
        env.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    else:
        env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", **thread_env)
    cmd = [str(PY), str(ROOT / runner), "--benchmark", benchmark,
           "--perturbation", perturbation,
           "--init-state-id", str(init_id), "--port", str(port), "--output-root", str(out)]
    cmd += ["--task-name", job["task_name"]] if job.get("task_name") else ["--task-id", str(task_id)]
    if episode_id_base is not None:
        cmd += ["--episode-id", str(episode_id_base + job["ordinal"]), "--capture-tag", "%s-init%02d" % (label, init_id)]
    t0 = time.time()
    summaries = list(out.glob("episode-*/summary.json"))
    if summaries:  # resume: episode already finished in an earlier run
        rc = 0
    else:
        for attempt in range(2):  # retry once: LIBERO's config.yaml read/write race at import time
            with open(out / ("client.log" if attempt == 0 else "client.retry.log"), "w") as log:
                rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            summaries = list(out.glob("episode-*/summary.json"))
            if rc == 0 and summaries:
                break
            time.sleep(2)
    if ports:
        release_port(port)
    row = dict(job, exit_code=rc, wall_seconds=round(time.time() - t0, 1))
    if rc == 0 and summaries:
        s = json.loads(summaries[0].read_text())
        row.update(status=s.get("status"), success=s.get("success"),
                   action_steps=s.get("action_steps"), inference_calls=s.get("inference_calls"),
                   duration_seconds=round(s.get("duration_seconds", 0), 1),
                   task_name=s.get("selected_task_name"))
    else:
        row.update(status="failed", success=None)
    print(json.dumps(row), flush=True)
    return row


def plus_category_sample(per_category, seed):
    """Balanced LIBERO-Plus sample: K variants for every (base libero_10 task, category)."""
    import ast, random
    module = ast.parse((ROOT / "upstream/LIBERO-PRO/libero/libero/benchmark/libero_suite_task_map.py").read_text())
    base_tasks = next(ast.literal_eval(n.value) for n in module.body
                      if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "libero_task_map")["libero_10"]
    classification = json.loads(
        (ROOT / "upstream/LIBERO-plus/libero/libero/benchmark/task_classification.json").read_text())["libero_10"]
    rng = random.Random(seed); specs = []
    for base_id, base in enumerate(base_tasks):
        for cat in sorted({i["category"] for i in classification}):
            cands = sorted(i["name"] for i in classification
                           if i["category"] == cat and i["name"].startswith(base + "_"))
            for k, name in enumerate(rng.sample(cands, min(per_category, len(cands)))):
                specs.append({"task_id": base_id, "task_name": name, "category": cat,
                              "label": "task%02d-%s-%d" % (base_id, cat.replace(" ", ""), k)})
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ports", default="9510", help="comma separated server ports")
    p.add_argument("--clients-per-server", type=int, default=4)
    p.add_argument("--benchmark", choices=("plus", "pro"), default="pro")
    p.add_argument("--perturbation", default="swap")
    p.add_argument("--tasks", default="0-9", help="e.g. 0-9 or 0,3,5")
    p.add_argument("--init-states", default="0-7")
    p.add_argument("--plus-per-category", type=int, default=0,
                   help="plus only: sample K variants per (base task x perturbation category) instead of --tasks")
    p.add_argument("--sample-seed", type=int, default=20260916)
    p.add_argument("--extra", action="append", default=[],
                   help="additional job set sharing this driver, e.g. 'benchmark=plus,plus_per_category=1,init_states=0' "
                        "or 'benchmark=pro,perturbation=swap,tasks=0-9,init_states=0-3'")
    p.add_argument("--runner", default="run_benchmark.py", help="run_benchmark.py or run_benchmark_capture.py")
    p.add_argument("--episode-id-base", type=int, default=None, help="with the capture runner: unique episode ids start here")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    def parse(spec):
        out = []
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return out

    ports = [int(x) for x in a.ports.split(",")]
    sets = [dict(benchmark=a.benchmark, perturbation=a.perturbation, tasks=a.tasks, init_states=a.init_states,
                 plus_per_category=a.plus_per_category)]
    for extra in a.extra:
        kv = dict(item.split("=", 1) for item in extra.split(","))
        sets.append(dict(benchmark=kv.get("benchmark", a.benchmark), perturbation=kv.get("perturbation", "swap"),
                         tasks=kv.get("tasks", "0-9"), init_states=kv.get("init_states", "0"),
                         plus_per_category=int(kv.get("plus_per_category", 0))))
    jobs = []
    for si, st in enumerate(sets):
        if st["plus_per_category"]:
            specs = plus_category_sample(st["plus_per_category"], a.sample_seed)
        else:
            specs = [{"task_id": t} for t in parse(st["tasks"])]
        for spec, s in [(spec, s) for spec in specs for s in parse(st["init_states"])]:
            label = spec.get("label") or ("task%02d" % spec["task_id"])
            jobs.append(dict(spec, benchmark=st["benchmark"], perturbation=st["perturbation"], port=None,
                             init_state_id=s, ordinal=len(jobs), label=label if len(sets) == 1 else "%s-%s" % (st["benchmark"], label)))
    # longest-first: episodes expected to run the full budget (perturbed benchmarks, low-success tasks) start first
    slow_tasks = {8, 9, 6, 0}          # observed lowest success on LIBERO-10 and Plus
    slow_categories = {"Sensor Noise": 0, "Light Conditions": 1}   # measured 2-5x longer per episode (image post-processing)
    def expected_length(job):
        long_bench = job["benchmark"] == "plus" or job["perturbation"] != "none"
        return (slow_categories.get(job.get("category"), 2), 0 if long_bench else 1,
                0 if job["task_id"] in slow_tasks else 1, job["ordinal"])
    jobs.sort(key=expected_length)
    a.out.mkdir(parents=True, exist_ok=True)
    workers = len(ports) * a.clients_per_server
    print("episodes=%d servers=%d workers=%d" % (len(jobs), len(ports), workers), flush=True)
    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, j, a.out, a.benchmark, a.perturbation, a.runner, a.episode_id_base, ports) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    wall = time.time() - t0
    rows.sort(key=lambda r: (r["task_id"], r["init_state_id"]))
    keys = ["benchmark", "perturbation", "task_id", "category", "init_state_id", "port", "exit_code", "status", "success",
            "action_steps", "inference_calls", "duration_seconds", "wall_seconds", "task_name"]
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["exit_code"] == 0]
    succ = [r for r in ok if r.get("success")]
    report = {"episodes": len(rows), "completed": len(ok), "failed": len(rows) - len(ok),
              "successes": len(succ), "success_rate": round(len(succ) / max(1, len(ok)), 3),
              "wall_seconds": round(wall, 1), "episodes_per_minute": round(60 * len(ok) / wall, 2),
              "mean_episode_seconds": round(sum(r["wall_seconds"] for r in ok) / max(1, len(ok)), 1)}
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
