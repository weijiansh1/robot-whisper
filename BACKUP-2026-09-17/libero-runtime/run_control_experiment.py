"""Fan control_branch.py runs over parents x branch points x strategies x seeds, then tabulate rescue rates.

Example:
  python3 run_control_experiment.py --parents controls/parents-topo-all-i.json --ports 9570,...,9577 \
      --strategies native,random,boundary,motion_max --branch alarm --seeds 0,1 --control-queries 1 \
      --out controls/exp-01
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "envs/libero/bin/python"
_LOCK = threading.Lock()
_ACTIVE = {}
_LAST = [0.0]


def acquire(ports):
    with _LOCK:
        port = min(ports, key=lambda p: (_ACTIVE.get(p, 0), ports.index(p)))
        _ACTIVE[port] = _ACTIVE.get(port, 0) + 1
        wait = max(0.0, _LAST[0] + 0.3 - time.time())
        _LAST[0] = time.time() + wait
    time.sleep(wait)
    return port


def release(port):
    with _LOCK:
        _ACTIVE[port] -= 1


def run_job(job, ports, out_root, render, capture_root, control_queries, candidates, noise_scale, trigger="branch", extra_args=""):
    port = acquire(ports)
    out = out_root / job["tag"] / ("q%02d" % job["branch_query"]) / job["strategy"] / ("s%d" % job["seed"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        release(port)
        return dict(job, **json.loads((out / "result.json").read_text()), cached=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MAGICK_THREAD_LIMIT="1", LP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if render == "egl":
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(ports.index(port) % 8),
                   __EGL_VENDOR_LIBRARY_FILENAMES=os.path.expanduser("~/.local/syslib/glvnd/egl_vendor.d/10_nvidia.json"))
        env.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
    else:
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    cmd = [str(PY), str(ROOT / "control_branch.py"), "--parent", job["dir"], "--tag", job["tag"], "--capture-root", capture_root,
           "--branch-query", str(job["branch_query"]), "--strategy", job["strategy"], "--candidates", str(candidates),
           "--control-queries", str(control_queries), "--seed", str(job["seed"]), "--port", str(port), "--out", str(out),
           "--noise-scale", str(noise_scale), "--trigger", trigger] + extra_args.split()
    t0 = time.time()
    with open(out / "client.log", "w") as log:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    release(port)
    row = dict(job, exit_code=rc, wall_seconds=round(time.time() - t0, 1))
    if (out / "result.json").exists():
        row.update({k: v for k, v in json.loads((out / "result.json").read_text()).items() if k != "decisions"})
    print(json.dumps({k: row.get(k) for k in ("tag", "branch_query", "strategy", "seed", "exit_code", "success", "action_steps", "replay_state_max_abs_diff", "wall_seconds")}), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parents", required=True)
    p.add_argument("--ports", required=True)
    p.add_argument("--clients-per-server", type=int, default=16)
    p.add_argument("--strategies", default="native,random,boundary,small_cluster,big_cluster,motion_max,route_exit,route_stay,flow_straight,flow_curved,v82_min,kick")
    p.add_argument("--branch", default="alarm", help="'alarm' = first v8.2 alarm query; 'alarm-4'; 'online' (start at query 0 with --trigger v82); or an integer query")
    p.add_argument("--trigger", default="branch", choices=("branch", "v82", "route_step", "either", "diam", "diam_or_v82", "knn", "knn_or_v82"))
    p.add_argument("--include-successes", action="store_true", help="also branch successful parents (needed for online false-alarm cost)")
    p.add_argument("--extra-args", default="", help="extra arguments passed verbatim to control_branch.py")
    p.add_argument("--seeds", default="0")
    p.add_argument("--control-queries", type=int, default=1)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--benchmarks", default="pro,plus")
    p.add_argument("--max-parents", type=int, default=0)
    p.add_argument("--render", default="egl")
    p.add_argument("--capture-root", default=str(ROOT.parent / "moe-capture" / "topo-20260916c"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parents = [x for x in json.load(open(a.parents)) if (a.include_successes or not x["success"]) and x["benchmark"] in a.benchmarks.split(",")]
    if a.branch == "online":
        for x in parents:
            x["branch_query"] = 0
    elif a.branch.startswith("alarm"):
        offset = int(a.branch[5:]) if len(a.branch) > 5 else 0
        parents = [x for x in parents if x["first_alarm"] is not None]
        for x in parents:
            x["branch_query"] = max(1, min(x["first_alarm"] + offset, x["queries"] - 2))
    else:
        for x in parents:
            x["branch_query"] = int(a.branch)
    if a.max_parents:
        parents = parents[: a.max_parents]
    ports = [int(x) for x in a.ports.split(",")]
    jobs = [dict(tag=x["tag"], dir=x["dir"], benchmark=x["benchmark"], task_id=x["task_id"], branch_query=x["branch_query"],
                 strategy=s, seed=int(seed)) for x in parents for s in a.strategies.split(",") for seed in a.seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    print("parents=%d jobs=%d workers=%d" % (len(parents), len(jobs), len(ports) * a.clients_per_server), flush=True)
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(len(ports) * a.clients_per_server) as ex:
        futs = [ex.submit(run_job, j, ports, a.out, a.render, a.capture_root, a.control_queries, a.candidates, a.noise_scale, a.trigger, a.extra_args) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    keys = ["tag", "benchmark", "task_id", "branch_query", "strategy", "seed", "exit_code", "success", "action_steps",
            "controlled_queries", "replay_state_max_abs_diff", "wall_seconds"]
    with (a.out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report = {}
    for s in a.strategies.split(","):
        rs = [r for r in rows if r["strategy"] == s and r.get("exit_code", 1) == 0]
        n, k = len(rs), sum(1 for r in rs if r.get("success"))
        report[s] = dict(branches=n, rescued=k, rate=round(k / n, 3) if n else None,
                         parent_success_kept=[sum(1 for r in rs if r.get("parent_success") and r.get("success")), sum(1 for r in rs if r.get("parent_success"))],
                         parent_failure_rescued=[sum(1 for r in rs if not r.get("parent_success") and r.get("success")), sum(1 for r in rs if not r.get("parent_success"))],
                         triggered=sum(1 for r in rs if r.get("trigger_query") is not None),
                         by_benchmark={b: [sum(1 for r in rs if r["benchmark"] == b and r.get("success")), sum(1 for r in rs if r["benchmark"] == b)]
                                       for b in sorted({r["benchmark"] for r in rs})})
    report["_meta"] = dict(parents=len(parents), jobs=len(jobs), failed_jobs=sum(1 for r in rows if r.get("exit_code", 1) != 0),
                           wall_seconds=round(time.time() - t0, 1), branch=a.branch, control_queries=a.control_queries, candidates=a.candidates, noise_scale=a.noise_scale,
                           replay_mismatch=sum(1 for r in rows if (r.get("replay_state_max_abs_diff") or 0) > 1e-6))
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
