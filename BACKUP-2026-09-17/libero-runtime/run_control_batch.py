"""Parallel driver for run_control_episode.py over the 170 topology episodes x arms (API-service style)."""
import argparse, csv, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "envs/libero/bin/python"


def jobs_from_topology():
    jobs = []
    for ds, bench in (("topo-libero10", "pro"), ("topo-plus", "plus")):
        for tag_dir in sorted((ROOT / "simulations" / ds).iterdir()):
            eps = sorted(tag_dir.glob("episode-*/summary.json"))
            if not eps:
                continue
            s = json.loads(eps[0].read_text())
            job = dict(dataset=ds, tag=tag_dir.name, benchmark=bench, task_name=s["task_name"], task_id=int(s["task_id"]),
                       init_state_id=int(s["init_state_id"]), native_success=bool(s["success"]), native_steps=int(s["action_steps"]))
            jobs.append(job)
    return jobs


def run_one(job, arm, out_root, ports, theta, confirm, log_lock, rows):
    out = out_root / arm / job["tag"]
    if (out / "summary.json").exists():
        s = json.loads((out / "summary.json").read_text())
        if s.get("status") == "completed":
            rows.append(dict(job, arm=arm, **{k: s.get(k) for k in ("success", "action_steps", "inference_calls", "candidate_calls", "physical_steps", "first_trigger_query", "max_rr")}, interventions=len(s.get("interventions", [])), skipped=True))
            return
    out.mkdir(parents=True, exist_ok=True)
    primary = ports[run_one.counter % len(ports)]
    run_one.counter += 1
    order = [primary] + [p for p in ports if p != primary]
    cmd = [str(PY), str(ROOT / os.environ.get("CONTROL_RUNNER", "run_control_episode.py")), "--benchmark", job["benchmark"], "--arm", arm, "--theta", str(theta), "--confirm", str(confirm),
           "--ports", ",".join(str(p) for p in order), "--out", str(out), "--init-state-id", str(job["init_state_id"])]
    if job["benchmark"] == "plus":
        cmd += ["--task-name", job["task_name"]]
    else:
        cmd += ["--perturbation", "none", "--task-id", str(job["task_id"])]
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", OMP_NUM_THREADS="1", MAGICK_THREAD_LIMIT="1", LP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    for attempt in range(2):
        with (out / "client.log").open("w") as log:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc == 0 and (out / "summary.json").exists():
            break
        time.sleep(3)
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    row = dict(job, arm=arm, rc=rc, wall=round(time.time() - t0, 1), status=s.get("status", "failed"),
               **{k: s.get(k) for k in ("success", "action_steps", "inference_calls", "candidate_calls", "physical_steps", "first_trigger_query", "max_rr")},
               interventions=len(s.get("interventions", [])), skipped=False)
    with log_lock:
        rows.append(row)
        print(json.dumps(row), flush=True)


run_one.counter = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="withdraw,escalate,resample_escape,resample_random")
    ap.add_argument("--ports", default="9510,9511,9512,9513,9514,9515,9516,9517")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (smoke)")
    ap.add_argument("--only-failures", action="store_true")
    ap.add_argument("--tags", default="", help="comma separated tag filter (smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = jobs_from_topology()
    if args.only_failures:
        jobs = [j for j in jobs if not j["native_success"]]
    if args.tags:
        want = set(args.tags.split(","))
        jobs = [j for j in jobs if j["tag"] in want]
    if args.limit:
        jobs = jobs[: args.limit]
    ports = [int(p) for p in args.ports.split(",")]
    arms = args.arms.split(",")
    # long (failed) episodes first so the tail is short
    work = [(j, a) for a in arms for j in sorted(jobs, key=lambda j: -j["native_steps"])]
    print("jobs %d x arms %d = %d runs, workers %d" % (len(jobs), len(arms), len(work), args.workers), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, lock = [], threading.Lock()
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda w: run_one(w[0], w[1], args.out, ports, args.theta, args.confirm, lock, rows), work))
    keys = sorted({k for r in rows for k in r})
    with (args.out / "summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    report = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a and r.get("status") == "completed"]
        rescued = sum(1 for r in rs if (not r["native_success"]) and r["success"])
        damaged = sum(1 for r in rs if r["native_success"] and not r["success"])
        report[a] = dict(completed=len(rs), failed_runs=sum(1 for r in rows if r["arm"] == a and r.get("status") != "completed"),
                         successes=sum(1 for r in rs if r["success"]), rescued=rescued, damaged=damaged,
                         triggered=sum(1 for r in rs if r["first_trigger_query"] is not None),
                         triggered_failures=sum(1 for r in rs if r["first_trigger_query"] is not None and not r["native_success"]),
                         triggered_successes=sum(1 for r in rs if r["first_trigger_query"] is not None and r["native_success"]))
    report["native"] = dict(successes=sum(1 for j in jobs if j["native_success"]), episodes=len(jobs))
    report["wall_seconds"] = round(time.time() - t0, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
