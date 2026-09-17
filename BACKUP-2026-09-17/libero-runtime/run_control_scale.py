"""Large-scale generalisation of the recurrence-triggered pause: new episodes on LIBERO-10, Pro (swap) and Plus,
arms native / hold16 / hold16_random, parallel requests to the API servers, paired analysis per benchmark."""
import argparse, csv, json, os, subprocess, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
PY = ROOT / "envs/libero/bin/python"
RUNNER = os.environ.get("CONTROL_RUNNER", "run_control_episode_v4.py")
sys.path.insert(0, str(ROOT))
from run_parallel_batch import plus_category_sample


def parse_range(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_jobs(a):
    jobs = []
    for t in parse_range(a.tasks):
        for i in parse_range(a.libero10_inits):
            jobs.append(dict(bench="libero10", benchmark="pro", perturbation="none", task_id=t, task_name=None, init_state_id=i, category="", tag="l10-task%02d-init%02d" % (t, i)))
        for i in parse_range(a.pro_inits):
            jobs.append(dict(bench="pro-swap", benchmark="pro", perturbation="swap", task_id=t, task_name=None, init_state_id=i, category="", tag="pro-swap-task%02d-init%02d" % (t, i)))
    if a.plus_per_category:
        for spec in plus_category_sample(a.plus_per_category, a.plus_seed):
            jobs.append(dict(bench="plus", benchmark="plus", perturbation="", task_id=spec["task_id"], task_name=spec["task_name"], init_state_id=0,
                             category=spec["category"], tag="plus-" + spec["label"]))
    return jobs


def run_one(job, arm, out_root, ports, lock, rows, counter):
    out = out_root / arm / job["tag"]
    f = out / "summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        if s.get("status") == "completed":
            rows.append(dict(job, arm=arm, status="completed", skipped=True, **{k: s.get(k) for k in ("success", "action_steps", "inference_calls", "physical_steps", "first_trigger_query", "max_rr", "random_first_query")}, interventions=len(s.get("interventions", []))))
            return
    out.mkdir(parents=True, exist_ok=True)
    with lock:
        primary = ports[counter[0] % len(ports)]; counter[0] += 1
    order = [primary] + [p for p in ports if p != primary]
    cmd = [str(PY), str(ROOT / RUNNER), "--benchmark", job["benchmark"], "--arm", arm, "--ports", ",".join(map(str, order)), "--out", str(out), "--init-state-id", str(job["init_state_id"])]
    cmd += ["--task-name", job["task_name"]] if job["benchmark"] == "plus" else ["--perturbation", job["perturbation"], "--task-id", str(job["task_id"])]
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", OMP_NUM_THREADS="1", MAGICK_THREAD_LIMIT="1", LP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time(); rc = -1
    for attempt in range(2):
        with (out / "client.log").open("w") as log:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc == 0 and f.exists():
            break
        time.sleep(5)
    s = json.loads(f.read_text()) if f.exists() else {}
    row = dict(job, arm=arm, rc=rc, wall=round(time.time() - t0, 1), status=s.get("status", "failed"), skipped=False,
               **{k: s.get(k) for k in ("success", "action_steps", "inference_calls", "physical_steps", "first_trigger_query", "max_rr", "random_first_query")}, interventions=len(s.get("interventions", [])))
    with lock:
        rows.append(row); print(json.dumps(row), flush=True)


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


def report(rows, arms, out_root):
    done = [r for r in rows if r.get("status") == "completed"]
    native = {r["tag"]: r for r in done if r["arm"] == "native"}
    rep = {}
    for bench in sorted({r["bench"] for r in rows}):
        nat = [r for r in native.values() if r["bench"] == bench]
        b = dict(native_episodes=len(nat), native_successes=sum(1 for r in nat if r["success"]))
        for arm in arms:
            if arm == "native": continue
            rs = [r for r in done if r["arm"] == arm and r["bench"] == bench and r["tag"] in native]
            fails = [r for r in rs if not native[r["tag"]]["success"]]; succs = [r for r in rs if native[r["tag"]]["success"]]
            resc = [r for r in fails if r["success"]]; dam = [r for r in succs if not r["success"]]
            per_task = {}
            for r in fails:
                per_task.setdefault(r["task_id"], [0, 0]); per_task[r["task_id"]][1] += 1; per_task[r["task_id"]][0] += int(r["success"])
            b[arm] = dict(paired=len(rs), failures=len(fails), rescued=len(resc), rescued_ci=wilson(len(resc), len(fails)), successes=len(succs), damaged=len(dam),
                          damaged_ci=wilson(len(dam), len(succs)), net=len(resc) - len(dam), triggered_failures=sum(1 for r in fails if r["first_trigger_query"] is not None),
                          triggered_successes=sum(1 for r in succs if r["first_trigger_query"] is not None),
                          slowed_successes=sum(1 for r in succs if r["success"] and r["action_steps"] > native[r["tag"]]["action_steps"]),
                          rescued_per_task={str(k): "%d/%d" % tuple(v) for k, v in sorted(per_task.items())}, rescued_tags=[r["tag"] for r in resc], damaged_tags=[r["tag"] for r in dam])
        rep[bench] = b
    rep["runs"] = dict(total=len(rows), completed=len(done), failed=len(rows) - len(done))
    (out_root / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="native,hold16,hold16_random")
    ap.add_argument("--ports", default="9510,9511,9512,9513,9514,9515,9516,9517")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--libero10-inits", default="10-29")
    ap.add_argument("--pro-inits", default="0-19")
    ap.add_argument("--plus-per-category", type=int, default=3)
    ap.add_argument("--plus-seed", type=int, default=20260917)
    ap.add_argument("--limit-per-benchmark", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = build_jobs(a)
    if a.limit_per_benchmark:
        keep = []
        for bench in ("libero10", "pro-swap", "plus"):
            keep += [j for j in jobs if j["bench"] == bench][: a.limit_per_benchmark]
        jobs = keep
    arms = a.arms.split(",")
    ports = [int(p) for p in a.ports.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    heavy = {"Sensor Noise": 0, "Light Conditions": 1}
    order = sorted(jobs, key=lambda j: (heavy.get(j["category"], 2), 0 if j["bench"] == "pro-swap" else 1))
    work = [(j, arm) for j in order for arm in arms]     # native and control arms of one episode run close together
    print("jobs %d x arms %d = %d runs, workers %d, runner %s" % (len(jobs), len(arms), len(work), a.workers, RUNNER), flush=True)
    rows, lock, counter = [], threading.Lock(), [0]
    t0 = time.time()
    if not a.report_only:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(lambda w: run_one(w[0], w[1], a.out, ports, lock, rows, counter), work))
    else:
        for j, arm in work:
            run_one(j, arm, a.out, ports, lock, rows, counter) if (a.out / arm / j["tag"] / "summary.json").exists() else None
    keys = sorted({k for r in rows for k in r})
    with (a.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    rep = report(rows, arms, a.out)
    rep["wall_seconds"] = round(time.time() - t0, 1)
    (a.out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
