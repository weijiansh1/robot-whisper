"""Analyse a control batch (run_control_batch.py output): rescue/damage per arm, trigger timing, RR dynamics after interventions."""
import json, sys, csv
from pathlib import Path
import numpy as np

ROOT = Path("/home/swj/data/libero-runtime")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "simulations/control-r1"
THETA = 0.4


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


sys.path.insert(0, str(ROOT))
from run_control_batch import jobs_from_topology
jobs = {j["tag"]: j for j in jobs_from_topology()}
arms = sorted(d.name for d in OUT.iterdir() if d.is_dir())
res = {"native": dict(successes=sum(j["native_success"] for j in jobs.values()), episodes=len(jobs))}
print("native: %d/%d successes" % (res["native"]["successes"], len(jobs)))
for arm in arms:
    rows = []
    for d in sorted((OUT / arm).iterdir()):
        f = d / "summary.json"
        if not f.exists(): continue
        s = json.loads(f.read_text())
        if s.get("status") != "completed": continue
        j = jobs[d.name]
        c = np.load(d / "control.npz") if (d / "control.npz").exists() else None
        rr_drop, rr_after = [], []
        if c is not None and s["interventions"]:
            pq = c["per_query"]
            for r in s["interventions"]:
                q = r["query"]
                after = pq[(pq[:, 0] > q) & (pq[:, 0] <= q + 6), 2]
                after = after[np.isfinite(after)]
                if after.size:
                    rr_after.append(float(after.mean())); rr_drop.append(bool((after < THETA).any()))
        rows.append(dict(tag=d.name, ds=j["dataset"], native=j["native_success"], success=s["success"], steps=s["action_steps"],
                         first=s["first_trigger_query"], n_int=len(s["interventions"]), phys=s["physical_steps"], cand=s["candidate_calls"],
                         rr_after=np.mean(rr_after) if rr_after else np.nan, rr_drop=np.mean(rr_drop) if rr_drop else np.nan,
                         native_steps=j["native_steps"]))
    n = len(rows)
    if not n: continue
    fails = [r for r in rows if not r["native"]]; succs = [r for r in rows if r["native"]]
    rescued = [r for r in fails if r["success"]]; damaged = [r for r in succs if not r["success"]]
    trig_f = [r for r in fails if r["first"] is not None]; trig_s = [r for r in succs if r["first"] is not None]
    slower = [r for r in succs if r["success"] and r["steps"] > r["native_steps"]]
    line = dict(runs=n, failures=len(fails), successes_native=len(succs), rescued=len(rescued), rescued_ci=wilson(len(rescued), len(fails)),
                damaged=len(damaged), damaged_ci=wilson(len(damaged), len(succs)), net=len(rescued) - len(damaged),
                triggered_failures=len(trig_f), triggered_successes=len(trig_s),
                median_first_trigger_q=float(np.median([r["first"] for r in trig_f])) if trig_f else None,
                interventions_per_triggered=float(np.mean([r["n_int"] for r in trig_f + trig_s])) if (trig_f or trig_s) else 0,
                rr_after_mean=float(np.nanmean([r["rr_after"] for r in rows])), rr_drop_frac=float(np.nanmean([r["rr_drop"] for r in rows])),
                slowed_successes=len(slower), extra_steps_on_slowed=float(np.mean([r["steps"] - r["native_steps"] for r in slower])) if slower else 0,
                mean_physical_steps=float(np.mean([r["phys"] for r in rows])), mean_candidate_calls=float(np.mean([r["cand"] for r in rows])),
                by_dataset={ds: dict(rescued=sum(1 for r in rescued if r["ds"] == ds), failures=sum(1 for r in fails if r["ds"] == ds),
                                     damaged=sum(1 for r in damaged if r["ds"] == ds), successes=sum(1 for r in succs if r["ds"] == ds)) for ds in ("topo-libero10", "topo-plus")},
                rescued_tags=[r["tag"] for r in rescued], damaged_tags=[r["tag"] for r in damaged])
    res[arm] = line
    print("%-16s runs %3d | rescued %d/%d [%.2f,%.2f] | damaged %d/%d [%.2f,%.2f] | net %+d | triggered fail %d succ %d, median first q %s | RR after interventions %.2f, dropped<theta within 6q in %.0f%% | slowed successes %d (+%.0f steps)" % (
        arm, n, len(rescued), len(fails), *line["rescued_ci"], len(damaged), len(succs), *line["damaged_ci"], line["net"], len(trig_f), len(trig_s),
        line["median_first_trigger_q"], line["rr_after_mean"], 100 * line["rr_drop_frac"], len(slower), line["extra_steps_on_slowed"]))
    print("     by dataset:", line["by_dataset"], "| rescued:", line["rescued_tags"], "| damaged:", line["damaged_tags"])
(OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=float))
