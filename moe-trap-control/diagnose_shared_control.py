#!/usr/bin/env python3
"""Stage-wise mechanism analysis of shared-control branches: evidence -> reach -> close -> grasp -> place."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, records

ROBOT_QPOS, OBJECT_QPOS = 9, 7
LIFT_M, FULLY_CLOSED = 0.02, 0.012


def object_slice(names, name):
    start = 1 + ROBOT_QPOS + OBJECT_QPOS * names.index(name)
    return slice(start, start + 3)


def per_step_series(rows, names, initial):
    """Approximate per-step series (10 steps per query) from stored chunk rows."""
    steps = []
    for r in rows:
        n = int(r["executed_action_count"])
        log = np.asarray(r["regulator_log"], np.float32)[:n]
        executed = np.asarray(r["executed_actions"], np.float32)[:n]
        vla = np.asarray(r["actions"], np.float32)[:n]
        target = str(r["target_name"].astype(str))
        before, after = np.asarray(r["sim_before"], float), np.asarray(r["sim_after"], float)
        eef0 = np.asarray(r["proprio"][:3], float)
        ap0 = float(r["proprio"][6]) - float(r["proprio"][7])
        if target in names:
            o0, o1 = before[object_slice(names, target)], after[object_slice(names, target)]
        else:
            o0 = o1 = np.full(3, np.nan)
        for k in range(n):
            f = (k + 1) / n
            steps.append(dict(query=int(r["query"]), k=k, target=target, eef=eef0, ap=ap0,
                              obj=o0 + f * (o1 - o0), rise=float((o0 + f * (o1 - o0))[2] - initial.get(target, np.array([0, 0, np.nan]))[2]) if target in initial else np.nan,
                              gate_close=bool(log[k, 2]), gate_open=bool(log[k, 3]), in_hand=bool(log[k, 4]), modified=bool(log[k, 5]),
                              servo=float(np.abs(log[k, :2]).sum()), exec_g=float(executed[k, 6]), vla_g=float(vla[k, 6]),
                              vla_xy=vla[k, :2], success=bool(r["success"]) and k == n - 1))
    return steps


def classify(rows, names, initial, gate_xy, dz_min, dz_max):
    """Stage reached on the LAST target segment (the object the branch ended on), plus segment counts."""
    targets = [str(r["target_name"].astype(str)) for r in rows]
    segments = []
    for i, t in enumerate(targets):
        if not segments or segments[-1][0] != t:
            segments.append((t, i))
    all_rows, rows = rows, rows[segments[-1][1]:]
    steps = per_step_series(rows, names, initial)
    if not steps:
        return dict(stage="empty")
    last_target = segments[-1][0]
    end_state = None
    if last_target in names:
        last = all_rows[-1]
        ap_end = float(last["proprio"][6]) - float(last["proprio"][7])
        obj_end = np.asarray(last["sim_after"], float)[object_slice(names, last_target)]
        rise_end = float(obj_end[2] - initial[last_target][2])
        moved_end = float(np.linalg.norm(obj_end[:2] - initial[last_target][:2]))
        end_state = ("held_at_end" if ap_end < 0.05 and rise_end >= LIFT_M else
                     "resting_elsewhere" if moved_end > 0.05 else "resting_at_start")
    evidence = next((i for i, s in enumerate(steps) if s["gate_close"]), None)
    out = dict(evidence_step=evidence, authority_steps=sum(s["modified"] and s["servo"] > 0 for s in steps),
               gate_close_count=sum(s["gate_close"] for s in steps), gate_open_count=sum(s["gate_open"] for s in steps))
    success = any(s["success"] for s in steps) or bool(rows[-1]["success"])
    # distances are known only at chunk starts (proprio); evaluate reach at chunk granularity
    reach = None
    for r in rows:
        t = str(r["target_name"].astype(str))
        if t not in names:
            continue
        eef = np.asarray(r["proprio"][:3], float)
        obj = np.asarray(r["sim_before"], float)[object_slice(names, t)]
        horiz = float(np.linalg.norm(eef[:2] - obj[:2])); dz = float(eef[2] - obj[2])
        if horiz <= gate_xy and dz_min <= dz <= dz_max:
            reach = int(r["query"]); break
    out["reached_query"] = reach
    min_h = min((float(np.linalg.norm(np.asarray(r["proprio"][:2], float) - np.asarray(r["sim_before"], float)[object_slice(names, str(r["target_name"].astype(str)))][:2]))
                 for r in rows if str(r["target_name"].astype(str)) in names), default=np.nan)
    out["min_horizontal_m"] = round(min_h, 3)
    # allowed closures: executed gripper > 0 while previous aperture open, at a chunk start with the object in the envelope
    closures, misses, grasps = 0, 0, 0
    lifted_any = any(s["rise"] >= LIFT_M for s in steps if not np.isnan(s["rise"]))
    prev_ap = None
    for r in rows:
        ap = float(r["proprio"][6]) - float(r["proprio"][7])
        if prev_ap is not None and prev_ap >= 0.05 and ap < 0.05:
            closures += 1
            if ap < FULLY_CLOSED:
                misses += 1
            else:
                grasps += 1
        prev_ap = ap
    out.update(closures=closures, closed_fully=misses, closed_holding=grasps, lifted=lifted_any, success=success,
               segments=len(segments), last_target=last_target, end_state=end_state,
               gate_open_last=sum(s["gate_open"] for s in steps))
    if success:
        stage = "success"
    elif evidence is None and not lifted_any:
        stage = "no_evidence"
    elif reach is None and not lifted_any:
        stage = "never_reached"
    elif not lifted_any and closures == 0:
        stage = "reached_never_closed"
    elif not lifted_any:
        stage = "closed_missed" if misses and not grasps else "closed_not_lifted"
    else:
        stage = "lifted_not_placed"
    out["stage"] = stage
    return out


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    reg = plan["regulator"]
    replay_run = Path(plan["replay_run"])
    rows_out = []
    for task in plan["tasks"]:
        replay = json.loads((replay_run / "tasks" / task["main_id"] / "replay/result.json").read_text())
        names = list(replay["c0"]["initial_positions"])
        initial = {n: np.asarray(p, float) for n, p in replay["c0"]["initial_positions"].items()}
        bpath = args.run / "tasks" / task["main_id"] / "branches/result.json"
        if bpath.exists():
            for b in json.loads(bpath.read_text())["branches"]:
                if args.arms and b["arm"] not in args.arms:
                    continue
                d = args.run / "tasks" / task["main_id"] / "branches/branches" / b["event_id"] / ("repeat%d" % b["replicate"]) / b["arm"] / "suffix"
                if not (d / "manifest.json").exists():
                    continue
                rows = list(records(d))
                info = classify(rows, names, initial, reg["close_gate_xy_m"], reg["close_gate_dz_min_m"], reg["close_gate_dz_max_m"])
                info.update(kind="fork", main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], arm=b["arm"],
                            timing=b["timing"], replicate=b["replicate"], physical_class=b["physical_class"], target=b["target_name"])
                rows_out.append(info)
        epath = args.run / "tasks" / task["main_id"] / "episodes/result.json"
        if epath.exists():
            for e in json.loads(epath.read_text()).get("episodes", []):
                if args.arms and e["arm"] not in args.arms:
                    continue
                d = args.run / "tasks" / task["main_id"] / "episodes/episodes" / e["arm"] / ("repeat%d" % e["replicate"]) / "suffix"
                if not (d / "manifest.json").exists():
                    continue
                rows = list(records(d))
                info = classify(rows, names, initial, reg["close_gate_xy_m"], reg["close_gate_dz_min_m"], reg["close_gate_dz_max_m"])
                info.update(kind="episode", main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], arm=e["arm"],
                            timing="q0", replicate=e["replicate"], physical_class="", target="")
                rows_out.append(info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    table = defaultdict(Counter)
    for r in rows_out:
        if r["failed"]:
            table["%s|%s|%s" % (r["kind"], r["arm"], r["timing"])][r["stage"]] += 1
    atomic_json(args.out, dict(run=str(args.run), rows=len(rows_out), stages={k: dict(v) for k, v in table.items()}))
    for k, v in sorted(table.items()):
        print("%-40s" % k, dict(sorted(v.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
