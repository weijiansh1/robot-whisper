#!/usr/bin/env python3
"""Post-handback failure-mode diagnostics for one repair run (reads stored sim states only)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, records

ROBOT_QPOS = 9
OBJECT_QPOS = 7
CLOSE_APERTURE = 0.05
LIFT_M = 0.02
MOVED_M = 0.01
NEAR_M = 0.12


def object_index(replay, name):
    names = list(replay["c0"]["initial_positions"])
    return 1 + ROBOT_QPOS + OBJECT_QPOS * names.index(name)


def branch_mode(rows, target_slice, initial):
    """Classify what the policy did with the object after the handback."""
    aperture = np.array([float(r["proprio"][6]) - float(r["proprio"][7]) for r in rows])
    eef = np.array([np.asarray(r["proprio"][:3], float) for r in rows])
    obj = np.array([np.asarray(r["sim_after"][target_slice], float)[:3] for r in rows])
    rise = obj[:, 2] - initial[2]
    moved = np.linalg.norm(obj[:, :2] - initial[:2], axis=1)
    dist = np.linalg.norm(eef - obj, axis=1)
    closed = aperture < CLOSE_APERTURE
    first_close = int(np.argmax(closed)) if closed.any() else None
    lifted = bool((rise >= LIFT_M).any())
    if lifted:
        mode = "lifted_not_placed"
    elif first_close is not None and moved[first_close:].max() < MOVED_M:
        mode = "phantom_regrasp"
    elif first_close is not None:
        mode = "closed_object_pushed"
    elif dist.min() > NEAR_M:
        mode = "never_approached"
    else:
        mode = "approached_never_closed"
    return dict(mode=mode, lifted=lifted, first_close_query=first_close, max_rise_m=float(rise.max()),
                max_moved_m=float(moved.max()), min_dist_m=float(dist.min()), final_dist_m=float(dist[-1]),
                queries=len(rows))


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    out = []
    for task in plan["tasks"]:
        task_dir = args.run / "tasks" / task["main_id"]
        replay = json.loads((task_dir / "replay/result.json").read_text())
        result = json.loads((task_dir / "branches/result.json").read_text())
        for branch in result["branches"]:
            directory = task_dir / "branches/branches" / branch["event_id"] / ("repeat%d" % branch["replicate"]) / branch["arm"]
            rows = list(records(directory / "suffix"))
            if not rows:
                continue
            name = branch["target_name"]
            start = object_index(replay, name)
            initial = np.asarray(json.loads((task_dir / "replay/events" / branch["event_id"] / "physics.json").read_text())["target_position"], float)
            info = branch_mode(rows, slice(start, start + 3), initial)
            if branch["success"]:
                info["mode"] = "rescued"
            info.update(main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], arm=branch["arm"],
                        timing=branch["timing"], physical_class=branch["physical_class"], target=name,
                        success=branch["success"], event_id=branch["event_id"], replicate=branch["replicate"])
            out.append(info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    table = defaultdict(Counter)
    for row in out:
        if row["failed"]:
            table[row["arm"]][row["mode"]] += 1
    by_task = defaultdict(Counter)
    for row in out:
        if row["failed"] and row["arm"] == "retract_above_target":
            by_task[row["base_task"]][row["mode"]] += 1
    report = dict(run=str(args.run), branches=len(out), modes_by_arm={a: dict(c) for a, c in table.items()},
                  retract_above_target_modes_by_task={t: dict(c) for t, c in by_task.items()})
    atomic_json(args.out, report)
    for arm, counter in sorted(table.items()):
        print("%-28s" % arm, dict(sorted(counter.items())))
    print()
    for t, counter in sorted(by_task.items()):
        print("%-90s" % t[:90], dict(sorted(counter.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
