#!/usr/bin/env python3
"""Goal-predicate timeline for v3 shared-control rows (forks and episodes) inside the LIBERO env."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import records
from evaluate_repair_end_states import build_env, evaluate, region_center


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    replay_run = Path(plan["replay_run"])
    out = []
    for task in plan["tasks"]:
        if task["benchmark"] != args.benchmark or not task["failed"]:
            continue
        replay = json.loads((replay_run / "tasks" / task["main_id"] / "replay/result.json").read_text())
        original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
        env = build_env(replay["variant"], original, task["init_index"], args.render_gpu, 520 + 11 + 300)
        inner = env.env
        goals = [list(g) for g in inner.parsed_problem["goal_state"]]
        jobs = []
        bpath = args.run / "tasks" / task["main_id"] / "branches/result.json"
        if bpath.exists():
            for b in json.loads(bpath.read_text())["branches"]:
                if b["arm"] in args.arms:
                    jobs.append(("fork", b["arm"], b["timing"], b["replicate"], b["success"],
                                 args.run / "tasks" / task["main_id"] / "branches/branches" / b["event_id"] / ("repeat%d" % b["replicate"]) / b["arm"] / "suffix"))
        epath = args.run / "tasks" / task["main_id"] / "episodes/result.json"
        if epath.exists():
            for e in json.loads(epath.read_text()).get("episodes", []):
                if e["arm"] in args.arms:
                    jobs.append(("episode", e["arm"], "q0", e["replicate"], e["success"],
                                 args.run / "tasks" / task["main_id"] / "episodes/episodes" / e["arm"] / ("repeat%d" % e["replicate"]) / "suffix"))
        for kind, arm, timing, replicate, success, directory in jobs:
            if not (directory / "manifest.json").exists():
                continue
            rows = list(records(directory))
            timeline = [evaluate(env, r["sim_after"]) for r in rows]
            last_target = str(rows[-1]["target_name"].astype(str))
            tg = next((i for i, g in enumerate(goals) if last_target in g[1:]), None)
            ever = next((i for i, t in enumerate(timeline) if tg is not None and t[tg]), None)
            satisfied_end = sum(timeline[-1])
            body = inner.obj_body_id.get(last_target)
            obj = np.asarray(inner.sim.data.body_xpos[body], float) if body is not None else None
            region = region_center(inner, goals[tg][2]) if tg is not None else None
            offset = (obj - region).tolist() if (region is not None and obj is not None) else [None] * 3
            ap_end = float(rows[-1]["proprio"][6]) - float(rows[-1]["proprio"][7])
            out.append(dict(main_id=task["main_id"], base_task=task["base_task"], kind=kind, arm=arm, timing=timing, replicate=replicate,
                            success=success, last_target=last_target, goals=len(goals), satisfied_end=satisfied_end,
                            last_goal_ever_query=ever, last_goal_end=timeline[-1][tg] if tg is not None else None,
                            offset_x=offset[0], offset_y=offset[1], offset_z=offset[2], aperture_end=ap_end, queries=len(rows)))
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(json.dumps(dict(benchmark=args.benchmark, rows=len(out))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--render-gpu", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=["shared_full", "shared_full_z10"])
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
