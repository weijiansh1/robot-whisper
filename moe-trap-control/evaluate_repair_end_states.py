#!/usr/bin/env python3
"""Evaluate goal predicates along stored repair branches by re-instantiating the LIBERO env (no model, no rendering use)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from collection_storage import records

HERE = Path(__file__).resolve().parent


def build_env(row, original, init_index, render_gpu, horizon):
    from benchmarks.run_benchmarks import load_suite
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = load_suite(row["registry"], {})
    task = suite.get_task(row["registry_index"])
    initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    random.seed(original["seed"])
    np.random.seed(original["seed"])
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
                             render_gpu_device_id=render_gpu, horizon=horizon)
    env.seed(original["seed"])
    env.reset()
    env.set_init_state(initial[init_index])
    return env


def region_center(inner, name):
    try:
        return np.asarray(inner.sim.data.site_xpos[inner.sim.model.site_name2id(name)], float).copy()
    except Exception:
        try:
            return np.asarray(inner.sim.data.body_xpos[inner.sim.model.body_name2id(name)], float).copy()
        except Exception:
            return None


def evaluate(env, state):
    inner = env.env
    inner.sim.set_state_from_flattened(np.asarray(state, np.float64))
    inner.sim.forward()
    return [bool(inner._eval_predicate(list(goal))) for goal in inner.parsed_problem["goal_state"]]


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    out = []
    for task in plan["tasks"]:
        if task["benchmark"] != args.benchmark or not task["failed"]:
            continue
        task_dir = args.run / "tasks" / task["main_id"]
        replay = json.loads((task_dir / "replay/result.json").read_text())
        original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
        env = build_env(replay["variant"], original, task["init_index"], args.render_gpu, 520 + 11 + 300)
        inner = env.env
        goals = [list(g) for g in inner.parsed_problem["goal_state"]]
        result = json.loads((task_dir / "branches/result.json").read_text())
        for branch in result["branches"]:
            if branch["arm"] not in args.arms:
                continue
            directory = task_dir / "branches/branches" / branch["event_id"] / ("repeat%d" % branch["replicate"]) / branch["arm"]
            rows = list(records(directory / "suffix"))
            if not rows:
                continue
            target = branch["target_name"]
            target_goal = next((i for i, g in enumerate(goals) if target in g[1:]), None)
            timeline = [evaluate(env, r["sim_after"]) for r in rows]
            end = timeline[-1]
            ever = next((i for i, t in enumerate(timeline) if target_goal is not None and t[target_goal]), None)
            all_ever = next((i for i, t in enumerate(timeline) if all(t)), None)
            body = inner.obj_body_id[target]
            obj = np.asarray(inner.sim.data.body_xpos[body], float)
            region = region_center(inner, goals[target_goal][2]) if target_goal is not None else None
            offset = (obj - region).tolist() if region is not None else [None] * 3
            out.append(dict(main_id=task["main_id"], base_task=task["base_task"], arm=branch["arm"], timing=branch["timing"],
                event_id=branch["event_id"], replicate=branch["replicate"], success=branch["success"], target=target,
                goals=len(goals), target_goal_end=end[target_goal] if target_goal is not None else None,
                target_goal_first_query=ever, all_goals_first_query=all_ever, others_end=sum(end) - (end[target_goal] if target_goal is not None else 0),
                unsatisfied_end=len(goals) - sum(end), offset_x=offset[0], offset_y=offset[1], offset_z=offset[2],
                final_aperture=float(rows[-1]["proprio"][6]) - float(rows[-1]["proprio"][7]), queries=len(rows)))
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(json.dumps(dict(benchmark=args.benchmark, rows=len(out), out=str(args.out))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--render-gpu", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=["retract_above_target", "retract_above_target_noisy"])
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
