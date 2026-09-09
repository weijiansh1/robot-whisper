#!/usr/bin/env python3
"""Independently re-execute native, v8, and v8.2 selected actions in MuJoCo."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from collection_storage import atomic_json, digest, load_snapshot, records


def verify(args):
    from libero.libero.envs import OffScreenRenderEnv
    from collect_preflight_worker import restore, verify_egl_device
    from fixed_recovery_control import set_environment_rng

    verify_egl_device(0)
    plan = json.loads((args.run / "plan.json").read_text())
    task = next(t for t in plan["tasks"] if t["main_id"] == args.main_id)
    original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
    replay = args.run / "replays" / args.main_id
    event = json.loads((replay / "result.json").read_text())["events"][0]
    location = replay / "events" / event["event_id"] / "snapshot"
    saved = load_snapshot(location)
    tape = list(records(replay / "environment_rng"))
    outcomes, first_rows, traces = [], {}, {}
    for arm in ("native", "iid_v8", "guided_v8", "iid_v82", "guided_v82"):
        path = args.run / "events" / event["event_id"] / arm
        rows = list(records(path / "suffix"))
        traces[arm] = rows
        first_rows[arm] = rows[0]
        random.seed(original["seed"])
        np.random.seed(original["seed"])
        env = OffScreenRenderEnv(bddl_file_name=original["bddl_path"], camera_heights=224,
            camera_widths=224, render_gpu_device_id=0, horizon=531)
        try:
            env.seed(original["seed"])
            env.reset()
            restore(env, saved, np.random.default_rng())
            changed, dispatched, maximum = 0, 0, 0.
            for row in rows:
                np.testing.assert_array_equal(np.asarray(env.get_sim_state()), row["sim_before"])
                count, step = int(row["executed_action_count"]), int(row["action_steps_before"])
                if int(row["candidate_count"]) == 16:
                    pool = list(records(path / "pools" / str(int(row["relative_query"])) / "candidates"))
                    chosen = pool[int(row["candidate_id"])]
                    np.testing.assert_array_equal(row["actions"], chosen["actions"])
                    changed += int(not np.array_equal(row["actions"][:count], pool[0]["actions"][:count]))
                for offset, action in enumerate(row["actions"][:count]):
                    set_environment_rng(tape, args.main_id, step+offset)
                    env.step(np.asarray(action).tolist())
                    dispatched += 1
                    if bool(env.check_success()) and offset+1 != count:
                        raise ValueError("Recorded suffix executed beyond success")
                actual = np.asarray(env.get_sim_state())
                maximum = max(maximum, float(np.max(np.abs(actual-row["sim_after"]))))
                np.testing.assert_array_equal(actual, row["sim_after"])
                if bool(env.check_success()) != bool(row["success"]):
                    raise ValueError("Re-simulated success differs")
            outcomes.append(dict(arm=arm, queries=len(rows), env_step_calls=dispatched,
                changed_executed_chunks=changed, maximum_state_error=maximum,
                final_success=bool(env.check_success()), suffix_manifest_sha256=digest(path / "suffix/manifest.json")))
        finally:
            env.close()
    comparisons = []
    baseline = first_rows["native"]
    for arm in ("iid_v8", "guided_v8", "iid_v82", "guided_v82"):
        row = first_rows[arm]
        np.testing.assert_array_equal(row["sim_before"], baseline["sim_before"])
        np.testing.assert_array_equal(row["input_sha256"], baseline["input_sha256"])
        count = min(int(row["executed_action_count"]), int(baseline["executed_action_count"]))
        comparisons.append(dict(arm=arm, same_initial_state=True, same_observation=True,
            query=int(row["query"]), executed_steps=count,
            first_command_rms_6d=float(np.sqrt(np.mean((row["actions"][:count, :6]-baseline["actions"][:count, :6])**2))),
            first_state_l2=float(np.linalg.norm(row["sim_after"]-baseline["sim_after"])),
            first_state_max_abs=float(np.max(np.abs(row["sim_after"]-baseline["sim_after"])))))
    first_changes = []
    for arm in ("iid_v8", "guided_v8", "iid_v82", "guided_v82"):
        for row, baseline in zip(traces[arm], traces["native"]):
            count = min(int(row["executed_action_count"]), int(baseline["executed_action_count"]))
            if np.array_equal(row["actions"][:count], baseline["actions"][:count]):
                continue
            np.testing.assert_array_equal(row["sim_before"], baseline["sim_before"])
            np.testing.assert_array_equal(row["input_sha256"], baseline["input_sha256"])
            delta = row["actions"][:count, :6].astype(float)-baseline["actions"][:count, :6].astype(float)
            first_changes.append(dict(arm=arm, query=int(row["query"]), same_before=True,
                same_observation=True, compared_action_steps=count,
                command_rms_6d=float(np.sqrt(np.mean(delta**2))),
                native_executed_steps=int(baseline["executed_action_count"]),
                selected_executed_steps=int(row["executed_action_count"]),
                state_l2=float(np.linalg.norm(row["sim_after"]-baseline["sim_after"])),
                state_max_abs=float(np.max(np.abs(row["sim_after"]-baseline["sim_after"])))))
            break
    result = dict(status="passed", run=str(args.run), main_id=args.main_id, benchmark=task["benchmark"],
        verification="Independent MuJoCo re-execution of every recorded action in five complete suffixes; no policy calls",
        snapshot_manifest_sha256=digest(location / "manifest.json"), actual_policy_queries=0,
        physical_render_gpu=0, gpu6_used=False, arms=outcomes, first_chunk_comparisons=comparisons,
        first_changed_chunk_comparisons=first_changes, verifier_sha256=digest(__file__))
    atomic_json(args.output, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--main-id")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        verify(args)
        return
    from benchmarks.run_benchmarks import BASE, environment, ROOTS
    plan = json.loads((args.run / "plan.json").read_text())
    tasks = plan["tasks"] if args.all else [next(t for t in plan["tasks"] if t["main_id"] == args.main_id)]
    verified = []
    for task in tasks:
        if task["benchmark"] == "native_long":
            from native_long_runtime import environment as native_environment, ROOT
            env, root = native_environment(0, initialize=True), ROOT
        else:
            env, root = environment(task["benchmark"], 0, "egl"), ROOTS[task["benchmark"]]
        target = args.output.with_suffix("") / (task["main_id"]+".json") if args.all else args.output
        subprocess.run([str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()),
            "--worker", "--run", str(args.run.resolve()), "--main-id", task["main_id"],
            "--output", str(target.resolve())], env=env, cwd=root, check=True)
        verified.append(json.loads(target.read_text()))
    if args.all:
        result = dict(status="passed", run=str(args.run.resolve()), parents=len(verified),
            suffixes=sum(len(row["arms"]) for row in verified),
            env_step_calls=sum(arm["env_step_calls"] for row in verified for arm in row["arms"]),
            maximum_state_error=max((arm["maximum_state_error"] for row in verified for arm in row["arms"]), default=0.),
            actual_policy_queries=0, physical_render_gpu=0, gpu6_used=False, cases=verified,
            verifier_sha256=digest(__file__), plan_sha256=digest(args.run / "plan.json"))
        atomic_json(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
