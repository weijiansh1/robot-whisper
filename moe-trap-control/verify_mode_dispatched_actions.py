#!/usr/bin/env python3
"""Re-execute every mode experiment suffix in MuJoCo without policy calls."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import random
import subprocess

import numpy as np

from collection_storage import atomic_json, digest, load_snapshot, records
from mode_control import ARMS, PROTOCOL


def verify(args):
    from libero.libero.envs import OffScreenRenderEnv
    from collect_preflight_worker import restore, verify_egl_device
    from fixed_recovery_control import set_environment_rng

    verify_egl_device(0)
    plan = json.loads((args.run / "plan.json").read_text())
    if plan["protocol"] != PROTOCOL or plan["arms"] != list(ARMS):
        raise ValueError("Wrong mode experiment")
    task = next(t for t in plan["tasks"] if t["main_id"] == args.main_id)
    original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
    replay = args.run / "replays" / args.main_id
    event = json.loads((replay / "result.json").read_text())["events"][0]
    snapshot = replay / "events" / event["event_id"] / "snapshot"
    saved, tape = load_snapshot(snapshot), list(records(replay / "environment_rng"))
    outcomes = []
    for arm in ARMS:
        path = args.run / "events" / event["event_id"] / arm
        branch = json.loads((path / "branch.json").read_text())
        physical = list(records(path / "physical")) if (path / "physical").exists() else []
        rows = list(records(path / "suffix"))
        random.seed(original["seed"])
        np.random.seed(original["seed"])
        env = OffScreenRenderEnv(bddl_file_name=original["bddl_path"], camera_heights=224,
            camera_widths=224, render_gpu_device_id=0, horizon=531)
        try:
            env.seed(original["seed"])
            env.reset()
            obs = restore(env, saved, np.random.default_rng())
            dispatched, changed, maximum = 0, 0, 0.
            for row in physical:
                np.testing.assert_array_equal(np.asarray(env.get_sim_state()), row["sim_before"])
                np.testing.assert_array_equal(obs["robot0_eef_pos"], row["eef_before"])
                set_environment_rng(tape, args.main_id, int(row["action_step"]))
                obs, _, _, _ = env.step(row["action"].tolist())
                np.testing.assert_array_equal(np.asarray(env.get_sim_state()), row["sim_after"])
                np.testing.assert_array_equal(obs["robot0_eef_pos"], row["eef_after"])
                if bool(env.check_success()) != bool(row["success"]):
                    raise ValueError("Physical recovery success differs")
                dispatched += 1
            for row in rows:
                np.testing.assert_array_equal(np.asarray(env.get_sim_state()), row["sim_before"])
                count, step = int(row["executed_action_count"]), int(row["action_steps_before"])
                if int(row["candidate_count"]) == 16:
                    pool = list(records(path / "pools" / str(int(row["relative_query"])) / "candidates"))
                    np.testing.assert_array_equal(row["actions"], pool[int(row["candidate_id"])]["actions"])
                    changed += int(not np.array_equal(row["actions"][:count], pool[0]["actions"][:count]))
                for offset, action in enumerate(row["actions"][:count]):
                    set_environment_rng(tape, args.main_id, step+offset)
                    env.step(action.tolist())
                    dispatched += 1
                    if bool(env.check_success()) and offset+1 != count:
                        raise ValueError("Recorded suffix executed beyond success")
                actual = np.asarray(env.get_sim_state())
                maximum = max(maximum, float(np.max(np.abs(actual-row["sim_after"]))))
                np.testing.assert_array_equal(actual, row["sim_after"])
                if bool(env.check_success()) != bool(row["success"]):
                    raise ValueError("Re-simulated suffix success differs")
            if bool(env.check_success()) != branch["success"] or dispatched != branch["action_steps"]:
                raise ValueError("Full suffix outcome/action budget differs")
            outcomes.append(dict(arm=arm, queries=len(rows), physical_steps=len(physical), env_step_calls=dispatched,
                changed_executed_chunks=changed, maximum_state_error=maximum, final_success=bool(env.check_success()),
                suffix_manifest_sha256=digest(path / "suffix/manifest.json")))
        finally:
            env.close()
    result = dict(status="passed", run=str(args.run), main_id=args.main_id, arms=outcomes,
        actual_policy_queries=0, physical_render_gpu=0, gpu6_used=False,
        snapshot_manifest_sha256=digest(snapshot / "manifest.json"), verifier_sha256=digest(__file__))
    atomic_json(args.output, result)
    print(json.dumps(dict(main_id=args.main_id, status="passed", suffixes=len(outcomes),
        env_step_calls=sum(row["env_step_calls"] for row in outcomes))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-id")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        verify(args)
        return
    from benchmarks.run_benchmarks import BASE
    from native_long_runtime import ROOT, environment
    plan = json.loads((args.run / "plan.json").read_text())
    env = environment(0, initialize=True)
    tasks = [t for t in plan["tasks"] if args.main_id is None or t["main_id"] == args.main_id]
    if not tasks:
        raise ValueError("No matching original parent")
    directory = args.output.with_suffix("")
    directory.mkdir(parents=True, exist_ok=False)

    def launch(task):
        target = directory / (task["main_id"]+".json")
        with (directory / (task["main_id"]+".log")).open("w") as log:
            subprocess.run([str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()),
                "--worker", "--run", str(args.run.resolve()), "--main-id", task["main_id"],
                "--output", str(target.resolve())], env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(target.read_text())
        print(json.dumps(dict(main_id=task["main_id"], status=result["status"])), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(launch, tasks))
    result = dict(status="passed", run=str(args.run.resolve()), parents=len(cases),
        suffixes=sum(len(row["arms"]) for row in cases),
        env_step_calls=sum(arm["env_step_calls"] for row in cases for arm in row["arms"]),
        maximum_state_error=max(arm["maximum_state_error"] for row in cases for arm in row["arms"]),
        actual_policy_queries=0, physical_render_gpu=0, gpu6_used=False, cases=cases,
        verifier_sha256=digest(__file__), plan_sha256=digest(args.run / "plan.json"))
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
