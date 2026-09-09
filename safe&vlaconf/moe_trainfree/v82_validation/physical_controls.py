"""Restore recorded B states and extract outcome-neutral physical events."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HUB = ROOT / "VLA_MUI_HUB"
sys.path.insert(0, str(HUB / "physical-failure-labels"))
from annotate import PhysicsTask, SUITE_TO_BENCHMARK  # noqa: E402

CAPS = {"libero_goal": 300, "libero_long": 520, "libero_object": 280, "libero_spatial": 220}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first_true(values, start=0):
    positions = np.flatnonzero(np.asarray(values)[start:])
    return int(positions[0] + start) if len(positions) else -1


def events_for_episode(meta, goals, grasps, xyz, subjects, expressions):
    result = []
    for si, subject in enumerate(subjects):
        related = [i for i, expr in enumerate(expressions) if len(expr) >= 2 and expr[1] == subject]
        satisfied = goals[:, related].all(axis=1)
        releases = np.flatnonzero(grasps[:-1, si] & ~grasps[1:, si]) + 1
        peak = np.maximum.accumulate(xyz[:, si, 2])
        for q in releases:
            record = dict(meta, query=int(q), subject=subject,
                          kind="release_in_goal" if satisfied[q] else "release_outside_goal",
                          regrasp_q=first_true(grasps[:, si], int(q) + 1),
                          goal_reached_q=first_true(satisfied, int(q)), release_q=int(q))
            result.append(record)
            candidates = [j for j in range(int(q), min(len(goals), int(q) + 3))
                          if not grasps[j, si] and peak[j] - xyz[j, si, 2] >= .04]
            if candidates and not satisfied[candidates[0]]:
                result.append(dict(record, kind="release_height_loss", query=candidates[0]))
    for gi, expression in enumerate(expressions):
        regressions = np.flatnonzero(goals[:-1, gi] & ~goals[1:, gi]) + 1
        for q in regressions:
            result.append(dict(meta, query=int(q), subject=str(expression[1]), kind="goal_regression",
                               regrasp_q=-1, goal_reached_q=first_true(goals[:, gi], int(q)), release_q=-1))
    return result


def inspect_run(job):
    source, rows, output = job
    started = time.perf_counter()
    path = HUB / source
    summary_path = path / "client/summaries.json"
    summaries = {int(x["episode_index"]): x for x in json.loads(summary_path.read_text())}
    name = hashlib.sha256(source.encode()).hexdigest()[:16]
    out = Path(output)
    if (out / (name + ".json")).exists() and (out / (name + ".npz")).exists():
        audit = json.loads((out / (name + ".json")).read_text())
        assert audit["summary_sha256"] == digest(summary_path)
        assert audit["output_sha256"] == digest(out / (name + ".npz"))
        events, episodes = [], []
        with np.load(out / (name + ".npz")) as z:
            subjects = z["subjects"].astype(str).tolist()
            expressions = [json.loads(g) for g in z["goal_expressions"].astype(str)]
            offset = 0
            for row, global_row, n in zip(rows, z["global_rows"], z["lengths"]):
                assert int(row["global_row"]) == global_row and int(row["length"]) == n
                ep = int(row["episode"])
                client = path / "client" / ("episode_%02d.npz" % ep)
                assert audit["episode_sha256"][client.name] == digest(client)
                meta = dict(global_row=int(global_row), source=source, task=row["task"], suite=row["suite"],
                            episode=ep, init_state_id=int(row["init_state_id"]), length=int(n),
                            failure=not bool(summaries[ep]["success"]))
                block = slice(offset, offset + n)
                goals = z["goals"][block]
                events.extend(events_for_episode(meta, goals, z["grasps"][block], z["xyz"][block], subjects, expressions))
                episodes.append(dict(meta, action_steps=int(summaries[ep]["action_steps"]), snapshots=int(n),
                                     subjects=len(subjects), predicates=len(expressions),
                                     first_full_goal_q=first_true(goals.all(axis=1))))
                offset += n
        audit["reused"] = True
        return audit, events, episodes
    first = summaries[int(rows[0]["episode"])]
    suite = rows[0]["suite"]
    init_log = io.StringIO()
    with redirect_stdout(init_log):
        task = PhysicsTask(SUITE_TO_BENCHMARK[suite], int(first["task_id"]),
                           first["task_name"], int(first["seed"]), CAPS[suite])
    event_rows, episode_rows, hashes = [], [], {}
    all_goals, all_grasps, all_contacts, all_xyz = [], [], [], []
    subjects = task.mobile_subjects
    full_goal_count = 0
    try:
        for row in rows:
            ep = int(row["episode"])
            summary = summaries[ep]
            client = path / "client" / ("episode_%02d.npz" % ep)
            with np.load(client, allow_pickle=False) as z:
                states = np.asarray(z["sim_state"], np.float64)
            n = len(states)
            if n != int(row["length"]) or n != int(summary["inference_calls"]):
                raise ValueError("physical source length mismatch")
            if states.shape[1] != 1 + task.core.sim.model.nq + task.core.sim.model.nv:
                raise ValueError("physical state topology mismatch")
            goals = np.zeros((n, len(task.goals)), bool)
            grasps = np.zeros((n, len(subjects)), bool)
            contacts = np.zeros_like(grasps)
            xyz = np.zeros((n, len(subjects), 3), np.float32)
            for q, state in enumerate(states):
                task.core.sim.set_state_from_flattened(state)
                task.core.sim.forward()
                goals[q] = [bool(task.core._eval_predicate(g)) for g in task.goals]
                if bool(task.environment.check_success()) != bool(goals[q].all()):
                    raise ValueError("predicate/API mismatch")
                if goals[q].all():
                    full_goal_count += 1
                    if not bool(summary["success"]):
                        raise ValueError("failed episode has restored full-goal state: %s/%d/q%d" % (source, ep, q))
                for si, subject in enumerate(subjects):
                    model = task.core.get_object(subject)
                    grasps[q, si] = task.core._check_grasp(task.robot.gripper, model)
                    contacts[q, si] = task.core.check_contact(task.robot.gripper, model)
                    xyz[q, si] = task.core.object_states_dict[subject].get_geom_state()["pos"]
            meta = dict(global_row=int(row["global_row"]), source=source, task=row["task"], suite=suite,
                        episode=ep, init_state_id=int(row["init_state_id"]), length=n,
                        failure=not bool(summary["success"]))
            event_rows.extend(events_for_episode(meta, goals, grasps, xyz, subjects, task.goals))
            episode_rows.append(dict(meta, action_steps=int(summary["action_steps"]),
                                     snapshots=n, subjects=len(subjects), predicates=len(task.goals),
                                     first_full_goal_q=first_true(goals.all(axis=1))))
            hashes[client.name] = digest(client)
            all_goals.append(goals)
            all_grasps.append(grasps)
            all_contacts.append(contacts)
            all_xyz.append(xyz)
    finally:
        task.close()
    np.savez_compressed(out / (name + ".npz"), goals=np.concatenate(all_goals),
                        grasps=np.concatenate(all_grasps), contacts=np.concatenate(all_contacts),
                        xyz=np.concatenate(all_xyz), subjects=np.asarray(subjects),
                        goal_expressions=np.asarray([json.dumps(g) for g in task.goals]),
                        global_rows=np.asarray([r["global_row"] for r in episode_rows]),
                        lengths=np.asarray([r["length"] for r in episode_rows]))
    audit = dict(source=source, episodes=len(rows), snapshots=sum(r["length"] for r in episode_rows),
                 success=sum(not r["failure"] for r in episode_rows), failures=sum(r["failure"] for r in episode_rows),
                 summary_sha256=digest(summary_path), episode_sha256=hashes, seconds=time.perf_counter() - started,
                 output=name + ".npz", output_sha256=digest(out / (name + ".npz")),
                 predicate_api_mismatches=0, full_goal_true_at_saved_checkpoint=full_goal_count,
                 initialization_log=init_log.getvalue())
    (out / (name + ".json")).write_text(json.dumps(audit, indent=2) + "\n")
    return audit, event_rows, episode_rows


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=HERE.parent / "results/round9_full_corpus/index.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "runs").mkdir(exist_ok=args.resume)
    with args.index.open() as stream:
        rows = [dict(row, global_row=i) for i, row in enumerate(csv.DictReader(stream))]
    groups = {}
    for row in rows:
        if row["run_id"] == "right-50x8b-20260903":
            groups.setdefault(row["source"], []).append(row)
    jobs = [(source, values, str(args.output / "runs")) for source, values in sorted(groups.items())]
    if args.limit_tasks:
        jobs = jobs[:args.limit_tasks]
    audits, events, episodes = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(inspect_run, job): job[0] for job in jobs}
        for future in as_completed(pending):
            audit, current_events, current_episodes = future.result()
            audits.append(audit)
            events.extend(current_events)
            episodes.extend(current_episodes)
            print("PHYSICS %d/%d %s: %d states, %.1fs" %
                  (len(audits), len(jobs), audit["source"], audit["snapshots"], audit["seconds"]), flush=True)
    events.sort(key=lambda r: (r["global_row"], r["query"], r["kind"], r["subject"]))
    episodes.sort(key=lambda r: r["global_row"])
    write_csv(args.output / "events.csv", events)
    write_csv(args.output / "episodes.csv", episodes)
    manifest = dict(schema="himoe.v82.physics_controls.v1", runs=sorted(audits, key=lambda x: x["source"]),
                    episodes=len(episodes), snapshots=sum(r["snapshots"] for r in episodes), events=len(events),
                    input_index_sha256=digest(args.index), code_sha256=digest(Path(__file__)),
                    physics_helper_sha256=digest(HUB / "physical-failure-labels/annotate.py"),
                    source_rollouts_modified=False, new_rollouts=False, model_loaded=False,
                    snapshot_stride_actions=10, outcome_neutral_event_definition=True)
    (args.output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PHYSICS COMPLETE %d episodes / %d events" % (len(episodes), len(events)), flush=True)


if __name__ == "__main__":
    main()
