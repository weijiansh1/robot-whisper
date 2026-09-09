"""Restore recorded states in the pinned Python 3.8 simulator, without actions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parents[1]
SCHEMA = "himoe.physical_progress.primitives.v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def physics_module(hub):
    path = Path(hub) / "physical-failure-labels/annotate.py"
    spec = importlib.util.spec_from_file_location("progress_physics_source", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_layout(task, layout):
    model = task.core.sim.model
    expected = (model.nq, model.nv, 1 + model.nq + model.nv)
    if tuple(layout[k] for k in ("nq", "nv", "state_dim")) != expected:
        raise ValueError("simulator dimensions differ from collection")
    joints = []
    for name in model.joint_names:
        address = model.get_joint_qpos_addr(name)
        low, high = address if isinstance(address, tuple) else (address, address + 1)
        joints.append((str(name), int(low) + 1, int(high) + 1))
    if joints != [(j["joint"], j["state_lo"], j["state_hi"]) for j in layout["joints"]]:
        raise ValueError("simulator joint order differs from collection")


def extract_episode(task, path, length, physics):
    with np.load(str(path), allow_pickle=False) as archive:
        states = np.asarray(archive["sim_state"], dtype=np.float64)
        policy_states = np.asarray(archive["state"], dtype=np.float64)
        action_shape = archive["actions"].shape
    dimension = 1 + task.core.sim.model.nq + task.core.sim.model.nv
    if states.shape != (length, dimension) or policy_states.shape != (length, 8):
        raise ValueError("episode array shapes do not align")
    if action_shape != (length, 10, 7) or not np.isfinite(states).all():
        raise ValueError("invalid recorded action/state contract")
    goal = np.zeros((length, len(task.goals)), dtype=bool)
    grasp = np.zeros((length, len(task.mobile_subjects)), dtype=bool)
    position = np.zeros((length, len(task.mobile_subjects), 3), dtype=np.float32)
    distance = np.zeros((length, len(task.mobile_subjects)), dtype=np.float32)
    alignment = np.zeros((length, 4), dtype=np.float64)
    max_robot_error = 0.0
    from robosuite.utils.transform_utils import convert_quat

    for q, state in enumerate(states):
        task.core.sim.set_state_from_flattened(state)
        task.core.sim.forward()
        goal[q] = [bool(task.core._eval_predicate(g)) for g in task.goals]
        if bool(task.environment.check_success()) != bool(goal[q].all()):
            raise ValueError("original predicate/check_success mismatch")
        data = task.core.sim.data
        observation = dict(
            robot0_eef_pos=data.site_xpos[task.robot.eef_site_id],
            robot0_eef_quat=convert_quat(data.get_body_xquat(task.robot.robot_model.eef_name), to="xyzw"),
            robot0_gripper_qpos=data.qpos[task.robot._ref_gripper_joint_pos_indexes],
        )
        error = float(np.max(np.abs(physics.policy_state_from_observation(observation) - policy_states[q])))
        max_robot_error = max(max_robot_error, error)
        axis_angle = policy_states[q, 3:6]
        angle = np.linalg.norm(axis_angle)
        logged_quat = np.r_[np.cos(angle/2), np.sin(angle/2)*axis_angle/max(angle, 1e-12)]
        restored_quat = data.get_body_xquat(task.robot.robot_model.eef_name)
        similarity = abs(np.dot(logged_quat, restored_quat)) / (np.linalg.norm(logged_quat)*np.linalg.norm(restored_quat))
        alignment[q] = (error, np.linalg.norm(observation["robot0_eef_pos"]-policy_states[q, :3]),
                        2*np.arccos(np.clip(similarity, 0, 1)),
                        np.max(np.abs(observation["robot0_gripper_qpos"]-policy_states[q, 6:])))
        for k, name in enumerate(task.mobile_subjects):
            obj = task.core.get_object(name)
            position[q, k] = task.core.object_states_dict[name].get_geom_state()["pos"]
            distance[q, k] = task.core._gripper_to_target(task.robot.gripper, obj, return_distance=True)
            grasp[q, k] = bool(task.core._check_grasp(task.robot.gripper, obj))
    return dict(goal=goal, grasp=grasp, position=position, distance=distance, alignment=alignment), max_robot_error


def extract_task(job):
    hub, libero_root, output, rows, input_hash = job
    hub, output = Path(hub), Path(output)
    from himoe_libero_bridge.libero_runtime import _configure_libero
    _configure_libero(Path(libero_root))
    physics = physics_module(hub)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_run"], []).append(row)
    first_run = hub / rows[0]["source_run"]
    first = json.loads((first_run / "client/summaries.json").read_text())[0]
    task = physics.PhysicsTask(physics.SUITE_TO_BENCHMARK[rows[0]["suite"]],
                               int(first["task_id"]), rows[0]["task"], 7,
                               int(rows[0]["max_steps"]))
    audits = []
    try:
        for source, group in sorted(grouped.items()):
            start = time.time()
            run = hub / source
            summaries = json.loads((run / "client/summaries.json").read_text())
            records = {int(r["episode_index"]): r for r in summaries}
            validate_layout(task, json.loads((run / "client/sim_layout.json").read_text()))
            paths = [run / "meta.json", run / "client/summaries.json", run / "client/sim_layout.json"]
            paths.extend(run / "client" / physics.episode_filename(int(row["episode"])) for row in group)
            hashes = {str(p.relative_to(hub)): digest(p) for p in paths}
            key = hashlib.sha256(source.encode()).hexdigest()[:20]
            cache, audit_path = output / (key + ".npz"), output / (key + ".json")
            if cache.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if (audit["source_hashes"] == hashes and audit["input_hash"] == input_hash
                        and audit["extractor_sha256"] == digest(__file__)
                        and audit["physics_source_sha256"] == digest(hub / "physical-failure-labels/annotate.py")
                        and audit["cache_sha256"] == digest(cache)):
                    audits.append(audit)
                    continue
            blocks = {k: [] for k in ("goal", "grasp", "position", "distance", "alignment")}
            episode_rows, queries, errors = [], [], []
            for row in group:
                episode, length = int(row["episode"]), int(row["length"])
                record = records[episode]
                expected = (row["task"], int(row["init_state"]), int(row["seed"]), length,
                            row["success"] == "True", int(row["action_steps"]))
                actual = tuple(record[k] for k in ("task_name", "init_state_id", "flow_noise_seed",
                                                   "inference_calls", "success", "action_steps"))
                if expected != actual or int(record["seed"]) != 7:
                    raise ValueError("source metadata differs from validated episode table")
                arrays, error = extract_episode(task, run / "client" / physics.episode_filename(episode),
                                                length, physics)
                for name, values in arrays.items():
                    blocks[name].append(values)
                episode_rows.append(np.full(length, int(row["episode_row"]), dtype=np.int32))
                queries.append(np.arange(length, dtype=np.int16))
                errors.append(error)
            membership = np.asarray([[len(g) == 3 and g[1] == name for g in task.goals]
                                     for name in task.mobile_subjects], dtype=bool).reshape(
                                         len(task.mobile_subjects), len(task.goals))
            alignment = np.concatenate(blocks["alignment"])
            np.savez_compressed(str(cache), episode_row=np.concatenate(episode_rows), query=np.concatenate(queries),
                                subject_goal=membership, **{k: np.concatenate(v) for k, v in blocks.items()})
            audit = dict(schema=SCHEMA, source_run=source, suite=rows[0]["suite"], task=rows[0]["task"],
                         episodes=len(group), queries=sum(len(v) for v in queries), source_hashes=hashes,
                         input_hash=input_hash, extractor_sha256=digest(__file__),
                         physics_source_sha256=digest(hub / "physical-failure-labels/annotate.py"),
                         cache_file=cache.name, cache_sha256=digest(cache),
                         goals=task.goals, mobile_subjects=task.mobile_subjects,
                         max_robot_state_error=max(errors), predicate_api_mismatches=0,
                         max_eef_position_error_m=float(alignment[:, 1].max()),
                         max_eef_rotation_error_rad=float(alignment[:, 2].max()),
                         max_gripper_error_m=float(alignment[:, 3].max()),
                         robot_alignment_outliers=int(((alignment[:, 1] > 0.005) |
                             (alignment[:, 2] > 0.03) | (alignment[:, 3] > 0.001)).sum()),
                         alignment_thresholds=dict(position_m=0.005, rotation_rad=0.03, gripper_m=0.001),
                         restored_completed_checkpoints=int(np.concatenate(blocks["goal"]).all(axis=1).sum()),
                         layout_verified=True,
                         elapsed_seconds=round(time.time() - start, 2))
            write_json(audit_path, audit)
            print("physics %s/%s: %d episodes, %d checkpoints, %.1fs" %
                  (rows[0]["task"], Path(source).name, len(group), audit["queries"], audit["elapsed_seconds"]), flush=True)
            audits.append(audit)
    finally:
        task.close()
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, default=HERE / "results/episodes.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results_progress/physical")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--episodes-per-run", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_hash = digest(args.episodes)
    expected = json.loads((args.episodes.parent / "artifact_manifest.json").read_text())["episodes.csv"]
    if input_hash != expected:
        raise ValueError("episode index differs from original verified dataset")
    with args.episodes.open() as stream:
        rows = list(csv.DictReader(stream))
    tasks = sorted(set(r["task"] for r in rows))
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    jobs = []
    for name in tasks:
        group = [r for r in rows if r["task"] == name and
                 (not args.episodes_per_run or int(r["episode"]) < args.episodes_per_run)]
        jobs.append((str(args.hub.resolve()), str(args.libero_root.resolve()), str(args.output.resolve()), group, input_hash))
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_task, job) for job in jobs]
        for future in as_completed(futures):
            audits.extend(future.result())
    audits.sort(key=lambda r: r["source_run"])
    write_json(args.output / "manifest.json", dict(schema=SCHEMA, episodes=sum(a["episodes"] for a in audits),
               queries=sum(a["queries"] for a in audits), tasks=len(tasks), runs=audits,
               simulator_restoration_only=True, sampled_or_executed_actions=0,
               partial=bool(args.limit_tasks or args.episodes_per_run)))


if __name__ == "__main__":
    main()
