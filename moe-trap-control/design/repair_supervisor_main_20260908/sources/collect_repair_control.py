#!/usr/bin/env python3
"""Exact native replay with three fork snapshots, then REPAIR-then-VLA branches per arm.

Derived from collect_adaptive_control.py.  Differences: the replay evaluates the physical
grasp-verification rule online and persists an `early` snapshot when it first fires; every
snapshot carries the physical class of the fork state; branches may run a train-free REPAIR
phase through env.step before handing control back to the policy; the branch window is a
fixed 300 environment steps from the fork, with the original-cap endpoint recorded as well.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

import numpy as np

from repair_control import (ARMS, CONTRACT, CONTRACT_V2, CONTROLLER, GPUS, RENDER_GPUS, PROTOCOL, PROTOCOL_V2,
                            HORIZON_STEPS, SUPERVISOR_ARMS, WINDOW_STEPS, RouteRisk, early_event, noise_for, seed_for)
from repair_controller import (CLOSE, CLOSE_APERTURE, DEPARTURE_M, NEAR_M, OPEN, STILL_M, CarryGuard, ContactDescent,
                               HandState, PhantomGuard, ProportionalRetract, aperture, eef_position, goal_region,
                               object_top_offset, physical_class, place_point, place_waypoints, resolve_target,
                               retract_chunk, tilt_degrees)
from collect_preflight_worker import (component_digests, input_digest, restore, snapshot,
                                      valid_response, verify_egl_device)
from collection_routes import ALL_FIELDS, CAPTURE_KEY, PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records, save_snapshot

GRASP_NEAR_M = 0.05
IN_HAND_LIFT_M = 0.01


def _listify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _listify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _json_safe(value):
    """JSON without NaN: non-finite floats become null, numpy scalars/arrays become Python values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def repair_plan(arm, physics, eef, main_id, replicate):
    """Pure planning: returns (kind, controller_or_none, target_or_none) for one arm at one fork state."""
    spec = ARMS[arm] if arm in ARMS else SUPERVISOR_ARMS[arm]
    if spec["repair"] == "none":
        return "none", None, None
    if spec["repair"] == "open":
        return "open", None, np.asarray(eef, np.float64).copy()
    if spec["target"] == "history":
        if physics.get("history_pose") is not None:
            target = np.asarray(physics["history_pose"], np.float64)
        else:
            target = np.asarray(eef, np.float64) + np.array([0.0, 0.0, CONTROLLER["above_m"]])
    else:
        if physics.get("target_position") is None:
            return "unavailable", None, None
        position = np.asarray(physics["target_position"], np.float64).copy()
        if spec.get("xy_noise_m"):
            rng = np.random.default_rng(seed_for(main_id, replicate, 0, "policy", 1))
            position[:2] += rng.normal(0.0, spec["xy_noise_m"], 2)
        target = position + np.array([0.0, 0.0, CONTROLLER["above_m"]])
    controller = ProportionalRetract(target, CONTROLLER["gain"], CONTROLLER["unit_metres"], CONTROLLER["tolerance_m"],
                                     CONTROLLER["settle_steps"], CONTROLLER["max_chunks"], gripper=OPEN)
    return spec["repair"], controller, target


class RepairSession:
    def __init__(self, args, cache):
        from benchmarks.run_benchmarks import ROOTS, variants, load_suite
        from libero.libero import get_libero_path
        import libero.libero
        from libero.libero.envs import OffScreenRenderEnv
        from himoe_libero_bridge.client import PolicyClient
        from himoe_libero_bridge.preprocess import build_policy_observation
        from himoe_libero_bridge.suites import get_suite

        self.args, self.cache, self.task = args, cache, args.control["parent"]
        self.parent = Path(self.task["parent_directory"])
        self.original = json.loads((self.parent / "result.json").read_text())
        original = self.original
        if (args.gpu not in GPUS or args.render_gpu not in RENDER_GPUS or
                os.environ.get("CUDA_VISIBLE_DEVICES") != "" or ROOTS[args.benchmark] not in Path(libero.libero.__file__).parents):
            raise ValueError("Repair GPU/benchmark isolation")
        if original["status"] != "completed" or not original["main_complete"] or original["main_intervention"]:
            raise ValueError("Uncommitted or intervened parent")
        for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(self.parent / filename) != self.task[key]:
                raise ValueError("Native parent changed")
        if args.exact_noise_fastpath and args.benchmark == "plus" and "noise_fastpath" not in cache:
            from collection_noise import install
            cache["noise_fastpath"] = install()
        if "variants" not in cache:
            cache["variants"] = {row["variant_id"]: row for row in variants(args.benchmark)}
            cache["risk"] = RouteRisk()
        self.row, self.risk = cache["variants"][args.variant], cache["risk"]
        if (args.model != "long" or self.row != original["variant"] or args.main_id != original["main_id"] or
                args.seed != original["seed"] or args.init_index != original["init_index"]):
            raise ValueError("Repair parent identity")
        suite = load_suite(self.row["registry"], cache.setdefault("suites", {}))
        task = suite.get_task(self.row["registry_index"])
        self.initial = np.asarray(suite.get_task_init_states(self.row["registry_index"]))
        if (task.name != self.row["task_name"] or hashlib.sha256(np.ascontiguousarray(
                self.initial[args.init_index]).tobytes()).hexdigest() != original["initial_state_sha256"]):
            raise ValueError("Changed initial state")
        if digest(original["bddl_path"]) != original["bddl_sha256"]:
            raise ValueError("Changed BDDL")
        self.bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env_class, self.build_observation = OffScreenRenderEnv, build_policy_observation
        self.prompt, self.horizon = original["prompt"], self.row["horizon_steps"]
        if self.horizon != HORIZON_STEPS:
            raise ValueError("Repair protocol assumes the 520-step Long horizon")
        self.rng, self.env = np.random.default_rng(args.seed), None
        self.client = cache.get("client")
        if self.client is None:
            self.client = PolicyClient(port=args.port, connect_timeout=30, inference_timeout=180)
            cache["client"] = self.client
        metadata = self.client.metadata
        if metadata.get("bundle_physical_gpu") != args.gpu or metadata.get("checkpoint_sha256") != get_suite(args.model).weights_sha256:
            raise ValueError("Wrong repair model endpoint")
        for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                    "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
            if metadata[key] != original["model_metadata"][key]:
                raise ValueError("Model identity differs: " + key)
        args.output.mkdir(parents=True, exist_ok=False)
        v2 = args.control.get("supervisor") is not None
        self.report = dict(status="running", protocol=PROTOCOL_V2 if v2 else PROTOCOL, contract=CONTRACT_V2 if v2 else CONTRACT,
            controller=CONTROLLER, supervisor=args.control.get("supervisor"),
            main_id=args.main_id, job_kind=args.control["kind"], variant=self.row, parent_directory=str(self.parent),
            parent_commit_sha256=self.task["parent_commit_sha256"], parent_manifest_sha256=self.task["parent_manifest_sha256"],
            model=args.model, model_metadata=metadata, gpu=args.gpu, render_gpu=args.render_gpu,
            egl_device_uuid=verify_egl_device(args.render_gpu), seed=args.seed, init_index=args.init_index,
            hidden_capture=False, main_intervention=False, main_complete=True, reused_main=True,
            queries=0, action_steps=original["action_steps"], success=original["success"],
            first_alarm_query=original["first_alarm_query"], first_knn_alarm=self.task["first_alarm"],
            analysis_role=self.task["analysis_role"], failed=self.task["failed"], c0=None, events=[], branches=[],
            invalid_pair=False, actual_model_queries=0, candidate_queries=0, reused_candidate_queries=0,
            environment_pid=os.getpid(), worker_job_index=cache["jobs"])
        self.save()

    def save(self):
        atomic_json(self.args.output / "result.json", self.report)

    def new_env(self, extended=False):
        self.close_env()
        args = self.args
        lock_path = Path(tempfile.gettempdir()) / ("himoe-collection-egl-%d-%d.lock" % (os.getuid(), args.render_gpu))
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            random.seed(args.seed)
            np.random.seed(args.seed)
            horizon = self.horizon + 11 + (WINDOW_STEPS if extended else 0)
            self.env = self.env_class(bddl_file_name=str(self.bddl), camera_heights=224,
                camera_widths=224, render_gpu_device_id=args.render_gpu, horizon=horizon)
            self.env.seed(args.seed)
            self.env.reset()
            self.env.set_init_state(self.initial[args.init_index])

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def query(self, obs, noise):
        request = self.build_observation(obs, self.prompt)
        request.update({"flow/noise": noise, "routing/capture": True, CAPTURE_KEY: True})
        tick = time.monotonic()
        response = self.client.infer(request)
        duration = time.monotonic() - tick
        self.report["actual_model_queries"] += 1
        valid_response(response)
        if response["flow/noise_sha256"] != hashlib.sha256(noise.tobytes()).hexdigest():
            raise ValueError("Repair noise identity")
        return request, response, duration

    def advance(self, obs, actions, limit, steps, replicate=None, observer=None):
        tick, count, success = time.monotonic(), 0, False
        for action in actions[:limit]:
            if replicate is not None:
                seed = seed_for(self.args.main_id, replicate, steps + count, "environment")
                random.seed(seed)
                np.random.seed(seed)
            obs, _, done, _ = self.env.step(action.tolist())
            count += 1
            success = bool(self.env.check_success())
            if observer is not None:
                observer(obs)
            if success:
                break
            if done:
                raise ValueError("Premature termination inside the branch window")
        return obs, count, success, time.monotonic() - tick

    @staticmethod
    def record(request, response, q, steps, count, before, after, success, alarm, inference, simulation):
        row = {field: np.asarray(response[field]) for field in ALL_FIELDS}
        row.update(query=np.int32(q), action_steps_before=np.int32(steps), executed_action_count=np.int16(count),
            actions=response["actions"], noise=request["flow/noise"], proprio=np.asarray(request["observation/state"]).copy(),
            input_sha256=np.asarray(input_digest(request), dtype="S64"), input_component_sha256=component_digests(request),
            sim_before=before, sim_after=after, success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
            alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], np.float32),
            inference_seconds=np.float64(inference), environment_seconds=np.float64(simulation))
        return row

    # ------------------------------------------------------------------ physics helpers
    def movable_positions(self):
        inner = self.env.env
        out = {}
        for group in inner.parsed_problem["objects"].values():
            for name in group:
                out[name] = np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]], np.float64).copy()
        return out

    def fork_physics(self, obs, target, initial_positions, proprio_history, closure, q, steps):
        eef, ap = eef_position(obs), aperture(obs)
        history_pose = None
        if closure is not None:
            for past_q, past in reversed(proprio_history[:closure["query"]]):
                if past[6] - past[7] >= CLOSE_APERTURE:
                    history_pose = past[:3].astype(np.float64).tolist()
                    break
        physics = dict(query=int(q), action_steps_before=int(steps), eef=eef.tolist(), aperture=ap,
                       target_name=target["name"], unsatisfied=target["unsatisfied"],
                       articulated_pending=target["articulated_pending"], history_pose=history_pose,
                       closure=None if closure is None else _listify(closure))
        if target["name"] is None:
            physics.update(target_position=None, target_quaternion=None, initial_target_position=None,
                           physical_class="articulated_pending" if target["articulated_pending"] else "complete",
                           grasped=False, tilt_deg=None, table_z=None)
            return physics
        position, quaternion = target["position"], target["quaternion"]
        initial = initial_positions.get(target["name"], position)
        tilt = tilt_degrees(quaternion)
        grasped = bool(ap < CLOSE_APERTURE and np.linalg.norm(eef - position) < GRASP_NEAR_M
                       and position[2] > initial[2] + IN_HAND_LIFT_M)
        table_z = float(initial[2])
        physics.update(target_position=position.tolist(), target_quaternion=quaternion.tolist(),
                       initial_target_position=np.asarray(initial).tolist(), tilt_deg=tilt, table_z=table_z,
                       grasped=grasped, physical_class=physical_class(position, initial, eef, table_z, tilt, grasped),
                       eef_to_target_m=float(np.linalg.norm(eef - position)),
                       top_offset_m=float(object_top_offset(self.env, target["name"])))
        return physics

    # ------------------------------------------------------------------ replay with fork snapshots
    def replay(self):
        self.new_env(extended=False)
        obs = restore(self.env, load_snapshot(self.parent / "preflight_q000"), self.rng)
        monitor, first = self.risk.monitor(), -1
        self.report["events"] = copy.deepcopy(self.task["events"])
        targets = {event["start_query"]: event for event in self.report["events"]}
        c0 = dict(status="running", start_query=0, compared_queries=0, starts_after_main_complete=True,
            parent_main_id=self.args.main_id, parent_commit_sha256=self.task["parent_commit_sha256"], final_success=None)
        self.report["c0"] = c0
        initial_positions = self.movable_positions()
        proprio_history, physics_rows = [], []
        closure, previous_ap, early = None, None, None
        writer = ShardWriter(self.args.output / "c0")
        try:
            for expected in records(self.parent / "main"):
                q, steps = int(expected["query"]), int(expected["action_steps_before"])
                target = resolve_target(self.env)
                ap, eef = aperture(obs), eef_position(obs)
                if target["name"] is not None and previous_ap is not None:
                    near = float(np.linalg.norm(eef - target["position"])) < NEAR_M
                    if closure is None and ap < CLOSE_APERTURE and previous_ap >= CLOSE_APERTURE and near:
                        closure = dict(query=q, eef=eef.copy(), object=target["position"].copy(), name=target["name"])
                    if closure is not None and early is None and ap < CLOSE_APERTURE and target["name"] == closure["name"]:
                        departed = float(np.linalg.norm(eef - closure["eef"])) >= DEPARTURE_M
                        still = float(np.linalg.norm(target["position"] - closure["object"])) < STILL_M
                        if departed and still:
                            early = q
                            if q in targets:
                                targets[q]["early_coincident"] = True
                            else:
                                event = early_event(self.args.main_id, q)
                                self.report["events"].append(event)
                                targets[q] = event
                if closure is not None and ap >= CLOSE_APERTURE:
                    closure = None
                previous_ap = ap
                if q in targets:
                    location = self.args.output / "events" / targets[q]["event_id"] / "snapshot"
                    save_snapshot(location, snapshot(self.env, obs, self.rng, q, steps, self.prompt))
                    physics = self.fork_physics(obs, target, initial_positions, proprio_history, closure, q, steps)
                    atomic_json(location.parent / "physics.json", _json_safe(physics))
                    targets[q].update(snapshot_manifest_sha256=digest(location / "manifest.json"), action_steps_before=steps,
                                      physical_class=physics["physical_class"], target_name=physics["target_name"],
                                      physics_sha256=digest(location.parent / "physics.json"))
                request, response, inference = self.query(obs, self.rng.standard_normal((10, 24)).astype(np.float32))
                proprio_history.append((q, np.asarray(request["observation/state"], np.float64).copy()))
                alarm = monitor.update(response[PROBS_KEY])
                _, score = self.risk.current(monitor)
                if first < 0 and score > self.risk.threshold:
                    first = q
                before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                obs, count, success, simulation = self.advance(obs, response["actions"], int(expected["executed_action_count"]), steps)
                after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                actual = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                for field in expected:
                    if field not in ("inference_seconds", "environment_seconds"):
                        np.testing.assert_array_equal(actual[field], expected[field], err_msg="C0 %d: %s" % (q, field))
                actual["knn_score"] = np.float32(score)
                writer.append(actual)
                physics_rows.append(dict(query=q, action_steps_before=steps, aperture=ap, eef=eef.tolist(),
                    target_name=target["name"], target_position=None if target["position"] is None else target["position"].tolist(),
                    unsatisfied=len(target["unsatisfied"]), knn_score=float(score),
                    freeze_score=float(alarm["freeze_score"]), acceleration_score=float(alarm["acceleration_score"]),
                    periodicity_score=float(alarm["periodicity_score"]), v7_alarm=bool(alarm["alarm"]),
                    closure_query=None if closure is None else int(closure["query"])))
                c0.update(compared_queries=q + 1, final_success=success)
                if (q + 1) % 8 == 0:
                    self.save()
            c0["shards"] = writer.close()
        finally:
            writer.close()
        if (c0["compared_queries"] != self.original["queries"] or c0["final_success"] != self.original["success"] or
                first != self.task["first_alarm"]):
            raise ValueError("Incomplete C0 or online kNN trigger mismatch")
        c0.update(status="passed", online_knn_first=first, frozen_threshold=self.risk.threshold,
                  early_query=early, initial_positions=_listify(initial_positions))
        atomic_json(self.args.output / "c0/branch.json", c0)
        atomic_json(self.args.output / "c0_physics.json", _json_safe(physics_rows))
        for event in self.report["events"]:
            atomic_json(self.args.output / "events" / event["event_id"] / "event.json", event)
        self.close_env()

    # ------------------------------------------------------------------ branches
    def repair_rows(self, writer, index, phase_name, actions, before, after, eef_before, obs_after, count, success, target, distance, reason):
        row = dict(query=np.int32(index), phase=np.asarray(phase_name, dtype="S16"), actions=np.asarray(actions, np.float32),
            executed_action_count=np.int16(count), sim_before=before, sim_after=after,
            eef_before=np.asarray(eef_before, np.float64), eef_after=eef_position(obs_after),
            aperture_after=np.float64(aperture(obs_after)), target=np.asarray(target, np.float64),
            distance_after=np.float64(distance), success=np.bool_(success), reason=np.asarray(reason or "", dtype="S16"))
        writer.append(row)

    def run_repair(self, arm, physics, obs, steps, replicate, directory, branch):
        kind, controller, target = repair_plan(arm, physics, eef_position(obs), self.args.main_id, replicate)
        branch.update(repair_kind=kind, repair_target=None if target is None else np.asarray(target).tolist())
        if kind in ("none", "unavailable"):
            branch.update(repair_steps=0, repair_reason=kind, repair_chunks=0)
            return obs, steps, False
        writer = ShardWriter(directory / "repair", block_size=4)
        index, success = 0, False
        try:
            def run_phase(name, ctrl, gripper):
                nonlocal obs, steps, index, success
                phase = ctrl.phase
                while not phase.done and not phase.exhausted and not success:
                    eef0 = eef_position(obs)
                    chunk = ctrl.next_chunk(eef0)
                    chunk[:, 6] = gripper
                    before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    obs, count, success, _ = self.advance(obs, chunk, 10, steps, replicate,
                                                          observer=lambda o: phase.observe(eef_position(o)))
                    after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    self.repair_rows(writer, index, name, chunk, before, after, eef0, obs, count, success,
                                     phase.target, phase.history[-1] if phase.history else float("nan"), phase.reason)
                    steps += count
                    index += 1
                return phase.reason

            if kind == "open":
                for _ in range(CONTROLLER["open_chunks"]):
                    eef0 = eef_position(obs)
                    chunk = retract_chunk(eef0, eef0, 0.0, CONTROLLER["unit_metres"], gripper=OPEN)
                    before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    obs, count, success, _ = self.advance(obs, chunk, 10, steps, replicate)
                    after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    self.repair_rows(writer, index, "open", chunk, before, after, eef0, obs, count, success, eef0, 0.0, "open")
                    steps += count
                    index += 1
                reason = "open"
            else:
                reason = run_phase("retract", controller, OPEN)
                branch["retract_reason"] = reason
                if kind == "regrasp" and not success:
                    name = physics["target_name"]
                    inner = self.env.env
                    position = np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]], np.float64).copy()
                    z_before = float(position[2])
                    descend_target = position + np.array([0.0, 0.0, physics["top_offset_m"] + 0.005])
                    descend = ProportionalRetract(descend_target, CONTROLLER["gain"], CONTROLLER["unit_metres"], 0.015,
                                                  CONTROLLER["settle_steps"], CONTROLLER["descend_max_chunks"], gripper=OPEN)
                    branch["descend_reason"] = run_phase("descend", descend, OPEN)
                    for _ in range(CONTROLLER["grasp_close_chunks"]):
                        if success:
                            break
                        eef0 = eef_position(obs)
                        chunk = retract_chunk(eef0, eef0, 0.0, CONTROLLER["unit_metres"], gripper=CLOSE)
                        before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        obs, count, success, _ = self.advance(obs, chunk, 10, steps, replicate)
                        after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        self.repair_rows(writer, index, "close", chunk, before, after, eef0, obs, count, success, eef0, 0.0, "close")
                        steps += count
                        index += 1
                    lift_target = eef_position(obs) + np.array([0.0, 0.0, CONTROLLER["lift_m"]])
                    lift = ProportionalRetract(lift_target, CONTROLLER["gain"], CONTROLLER["unit_metres"], CONTROLLER["tolerance_m"],
                                               CONTROLLER["settle_steps"], CONTROLLER["lift_max_chunks"], gripper=CLOSE)
                    branch["lift_reason"] = run_phase("lift", lift, CLOSE)
                    z_after = float(inner.sim.data.body_xpos[inner.obj_body_id[name]][2])
                    branch.update(object_z_before_grasp=z_before, object_z_after_lift=z_after,
                                  lifted=bool(z_after - z_before >= 0.02))
                    reason = "regrasp"
            branch["shards_repair"] = writer.close()
        finally:
            writer.close()
        branch.update(repair_steps=int(steps - branch["fork_action_steps"]), repair_reason=reason, repair_chunks=index,
                      repair_success=bool(success))
        return obs, steps, success

    # ------------------------------------------------------------------ supervisor v2
    def _phase(self, writer, index, name, ctrl, gripper, obs, steps, replicate, observer=None, limit=None, target_name=None):
        """Run one proportional phase to completion; returns (obs, steps, index, success, reason)."""
        phase, success = ctrl.phase, False
        while not phase.done and not phase.exhausted and not success and (limit is None or limit(steps) > 0):
            eef0 = eef_position(obs)
            chunk = ctrl.next_chunk(eef0)
            chunk[:, 6] = gripper

            def watch(o, phase=phase):
                phase.observe(eef_position(o), self._target_state(target_name)[0] if target_name else None)
                if observer is not None:
                    observer(o)
            before = np.asarray(self.env.get_sim_state(), np.float64).copy()
            obs, count, success, _ = self.advance(obs, chunk, 10 if limit is None else min(10, limit(steps)), steps, replicate, observer=watch)
            after = np.asarray(self.env.get_sim_state(), np.float64).copy()
            self.repair_rows(writer, index, name, chunk, before, after, eef0, obs, count, success,
                             phase.target, phase.history[-1] if phase.history else float("nan"), phase.reason)
            steps += count
            index += 1
        return obs, steps, index, success, phase.reason

    def _still_chunk(self, writer, index, name, gripper, obs, steps, replicate, limit=None):
        eef0 = eef_position(obs)
        chunk = retract_chunk(eef0, eef0, 0.0, CONTROLLER["unit_metres"], gripper=gripper)
        before = np.asarray(self.env.get_sim_state(), np.float64).copy()
        obs, count, success, _ = self.advance(obs, chunk, 10 if limit is None else min(10, limit(steps)), steps, replicate)
        after = np.asarray(self.env.get_sim_state(), np.float64).copy()
        self.repair_rows(writer, index, name, chunk, before, after, eef0, obs, count, success, eef0, 0.0, name)
        return obs, steps + count, index + 1, success

    def _target_state(self, name):
        inner = self.env.env
        position = np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]], np.float64).copy()
        unsatisfied = sum(not inner._eval_predicate(list(g)) for g in inner.parsed_problem["goal_state"])
        return position, int(unsatisfied)

    def run_supervised(self, arm, physics, obs, steps, replicate, directory, branch, q0, monitor, writer):
        """First stage identical to retract_above_target, then VLA chunks under physical guards G2/G3."""
        spec = SUPERVISOR_ARMS[arm]
        sup = self.args.control["supervisor"]
        fork_steps = int(steps)
        name = physics.get("target_name")
        repair_writer = ShardWriter(directory / "repair", block_size=4)
        interventions, rindex, success = [], 0, False
        branch.update(interventions=interventions, first_extra_intervention_query=None, guards=list(spec["guards"]))
        try:
            if name is None:
                branch.update(repair_kind="unavailable", repair_target=None)
            else:
                target0 = np.asarray(physics["target_position"], np.float64)
                initial0 = np.asarray(physics["initial_target_position"], np.float64)
                rest_z = float(min(target0[2], initial0[2]))
                lifted_at_fork = bool(target0[2] - initial0[2] >= sup["lifted_m"])
                if "G0" in spec["guards"] and lifted_at_fork:
                    interventions.append(dict(guard="G0", kind="veto", query_index=0, start_step=0, steps=0, reason="lifted_at_fork"))
                    branch.update(repair_kind="veto", repair_target=None, twin="new_noise")
                else:
                    kind, controller, target = repair_plan("retract_above_target", physics, eef_position(obs), self.args.main_id, replicate)
                    branch.update(repair_kind=kind, repair_target=np.asarray(target).tolist(), twin="retract_above_target")
                    obs, steps, rindex, success, reason = self._phase(repair_writer, rindex, "retract", controller, OPEN, obs, steps, replicate)
                    interventions.append(dict(guard="G1", kind="retract", query_index=0, start_step=0,
                                              steps=int(steps - fork_steps), reason=reason))
                    branch["retract_reason"] = reason
            branch["handback_step"] = int(steps - fork_steps)
            branch["action_steps"] = int(steps - fork_steps)
            if success:
                branch.update(success=True, success_step=branch["action_steps"])
            phantom = carry = hand = None
            if name is not None:
                hand = HandState(rest_z, sup["lifted_m"])
                if "G2" in spec["guards"]:
                    phantom = PhantomGuard(sup["phantom_departure_m"], sup["phantom_hold_steps"], sup["phantom_moved_m"])
                if "G3" in spec["guards"]:
                    carry = CarryGuard(sup["carry_patience_steps"])
            fired = {}

            def remaining(current_steps):
                return WINDOW_STEPS - int(current_steps - fork_steps)

            def observe(o):
                if hand is not None:
                    position, unsatisfied = self._target_state(name)
                    eef, ap, step = eef_position(o), aperture(o), self._guard_step
                    in_hand = hand.update(eef, ap, position)
                    if not fired:
                        verdict = phantom.observe(eef, ap, position, step, in_hand) if phantom is not None else None
                        if verdict is None and carry is not None:
                            verdict = carry.observe(unsatisfied, step, in_hand)
                        if verdict is not None:
                            fired["guard"] = "G2" if verdict.startswith("phantom") else "G3"
                            fired["reason"] = verdict
                self._guard_step += 1

            self._guard_step = int(steps)
            while not branch["success"] and branch["action_steps"] < WINDOW_STEPS:
                index = branch["queries"]
                q = q0 + index
                request, response, inference = self.query(obs, noise_for(self.args.main_id, replicate, index, 0))
                branch["deployment_model_queries"] += 1
                alarm = monitor.update(response[PROBS_KEY])
                vector, score = self.risk.current(monitor)
                before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                limit = min(10, WINDOW_STEPS - branch["action_steps"])
                self._guard_step = int(steps)
                obs, count, success, simulation = self.advance(obs, response["actions"], limit, steps, replicate, observer=observe)
                after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                if not np.isfinite(after).all():
                    raise ValueError("Non-finite branch physics")
                row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                row.update(relative_query=np.int32(index), candidate_id=np.int16(0),
                    policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy", 0)),
                    environment_first_seed=np.uint32(seed_for(self.args.main_id, replicate, steps, "environment")),
                    environment_last_seed=np.uint32(seed_for(self.args.main_id, replicate, steps + count - 1, "environment")),
                    requested_action_count=np.int16(limit), knn_score=np.float32(score), knn_vector=vector,
                    steps_since_fork=np.int32(steps - fork_steps), intervention_count=np.int16(len(interventions)))
                writer.append(row)
                steps += count
                branch.update(queries=index + 1, action_steps=steps - fork_steps, success=success)
                if success:
                    branch["success_step"] = branch["action_steps"]
                    break
                if fired and len(interventions) < sup["max_interventions"] and branch["action_steps"] < WINDOW_STEPS:
                    guard, reason = fired["guard"], fired["reason"]
                    start = int(steps - fork_steps)
                    if branch["first_extra_intervention_query"] is None:
                        branch["first_extra_intervention_query"] = index + 1
                    record = dict(guard=guard, kind=None, query_index=index + 1, start_step=start, steps=0, reason=reason)
                    interventions.append(record)
                    if guard == "G2":
                        position, _ = self._target_state(name)
                        repeats = sum(i["guard"] == "G2" for i in interventions) - 1
                        target = position + np.array([0.0, 0.0, CONTROLLER["above_m"] + repeats * sup["phantom_escalation_m"]])
                        ctrl = ProportionalRetract(target, CONTROLLER["gain"], CONTROLLER["unit_metres"], CONTROLLER["tolerance_m"],
                                                   CONTROLLER["settle_steps"], CONTROLLER["max_chunks"], gripper=OPEN)
                        obs, steps, rindex, success, phase_reason = self._phase(repair_writer, rindex, "retract", ctrl, OPEN, obs, steps, replicate, limit=remaining)
                        record.update(kind="retract", target=target.tolist(), phase_reason=phase_reason)
                    else:
                        region = goal_region(self.env, name)
                        if region is None:
                            record.update(kind="no_region")
                            carry = None
                        else:
                            position, _ = self._target_state(name)
                            others = [np.asarray(p, np.float64) for n, p in self.movable_positions().items() if n != name]
                            point, occupied = place_point(region, others)
                            way = place_waypoints(eef_position(obs), position, region, physics.get("top_offset_m", 0.02),
                                                  sup["place_height_m"], sup["place_clearance_m"], sup["place_rise_m"], point=point)
                            record.update(kind="place", region=region["name"], region_centre=region["centre"].tolist(),
                                          region_half=region["half"].tolist(), place_point=point.tolist(), occupied=occupied,
                                          waypoints={k: np.asarray(v).tolist() for k, v in way.items()})
                            reasons = {}
                            for phase_name, waypoint, chunks in (("rise", way["rise"], sup["place_rise_chunks"]),
                                                                 ("transfer", way["transfer"], sup["place_transfer_chunks"])):
                                if success:
                                    break
                                ctrl = ProportionalRetract(waypoint, CONTROLLER["gain"], CONTROLLER["unit_metres"], sup["place_tolerance_m"],
                                                           CONTROLLER["settle_steps"], chunks, gripper=CLOSE)
                                obs, steps, rindex, success, reasons[phase_name] = self._phase(repair_writer, rindex, phase_name, ctrl, CLOSE, obs, steps, replicate, limit=remaining)
                            if not success:
                                floor_target = way["lower"] - np.array([0.0, 0.0, sup["place_clearance_m"] + physics.get("top_offset_m", 0.02)])
                                ctrl = ContactDescent(floor_target, CONTROLLER["gain"] / 2, CONTROLLER["unit_metres"], CONTROLLER["settle_steps"], sup["place_lower_chunks"])
                                obs, steps, rindex, success, reasons["lower"] = self._phase(repair_writer, rindex, "lower", ctrl, CLOSE, obs, steps, replicate, limit=remaining, target_name=name)
                            if not success:
                                obs, steps, rindex, success = self._still_chunk(repair_writer, rindex, "release", OPEN, obs, steps, replicate, limit=remaining)
                            if not success:
                                retreat = eef_position(obs) + np.array([0.0, 0.0, sup["place_rise_m"]])
                                ctrl = ProportionalRetract(retreat, CONTROLLER["gain"], CONTROLLER["unit_metres"], sup["place_tolerance_m"],
                                                           CONTROLLER["settle_steps"], sup["place_retreat_chunks"], gripper=OPEN)
                                obs, steps, rindex, success, reasons["retreat"] = self._phase(repair_writer, rindex, "retreat", ctrl, OPEN, obs, steps, replicate, limit=remaining)
                            record["phase_reasons"] = reasons
                    record["steps"] = int(steps - fork_steps - start)
                    if phantom is not None:
                        phantom.reset()
                    if carry is not None:
                        carry.reset()
                    fired = {}
                    branch.update(action_steps=steps - fork_steps, success=success)
                    if success:
                        branch["success_step"] = branch["action_steps"]
                elif fired:
                    fired = {}
                if branch["queries"] % 8 == 0:
                    self.save()
            branch["shards_repair"] = repair_writer.close()
        finally:
            repair_writer.close()
        branch.update(repair_steps=int(sum(i["steps"] for i in interventions)), repair_chunks=rindex,
                      repair_reason=interventions[0]["reason"] if interventions else "none",
                      repair_success=bool(interventions and branch["success"] and branch["success_step"] <= interventions[0]["steps"]))
        return obs, steps

    def branches(self):
        spec = self.args.control
        replay_dir = Path(spec["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if replay["status"] != "completed" or replay["c0"]["status"] != "passed" or replay["main_id"] != self.args.main_id:
            raise ValueError("Branches require a completed C0 for this parent")
        if replay["parent_commit_sha256"] != self.task["parent_commit_sha256"]:
            raise ValueError("Replay parent differs")
        events = replay["events"]
        self.report["events"] = copy.deepcopy(events)
        self.report["c0"] = dict(status="passed", compared_queries=0, replay_directory=str(replay_dir),
            replay_result_sha256=digest(replay_dir / "result.json"), replay_branch_sha256=digest(replay_dir / "c0/branch.json"))
        main_rows = list(records(self.parent / "main"))
        timings = spec.get("timings")
        for event in events:
            if timings and event["timing"] not in timings:
                continue
            saved = load_snapshot(replay_dir / "events" / event["event_id"] / "snapshot")
            physics = json.loads((replay_dir / "events" / event["event_id"] / "physics.json").read_text())
            if digest(replay_dir / "events" / event["event_id"] / "physics.json") != event["physics_sha256"]:
                raise ValueError("Fork physics changed")
            q0 = event["start_query"]
            prefix_monitor = self.risk.monitor()
            for row in main_rows:
                if int(row["query"]) >= q0:
                    break
                prefix_monitor.update(row[PROBS_KEY])
            for replicate in range(int(spec["replicates"])):
                for arm in spec["arms"]:
                    directory = self.args.output / "branches" / event["event_id"] / ("repeat%d" % replicate) / arm
                    self.new_env(extended=True)
                    obs = restore(self.env, saved, self.rng)
                    monitor = copy.deepcopy(prefix_monitor)
                    fork_steps = int(saved["action_steps"])
                    branch = dict(status="running", arm=arm, control=ARMS[arm] if arm in ARMS else SUPERVISOR_ARMS[arm],
                        replicate=replicate, event_id=event["event_id"],
                        timing=event["timing"], deployable=event["deployable"], start_query=q0,
                        physical_class=physics["physical_class"], target_name=physics["target_name"],
                        parent_main_id=self.args.main_id, parent_commit_sha256=self.task["parent_commit_sha256"],
                        snapshot_manifest_sha256=event["snapshot_manifest_sha256"], fork_action_steps=fork_steps,
                        original_remaining_steps=HORIZON_STEPS - fork_steps, window_steps=WINDOW_STEPS,
                        queries=0, action_steps=0, success=False, success_step=None, success_within_original=False,
                        deployment_model_queries=0, handback_step=None)
                    self.report["branches"].append(branch)
                    directory.mkdir(parents=True, exist_ok=False)
                    if arm in SUPERVISOR_ARMS:
                        writer = ShardWriter(directory / "suffix")
                        try:
                            self.run_supervised(arm, physics, obs, fork_steps, replicate, directory, branch, q0, monitor, writer)
                            branch["shards"] = writer.close()
                        finally:
                            writer.close()
                        branch["success_within_original"] = bool(branch["success"] and branch["success_step"] <= branch["original_remaining_steps"])
                        branch["status"] = "completed"
                        atomic_json(directory / "branch.json", branch)
                        self.close_env()
                        self.save()
                        continue
                    obs, steps, success = self.run_repair(arm, physics, obs, fork_steps, replicate, directory, branch)
                    branch["action_steps"] = steps - fork_steps
                    branch["handback_step"] = branch["action_steps"]
                    if success:
                        branch.update(success=True, success_step=branch["action_steps"])
                    writer = ShardWriter(directory / "suffix")
                    try:
                        while not branch["success"] and branch["action_steps"] < WINDOW_STEPS:
                            index = branch["queries"]
                            q = q0 + index
                            request, response, inference = self.query(obs, noise_for(self.args.main_id, replicate, index, 0))
                            branch["deployment_model_queries"] += 1
                            alarm = monitor.update(response[PROBS_KEY])
                            vector, score = self.risk.current(monitor)
                            before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                            limit = min(10, WINDOW_STEPS - branch["action_steps"])
                            obs, count, success, simulation = self.advance(obs, response["actions"], limit, steps, replicate)
                            after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                            if not np.isfinite(after).all():
                                raise ValueError("Non-finite branch physics")
                            row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                            row.update(relative_query=np.int32(index), candidate_id=np.int16(0),
                                policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy", 0)),
                                environment_first_seed=np.uint32(seed_for(self.args.main_id, replicate, steps, "environment")),
                                environment_last_seed=np.uint32(seed_for(self.args.main_id, replicate, steps + count - 1, "environment")),
                                requested_action_count=np.int16(limit), knn_score=np.float32(score), knn_vector=vector,
                                steps_since_fork=np.int32(steps - fork_steps))
                            writer.append(row)
                            steps += count
                            branch.update(queries=index + 1, action_steps=steps - fork_steps, success=success)
                            if success:
                                branch["success_step"] = branch["action_steps"]
                            if branch["queries"] % 8 == 0:
                                self.save()
                        branch["shards"] = writer.close()
                    finally:
                        writer.close()
                    branch["success_within_original"] = bool(branch["success"] and branch["success_step"] <= branch["original_remaining_steps"])
                    branch["status"] = "completed"
                    atomic_json(directory / "branch.json", branch)
                    self.close_env()
                    self.save()


def run(args, cache):
    started = time.monotonic()
    session = RepairSession(args, cache)
    try:
        if args.control["kind"] == "replay":
            session.replay()
        elif args.control["kind"] == "branches":
            session.branches()
        else:
            raise ValueError("Unknown repair job")
        session.report["status"] = "completed"
    except BaseException:
        session.report.update(status="failed", invalid_pair=True, error=traceback.format_exc())
        raise
    finally:
        session.close_env()
        session.report["elapsed_seconds"] = time.monotonic() - started
        session.save()


def worker_loop(args):
    cache = {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            task.variant, task.seed, task.output = message["variant"], message["seed"], Path(message["output"])
            task.main_id, task.init_index, task.control = message["main_id"], int(message["init_index"]), message["control"]
            cache["jobs"] = cache.get("jobs", 0) + 1
            result = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                result["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"] = 1
            print("COLLECTION_RESULT " + json.dumps(result), flush=True)
    finally:
        if cache.get("client") is not None:
            cache["client"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--model", choices=("long",), required=True)
    parser.add_argument("--gpu", type=int, choices=GPUS, required=True)
    parser.add_argument("--render-gpu", type=int, choices=RENDER_GPUS, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-loop", action="store_true", required=True)
    parser.add_argument("--paired-branches", action="store_true", required=True)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    worker_loop(parser.parse_args())
