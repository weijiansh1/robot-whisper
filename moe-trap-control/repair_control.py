"""Frozen protocol for the late-alarm repair experiment: GPUs 4/5, six arms, three fork timings.

Nothing here fits a threshold or trains anything.  The mid trigger is the frozen global
Euclidean kNN-20 alarm reused from the adaptive-control experiment; the early trigger is a
physical grasp-verification rule evaluated online during the exact C0 replay; the late
trigger is a fixed query.  Arms differ only in the target x* of the REPAIR controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_storage import digest
from adaptive_control import RouteRisk, REFERENCE, PARAMETERS  # noqa: F401  frozen kNN-20 + v7 monitor

PROTOCOL = "moe_control.repair_recovery.v1"
GPUS = (4, 5)
RENDER_GPUS = (4, 5)
LATE_QUERY = 44
WINDOW_STEPS = 300
HORIZON_STEPS = 520
ARMS = {
    "new_noise": dict(repair="none"),
    "open_only": dict(repair="open"),
    "retract_history": dict(repair="retract", target="history"),
    "retract_above_target": dict(repair="retract", target="above_target"),
    "retract_above_target_noisy": dict(repair="retract", target="above_target", xy_noise_m=0.02),
    "scripted_regrasp": dict(repair="regrasp", target="above_target"),
}
CONTROLLER = dict(gain=0.8, unit_metres=0.05, tolerance_m=0.02, settle_steps=5, max_chunks=6,
                  open_chunks=1, above_m=0.10, lift_m=0.08, grasp_close_chunks=1, descend_max_chunks=4,
                  lift_max_chunks=2)
CONTRACT = dict(
    trigger_mid="frozen global Euclidean kNN-20 first alarm + 1, deployable",
    trigger_early="physical grasp-verification rule evaluated online during C0: closed aperture, "
                  "eef departed >= 0.10 m from the closure pose, target moved < 0.01 m since closure",
    trigger_late="fixed query 44 (80 steps left)",
    repair="saturated proportional task-space controller through env.step only; no model call; one switch per suffix",
    handback="guard: within 0.02 m for 5 consecutive steps or 6 chunks; then native 10-step chunks with the "
             "replicate's new noise stream",
    endpoints="B: success within 300 steps from the fork including repair; A: success within 520 - steps_before",
    internal_readouts="full HB probs stored for every VLA query; the first post-handback query is the competence observer",
    same_noise_anchor="the exact C0 replay is the same-noise continuation through every fork state",
    hidden_capture=False, batch_size=1, threshold_fitting=False, training=False,
)


def event_id(main_id, timing, start):
    return stable_id(PROTOCOL, main_id, timing, int(start))


def events_for(main_id, knn_first, length, failed):
    events = []
    q = int(knn_first)
    if q >= 0 and q + 1 < length:
        events.append(dict(event_id=event_id(main_id, "mid", q + 1), timing="mid", start_query=q + 1,
                           alarm_query=q, deployable=True))
    if failed and LATE_QUERY < length and not any(e["start_query"] == LATE_QUERY for e in events):
        events.append(dict(event_id=event_id(main_id, "late", LATE_QUERY), timing="late", start_query=LATE_QUERY,
                           alarm_query=q, deployable=False))
    return events


def early_event(main_id, start):
    return dict(event_id=event_id(main_id, "early", start), timing="early", start_query=int(start),
                alarm_query=-1, deployable=True)


def seed_for(main_id, replicate, index, stream, candidate=0):
    if stream not in ("policy", "environment") or replicate < 0 or index < 0:
        raise ValueError("Invalid repair random stream")
    return int(stable_id(PROTOCOL, main_id, int(replicate), int(index), stream, int(candidate))[:8], 16)


def noise_for(main_id, replicate, index, candidate=0):
    return np.random.default_rng(seed_for(main_id, replicate, index, "policy", candidate)).standard_normal(
        (10, 24)).astype(np.float32)


BYTES_PER_QUERY = 80 * 1024   # smoke 2026-09-08 measured 74.3 MB for ~1,120 stored queries (~66 KiB each)


def branch_bound(arms, replicates, events=3):
    queries = (WINDOW_STEPS // 10 + 1) * len(arms) * replicates * events
    return queries * BYTES_PER_QUERY + 8 * 1024**2


# ---------------------------------------------------------------------------- supervisor v2
PROTOCOL_V2 = "moe_control.repair_supervisor.v2"
SUPERVISOR_ARMS = {
    "supervisor_regrasp_guard": dict(repair="retract", target="above_target", guards=["G0", "G2"]),
    "supervisor_place_assist": dict(repair="retract", target="above_target", guards=["G0", "G3"]),
    "supervisor_full": dict(repair="retract", target="above_target", guards=["G0", "G2", "G3"]),
}
SUPERVISOR = dict(max_interventions=3, lifted_m=0.02, phantom_departure_m=0.05, phantom_hold_steps=40,
                  phantom_moved_m=0.01, phantom_escalation_m=0.05, carry_patience_steps=120, place_height_m=0.15, place_clearance_m=0.02,
                  place_rise_m=0.08, place_rise_chunks=3, place_transfer_chunks=6, place_lower_chunks=4,
                  place_retreat_chunks=2, place_tolerance_m=0.02)
CONTRACT_V2 = dict(CONTRACT,
    repair="supervisory switching {VLA, RETRACT, PLACE}; physical guards evaluated every env step, switches at chunk "
           "boundaries; at most 3 interventions per suffix; no model call inside a primitive",
    guards="G0 veto: target lifted >= 0.02 m at the fork -> no first-stage repair; in-hand := closed aperture, target lifted "
           ">= 0.02 m above rest, within 0.20 m of the eef and moving with it; G2 phantom: closed aperture, target not in hand "
           "and unmoved since closure, eef departed >= 0.05 m or closure held >= 40 steps -> RETRACT above target, 0.05 m "
           "higher at each repeat; G3 carry: target in hand >= 120 consecutive steps without a goal predicate turning true -> "
           "PLACE: rise, transfer above the goal region, descend until contact, release, retreat; every primitive is cut at "
           "the 300-step window",
    first_stage="identical to retract_above_target of moe_control.repair_recovery.v1 (same fork snapshot, same noise "
                "streams) so every branch is paired with its v1 twin up to the first extra intervention",
    replay="fork snapshots and fork physics reused from the audited v1 run; no C0 replay")


def scheduled_jobs(plan, inventory, output):
    jobs, replay_run = [], plan.get("replay_run")
    for task in plan["tasks"]:
        parent_id = task["main_id"]
        base = dict(inventory[task["variant_id"]], main_id=parent_id,
                    noise_seed=int(task["noise_seed"]), init_index=int(task["init_index"]))
        if replay_run:
            replay_directory = str(Path(replay_run) / "tasks" / parent_id / "replay")
            if plan["arms"]:
                jobs.append(dict(base, job_id=parent_id + "/branches", depends_on=None,
                             sampling=dict(kind="branches", parent=task, arms=plan["arms"], replicates=plan["replicates"],
                                           timings=plan.get("timings"), supervisor=plan.get("supervisor"),
                                               regulator=plan.get("regulator"), replay_directory=replay_directory,
                                               max_output_bytes=branch_bound(plan["arms"], plan["replicates"]))))
            if plan.get("episode_arms"):
                jobs.append(dict(base, job_id=parent_id + "/episodes", depends_on=None,
                                 sampling=dict(kind="episodes", parent=task, arms=plan["episode_arms"],
                                               replicates=plan["replicates"], regulator=plan["regulator"],
                                               max_output_bytes=episode_bound(plan["episode_arms"], plan["replicates"]))))
            continue
        replay_id = parent_id + "/replay"
        jobs.append(dict(base, job_id=replay_id, depends_on=None,
                         sampling=dict(kind="replay", parent=task, max_output_bytes=task["max_output_bytes"])))
        jobs.append(dict(base, job_id=parent_id + "/branches", depends_on=replay_id,
                         sampling=dict(kind="branches", parent=task, arms=plan["arms"], replicates=plan["replicates"],
                                       replay_directory=str(Path(output) / "tasks" / replay_id),
                                       max_output_bytes=branch_bound(plan["arms"], plan["replicates"]))))
    jobs.sort(key=lambda row: (row["depends_on"] is not None, -row["sampling"]["max_output_bytes"], row["job_id"]))
    return jobs


def load_plan(path, model="long"):
    plan = json.loads(Path(path).read_text())
    verify_frozen_alarm()
    v2 = plan["protocol"] in (PROTOCOL_V2, PROTOCOL_V3)
    if plan["protocol"] == PROTOCOL_V2:
        if (plan["contract"] != CONTRACT_V2 or plan["arms_registry"] != SUPERVISOR_ARMS or plan["supervisor"] != SUPERVISOR
                or not set(plan["arms"]) <= set(SUPERVISOR_ARMS) or not plan.get("replay_run")):
            raise ValueError("Supervisor plan contract changed")
    elif plan["protocol"] == PROTOCOL_V3:
        if (plan["contract"] != CONTRACT_V3 or plan["arms_registry"] != REGULATOR_ARMS or plan["regulator"] != REGULATOR
                or plan["episode_arms_registry"] != EPISODE_ARMS or not set(plan["arms"]) <= set(REGULATOR_ARMS)
                or not set(plan["episode_arms"]) <= set(EPISODE_ARMS) or not plan.get("replay_run")):
            raise ValueError("Regulator plan contract changed")
    if v2:
        audit = json.loads(Path(plan["replay_audit"]).read_text())
        if audit["status"] != "passed" or audit["plan_sha256"] != digest(Path(plan["replay_run"]) / "plan.json"):
            raise ValueError("Derived plan needs an audited v1 run")
    elif plan["protocol"] != PROTOCOL or plan["contract"] != CONTRACT or plan["arms_registry"] != ARMS:
        raise ValueError("Repair plan contract changed")
    elif not set(plan["arms"]) <= set(ARMS) or "new_noise" not in plan["arms"]:
        raise ValueError("Arms must be registered and include the new_noise baseline")
    if (model != "long" or plan["allowed_gpus"] != list(GPUS) or plan["render_gpus"] != list(RENDER_GPUS) or
            plan["frozen_parameters_sha256"] != PARAMETERS_SHA256 or plan["controller"] != CONTROLLER):
        raise ValueError("Repair plan contract changed")
    for source, expected in plan["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError("Repair plan source changed: " + source)
    if plan["parameter_fitting"] or plan["training"] or plan["model"] != "long":
        raise ValueError("Repair plan must be train-free")
    if not plan["tasks"] or len({t["main_id"] for t in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Empty/duplicated repair parents")
    for task in plan["tasks"]:
        directory = Path(task["parent_directory"])
        original = json.loads((directory / "result.json").read_text())
        if task["noise_seed"] != original["seed"] or task["init_index"] != original["init_index"]:
            raise ValueError("Native random seed/init changed")
        if task["failed"] != (not original["success"]):
            raise ValueError("Parent outcome label changed")
        for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(directory / filename) != task[key]:
                raise ValueError("Parent commit changed")
        if not v2 and task["events"] != events_for(task["main_id"], task["first_alarm"], task["parent_queries"], task["failed"]):
            raise ValueError("Repair trigger positions changed")
    return plan


# ---------------------------------------------------------------------------- feedback regulator v3
PROTOCOL_V3 = "moe_control.repair_regulator.v3"
REGULATOR = dict(law="shared", alpha=0.7, authority_steps=60, kp_shared=10.0, z_offset_m=0.05, lifted_m=0.02,
                 carry_patience_steps=120, close_gate_xy_m=0.07, close_gate_dz_min_m=-0.03, close_gate_dz_max_m=0.13,
                 close_gate_extent_margin_m=0.02,
                 release_margin_xy_m=0.02, release_margin_z_m=0.10, body_region_radius_m=0.06, body_region_height_m=0.08)
REGULATOR_ARMS = {
    "gate_close": dict(features=["close_gate"]),
    "shared_approach": dict(features=["close_gate", "shared"]),
    "shared_full": dict(features=["close_gate", "shared", "carry", "release_gate"]),
    "shared_full_alpha1": dict(features=["close_gate", "shared", "carry", "release_gate"], alpha=1.0),
}
EPISODE_ARMS = {
    "vla": dict(features=[], engage="never"),
    "shared_full": dict(features=["close_gate", "shared", "carry", "release_gate"], engage="always"),
    "shared_full_alarmed": dict(features=["close_gate", "shared", "carry", "release_gate"], engage="knn_alarm"),
    "shared_full_alpha1": dict(features=["close_gate", "shared", "carry", "release_gate"], alpha=1.0, engage="always"),
}
CONTRACT_V3 = dict(CONTRACT,
    repair="no switching: the VLA runs every chunk; a shared-control law edits each executed env step: "
           "u = (1 - a) u_VLA + a u_servo with a = 0 until the graspable-envelope gate blocks a closure (CLOSE commanded while "
           "neither the target (extent-based envelope) nor any other movable object (strict 0.07 m envelope) sits between "
           "the fingers: physical evidence of a phantom grasp; an already closed empty gripper counts the same way because "
           "the policy keeps commanding CLOSE); then a = 0.7 (alpha1 arms: 1.0) for 60 "
           "steps pulling the end effector above the nearest unsatisfied object; "
           "in hand the policy keeps authority unless the carry stalls 120 steps (pull toward the place point); "
           "OPEN executes only inside the goal region",
    calibration="closure envelope from 22 lifted closures in the v1 replays: horizontal <= 0.063 m, dz 0.00 to 0.10 m; "
                "gate uses max(0.07 m, object bounding radius + 0.02 m) and -0.03 to max(0.13 m, radius + 0.02 m) because "
                "handle grasps (moka pot) close farther from the body origin; a first always-on PI servo (kp 6, ki 0.3, u_max 0.5) was rejected "
                "in the smoke of 2026-09-09 because it edited every step and broke a success parent; gains fixed before the "
                "main run, not fitted to outcomes",
    fork_branches="start at the v1 fork snapshots; twin = new_noise until the first modified env step",
    episodes="full 520-step episodes from the parent's q0 snapshot: vla (replicate 0 must reproduce the parent bitwise), "
             "shared control always on, shared control engaged at the online kNN-20 first alarm",
    replay="fork snapshots reused from the audited v1 run; no C0 replay")


def episode_bound(arms, replicates):
    return (HORIZON_STEPS // 10 + 1) * len(arms) * replicates * BYTES_PER_QUERY + 8 * 1024**2
