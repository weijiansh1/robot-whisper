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


def scheduled_jobs(plan, inventory, output):
    jobs = []
    for task in plan["tasks"]:
        parent_id = task["main_id"]
        base = dict(inventory[task["variant_id"]], main_id=parent_id,
                    noise_seed=int(task["noise_seed"]), init_index=int(task["init_index"]))
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
    if (plan["protocol"] != PROTOCOL or plan["contract"] != CONTRACT or model != "long" or
            plan["allowed_gpus"] != list(GPUS) or plan["render_gpus"] != list(RENDER_GPUS) or
            plan["frozen_parameters_sha256"] != PARAMETERS_SHA256 or plan["arms_registry"] != ARMS or
            plan["controller"] != CONTROLLER):
        raise ValueError("Repair plan contract changed")
    if not set(plan["arms"]) <= set(ARMS) or "new_noise" not in plan["arms"]:
        raise ValueError("Arms must be registered and include the new_noise baseline")
    for source, expected in plan["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError("Repair input changed: " + source)
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
        if task["events"] != events_for(task["main_id"], task["first_alarm"], task["parent_queries"], task["failed"]):
            raise ValueError("Repair trigger positions changed")
    return plan
