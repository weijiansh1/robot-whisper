"""Freeze a state-disjoint paired MPC pilot before any new intervention outcome."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from adaptive_control import PARAMETERS, REFERENCE
from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from fixed_recovery_control import MODULE, METHODS, STALL, TriggerMonitor, PROTOCOL as TRIGGER_PROTOCOL
from recovery_mpc import PROTOCOL, ARMS, SETTINGS


def prepare(args):
    verify_frozen_alarm()
    model_path = args.model.resolve()
    model = json.loads(model_path.read_text())
    if model["status"] != "completed" or model["protocol"] != PROTOCOL or model["settings"] != SETTINGS:
        raise ValueError("Identified dynamics model mismatch")
    if digest(HERE/"identify_recovery_dynamics.py") != model["identifier_sha256"]:
        raise ValueError("Identification code changed")
    for path, expected in model["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Identification evidence changed: "+path)
    audit_path = HERE/"design/experiment_long_batch1_audit_20260908.json"
    alarm_dir = HERE/"design/experiment_long_batch1_alarm_comparison_20260908"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Native audit required")
    with (alarm_dir/"first_alarms.csv").open(newline="") as stream:
        alarms = {r["main_id"]: r for r in csv.DictReader(stream)}
    excluded = set(model["development_main_ids"])
    groups = defaultdict(list)
    for parent in audit["tasks"]:
        if parent["main_id"] not in excluded:
            groups[parent["benchmark"], parent["category"]].append(parent)
    selected = []
    for key in sorted(groups):
        if len(groups[key]) < 2:
            raise ValueError("Insufficient unused mains in stratum")
        selected.extend(sorted(groups[key], key=lambda r: stable_id(PROTOCOL, "cohort", r["main_id"]))[:2])
    tasks, bound = [], 0
    for parent in selected:
        directory = Path(parent["directory"])
        original = json.loads((directory/"result.json").read_text())
        monitor = TriggerMonitor()
        for row in records(directory/"main"):
            monitor.update(row[PROBS_KEY], row["proprio"][:3])
        first = dict(monitor.first)
        for method in METHODS[:4]:
            if first[method] != int(alarms[parent["main_id"]][method]):
                raise ValueError("Frozen alarm mismatch")
        first["random_time"] = int(np.random.default_rng(int(stable_id(TRIGGER_PROTOCOL,
            parent["main_id"], "random_time")[:8], 16)).integers(8, 40))
        positions = defaultdict(list)
        for method, q in first.items():
            if 0 <= q < parent["main_queries"]-1:
                positions[q+1].append(method)
        events = [dict(event_id=stable_id(PROTOCOL, parent["main_id"], q), start_query=q,
            alarm_query=q-1, methods=positions[q], deployable=True) for q in sorted(positions)]
        task = {k: parent[k] for k in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
        maximum = parent["main_queries"]+sum(len(ARMS)*(52-e["start_query"]) for e in events)
        task.update(parent_directory=str(directory), parent_queries=parent["main_queries"],
            parent_commit_sha256=digest(directory/"main_complete.json"),
            parent_manifest_sha256=digest(directory/"main/manifest.json"), noise_seed=original["seed"],
            init_index=original["init_index"], first_alarm=first["knn20"], first_alarms=first,
            base_task=original["variant"]["base_task"], native_success=original["success"], events=events,
            maximum_queries=maximum,
            max_output_bytes=maximum*180*1024+(len(events)+2)*2*1024**2+520*6*1024)
        tasks.append(task)
        bound += task["max_output_bytes"]
    sources = [Path(__file__).resolve(), model_path, audit_path, alarm_dir/"first_alarms.csv",
        alarm_dir/"verification.json", alarm_dir/"contract.json", PARAMETERS, REFERENCE]
    sources.extend(HERE/name for name in ("recovery_mpc.py", "collect_mpc_recovery.py", "run_mpc_recovery.py",
        "test_recovery_mpc.py", "fixed_recovery_control.py", "collect_fixed_recovery.py"))
    plan = dict(protocol=PROTOCOL, model="long", methods=list(METHODS), arms=list(ARMS), module=MODULE,
        behavior_trigger=STALL, tasks=tasks, allowed_gpus=[0,1,2,3], render_gpus=[0,3],
        replicas_per_gpu=8, workers_per_gpu=8, batch_size=1, hidden_capture=False, threshold_fitting=False,
        dynamics_model=str(model_path), dynamics_sha256=digest(model_path), mpc_settings=SETTINGS,
        selection="fixed hash, two per benchmark/category, exclude all 36 identification development mains; no alarm/outcome filtering",
        evaluation_scope="state-disjoint from motion identification, historical cohort and alarms previously explored; not globally blind",
        excluded_main_ids=sorted(excluded),
        timing="first trigger q, execute native q, intervene before q+1; no effective trigger retains native outcome",
        random_trigger="unchanged fixed-recovery random schedule: one uniform query in [8,39]; termination censors deployment",
        random_stream="original policy RNG indexed by query; exact C0 environment RNG by absolute action step; unchanged deterministic extension beyond native terminal",
        recovery="once, same 8 lift plus 8 retreat steps and last gripper sign; total520 budget; fixed P, constrained MPC and disturbance-compensated MPC; hold and native controls",
        estimand="controller comparison within identical trigger; trigger comparison within fixed controller; rescue, harm, full policy success, recurrence and cost",
        secondary_metrics="phase target tracking; first-step prediction residual; same-target LQR energy change (diagnostic, not a stability guarantee); optimizer residual and fallback",
        new_main_coverage=0, parent_audit=str(audit_path), frozen_parameters_sha256=PARAMETERS_SHA256,
        source_sha256={str(p): digest(p) for p in sources}, maximum_output_bytes=bound,
        storage_quota_gib=6, disk_floor_gib=8)
    if args.output.exists() or shutil.disk_usage(HERE).free-bound < 8*1024**3:
        raise ValueError("Plan exists or insufficient disk headroom")
    atomic_json(args.output, plan)
    print(json.dumps(dict(parents=len(tasks), events=sum(len(t["events"]) for t in tasks),
        suffixes=len(ARMS)*sum(len(t["events"]) for t in tasks),
        native_successes=sum(t["native_success"] for t in tasks), maximum_gib=bound/1024**3,
        maximum_queries=sum(t["maximum_queries"] for t in tasks))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
