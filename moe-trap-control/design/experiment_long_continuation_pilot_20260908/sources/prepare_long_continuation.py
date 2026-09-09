#!/usr/bin/env python3
"""Freeze an outcome-independent 12-parent pilot and 108-parent Long continuation."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id
from collection_storage import atomic_json, digest
from continuation_experiment import (ARMS, CANDIDATES, INTERVENTIONS, METHODS, PROTOCOL, REPLICATES,
                                     events_for, load_plan, task_bound)


def run(output):
    audit_path = HERE / "design/experiment_long_batch1_audit_20260908.json"
    alarm_dir = HERE / "design/experiment_long_batch1_alarm_comparison_20260908"
    audit = json.loads(audit_path.read_text())
    with (alarm_dir / "first_alarms.csv").open(newline="") as stream:
        alarms = {r["main_id"]: r for r in csv.DictReader(stream)}
    groups = {}
    for parent in audit["tasks"]:
        directory = Path(parent["directory"])
        result = json.loads((directory / "result.json").read_text())
        main_id = parent["main_id"]
        first = {method: int(alarms[main_id][method]) for method in METHODS}
        events, terminal = events_for(main_id, first, parent["main_queries"])
        task = {key: parent[key] for key in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
        task.update(parent_directory=str(directory), parent_queries=parent["main_queries"],
            parent_commit_sha256=digest(directory / "main_complete.json"),
            parent_manifest_sha256=digest(directory / "main/manifest.json"),
            noise_seed=result["seed"], init_index=result["init_index"], first_alarms=first,
            events=events, terminal_alarms=terminal)
        task["max_output_bytes"] = task_bound(task)
        groups.setdefault((parent["benchmark"], parent["category"]), []).append(task)
    for group in groups.values():
        group.sort(key=lambda t: stable_id(PROTOCOL, "pilot_selection", t["main_id"]))
    pilot = [groups[key][0] for key in sorted(groups)]
    rest = [task for key in sorted(groups) for task in groups[key][1:]]
    rest.sort(key=lambda t: stable_id(PROTOCOL, "dispatch", t["main_id"]))
    sources = [audit_path, alarm_dir / "first_alarms.csv", alarm_dir / "verification.json",
               alarm_dir / "contract.json", HERE / "prepare_long_continuation.py",
               HERE / "continuation_experiment.py", HERE.parent / "himoe-route-capture/route_noise_selector.py"]
    output.mkdir(parents=True, exist_ok=False)
    for stage, tasks, replicas in (("pilot", pilot, 2), ("main", rest, 8)):
        bound = sum(t["max_output_bytes"] for t in tasks)
        plan = dict(protocol=PROTOCOL, model="long", methods=METHODS, arms=ARMS, replicates=REPLICATES,
            candidates=CANDIDATES, interventions=INTERVENTIONS, stage=stage,
            source_sha256={str(p): digest(p) for p in sources}, parent_audit=str(audit_path),
            alarm_table=str(alarm_dir / "first_alarms.csv"), alarm_verification=str(alarm_dir / "verification.json"),
            alarm_contract=str(alarm_dir / "contract.json"), frozen_parameters_sha256=PARAMETERS_SHA256,
            tasks=tasks, allowed_gpus=[0, 1, 2, 3, 4, 5, 7], render_gpus=[0, 3, 4, 5, 7],
            replicas_per_gpu=replicas, workers_per_gpu=replicas, mps=True,
            selection="all first 120 audited Long parents; one hash-selected parent/category in pilot, remaining 108 in main",
            repeated_parents_not_new_coverage=True, parameter_fitting=False,
            benchmark_counts=dict(Counter(t["benchmark"] for t in tasks)),
            event_states=sum(len(t["events"]) for t in tasks),
            maximum_output_bytes=bound, disk_free_bytes=shutil.disk_usage(HERE).free,
            disk_floor_gib=8, storage_quota_gib=16)
        if shutil.disk_usage(HERE).free - sum(t["max_output_bytes"] for t in pilot + rest) < 8 * 1024**3:
            raise ValueError("Full experiment exceeds storage headroom")
        path = output / (stage + "_plan.json")
        atomic_json(path, plan)
        load_plan(path)
        print(json.dumps(dict(stage=stage, parents=len(tasks), benchmarks=plan["benchmark_counts"],
            events=plan["event_states"], branches=plan["event_states"] * REPLICATES * len(ARMS),
            maximum_gib=bound / 1024**3, plan=str(path), sha256=digest(path))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
