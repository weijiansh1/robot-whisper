#!/usr/bin/env python3
"""Describe an audited batch without treating branch replicates as independent mains."""

import argparse
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np

from collection_protocol import HERE, MANIFEST_SHA256
from collection_storage import atomic_json, digest, records


def completion_counts(completed):
    manifest = HERE / "design/collection_manifest.csv"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen source task manifest changed")
    with manifest.open(newline="") as stream:
        inventory = {row["main_id"]: row for row in csv.DictReader(stream)}
    identities = set(completed)
    if identities - inventory.keys():
        raise ValueError("Completion index contains an unknown main ID")
    screen = {main_id for main_id, row in inventory.items() if row["screen"] == "1"}
    return dict(completed_count=len(identities), completed_screen_count=len(identities & screen),
        completed_extension_count=len(identities - screen),
        screen_planned_remaining=len(screen - identities),
        coverage_planned_remaining=len(inventory) - len(identities))


def summarize(tasks):
    alarms = [task for task in tasks if task["first_alarm_query"] is not None]
    success = [task for task in tasks if task["success"]]
    failures = [task for task in tasks if not task["success"]]
    fp = sum(task["first_alarm_query"] is not None for task in success)
    tp = sum(task["first_alarm_query"] is not None for task in failures)
    result = dict(mains=len(tasks), successes=len(success), failures=len(failures), alarmed=len(alarms),
        false_positive_mains=fp, false_positive_denominator=len(success),
        false_positive_rate=fp / len(success) if success else None,
        true_positive_mains=tp, recall_denominator=len(failures), recall=tp / len(failures) if failures else None,
        paired_states=len(alarms), paired_replicates=sum(len(task["branches"]) // 2 for task in alarms))
    state_values = []
    for task in alarms:
        values = {arm: float(np.mean([b["success"] for b in task["branches"] if b["arm"] == arm])) for arm in ("C1", "T1")}
        state_values.append(dict(main_id=task["main_id"], main_success=task["success"], **values,
                                 delta=values["T1"] - values["C1"]))
    for arm in ("C1", "T1"):
        rescue = [row[arm] for row in state_values if not row["main_success"]]
        harm = [1 - row[arm] for row in state_values if row["main_success"]]
        result[arm] = dict(success_probability_on_alarmed_states=float(np.mean([row[arm] for row in state_values])) if state_values else None,
            rescue_probability=float(np.mean(rescue)) if rescue else None, rescue_states=len(rescue),
            harm_probability=float(np.mean(harm)) if harm else None, harm_states=len(harm))
    result["paired_delta_T1_minus_C1"] = float(np.mean([row["delta"] for row in state_values])) if state_values else None
    result["state_pairs"] = state_values
    return result


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed":
        raise ValueError("Only a fully audited batch can enter the descriptive result table")
    root = Path(audit["run"])
    summary = json.loads((root / "summary.json").read_text())
    tasks = audit["tasks"]
    primary = [row for row in tasks if row["analysis_role"] == "perturbation"]
    controls = [row for row in tasks if row["analysis_role"] != "perturbation"]
    samples = json.loads((root / "gpu_samples.json").read_text())
    supplied = [sample for sample in samples if all(sample["active_tasks_by_gpu"][str(gpu)] >= summary["replicas_per_gpu"]
                                                  for gpu in summary["gpus"])]
    action_effects = []
    for task in tasks:
        if not task["branches"]:
            continue
        for replicate in range(4):
            c1 = next(records(Path(task["directory"]) / "branches/C1" / str(replicate) / "suffix"))
            t1 = next(records(Path(task["directory"]) / "branches/T1" / str(replicate) / "suffix"))
            delta = t1["actions"].astype(np.float64) - c1["actions"]
            action_effects.append(dict(main_id=task["main_id"], replicate=replicate,
                max_abs_delta=float(np.max(np.abs(delta))), rms_delta=float(np.sqrt(np.mean(delta**2)))))
    free_bytes = shutil.disk_usage(root).free
    mean_bytes = audit["total_storage_bytes"] / len(tasks)
    report = dict(run=str(root), audit_sha256=digest(args.audit), formal_collection=audit["formal_collection"],
        analyzer_sha256=digest(Path(__file__)),
        estimand="Descriptive batch averages; one paired estimate per alarm state; no best-of-four selection",
        benchmark_score=False, primary=summarize(primary), unchanged_scene_controls=summarize(controls),
        per_benchmark={benchmark: summarize([task for task in primary if task["benchmark"] == benchmark]) for benchmark in ("pro", "plus")},
        per_category={benchmark + ":" + category: summarize([task for task in primary if task["benchmark"] == benchmark and task["category"] == category])
                      for benchmark, category in sorted({(task["benchmark"], task["category"]) for task in primary})},
        total_main_queries=audit["main_queries"], total_c0_queries=audit["c0_queries"],
        total_paired_queries=audit["paired_branch_queries"], total_paired_branches=audit["paired_branches"],
        collection_elapsed_seconds=summary["collection_elapsed_seconds"], queries_per_second=summary["queries_per_second"],
        gpu_summary=summary["gpu_summary"], supplied_samples=len(supplied),
        host_peak_memory_bytes=summary.get("host_peak_memory_bytes"), host_mean_cpu_cores=summary.get("host_mean_cpu_cores"),
        supplied_power_mean_w=float(np.mean([row["power_w"] for sample in supplied for row in sample["gpus"]])) if supplied else None,
        supplied_utilization_mean_percent=float(np.mean([row["utilization_percent"] for sample in supplied for row in sample["gpus"]])) if supplied else None,
        total_storage_bytes=audit["total_storage_bytes"], mean_bytes_per_main=mean_bytes,
        c0_alarm_suffixes=audit["alarmed_mains"], c0_quality_only_suffixes=len(tasks) - audit["alarmed_mains"],
        first_query_action_effects=action_effects,
        first_query_pairs_with_changed_actions=sum(row["max_abs_delta"] > 0 for row in action_effects),
        storage_projection=dict(disk_free_bytes=free_bytes, reserve_bytes=8 * 1024**3,
            safety_factor=1.15, additional_mains_at_observed_mix=max(0, int((free_bytes - 8 * 1024**3) / (1.15 * mean_bytes))),
            limitation="Estimate at this batch's alarm frequency and suffix lengths, not a guaranteed maximum or an all-suite estimate"))
    atomic_json(args.output, report)
    if audit["formal_collection"]:
        index = dict(schema="moe_control.collection_status.v1", source_manifest_sha256=MANIFEST_SHA256,
            run=str(root), audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
            completed_mains=[dict(main_id=task["main_id"], variant_id=task["variant_id"], directory=task["directory"],
                status="audited_complete", analysis_role=task["analysis_role"], paired_branches=len(task["branches"])) for task in tasks],
            counts=dict(Counter(task["benchmark"] for task in tasks)))
        atomic_json(root / "collection_status.json", index)
        index_path = HERE / "design/collection_status.json"
        with index_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            combined = json.loads(index_path.read_text()) if index_path.exists() else dict(
                schema="moe_control.collection_index.v1", source_manifest_sha256=MANIFEST_SHA256, completed={})
            if combined["source_manifest_sha256"] != MANIFEST_SHA256:
                raise ValueError("Global completion index manifest mismatch")
            for task in index["completed_mains"]:
                old = combined["completed"].get(task["main_id"])
                if old is not None and old["directory"] != task["directory"]:
                    raise ValueError("Main ID was already accepted from another run")
                combined["completed"][task["main_id"]] = dict(task, model=summary["model"],
                    audit=str(args.audit.resolve()), audit_sha256=digest(args.audit))
            combined.update(completion_counts(combined["completed"]))
            atomic_json(index_path, combined)
    print(json.dumps({key: value for key, value in report.items() if key not in
        ("primary", "unchanged_scene_controls", "per_benchmark", "per_category", "first_query_action_effects")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
