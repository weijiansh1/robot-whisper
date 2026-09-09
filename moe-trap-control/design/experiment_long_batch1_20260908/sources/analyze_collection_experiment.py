#!/usr/bin/env python3
"""Describe an audited batch without treating branch replicates as independent mains."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from collection_protocol import MANIFEST_SHA256
from collection_storage import atomic_json, digest


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
    report = dict(run=str(root), audit_sha256=digest(args.audit), formal_collection=audit["formal_collection"],
        estimand="Descriptive first-batch averages; one paired estimate per alarm state; no best-of-four selection",
        benchmark_score=False, primary=summarize(primary), unchanged_scene_controls=summarize(controls),
        per_benchmark={benchmark: summarize([task for task in primary if task["benchmark"] == benchmark]) for benchmark in ("pro", "plus")},
        per_category={benchmark + ":" + category: summarize([task for task in primary if task["benchmark"] == benchmark and task["category"] == category])
                      for benchmark, category in sorted({(task["benchmark"], task["category"]) for task in primary})},
        total_main_queries=audit["main_queries"], total_c0_queries=audit["c0_queries"],
        total_paired_queries=audit["paired_branch_queries"], total_paired_branches=audit["paired_branches"],
        collection_elapsed_seconds=summary["collection_elapsed_seconds"], queries_per_second=summary["queries_per_second"],
        gpu_summary=summary["gpu_summary"], supplied_samples=len(supplied),
        supplied_power_mean_w=float(np.mean([row["power_w"] for sample in supplied for row in sample["gpus"]])) if supplied else None,
        supplied_utilization_mean_percent=float(np.mean([row["utilization_percent"] for sample in supplied for row in sample["gpus"]])) if supplied else None,
        total_storage_bytes=audit["total_storage_bytes"], mean_bytes_per_main=audit["total_storage_bytes"] / len(tasks))
    atomic_json(args.output, report)
    if audit["formal_collection"]:
        index = dict(schema="moe_control.collection_status.v1", source_manifest_sha256=MANIFEST_SHA256,
            run=str(root), audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
            completed_mains=[dict(main_id=task["main_id"], variant_id=task["variant_id"], directory=task["directory"],
                status="audited_complete", analysis_role=task["analysis_role"], paired_branches=len(task["branches"])) for task in tasks],
            counts=dict(Counter(task["benchmark"] for task in tasks)))
        atomic_json(root / "collection_status.json", index)
    print(json.dumps({key: value for key, value in report.items() if key not in ("primary", "unchanged_scene_controls", "per_benchmark", "per_category")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
