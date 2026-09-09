#!/usr/bin/env python3
"""Freeze the first Long screening batch before observing any formal outcomes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from collection_protocol import (HERE, MANIFEST_SHA256, PARAMETERS_SHA256, PROTOCOL, REPLICATES, load_plan)
from collection_storage import atomic_json, digest


def run(output):
    source = HERE / "design/collection_manifest.csv"
    if digest(source) != MANIFEST_SHA256 or output.exists():
        raise ValueError("Changed source manifest or existing frozen plan")
    with source.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["suite"] == "libero_10" and row["screen"] == "1"]
    pro = [row for row in rows if row["benchmark"] == "pro" and row["init_index"] == "0"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["benchmark"] == "plus":
            groups[row["category"]][row["base_task"]].append(row)

    def order(row):
        return hashlib.sha256(("long-batch1-20260908:" + row["main_id"]).encode()).hexdigest()

    plus = []
    for category, tasks in sorted(groups.items()):
        # One fixed-hash variant per base task; sparse category cells redistribute without replacement.
        ordered = {task: sorted(items, key=order) for task, items in tasks.items()}
        chosen = []
        for rank in range(100):
            for task in sorted(ordered):
                if rank < len(ordered[task]) and len(chosen) < 10:
                    row = dict(ordered[task][rank])
                    row["batch_base_category_population_in_screen"] = len(ordered[task])
                    chosen.append(row)
            if len(chosen) == 10:
                break
        plus.extend(chosen)
    if len(pro) != 50 or len(plus) != 70:
        raise ValueError("Unexpected Long screening coverage")
    controls = set(json.loads((HERE / "benchmarks/READINESS.json").read_text())["pro_environment_generation"]["bddl_unchanged_from_base"])
    tasks = sorted(pro + plus, key=order)
    for row in tasks:
        row["analysis_role"] = "unchanged_scene_control" if row["variant_id"] in controls else "perturbation"
        row["status"] = "planned"
    plan = dict(schema="moe_control.run_plan.v1", protocol=PROTOCOL, model="long", stage="P1_long_batch1",
        source_manifest_sha256=MANIFEST_SHA256, frozen_parameters_sha256=PARAMETERS_SHA256,
        replicates=REPLICATES, main_intervention=False, hidden_capture=False,
        allowed_gpus=[0, 1, 2, 3, 4, 5, 7], render_gpus=[0, 3, 4, 5, 7], mps=True,
        replicas_per_gpu=8, workers_per_gpu=8, batch_size=1, storage_quota_gib=12, disk_floor_gib=8,
        selection="Pro all 50 Long entries at init=0; Plus 10/category from frozen screen, balanced by base task and stable hash",
        inference_scope="first fixed Long batch; descriptive results only, not a full benchmark score",
        no_alarm_c0="One extra native replay from q5 or q0 for fidelity QC; not an intervention event",
        branch_arms={"C0": 1, "C1": 4, "T1": 4}, checkpoint_model="long", horizon_steps=520,
        branch_trigger="all first alarms, independent of final main success; only after main commit and exact C0",
        cohort_counts=dict(Counter(row["benchmark"] for row in tasks)),
        role_counts=dict(Counter(row["analysis_role"] for row in tasks)),
        base_task_count=len({row["base_task"] for row in tasks}), tasks=tasks)
    atomic_json(output, plan)
    load_plan(output, "long")
    print(json.dumps({"plan": str(output), "sha256": digest(output), "counts": plan["cohort_counts"],
                      "roles": plan["role_counts"], "base_tasks": plan["base_task_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
