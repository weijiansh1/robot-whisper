#!/usr/bin/env python3
"""Freeze pilot and reserved confirmation states before closed-loop outcomes."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from adaptive_control import PARAMETERS, REFERENCE
from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from fixed_recovery_control import TriggerMonitor
from v8_closed_loop import PROTOCOL, ARMS, SETTINGS, limits

RUNTIME = ("v8_closed_loop.py", "test_v8_closed_loop.py", "collect_v8_closed_loop.py",
    "prepare_v8_closed_loop.py", "run_v8_closed_loop.py", "fixed_recovery_control.py",
    "collect_fixed_recovery.py", "serve_v8_feature_model.py", "v8_feature_control.py",
    "collect_adaptive_control.py", "adaptive_control.py", "collection_routes.py", "collection_storage.py",
    "collection_protocol.py", "collect_preflight_worker.py", "collection_noise.py", "collection_mps.py",
    "collection_worker_pool.py", "serve_isolated_model.py", "run_collection_preflight.py")


def balanced(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["benchmark"], row["category"]].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: stable_id(PROTOCOL, "cohort", row["main_id"]))
    keys = sorted(groups, key=lambda key: stable_id(PROTOCOL, "stratum", *key))
    ordered = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def prepare(args):
    verify_frozen_alarm()
    audit_path = HERE / "design/experiment_long_scale_audit_20260908.json"
    alarm_path = HERE / "design/experiment_long_scale_alarm_comparison_20260908/first_alarms.csv"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Audited native mains required")
    with alarm_path.open(newline="") as stream:
        alarms = {row["main_id"]: row for row in csv.DictReader(stream)}
    eligible = [row for row in audit["tasks"] if row["analysis_role"] == "perturbation" and
        0 <= int(alarms[row["main_id"]]["v8_frozen"]) < row["main_queries"]-1]
    failed = balanced([row for row in eligible if not row["success"]])
    successful = balanced([row for row in eligible if row["success"]])
    if len(failed) < 24 or len(successful) < 8:
        raise ValueError("Insufficient disjoint eligible pilot and confirmation states")
    thresholds, margins = limits()
    sources = [HERE / name for name in RUNTIME]+[audit_path, alarm_path, PARAMETERS, REFERENCE]
    sources += [HERE.parent / "moe-v7-0905/method/intrinsic_guard_monitor.py",
                HERE.parent / "himoe-route-capture/route_noise_selector.py"]
    for stage, offset in (("pilot", 0), ("confirmation", 1)):
        tasks = []
        for parent in failed[offset*12:(offset+1)*12]+successful[offset*4:(offset+1)*4]:
            directory = Path(parent["directory"])
            original = json.loads((directory / "result.json").read_text())
            live = TriggerMonitor()
            for row in records(directory / "main"):
                live.update(row[PROBS_KEY], row["proprio"][:3])
            first = dict(live.first)
            for method in ("v7_frozen", "v8_frozen", "knn20", "knn_euclidean_OR_cosine"):
                if first[method] != int(alarms[parent["main_id"]][method]):
                    raise ValueError("Frozen trigger differs: "+method)
            q0 = first["v8_frozen"]+1
            event = dict(event_id=stable_id(PROTOCOL, parent["main_id"], q0),
                start_query=q0, alarm_query=q0-1, methods=["v8_frozen"], deployable=True)
            maximum = parent["main_queries"]+5*(52-q0)+4*15*min(12, 52-q0)
            task = {key: parent[key] for key in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
            task.update(parent_directory=str(directory), parent_queries=parent["main_queries"],
                parent_commit_sha256=digest(directory / "main_complete.json"),
                parent_manifest_sha256=digest(directory / "main/manifest.json"), noise_seed=original["seed"],
                init_index=original["init_index"], first_alarm=first["knn20"], first_alarms=first,
                base_task=original["variant"]["base_task"], native_success=original["success"], events=[event],
                maximum_queries=maximum, max_output_bytes=maximum*200*1024+16*1024**2)
            tasks.append(task)
        plan = dict(protocol=PROTOCOL, stage=stage, model="long", tasks=tasks, settings=SETTINGS,
            thresholds=thresholds.tolist(), margins=margins.tolist(), recovery_targets=(thresholds-margins).tolist(),
            arms=list(ARMS), methods=["v8_frozen"], allowed_gpus=[0, 1, 2, 3], render_gpus=[0, 3],
            replicas_per_gpu=8, workers_per_gpu=8, batch_size=1, hidden_capture=False, threshold_fitting=False,
            selection="v8-alarmed perturbation mains; 12 native failures plus 4 successes; hash ordered round-robin benchmark/category",
            evaluation_scope="conditional recovery feasibility; historical native outcomes used for stratification; no overall success-rate claim",
            confirmation_rule="proceed if a v8 chooser has positive rescue-minus-harm and more paired wins than losses versus its matching random chooser; report selection and test disjoint reserved parents",
            timing="complete native main, complete exact C0, first frozen v8 alarm q+1; native chunk10; original total520",
            random_stream="native RNG draw once per executed query; extra streams paired across arms; exact native environment tape by absolute step",
            new_main_coverage=0, source_audit=str(audit_path), source_audit_sha256=digest(audit_path),
            runtime_sources=list(RUNTIME), source_sha256={str(path): digest(path) for path in sources},
            maximum_output_bytes=sum(t["max_output_bytes"] for t in tasks), storage_quota_gib=6, disk_floor_gib=8)
        path = args.output_prefix.with_name(args.output_prefix.name+"_"+stage+".json")
        if path.exists():
            raise ValueError("Refusing to overwrite frozen plan")
        atomic_json(path, plan)
        print(json.dumps(dict(stage=stage, path=str(path), parents=len(tasks), failed=12, successful=4,
            benchmarks={name: sum(t["benchmark"] == name for t in tasks) for name in ("pro", "plus")},
            max_queries=sum(t["maximum_queries"] for t in tasks), max_gib=plan["maximum_output_bytes"]/1024**3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    prepare(parser.parse_args())
