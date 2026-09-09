#!/usr/bin/env python3
"""Combine audited runs and frozen alarm replays for a complete screening suite."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from compare_alarm_methods import metrics
import numpy as np
import pandas as pd

from analyze_collection_experiment import summarize
from collection_protocol import HERE, MANIFEST_SHA256, PARAMETERS_SHA256
from collection_storage import atomic_json, digest

SUITES = dict(long="libero_10", goal="libero_goal", spatial="libero_spatial", object="libero_object")


def run(args):
    manifest = HERE / "design/collection_manifest.csv"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen task manifest changed")
    with manifest.open(newline="") as stream:
        planned = {row["main_id"] for row in csv.DictReader(stream)
                   if row["screen"] == "1" and row["suite"] == SUITES[args.model]}
    audits, tasks, sources = {}, [], []
    for path in args.audits:
        value = json.loads(path.read_text())
        if value["status"] != "passed" or not value["formal_collection"]:
            raise ValueError("Only passed formal audits may be combined")
        summary_path = Path(value["run"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "completed" or summary["model"] != args.model:
            raise ValueError("Incomplete or mismatched collection run")
        identity = digest(path)
        if identity in audits:
            raise ValueError("Duplicate audit input")
        audits[identity] = value
        tasks.extend(value["tasks"])
        sources.append(dict(audit=str(path.resolve()), audit_sha256=identity,
            summary=str(summary_path), summary_sha256=digest(summary_path),
            mains=len(value["tasks"]), collection_elapsed_seconds=summary["collection_elapsed_seconds"],
            queries_per_second=summary["queries_per_second"], gpu_summary=summary["gpu_summary"]))
    identities = [task["main_id"] for task in tasks]
    if len(set(identities)) != len(identities) or set(identities) != planned:
        raise ValueError("Combined mains must cover the frozen suite screen exactly once")
    by_id = {task["main_id"]: task for task in tasks}
    frames, compared_audits, replay_sources = [], set(), []
    methods = None
    for directory in args.comparisons:
        verification = json.loads((directory / "verification.json").read_text())
        if verification["status"] != "passed":
            raise ValueError("Alarm replay verification did not pass")
        for name, expected in verification["artifacts"].items():
            if digest(directory / name) != expected:
                raise ValueError("Alarm replay artifact changed: " + name)
        contract = json.loads((directory / "contract.json").read_text())
        identity = contract["collection_audit_sha256"]
        if identity not in audits or identity in compared_audits:
            raise ValueError("Unmatched or duplicated alarm replay audit")
        compared_audits.add(identity)
        if contract["parameters_sha256"] != PARAMETERS_SHA256 or contract["reference_fit"] or contract["threshold_fit"]:
            raise ValueError("Alarm profiles differ from the frozen experiment")
        if methods is not None and methods != contract["methods"]:
            raise ValueError("Different alarm method sets")
        methods = contract["methods"]
        frame = pd.read_csv(directory / "first_alarms.csv")
        if frame.main_id.duplicated().any() or set(frame.main_id) != {task["main_id"] for task in audits[identity]["tasks"]}:
            raise ValueError("Alarm replay main identities differ from audit")
        for row in frame.itertuples():
            task = by_id[row.main_id]
            if (row.length != task["main_queries"] or bool(row.failure) == task["success"] or
                    row.benchmark != task["benchmark"] or row.analysis_role != task["analysis_role"]):
                raise ValueError("Alarm replay outcome or cohort mismatch")
        frames.append(frame)
        replay_sources.append(dict(directory=str(directory.resolve()),
                                   verification_sha256=digest(directory / "verification.json")))
    if compared_audits != set(audits):
        raise ValueError("An audited run is missing alarm replay results")
    frame = pd.concat(frames, ignore_index=True)
    cohorts = dict(all=frame, primary=frame.loc[frame.analysis_role.eq("perturbation")],
                   unchanged_scene_controls=frame.loc[~frame.analysis_role.eq("perturbation")])
    for benchmark in sorted(frame.benchmark.unique()):
        cohorts["all_" + benchmark] = frame.loc[frame.benchmark.eq(benchmark)]
        cohorts["primary_" + benchmark] = cohorts["primary"].loc[cohorts["primary"].benchmark.eq(benchmark)]
    metric_rows = [dict(cohort=name, method=method,
        **metrics(part[method].to_numpy(int), part, cutoff=int(frame.length.max()) - 1))
        for name, part in cohorts.items() for method in methods]
    primary = [task for task in tasks if task["analysis_role"] == "perturbation"]
    controls = [task for task in tasks if task["analysis_role"] != "perturbation"]
    queries = {key: sum(value[key] for value in audits.values())
               for key in ("main_queries", "c0_queries", "paired_branch_queries")}
    elapsed = sum(row["collection_elapsed_seconds"] for row in sources)
    report = dict(
        schema="moe_control.complete_screen_analysis.v1", status="passed", model=args.model,
        source_manifest_sha256=MANIFEST_SHA256, frozen_parameters_sha256=PARAMETERS_SHA256,
        analyzer_sha256=digest(Path(__file__)), collection_runs=sources, alarm_replays=replay_sources,
        complete_frozen_suite_screen=True, unique_mains=len(tasks), planned_mains=len(planned),
        benchmark_score=False, counts=dict(Counter(task["benchmark"] for task in tasks)),
        roles=dict(Counter(task["analysis_role"] for task in tasks)),
        all=summarize(tasks), primary=summarize(primary), unchanged_scene_controls=summarize(controls),
        per_benchmark={benchmark: summarize([task for task in primary if task["benchmark"] == benchmark])
                       for benchmark in sorted(frame.benchmark.unique())},
        alarm_metrics=metric_rows, queries=queries, total_queries=sum(queries.values()),
        paired_branches=sum(value["paired_branches"] for value in audits.values()),
        total_storage_bytes=sum(value["total_storage_bytes"] for value in audits.values()),
        collection_elapsed_seconds=elapsed, queries_per_second=sum(queries.values()) / elapsed,
        estimand="One native outcome and one mean paired intervention effect per main alarm state; no best-of-four",
        limitation="Complete planned screen, not the full benchmark distribution; only v7 alarm states have intervention pairs",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "first_alarms.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output / "metrics.csv", index=False)
    atomic_json(args.output / "summary.json", report)
    atomic_json(args.output / "verification.json", dict(status="passed", unique_mains=len(tasks),
        exact_screen_coverage=True, all_replay_artifacts_verified=True, shared_parameters=True,
        artifacts={path.name: digest(path) for path in args.output.iterdir() if path.is_file()}))
    print(json.dumps({key: report[key] for key in
        ("status", "unique_mains", "counts", "roles", "total_queries", "paired_branches", "total_storage_bytes")}))
    print(pd.DataFrame(metric_rows).loc[lambda table: table.cohort.eq("all"),
        ["method", "tp", "fp", "precision", "recall", "fpr"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(SUITES), default="long")
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
