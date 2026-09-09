#!/usr/bin/env python3
"""Seal validation parent IDs before development outcomes choose a controller."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from adaptive_control import PROTOCOL, ARMS
from collection_protocol import HERE, PARAMETERS_SHA256, stable_id
from collection_storage import atomic_json, digest


def chosen_parents(per_category):
    audit_path = HERE / "design/experiment_long_scale_audit_20260908.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Validation native parents are not audited")
    groups = defaultdict(list)
    for task in audit["tasks"]:
        groups[task["benchmark"], task["category"]].append(task)
    chosen = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda t: stable_id(PROTOCOL, "parent_selection", t["main_id"]))
        if len(rows) < per_category:
            raise ValueError("Insufficient validation parents")
        chosen.extend(rows[:per_category])
    return audit_path, chosen


def run(args):
    if args.selection is None:
        if args.cohort.exists():
            raise ValueError("Validation cohort is already sealed")
        audit_path, chosen = chosen_parents(args.per_category)
        development = [HERE / "design/adaptive_control_plans_20260908" / name for name in ("smoke.json", "development.json")]
        development_ids = {t["main_id"] for path in development for t in json.loads(path.read_text())["tasks"]}
        ids = [t["main_id"] for t in chosen]
        if set(ids) & development_ids:
            raise ValueError("Development/validation overlap")
        atomic_json(args.cohort, dict(protocol=PROTOCOL, per_category=args.per_category, main_ids=ids,
            benchmark_counts=dict(Counter(t["benchmark"] for t in chosen)),
            parent_audit=str(audit_path), parent_audit_sha256=digest(audit_path),
            development_plans={str(p): digest(p) for p in development},
            disjoint_development_ids=sorted(development_ids), frozen_parameters_sha256=PARAMETERS_SHA256,
            control_outcomes_read=False, rule="first fixed hashes within benchmark/category, including unalarmed parents",
            selector_sha256=digest(__file__)))
        print(json.dumps(dict(cohort=str(args.cohort), parents=len(ids), benchmarks=dict(Counter(t["benchmark"] for t in chosen)))))
        return
    if args.output is None:
        raise ValueError("A validation plan output is required")
    cohort = json.loads(args.cohort.read_text())
    selection = json.loads(args.selection.read_text())
    if cohort["protocol"] != PROTOCOL or selection["protocol"] != PROTOCOL or selection["validation_outcomes_read"]:
        raise ValueError("Invalid validation selection provenance")
    for path, expected in cohort["development_plans"].items():
        if digest(path) != expected:
            raise ValueError("Development plan changed")
    audit_path, chosen = chosen_parents(cohort["per_category"])
    if digest(audit_path) != cohort["parent_audit_sha256"] or [t["main_id"] for t in chosen] != cohort["main_ids"]:
        raise ValueError("Sealed validation cohort changed")
    for path, expected in selection["development_audits"].items():
        if digest(path) != expected or json.loads(Path(path).read_text())["status"] != "passed":
            raise ValueError("Selected policy lacks verified development evidence")
    if not set(selection["validation_arms"]) <= set(ARMS):
        raise ValueError("Unsupported validation controller")
    from prepare_adaptive_control import run as prepare
    prepare(argparse.Namespace(stage="validation", per_category=cohort["per_category"], replicates=4, replicas=8,
        exclude_plan=None, arms=selection["validation_arms"], selection=args.selection, output=args.output))
    plan = json.loads(args.output.read_text())
    if [t["main_id"] for t in plan["tasks"]] != cohort["main_ids"]:
        raise ValueError("Plan differs from the sealed validation cohort")
    atomic_json(args.output.with_suffix(".cohort_verification.json"), dict(status="passed", parents=len(chosen),
        cohort_sha256=digest(args.cohort), selection_sha256=digest(args.selection), plan_sha256=digest(args.output),
        zero_development_overlap=True, selected_policies=selection["selected_policies"],
        validator_sha256=digest(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
