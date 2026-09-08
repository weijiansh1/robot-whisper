#!/usr/bin/env python3
"""Outcome tables for a supervisor (v2) run, paired with the v1 twins (retract_above_target / new_noise)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json
from repair_control import PROTOCOL_V2


def v2_rows(run_dir, plan):
    rows = []
    for task in plan["tasks"]:
        result = json.loads((run_dir / "tasks" / task["main_id"] / "branches/result.json").read_text())
        for b in result["branches"]:
            interventions = b.get("interventions", [])
            rows.append(dict(main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], timing=b["timing"],
                arm=b["arm"], event_id=b["event_id"], replicate=b["replicate"], physical_class=b["physical_class"],
                success_window=bool(b["success"]), success_original=bool(b["success_within_original"]),
                success_step=b["success_step"], action_steps=b["action_steps"], twin=b.get("twin"),
                first_extra=b.get("first_extra_intervention_query"), interventions=len(interventions),
                g0=sum(i["guard"] == "G0" for i in interventions), g2=sum(i["guard"] == "G2" for i in interventions),
                g3=sum(i["guard"] == "G3" and i.get("kind") == "place" for i in interventions),
                g3_no_region=sum(i["guard"] == "G3" and i.get("kind") == "no_region" for i in interventions),
                repair_steps=b.get("repair_steps", 0)))
    return rows


def v1_lookup(path):
    table = {}
    with Path(path).open(newline="") as stream:
        for r in csv.DictReader(stream):
            table[(r["event_id"], int(r["replicate"]), r["arm"])] = dict(window=r["success_window"] == "True",
                                                                          original=r["success_original"] == "True")
    return table


def cell(rows):
    return dict(n=len(rows), window=sum(r["success_window"] for r in rows), original=sum(r["success_original"] for r in rows),
                twin_window=sum(r["twin_window"] for r in rows), twin_original=sum(r["twin_original"] for r in rows),
                new_noise_window=sum(r["nn_window"] for r in rows),
                wins=sum(r["success_window"] and not r["twin_window"] for r in rows),
                losses=sum(r["twin_window"] and not r["success_window"] for r in rows),
                g2_fired=sum(r["g2"] > 0 for r in rows), g3_fired=sum(r["g3"] > 0 for r in rows),
                g3_no_region=sum(r["g3_no_region"] > 0 for r in rows), vetoed=sum(r["g0"] > 0 for r in rows),
                diverged=sum(r["first_extra"] is not None for r in rows))


def bootstrap(rows, key="twin_window", n_boot=2000, seed=20260908):
    rng = np.random.default_rng(seed)
    per_main = defaultdict(list)
    for r in rows:
        per_main[r["main_id"]].append(int(r["success_window"]) - int(r[key]))
    mains = list(per_main)
    if not mains:
        return None
    means = np.array([np.mean(per_main[m]) for m in mains])
    draws = [means[rng.integers(0, len(mains), len(mains))].mean() for _ in range(n_boot)]
    return dict(mains=len(mains), point=float(means.mean()), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    if plan["protocol"] != PROTOCOL_V2:
        raise ValueError("Not a supervisor run")
    v1 = v1_lookup(Path(plan["replay_run"]).parent / (Path(plan["replay_run"]).name.replace("repair_main", "repair_main_analysis")) / "branches.csv"
                   if args.v1_branches is None else args.v1_branches)
    rows = v2_rows(args.run, plan)
    for r in rows:
        twin = v1.get((r["event_id"], r["replicate"], r["twin"] or "retract_above_target"), dict(window=False, original=False))
        nn = v1.get((r["event_id"], r["replicate"], "new_noise"), dict(window=False, original=False))
        r.update(twin_window=twin["window"], twin_original=twin["original"], nn_window=nn["window"])
    failed = [r for r in rows if r["failed"]]
    success = [r for r in rows if not r["failed"]]
    summary = dict(protocol=PROTOCOL_V2, run=str(args.run), branches=len(rows), failed_parents=len({r["main_id"] for r in failed}),
                   success_parents=len({r["main_id"] for r in success}), supervisor=plan["supervisor"])
    summary["by_timing_arm"] = {"%s|%s" % (t, a): cell([r for r in failed if r["timing"] == t and r["arm"] == a])
                                for t in ("early", "mid", "late") for a in plan["arms"]
                                if any(r["timing"] == t and r["arm"] == a for r in failed)}
    summary["by_class_arm"] = {"%s|%s" % (c, a): cell([r for r in failed if r["physical_class"] == c and r["arm"] == a])
                               for c in sorted({r["physical_class"] for r in failed}) for a in plan["arms"]}
    summary["by_base_task_full"] = {b: cell([r for r in failed if r["base_task"] == b and r["arm"] == "supervisor_full"])
                                    for b in sorted({r["base_task"] for r in failed})}
    summary["conditional_on_guard"] = {}
    for a in plan["arms"]:
        arm_rows = [r for r in failed if r["arm"] == a]
        summary["conditional_on_guard"][a] = dict(
            g2_fired=cell([r for r in arm_rows if r["g2"] > 0]), g3_fired=cell([r for r in arm_rows if r["g3"] > 0]),
            no_extra=cell([r for r in arm_rows if r["first_extra"] is None]))
    summary["harm_success_parents"] = {a: cell([r for r in success if r["arm"] == a]) for a in plan["arms"]}
    summary["bootstrap_vs_twin"] = {"%s|%s" % (t, a): bootstrap([r for r in failed if r["timing"] == t and r["arm"] == a])
                                    for t in ("early", "mid", "late") for a in plan["arms"]}
    summary["bootstrap_vs_new_noise"] = {"%s|%s" % (t, a): bootstrap([r for r in failed if r["timing"] == t and r["arm"] == a], "nn_window")
                                         for t in ("early", "mid", "late") for a in plan["arms"]}
    rescued = defaultdict(set)
    for r in failed:
        if r["success_window"]:
            rescued[r["arm"]].add(r["main_id"])
        if r["twin_window"]:
            rescued["twin:" + (r["twin"] or "retract_above_target")].add(r["main_id"])
    summary["parents_rescued"] = {k: len(v) for k, v in rescued.items()}
    summary["parents_rescued_any_arm"] = len(set().union(*[v for k, v in rescued.items() if not k.startswith("twin")]) if rescued else set())
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "summary.json", summary)
    with (args.out / "branches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for k, v in summary["by_timing_arm"].items():
        print("%-34s n=%3d B=%2d (twin %2d, new_noise %2d) A=%2d (twin %2d) wins=%2d losses=%2d G2=%2d G3=%2d veto=%d" % (
            k, v["n"], v["window"], v["twin_window"], v["new_noise_window"], v["original"], v["twin_original"], v["wins"], v["losses"],
            v["g2_fired"], v["g3_fired"], v["vetoed"]))
    print("harm:", {a: "%d/%d (twin %d)" % (v["window"], v["n"], v["twin_window"]) for a, v in summary["harm_success_parents"].items()})
    print("parents rescued:", summary["parents_rescued"], "any arm", summary["parents_rescued_any_arm"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v1-branches", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
