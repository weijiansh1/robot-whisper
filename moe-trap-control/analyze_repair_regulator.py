#!/usr/bin/env python3
"""Outcome tables for a feedback-regulator (v3) run: fork branches paired with v1 twins, and full episodes paired by parent."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json
from repair_control import PROTOCOL_V3
from analyze_repair_supervisor import v1_lookup, bootstrap


def fork_rows(run_dir, plan):
    rows = []
    for task in plan["tasks"]:
        path = run_dir / "tasks" / task["main_id"] / "branches/result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for b in result["branches"]:
            rows.append(dict(main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], timing=b["timing"],
                arm=b["arm"], event_id=b["event_id"], replicate=b["replicate"], physical_class=b["physical_class"],
                success_window=bool(b["success"]), success_original=bool(b["success_within_original"]),
                success_step=b["success_step"], first_modified=b.get("first_extra_intervention_query"),
                modified_steps=b.get("modified_steps", 0), gate_close_steps=b.get("gate_close_steps", 0),
                gate_open_steps=b.get("gate_open_steps", 0), servo_effort=round(float(b.get("servo_effort", 0.0)), 2)))
    return rows


def episode_rows(run_dir, plan):
    rows = []
    for task in plan["tasks"]:
        path = run_dir / "tasks" / task["main_id"] / "episodes/result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for e in result.get("episodes", []):
            rows.append(dict(main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], arm=e["arm"],
                replicate=e["replicate"], success=bool(e["success"]), success_step=e["success_step"],
                first_alarm_query=e.get("first_alarm_query"), engaged_query=e.get("engaged_query"),
                first_modified_query=e.get("first_modified_query"), modified_steps=e.get("modified_steps", 0),
                gate_close_steps=e.get("gate_close_steps", 0), gate_open_steps=e.get("gate_open_steps", 0),
                servo_effort=round(float(e.get("servo_effort", 0.0)), 2), fidelity=e.get("fidelity")))
    return rows


def fork_cell(rows):
    return dict(n=len(rows), window=sum(r["success_window"] for r in rows), original=sum(r["success_original"] for r in rows),
                new_noise_window=sum(r["nn_window"] for r in rows), a4_window=sum(r["a4_window"] for r in rows),
                wins_vs_new_noise=sum(r["success_window"] and not r["nn_window"] for r in rows),
                losses_vs_new_noise=sum(r["nn_window"] and not r["success_window"] for r in rows),
                wins_vs_a4=sum(r["success_window"] and not r["a4_window"] for r in rows),
                losses_vs_a4=sum(r["a4_window"] and not r["success_window"] for r in rows),
                modified=sum(r["first_modified"] is not None for r in rows),
                gate_close_steps=sum(r["gate_close_steps"] for r in rows), gate_open_steps=sum(r["gate_open_steps"] for r in rows))


def episode_cell(rows):
    return dict(n=len(rows), success=sum(r["success"] for r in rows), engaged=sum(r["engaged_query"] is not None for r in rows),
                modified=sum(r["first_modified_query"] is not None for r in rows),
                median_success_step=float(np.median([r["success_step"] for r in rows if r["success"]])) if any(r["success"] for r in rows) else None)


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    if plan["protocol"] != PROTOCOL_V3:
        raise ValueError("Not a regulator run")
    v1 = v1_lookup(args.v1_branches)
    forks, episodes = fork_rows(args.run, plan), episode_rows(args.run, plan)
    for extra in args.extra_runs:
        other = json.loads((extra / "plan.json").read_text())
        if other["protocol"] != PROTOCOL_V3 or other["regulator"] != plan["regulator"] or other["replay_run"] != plan["replay_run"]:
            raise ValueError("Extra run uses a different regulator protocol")
        seen = {(r["main_id"], r["event_id"], r["replicate"], r["arm"]) for r in forks}
        forks += [r for r in fork_rows(extra, other) if (r["main_id"], r["event_id"], r["replicate"], r["arm"]) not in seen]
        seen_e = {(r["main_id"], r["replicate"], r["arm"]) for r in episodes}
        episodes += [r for r in episode_rows(extra, other) if (r["main_id"], r["replicate"], r["arm"]) not in seen_e]
        plan["arms"] = list(plan["arms"]) + [a for a in other["arms"] if a not in plan["arms"]]
        plan["episode_arms"] = list(plan["episode_arms"]) + [a for a in other["episode_arms"] if a not in plan["episode_arms"]]
    for r in forks:
        r["nn_window"] = v1.get((r["event_id"], r["replicate"], "new_noise"), dict(window=False))["window"]
        r["a4_window"] = v1.get((r["event_id"], r["replicate"], "retract_above_target"), dict(window=False))["window"]
    failed = [r for r in forks if r["failed"]]
    success = [r for r in forks if not r["failed"]]
    summary = dict(protocol=PROTOCOL_V3, run=str(args.run), extra_runs=[str(e) for e in args.extra_runs],
                   fork_branches=len(forks), episodes=len(episodes),
                   failed_parents=len({r["main_id"] for r in failed}), success_parents=len({r["main_id"] for r in success}),
                   regulator=plan["regulator"])
    summary["fork_by_timing_arm"] = {"%s|%s" % (t, a): fork_cell([r for r in failed if r["timing"] == t and r["arm"] == a])
                                     for t in ("early", "mid", "late") for a in plan["arms"]
                                     if any(r["timing"] == t and r["arm"] == a for r in failed)}
    summary["fork_by_class_arm"] = {"%s|%s" % (c, a): fork_cell([r for r in failed if r["physical_class"] == c and r["arm"] == a])
                                    for c in sorted({r["physical_class"] for r in failed}) for a in plan["arms"]}
    summary["fork_by_base_task"] = {"%s|%s" % (b, a): fork_cell([r for r in failed if r["base_task"] == b and r["arm"] == a])
                                    for b in sorted({r["base_task"] for r in failed}) for a in plan["arms"]}
    summary["fork_harm_success_parents"] = {a: fork_cell([r for r in success if r["arm"] == a]) for a in plan["arms"]}
    summary["fork_bootstrap_vs_new_noise"] = {"%s|%s" % (t, a): bootstrap([r for r in failed if r["timing"] == t and r["arm"] == a], "nn_window")
                                              for t in ("early", "mid", "late") for a in plan["arms"]}
    summary["fork_bootstrap_vs_a4"] = {"%s|%s" % (t, a): bootstrap([r for r in failed if r["timing"] == t and r["arm"] == a], "a4_window")
                                       for t in ("early", "mid", "late") for a in plan["arms"]}
    rescued = defaultdict(set)
    for r in failed:
        if r["success_window"]:
            rescued[r["arm"]].add(r["main_id"])
    summary["fork_parents_rescued"] = {a: len(v) for a, v in rescued.items()}
    # in-run pairing against the no-regulator fork arm (fresh replicates have no v1 twin)
    control = {(r["event_id"], r["replicate"]): r["success_window"] for r in forks if r["arm"] == "vla_fork"}
    if control:
        pairs = {}
        for a in plan["arms"]:
            if a == "vla_fork":
                continue
            for label, subset in (("failed", failed), ("success", success)):
                both = [(r, control[(r["event_id"], r["replicate"])]) for r in subset if r["arm"] == a and (r["event_id"], r["replicate"]) in control]
                pairs["%s|%s" % (a, label)] = dict(pairs=len(both), arm_success=sum(r["success_window"] for r, _ in both),
                                                 control_success=sum(c for _, c in both),
                                                 wins=sum(r["success_window"] and not c for r, c in both),
                                                 losses=sum(c and not r["success_window"] for r, c in both))
        summary["fork_paired_vs_vla_fork"] = pairs
        by_timing = {}
        for a in plan["arms"]:
            for t in ("mid", "late"):
                both = [(r, control[(r["event_id"], r["replicate"])]) for r in failed if r["arm"] == a and r["timing"] == t and (r["event_id"], r["replicate"]) in control]
                if both:
                    by_timing["%s|%s" % (t, a)] = dict(n=len(both), success=sum(r["success_window"] for r, _ in both),
                                                       control=sum(c for _, c in both), wins=sum(r["success_window"] and not c for r, c in both),
                                                       losses=sum(c and not r["success_window"] for r, c in both))
        summary["fork_by_timing_vs_vla_fork"] = by_timing
    ef, es = [r for r in episodes if r["failed"]], [r for r in episodes if not r["failed"]]
    summary["episodes_failed_parents"] = {a: episode_cell([r for r in ef if r["arm"] == a]) for a in plan["episode_arms"]}
    summary["episodes_success_parents"] = {a: episode_cell([r for r in es if r["arm"] == a]) for a in plan["episode_arms"]}
    for label, rows in (("failed", ef), ("success", es)):
        by = defaultdict(dict)
        for r in rows:
            by[(r["main_id"], r["replicate"])][r["arm"]] = r["success"]
        pairs = {}
        for a in plan["episode_arms"]:
            if a == "vla":
                continue
            both = [v for v in by.values() if a in v and "vla" in v]
            wins = sum(1 for v in both if v[a] and not v["vla"])
            losses = sum(1 for v in both if v["vla"] and not v[a])
            pairs[a] = dict(pairs=len(both), wins=wins, losses=losses)
        summary["episodes_paired_vs_vla_" + label] = pairs
    per_parent = defaultdict(lambda: defaultdict(list))
    for r in ef:
        per_parent[r["arm"]][r["main_id"]].append(int(r["success"]))
    summary["episodes_failed_parents_rescued"] = {a: sum(1 for v in d.values() if any(v)) for a, d in per_parent.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "summary.json", summary)
    for name, rows in (("fork_branches.csv", forks), ("episodes.csv", episodes)):
        if rows:
            with (args.out / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print("== fork branches (failed parents)")
    for k, v in summary["fork_by_timing_arm"].items():
        print("  %-26s n=%3d B=%2d A=%2d | new_noise %2d (w%d/l%d) | A4 %2d (w%d/l%d) | modified %3d gates close %4d open %4d" % (
            k, v["n"], v["window"], v["original"], v["new_noise_window"], v["wins_vs_new_noise"], v["losses_vs_new_noise"],
            v["a4_window"], v["wins_vs_a4"], v["losses_vs_a4"], v["modified"], v["gate_close_steps"], v["gate_open_steps"]))
    print("  harm (success parents, window):", {a: "%d/%d (new_noise %d, A4 %d)" % (v["window"], v["n"], v["new_noise_window"], v["a4_window"])
                                                for a, v in summary["fork_harm_success_parents"].items()})
    for k, v in summary.get("fork_by_timing_vs_vla_fork", {}).items():
        print("  vs vla_fork %-22s n=%3d success=%2d control=%2d wins=%2d losses=%2d" % (k, v["n"], v["success"], v["control"], v["wins"], v["losses"]))
    for k, v in summary.get("fork_paired_vs_vla_fork", {}).items():
        print("  vs vla_fork %-26s pairs=%3d arm=%2d control=%2d wins=%2d losses=%2d" % (k, v["pairs"], v["arm_success"], v["control_success"], v["wins"], v["losses"]))
    print("== full episodes")
    for label in ("failed", "success"):
        for a, v in summary["episodes_%s_parents" % label].items():
            print("  %-8s %-22s success %2d/%2d engaged %2d modified %2d" % (label, a, v["success"], v["n"], v["engaged"], v["modified"]))
        print("  paired vs vla (%s):" % label, summary["episodes_paired_vs_vla_" + label])
    print("  failed parents rescued at least once:", summary["episodes_failed_parents_rescued"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v1-branches", type=Path, default=Path("design/repair_main_analysis_20260908/branches.csv"))
    parser.add_argument("--extra-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
