#!/usr/bin/env python3
"""Separate overall original-Long utility, conditional recovery, and real dispatch."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from analyze_long_continuation import cluster_interval
from analyze_v8_closed_loop import summarize, plot
from analyze_v8_strength import resources
from collection_storage import atomic_json, digest, records
from v8_closed_loop import ARMS, PROTOCOL


def actual_dispatch(run, tasks):
    result = {}
    for arm in ARMS[1:]:
        changed, pools, grip_steps = 0, 0, 0
        distances, spans = [], []
        for task in tasks:
            event = json.loads((run / "replays" / task["main_id"] / "result.json").read_text())["events"][0]
            path = run / "events" / event["event_id"] / arm
            for row in records(path / "suffix"):
                if int(row["candidate_count"]) != 16:
                    continue
                candidates = list(records(path / "pools" / str(int(row["relative_query"])) / "candidates"))
                chosen, default = row["actions"], candidates[0]["actions"]
                count = int(row["executed_action_count"])
                np.testing.assert_array_equal(chosen, candidates[int(row["candidate_id"])]["actions"])
                delta = chosen[:count, :6].astype(float)-default[:count, :6].astype(float)
                changed += int(not np.array_equal(chosen[:count], default[:count]))
                grip_steps += int(np.sum((chosen[:count, 6] > 0) != (default[:count, 6] > 0)))
                pools += 1
                distances.append(float(np.sqrt(np.mean(delta**2))))
                spans.append(float(np.ptp([float(c["risk"]) for c in candidates])))
        result[arm] = dict(candidate_pools=pools, actually_changed_executed_chunks=changed,
            gripper_sign_changed_steps=grip_steps,
            command_rms_6d_mean=float(np.mean(distances)) if distances else None,
            command_rms_6d_median=float(np.median(distances)) if distances else None,
            candidate_risk_span_median=float(np.median(spans)) if spans else None,
            candidate_risk_span_mean=float(np.mean(spans)) if spans else None)
    return result


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed" or audit["protocol"] != PROTOCOL:
        raise ValueError("Complete audited original run required")
    tasks, mains = audit["tasks"], audit["cohort"]
    branch_lookup = {t["main_id"]: {b["arm"]: b for b in t["branches"]} for t in tasks}
    task_lookup = {t["main_id"]: t for t in tasks}
    conditional = summarize(tasks)
    overall, states = {}, {}
    for arm in ARMS:
        values = []
        for main in mains:
            branch = branch_lookup.get(main["main_id"], {}).get(arm)
            values.append(dict(main_id=main["main_id"], base_task=main["base_task"],
                success=main["native_success"] if branch is None else branch["success"],
                policy_queries=main["native_queries"] if branch is None else
                    task_lookup[main["main_id"]]["start_query"]+branch["actual_model_queries"]))
        states[arm] = values
    base_queries = sum(v["policy_queries"] for v in states["native"])
    failures = sum(not m["native_success"] for m in mains)
    for arm, values in states.items():
        deltas = np.array([int(v["success"])-int(m["native_success"]) for v, m in zip(values, mains)])
        pair = "native" if arm == "native" else arm.split("_")[0]+"_random"
        control = states[pair]
        paired = np.array([int(a["success"])-int(b["success"]) for a, b in zip(values, control)])
        ci = cluster_interval(deltas, [m["base_task"] for m in mains])
        queries = sum(v["policy_queries"] for v in values)
        overall[arm] = dict(mains=len(mains), successes=sum(v["success"] for v in values),
            success_rate=sum(v["success"] for v in values)/len(values),
            rescues=int((deltas > 0).sum()), harms=int((deltas < 0).sum()),
            native_failures=failures, native_successes=len(mains)-failures,
            rescued_main_ids=[v["main_id"] for v, delta in zip(values, deltas) if delta > 0],
            harmed_main_ids=[v["main_id"] for v, delta in zip(values, deltas) if delta < 0],
            delta_vs_native_pp=float(deltas.mean()*100),
            task_cluster_ci95_pp=None if ci is None else (100*np.asarray(ci)).tolist(),
            paired_comparison=pair, paired_wins=int((paired > 0).sum()), paired_losses=int((paired < 0).sum()),
            full_policy_queries=queries, query_cost_ratio=queries/base_queries)
    detection = {}
    for timing in ("any_alarm", "effective_alarm"):
        flags = [m["first_v8_alarm"] >= 0 if timing == "any_alarm" else m["effective_alarm"] for m in mains]
        tp = sum(flag and not m["native_success"] for flag, m in zip(flags, mains))
        fp = sum(flag and m["native_success"] for flag, m in zip(flags, mains))
        detection[timing] = dict(tp=tp, fp=fp, fn=failures-tp, tn=len(mains)-failures-fp,
            precision=tp/(tp+fp) if tp+fp else None, recall=tp/failures if failures else None,
            false_positive_rate=fp/(len(mains)-failures) if len(mains) > failures else None)
    confirmation = {}
    for arm in ARMS[1:]:
        values = [(t, next(b for b in t["branches"] if b["arm"] == arm)) for t in tasks]
        confirmed = [(t, b) for t, b in values if b["exit_reason"] == "population_confirmed"]
        confirmation[arm] = dict(confirmed=len(confirmed), final_failures=sum(not b["success"] for _, b in confirmed),
            originally_failed_and_still_failed=sum(not t["native_success"] and not b["success"] for t, b in confirmed),
            failed_without_recurrence=sum(not b["success"] and not b["post_exit_recurrence"] for _, b in confirmed))
    result = dict(status="completed", benchmark="original_libero_long", cohort_size=len(mains),
        audit=str(args.audit.resolve()), audit_sha256=digest(args.audit), analyzer_sha256=digest(__file__),
        overall=overall, conditional=conditional, detector=detection, convergence=confirmation,
        actual_dispatch=actual_dispatch(Path(audit["run"]), tasks),
        by_task={name: {arm: sum(v["success"] for v in values if v["base_task"] == name)
            for arm, values in states.items()} for name in sorted({m["base_task"] for m in mains})},
        trigger_heads=dict(Counter(name for t in tasks for name, value in t["trigger_heads"].items() if value)),
        resources=resources(Path(audit["run"])),
        continuation_gate=dict(passed=any(overall[arm]["rescues"] > overall[arm]["harms"] and
            overall[arm]["paired_wins"] > overall[arm]["paired_losses"] for arm in ("iid_v8", "guided_v8"))),
        notes=["Original upstream LIBERO libero_10, all ten tasks, ten official initial states per task, no Pro/Plus perturbation.",
            "All 100 fresh mains were predeclared before outcomes; all received exact complete C0 replay.",
            "Every arm retains the original outcome and original query cost when there is no usable first v8 alarm.",
            "Rescue and harm denominators use original policy outcomes, not alarm labels.",
            "Frozen alarm parameters and recovery margins match the preceding experiment; no fitting or hidden capture.",
            "Query costs are counterfactual single-policy deployment, excluding research-only C0 and other arms.",
            "Population confirmation means all 16 observed candidates below target for three successive executed query states; not proof of distributional convergence or task safety.",
            "The bounded selector can dispatch above-target actions when no lower candidate exists; maximum 12 recovery queries.",
            "Risk suppression and simulator action re-execution are distinct from improved task success.",
            "Intervals are descriptive bootstrap over ten task clusters, without multiplicity adjustment."])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "summary.json", result)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        fields = ("arm", "mains", "successes", "rescues", "harms", "delta_vs_native_pp", "paired_wins", "paired_losses", "full_policy_queries", "query_cost_ratio")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, value in overall.items():
            writer.writerow(dict(arm=arm, **{key: value[key] for key in fields[1:]}))
    if tasks:
        plot(tasks, args.output)
    print(json.dumps(dict(overall=overall, detector=detection, convergence=confirmation, actual_dispatch=result["actual_dispatch"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
