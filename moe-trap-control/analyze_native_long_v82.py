#!/usr/bin/env python3
"""Report paired v8.2 trigger and selector effects with actual native-Long outcomes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from analyze_long_continuation import cluster_interval
from analyze_v8_strength import resources
from collection_storage import atomic_json, digest, records
from v82_closed_loop import ARMS, PROTOCOL, base_arm
from collection_routes import ALL_FIELDS


def summarize(tasks):
    output = {}
    failures = sum(not t["native_success"] for t in tasks)
    lookup = {t["main_id"]: {b["arm"]: b for b in t["branches"]} for t in tasks}
    for arm in ARMS:
        values = [lookup[t["main_id"]][arm] for t in tasks]
        base = [lookup[t["main_id"]]["native"] for t in tasks]
        rescue = [t["main_id"] for t, b in zip(tasks, values) if not t["native_success"] and b["success"]]
        harm = [t["main_id"] for t, b in zip(tasks, values) if t["native_success"] and not b["success"]]
        comparison = "native" if arm == "native" else arm.split("_")[0]+"_random_"+arm.rsplit("_", 1)[-1]
        controls = [lookup[t["main_id"]][comparison] for t in tasks]
        deltas = np.array([int(a["success"])-int(b["success"]) for a, b in zip(values, controls)])
        interval = cluster_interval(deltas, [t["base_task"] for t in tasks]) if tasks else None
        total_queries = sum(t["start_query"]+b["actual_model_queries"] for t, b in zip(tasks, values))
        baseline_queries = sum(t["start_query"]+b["actual_model_queries"] for t, b in zip(tasks, base))
        pools = [p for b in values for p in b["pool_diagnostics"]]
        output[arm] = dict(parents=len(tasks), failed_parents=failures, successful_parents=len(tasks)-failures,
            rescues=len(rescue), harms=len(harm), rescued_main_ids=rescue, harmed_main_ids=harm,
            successes=sum(b["success"] for b in values), net_rescues=len(rescue)-len(harm),
            comparison=comparison, paired_wins=int((deltas > 0).sum()), paired_losses=int((deltas < 0).sum()),
            conditional_delta_pp=float(deltas.mean()*100) if tasks else None,
            cluster_ci95_pp=None if interval is None else (100*np.asarray(interval)).tolist(),
            candidate_pools=len(pools), changed_chunks=sum(b["changed_chunks"] for b in values),
            selected_below_target=sum(b["selected_below_target"] for b in values),
            selected_below_alarm=sum(b["selected_below_alarm"] for b in values),
            pool_below_target=sum(p["below_target"] == 16 for p in pools),
            initial_population_low_fraction=float(np.mean([p["original_population_below_target"]/8 for p in pools])) if pools else None,
            second_population_low_fraction=float(np.mean([p["second_population_below_target"]/8 for p in pools])) if pools else None,
            mean_selected_minus_default=float(np.mean([p["selected"]-p["default"] for p in pools])) if pools else None,
            dominant_default=dict(Counter(p["dominant_default"] for p in pools)),
            dominant_selected=dict(Counter(p["dominant_selected"] for p in pools)),
            exits=dict(Counter(b["exit_reason"] for b in values if b["exit_reason"] is not None)),
            post_exit_recurrences=sum(b["post_exit_recurrence"] for b in values),
            full_policy_model_queries=total_queries, baseline_model_queries=baseline_queries,
            query_overhead_percent=100*(total_queries/baseline_queries-1) if baseline_queries else None,
            mean_full_action_steps=float(np.mean([b["final_action_steps"] for b in values])) if values else None)
    return output



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
        pair = "native" if arm == "native" else arm.split("_")[0]+"_random_"+arm.rsplit("_", 1)[-1]
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
        flags = [m["first_v82_alarm"] >= 0 if timing == "any_alarm" else m["effective_alarm"] for m in mains]
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
            overall[arm]["paired_wins"] > overall[arm]["paired_losses"] for arm in ("iid_v8", "guided_v8", "iid_v82", "guided_v82"))),
        notes=["Original upstream LIBERO libero_10, all ten tasks, ten official initial states per task, no Pro/Plus perturbation.",
            "Reuse the prior complete audited 100-main cohort; all 14 v8.2-triggered parents received a fresh complete exact C0.",
            "All arms retain native outcomes and costs on the 86 mains without a usable first v8.2 alarm.",
            "Rescue and harm denominators use original policy outcomes, not alarm labels.",
            "Original fixed v8.2 slope and original absolute recovery margins; no fitting or hidden capture. Both moving and fixed selector targets are compared.",
            "Query costs are counterfactual single-policy deployment, excluding research-only C0 and other arms.",
            "Population confirmation means all 16 observed candidates below target for three successive executed query states; not proof of distributional convergence or task safety.",
            "The bounded selector can dispatch above-target actions when no lower candidate exists; maximum 12 recovery queries.",
            "Risk suppression and simulator action re-execution are distinct from improved task success.",
            "Intervals are descriptive bootstrap over ten task clusters, without multiplicity adjustment."])

    previous_plan = json.loads((Path(audit["run"]) / "plan.json").read_text())
    previous = json.loads(Path(previous_plan["source_audit"]).read_text())
    if previous["status"] != "passed" or digest(previous_plan["source_audit"]) != previous_plan["source_audit_sha256"]:
        raise ValueError("Prior original Long audit changed")
    previous_lookup = {t["main_id"]: t for t in previous["tasks"]}
    old_run = Path(previous_plan["source_run"])
    unchanged = []
    timing = {}
    for arm in ("iid_random_v8", "iid_v8", "guided_random_v8", "guided_v8"):
        old_arm = base_arm(arm)
        old_states = []
        for main in mains:
            prior = previous_lookup.get(main["main_id"])
            branch = next(b for b in prior["branches"] if b["arm"] == old_arm) if prior else None
            old_states.append(main["native_success"] if branch is None else branch["success"])
        current_states = [v["success"] for v in states[arm]]
        delta = np.asarray(current_states, int)-np.asarray(old_states, int)
        timing[arm] = dict(previous_v8_successes=sum(old_states), current_v82_trigger_successes=sum(current_states),
            wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
            improved_main_ids=[m["main_id"] for m, d in zip(mains, delta) if d > 0],
            worsened_main_ids=[m["main_id"] for m, d in zip(mains, delta) if d < 0])
    for task in tasks:
        prior = previous_lookup.get(task["main_id"])
        if prior is None or prior["start_query"] != task["start_query"]:
            continue
        current_event = json.loads((Path(audit["run"]) / "replays" / task["main_id"] / "result.json").read_text())["events"][0]
        prior_event = json.loads((old_run / "replays" / task["main_id"] / "result.json").read_text())["events"][0]
        for arm in ("iid_random_v8", "iid_v8", "guided_random_v8", "guided_v8"):
            current_path = Path(audit["run"]) / "events" / current_event["event_id"] / arm
            old_path = old_run / "events" / prior_event["event_id"] / base_arm(arm)
            current_rows, old_rows = list(records(current_path / "suffix")), list(records(old_path / "suffix"))
            if len(current_rows) != len(old_rows):
                raise ValueError("Unchanged trigger/selector changed suffix length")
            for current, old in zip(current_rows, old_rows):
                for key in (*ALL_FIELDS, "actions", "noise", "sim_before", "sim_after", "proprio",
                            "input_sha256", "candidate_id", "candidate_count", "selected_risk",
                            "default_risk", "executed_action_count", "success"):
                    np.testing.assert_array_equal(current[key], old[key], err_msg="Prior v8 replication: "+key)
            unchanged.append(dict(main_id=task["main_id"], arm=arm, queries=len(current_rows),
                current_manifest_sha256=digest(current_path / "suffix/manifest.json"),
                previous_manifest_sha256=digest(old_path / "suffix/manifest.json")))
    score_changes = {}
    for family in ("iid", "guided"):
        for kind in ("", "_random"):
            fixed, moving = family+kind+"_v8", family+kind+"_v82"
            delta = np.array([int(a["success"])-int(b["success"]) for a, b in zip(states[moving], states[fixed])])
            score_changes[moving] = dict(comparison=fixed, wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                improved_main_ids=[m["main_id"] for m, d in zip(mains, delta) if d > 0],
                worsened_main_ids=[m["main_id"] for m, d in zip(mains, delta) if d < 0])
    result.update(timing_only_vs_previous_v8=timing, moving_vs_fixed_selector=score_changes,
        previous_v8_unchanged_timing_replication=dict(status="passed", suffixes=len(unchanged),
            queries=sum(row["queries"] for row in unchanged), cases=unchanged),
        changed_trigger_cases=[dict(main_id=m["main_id"], first_v8=m["first_v8_alarm"],
            first_v82=m["first_v82_alarm"], native_success=m["native_success"],
            outcomes={arm: branch_lookup[m["main_id"]][arm]["success"] for arm in ARMS})
            for m in mains if m["first_v8_alarm"] != m["first_v82_alarm"]])

    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "summary.json", result)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        fields = ("arm", "mains", "successes", "rescues", "harms", "delta_vs_native_pp", "paired_wins", "paired_losses", "full_policy_queries", "query_cost_ratio")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, value in overall.items():
            writer.writerow(dict(arm=arm, **{key: value[key] for key in fields[1:]}))
    print(json.dumps(dict(overall=overall, detector=detection, convergence=confirmation, actual_dispatch=result["actual_dispatch"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
