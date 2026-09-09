#!/usr/bin/env python3
"""Separate identified-controller tracking, task utility, trigger value and cost."""

import argparse
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json,digest
from fixed_recovery_control import METHODS
from recovery_mpc import ARMS
from analyze_v8_strength import resources,interval



def tracking(branches):
    rows=[b["control_diagnostics"] for b in branches if b["control_diagnostics"]]
    flat=[r for values in rows for r in values]
    valid_prediction=[r for r in flat if r["step"]>0]
    completed_lift=[v[7]["target_error_mm"] for v in rows if len(v)>=8]
    completed_retreat=[v[15]["target_error_mm"] for v in rows if len(v)>=16]
    return dict(branches=len(rows),physical_steps=len(flat),
        mean_final_target_error_mm=float(np.mean([v[-1]["target_error_mm"] for v in rows])) if rows else None,
        mean_branch_tracking_rmse_mm=float(np.mean([np.sqrt(np.mean([r["target_error_mm"]**2 for r in v])) for v in rows])) if rows else None,
        completed_lift=len(completed_lift),completed_retreat=len(completed_retreat),
        mean_completed_lift_error_mm=float(np.mean(completed_lift)) if completed_lift else None,
        mean_completed_retreat_error_mm=float(np.mean(completed_retreat)) if completed_retreat else None,
        prediction_rmse_after_first_mm=float(np.sqrt(np.mean([r["prediction_error_mm"]**2 for r in valid_prediction]))) if valid_prediction else None,
        pooled_lyapunov_increase_steps=sum(r["lyapunov_delta"]>1e-8 for r in flat),
        optimizer_fallbacks=sum(r["fallback"] for r in flat),
        optimizer_calls=sum(r["fallback"] or r["optimizer_accepted"] for r in flat),
        controller_seconds=sum(r["controller_seconds"] for r in flat),
        mean_input_norm_cm=float(np.mean([r["input_norm_cm"] for r in flat])) if flat else None)


def outcome(task,method,arm):
    selected=[e for e in task["events"] if method in e["methods"]]
    if not selected:
        return dict(main_id=task["main_id"],triggered=False,success=task["native_success"],
                    model_queries=task["native_queries"],total_action_steps=task["native_action_steps"],
                    recovery_steps=0,branch=None,event_id=None,start_query=None)
    if len(selected)!=1:
        raise ValueError("Multiple interventions for one policy")
    event=selected[0]
    branch=next(b for b in event["branches"] if b["arm"]==arm)
    return dict(main_id=task["main_id"],triggered=True,success=branch["success"],
        model_queries=event["start_query"]+branch["queries"],total_action_steps=branch["final_action_steps"],
        recovery_steps=branch["recovery_steps"],branch=branch,event_id=event["event_id"],start_query=event["start_query"])


def paired(a,b):
    values=np.asarray([int(x["success"])-int(y["success"]) for x,y in zip(a,b)],float)
    return dict(parents=len(values),wins=int((values>0).sum()),losses=int((values<0).sum()),
        delta_pp=float(values.mean()*100) if len(values) else None,parent_bootstrap_95ci_pp=interval(values),
        winning_main_ids=[x["main_id"] for x,y in zip(a,b) if x["success"] and not y["success"]],
        losing_main_ids=[x["main_id"] for x,y in zip(a,b) if not x["success"] and y["success"]])


def summary(tasks):
    methods={}
    native=[dict(main_id=t["main_id"],success=t["native_success"]) for t in tasks]
    failures=sum(not t["native_success"] for t in tasks)
    successes=len(tasks)-failures
    for method in METHODS:
        arms={}
        for arm in ARMS[1:]:
            values=[outcome(t,method,arm) for t in tasks]
            active=[v for v in values if v["triggered"]]
            branches=[v["branch"] for v in active]
            known=[b for b in branches if b["recurrence_window_available"]]
            rescues=[t["main_id"] for t,v in zip(tasks,values) if not t["native_success"] and v["success"]]
            harms=[t["main_id"] for t,v in zip(tasks,values) if t["native_success"] and not v["success"]]
            triggered_failure=sum(v["triggered"] and not t["native_success"] for t,v in zip(tasks,values))
            triggered_success=sum(v["triggered"] and t["native_success"] for t,v in zip(tasks,values))
            arms[arm]=dict(parents=len(tasks),original_failures=failures,original_successes=successes,
                successes=sum(v["success"] for v in values),rescues=len(rescues),harms=len(harms),
                rescued_main_ids=rescues,harmed_main_ids=harms,
                effective_triggers=len(active),triggered_original_failures=triggered_failure,
                triggered_original_successes=triggered_success,untriggered=len(values)-len(active),
                versus_native=paired(values,native),
                total_injected_physical_steps=sum(v["recovery_steps"] for v in values),
                model_query_delta_vs_native=sum(v["model_queries"]-t["native_queries"] for t,v in zip(tasks,values)),
                action_step_delta_vs_native=sum(v["total_action_steps"]-t["native_action_steps"] for t,v in zip(tasks,values)),
                injected_step_execution_seconds=sum(b["recovery_environment_seconds"] for b in branches),
                median_observed_recovery_displacement_m=np.median([b["recovery_displacement_m"] for b in branches],axis=0).tolist() if branches else None,
                median_lift_phase_displacement_m=np.median([b["lift_phase_displacement_m"] for b in branches],axis=0).tolist() if branches else None,
                recurrence=dict(observable=len(known),censored=len(branches)-len(known),
                    any_stall=sum(b["any_post_recovery_stall"] for b in known),
                    never_clear=sum(b["never_clear_in_observed_windows"] for b in known),
                    clear_then_stall=sum(b["clear_then_stall"] for b in known)),
                active_main_ids=[v["main_id"] for v in active],
                tracking=tracking(branches),
                controller_seconds=sum(b["controller_seconds"] for b in branches),
                optimizer_fallbacks=sum(b["optimizer_fallbacks"] for b in branches))
        arms["withdraw_vs_hold"]=paired([outcome(t,method,"withdraw") for t in tasks],
                                       [outcome(t,method,"hold") for t in tasks])
        for arm in ("mpc","mpc_disturbance"):
            arms[arm+"_vs_withdraw"]=paired([outcome(t,method,arm) for t in tasks],
                                          [outcome(t,method,"withdraw") for t in tasks])
        methods[method]=arms
    comparisons={}
    for arm in ("withdraw","mpc","mpc_disturbance"):
        comparisons[arm]={}
        for reference in ("behavior_stall","random_time","v7_frozen","knn20","knn_euclidean_OR_cosine"):
            a=[outcome(t,"v8_frozen",arm) for t in tasks]
            b=[outcome(t,reference,arm) for t in tasks]
            both=[(x,y) for x,y in zip(a,b) if x["triggered"] and y["triggered"]]
            comparisons[arm][reference]=dict(full_cohort=paired(a,b),
                both_triggered=paired([x for x,y in both],[y for x,y in both]),
                exact_same_event_parents=sum(x["triggered"] and x["event_id"]==y["event_id"] for x,y in zip(a,b)))
    mechanical={}
    for arm in ARMS[1:]:
        parent_values=[]
        for task in tasks:
            branches=[next(b for b in e["branches"] if b["arm"]==arm) for e in task["events"]]
            value=tracking(branches)
            if value["branches"]:
                parent_values.append(dict(main_id=task["main_id"],**value))
        mechanical[arm]=dict(parents=len(parent_values),parent_values=parent_values,
            mean_parent_final_target_error_mm=float(np.mean([v["mean_final_target_error_mm"] for v in parent_values])) if parent_values else None,
            mean_parent_tracking_rmse_mm=float(np.mean([v["mean_branch_tracking_rmse_mm"] for v in parent_values])) if parent_values else None,
            physical_steps=sum(v["physical_steps"] for v in parent_values),
            optimizer_fallbacks=sum(v["optimizer_fallbacks"] for v in parent_values))
    return dict(parents=len(tasks),original_failures=failures,original_successes=successes,
        methods=methods,v8_controller_comparisons=comparisons,mechanistic=mechanical)


def run(args):
    audit=json.loads(args.audit.read_text())
    if audit["status"]!="passed":
        raise ValueError("Passed audit required")
    tasks=audit["tasks"]
    primary=[t for t in tasks if t["analysis_role"]=="perturbation"]
    report=dict(status="completed",audit=str(args.audit.resolve()),audit_sha256=digest(args.audit),
        analyzer_sha256=digest(__file__),stage="state_disjoint_identified_control_pilot",
        estimand="paired full-task success change between controllers within one trigger policy, then between triggers with one fixed controller; all selected parents retained",
        unit="one original main_id, one original policy-noise stream; shared events are not independent evidence",
        scope="24 fixed-hash prior Long mains, two per category, disjoint from all 36 motion-identification development mains; historical cohort previously explored, not a globally blind or official benchmark evaluation",
        coverage="one maximum intervention per policy; actual trigger coverage differs and is reported, not claimed budget-matched",
        control="same targets, 8+8 steps, <=1cm Cartesian command norm, gripper sign and original520 budget; P vs constrained MPC vs disturbance-compensated MPC; hold/native controls",
        randomization="policy RNG follows original query index; original C0 environment RNG tape shared by absolute action step; deterministic common extension after native termination",
        recurrence_definition="6 post-recovery query positions have diameter <=5mm; clear-then-stall requires an observed non-stall full window before a later stall window; no full window is censored",
        latched_alarm_caveat="persistent alarm flags are not new trap events; recurrence uses the separately specified behavior proxy",
        timing_caveat="all triggers act after executing the triggering native chunk, before q+1; comparison includes coverage and timing, not timing alone",
        uncertainty="descriptive paired main-level bootstrap; degenerate differences yield null, not equivalence; one pilot stream per original main",
        all=summary(tasks),primary=summary(primary),
        by_benchmark={b:summary([t for t in primary if t["benchmark"]==b]) for b in ("pro","plus")},
        dynamics_model=json.loads((Path(audit["run"])/"dynamics_model.json").read_text()),
        resources=resources(Path(audit["run"])),actual_model_queries=audit["actual_model_queries"],
        c0_queries=audit["c0_queries"],events=audit["events"],branches=audit["branches"],new_main_coverage=0,
        paired_recovery_labels=[dict(main_id=t["main_id"],benchmark=t["benchmark"],category=t["category"],
            native_success=t["native_success"],event_id=e["event_id"],start_query=e["start_query"],methods=e["methods"],
            outcomes={b["arm"]:b["success"] for b in e["branches"]}) for t in tasks for e in t["events"]])
    atomic_json(args.output,report)
    print(json.dumps(dict(parents=len(tasks),primary_parents=len(primary),
        methods={m:{a:{k:r[k] for k in ("successes","rescues","harms","effective_triggers")}
                    for a,r in arms.items() if a in ARMS[1:]} for m,arms in report["all"]["methods"].items()})))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
