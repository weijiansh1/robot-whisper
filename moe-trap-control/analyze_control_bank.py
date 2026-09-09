#!/usr/bin/env python3
"""Compare fixed controllers across separate exploratory and fresh native cohorts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest
from control_bank import FAMILIES, PHYSICAL


def telemetry(run):
    samples = json.loads((run / "gpu_samples.json").read_text())
    result = []
    for gpu in (0,1,2,3):
        values = [(sample,next(r for r in sample["gpus"] if r["gpu"] == gpu)) for sample in samples]
        full = [row for sample,row in values if sample["active_tasks_by_gpu"][str(gpu)] == 8]
        result.append(dict(gpu=gpu,all_samples=len(values),eight_worker_samples=len(full),
            mean_utilization=float(np.mean([r["utilization_percent"] for _,r in values])),
            eight_worker_mean_utilization=float(np.mean([r["utilization_percent"] for r in full])) if full else None,
            eight_worker_mean_power_w=float(np.mean([r["power_w"] for r in full])) if full else None,
            peak_power_w=max(r["power_w"] for _,r in values)))
    return result


def analyze(args):
    audit, physical = json.loads(args.audit.read_text()),json.loads(args.physical.read_text())
    if audit["status"] != "passed" or physical["status"] != "passed" or audit["plan_sha256"] != physical["plan_sha256"]:
        raise ValueError("Independent audit and complete physical re-execution required")
    run = Path(audit["run"])
    plan, summary = (json.loads((run / name).read_text()) for name in ("plan.json","summary.json"))
    fresh_summary = json.loads((Path(plan["fresh_run"]) / "summary.json").read_text())
    rows = [row for task in audit["tasks"] for row in task["branches"]]
    lookup = {(r["main_id"],r["arm"]):r for r in rows}
    groups, methods, rescues, harms, tracking = [], [], [], [], []
    for group in plan["cohorts"]:
        cohort = [t for t in plan["cohort"] if t["cohort_group"] == group]
        selected_rows = [r for r in rows if r["cohort_group"] == group]
        baseline = sum(t["native_success"] for t in cohort)
        alarms = [t for t in cohort if t["events"]]
        base_queries = sum(t["parent_queries"] for t in cohort)
        prefix_queries = sum(t["events"][0]["start_query"] if t["events"] else t["parent_queries"] for t in cohort)
        groups.append(dict(group=group,mains=len(cohort),baseline_success=baseline,baseline_success_rate=baseline/len(cohort),
            alarms=len(alarms),alarmed_original_failures=sum(not t["native_success"] for t in alarms),
            alarmed_original_successes=sum(t["native_success"] for t in alarms),baseline_queries=base_queries,
            any_operator_rescued_parents=len({r["main_id"] for r in selected_rows if r["rescued"]}),
            entry_modes=dict(Counter(r["entry_mode"] for r in selected_rows if r["family"] == "native"))))
        for family in FAMILIES:
            selected = [r for r in selected_rows if r["family"] == family]
            arms = [next(a for a in audit["arms"] if a["cohort_group"] == group and a["arm"] == family+"_r"+str(rep)) for rep in (0,1)]
            paired = {}
            for control in ("resample","random_switch"):
                wins = losses = 0
                repeated_pairs = []
                for repeat in (0,1):
                    these = [row for row in selected if row["repeat"] == repeat]
                    refs = [lookup[row["main_id"],control+"_r"+str(repeat)] for row in these]
                    repeated_pairs.append(dict(repeat=repeat,parents=len(these),
                        wins=sum(row["success"] and not ref["success"] for row,ref in zip(these,refs)),
                        losses=sum(ref["success"] and not row["success"] for row,ref in zip(these,refs))))
                for row in selected:
                    ref = lookup[row["main_id"],control+"_r"+str(row["repeat"])]
                    wins += int(row["success"] and not ref["success"])
                    losses += int(ref["success"] and not row["success"])
                paired[control] = dict(wins=wins,losses=losses,pairs=len(selected),parents=len(alarms),
                    repeats=repeated_pairs)
            counts = [arm["full_cohort_success"] for arm in arms]
            methods.append(dict(cohort_group=group,family=family,cohort_size=len(cohort),
                success_counts=counts,mean_success_rate=float(np.mean(counts)/len(cohort)),
                mean_gain_percentage_points=float((np.mean(counts)-baseline)/len(cohort)*100),
                rescues=[arm["rescued"] for arm in arms],harms=[arm["harmed"] for arm in arms],
                full_policy_queries=[prefix_queries+arm["actual_branch_model_queries"] for arm in arms],
                paired_controls=paired,selected_operators=dict(Counter(r["operator"] for r in selected))))
        for main_id in sorted({r["main_id"] for r in selected_rows if r["rescued"]}):
            these = [r for r in selected_rows if r["main_id"] == main_id]
            successes = [r for r in these if r["rescued"]]
            rescues.append(dict(cohort_group=group,main_id=main_id,base_task=these[0]["base_task"],entry_mode=these[0]["entry_mode"],
                successful_arms=[r["arm"] for r in successes],
                resample_success=[lookup[main_id,"resample_r"+str(rep)]["success"] for rep in (0,1)],
                improved_over_paired_resample=[r["arm"] for r in successes if not lookup[main_id,"resample_r"+str(r["repeat"])]["success"]]))
        for main_id in sorted({r["main_id"] for r in selected_rows if r["harmed"]}):
            these = [r for r in selected_rows if r["main_id"] == main_id]
            harms.append(dict(cohort_group=group,main_id=main_id,base_task=these[0]["base_task"],
                entry_mode=these[0]["entry_mode"],harmed_arms=[r["arm"] for r in these if r["harmed"]],
                resample_success=[lookup[main_id,"resample_r"+str(rep)]["success"] for rep in (0,1)]))
    for operator in PHYSICAL:
        selected = [r for r in rows if r["family"] == operator]
        steps = sum(r["physical_steps"] for r in selected)
        tracking.append(dict(operator=operator,physical_steps=steps,
            fraction_target_error_nonincreasing=sum(r["target_error_nonincreasing_steps"] for r in selected)/steps if steps else None,
            mean_nominal_prediction_error_m=sum(r["mean_nominal_prediction_error_m"]*r["physical_steps"] for r in selected if r["physical_steps"])/steps if steps else None))
    result = dict(status="passed",run=str(run),audit_sha256=digest(args.audit),physical_sha256=digest(args.physical),
        analyzer_sha256=digest(__file__),groups=groups,methods=methods,rescue_cases=rescues,harm_cases=harms,
        physical_tracking=tracking,mode_responses=audit["mode_responses"],
        bank_model_queries=summary["actual_model_queries"],fresh_main_and_c0_queries=fresh_summary["actual_model_queries"],
        total_model_queries=summary["actual_model_queries"]+fresh_summary["actual_model_queries"],
        bank_collection_seconds=summary["collection_elapsed_seconds"],
        fresh_collection_seconds=fresh_summary["collection_elapsed_seconds"],
        bank_queries_per_second=summary["queries_per_second"],gpu=telemetry(run),
        reexecuted_suffixes=physical["suffixes"],reexecuted_steps=physical["env_step_calls"],
        maximum_state_error=physical["maximum_state_error"],
        inference="All 15 families were compared prospectively on fresh initializations; selecting the observed winner needs another independent confirmation")
    atomic_json(args.output,result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,7),sharey=True,constrained_layout=True)
    for axis,group in zip(axes,groups):
        values = [next(m for m in methods if m["cohort_group"] == group["group"] and m["family"] == family) for family in FAMILIES]
        gain = np.array([m["mean_gain_percentage_points"] for m in values])
        y = np.arange(len(values))
        axis.barh(y,gain,color=["#22865c" if value > 0 else "#bc5663" if value < 0 else "#888888" for value in gain],height=.58)
        for repeat,marker in ((0,"o"),(1,"x")):
            individual = [(m["success_counts"][repeat]-group["baseline_success"])/group["mains"]*100 for m in values]
            axis.scatter(individual,y,marker=marker,color="#252525",s=24,label="repeat "+str(repeat),zorder=3)
        axis.axvline(0,color="#454545",linewidth=.8)
        axis.set_yticks(y,FAMILIES)
        axis.set_xlabel("Success-rate change from original (percentage points)")
        axis.set_xlim(-2.2,2.2)
        axis.set_title(group["group"].capitalize()+" cohort: "+str(group["mains"])+" mains\nOriginal success: "+str(group["baseline_success"])+"/"+str(group["mains"]))
        axis.grid(axis="x",alpha=.2)
        axis.legend(loc="lower right",fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Native Long: frozen controller bank, same total 520-step budget\n"
        "Bars: mean of two repeats; markers: individual repeats",fontsize=13)
    fig.savefig(args.output.with_suffix(".png"),dpi=180)
    plt.close(fig)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("response",["#b74655","#f5f5f5","#268466"])
    fig,axes = plt.subplots(2,1,figsize=(13,10),constrained_layout=True,
        gridspec_kw={"height_ratios":[group["alarms"] for group in groups]})
    for axis,group in zip(axes,groups):
        parents = [task for task in audit["tasks"] if task["cohort_group"] == group["group"]]
        matrix = np.array([[sum(int(lookup[t["main_id"],family+"_r"+str(rep)]["success"])-
            int(lookup[t["main_id"],family+"_r"+str(rep)]["original_success"]) for rep in (0,1))/2
            for family in FAMILIES] for t in parents])
        axis.imshow(matrix,cmap=cmap,vmin=-1,vmax=1,aspect="auto")
        labels = [task["main_id"][:6]+"  "+task["branches"][0]["entry_mode"]+"  (original "+
            ("success" if task["branches"][0]["original_success"] else "failure")+")" for task in parents]
        axis.set_yticks(np.arange(len(parents)),labels,fontsize=8)
        axis.set_xticks(np.arange(len(FAMILIES)),FAMILIES,rotation=40,ha="right",fontsize=8)
        for y,x in zip(*np.nonzero(matrix)):
            axis.text(x,y,("+" if matrix[y,x]>0 else "-")+str(int(abs(matrix[y,x])*2))+"/2",
                ha="center",va="center",fontsize=8,color="white" if abs(matrix[y,x]) == 1 else "#222222")
        axis.set_title(group["group"].capitalize()+": all "+str(len(parents))+" alarm states",fontsize=11)
    fig.suptitle("Intervention response by original state and entry signal pattern\n"
        "Green: rescued repeats; red: harmed repeats; blank: unchanged final outcome",fontsize=12)
    fig.savefig(args.output.with_name(args.output.stem+"_responses.png"),dpi=180)
    plt.close(fig)
    print(json.dumps({key:result[key] for key in ("status","groups","total_model_queries","reexecuted_suffixes","reexecuted_steps")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--physical",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    analyze(parser.parse_args())
