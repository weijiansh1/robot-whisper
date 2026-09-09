#!/usr/bin/env python3
"""Report independent online outcomes, matched assignment controls, and MoE diagnostics."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from analyze_control_bank import telemetry
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from moe_outcome_models import FAMILIES, MODELS, OPERATORS, FEATURE_NAMES, FeatureHistory, cluster, density_value, project


def landmarks(rows):
    from sklearn.metrics import roc_auc_score
    result = []
    for query in (8,16,24,32,40,"tail"):
        selected = [r for r in rows if r["query"] == query]
        y = [r["success"] for r in selected]
        result.append(dict(query=query,mains=len(y),successes=sum(y),failures=len(y)-sum(y),
            auc={name:float(roc_auc_score(y,[r[name] for r in selected])) if len(set(y))==2 else None
                 for name in ("density","markov")}))
    return result


def feature_contrasts(rows):
    result = []
    names = FEATURE_NAMES+("mean_entropy","mean_denoise_flow","mean_query_mobility","effective_experts")
    for query in (8,16,24,32,40,"tail"):
        selected = [row for row in rows if row["query"] == query]
        x = np.asarray([row["feature"] for row in selected])
        labels = np.asarray([row["success"] for row in selected],bool)
        tasks = np.asarray([row["task"] for row in selected])
        x = np.column_stack((x,x[:,20:28].mean(1),x[:,28:36].mean(1),x[:,36:44].mean(1),
                             1./np.square(x[:,44:76]).sum(1)))
        both = [task for task in sorted(set(tasks)) if len(set(labels[tasks==task])) == 2]
        effects = []
        if labels.any() and (~labels).any():
            difference = x[labels].mean(0)-x[~labels].mean(0)
            standard = difference/np.maximum(x.std(0),1e-6)
            paired = np.asarray([x[(tasks==task)&labels].mean(0)-x[(tasks==task)&~labels].mean(0) for task in both])
            for index,name in enumerate(names):
                effects.append(dict(feature=name,success_mean=float(x[labels,index].mean()),
                    failure_mean=float(x[~labels,index].mean()),standardized_difference=float(standard[index]),
                    task_balanced_mean_difference=float(paired[:,index].mean()) if both else None,
                    task_differences={task:float(paired[j,index]) for j,task in enumerate(both)}))
        result.append(dict(query=query,mains=len(selected),successes=int(labels.sum()),
            failures=int((~labels).sum()),tasks_with_both_classes=len(both),features=effects))
    return result


def prior_feature_rows(path,expected):
    if digest(path) != expected:
        raise ValueError("Frozen training feature cache changed")
    rows = []
    with np.load(path,allow_pickle=False) as data:
        for parent,task in enumerate(data["main_tasks"]):
            indices = np.flatnonzero(data["frame_parent"]==parent)
            for query in (8,16,24,32,40):
                matched = indices[data["frame_q"][indices]==query]
                if len(matched):
                    rows.append(dict(task=str(task),query=query,success=int(data["main_success"][parent]),
                        feature=data["frame_x"][matched[0]]))
            rows.append(dict(task=str(task),query="tail",success=int(data["main_success"][parent]),
                feature=data["frame_x"][indices[-3:]].mean(0)))
    return rows


def alarm_metrics(cohort):
    result = []
    for scope in ("ever_triggered","usable_next_query"):
        flags = [task["first_alarms"]["v82_frozen"]>=0 if scope=="ever_triggered" else bool(task["events"])
                 for task in cohort]
        tp = sum(flag and not task["native_success"] for flag,task in zip(flags,cohort))
        fp = sum(flag and task["native_success"] for flag,task in zip(flags,cohort))
        failures = sum(not task["native_success"] for task in cohort)
        successes = len(cohort)-failures
        result.append(dict(scope=scope,true_positive=tp,false_positive=fp,false_negative=failures-tp,
            true_negative=successes-fp,precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/failures if failures else None,false_positive_rate=fp/successes if successes else None))
    return result


def main(args):
    audit,physical = json.loads(args.audit.read_text()),json.loads(args.physical.read_text())
    if audit["status"] != "passed" or physical["status"] != "passed" or audit["plan_sha256"] != physical["plan_sha256"]:
        raise ValueError("Independent audits must complete")
    run = Path(audit["run"])
    plan,summary = (json.loads((run/name).read_text()) for name in ("plan.json","summary.json"))
    fitted = json.loads(Path(plan["outcome_model_path"]).read_text())
    model = fitted["model"]
    fresh_summary = json.loads((Path(plan["fresh_run"])/"summary.json").read_text())
    rows = [r for t in audit["tasks"] for r in t["branches"]]
    lookup = {(r["main_id"],r["arm"]):r for r in rows}
    baseline = sum(t["native_success"] for t in plan["cohort"])
    prefix = sum(t["events"][0]["start_query"] if t["events"] else t["parent_queries"] for t in plan["cohort"])
    methods = []
    for family in FAMILIES:
        chosen = [r for r in rows if r["family"] == family]
        arms = [next(r for r in audit["arms"] if r["arm"] == family+"_r"+str(rep)) for rep in (0,1)]
        comparisons = {}
        for control in ("resample",)+(("shuffle_"+family,) if family in MODELS else ()):
            pairs = []
            for rep in (0,1):
                actual = [r for r in chosen if r["repeat"] == rep]
                reference = [lookup[r["main_id"],control+"_r"+str(rep)] for r in actual]
                pairs.append(dict(repeat=rep,parents=len(actual),
                    wins=sum(r["success"] and not ref["success"] for r,ref in zip(actual,reference)),
                    losses=sum(ref["success"] and not r["success"] for r,ref in zip(actual,reference))))
            comparisons[control] = pairs
        methods.append(dict(family=family,success_counts=[r["full_cohort_success"] for r in arms],
            mean_success_rate=float(np.mean([r["full_cohort_success"] for r in arms])/len(plan["cohort"])),
            rescues=[r["rescued"] for r in arms],harms=[r["harmed"] for r in arms],paired=comparisons,
            full_policy_queries=[prefix+r["actual_branch_model_queries"] for r in arms],
            physical_steps=[sum(r["physical_steps"] for r in chosen if r["repeat"]==rep) for rep in (0,1)],
            operators=[dict(Counter(r["operator"] for r in chosen if r["repeat"]==rep)) for rep in (0,1)]))
    prediction_rows,feature_rows,prefixes = [],[],{}
    for task in plan["cohort"]:
        history,values = FeatureHistory(),[]
        q0 = task["events"][0]["start_query"] if task["events"] else -1
        for row in records(Path(task["parent_directory"])/"main"):
            if int(row["query"]) == q0:
                prefixes[task["main_id"]] = copy.deepcopy(history)
            feature = history.update(row[PROBS_KEY],int(row["action_steps_before"]))
            z = project(model,feature)
            values.append(dict(main_id=task["main_id"],query=int(row["query"]),success=int(task["native_success"]),
                density=float(density_value(model,z)[0]),markov=float(np.asarray(model["committor"])[cluster(model,z)[0]]),
                feature=feature))
        prediction_rows.extend({key:value for key,value in r.items() if key!="feature"}
            for r in values if r["query"] in (8,16,24,32,40))
        feature_rows.extend(dict(task=task["base_task"],query=r["query"],success=r["success"],feature=r["feature"])
            for r in values if r["query"] in (8,16,24,32,40))
        feature_rows.append(dict(task=task["base_task"],query="tail",success=int(task["native_success"]),
            feature=np.mean([r["feature"] for r in values[-3:]],axis=0)))
        prediction_rows.append(dict(main_id=task["main_id"],query="tail",success=int(task["native_success"]),
            **{key:float(np.mean([r[key] for r in values[-3:]])) for key in ("density","markov")}))
    responses,seen = [],set()
    for task in plan["tasks"]:
        event = task["events"][0]
        for arm in plan["arms"]:
            row = lookup[task["main_id"],arm]
            key = (task["main_id"],row["operator"],row["repeat"])
            if arm == "native" or key in seen:
                continue
            seen.add(key)
            directory = run/"events"/event["event_id"]/arm
            branch = json.loads((directory/"branch.json").read_text())
            history = copy.deepcopy(prefixes[task["main_id"]])
            for actual in records(directory/"suffix"):
                feature = history.update(actual[PROBS_KEY],int(actual["action_steps_before"]))
                elapsed = int(actual["action_steps_before"])-branch["initial_action_steps"]
                if elapsed >= 32:
                    start = project(model,task["entry_feature"])
                    observed = project(model,feature)
                    predicted = np.clip(np.asarray(model["response_coefficients"])[OPERATORS.index(row["operator"])] @ np.r_[1.,start],-6.,6.)
                    responses.append(dict(main_id=task["main_id"],operator=row["operator"],repeat=row["repeat"],
                        elapsed_steps=elapsed,success=row["success"],latent_rmse=float(np.sqrt(np.mean((predicted-observed)**2))),
                        identity_latent_rmse=float(np.sqrt(np.mean((start-observed)**2))),
                        predicted_density_change=float(density_value(model,predicted)[0]-density_value(model,start)[0]),
                        actual_density_change=float(density_value(model,observed)[0]-density_value(model,start)[0])))
                    break
    response_pairs = []
    response_lookup = {(r["main_id"],r["operator"],r["repeat"]):r for r in responses}
    for row in responses:
        reference = response_lookup.get((row["main_id"],"resample",row["repeat"]))
        if row["operator"] == "resample" or reference is None:
            continue
        response_pairs.append(dict(main_id=row["main_id"],operator=row["operator"],repeat=row["repeat"],
            elapsed_steps=row["elapsed_steps"],reference_elapsed_steps=reference["elapsed_steps"],
            predicted_gain=row["predicted_density_change"]-reference["predicted_density_change"],
            observed_gain=row["actual_density_change"]-reference["actual_density_change"],
            success_difference=int(row["success"])-int(reference["success"])))
    response_parents = []
    for parent in sorted({r["main_id"] for r in responses}):
        selected = [r for r in responses if r["main_id"]==parent]
        response_parents.append(dict(main_id=parent,responses=len(selected),
            latent_rmse=float(np.mean([r["latent_rmse"] for r in selected])),
            identity_latent_rmse=float(np.mean([r["identity_latent_rmse"] for r in selected]))))
    changed = []
    for task in plan["tasks"]:
        these = [r for r in rows if r["main_id"] == task["main_id"] and r["family"] != "native"]
        rescues,harms = [r for r in these if r["rescued"]],[r for r in these if r["harmed"]]
        if rescues or harms:
            changed.append(dict(main_id=task["main_id"],base_task=task["base_task"],init_index=task["init_index"],
                original_success=task["native_success"],entry_mode=these[0]["entry_mode"],
                rescued_arms=[r["arm"] for r in rescues],harmed_arms=[r["arm"] for r in harms],
                resample_success=[lookup[task["main_id"],"resample_r"+str(rep)]["success"] for rep in (0,1)]))
    result = dict(status="passed",plan_sha256=digest(run/"plan.json"),run=str(run),
        audit_sha256=digest(args.audit),physical_sha256=digest(args.physical),analyzer_sha256=digest(__file__),
        fitted_model_sha256=digest(plan["outcome_model_path"]),mains=len(plan["cohort"]),baseline_success=baseline,
        alarm_metrics=alarm_metrics(plan["cohort"]),
        alarms=len(plan["tasks"]),alarmed_failures=sum(not t["native_success"] for t in plan["tasks"]),
        alarmed_successes=sum(t["native_success"] for t in plan["tasks"]),methods=methods,changed_cases=changed,
        unique_rescued_parents=len({r["main_id"] for r in rows if r["rescued"]}),
        fresh_landmark_metrics=landmarks(prediction_rows),fresh_prediction_rows=prediction_rows,
        fresh_feature_contrasts=feature_contrasts(feature_rows),
        prior_feature_contrasts=feature_contrasts(prior_feature_rows(Path(plan["outcome_model_path"]).with_suffix(".npz"),fitted["data_sha256"])),
        prior_diagnostics=fitted["diagnostics"],response_validation=responses,
        paired_response_validation=response_pairs,response_parent_metrics=response_parents,
        response_latent_rmse=float(np.mean([r["latent_rmse"] for r in responses])) if responses else None,
        response_identity_latent_rmse=float(np.mean([r["identity_latent_rmse"] for r in responses])) if responses else None,
        response_parent_mean_rmse=float(np.mean([r["latent_rmse"] for r in response_parents])) if response_parents else None,
        response_parent_mean_identity_rmse=float(np.mean([r["identity_latent_rmse"] for r in response_parents])) if response_parents else None,
        fresh_main_and_c0_queries=fresh_summary["actual_model_queries"],branch_phase_queries=summary["actual_model_queries"],
        total_model_queries=fresh_summary["actual_model_queries"]+summary["actual_model_queries"],
        collection_seconds=fresh_summary["collection_elapsed_seconds"]+summary["collection_elapsed_seconds"],
        branch_queries_per_second=summary["queries_per_second"],gpu=telemetry(run),
        actual_recovery_steps=sum(r["physical_steps"] for r in rows),
        transformed_commands=sum(r["changed_commands"] for r in rows),
        verified_suffixes=physical["suffixes"],verified_steps=physical["env_step_calls"],maximum_state_error=physical["maximum_state_error"])
    atomic_json(args.output,result)
    plot(args.output,result)
    print(json.dumps({key:result[key] for key in ("status","mains","baseline_success","alarms","unique_rescued_parents","total_model_queries")}))
    print(json.dumps(methods))


def plot(output,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axis = plt.subplots(figsize=(10,6),constrained_layout=True)
    gain = [(np.mean(r["success_counts"])-result["baseline_success"])/result["mains"]*100 for r in result["methods"]]
    y = np.arange(len(FAMILIES))
    axis.barh(y,gain,color=["#278363" if v>0 else "#b34c5d" if v<0 else "#777777" for v in gain],height=.6)
    for rep,marker in ((0,"o"),(1,"x")):
        axis.scatter([(r["success_counts"][rep]-result["baseline_success"])/result["mains"]*100 for r in result["methods"]],
            y+(-.09 if rep==0 else .09),marker=marker,color="#222222",label="repeat "+str(rep),zorder=3)
    axis.set_yticks(y,FAMILIES);axis.invert_yaxis();axis.axvline(0,color="#555555",linewidth=.8)
    axis.set_xlim(min(-.12,min(gain)-.12),max(1.12,max(gain)+.12))
    axis.set_xlabel("Success-rate change from original (percentage points)")
    axis.set_title("Frozen MoE outcome models: "+str(result["mains"])+" fresh native Long mains\n"
        "Original successes: "+str(result["baseline_success"])+"; bars average two repeats")
    axis.grid(axis="x",alpha=.2)
    fig.legend(*axis.get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    fig.savefig(output.with_suffix(".png"),dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for axis,source,title in ((axes[0],result["prior_diagnostics"]["landmark_metrics"],"Prior 150: task-isolated validation"),
                              (axes[1],result["fresh_landmark_metrics"],"Fresh 100: frozen-model evaluation")):
        selected = [r for r in source if r["query"] != "tail"]
        for name,color in (("density","#287766"),("markov","#b25162")):
            axis.plot([r["query"] for r in selected],[r["auc"][name] for r in selected],"o-",color=color,label=name)
        for row in selected:
            axis.text(row["query"],1.025,str(row["successes"])+"/"+str(row["failures"]),ha="center",fontsize=8)
        axis.axhline(.5,color="#777777",linestyle=":");axis.set_ylim(0,1.08)
        axis.set_title(title);axis.set_xlabel("Executed query (labels: success/failure mains still running)")
        axis.set_ylabel("AUC for eventual success");axis.legend();axis.grid(alpha=.15)
    fig.savefig(output.with_name(output.stem+"_landmarks.png"),dpi=180);plt.close(fig)
    from matplotlib.lines import Line2D
    for case in result["changed_cases"]:
        selected = [r for r in result["response_validation"] if r["main_id"]==case["main_id"]]
        if not selected:
            continue
        operators = [op for op in OPERATORS if any(r["operator"]==op for r in selected)]
        fig,axes = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True,constrained_layout=True)
        for repeat,axis in enumerate(axes):
            for y,operator in enumerate(operators):
                if not any(r["operator"]==operator and r["repeat"]==repeat for r in selected):
                    axis.text(.03,y,"Not assigned",transform=axis.get_yaxis_transform(),fontsize=9,color="#777777",va="center")
            for row in (r for r in selected if r["repeat"]==repeat):
                y = operators.index(row["operator"])
                color,marker = ("#278363","o") if row["success"] else ("#b34c5d","x")
                axis.hlines(y,0,row["actual_density_change"],color=color,linewidth=2)
                axis.scatter(row["actual_density_change"],y,color=color,marker=marker,s=65,zorder=3)
            axis.axvline(0,color="#777777",linewidth=.8)
            axis.set_yticks(np.arange(len(operators)),operators)
            axis.set_xlabel("Observed change in MoE log-density ratio")
            axis.set_title("Paired noise repeat "+str(repeat));axis.grid(axis="x",alpha=.15)
        axes[0].invert_yaxis()
        fig.legend(handles=[Line2D([],[],marker="o",color="#278363",linestyle="none",label="Final success"),
                            Line2D([],[],marker="x",color="#b34c5d",linestyle="none",label="Final failure")],
                   loc="outside lower center",ncol=2,frameon=False)
        fig.suptitle("MoE density improvement and actual outcomes at one alarm state\n"
            "Moka pots, initialization "+str(case["init_index"])+"; first query at least 32 steps after intervention",fontsize=12)
        fig.savefig(output.with_name(output.stem+"_case_"+case["main_id"][:6]+".png"),dpi=180);plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--physical",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
