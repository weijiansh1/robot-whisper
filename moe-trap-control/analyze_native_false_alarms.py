#!/usr/bin/env python3
"""Analyze natural recovery after frozen v8.2 alarms using audited native Long traces."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, islice
import json
from pathlib import Path
import platform

import numpy as np

from collection_routes import PROBS_KEY, NATIVE_IDS_KEY, AS_FIELDS, HB_LAYERS, AS_LAYERS
from collection_storage import atomic_json, digest, records
from mode_control import active_mask, thresholds_at
from v8_closed_loop import limits, score_status
from v82_closed_loop import V82Monitor

PROTOCOL = "moe_control.native_false_alarm_dynamics.v1"
INPUTS = (("prior", Path("/tmp/himoe_control_bank_20260909"), Path("design/control_bank_audit_20260909.json")),
          ("recent", Path("/tmp/himoe_moe_outcomes_20260909"), Path("design/moe_outcomes_audit_20260909.json")))
LETTERS = "FAPIC"
SCALARS = tuple("r_"+letter for letter in LETTERS)+(
    "risk", "maximum_component", "hb_mobility_front", "hb_mobility_back",
    "hb_state_mobility", "hb_denoise_front", "hb_denoise_back", "hb_top4_turnover",
    "as_mobility", "as_top1_turnover", "hb_distance_from_alarm", "as_distance_from_alarm",
    "hb_return_lag3", "hb_escape_efficiency")
LEARNING_BASE = tuple("r_"+letter for letter in LETTERS)
LEARNING_DYNAMIC = ("risk", "maximum_component", "hb_mobility_back", "hb_top4_turnover",
    "as_mobility", "as_top1_turnover", "hb_distance_from_alarm", "hb_return_lag3", "hb_escape_efficiency")


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def roots(probabilities):
    p = np.maximum(np.asarray(probabilities, np.float64), 0.)
    return np.sqrt(p/p.sum(-1, keepdims=True))


def distance(left, right):
    return np.linalg.norm(left-right, axis=-1)/np.sqrt(2.)


def turnover(left, right):
    shared = (left[..., :, None] == right[..., None, :]).any(-1).sum(-1)
    return 1.-shared/left.shape[-1]


class TraceFeatures:
    def __init__(self, alarm):
        self.alarm = alarm
        self.monitor = V82Monitor()
        self.base, self.margins = limits()
        self.previous = None
        self.anchor = None
        self.hb_history = []
        self.path_length = 0.

    def update(self, row):
        status = self.monitor.update(row[PROBS_KEY])
        query = int(row["query"])
        if status["query"] != query:
            raise ValueError("Noncausal or discontinuous query history")
        score = score_status(status)
        if "component_scores" in row:
            np.testing.assert_allclose(score, row["component_scores"], rtol=2e-5, atol=2e-6)
        tau = thresholds_at(query, self.base)
        normalized = (score-tau)/self.margins
        mask = active_mask(score, tau)
        hb, as_root = roots(row[PROBS_KEY]), roots(row[AS_FIELDS[0]])
        ids, as_ids = row[NATIVE_IDS_KEY], row[AS_FIELDS[1]]
        action = hb[:, :, 1:]
        flow = distance(action[:, 1:], action[:, :-1]).mean(axis=(1, 2))
        result = {"r_"+letter: finite(value) for letter, value in zip(LETTERS, normalized)}
        if np.isfinite(normalized).all():
            risk = max(normalized[0], min(normalized[1], normalized[2]), *normalized[3:])
            result.update(risk=float(risk), maximum_component=float(normalized.max()))
        else:
            result.update(risk=None, maximum_component=None)
        result.update(query=query, steps_before=int(row["action_steps_before"]),
            executed_actions=int(row["executed_action_count"]), success_after=bool(row["success"]),
            mode="+".join(letter for letter, active in zip(LETTERS, mask) if active) or "none",
            latched_alarm=bool(status["v82_alarm"]), hb_denoise_front=float(flow[:4].mean()),
            hb_denoise_back=float(flow[4:].mean()))
        movement = None
        if self.previous is not None:
            prev_hb, prev_as, prev_ids, prev_as_ids = self.previous
            mobility = distance(action, prev_hb[:, :, 1:]).mean(axis=(1, 2))
            movement = float(mobility.mean())
            result.update(hb_mobility_front=float(mobility[:4].mean()), hb_mobility_back=float(mobility[4:].mean()),
                hb_state_mobility=float(distance(hb[:, :, 0], prev_hb[:, :, 0]).mean()),
                hb_top4_turnover=float(turnover(ids[:, :, 1:], prev_ids[:, :, 1:]).mean()),
                as_mobility=float(distance(as_root[:, :, 1:], prev_as[:, :, 1:]).mean()),
                as_top1_turnover=float(turnover(as_ids[:, :, 1:], prev_as_ids[:, :, 1:]).mean()))
        else:
            result.update({key: None for key in ("hb_mobility_front", "hb_mobility_back", "hb_state_mobility",
                "hb_top4_turnover", "as_mobility", "as_top1_turnover")})
        result["hb_return_lag3"] = float(distance(action, self.hb_history[-3]).mean()) if len(self.hb_history)>=3 else None
        if query == self.alarm:
            self.anchor = action.copy(), as_root[:, :, 1:].copy()
        if query > self.alarm and movement is not None:
            self.path_length += movement
        if self.anchor is not None:
            drift = float(distance(action, self.anchor[0]).mean())
            result.update(hb_distance_from_alarm=drift,
                as_distance_from_alarm=float(distance(as_root[:, :, 1:], self.anchor[1]).mean()),
                hb_escape_efficiency=drift/self.path_length if self.path_length>0 else 0.)
        else:
            result.update(hb_distance_from_alarm=None, as_distance_from_alarm=None, hb_escape_efficiency=None)
        self.hb_history.append(action.copy())
        self.hb_history = self.hb_history[-3:]
        self.previous = hb.copy(), as_root.copy(), ids.copy(), as_ids.copy()
        return result


def specifications():
    tasks, specs, sources = {}, [], {}
    cohort_counts = {}
    for split, run, audit_path in INPUTS:
        plan = json.loads((run/"plan.json").read_text())
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "passed" or audit["plan_sha256"] != digest(run/"plan.json"):
            raise ValueError("Verified original dataset required")
        for path in (run/"plan.json", audit_path):
            sources[str(path.resolve())] = digest(path)
        cohort_counts[split] = dict(mains=len(plan["cohort"]), successes=sum(t["native_success"] for t in plan["cohort"]))
        for task in plan["cohort"]:
            if task["main_id"] in tasks:
                raise ValueError("Repeated parent across cohorts")
            tasks[task["main_id"]] = task
            alarm = task["first_alarms"]["v82_frozen"]
            if alarm<0:
                continue
            directory = Path(task["parent_directory"])
            if digest(directory/"main/manifest.json") != task["parent_manifest_sha256"]:
                raise ValueError("Main manifest changed")
            original = json.loads((directory/"result.json").read_text())
            if original["success"] != task["native_success"] or original["hidden_capture"]:
                raise ValueError("Original label/capture mismatch")
            common = dict(main_id=task["main_id"], task=task["base_task"], init_index=task["init_index"],
                split=split, alarm_query=alarm, main_directory=str(directory/"main"),
                original_success=task["native_success"])
            specs.append(dict(common, trace_id=task["main_id"], group="natural_success" if task["native_success"] else "natural_failure",
                operator="native", repeat=-1, expected_success=task["native_success"], suffix_directory=None,
                start_query=None, suffix_manifest_sha256=None))
            sources[str(directory/"main/manifest.json")] = digest(directory/"main/manifest.json")
        seen = set()
        for audited in audit["tasks"]:
            task = tasks[audited["main_id"]]
            event = task["events"][0]
            for row in audited["branches"]:
                if not (row["rescued"] or row["harmed"]):
                    continue
                key = row["main_id"], row["operator"], row["repeat"]
                if key in seen:
                    continue
                seen.add(key)
                directory = run/"events"/event["event_id"]/row["arm"]
                if digest(directory/"suffix/manifest.json") != row["suffix_manifest_sha256"]:
                    raise ValueError("Audited suffix changed")
                specs.append(dict(main_id=row["main_id"], task=task["base_task"], init_index=task["init_index"], split=split,
                    alarm_query=task["first_alarms"]["v82_frozen"], main_directory=str(Path(task["parent_directory"])/"main"),
                    original_success=task["native_success"], trace_id="/".join((row["main_id"],row["operator"],str(row["repeat"]))),
                    group="rescued" if row["rescued"] else "harmed", operator=row["operator"], repeat=row["repeat"],
                    expected_success=row["success"], suffix_directory=str(directory/"suffix"),
                    start_query=event["start_query"], suffix_manifest_sha256=row["suffix_manifest_sha256"]))
    return specs, sources, cohort_counts


def extract_one(spec):
    main = records(Path(spec["main_directory"]))
    stream = main if spec["suffix_directory"] is None else chain(islice(main, spec["start_query"]), records(Path(spec["suffix_directory"])))
    feature = TraceFeatures(spec["alarm_query"])
    points = [feature.update(row) for row in stream]
    if feature.monitor.first_v82_alarm != spec["alarm_query"] or points[-1]["success_after"] != spec["expected_success"]:
        raise ValueError("Original trigger or committed outcome mismatch")
    if spec["suffix_directory"] is None and not all(p["latched_alarm"] for p in points[spec["alarm_query"]:]):
        raise ValueError("Frozen alarm latch changed")
    return dict(spec, points=points, final_steps=points[-1]["steps_before"]+points[-1]["executed_actions"],
        post_alarm_queries=len(points)-spec["alarm_query"]-1)


def clear_at(points, key, length, boundary):
    count = 0
    for offset, point in enumerate(points, 1):
        value = point[key]
        count = count+1 if value is not None and value<boundary else 0
        if count>=length:
            return offset
    return None


def trace_summary(trace):
    a = trace["alarm_query"]
    after = trace["points"][a+1:]
    row = {key: trace[key] for key in ("trace_id", "main_id", "task", "init_index", "split", "group", "operator", "repeat",
        "alarm_query", "post_alarm_queries", "final_steps")}
    row.update(alarm_steps=trace["points"][a]["steps_before"], alarm_mode=trace["points"][a]["mode"],
        alarm_risk=trace["points"][a]["risk"], last_risk=trace["points"][-1]["risk"], last_mode=trace["points"][-1]["mode"],
        first_risk_below=clear_at(after,"risk",1,0.), risk_below_three=clear_at(after,"risk",3,0.),
        all_below_three=clear_at(after,"maximum_component",3,0.), all_deep_below_three=clear_at(after,"maximum_component",3,-1.))
    for key in ("risk_below_three", "all_below_three"):
        start = row[key]
        row[key+"_recurred"] = (any(p["risk" if key=="risk_below_three" else "maximum_component"] is not None and
            p["risk" if key=="risk_below_three" else "maximum_component"]>=0 for p in after[start:]) if start is not None else None)
    return row


def aligned_summary(traces):
    result = []
    for group in ("natural_success", "natural_failure", "rescued", "harmed"):
        members = [t for t in traces if t["group"]==group]
        for offset in range(-4, 22):
            parents = {}
            for trace in members:
                query = trace["alarm_query"]+offset
                if 0<=query<len(trace["points"]):
                    parents.setdefault(trace["main_id"], []).append(trace["points"][query])
            values = {}
            for feature in SCALARS:
                per_parent = [float(np.mean([p[feature] for p in points if p[feature] is not None]))
                    for points in parents.values() if any(p[feature] is not None for p in points)]
                values[feature] = dict(median=float(np.median(per_parent)) if per_parent else None,
                    parents=len(per_parent))
            result.append(dict(group=group,offset=offset,parents=len(parents),
                traces=sum(len(points) for points in parents.values()),features=values))
    return result


def learning_rows(traces, offset, method):
    x, y, parents, tasks, splits = [], [], [], [], []
    for trace in traces:
        if not trace["group"].startswith("natural_"):
            continue
        a, t = trace["alarm_query"], trace["alarm_query"]+offset
        if t>=len(trace["points"]):
            continue
        origin, current = trace["points"][a], trace["points"][t]
        values = [current["steps_before"]/520.]
        if method!="clock":
            values += [origin[k] for k in LEARNING_BASE]
        if method=="post_history":
            values += [current[k] for k in LEARNING_BASE]
            values += [current[k]-origin[k] for k in LEARNING_BASE]
            values += [current[k] for k in LEARNING_DYNAMIC]
        if any(v is None or not np.isfinite(v) for v in values):
            continue
        x.append(values); y.append(trace["original_success"]); parents.append(trace["main_id"])
        tasks.append(trace["task"]); splits.append(trace["split"])
    return np.asarray(x), np.asarray(y,int), parents, np.asarray(tasks), np.asarray(splits)


def evaluate_predictions(y, probability):
    from sklearn.metrics import roc_auc_score
    good = np.isfinite(probability)
    y, p = y[good], probability[good]
    return dict(parents=len(y), natural_successes=int(y.sum()), failures=int((1-y).sum()),
        auc_success=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        spared_successes_at_half=int(((p>=.5)&(y==1)).sum()),
        missed_failure_alarms_at_half=int(((p>=.5)&(y==0)).sum()))


def fit_predict(x, y, train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(set(y[train]))<2 or not test.any():
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1,class_weight="balanced",max_iter=2000,random_state=20260909))
    model.fit(x[train], y[train])
    return model.predict_proba(x[test])[:,1]


def learning_experiment(traces):
    outputs = []
    for offset in (0, 1, 3, 5):
        for method in (("clock", "alarm_only") if offset==0 else ("clock", "alarm_only", "post_history")):
            x, y, parents, tasks, splits = learning_rows(traces, offset, method)
            probability = np.full(len(y), np.nan)
            folds = []
            for task in sorted(set(tasks)):
                test = tasks==task
                prediction = fit_predict(x,y,~test,test)
                folds.append(dict(task=task,training_parents=int((~test).sum()),training_successes=int(y[~test].sum()),
                    testing_parents=int(test.sum()),testing_successes=int(y[test].sum()),status="evaluated" if prediction is not None else "one_class_or_empty"))
                if prediction is not None:
                    probability[test] = prediction
            train, test = splits=="prior", splits=="recent"
            held_probability = np.full(len(y), np.nan)
            prediction = fit_predict(x,y,train,test)
            if prediction is not None:
                held_probability[test] = prediction
            outputs.append(dict(offset=offset,method=method,feature_dimensions=x.shape[1],
                task_out=evaluate_predictions(y,probability),prior_to_recent=evaluate_predictions(y,held_probability),folds=folds,
                predictions=[dict(main_id=parent,success=int(label),task_out=finite(p),prior_to_recent=finite(h))
                    for parent,label,p,h in zip(parents,y,probability,held_probability)]))
    return outputs


def analyze(args):
    data = json.loads(args.extract.read_text())
    if data["status"]!="passed" or data["protocol"]!=PROTOCOL:
        raise ValueError("Verified extraction required")
    traces = data["traces"]
    summary = dict(status="passed",protocol=PROTOCOL,extract_sha256=digest(args.extract),analyzer_sha256=digest(__file__),
        cohort_counts=data["cohort_counts"],group_counts={group:dict(parents=len({t["main_id"] for t in traces if t["group"]==group}),
            traces=sum(t["group"]==group for t in traces)) for group in sorted({t["group"] for t in traces})},
        cases=[trace_summary(t) for t in traces],aligned=aligned_summary(traces),learning=learning_experiment(traces),
        learning_settings=dict(C=.1,class_weight="balanced",decision_threshold=.5,preprocessing="fit inside each training fold",
            labels="natural success after alarm, versus eventual failure without intervention",
            use="retrospective diagnostic only; no new controller or threshold deployed",
            caveat="Both existing cohorts have previously been examined; recent-cohort split is not a new prospective test"),
        actual_gpu_queries=0,alarm_parameters_changed=False,hidden_capture=False)
    atomic_json(args.output,summary)
    plot(args.output,data,summary)
    print(json.dumps(dict(status="passed",groups=summary["group_counts"],
        natural_success_cases=[r for r in summary["cases"] if r["group"]=="natural_success"])))
    print(json.dumps([dict(offset=r["offset"],method=r["method"],task_out=r["task_out"],prior_to_recent=r["prior_to_recent"]) for r in summary["learning"]]))


def plot(output,data,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [t for t in data["traces"] if t["group"]=="natural_success"]
    fig,axes = plt.subplots(len(cases),2,figsize=(11,2.3*len(cases)),constrained_layout=True)
    for row,trace in enumerate(cases):
        a = trace["alarm_query"]
        selected = trace["points"][max(0,a-4):]
        x = [p["query"]-a for p in selected]
        for key,color in (("risk","#a9465a"),("maximum_component","#6e7785")):
            axes[row,0].plot(x,[p[key] for p in selected],"o-",markersize=3,color=color,label=key)
        axes[row,0].axhline(0,color="#666666",linestyle=":")
        axes[row,0].set_title(trace["main_id"][:6]+" / init "+str(trace["init_index"])+" / "+str(trace["post_alarm_queries"])+" later queries",fontsize=10)
        for key,color in (("hb_mobility_back","#247e68"),("hb_distance_from_alarm","#ab792c")):
            axes[row,1].plot(x,[p[key] for p in selected],"o-",markersize=3,color=color,label=key)
        axes[row,1].set_title("HB routing: local movement and displacement",fontsize=10)
        for axis in axes[row]:
            axis.axvline(0,color="#444444",linewidth=.8);axis.grid(alpha=.15);axis.set_xlabel("Query offset from first v8.2 alarm")
            if len(x)==5:
                axis.set_xlim(-4.4,1.4)
        axes[row,0].set_ylabel("Risk in margin units");axes[row,1].set_ylabel("Hellinger distance")
    handles,labels=[],[]
    for axis in axes[0]:
        h,l=axis.get_legend_handles_labels();handles+=h;labels+=l
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("All five original Long alarms followed by natural success\n"
        "Lines stop at the final policy query; no endpoint padding or future interpolation",fontsize=13)
    fig.savefig(output.with_suffix(".png"),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,key in zip(axes.flat,("risk","hb_mobility_back","hb_top4_turnover","as_mobility")):
        for group,color in (("natural_success","#247e68"),("natural_failure","#a9465a")):
            selected=[r for r in summary["aligned"] if r["group"]==group and r["offset"]>=0]
            axis.plot([r["offset"] for r in selected],[r["features"][key]["median"] for r in selected],"o-",markersize=3,color=color,label=group)
        axis.axvline(0,color="#666666",linewidth=.8);axis.set_title(key);axis.set_xlabel("Query offset from first alarm");axis.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Observed native suffixes: medians among trajectories still running\n"
        "Natural-success counts fall from 5 at alarm to 4 at +1 and 3 at +3",fontsize=12)
    fig.savefig(output.with_name(output.stem+"_aligned.png"),dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("extract","analyze"),required=True)
    parser.add_argument("--extract",type=Path,default=Path("design/native_false_alarm_traces_20260909.json"))
    parser.add_argument("--output",type=Path,default=Path("design/native_false_alarm_analysis_20260909.json"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.phase=="analyze":
        analyze(args)
        return
    specs,sources,counts=specifications()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        traces=list(pool.map(extract_one,specs))
    atomic_json(args.extract,dict(status="passed",protocol=PROTOCOL,source_sha256=sources,
        extractor_sha256=digest(__file__),python=platform.python_version(),numpy=np.__version__,cohort_counts=counts,
        layers=dict(hb=list(HB_LAYERS),as_layers=list(AS_LAYERS)),traces=traces))
    print(json.dumps(dict(status="passed",traces=len(traces),groups=dict(Counter(t["group"] for t in traces)))))


if __name__=="__main__":
    main()
