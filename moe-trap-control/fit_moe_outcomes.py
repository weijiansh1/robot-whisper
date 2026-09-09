#!/usr/bin/env python3
"""Identify success/failure geometry and intervention responses from audited data."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from moe_outcome_models import (PROTOCOL, MODEL_SETTINGS, OPERATORS, MODELS, FEATURE_NAMES, FeatureHistory,
    choose, cluster, density_value, project)


def dataset(bank):
    plan = json.loads((bank/"plan.json").read_text())
    audit = json.loads(Path("design/control_bank_audit_20260909.json").read_text())
    if audit["status"] != "passed" or audit["plan_sha256"] != digest(bank/"plan.json"):
        raise ValueError("Verified paired bank required")
    frame_x, frame_parent, frame_task, frame_y, frame_q, histories = [],[],[],[],[],{}
    event_x, event_parent, event_task, outcomes, responses, response_valid, terminal = [],[],[],[],[],[],[]
    sources = {str((bank/"plan.json").resolve()):digest(bank/"plan.json"),
               str(Path("design/control_bank_audit_20260909.json").resolve()):digest("design/control_bank_audit_20260909.json")}
    for parent, task in enumerate(plan["cohort"]):
        path = Path(task["parent_directory"])
        sources[str(path/"main/manifest.json")] = digest(path/"main/manifest.json")
        history = FeatureHistory()
        q0 = task["events"][0]["start_query"] if task["events"] else -1
        for row in records(path/"main"):
            q = int(row["query"])
            if q == q0:
                histories[task["main_id"]] = copy.deepcopy(history)
            feature = history.update(row[PROBS_KEY],int(row["action_steps_before"]))
            if q >= 8:
                frame_x.append(feature); frame_parent.append(parent); frame_task.append(task["base_task"])
                frame_y.append(task["native_success"]); frame_q.append(q)
            if q == q0:
                event_x.append(feature); event_parent.append(parent); event_task.append(task["base_task"])
        if not task["events"]:
            continue
        event = task["events"][0]
        y, response, valid, absorb = np.zeros((len(OPERATORS),2)),np.zeros((len(OPERATORS),2,77)),np.zeros((len(OPERATORS),2),bool),np.zeros((len(OPERATORS),2),int)
        for op, operator in enumerate(OPERATORS):
            for rep in (0,1):
                directory = bank/"events"/event["event_id"]/(operator+"_r"+str(rep))
                branch = json.loads((directory/"branch.json").read_text())
                sources[str(directory/"branch.json")] = digest(directory/"branch.json")
                sources[str(directory/"suffix/manifest.json")] = digest(directory/"suffix/manifest.json")
                y[op,rep] = branch["success"]
                history = copy.deepcopy(histories[task["main_id"]])
                for row in records(directory/"suffix"):
                    feature = history.update(row[PROBS_KEY],int(row["action_steps_before"]))
                    if int(row["action_steps_before"])-branch["initial_action_steps"] >= 32:
                        response[op,rep],valid[op,rep] = feature,True
                        break
                if not valid[op,rep]:
                    absorb[op,rep] = 1 if branch["success"] else -1
        outcomes.append(y); responses.append(response); response_valid.append(valid); terminal.append(absorb)
    return dict(frame_x=np.asarray(frame_x),frame_parent=np.asarray(frame_parent),frame_task=np.asarray(frame_task),
        frame_y=np.asarray(frame_y,int),frame_q=np.asarray(frame_q),event_x=np.asarray(event_x),
        event_parent=np.asarray(event_parent),event_task=np.asarray(event_task),outcomes=np.asarray(outcomes),
        responses=np.asarray(responses),response_valid=np.asarray(response_valid),terminal=np.asarray(terminal),
        main_ids=np.asarray([t["main_id"] for t in plan["cohort"]]),
        main_tasks=np.asarray([t["base_task"] for t in plan["cohort"]]),
        main_success=np.asarray([t["native_success"] for t in plan["cohort"]],int)),sources


def ridge(x,y,weight=None):
    x,y = np.asarray(x,float),np.asarray(y,float)
    if len(x) == 0:
        return np.zeros((y.shape[-1],9)) if y.ndim == 2 else np.zeros(9)
    weight = np.ones(len(x)) if weight is None else weight
    design = np.column_stack((np.ones(len(x)),x))
    penalty = np.diag([1.]+[10.]*x.shape[1])
    return np.linalg.solve(design.T @ (weight[:,None]*design)+penalty,design.T @ (weight[:,None]*y if y.ndim==2 else weight*y)).T


def fit(data, parents):
    from sklearn.cluster import KMeans
    mask = np.isin(data["frame_parent"],parents)
    x, ids, labels = data["frame_x"][mask],data["frame_parent"][mask],data["frame_y"][mask]
    unique, counts = np.unique(ids,return_counts=True)
    weights = np.array([1./dict(zip(unique,counts))[parent] for parent in ids])
    mean = np.average(x,axis=0,weights=weights)
    scale = np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),1e-4)
    normal = np.clip((x-mean)/scale,-8.,8.)
    covariance = normal.T @ (normal*weights[:,None])/weights.sum()
    values,vectors = np.linalg.eigh(covariance)
    take = np.argsort(values)[-8:][::-1]
    projection = vectors[:,take]/np.sqrt(np.maximum(values[take],.1))[None,:]
    model = dict(mean=mean.tolist(),scale=scale.tolist(),projection=projection.tolist(),
        settings=MODEL_SETTINGS,operators=list(OPERATORS),parent_indices=parents.tolist())
    z = project(model,x)
    gaussians = []
    for label in (0,1):
        these = labels == label
        if not these.any():
            raise ValueError("Both original outcome classes required in each fold")
        center = np.average(z[these],axis=0,weights=weights[these])
        delta = z[these]-center
        cov = delta.T @ (weights[these,None]*delta)/weights[these].sum()
        cov = .5*cov+.5*np.eye(8)
        gaussians.append(dict(mean=center.tolist(),precision=np.linalg.inv(cov).tolist(),logdet=float(np.linalg.slogdet(cov)[1])))
    model["gaussians"] = gaussians
    kmeans = KMeans(n_clusters=8,n_init=10,random_state=20260909).fit(z,sample_weight=weights)
    model["centers"] = kmeans.cluster_centers_.tolist()
    states = cluster(model,z)
    counts = np.full((8,10),.01)
    for parent in unique:
        indices = np.flatnonzero(ids == parent)
        for left,right in zip(indices[:-1],indices[1:]):
            counts[states[left],states[right]] += 1./len(indices)
        counts[states[indices[-1]],8+int(labels[indices[-1]])] += 1./len(indices)
    transition = counts/counts.sum(1,keepdims=True)
    committor = np.r_[np.linalg.solve(np.eye(8)-transition[:,:8],transition[:,9]),0.,1.]
    model.update(transition=transition.tolist(),committor=committor.tolist())
    events = np.isin(data["event_parent"],parents)
    entry = project(model,data["event_x"][events])
    observed = data["outcomes"][events].mean(-1)
    uplift = observed-observed[:,0,None]
    model.update(entry_latent=entry.tolist(),uplifts=uplift.tolist(),
        entry_clock=data["event_x"][events,-1].tolist(),event_parent_indices=data["event_parent"][events].tolist())
    coefficients, effects, transitions = [],[],[]
    for op in range(len(OPERATORS)):
        valid = data["response_valid"][events,op]
        repeated_x = np.repeat(entry,2,axis=0)
        flat_valid = valid.ravel()
        targets = project(model,data["responses"][events,op].reshape(-1,77)[flat_valid])
        # Regularize changes toward the identity response, keeping the fit in observed coordinates.
        coefficient = ridge(repeated_x[flat_valid],targets-repeated_x[flat_valid],np.full(flat_valid.sum(),.5))
        coefficient[:,1:] += np.eye(8)
        coefficients.append(coefficient.tolist())
        effects.append(ridge(entry,uplift[:,op]).tolist())
        op_counts = 2.*transition.copy()
        for index,state in enumerate(cluster(model,entry)):
            for rep in (0,1):
                destination = (int(cluster(model,project(model,data["responses"][events,op][index,rep]))[0])
                    if valid[index,rep] else (9 if data["terminal"][events,op][index,rep] > 0 else 8))
                op_counts[state,destination] += .5
        transitions.append((op_counts/op_counts.sum(1,keepdims=True)).tolist())
    model.update(response_coefficients=coefficients,uplift_coefficients=effects,operator_transitions=transitions)
    return model


def cross_validate(data):
    from sklearn.metrics import roc_auc_score
    all_parents = np.arange(len(data["main_ids"]))
    folds, diagnostics, landmarks = [],[],[]
    for task in sorted(set(data["main_tasks"])):
        train = all_parents[data["main_tasks"] != task]
        model = fit(data,train)
        held = np.flatnonzero(data["event_task"] == task)
        results = []
        for index in held:
            for method in MODELS+("clock_knn",):
                operator,scores = choose(model,method,data["event_x"][index])
                op = OPERATORS.index(operator)
                reference = data["outcomes"][index,0].astype(int)
                selected = data["outcomes"][index,op].astype(int)
                results.append(dict(main_id=str(data["main_ids"][data["event_parent"][index]]),method=method,
                    operator=operator,success=selected.tolist(),resample=reference.tolist(),delta=(selected-reference).tolist(),
                    training_main_ids=[str(data["main_ids"][i]) for i in train]))
        frame = data["frame_task"] == task
        latent = project(model,data["frame_x"][frame])
        potential = density_value(model,latent)
        markov = np.asarray(model["committor"])[cluster(model,latent)]
        for parent in np.unique(data["frame_parent"][frame]):
            indices = np.flatnonzero(data["frame_parent"][frame] == parent)
            diagnostics.append(dict(main_id=str(data["main_ids"][parent]),task=task,success=int(data["main_success"][parent]),
                density_tail=float(potential[indices[-3:]].mean()),markov_tail=float(markov[indices[-3:]].mean())))
            for query in (8,16,24,32,40):
                selected = indices[data["frame_q"][frame][indices] == query]
                if len(selected):
                    index = int(selected[0])
                    landmarks.append(dict(main_id=str(data["main_ids"][parent]),query=query,
                        success=int(data["main_success"][parent]),density=float(potential[index]),markov=float(markov[index])))
        folds.append(dict(heldout_task=task,training_mains=len(train),heldout_alarm_parents=len(held),choices=results))
        print(json.dumps(dict(heldout_task=task,alarms=len(held))),flush=True)
    outcomes = [r for fold in folds for r in fold["choices"]]
    metrics = []
    for method in MODELS+("clock_knn",):
        rows = [r for r in outcomes if r["method"] == method]
        metrics.append(dict(method=method,parents=len(rows),wins=[sum(r["delta"][rep]>0 for r in rows) for rep in (0,1)],
            losses=[sum(r["delta"][rep]<0 for r in rows) for rep in (0,1)]))
    auc = {key:float(roc_auc_score([r["success"] for r in diagnostics],[r[key] for r in diagnostics]))
           for key in ("density_tail","markov_tail")}
    landmark_metrics = []
    for query in (8,16,24,32,40):
        rows = [r for r in landmarks if r["query"] == query]
        y = [r["success"] for r in rows]
        landmark_metrics.append(dict(query=query,mains=len(rows),successes=sum(y),failures=len(y)-sum(y),
            auc={method:float(roc_auc_score(y,[r[method] for r in rows])) if len(set(y))==2 else None
                for method in ("density","markov")}))
    tail = np.stack([data["frame_x"][data["frame_parent"]==parent][-3:].mean(0) for parent in all_parents])
    positive,negative = tail[data["main_success"]==1],tail[data["main_success"]==0]
    difference = (positive.mean(0)-negative.mean(0))/np.maximum(tail.std(0),1e-6)
    effects = [dict(feature=name,success_mean=float(positive[:,i].mean()),failure_mean=float(negative[:,i].mean()),
        standardized_difference=float(difference[i])) for i,name in enumerate(FEATURE_NAMES)]
    return dict(folds=folds,methods=metrics,main_diagnostics=diagnostics,tail_auc=auc,
        landmarks=landmarks,landmark_metrics=landmark_metrics,tail_feature_effects=effects,
        caveat="Tail scores are retrospective diagnostics, not early alarms; task-isolated folds refit all preprocessing")


def main(args):
    if args.output.exists():
        raise ValueError("Refusing to overwrite frozen outcome models")
    data,sources = dataset(args.bank.resolve())
    np.savez_compressed(args.output.with_suffix(".npz"),**data)
    diagnostics = cross_validate(data)
    model = fit(data,np.arange(len(data["main_ids"])))
    result = dict(status="frozen",protocol=PROTOCOL,model=model,diagnostics=diagnostics,
        data_sha256=digest(args.output.with_suffix(".npz")),source_sha256=sources,
        code_sha256={str(Path(name).resolve()):digest(name) for name in ("moe_outcome_models.py","fit_moe_outcomes.py")},
        training_main_ids=data["main_ids"].tolist(),training_init_indices=list(range(15)),
        training_mains=len(data["main_ids"]),training_alarm_states=len(data["event_parent"]),
        policy_weights_frozen=True,alarm_parameters_frozen=True,hidden_capture=False)
    atomic_json(args.output,result)
    print(json.dumps(dict(status="frozen",methods=diagnostics["methods"],tail_auc=diagnostics["tail_auc"],
        training_mains=result["training_mains"],training_alarm_states=result["training_alarm_states"],sha256=digest(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
