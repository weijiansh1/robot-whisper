#!/usr/bin/env python3
"""Freeze untouched original Long mains, then causal matched-choice interventions."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from moe_outcome_models import PROTOCOL, SETTINGS, ARMS, MODELS, FeatureHistory, choose, derangement
from native_long_runtime import BASE, ROOT, environment, inventory, verify_source
from v8_closed_loop import PROTOCOL as MAIN_PROTOCOL, SETTINGS as MAIN_SETTINGS
from v82_closed_loop import V82TriggerMonitor

RUNTIME = ["moe_outcome_models.py","fit_moe_outcomes.py","test_moe_outcome_models.py",
    "collect_moe_outcomes.py","run_moe_outcomes.py","prepare_moe_outcomes.py",
    "MOE_OUTCOME_MODELS_THEORY.zh.md"]


def verified_model(args):
    fitted = json.loads(args.model.read_text())
    if fitted["status"] != "frozen" or fitted["protocol"] != PROTOCOL:
        raise ValueError("Frozen outcome estimators required")
    for sources in (fitted["code_sha256"],fitted["source_sha256"]):
        for path,expected in sources.items():
            if digest(path) != expected:
                raise ValueError("Training commitment changed: "+path)
    return fitted


def prepare_mains(args):
    from benchmarks.run_benchmarks import load_suite
    fitted = verified_model(args)
    prior = json.loads((HERE/"design/control_bank_plan_20260909.json").read_text())
    for path,expected in prior["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Previous frozen source changed: "+path)
    suite,tasks = load_suite("libero_10",{}),[]
    for row in inventory():
        initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
        for index in range(15,25):
            seed = int(stable_id(PROTOCOL,row["variant_id"],index,"main_noise")[:8],16)
            tasks.append(dict(main_id=stable_id(PROTOCOL,row["variant_id"],index,seed),
                variant_id=row["variant_id"],variant=row,benchmark="native_long",category="Original",
                analysis_role="native",base_task=row["base_task"],noise_seed=seed,init_index=index,
                initial_state_sha256=hashlib.sha256(np.ascontiguousarray(initial[index]).tobytes()).hexdigest()))
    if set(fitted["training_main_ids"]) & {task["main_id"] for task in tasks}:
        raise ValueError("Fresh and training parents overlap")
    runtime = list(dict.fromkeys(prior["runtime_sources"]+RUNTIME))
    sources = dict(prior["source_sha256"])
    sources.update({str(HERE/name):digest(HERE/name) for name in runtime})
    sources[str(args.model.resolve())] = digest(args.model)
    maximum = 2*52*len(tasks)
    return dict(protocol=MAIN_PROTOCOL,stage="moe_outcome_fresh_mains",experiment="original_libero_long",
        settings=MAIN_SETTINGS,arms=[],tasks=tasks,model="long",benchmark="native_long",
        original_source=verify_source(),allowed_gpus=[0,1,2,3],render_gpus=[0,3],replicas_per_gpu=8,
        workers_per_gpu=8,batch_size=1,hidden_capture=False,threshold_fitting=False,
        outcome_protocol=PROTOCOL,outcome_settings=SETTINGS,outcome_arms=list(ARMS),
        outcome_model_path=str(args.model.resolve()),outcome_model_sha256=digest(args.model),
        fresh_init_indices=list(range(15,25)),new_main_coverage=100,
        selection="all ten native Long tasks, official initializations 15..24; no outcome-dependent selection",
        runtime_sources=runtime,source_sha256=sources,thresholds=prior["thresholds"],margins=prior["margins"],
        maximum_model_queries=maximum,maximum_output_bytes=maximum*200*1024+100*20*1024**2,
        storage_quota_gib=6,disk_floor_gib=8)


def prepare_branches(args):
    fitted = verified_model(args)
    fresh = json.loads((args.fresh_run/"plan.json").read_text())
    summary = json.loads((args.fresh_run/"summary.json").read_text())
    if summary["status"] != "completed" or len(fresh["cohort"]) != 100:
        raise ValueError("All 100 original mains and full C0 must complete first")
    if fresh["outcome_settings"] != SETTINGS or fresh["outcome_model_sha256"] != digest(args.model):
        raise ValueError("Outcome controller changed after preregistration")
    for path,expected in fresh["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: "+path)
    cohort = []
    for parent in fresh["cohort"]:
        task = copy.deepcopy(parent)
        task.update(cohort_group="fresh",events=[])
        path = Path(task["parent_directory"])
        for name,key in (("main_complete.json","parent_commit_sha256"),("main/manifest.json","parent_manifest_sha256")):
            if digest(path/name) != task[key]:
                raise ValueError("Parent changed")
        history,monitor,features = FeatureHistory(),V82TriggerMonitor(),[]
        for row in records(path/"main"):
            monitor.update(row[PROBS_KEY],row["proprio"][:3])
            features.append(history.update(row[PROBS_KEY],int(row["action_steps_before"])))
        for key,value in task["first_alarms"].items():
            if monitor.first[key] != value:
                raise ValueError("Original alarm changed")
        task["first_alarms"] = monitor.first
        q0 = monitor.first["v82_frozen"]+1
        if q0 > 0 and q0 < len(features):
            task["events"] = [dict(event_id=stable_id(PROTOCOL,task["main_id"],q0),start_query=q0,
                alarm_query=q0-1,methods=["v82_frozen"],deployable=True)]
            selections = {}
            for method in MODELS+("clock_knn",):
                operator,scores = choose(fitted["model"],method,features[q0])
                selections[method] = dict(operator=operator,scores=scores.tolist())
            task.update(entry_feature=features[q0].tolist(),selector_choices=selections,
                outcome_model_path=str(args.model.resolve()),outcome_model_sha256=digest(args.model),shuffled_choices={})
        cohort.append(task)
    tasks = [t for t in cohort if t["events"]]
    lookup = {t["main_id"]:t for t in tasks}
    for method in MODELS:
        for repeat in (0,1):
            for main_id,donor in derangement(lookup,method,repeat).items():
                lookup[main_id]["shuffled_choices"]["shuffle_"+method+"_r"+str(repeat)] = dict(
                    donor_main_id=donor,operator=lookup[donor]["selector_choices"][method]["operator"])
    runtime = list(dict.fromkeys(fresh["runtime_sources"]+["audit_moe_outcomes.py","verify_moe_outcome_actions.py"]))
    sources = dict(fresh["source_sha256"])
    sources.update({str(HERE/name):digest(HERE/name) for name in runtime})
    sources.update({str((args.fresh_run/name).resolve()):digest(args.fresh_run/name) for name in ("cohort_plan.json","plan.json","summary.json")})
    maximum = sum(t["parent_queries"]+len(ARMS)*(52-t["events"][0]["start_query"]+8) for t in tasks)
    bound = maximum*200*1024+len(tasks)*32*1024**2
    return dict(protocol=PROTOCOL,stage="moe_outcome_online",benchmark="native_long",model="long",
        settings=SETTINGS,arms=list(ARMS),tasks=tasks,cohort=cohort,cohorts=["fresh"],
        original_source=verify_source(),thresholds=fresh["thresholds"],margins=fresh["margins"],methods=["v82_frozen"],
        allowed_gpus=[0,1,2,3],render_gpus=[0,3],replicas_per_gpu=8,workers_per_gpu=8,batch_size=1,
        hidden_capture=False,threshold_fitting=False,outcome_model_path=str(args.model.resolve()),
        outcome_model_sha256=digest(args.model),fresh_run=str(args.fresh_run.resolve()),new_main_coverage=100,
        runtime_sources=runtime,source_sha256=sources,maximum_model_queries=maximum,maximum_output_bytes=bound,
        storage_quota_gib=max(6,math.ceil(bound/1024**3)+1),disk_floor_gib=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("mains","branches"),required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--fresh-run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    if args.phase == "mains" and not args.worker:
        subprocess.run([str(BASE/"envs/libero/bin/python"),str(Path(__file__).resolve()),"--worker","--phase","mains",
            "--model",str(args.model.resolve()),"--output",str(args.output.resolve())],
            env=environment(0,initialize=True),cwd=ROOT,check=True)
    else:
        if args.output.exists():
            raise ValueError("Cannot overwrite frozen plan")
        verify_frozen_alarm()
        result = prepare_mains(args) if args.phase == "mains" else prepare_branches(args)
        atomic_json(args.output,result)
        print(json.dumps(dict(status="frozen",phase=args.phase,tasks=len(result["tasks"]),
            arms=len(result["arms"]),maximum_model_queries=result["maximum_model_queries"],sha256=digest(args.output))))
