#!/usr/bin/env python3
"""Independently reconstruct gate arithmetic and verify the paired strength sweep."""

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest, load_snapshot, records
from collection_routes import (ALL_FIELDS, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                                EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)
from audit_v8_feature_experiment import require, probability_weights
from audit_long_continuation import route_check, audit_as
from v8_feature_control import normalize
from v8_strength_control import PROTOCOL, ARMS, LOGIT_FIELDS, strength_bias, seed_for, noise_for, V8Monitor, flow_features


def bf16_round(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite gate arithmetic")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def softmax64(logits):
    value = np.asarray(logits, np.float64)
    value = np.exp(value - value.max(-1, keepdims=True))
    return value / value.sum(-1, keepdims=True)


def audit_job(payload):
    run, task, job = payload
    directory = Path(run) / "jobs" / job["job_id"]
    result = json.loads((directory / "result.json").read_text())
    require(result["status"] == "completed" and not result["invalid_pair"], "Incomplete strength job")
    require(result["protocol"] == PROTOCOL and result["gpu"] in (0,1,2,3) and result["render_gpu"] in (0,3), "Strength protocol/GPU")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["reused_main"], "Main/no-hidden contract")
    require(result["weak_full_suffix_exact"] and result["baseline_first_query_exact"], "Missing baseline equivalence")
    require(result["main_id"] == task["main_id"] and result["replicate"] == job["replicate"], "Job identity")
    parent, replay = Path(task["parent_directory"]), Path(task["replay_directory"])
    original = json.loads((parent / "result.json").read_text())
    require(result["seed"] == original["seed"] and result["init_index"] == original["init_index"], "Original seed/init")
    for file,key in (("main_complete.json","parent_commit_sha256"),("main/manifest.json","parent_manifest_sha256")):
        require(digest(parent/file)==task[key]==result[key], "Native commitment")
    for file,key in (("result.json","replay_result_sha256"),("c0/branch.json","replay_c0_sha256")):
        require(digest(replay/file)==task[key]==result["c0"][key], "Old C0 commitment")
    for key in ("checkpoint_sha256","normalization_stats_sha256","libero_wrist_layout",
                "himoe_upstream_commit","himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key]==original["model_metadata"][key], "Model identity")
    main, c0 = list(records(parent/"main")), list(records(replay/"c0"))
    require(len(main)==len(c0)==task["parent_queries"], "Original complete C0")
    event, rep = task["event"], job["replicate"]
    q0 = event["start_query"]
    prefix, monitor = None, V8Monitor()
    for expected,actual in zip(main,c0):
        for key in expected:
            if key not in ("inference_seconds","environment_seconds"):
                np.testing.assert_array_equal(expected[key],actual[key],err_msg="Original C0 "+key)
        monitor.update(actual[PROBS_KEY])
        if int(actual["query"])==q0-1:
            prefix=copy.deepcopy(monitor)
    require(monitor.first_alarm==prefix.first_alarm==q0-1==task["first_v8_alarm"], "Frozen v8 first alarm")
    location=replay/"events"/event["event_id"]/"snapshot"
    require(digest(location/"manifest.json")==event["snapshot_manifest_sha256"], "C0 snapshot commitment")
    saved=load_snapshot(location)
    require(saved["query"]==q0 and saved["action_steps"]==int(main[q0]["action_steps_before"]), "Snapshot placement")
    native=list(records(replay/"branches"/("r%d_native"%rep)/"suffix"))
    weak=list(records(replay/"branches"/("r%d_combined"%rep)/"suffix"))
    require(len(result["branches"])==len(ARMS), "Missing strength arms")
    outputs, prefix_rows, seen, total = [], {}, set(), 0
    dtypes_seen=set()
    for branch in result["branches"]:
        arm=branch["arm"]
        require(arm in ARMS and arm not in seen and branch["control"]==ARMS[arm], "Invalid arm")
        seen.add(arm)
        path=directory/"branches"/arm
        require(branch==json.loads((path/"branch.json").read_text()) and branch["status"]=="completed", "Branch commitment")
        suffix, controls=list(records(path/"suffix")),list(records(path/"control"))
        require(len(suffix)==branch["queries"] and len(controls)==min(ARMS[arm]["duration"],len(suffix)), "Incomplete suffix/evidence")
        require(branch["starts_after_c0"] and branch["starts_after_main_complete"], "Branch too early")
        previous=main[q0-1][PROBS_KEY]
        before=main[q0]["sim_before"]
        steps=saved["action_steps"]
        effective_monitor=copy.deepcopy(prefix)
        native_monitor=copy.deepcopy(prefix.v7)
        measures, release=[],None
        for index,row in enumerate(suffix):
            require(int(row["query"])==q0+index and int(row["relative_query"])==index, "Query index")
            require(int(row["action_steps_before"])==steps, "Action index")
            np.testing.assert_array_equal(row["sim_before"],before)
            count,success=int(row["executed_action_count"]),bool(row["success"])
            require(0<count<=min(10,520-steps) and (success or count==min(10,520-steps)), "Execution chunk")
            require(not success or index==len(suffix)-1, "Executed after success")
            np.testing.assert_array_equal(row["noise"],noise_for(task["main_id"],rep,index))
            require(int(row["policy_seed"])==seed_for(task["main_id"],rep,index,"policy"), "Policy seed")
            for key,offset in (("environment_first_seed",0),("environment_last_seed",count-1)):
                require(int(row[key])==seed_for(task["main_id"],rep,steps+offset,"environment"), "Environment seed")
            active=index<ARMS[arm]["duration"]
            require(bool(row["control_active"])==active, "Control duration")
            if active:
                evidence=controls[index]
                require(int(evidence["query"])==q0+index and int(evidence["relative_query"])==index, "Control evidence index")
                np.testing.assert_array_equal(evidence["input_sha256"],row["input_sha256"])
                np.testing.assert_array_equal(evidence["noise"],row["noise"])
                np.testing.assert_array_equal(evidence["previous_shadow"],previous)
                shadow={k:evidence["shadow/"+k] for k in ALL_FIELDS}
                shadow.update(actions=evidence["shadow_actions"],noise=evidence["noise"])
                route_check(shadow)
                audit_as(row)
                bias=strength_bias(shadow[PROBS_KEY],previous,ARMS[arm],seed_for(task["main_id"],rep,index,"direction"))
                np.testing.assert_allclose(evidence["logit_bias"],bias,atol=3e-5,rtol=2e-5)
                types=tuple(x.decode() for x in evidence["arithmetic_dtypes"])
                dtypes_seen.add(types)
                logits,changed_logits,rounded_bias=[evidence[k] for k in LOGIT_FIELDS]
                if types[0]=="torch.bfloat16":
                    np.testing.assert_array_equal(logits,bf16_round(logits))
                    np.testing.assert_array_equal(rounded_bias,bf16_round(evidence["logit_bias"]))
                    np.testing.assert_array_equal(changed_logits,bf16_round(logits+rounded_bias))
                elif types[0]=="torch.float32":
                    np.testing.assert_array_equal(rounded_bias,evidence["logit_bias"])
                    np.testing.assert_array_equal(changed_logits,logits+rounded_bias)
                else:
                    raise ValueError("Unverified logit dtype")
                p,effective=evidence["native_probs_fp32"],evidence["effective_probs_fp32"]
                np.testing.assert_allclose(p,softmax64(logits),atol=2e-7,rtol=2e-6)
                np.testing.assert_allclose(effective,softmax64(changed_logits),atol=2e-7,rtol=2e-6)
                np.testing.assert_array_equal(row[PROBS_KEY],p.astype(np.float16))
                probability_weights(p,row[NATIVE_IDS_KEY],row[NATIVE_WEIGHTS_KEY])
                probability_weights(effective,row[EFFECTIVE_IDS_KEY],row[EFFECTIVE_WEIGHTS_KEY])
                for key in (NATIVE_IDS_KEY,EFFECTIVE_IDS_KEY):
                    require(np.all(np.diff(np.sort(row[key],-1),axis=-1)>0), "Duplicate routed expert")
                np.testing.assert_array_equal(row[NATIVE_IDS_KEY][:4],row[EFFECTIVE_IDS_KEY][:4])
                np.testing.assert_array_equal(row[NATIVE_IDS_KEY][:,:,0],row[EFFECTIVE_IDS_KEY][:,:,0])
                np.testing.assert_array_equal(row[NATIVE_WEIGHTS_KEY][:,:,0],row[EFFECTIVE_WEIGHTS_KEY][:,:,0])
                delta=row["actions"].astype(np.float64)-shadow["actions"]
                action_rms=float(np.sqrt(np.square(delta).mean()))
                np.testing.assert_allclose(row["action_rms_vs_shadow"],action_rms,rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(row["action_max_abs_vs_shadow"],np.abs(delta).max(),atol=1e-7)
                top1=float(row[EFFECTIVE_WEIGHTS_KEY][4:,:,1:].max(-1).mean())
                np.testing.assert_allclose(row["top1_combine_mean"],top1,atol=1e-7)
                changed=float(np.any(np.sort(row[EFFECTIVE_IDS_KEY][4:,:,1:],-1)!=np.sort(shadow[EFFECTIVE_IDS_KEY][4:,:,1:],-1),-1).mean())
                np.testing.assert_allclose(row["changed_top4_fraction"],changed,atol=1e-7)
                np.testing.assert_allclose(row["bias_rms"],np.sqrt(np.square(bias[4:,:,1:]).mean()),atol=2e-6)
                raw_shadow=flow_features(shadow[PROBS_KEY])[0]
                raw_effective=flow_features(effective)[0]
                root_previous=np.sqrt(normalize(previous)[4:,-1,1:])
                m0=np.linalg.norm(np.sqrt(normalize(shadow[PROBS_KEY])[4:,-1,1:])-root_previous,axis=-1).mean()
                m1=np.linalg.norm(np.sqrt(normalize(effective)[4:,-1,1:])-root_previous,axis=-1).mean()
                measures.append([action_rms,float(np.sqrt(np.square(delta[:,:6]).mean())),
                    float(np.mean(np.sign(row["actions"][:,6])!=np.sign(shadow["actions"][:,6]))),
                    top1,float(row["bias_rms"]),changed,float(raw_shadow[0]-raw_effective[0]),
                    float(raw_effective[1]/max(raw_shadow[1],1e-12)),float(m1/max(m0,1e-12))])
                previous=shadow[PROBS_KEY]
            else:
                route_check(row)
                shadow=row
                effective=previous=row[PROBS_KEY]
            if index==0:
                for key in (*ALL_FIELDS,"actions"):
                    np.testing.assert_array_equal(shadow[key],native[0][key],err_msg="Historical native reference")
                np.testing.assert_array_equal(row["input_sha256"],native[0]["input_sha256"])
            status=effective_monitor.update(effective)
            native_status=native_monitor.update(row[PROBS_KEY])
            np.testing.assert_allclose(row["v8_effective_raw"],status["v8_raw"],atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(row["v8_effective_scores"],status["v8_scores"],atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(row["v8_effective_freeze"],status["freeze_score"],atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(row["alarm_scores"],[native_status[k] for k in ("freeze_score","acceleration_score","periodicity_score")],atol=2e-6,rtol=2e-5)
            if index==ARMS[arm]["duration"]+6:
                release=[float(status["v8_raw"][0]),float(status["v8_raw"][1]),float(status["freeze_score"])]
            if arm=="combined1_5":
                require(index<len(weak), "Weak suffix too long")
                for key in weak[index]:
                    if key not in ("inference_seconds","environment_seconds"):
                        np.testing.assert_array_equal(row[key],weak[index][key],err_msg="Complete weak reference")
            steps+=count
            before=row["sim_after"]
        require(steps==520 or bool(suffix[-1]["success"]), "Truncated complete suffix")
        require(branch["action_steps"]==steps-saved["action_steps"] and branch["success"]==bool(suffix[-1]["success"]), "Final label/count")
        require(branch["deployment_model_queries"]==len(suffix)+len(controls), "Missing shadow cost")
        if arm=="combined1_5":
            require(len(suffix)==len(weak), "Weak suffix length differs")
        prefix_rows[arm]=suffix[:5]
        total+=len(suffix)+len(controls)
        outputs.append(dict(arm=arm,replicate=rep,success=branch["success"],queries=len(suffix),
            controlled_queries=len(controls),actual_model_queries=branch["deployment_model_queries"],
            first_query=measures[0],first5_mean=np.mean(measures[:5],axis=0).tolist(),
            control_mean=np.mean(measures,axis=0).tolist(),released_features=release))
    for multiplier in (1,4,16):
        short,long=prefix_rows["combined%d_5"%multiplier],prefix_rows["combined%d_20"%multiplier]
        require(len(short)==len(long), "Duration pair prefix lengths differ")
        for a,b in zip(short,long):
            for key in a:
                if key not in ("inference_seconds","environment_seconds"):
                    np.testing.assert_array_equal(a[key],b[key],err_msg="Duration pair first5 "+key)
    require(total==result["actual_model_queries"], "Job forward ledger")
    return dict(main_id=task["main_id"],job_id=job["job_id"],replicate=rep,benchmark=task["benchmark"],
        category=task["category"],analysis_role=task["analysis_role"],active_heads=task["active_heads"],native_success=original["success"],
        reference_native_success=bool(native[-1]["success"]),reference_weak_success=bool(weak[-1]["success"]),
        first_v8_alarm=task["first_v8_alarm"],branches=outputs,actual_model_queries=total,
        arithmetic_dtypes=[list(t) for t in sorted(dtypes_seen)])


def run(args):
    plan=json.loads((args.run/"plan.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    require(summary["status"]=="completed" and plan["protocol"]==PROTOCOL, "Incomplete run")
    require(summary["plan_sha256"]==digest(args.run/"plan.json"), "Plan changed")
    for name,expected in summary["sources"].items():
        require(digest(args.run/"sources"/name)==expected==digest(Path(__file__).parent/name), "Runtime source changed: "+name)
    for path,expected in plan["source_sha256"].items():
        require(digest(path)==expected, "Frozen upstream source changed")
    require(summary["temporary_models_stopped"] and summary["environment_workers_stopped"] and not summary["live_replica_pids_after_cleanup"], "Incomplete cleanup")
    require(all(not Path("/proc/%d"%r["pid"]).exists() for r in summary["tasks"]), "Live environment worker")
    for check in summary["model_equivalence"]:
        require(check["zero_strength_exact"] and check["release_exact"] and check["actions_and_top4_exact"], "Failed model preflight")
    tasks={t["main_id"]:t for t in plan["tasks"]}
    payloads=[(str(args.run),tasks[j["main_id"]],j) for j in plan["jobs"]]
    audited=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(audit_job,payloads):
            audited.append(row)
            print("AUDITED "+row["job_id"],flush=True)
    total=sum(r["actual_model_queries"] for r in audited)
    require(total==summary["actual_model_queries"], "Run query accounting")
    output=dict(status="passed",run=str(args.run.resolve()),auditor_sha256=digest(__file__),jobs=audited,
        parents=len(tasks),branches=sum(len(r["branches"]) for r in audited),actual_model_queries=total,
        weak_complete_reproductions=len(audited),duration_prefix_equivalences=3*len(audited),
        feature_columns=["action_rms","continuous_action_rms","gripper_sign_flip_fraction","top1_combine_weight",
                         "bias_rms","changed_top4_fraction","frontback_log_ratio_gain","curvature_ratio","mobility_ratio"],
        numerical_audit="reconstruct BF16 round-to-nearest-even bias/addition and FP64 softmax; verify real dispatched top4/weights",
        hidden_capture=False,gpu6_used=False,new_main_coverage=0)
    atomic_json(args.output,output)
    print(json.dumps({k:output[k] for k in ("status","parents","branches","actual_model_queries")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args())
