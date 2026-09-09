#!/usr/bin/env python3
"""Audit exact pairs, constrained MPC decisions, observer causality and complete outcomes."""

import argparse
import ast
import concurrent.futures
import copy
import json
from pathlib import Path

import numpy as np

from audit_long_continuation import require, route_check
from collection_storage import atomic_json, digest, load_snapshot, records
from collection_routes import PROBS_KEY
from collection_protocol import stable_id
from fixed_recovery_control import MODULE, METHODS, STALL, TriggerMonitor, targets, PROTOCOL as TRIGGER_PROTOCOL
from recovery_mpc import PROTOCOL, ARMS, SETTINGS, RecoveryMPC


def runtime_sources(run, sources):
    drift = []
    for name, expected in sources.items():
        archived, current = run/"sources"/name, Path(__file__).parent/name
        require(digest(archived)==expected,"Archived runtime changed: "+name)
        actual = digest(current)
        if actual == expected:
            continue
        require(name=="run_collection_preflight.py","Runtime changed: "+name)
        normalized = []
        for path in (archived, current):
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.FunctionDef) and node.name=="run" or
                isinstance(node, ast.If) and ast.dump(node.test)==ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body))]
            normalized.append(ast.dump(tree, include_attributes=False))
        require(normalized[0]==normalized[1],"Imported scheduler helpers changed")
        runner = ast.parse((run/"sources/run_mpc_recovery.py").read_text())
        imports = [alias.name for node in ast.walk(runner) if isinstance(node, ast.ImportFrom)
                   and node.module=="run_collection_preflight" for alias in node.names]
        require("run" not in imports and "*" not in imports,"Modified scheduler entrypoint used")
        drift.append(dict(file=name, archived_sha256=expected, current_sha256=actual,
            used_imports=imports, imported_helpers_and_module_initialization_ast_identical=True,
            scope="only unused run(args) and __main__ CLI differ; separate repair entry added during this run"))
    return drift


def stall_check(positions):
    if len(positions)<6:
        return False
    return max(np.linalg.norm(a-b) for a in positions[-6:] for b in positions[-6:])<=.005



def ball(x, radius=1.):
    return np.asarray(x)/np.maximum(1.,np.linalg.norm(x,axis=-1,keepdims=True)/radius)


def control_check(mpc,row,arm,position,target,velocity,observer,scale):
    state=np.r_[100*(position-target),velocity]
    d=observer if arm=="mpc_disturbance" else np.zeros(3)
    u=100*row["action"][:3]*scale
    observed_velocity=100*(row["eef_after"]-position)
    following=np.r_[100*(row["eef_after"]-target),observed_velocity]
    predicted=mpc.A@state+mpc.B@u+mpc.D@d
    innovation=observed_velocity-mpc.a*velocity-mpc.b*u
    updated=ball(.5*observer+.5*innovation,.5)
    expected=dict(state_cm=state,velocity_cm=velocity,input_cm=u,disturbance_cm=d,
        observer_before_cm=observer,observer_after_cm=updated,next_state_cm=following,
        predicted_next_state_cm=predicted,lyapunov_before=state@mpc.P@state,
        lyapunov_after=following@mpc.P@following)
    for key,value in expected.items():
        np.testing.assert_allclose(row[key],value,atol=2e-10,rtol=2e-10,err_msg=key)
    optimized=arm in ("mpc","mpc_disturbance")
    if optimized:
        plan=row["plan_cm"]
        require(plan.shape==(6,3) and np.isfinite(plan).all(),"Invalid MPC plan")
        require(np.linalg.norm(plan,axis=1).max()<=1+1e-10,"MPC future action feasibility")
        np.testing.assert_allclose(plan[0],u,atol=2e-12,rtol=0)
        x=state.copy()
        trajectory=[]
        objective=0.
        for k,control in enumerate(plan):
            x=mpc.A@x+mpc.B@control+mpc.D@d
            trajectory.append(x.copy())
            weight=mpc.P if k==5 else mpc.Q
            objective+=x@weight@x+control@mpc.R@control
        np.testing.assert_allclose(row["predicted_states_cm"],trajectory,atol=2e-10)
        np.testing.assert_allclose(row["objective"],objective,atol=2e-9)
        require(bool(row["fallback"])!=bool(row["optimizer_accepted"]),"Optimizer status inconsistency")
        if row["optimizer_accepted"]:
            require(bool(row["solver_success"]),"Accepted failed solver")
            gradient=np.zeros((6,3))
            costate=np.zeros(6)
            for k in reversed(range(6)):
                weight=mpc.P if k==5 else mpc.Q
                costate=2*weight@trajectory[k]+mpc.A.T@costate
                gradient[k]=2*mpc.R@plan[k]+mpc.B.T@costate
            residual=float(np.abs(plan-ball(plan-gradient/mpc.lipschitz)).max())
            require(residual<=SETTINGS["projected_gradient_tolerance"]+1e-10,"MPC stationarity")
            np.testing.assert_allclose(row["projected_gradient_residual"],residual,atol=2e-10)
        else:
            np.testing.assert_allclose(plan[0],ball(-state[:3]),atol=1e-12)
            np.testing.assert_array_equal(plan[1:],np.zeros((5,3)))
    else:
        require(not row["optimizer_accepted"] and not row["fallback"],"Non-MPC optimization")
    diagnostic=dict(step=int(row["query"]),phase=int(row["phase"]),
        target_error_mm=float(np.linalg.norm(following[:3])*10),
        prediction_error_mm=float(np.linalg.norm(observed_velocity-predicted[3:])*10),
        motion_mm=float(np.linalg.norm(observed_velocity)*10),input_norm_cm=float(np.linalg.norm(u)),
        lyapunov_delta=float(following@mpc.P@following-state@mpc.P@state),
        controller_seconds=float(row["controller_seconds"]),fallback=bool(row["fallback"]),
        optimizer_accepted=bool(row["optimizer_accepted"]),
        projected_gradient_residual=float(row["projected_gradient_residual"]))
    return diagnostic,observed_velocity,updated


def audit_parent(payload):
    run,task=payload
    run,parent=Path(run),Path(task["parent_directory"])
    replay=run/"replays"/task["main_id"]
    model_path=run/"dynamics_model.json"
    mpc=RecoveryMPC(model_path)
    original=json.loads((parent/"result.json").read_text())
    result=json.loads((replay/"result.json").read_text())
    require(result["status"]=="completed" and result["c0"]["status"]=="passed" and result["online_triggers_exact"],"Incomplete C0")
    require(result["protocol"]==PROTOCOL and not result["invalid_pair"],"C0 identity")
    require(result["dynamics_sha256"]==digest(model_path),"C0 dynamics commitment")
    require(result["gpu"] in (0,1,2,3) and result["render_gpu"] in (0,3) and not result["hidden_capture"],"C0 GPU/no-hidden")
    require(not result["main_intervention"] and result["reused_main"] and original["main_complete"],"Changed native main")
    for filename,key in (("main_complete.json","parent_commit_sha256"),("main/manifest.json","parent_manifest_sha256")):
        require(digest(parent/filename)==task[key]==result[key],"Main commitment")
    require(digest(replay/"environment_rng/manifest.json")==result["rng_tape_sha256"],"RNG tape commitment")
    tape=list(records(replay/"environment_rng"))
    require(len(tape)==original["action_steps"],"Missing original environment RNG states")
    main,c0=list(records(parent/"main")),list(records(replay/"c0"))
    require(len(main)==len(c0)==task["parent_queries"]==result["c0"]["compared_queries"],"Complete C0 length")
    monitor=TriggerMonitor()
    for expected,actual in zip(main,c0):
        for key in expected:
            if key not in ("inference_seconds","environment_seconds"):
                np.testing.assert_array_equal(actual[key],expected[key],err_msg="Complete C0 "+key)
        monitor.update(actual[PROBS_KEY],actual["proprio"][:3])
    require(all(task["first_alarms"][k]==v for k,v in monitor.first.items()),"Causal trigger mismatch")
    random_query=int(np.random.default_rng(int(stable_id(TRIGGER_PROTOCOL,task["main_id"],"random_time")[:8],16)).integers(8,40))
    require(task["first_alarms"]["random_time"]==random_query,"Changed random trigger schedule")
    expected_positions={}
    for method in METHODS:
        q=task["first_alarms"][method]
        if 0<=q<len(main)-1:
            expected_positions.setdefault(q+1,[]).append(method)
    require({e["start_query"]:e["methods"] for e in task["events"]}==expected_positions,"Trigger event union")
    total=len(c0)
    events=[]
    for event in result["events"]:
        q0,initial_steps=event["start_query"],event["action_steps_before"]
        require(event["methods"]==expected_positions[q0] and initial_steps==int(main[q0]["action_steps_before"]),"Event placement")
        location=replay/"events"/event["event_id"]/"snapshot"
        require(digest(location/"manifest.json")==event["snapshot_manifest_sha256"],"Snapshot commitment")
        saved=load_snapshot(location)
        require(saved["query"]==q0 and saved["action_steps"]==initial_steps,"Snapshot time")
        require(saved["controller_numeric"][0]["use_delta"],"Recovery requires delta-position control")
        directory=run/"events"/event["event_id"]
        outcome=json.loads((directory/"result.json").read_text())
        require(outcome["status"]=="completed" and outcome["all_native_suffixes_exact"] and not outcome["invalid_pair"],"Incomplete event")
        require(outcome["events"]==[event] and outcome["protocol"]==PROTOCOL,"Event identity")
        require(outcome["gpu"] in (0,1,2,3) and outcome["render_gpu"] in (0,3),"Event GPU")
        require(not outcome["hidden_capture"] and not outcome["main_intervention"],"Hidden/main changed")
        require(outcome["c0"]["replay_result_sha256"]==digest(replay/"result.json"),"C0 result changed")
        require(outcome["c0"]["replay_branch_sha256"]==digest(replay/"c0/branch.json"),"C0 branch changed")
        for metadata in (result["model_metadata"],outcome["model_metadata"]):
            for key in ("checkpoint_sha256","normalization_stats_sha256","libero_wrist_layout",
                        "himoe_upstream_commit","himoe_working_tree_diff_sha256"):
                require(metadata[key]==original["model_metadata"][key],"Model identity")
        start=np.asarray(saved["observation"]["robot0_eef_pos"],float)
        target=targets(start,main[max(0,q0-3)]["proprio"][:3])
        np.testing.assert_allclose(target[0]-start,[0,0,.04],atol=1e-15)
        require(np.linalg.norm((target[1]-target[0])[:2])<=.04000001,"Retreat target bound")
        gripper=float(main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1,6])
        prefix=TriggerMonitor()
        for row in main[:q0]:
            prefix.update(row[PROBS_KEY],row["proprio"][:3])
        branches=[]
        require([b["arm"] for b in outcome["branches"]]==list(ARMS),"Missing event arms")
        event_queries=0
        for branch in outcome["branches"]:
            arm=branch["arm"]
            path=directory/"branches"/arm
            require(branch==json.loads((path/"branch.json").read_text()) and branch["status"]=="completed","Branch commitment")
            require(branch["starts_after_main_complete"] and branch["starts_after_c0"],"Premature intervention")
            physical,suffix=list(records(path/"physical")),list(records(path/"suffix"))
            np.testing.assert_allclose(branch["recovery_targets"],target,atol=1e-15,rtol=0)
            scale=np.asarray(branch["controller_translation_scale"])
            require(np.all(scale>=.01),"Invalid controller scale")
            np.testing.assert_array_equal(branch["initial_eef"],start)
            require(branch["last_native_gripper"]==gripper,"Changed gripper history")
            steps,before,success=initial_steps,main[q0]["sim_before"],False
            eef=start.copy()
            physical_seconds=0.
            velocity=100*(start-main[q0-1]["proprio"][:3])/int(main[q0-1]["executed_action_count"])
            observer=np.zeros(3)
            np.testing.assert_array_equal(branch["initial_velocity_cm"],velocity)
            require(branch["dynamics_sha256"]==digest(model_path),"Branch dynamics commitment")
            require(outcome["dynamics_sha256"]==digest(model_path),"Event dynamics commitment")
            diagnostics=[]
            for index,row in enumerate(physical):
                require(arm!="native" and int(row["query"])==index and int(row["action_step"])==steps,"Physical action placement")
                require(int(row["phase"])==index//8 and index<16,"Physical duration")
                np.testing.assert_array_equal(row["sim_before"],before)
                np.testing.assert_array_equal(row["eef_before"],eef)
                np.testing.assert_array_equal(row["target"],np.asarray(branch["recovery_targets"])[index//8])
                action=row["action"]
                np.testing.assert_array_equal(action[3:],[0.,0.,0.,1. if gripper>0 else -1.])
                error=target[index//8]-eef
                expected=(error/max(1.,np.linalg.norm(error)/.01)) if arm=="withdraw" else np.zeros(3)
                if arm in ("hold","withdraw"):
                    np.testing.assert_allclose(action[:3]*scale,expected,atol=1e-15,rtol=1e-12)
                require(np.linalg.norm(action[:3]*scale)<=.01000001 and np.max(np.abs(action))<=1.000001,"Action bound")
                diagnostic,velocity,observer=control_check(mpc,row,arm,eef,target[index//8],velocity,observer,scale)
                diagnostics.append(diagnostic)
                success=bool(row["success"])
                require(not success or index==len(physical)-1,"Physical action after success")
                steps+=1
                before,eef=row["sim_after"],row["eef_after"]
                physical_seconds+=float(row["environment_seconds"])
            require(len(physical)==branch["recovery_steps"],"Physical action count")
            require(sum(d["fallback"] for d in diagnostics)==branch["optimizer_fallbacks"],"Fallback ledger")
            np.testing.assert_allclose(sum(d["controller_seconds"] for d in diagnostics),branch["controller_seconds"],atol=1e-12)
            require((len(physical)==0 if arm=="native" else success or len(physical)==min(16,520-initial_steps)),"Short physical intervention")
            np.testing.assert_array_equal(branch["after_recovery_eef"],eef)
            require(success==branch["success_during_recovery"],"Physical success label")
            rng=np.random.default_rng()
            rng.bit_generator.state=copy.deepcopy(saved["policy_rng"])
            live=copy.deepcopy(prefix)
            positions,stall_flags=[],[]
            for index,row in enumerate(suffix):
                require(not success,"Policy continued after success")
                q=q0+index
                require(int(row["query"])==q and int(row["relative_query"])==index and int(row["action_steps_before"])==steps,"Suffix placement")
                np.testing.assert_array_equal(row["sim_before"],before)
                np.testing.assert_array_equal(row["noise"],rng.standard_normal((10,24)).astype(np.float32))
                route_check(row)
                status=live.update(row[PROBS_KEY],row["proprio"][:3])
                np.testing.assert_allclose(row["alarm_scores"],[status[k] for k in ("freeze_score","acceleration_score","periodicity_score")],atol=2e-6,rtol=2e-5)
                for key in ("knn_score","cosine_score","v8_scores"):
                    np.testing.assert_allclose(row[key],status[key],atol=2e-6,rtol=2e-5)
                require(bool(row["behavior_stall"])==status["behavior_stall"],"Behavior trigger drift")
                positions.append(row["proprio"][:3])
                require(bool(row["post_recovery_stall_valid"])==(index>=5),"Post-recovery window")
                flag=stall_check(positions)
                require(bool(row["post_recovery_stall"])==flag,"Post-recovery stall changed")
                if index>=5:
                    stall_flags.append(flag)
                count,success=int(row["executed_action_count"]),bool(row["success"])
                require(0<count<=min(10,520-steps) and (success or count==min(10,520-steps)),"Suffix execution chunk")
                if arm=="native":
                    require(q<len(main),"Native suffix too long")
                    for key in main[q]:
                        if key not in ("inference_seconds","environment_seconds"):
                            np.testing.assert_array_equal(row[key],main[q][key],err_msg="Complete native suffix "+key)
                steps+=count
                before=row["sim_after"]
            require(success or steps==520,"Truncated suffix")
            require(branch["success"]==success and branch["final_action_steps"]==steps and branch["action_steps"]==steps-initial_steps,"Final label/budget")
            require(len(suffix)==branch["queries"]==branch["actual_model_queries"],"Forward ledger")
            if arm=="native":
                require(success==original["success"] and len(suffix)==len(main)-q0,"Native full outcome")
            event_queries+=len(suffix)
            flags=np.asarray(stall_flags,bool)
            clear=np.flatnonzero(~flags)
            recurred=bool(len(clear) and flags[clear[0]+1:].any())
            branches.append(dict(arm=arm,success=success,queries=len(suffix),recovery_steps=len(physical),
                suffix_action_steps=steps-initial_steps,final_action_steps=steps,
                inference_seconds=branch["inference_seconds"],environment_seconds=branch["environment_seconds"],
                elapsed_seconds=branch["elapsed_seconds"],recovery_environment_seconds=physical_seconds,
                recovery_displacement_m=(eef-start).tolist(),
                lift_phase_displacement_m=(physical[min(7,len(physical)-1)]["eef_after"]-start).tolist() if physical else [0.,0.,0.],
                recovery_target_error_m=float(np.linalg.norm(eef-target[min(1,(len(physical)-1)//8)])) if physical else None,
                recurrence_window_available=bool(len(flags)),any_post_recovery_stall=bool(flags.any()),
                never_clear_in_observed_windows=bool(len(flags) and flags.all()),clear_then_stall=recurred,
                post_recovery_windows=len(flags),success_during_recovery=branch["success_during_recovery"],
                control_diagnostics=diagnostics,controller_seconds=branch["controller_seconds"],
                optimizer_fallbacks=branch["optimizer_fallbacks"]))
        require(event_queries==outcome["actual_model_queries"],"Event query ledger")
        total+=event_queries
        events.append(dict(event_id=event["event_id"],start_query=q0,methods=event["methods"],branches=branches))
    require(len(events)==len(task["events"]),"Missing physical events")
    return dict(main_id=task["main_id"],benchmark=task["benchmark"],category=task["category"],base_task=task["base_task"],
        analysis_role=task["analysis_role"],native_success=original["success"],native_queries=len(main),
        native_action_steps=original["action_steps"],first_alarms=task["first_alarms"],events=events,
        c0_queries=len(c0),actual_model_queries=total)


def run(args):
    plan=json.loads((args.run/"plan.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    require(summary["status"]=="completed" and plan["protocol"]==PROTOCOL and plan["module"]==MODULE and plan["behavior_trigger"]==STALL,"Incomplete/floating experiment")
    require(summary["plan_sha256"]==digest(args.run/"plan.json"),"Plan changed")
    require(plan["mpc_settings"]==SETTINGS and plan["arms"]==list(ARMS),"MPC settings drift")
    require(plan["dynamics_sha256"]==digest(args.run/"dynamics_model.json"),"Model artifact changed")
    model=json.loads((args.run/"dynamics_model.json").read_text())
    require(not set(t["main_id"] for t in plan["tasks"]) & set(model["development_main_ids"]),"Identification/evaluation leakage")
    original_audit=json.loads(Path(plan["parent_audit"]).read_text())
    from collections import defaultdict
    groups=defaultdict(list)
    for t in original_audit["tasks"]:
        if t["main_id"] not in model["development_main_ids"]:
            groups[t["benchmark"],t["category"]].append(t["main_id"])
    selected=[m for key in sorted(groups) for m in sorted(groups[key],key=lambda m:stable_id(PROTOCOL,"cohort",m))[:2]]
    require([t["main_id"] for t in plan["tasks"]]==selected,"Cohort selection changed")
    source_drift=runtime_sources(args.run,summary["sources"])
    for path,expected in plan["source_sha256"].items():
        require(digest(path)==expected,"Frozen source changed")
    require(summary["temporary_models_stopped"] and summary["environment_workers_stopped"] and not summary["live_replica_pids_after_cleanup"],"Incomplete cleanup")
    require(all(j["status"]=="completed" and not Path("/proc/%d"%j["pid"]).exists() for j in summary["tasks"]),"Incomplete job/worker cleanup")
    require(all(c["actions_and_top4_exact"] and c["concurrent_full_hb_exact"] for c in summary["model_equivalence"]),"Model preflight failed")
    audited=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(audit_parent,[(str(args.run),t) for t in plan["tasks"]]):
            audited.append(result)
            print("AUDITED "+result["main_id"],flush=True)
    require(sum(r["actual_model_queries"] for r in audited)==summary["actual_model_queries"],"Total query ledger")
    output=dict(status="passed",run=str(args.run.resolve()),auditor_sha256=digest(__file__),tasks=audited,
        parents=len(audited),events=sum(len(r["events"]) for r in audited),
        branches=sum(len(e["branches"]) for r in audited for e in r["events"]),
        actual_model_queries=summary["actual_model_queries"],c0_queries=sum(r["c0_queries"] for r in audited),
        original_rng_tapes_complete=True,native_complete_suffixes_exact=True,physical_action_bounds_checked=True,
        hidden_capture=False,gpu6_used=False,new_main_coverage=0,
        dynamics_sha256=plan["dynamics_sha256"],state_disjoint_from_identification=True,
        optimizer_and_observer_checked=True,runtime_source_drift=source_drift)
    atomic_json(args.output,output)
    print(json.dumps({k:output[k] for k in ("status","parents","events","branches","actual_model_queries")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args())
