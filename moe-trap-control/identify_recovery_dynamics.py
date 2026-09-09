#!/usr/bin/env python3
"""Identify local Cartesian ARX response without reading terminal outcome labels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from collection_protocol import stable_id
from collection_storage import atomic_json, digest, records
from recovery_mpc import PROTOCOL, SETTINGS, certificate


def fit(rows):
    velocity, control, next_velocity = [np.asarray([r[key] for r in rows]) for key in ("velocity", "control", "next_velocity")]
    coefficients, diagnostics = [], []
    counts = defaultdict(int)
    for row in rows:
        counts[row["main_id"]] += 1
    weights = np.sqrt([1/counts[row["main_id"]] for row in rows])
    for axis in range(3):
        X = np.column_stack((velocity[:,axis],control[:,axis]))
        weighted = X*weights[:,None]
        solved = lsq_linear(weighted,next_velocity[:,axis]*weights,bounds=([-.95,.02],[.99,2.]),tol=1e-12)
        coefficients.append(solved.x)
        diagnostics.append(dict(rank=int(np.linalg.matrix_rank(weighted)),
            singular_values=np.linalg.svd(weighted,compute_uv=False).tolist(),
            condition=float(np.linalg.cond(weighted)),active_constraints=solved.active_mask.tolist()))
    a,b = np.asarray(coefficients).T
    return dict(a=a.tolist(),b=b.tolist(),fit_diagnostics=diagnostics)


def evaluate(rows, model):
    a,b = np.asarray(model["a"]),np.asarray(model["b"])
    by_parent = defaultdict(list)
    errors, norm_errors, signals = defaultdict(list),defaultdict(list),[]
    for row in rows:
        truth = row["next_velocity"]
        predictions = dict(arx=a*row["velocity"]+b*row["control"],
                           persistence=row["velocity"],kinematic=row["control"],zero=np.zeros(3))
        signals.append(truth)
        for name,prediction in predictions.items():
            delta = prediction-truth
            errors[name].append(np.square(delta))
            norm_errors[name].append(np.linalg.norm(delta))
        by_parent[row["main_id"]].append(len(signals)-1)
    outputs={}
    for name,err in errors.items():
        err=np.asarray(err)
        parent_mse=[err[indices].mean(0) for indices in by_parent.values()]
        outputs[name]=dict(equal_parent_axis_rmse_mm=(10*np.sqrt(np.mean(parent_mse,axis=0))).tolist(),
            equal_parent_vector_rmse_mm=float(10*np.sqrt(np.mean(parent_mse,axis=0).sum())),
            transition_vector_error_p95_mm=float(10*np.quantile(norm_errors[name],.95)),
            transition_vector_error_max_mm=float(10*np.max(norm_errors[name])))
    return dict(parents=len(by_parent),transitions=len(rows),models=outputs)


def run(args):
    audit=json.loads(args.audit.read_text())
    if audit["status"]!="passed":
        raise ValueError("Audited physical data required")
    run=Path(audit["run"])
    rows, source_files = [], [args.audit,run/"plan.json",run/"summary.json"]
    for task in audit["tasks"]:
        for event in task["events"]:
            for arm in ("hold","withdraw"):
                path=run/"events"/event["event_id"] / "branches" / arm
                branch=json.loads((path/"branch.json").read_text())
                physical=list(records(path/"physical"))
                source_files.extend((path/"branch.json",path/"physical/manifest.json"))
                scale=np.asarray(branch["controller_translation_scale"])
                for previous,current in zip(physical,physical[1:]):
                    rows.append(dict(main_id=task["main_id"],base_task=task["base_task"],benchmark=task["benchmark"],
                        velocity=100*(previous["eef_after"]-previous["eef_before"]),
                        control=100*current["action"][:3]*scale,
                        next_velocity=100*(current["eef_after"]-current["eef_before"])))
    base_tasks=sorted({r["base_task"] for r in rows},key=lambda t:stable_id(PROTOCOL,"dynamics_split",t))
    heldout=set(base_tasks[-3:])
    training=[r for r in rows if r["base_task"] not in heldout]
    validation=[r for r in rows if r["base_task"] in heldout]
    trial=fit(training)
    final=fit(rows)
    proof=certificate(final["a"],final["b"])
    if (proof["closed_loop_spectral_radius"]>=1 or proof["minimum_P_eigenvalue"]<=0 or
            proof["riccati_identity_max_residual"]>1e-8 or proof["controllability_rank"]!=6):
        raise ValueError("Nominal model certificate failed")
    report=dict(status="completed",protocol=PROTOCOL,identifier_sha256=digest(__file__),
        source_sha256={str(p.resolve()):digest(p) for p in source_files},
        development_main_ids=sorted(t["main_id"] for t in audit["tasks"]),
        physical_data_main_ids=sorted({r["main_id"] for r in rows}),
        equations="v[t+1]=diag(a)*v[t]+diag(b)*u[t]+w[t]; p[t+1]=p[t]+v[t+1]",
        units="cm, cm per environment step; u is actual Cartesian delta requested from OSC",
        fitting="one global diagonal ARX, bounds a in [-0.95,0.99], b in [0.02,2]; equal weight per original main; no outcome fitting",
        split="fixed-hash base-task split, final 3 tasks held out for model diagnostic; final deployment model then refit on all prior physical data",
        diagnostic_training_tasks=base_tasks[:-3],diagnostic_heldout_tasks=base_tasks[-3:],
        diagnostic_model=trial,diagnostic_training=evaluate(training,trial),diagnostic_heldout=evaluate(validation,trial),
        final_fit=evaluate(rows,final),a=final["a"],b=final["b"],fit_diagnostics=final["fit_diagnostics"],
        nominal_certificate=proof,settings=SETTINGS,
        limitations="closed-loop observational identification with directional excitation; no object/orientation/gripper dynamics; measured residuals are not worst-case robust bounds",
        final_model_evaluation="future controller experiment must exclude every development_main_id")
    if args.output.exists():
        raise ValueError("Refusing to replace identified model")
    atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ("a","b","diagnostic_heldout","fit_diagnostics")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
