#!/usr/bin/env python3
"""Check feature/choice compatibility inside the exact simulator Python runtime."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, records
from moe_outcome_models import MODELS, FeatureHistory, choose


def probe(args):
    data = np.load(args.model.with_suffix(".npz"))
    fitted = json.loads(args.model.read_text())
    old = json.loads(Path("/home/jovyan/work/himoe-vla/moe-trap-control/design/control_bank_plan_20260909.json").read_text())
    maximum,maximum_score_error,results = 0.,0.,[]
    for index,parent in enumerate(data["event_parent"]):
        task = old["cohort"][int(parent)]
        q0 = task["events"][0]["start_query"]
        history = FeatureHistory()
        for row in records(Path(task["parent_directory"])/"main"):
            feature = history.update(row[PROBS_KEY],int(row["action_steps_before"]))
            if int(row["query"]) == q0:
                break
        expected = data["event_x"][index]
        maximum = max(maximum,float(np.max(np.abs(feature-expected))))
        np.testing.assert_allclose(feature,expected,rtol=0,atol=3e-5)
        operators = {}
        for method in MODELS+("clock_knn",):
            actual,scores = choose(fitted["model"],method,feature)
            ref,reference = choose(fitted["model"],method,expected)
            if actual != ref:
                raise ValueError("Runtime feature difference changes controller")
            maximum_score_error = max(maximum_score_error,float(np.max(np.abs(scores-reference))))
            np.testing.assert_allclose(scores,reference,rtol=0,atol=3e-5)
            operators[method] = actual
        results.append(dict(main_id=task["main_id"],operators=operators))
    atomic_json(args.output,dict(status="passed",maximum_feature_error=maximum,parents=len(results),
        maximum_score_error=maximum_score_error,canonical_online_environment="original simulator Python and NumPy",
        policy_queries=0,simulator_steps=0,results=results))
    print(json.dumps(dict(status="passed",maximum_feature_error=maximum,maximum_score_error=maximum_score_error,parents=len(results))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    if args.worker:
        probe(args)
    else:
        from native_long_runtime import BASE,ROOT,environment
        subprocess.run([str(BASE/"envs/libero/bin/python"),str(Path(__file__).resolve()),"--worker",
            "--model",str(args.model.resolve()),"--output",str(args.output.resolve())],
            env=environment(0),cwd=ROOT,check=True)
