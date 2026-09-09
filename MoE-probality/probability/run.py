"""Reproduce the complete retrospective audit with `python -m probability.run`."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .branches import branch_candidates, summarize_branches
from .data import build_dataset, json_records, sha256, write_json
from .evaluation import MODELS, alarm_sensitivity, alarm_statistics, evaluate, matched_alarm_controls
from .features import detector
from .model import fit_readout, predict_readout
from .report import make_figures, make_report
from .verify import verify_raw


HERE = Path(__file__).resolve().parents[1]


def write_manifest(output):
    hashes = {str(path.relative_to(HERE)): sha256(path) for path in sorted(HERE.glob("probability/*.py"))}
    hashes.update({str(path.relative_to(HERE)): sha256(path) for path in sorted(HERE.glob("tests/*.py"))})
    hashes.update({str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                   if path.is_file() and path.name != "artifact_manifest.json"})
    write_json(output / "artifact_manifest.json", hashes)


def render_saved(output):
    def load_json(name):
        return json.loads((output / name).read_text())

    episodes = pd.read_csv(output / "episodes.csv")
    scores = pd.read_csv(output / "metrics.csv")
    sensitivity = pd.read_csv(output / "alarm_rule_sensitivity.csv")
    predictions = pd.read_csv(output / "predictions.csv.gz")
    make_figures(output, scores, pd.read_csv(output / "calibration.csv"), sensitivity, predictions, episodes)
    make_report(output, load_json("data_audit.json"), scores, pd.read_csv(output / "alarm_success.csv"),
                pd.read_csv(output / "cluster_intervals.csv"), load_json("raw_verification.json"),
                pd.read_csv(output / "branch_candidates.csv"), sensitivity)
    write_manifest(output)


def run(hub, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    episodes, x, ep, q, audit = build_dataset(hub, output)
    index = episodes.set_index("episode_row")
    suite = index.loc[ep, "suite"].to_numpy()
    split = index.loc[ep, "split"].to_numpy()
    y = index.loc[ep, "success"].to_numpy(int)
    predictions = pd.DataFrame(dict(episode_row=ep, query=q,
                                    remaining_action_steps=x[:, 0].astype(int)))
    for name in MODELS:
        predictions[name] = np.nan
    bundles = {}
    for key in sorted(episodes.suite.unique()):
        train = (suite == key) & (split == "train")
        cal = (suite == key) & (split == "calibration")
        take = suite == key
        print(f"fit {key}: {train.sum()} train queries, {cal.sum()} calibration queries", flush=True)
        bundle = fit_readout(x[train], y[train], x[cal], y[cal], audit["policy_by_suite"][key])
        bundles[key] = bundle
        joblib.dump(bundle, output / "models" / (key + ".joblib"), compress=3)
        for name, value in predict_readout(bundle, x[take]).items():
            predictions.loc[take, name] = value
    if predictions[list(MODELS)].isna().any().any():
        raise AssertionError("missing probability predictions")
    print("evaluate held-out probabilities and cluster intervals", flush=True)
    scores, reliability, uncertainty = evaluate(predictions, episodes)
    alarms = alarm_statistics(episodes)
    sensitivity = alarm_sensitivity(episodes)
    controls = matched_alarm_controls(episodes)
    candidates = branch_candidates(episodes)
    half_episodes = episodes.copy()
    half_episodes["first_alarm_q"] = half_episodes.first_q_freeze_back_half_k4
    half_candidates = branch_candidates(half_episodes)
    scores.to_csv(output / "metrics.csv", index=False)
    reliability.to_csv(output / "calibration.csv", index=False)
    uncertainty.to_csv(output / "cluster_intervals.csv", index=False)
    alarms.to_csv(output / "alarm_success.csv", index=False)
    sensitivity.to_csv(output / "alarm_rule_sensitivity.csv", index=False)
    alarm_statistics(half_episodes).to_csv(output / "alarm_success_half_k4.csv", index=False)
    controls.to_csv(output / "matched_controls.csv", index=False)
    candidates.to_csv(output / "branch_candidates.csv", index=False)
    half_candidates.to_csv(output / "branch_candidates_half_k4.csv", index=False)
    episodes.groupby(["split", "suite"]).agg(episodes=("episode_row", "size"),
                    successes=("success", "sum"), queries=("length", "sum")).to_csv(output / "split_counts.csv")
    predictions.to_csv(output / "predictions.csv.gz", index=False, float_format="%.10g")
    first = predictions.merge(episodes[["episode_row", "first_alarm_q"]], on="episode_row", validate="many_to_one")
    first[first["query"] == first.first_alarm_q].drop(columns="first_alarm_q").to_csv(output / "first_alarm_predictions.csv", index=False)
    print("verify features and online predictions against raw routes", flush=True)
    raw = verify_raw(hub, episodes, x, ep, bundles)
    write_json(output / "raw_verification.json", raw)
    make_figures(output, scores, reliability, sensitivity, predictions, episodes)
    make_report(output, audit, scores, alarms, uncertainty, raw, candidates, sensitivity)
    primary = scores[(scores.split == "test_unseen_init") & (scores.suite == "all")
                     & scores.selection.isin(["all_queries", "first_alarm"])]
    summary = dict(episodes=audit["episodes"], queries=audit["queries"], successes=audit["successes"],
                   primary_rule=asdict(detector.PRIMARY), primary_metrics=json_records(primary),
                   alarm_success=json_records(alarms[alarms.suite == "all"]),
                   alarm_rule_sensitivity=json_records(sensitivity),
                   cluster_intervals=json_records(uncertainty), rho_available=False,
                   branch_candidates=len(candidates), actual_branch_rollouts=0,
                   supplementary_half_k4_branch_candidates=len(half_candidates),
                   raw_verification_passed=True,
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    pandas=pd.__version__, sklearn=sklearn.__version__))
    write_json(output / "summary.json", summary)
    write_manifest(output)
    print(primary[["selection", "model", "brier", "log_loss", "auroc"]].to_string(index=False), flush=True)
    print(f"report: {output / 'REPORT.zh.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=HERE.parent / "VLA_MUI_HUB")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--branch-results", type=Path, help="validate a completed aligned branch CSV instead of training")
    parser.add_argument("--render-only", action="store_true", help="regenerate figures/report from saved measurements")
    args = parser.parse_args()
    if args.branch_results and args.render_only:
        parser.error("--branch-results and --render-only are mutually exclusive")
    if args.render_only:
        render_saved(args.output.resolve())
    elif args.branch_results:
        args.output.mkdir(parents=True, exist_ok=True)
        summarize_branches(pd.read_csv(args.branch_results)).to_csv(args.output / "branch_soft_labels.csv", index=False)
    else:
        with threadpool_limits(limits=args.threads):
            run(args.hub.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
