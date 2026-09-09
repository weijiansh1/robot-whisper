"""Evaluate sealed routing kNN predictions using the unchanged Round5 protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import HERE, ROOT, METHODS
from core import RUNS, digest, write_json, trajectory_peak
from evaluate import attach_labels, ranking, alarm_metrics, PHYSICAL
from run_jaccard import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round6_jaccard_knn")
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output, baseline, parent = args.input.resolve(), args.baseline.resolve(), args.parent.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for category, prefix in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(prefix / name) == expected, name
    frame = attach_labels(pd.read_csv(output / "index.csv"))
    frame.to_csv(output / "outcome_alignment.csv", index=False)
    common = frame.loc[frame.run_id == RUNS[1]].groupby("task").length.min().to_dict()
    events = pd.read_csv(parent / "physical_events.csv").set_index("global_row", drop=False)
    records = {name: [] for name in ("ranking_metrics", "task_ranking_metrics", "alarm_metrics", "task_alarm_metrics", "physical_timing")}
    decisions = []
    for info in manifest["folds"]:
        fold = info["fold"]
        data = load(output / "predictions" / f"{fold}.npz")
        old = load(baseline / "predictions" / f"{fold}.npz")
        for key in ("reference_rows", "calibration_rows", "test_rows", "test_unseen"):
            np.testing.assert_array_equal(data[key], old[key])
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        metadata = {key: info[key] for key in ("fold", "suite", "seed")}
        metadata["origin"] = "round6_jaccard"
        common_end = part.task.map(common).to_numpy()
        valid = np.arange(52)[None] < part.length.to_numpy()[:, None]
        for mi, method in enumerate(data["methods"].astype(str)):
            raw = data["scores"][mi]
            assert not np.isfinite(raw[~valid]).any() and not np.isfinite(raw[:, :7]).any()
            views = [("full", trajectory_peak(raw), np.ones(len(part), bool)),
                ("common_horizon", trajectory_peak(np.where(np.arange(52)[None] < common_end[:,None], raw, np.nan)), np.ones(len(part), bool))]
            views.extend((f"q{q}", trajectory_peak(raw[:, :q+1]), part.length.to_numpy() > q) for q in (7,14,21))
            for view, score, present in views:
                for scope in ("seen", "unseen"):
                    mask = present & (part.scope.to_numpy() == scope)
                    group = part.loc[mask].reset_index(drop=True)
                    if not len(group):
                        continue
                    records["ranking_metrics"].append(dict(metadata,method=method,view=view,scope=scope,**ranking(group,score[mask])))
                    if view in ("full", "common_horizon"):
                        for task, ix in group.groupby("task").indices.items():
                            records["task_ranking_metrics"].append(dict(metadata,method=method,view=view,scope=scope,task=task,
                                **ranking(group.iloc[ix].reset_index(drop=True),score[mask][ix])))
            for ki, kind in enumerate(("episode", "task_init")):
                for ai, alpha in enumerate(data["alphas"]):
                    first = data["first"][ki,ai,mi]
                    assert not ((first < -1) | (first >= part.length.to_numpy())).any()
                    for scope in ("seen", "unseen"):
                        mask = part.scope.to_numpy() == scope
                        records["alarm_metrics"].append(dict(metadata,method=method,calibration=kind,alpha=alpha,scope=scope,
                            **alarm_metrics(part.loc[mask],first[mask])))
                        if np.isclose(alpha,.05):
                            for task, ix in part.loc[mask].groupby("task").indices.items():
                                records["task_alarm_metrics"].append(dict(metadata,method=method,calibration=kind,alpha=alpha,scope=scope,task=task,
                                    **alarm_metrics(part.loc[mask].iloc[ix],first[mask][ix])))
                    if np.isclose(alpha,.05):
                        for i,row in enumerate(part.itertuples()):
                            decisions.append(dict(metadata,method=method,calibration=kind,global_row=row.global_row,
                                scope=row.scope,task=row.task,episode=row.episode,length=row.length,failure=row.failure,first_alarm=int(first[i])))
                            if row.global_row in events.index:
                                event=events.loc[row.global_row]
                                records["physical_timing"].append(dict(metadata,method=method,calibration=kind,scope=row.scope,
                                    global_row=row.global_row,first_alarm=int(first[i]),drop_goal_release=event.drop_goal_release,
                                    failed_goal_release=event.failed_goal_release,reason=event.reason))
        print(f"EVALUATED {fold}",flush=True)
    previous=json.loads((baseline / "evaluation_summary.json").read_text())
    inputs = {str((parent / "physical_events.csv").relative_to(ROOT)): digest(parent / "physical_events.csv")}
    for name,rows in records.items():
        path=baseline / f"{name}.csv"
        assert digest(path) == previous["artifacts"][f"{name}.csv"]
        inputs[str(path.relative_to(ROOT))]=digest(path)
        pd.concat((pd.read_csv(path),pd.DataFrame(rows)),ignore_index=True).to_csv(output / f"{name}.csv",index=False)
    pd.DataFrame(decisions).to_csv(output / "episode_decisions.csv",index=False)
    write_json(output / "evaluation_summary.json", {"inputs":inputs,"common_horizons":common,
        "evaluator_sha256":digest(Path(__file__)), "shared_evaluator_sha256":digest(HERE.parent / "safe_protocol/evaluate.py"),
        "sealed_manifest_sha256":digest(output / "sealed_manifest.json"),"test_labels_sha256":digest(PHYSICAL / "episodes.csv"),
        "artifacts":{f"{name}.csv":digest(output / f"{name}.csv") for name in (*records,"episode_decisions")},
        "new_methods":METHODS,"historically_explored_data":True,"nominal_fpr_guarantee_on_unseen_tasks":False})
    alarms=pd.read_csv(output / "alarm_metrics.csv")
    main=alarms.loc[(alarms.scope=="unseen") & (alarms.calibration=="task_init") & np.isclose(alarms.alpha,.05)]
    summary=main.groupby("method")[["recall","fpr","t_det"]].mean()
    summary.to_csv(output / "unseen_group5_summary.csv")
    ranks=pd.read_csv(output / "ranking_metrics.csv")
    common_ranks=ranks.loc[(ranks.scope=="unseen") & (ranks.view=="common_horizon")]
    common_ranks.groupby("method")[["task_macro_auc","within_init_macro_auc","scorable_fraction"]].mean().to_csv(output / "common_horizon_summary.csv")
    print(summary.loc[["dyn_success_knn_k20","route_success_knn_k20",*METHODS,"eef_motion_low"]].to_string(),flush=True)


if __name__ == "__main__":
    main()
