"""Evaluate sealed boundary scores and the matching historical baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from knn import HERE, ROOT, PRIMARY
from core import RUNS, digest, write_json, trajectory_peak
from evaluate import attach_labels, ranking, alarm_metrics, PHYSICAL


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output, parent = args.input.resolve(), args.parent.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    for name, expected in manifest["sources"].items():
        assert digest(ROOT / name) == expected, name
    frame = attach_labels(pd.read_csv(output / "index.csv"))
    frame.to_csv(output / "outcome_alignment.csv", index=False)
    common = frame.loc[frame.run_id == RUNS[1]].groupby("task").length.min().to_dict()
    events = pd.read_csv(parent / "physical_events.csv").set_index("global_row", drop=False)
    baselines = {name: json.loads((parent / name / "sealed_manifest.json").read_text()) for name in ("v7", "followup")}
    inputs = {str((parent / "physical_events.csv").relative_to(ROOT)): digest(parent / "physical_events.csv")}
    ranks, task_ranks, alarms, task_alarms, physical = [], [], [], [], []

    def record_alarms(info, part, method, calibration, alpha, first, origin):
        if ((first >= part.length.to_numpy()) | (first < -1)).any():
            raise AssertionError("invalid first alarm query")
        metadata = {k: info[k] for k in ("fold", "suite", "seed")}
        for scope in ("seen", "unseen"):
            mask = part.scope.to_numpy() == scope
            if not mask.any():
                continue
            alarms.append(dict(metadata, method=method, calibration=calibration, alpha=alpha,
                scope=scope, origin=origin, **alarm_metrics(part.loc[mask], first[mask])))
            if np.isclose(alpha, .05):
                for task, ix in part.loc[mask].groupby("task").indices.items():
                    group = part.loc[mask].iloc[ix]
                    task_alarms.append(dict(metadata, method=method, calibration=calibration, alpha=alpha,
                        scope=scope, task=task, origin=origin, **alarm_metrics(group, first[mask][ix])))
        if np.isclose(alpha, .05):
            for i, row in enumerate(part.itertuples()):
                if row.global_row in events.index:
                    event = events.loc[row.global_row]
                    physical.append(dict(metadata, method=method, calibration=calibration, scope=row.scope,
                        global_row=row.global_row, first_alarm=int(first[i]), drop_goal_release=event.drop_goal_release,
                        failed_goal_release=event.failed_goal_release, reason=event.reason, origin=origin))

    def process_scores(info, part, data, chosen, origin):
        metadata = {k: info[k] for k in ("fold", "suite", "seed")}
        common_end = part.task.map(common).to_numpy()
        valid = np.arange(52)[None] < part.length.to_numpy()[:, None]
        for method_i, method in enumerate(data["methods"].astype(str)):
            if method not in chosen:
                continue
            raw = data["scores"][method_i]
            assert not np.isfinite(raw[~valid]).any()
            views = [("full", trajectory_peak(raw), np.ones(len(part), bool)),
                ("common_horizon", trajectory_peak(np.where(np.arange(52)[None] < common_end[:, None], raw, np.nan)), np.ones(len(part), bool))]
            views.extend((f"q{q}", trajectory_peak(raw[:, :q + 1]), part.length.to_numpy() > q) for q in (7, 14, 21))
            for view, values, present in views:
                for scope in ("seen", "unseen"):
                    selected = present & (part.scope.to_numpy() == scope)
                    if not selected.any():
                        continue
                    group = part.loc[selected].reset_index(drop=True)
                    ranks.append(dict(metadata, method=method, view=view, scope=scope, origin=origin,
                        **ranking(group, values[selected])))
                    if view in ("common_horizon", "full"):
                        for task, ix in group.groupby("task").indices.items():
                            task_ranks.append(dict(metadata, method=method, view=view, scope=scope, task=task,
                                origin=origin, **ranking(group.iloc[ix].reset_index(drop=True), values[selected][ix])))
            for kind_i, calibration in enumerate(("episode", "task_init")):
                for alpha_i, alpha in enumerate(data["alphas"]):
                    record_alarms(info, part, method, calibration, float(alpha),
                        data["first"][kind_i, alpha_i, method_i], origin)

    for info in manifest["folds"]:
        fold = info["fold"]
        data = load_npz(output / "predictions" / f"{fold}.npz")
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        process_scores(info, part, data, set(data["methods"].astype(str)), "round5")
        for name, chosen in (("v7", {"v7_guard_constant", "v7_turbulence_constant", "v7_success_fusion_constant"}),
                             ("followup", {"success_only__current", "success_only__cumsum"})):
            relative = f"predictions/{fold}.npz"
            path = parent / name / relative
            expected = baselines[name]["artifacts"][relative]
            assert digest(path) == expected
            inputs[str(path.relative_to(ROOT))] = expected
            old = load_npz(path)
            for field in ("reference_rows", "calibration_rows", "test_rows", "test_unseen"):
                np.testing.assert_array_equal(old[field], data[field])
            process_scores(info, part, old, chosen, f"round3_{name}")
        relative = f"budget_predictions/{fold}.npz"
        path = parent / "v7" / relative
        expected = baselines["v7"]["artifacts"][relative]
        assert digest(path) == expected
        inputs[str(path.relative_to(ROOT))] = expected
        budget = load_npz(path)
        np.testing.assert_array_equal(budget["test_rows"], data["test_rows"])
        np.testing.assert_array_equal(budget["test_unseen"], data["test_unseen"])
        for kind_i, kind in enumerate(("unlabeled_reference_budget", "success_calibration_budget")):
            for alpha_i, alpha in enumerate(budget["alphas"]):
                record_alarms(info, part, f"v7_{kind}", kind, float(alpha), budget["first"][kind_i, alpha_i], "round3_v7_budget")
        print(f"EVALUATED {fold}", flush=True)
    tables = {"ranking_metrics": ranks, "task_ranking_metrics": task_ranks, "alarm_metrics": alarms,
              "task_alarm_metrics": task_alarms, "physical_timing": physical}
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    write_json(output / "evaluation_summary.json", {"inputs": inputs,
        "test_labels_sha256": digest(PHYSICAL / "episodes.csv"),
        "evaluator_sha256": digest(Path(__file__)), "shared_evaluator_sha256": digest(HERE.parent / "safe_protocol/evaluate.py"),
        "sealed_manifest_sha256": digest(output / "sealed_manifest.json"),
        "artifacts": {f"{name}.csv": digest(output / f"{name}.csv") for name in tables},
        "rows": {name: len(records) for name, records in tables.items()}, "common_horizons": common,
        "primary": PRIMARY, "historically_explored_data": True, "nominal_fpr_guarantee_on_unseen_tasks": False})
    print("BOUNDARY/KNN EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
