"""Evaluate sealed scores with task-matched horizons and all online false alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from core import ROOT, RUNS, PRIMARY, digest, write_json, trajectory_peak

HERE = Path(__file__).resolve().parent
PHYSICAL = ROOT / "VLA_MUI_HUB/physical-failure-labels/results"
LANDMARKS = (0, 4, 9, 14, 19, 29, 39)
TIMING_METHODS = (PRIMARY, "history_contrast__cumsum", "behavior_contrast__cumsum", "freeze__current", "clock",
                  "stats_positive__current", "stats_ratio__cumsum", "success_only__cumsum", "success_only__current",
                  "v7_guard_constant", "v7_guard_timeband", "v7_freeze_constant", "v7_turbulence_constant",
                  "v7_success_fusion_constant")


def attach_labels(frame):
    labels = pd.read_csv(PHYSICAL / "episodes.csv")
    labels["task"] = labels.suite + "/" + labels.task_name
    labels = labels.rename(columns={"episode_index": "episode", "flow_noise_seed": "noise_seed"})
    labels = labels.loc[labels.run_id.isin(RUNS)]
    keys = ["task", "run_id", "episode", "init_state_id", "noise_seed"]
    result = frame.merge(labels[keys + ["recorded_success", "primary_failure_reason"]], on=keys,
                         how="left", validate="one_to_one", sort=False)
    if result.recorded_success.isna().any() or result.recorded_success.dtype != bool:
        raise ValueError("outcome alignment failure")
    result["failure"] = ~result.recorded_success
    return result


def finite_scores(score):
    finite = np.isfinite(score)
    floor = min(float(score[finite].min()) - 1, -1e6) if finite.any() else 0.
    return np.where(finite, score, floor)


def auc(y, score):
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return np.nan
    ranks = rankdata(finite_scores(score))
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))


def ranking(frame, score):
    y = frame.failure.to_numpy(bool)
    task_aucs, init_aucs = [], []
    for _, rows in frame.groupby("task").indices.items():
        task_aucs.append(auc(y[rows], score[rows]))
        task_frame = frame.iloc[rows]
        values = [auc(y[rows[ix]], score[rows[ix]]) for ix in task_frame.groupby("init_state_id").indices.values()]
        if np.isfinite(values).any():
            init_aucs.append(np.nanmean(values))
    return {"episodes": len(frame), "failures": int(y.sum()), "auc": auc(y, score),
            "ap": float(average_precision_score(y, finite_scores(score))) if y.any() else np.nan,
            "task_macro_auc": float(np.nanmean(task_aucs)) if np.isfinite(task_aucs).any() else np.nan,
            "within_init_macro_auc": float(np.mean(init_aucs)) if init_aucs else np.nan,
            "scorable_fraction": float(np.isfinite(score).mean()),
            "mixed_tasks": int(np.isfinite(task_aucs).sum())}


def alarm_metrics(frame, first):
    y = frame.failure.to_numpy(bool)
    fired = first >= 0
    p, n = int(y.sum()), int((~y).sum())
    tp, fp = int((y & fired).sum()), int((~y & fired).sum())
    recall, fpr = tp / p if p else np.nan, fp / n if n else np.nan
    timing = np.where(fired, (first + 1) / frame.length.to_numpy(), 1.)
    return {"episodes": len(y), "failures": p, "successes": n, "tp": tp, "fp": fp,
            "recall": recall, "fpr": fpr, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "balanced_accuracy": (recall + 1 - fpr) / 2, "t_det": timing[y].mean() if p else np.nan,
            "median_alarm_query": np.median(first[y & fired]) if tp else np.nan}


def goal_release(record, drop_only):
    allowed = {"object_released_or_dropped_before_goal", "object_released_outside_goal"}
    expressions = {goal["id"]: goal["expression"] for goal in record["goal_predicates"]}
    subjects = {expressions[goal["goal_id"]][1] for goal in record["goal_failure_labels"]
                if goal["reason"] in allowed or (not drop_only and goal["reason"] != "goal_satisfied_at_last_checkpoint")}
    events = [value["first_release_snapshot"] for subject, value in record["goal_subject_physics"].items()
              if subject in subjects and value.get("first_release_snapshot") is not None]
    return min(events) if events else np.nan


def physical_events(frame):
    data = {}
    with (PHYSICAL / "failures.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["run_id"] != RUNS[1]:
                continue
            task = record["suite"] + "/" + record["task_name"]
            data[(task, int(record["episode_index"]))] = record
    rows = []
    for row in frame.loc[(frame.run_id == RUNS[1]) & frame.failure].itertuples():
        record = data[(row.task, int(row.episode))]
        rows.append({"global_row": row.Index, "task": row.task, "episode": row.episode,
                     "reason": record["primary_failure_reason"], "failed_goal_release": goal_release(record, False),
                     "drop_goal_release": goal_release(record, True)})
    return pd.DataFrame(rows).set_index("global_row", drop=False)


def paired_task_bootstrap(task_ranks):
    data = task_ranks.loc[(task_ranks.reference_cap == 4096) & (task_ranks.view == "common_horizon") & (task_ranks.scope == "unseen")]
    table = data.groupby(["suite", "task", "method"]).auc.mean().unstack("method")
    if PRIMARY not in table:
        return pd.DataFrame(columns=["method", "baseline", "tasks", "macro_auc_difference", "lo", "hi", "scope"])
    results = []
    rng = np.random.default_rng(20260907)
    for baseline in ("behavior_contrast__cumsum", "stats_contrast__current", "stats_success__cumsum", "clock"):
        delta = (table[PRIMARY] - table[baseline]).dropna()
        if not len(delta):
            continue
        boot = []
        for _ in range(2000):
            values = []
            for _, block in delta.groupby(level="suite"):
                values.extend(rng.choice(block.to_numpy(), size=len(block), replace=True))
            boot.append(np.mean(values))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        results.append({"method": PRIMARY, "baseline": baseline, "tasks": len(delta),
                        "macro_auc_difference": delta.mean(), "lo": lo, "hi": hi,
                        "scope": "conditional on recorded tasks, fitted references and shared noise seeds"})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"sealed artifact changed: {name}")
    frame = attach_labels(pd.read_csv(output / "index.csv"))
    frame.to_csv(output / "outcome_alignment.csv", index=False)
    common = frame.loc[frame.run_id == RUNS[1]].groupby("task").length.min().to_dict()
    events = physical_events(frame)
    events.to_csv(output / "physical_events.csv", index=False)
    ranks, task_ranks, alarms, task_alarms, physical, decisions = [], [], [], [], [], []
    for info in manifest["folds"]:
        with np.load(output / "predictions" / f"{info['fold']}.npz", allow_pickle=False) as stored:
            data = {k: stored[k] for k in stored.files}
        part = frame.iloc[data["test_rows"]].copy().reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        common_end = part.task.map(common).to_numpy()
        full_valid = np.arange(52)[None] < part.length.to_numpy()[:, None]
        metadata = {k: info[k] for k in ("fold", "suite", "seed", "reference_cap")}
        for method_i, method in enumerate(data["methods"].astype(str)):
            raw = data["scores"][method_i]
            if np.isfinite(raw[~full_valid]).any():
                raise AssertionError("scores after termination")
            full_score = trajectory_peak(raw)
            common_score = trajectory_peak(np.where(np.arange(52)[None] < common_end[:, None], raw, np.nan))
            views = [("full", full_score, np.ones(len(part), bool)), ("common_horizon", common_score, np.ones(len(part), bool))]
            if info["reference_cap"] == 4096:
                views += [(f"q{q}", trajectory_peak(raw[:, :q + 1]), part.length.to_numpy() > q) for q in LANDMARKS]
            for view, values, present in views:
                for scope in ("seen", "unseen"):
                    selected = present & (part.scope.to_numpy() == scope)
                    if not selected.any():
                        continue
                    group = part.loc[selected].reset_index(drop=True)
                    ranks.append(dict(metadata, method=method, view=view, scope=scope, **ranking(group, values[selected])))
                    if view in ("full", "common_horizon"):
                        for task, ix in group.groupby("task").indices.items():
                            task_ranks.append(dict(metadata, method=method, view=view, scope=scope, task=task,
                                                   **ranking(group.iloc[ix].reset_index(drop=True), values[selected][ix])))
            for calibration_i, calibration_kind in enumerate(("episode", "task_init")):
                for alpha_i, alpha in enumerate(data["alphas"]):
                    first = data["first"][calibration_i, alpha_i, method_i]
                    if ((first >= part.length.to_numpy()) | (first < -1)).any():
                        raise AssertionError("invalid alarm time")
                    for scope in ("seen", "unseen"):
                        selected = part.scope.to_numpy() == scope
                        alarms.append(dict(metadata, method=method, calibration=calibration_kind, alpha=alpha,
                                           scope=scope, **alarm_metrics(part.loc[selected], first[selected])))
                        if alpha == 0.05:
                            for task, ix in part.loc[selected].groupby("task").indices.items():
                                current = part.loc[selected].iloc[ix]
                                task_alarms.append(dict(metadata, method=method, calibration=calibration_kind, alpha=alpha,
                                                       scope=scope, task=task, **alarm_metrics(current, first[selected][ix])))
                    if info["reference_cap"] == 4096 and alpha == 0.05 and method in TIMING_METHODS:
                        for i, row in enumerate(part.itertuples()):
                            if method in (PRIMARY, "success_only__cumsum", "v7_guard_constant", "v7_success_fusion_constant"):
                                decisions.append(dict(metadata, method=method, calibration=calibration_kind,
                                                      global_row=row.global_row, task=row.task, scope=row.scope,
                                                      episode=row.episode, failure=row.failure, length=row.length,
                                                      first_alarm=int(first[i])))
                            if row.global_row in events.index:
                                event = events.loc[row.global_row]
                                physical.append(dict(metadata, method=method, calibration=calibration_kind,
                                    global_row=row.global_row, scope=row.scope, first_alarm=int(first[i]),
                                    failed_goal_release=event.failed_goal_release, drop_goal_release=event.drop_goal_release,
                                    reason=event.reason))
        print(f"EVALUATED {info['fold']}", flush=True)
    frames = {"ranking_metrics": ranks, "task_ranking_metrics": task_ranks, "alarm_metrics": alarms,
              "task_alarm_metrics": task_alarms, "physical_timing": physical, "episode_decisions": decisions}
    for name, values in frames.items():
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)
    paired_task_bootstrap(pd.DataFrame(task_ranks)).to_csv(output / "paired_comparisons.csv", index=False)
    write_json(output / "evaluation_summary.json", {"test_labels_sha256": digest(PHYSICAL / "episodes.csv"),
        "physical_labels_sha256": digest(PHYSICAL / "failures.jsonl"), "evaluator_sha256": digest(Path(__file__)),
        "sealed_manifest_sha256": digest(output / "sealed_manifest.json"), "rows": {k: len(v) for k, v in frames.items()},
        "common_horizons": common, "B_episodes": int((frame.run_id == RUNS[1]).sum()),
        "B_failures": int(frame.loc[frame.run_id == RUNS[1], "failure"].sum()),
        "nominal_fpr_guarantee_on_unseen_tasks": False})


if __name__ == "__main__":
    main()
