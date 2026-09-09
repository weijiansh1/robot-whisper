"""Distributions, matched comparisons, and timing for frozen fusion candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion import HERE, ROOT, METHODS, PRIMARY
from core import digest, write_json, trajectory_peak
from evaluate import attach_labels, alarm_metrics, ranking
from run_fusion import load_npz

BIN_EDGES = [-.5, 7.5, 10.5, 14.5, 19.5, 29.5, 39.5, 51.5]
BIN_NAMES = ["q0-7", "q8-10", "q11-14", "q15-19", "q20-29", "q30-39", "q40-51"]
LEADS = (0, 2, 4, 8, 12, 16, 20)


def timing_metrics(frame, first):
    result = alarm_metrics(frame, first)
    y = frame.failure.to_numpy(bool)
    length = frame.length.to_numpy()
    alarm = first >= 0
    for lead in LEADS:
        timely = alarm & (length-first >= lead)
        result[f"tp_lead{lead}"] = int((y & timely).sum())
        result[f"recall_lead{lead}"] = float((y & timely).sum() / y.sum()) if y.any() else np.nan
        result[f"legacy_filtered_fp_lead{lead}"] = int((~y & timely).sum())
    for query in (7, 10, 14, 21, 29, 39, 51):
        by = alarm & (first <= query)
        result[f"recall_by_q{query}"] = float((y & by).sum() / y.sum()) if y.any() else np.nan
        result[f"fpr_by_q{query}"] = float((~y & by).sum() / (~y).sum()) if (~y).any() else np.nan
    return result


def distributions(decisions, output):
    bins, curves, relative = [], [], []
    for (scope, method), scoped in decisions.groupby(["scope", "method"]):
        for suite, part in [("all", scoped), *list(scoped.groupby("suite"))]:
            for failure, group in part.groupby("failure"):
                first, length = group.first_alarm_query.to_numpy(), group.length.to_numpy()
                meta = dict(scope=scope, suite=suite, method=method, failure=bool(failure),
                            episodes=len(group), unique_episodes=group.global_row.nunique())
                labels = pd.cut(first, BIN_EDGES, labels=BIN_NAMES).astype(str)
                labels[first < 0] = "no_alarm"
                for label in [*BIN_NAMES, "no_alarm"]:
                    count = int((labels == label).sum())
                    bins.append(dict(meta, bin=label, count=count, fraction=count/len(group)))
                progress = (first + 1) / length
                bands = np.minimum((progress * 4).astype(int), 3)
                for band, name in enumerate(("0-25%", "25-50%", "50-75%", "75-100%")):
                    count = int(((first >= 0) & (bands == band)).sum())
                    relative.append(dict(meta, bin=name, count=count, fraction=count/len(group)))
                relative.append(dict(meta, bin="no_alarm", count=int((first < 0).sum()), fraction=float((first < 0).mean())))
                for query in range(52):
                    at_risk = (length > query) & ((first < 0) | (first >= query))
                    count = int((first == query).sum())
                    cumulative = int(((first >= 0) & (first <= query)).sum())
                    curves.append(dict(meta, query=query, executed_action_steps=10*query,
                        new_alarms=count, cumulative_alarms=cumulative, cumulative_fraction=cumulative/len(group),
                        at_risk=int(at_risk.sum()), new_alarm_rate=count/at_risk.sum() if at_risk.any() else np.nan,
                        ended_without_alarm=int(((length <= query) & (first < 0)).sum())))
    for name, records in (("alarm_bins", bins), ("query_distribution", curves), ("relative_alarm_bins", relative)):
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)


def paired_comparisons(decisions, output):
    data = decisions.loc[decisions.scope.eq("unseen")].copy()
    pivot = data.pivot(index=["fold", "global_row"], columns="method", values="first_alarm_query")
    meta = data.drop_duplicates(["fold", "global_row"]).set_index(["fold", "global_row"]).loc[pivot.index]
    pairs = []
    for baseline in ("knn10", "v7_guard", "v8_guard"):
        old = pivot[baseline].to_numpy()
        for method in METHODS:
            current = pivot[method].to_numpy()
            for failure in (False, True):
                mask = meta.failure.to_numpy() == failure
                both = mask & (old >= 0) & (current >= 0)
                pairs.append(dict(baseline=baseline, method=method, failure=failure, episodes=int(mask.sum()),
                    both=int(both.sum()), added=int((mask & (old < 0) & (current >= 0)).sum()),
                    removed=int((mask & (old >= 0) & (current < 0)).sum()),
                    earlier=int((both & (current < old)).sum()), same=int((both & (current == old)).sum()),
                    later=int((both & (current > old)).sum())))
    pd.DataFrame(pairs).to_csv(output / "paired_alarm_changes.csv", index=False)
    data["tp"] = data.failure & data.first_alarm_query.ge(0)
    data["fp"] = ~data.failure & data.first_alarm_query.ge(0)
    data["tp_lead4"] = data.tp & (data.length - data.first_alarm_query >= 4)
    data["success"] = ~data.failure
    counts = data.groupby(["suite", "task", "method"])[["tp", "fp", "failure", "success", "tp_lead4"]].sum()
    tasks = counts.index.droplevel("method").unique()
    values = np.stack([counts.xs(method, level="method").loc[tasks].to_numpy(float) for method in METHODS])
    rng = np.random.default_rng(20260910)
    draws = np.concatenate([rng.choice(np.flatnonzero(tasks.get_level_values("suite") == suite),
        size=(2000, int((tasks.get_level_values("suite") == suite).sum())), replace=True)
        for suite in tasks.get_level_values("suite").unique()], axis=1)
    totals = values[:, draws].sum(axis=2)
    observed = values.sum(axis=1)
    boot_rows = []
    for name, numerator, denominator in (("recall", 0, 2), ("fpr", 1, 3), ("recall_lead4", 4, 2)):
        ratio = totals[..., numerator] / np.maximum(totals[..., denominator], 1)
        point = observed[:, numerator] / observed[:, denominator]
        for mi, method in enumerate(METHODS):
            delta = ratio[mi] - ratio[0]
            lo, hi = np.quantile(delta, [.025, .975])
            boot_rows.append(dict(method=method, baseline="knn10", metric=name, difference=point[mi]-point[0],
                                  low=lo, high=hi, tasks=len(tasks), bootstrap_samples=2000,
                                  unit="task with all fold appearances, stratified by suite"))
    pd.DataFrame(boot_rows).to_csv(output / "paired_task_bootstrap.csv", index=False)


def historical_anchors(frame, decisions, output):
    with np.load(ROOT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz") as meta:
        tasks = meta["task_names"].astype(str)[meta["task_index"]]
        lookup = {(r.task, int(r.episode)): i for i, r in enumerate(frame.itertuples()) if r.run_id == "right-50x8b-20260903"}
        rows = np.asarray([lookup[(task, int(ep))] for task, ep in zip(tasks, meta["episode"])])
        np.testing.assert_array_equal(frame.iloc[rows].length, meta["length"])
    part = frame.iloc[rows].copy()
    part["global_row"] = rows
    part = part.set_index("global_row", drop=False)
    firsts = {}
    v8_root = ROOT / "moe-v8-0906/results"
    paths = {"v7_frozen": ("v8_full_corpus_alarms.npz", "v7"), "v8_frozen": ("v8_full_corpus_alarms.npz", "v8"),
             "v82_frozen": ("v82_alarms.npz", "v8.2"), "v83_frozen": ("v83_alarms.npz", "v8.3")}
    anchor_records, summaries, audit = [], [], []
    for method, (filename, key) in paths.items():
        data = load_npz(v8_root / filename)
        first = data[f"external_8b|{key}"].astype(int)
        valid = (first >= 0) & (first < part.length.to_numpy())
        original = first.copy()
        first = np.where(valid, first, -1)
        firsts[method] = pd.Series(first, index=rows)
        audit.append(dict(method=method, removed_after_end=int(((original >= 0) & ~valid).sum()),
                          file=filename, sha256=digest(v8_root / filename)))
        summaries.append(dict(method=method, sample="historical_external_15600", **timing_metrics(part, first)))
        rec = part[["global_row", "suite", "task", "episode", "length", "failure"]].copy()
        rec["method"], rec["first_alarm_query"], rec["original_first_alarm_query"] = method, first, original
        anchor_records.append(rec)
    pd.concat(anchor_records).to_csv(output / "historical_episode_alarms.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "historical_metrics.csv", index=False)
    comparisons, combinations = [], []
    # These are diagnostic boolean combinations of already frozen alarms.
    # They retain each branch's old threshold; they are not a matched-budget fit.
    for (fold, scope), group in decisions.loc[decisions.method.eq("knn10")].groupby(["fold", "scope"]):
        group = group.loc[group.global_row.isin(rows)]
        k = group.first_alarm_query.to_numpy()
        comparisons.append(dict(fold=fold, scope=scope, method="knn10", **timing_metrics(group, k)))
        for method, source in firsts.items():
            old = source.loc[group.global_row].to_numpy()
            comparisons.append(dict(fold=fold, scope=scope, method=method, **timing_metrics(group, old)))
            for rule in ("or", "and"):
                merged = np.where(k < 0, old, np.where(old < 0, k, np.minimum(k, old))) if rule == "or" else np.where((k >= 0) & (old >= 0), np.maximum(k, old), -1)
                combinations.append(dict(fold=fold, scope=scope, method=f"{method}_{rule}_knn10", **timing_metrics(group, merged)))
    pd.DataFrame(comparisons).to_csv(output / "historical_matched_folds.csv", index=False)
    pd.DataFrame(combinations).to_csv(output / "historical_boolean_combinations.csv", index=False)
    write_json(output / "historical_audit.json", dict(episodes=len(rows), sources=audit,
        scope="Historical global profiles saw additional A tasks. Boolean combinations keep branch thresholds; budgets differ."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    frame = attach_labels(pd.read_csv(output / "index.csv"))
    frame.to_csv(output / "outcome_alignment.csv", index=False)
    common = frame.loc[frame.run_id.eq("right-50x8b-20260903")].groupby("task").length.min()
    metrics, ranks, task_metrics, decision_frames = [], [], [], []
    for info in manifest["folds"]:
        fold = info["fold"]
        data = load_npz(output / "predictions" / f"{fold}.npz")
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        for mi, method in enumerate(METHODS):
            for ki, calibration in enumerate(("episode", "task_init")):
                for ai, alpha in enumerate(data["alphas"]):
                    first = data["first"][ki, ai, mi]
                    assert ((first >= -1) & (first < part.length)).all()
                    for scope in ("seen", "unseen"):
                        mask = part.scope.eq(scope).to_numpy()
                        metrics.append(dict(fold=fold, suite=info["suite"], method=method, calibration=calibration,
                            alpha=float(alpha), scope=scope, **timing_metrics(part.loc[mask], first[mask])))
                    if ki == 1 and np.isclose(alpha, .05):
                        rec = part.drop(columns=["source", "checkpoint", "recorded_success", "primary_failure_reason"]).copy()
                        rec["fold"], rec["method"], rec["first_alarm_query"] = fold, method, first
                        rec["executed_action_steps_before_alarm"] = np.where(first >= 0, first*10, -1)
                        rec["threshold"] = data["thresholds"][ki, ai, mi]
                        decision_frames.append(rec)
                        for (scope, task), indices in part.groupby(["scope", "task"]).indices.items():
                            task_metrics.append(dict(fold=fold, suite=info["suite"], task=task, scope=scope,
                                method=method, **timing_metrics(part.iloc[indices], first[indices])))
            score = data["scores"][mi]
            end = part.task.map(common).to_numpy()
            views = {"full": trajectory_peak(score), "common_horizon": trajectory_peak(np.where(np.arange(52)[None] < end[:, None], score, np.nan))}
            for scope in ("seen", "unseen"):
                mask = part.scope.eq(scope).to_numpy()
                for view, values in views.items():
                    ranks.append(dict(fold=fold, suite=info["suite"], scope=scope, method=method,
                                      view=view, **ranking(part.loc[mask], values[mask])))
        print(f"FUSION EVALUATED {fold}", flush=True)
    pd.DataFrame(metrics).to_csv(output / "alarm_metrics.csv", index=False)
    pd.DataFrame(task_metrics).to_csv(output / "task_alarm_metrics.csv", index=False)
    rank_frame = pd.DataFrame(ranks)
    rank_frame.to_csv(output / "ranking_metrics.csv", index=False)
    decisions = pd.concat(decision_frames, ignore_index=True)
    decisions.to_csv(output / "episode_decisions.csv", index=False)
    metrics = pd.DataFrame(metrics)
    main = metrics.loc[metrics.calibration.eq("task_init") & np.isclose(metrics.alpha, .05)]
    fields = ["recall", "fpr", "precision", "recall_lead4", "recall_by_q10", "recall_by_q14", "recall_by_q21", "fpr_by_q14"]
    summary = main.groupby(["scope", "method"])[fields].mean().reset_index()
    summary.to_csv(output / "fold_macro_summary.csv", index=False)
    main.groupby(["scope", "suite", "method"])[fields].mean().to_csv(output / "suite_summary.csv")
    metrics.groupby(["scope", "calibration", "method", "alpha"])[fields].mean().to_csv(output / "operating_points.csv")
    rank_frame.groupby(["scope", "view", "method"])[["task_macro_auc", "within_init_macro_auc", "scorable_fraction"]].mean().to_csv(output / "common_horizon_summary.csv")
    distributions(decisions, output)
    paired_comparisons(decisions, output)
    events = pd.read_csv(HERE.parent / "results/round3_safe/physical_events.csv")
    physical = decisions.merge(events[["global_row", "drop_goal_release", "failed_goal_release", "reason"]], on="global_row", how="inner", validate="many_to_one")
    physical["lag_from_drop"] = np.where(physical.first_alarm_query >= 0, physical.first_alarm_query-physical.drop_goal_release, np.nan)
    physical.to_csv(output / "physical_timing.csv", index=False)
    event_metrics = []
    for (scope, method), group in physical.loc[physical.drop_goal_release.notna()].groupby(["scope", "method"]):
        first, event = group.first_alarm_query, group.drop_goal_release
        event_metrics.append(dict(scope=scope, method=method, events=len(group), unique_events=group.global_row.nunique(),
            before=int(((first >= 0) & (first < event)).sum()), at_event=int((first == event).sum()),
            after=int((first > event).sum()), missed=int((first < 0).sum())))
    pd.DataFrame(event_metrics).to_csv(output / "physical_summary.csv", index=False)
    historical_anchors(frame, decisions, output)
    write_json(output / "evaluation_summary.json", dict(primary=PRIMARY, methods=METHODS, folds=len(manifest["folds"]),
        decision_appearances=len(decisions), new_model_training=False, historically_explored_data=True,
        interpretation="All FP retained at every TP lead cutoff. Distributions pool appearances; macro metrics average folds.",
        evaluator_sha256=digest(Path(__file__)), sealed_manifest_sha256=digest(output / "sealed_manifest.json"),
        artifacts={path.name: digest(path) for path in sorted(output.glob("*.csv"))}))
    print(summary.loc[summary.scope.eq("unseen")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
