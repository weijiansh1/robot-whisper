"""Evaluate sealed guards using complete-trajectory peaks and conditional AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import auc
from run_analysis import BASE, RUN_B, bootstrap_indices, digest, write_json


METHODS = ("v7", "v8_fixed", "v82")


def pairwise_auc(labels, scores):
    positive = np.asarray(scores)[labels]
    negative = np.sort(np.asarray(scores)[~labels])
    if not len(positive) or not len(negative):
        return np.nan
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(((lower + upper) / 2).sum() / (len(positive) * len(negative)))


def macro_row(view, scope, method, part):
    part = part.loc[part.auc.notna()].sort_values(["suite", "task"])
    values = part.auc.to_numpy()
    draws = bootstrap_indices(part.task, part.suite)
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(view=view, scope=scope, method=method, estimate=float(values.mean()),
                lo=float(lo), hi=float(hi), tasks=len(part))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "v82_validation_20260908")
    parser.add_argument("--output", type=Path, default=BASE / "v82_full_auroc_20260908")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_contract.json", dict(
        methods=METHODS, primary_score="maximum finite score over all valid execution queries",
        summaries=["pooled trajectory AUC", "mean within-task trajectory AUC",
                   "mean within-task/time AUC over all comparable queries q7 onward"],
        confidence_interval="2000 suite-stratified task bootstrap draws for task means; fixed profiles",
        threshold_or_score_refitting=False, retrospective_user_requested_extension=True,
        episode_duration_used_as_score=False,
        test_cohort=RUN_B, source_sha256=digest(__file__),
    ))
    previous = json.loads((source / "analysis_verification.json").read_text())
    names = ("index.csv", "crossfit_predictions.npz", "conditional_auc_strata.csv")
    hashes = {name: digest(source / name) for name in names}
    for name, value in hashes.items():
        assert value == previous["artifacts"][name], name
    frame = pd.read_csv(source / "index.csv")
    with np.load(source / "crossfit_predictions.npz", allow_pickle=False) as saved:
        b = frame.iloc[saved["global_rows"]].reset_index(drop=True)
        positions = [saved["methods"].tolist().index(method) for method in METHODS]
        scores, valid = saved["scores"][positions], saved["valid"]
    assert len(b) == 16000 and b.run_id.eq(RUN_B).all()
    assert b.failure.sum() == 564 and not b.duplicated(["source", "episode"]).any()
    np.testing.assert_array_equal(valid, np.arange(52)[None] < b.length.to_numpy()[:, None])
    peaks = np.where(valid[None] & np.isfinite(scores), scores, -np.inf).max(axis=2)
    assert np.isfinite(peaks).all()
    for i, length in enumerate(b.length):
        for mi in range(len(METHODS)):
            values = scores[mi, i, :length]
            assert peaks[mi, i] == max(values[np.isfinite(values)])
    exported = b[["global_row", "source", "episode", "task", "suite", "failure", "length"]].copy()
    for mi, method in enumerate(METHODS):
        exported[method + "_peak"] = peaks[mi]
    exported.to_csv(output / "trajectory_peaks.csv", index=False)
    summary, tasks, checked = [], [], 0
    scopes = [("all", b)] + list(b.groupby("suite", sort=True))
    for scope, current in scopes:
        labels = current.failure.to_numpy(bool)
        for mi, method in enumerate(METHODS):
            values = peaks[mi, current.index]
            estimate = auc(labels, values)
            np.testing.assert_allclose(estimate, pairwise_auc(labels, values), atol=1e-12, rtol=0)
            checked += 1
            summary.append(dict(view="full_peak_pooled", scope=scope, method=method,
                                estimate=estimate, lo=np.nan, hi=np.nan, tasks=current.task.nunique(),
                                failures=int(labels.sum()), successes=int((~labels).sum())))
    for task, current in b.groupby("task", sort=True):
        labels = current.failure.to_numpy(bool)
        for mi, method in enumerate(METHODS):
            values = peaks[mi, current.index]
            estimate = auc(labels, values)
            np.testing.assert_allclose(estimate, pairwise_auc(labels, values), atol=1e-12, rtol=0, equal_nan=True)
            checked += 1
            tasks.append(dict(task=task, suite=current.suite.iloc[0], method=method, auc=estimate,
                              failures=int(labels.sum()), successes=int((~labels).sum())))
    task_scores = pd.DataFrame(tasks)
    task_scores.to_csv(output / "task_peak_auc.csv", index=False)
    old = pd.read_csv(source / "conditional_auc_strata.csv")
    conditional = old.loc[old.level.eq("task") & old.method.isin(METHODS)]
    assert conditional["query"].between(7, 51).all()
    by_task = {task: current for task, current in b.groupby("task", sort=True)}
    for row in conditional.itertuples():
        current = by_task[row.task]
        current = current.loc[current.length > row.query]
        values = scores[METHODS.index(row.method), current.index, row.query]
        assert np.isfinite(values).all()
        labels = current.failure.to_numpy(bool)
        np.testing.assert_allclose(row.auc, pairwise_auc(labels, values), atol=1e-12, rtol=0)
        assert row.failures == labels.sum() and row.successes == (~labels).sum()
        checked += 1
    task_conditional = conditional.groupby(["suite", "task", "method"], as_index=False).agg(
        auc=("auc", "mean"), queries=("query", "nunique"), first_query=("query", "min"),
        last_query=("query", "max"), failure_query_observations=("failures", "sum"),
        success_query_observations=("successes", "sum"))
    task_conditional.to_csv(output / "task_conditional_auc.csv", index=False)
    for view, table in (("full_peak_task_macro", task_scores),
                        ("full_conditional_task_macro", task_conditional)):
        for method, part in table.groupby("method", sort=True):
            for scope, current in [("all", part)] + list(part.groupby("suite", sort=True)):
                summary.append(macro_row(view, scope, method, current))
    result = pd.DataFrame(summary)
    result.to_csv(output / "auc_summary.csv", index=False)
    write_json(output / "verification.json", dict(
        all_checks_passed=True, independent_peak_checks=int(peaks.size),
        independent_auc_checks=checked, test_episodes=len(b), input_sha256=hashes,
        source_sha256=digest(__file__), episode_duration_used_as_score=False,
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    ))
    print(result.loc[result.scope.eq("all")].to_string(index=False))
    print("Independent peak checks:", peaks.size, "; independent AUC checks:", checked)


if __name__ == "__main__":
    main()
