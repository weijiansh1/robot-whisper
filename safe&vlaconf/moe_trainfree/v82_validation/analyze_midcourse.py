"""Summarize sealed guard scores at 50-60% of each task's action budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_analysis import BASE, RUN_B, bootstrap_indices, digest, write_json


CAPS = dict(libero_goal=300, libero_long=520, libero_object=280, libero_spatial=220)
METHODS = ("v7", "v8_fixed", "v82", "eef_motion_low", "clock")
WINDOWS = ("early_q7_q13", "through_50pct", "through_60pct", "band_50_60pct")


def bounds(suite, window):
    cap = CAPS[suite]
    if window == "early_q7_q13":
        return 7, 13
    end = cap // 20 if window == "through_50pct" else 3 * cap // 50
    start = (cap + 19) // 20 if window == "band_50_60pct" else 7
    return start, end


def interval(values, tasks, suites):
    values = np.asarray(values, float)
    draws = bootstrap_indices(tasks, suites)
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(estimate=float(values.mean()), lo=float(lo), hi=float(hi), tasks=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "v82_validation_20260908")
    parser.add_argument("--output", type=Path, default=BASE / "v82_midcourse_20260908")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_contract.json", dict(
        task_action_budgets=CAPS, methods=METHODS, windows=WINDOWS,
        progress_definition="10 * query / configured_task_action_budget",
        conditional_auc="same task and query, active episodes; average query then task",
        confidence_interval="2000 suite-stratified task bootstrap draws; fixed profiles",
        frozen_alarm_denominator="all B successes/failures, including ended episodes",
        checkpoint_recalibration=False, retrospective_user_requested_extension=True,
        per_episode_final_length_normalization=False, source_sha256=digest(__file__),
    ))
    import json

    audit = json.loads((source / "analysis_verification.json").read_text())
    input_names = ("index.csv", "conditional_auc_strata.csv", "early_conditional_summary.csv",
                   "crossfit_predictions.npz", "frozen_first_alarms.csv")
    hashes = {name: digest(source / name) for name in input_names}
    for name, value in hashes.items():
        assert value == audit["artifacts"][name], name
    frame = pd.read_csv(source / "index.csv")
    with np.load(source / "crossfit_predictions.npz", allow_pickle=False) as saved:
        b = frame.iloc[saved["global_rows"]].reset_index(drop=True)
        positions = [saved["methods"].tolist().index(name) for name in METHODS]
        scores = saved["scores"][positions]
        valid = saved["valid"]
    assert len(b) == 16000 and b.run_id.eq(RUN_B).all()
    assert not b.duplicated(["source", "episode"]).any()
    np.testing.assert_array_equal(valid, np.arange(52)[None] < b.length.to_numpy()[:, None])
    np.testing.assert_array_equal(b.loc[b.failure, "action_steps"], b.loc[b.failure, "suite"].map(CAPS))
    old = pd.read_csv(source / "conditional_auc_strata.csv")
    old = old.loc[old.level.eq("task") & old.method.isin(METHODS)]
    lookup = old.set_index(["task", "query", "method"])
    assert not lookup.index.duplicated().any()
    checked, records, window_rows = 0, [], []
    for task, part in b.groupby("task", sort=True):
        ix = part.index.to_numpy()
        suite = part.suite.iloc[0]
        failure = part.failure.to_numpy(bool)
        length = part.length.to_numpy(int)
        for window in WINDOWS:
            start, end = bounds(suite, window)
            window_rows.append(dict(task=task, suite=suite, window=window,
                                    budget_actions=CAPS[suite], first_query=start, last_query=end))
        last = max(bounds(suite, window)[1] for window in WINDOWS)
        for q in range(7, last + 1):
            active = length > q
            for mi, method in enumerate(METHODS):
                values = scores[mi, ix, q]
                assert np.isfinite(values[active]).all()
                positive, negative = values[active & failure], values[active & ~failure]
                key = (task, q, method)
                if not len(positive) or not len(negative):
                    assert key not in lookup.index
                    continue
                # Direct pair comparisons independently check the saved rank-based AUC.
                pairs = positive[:, None] - negative[None, :]
                value = float(((pairs > 0) + .5 * (pairs == 0)).mean())
                previous = lookup.loc[key]
                np.testing.assert_allclose(value, previous.auc, rtol=0, atol=1e-12)
                assert len(positive) == previous.failures and len(negative) == previous.successes
                checked += 1
                for window in WINDOWS:
                    start, end = bounds(suite, window)
                    if start <= q <= end:
                        records.append(dict(window=window, task=task, suite=suite, query=q,
                                            method=method, auc=value, failures=len(positive),
                                            successes=len(negative)))
    data = pd.DataFrame(records)
    data.to_csv(output / "conditional_auc_strata.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output / "task_windows.csv", index=False)
    task_values = data.groupby(["window", "suite", "task", "method"], as_index=False).agg(
        auc=("auc", "mean"), queries=("query", "nunique"),
        failure_query_observations=("failures", "sum"), success_query_observations=("successes", "sum"))
    task_values.to_csv(output / "task_auc.csv", index=False)
    summaries, paired = [], []
    for (window, method), part in task_values.groupby(["window", "method"], sort=True):
        for scope, current in [("all", part)] + list(part.groupby("suite", sort=True)):
            summaries.append(dict(window=window, method=method, scope=scope,
                                  **interval(current.auc, current.task, current.suite)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "auc_summary.csv", index=False)
    old_early = pd.read_csv(source / "early_conditional_summary.csv")
    for method in METHODS:
        actual = summary.loc[summary.window.eq("early_q7_q13") & summary.scope.eq("all") & summary.method.eq(method)].iloc[0]
        expected = old_early.loc[old_early.level.eq("task") & old_early.scope.eq("all") & old_early.method.eq(method) & old_early.comparison.eq("auc")].iloc[0]
        np.testing.assert_allclose(actual[["estimate", "lo", "hi"]].to_numpy(float),
                                   expected[["estimate", "lo", "hi"]].to_numpy(float), atol=1e-12, rtol=0)
    for method, part in task_values.groupby("method", sort=True):
        pivot = part.pivot(index=["suite", "task"], columns="window", values="auc")
        for window in WINDOWS[1:]:
            common = pivot[["early_q7_q13", window]].dropna()
            suites, tasks = common.index.get_level_values("suite"), common.index.get_level_values("task")
            paired.append(dict(method=method, comparison=window + "_minus_early", window=window,
                               early_on_common_tasks=float(common.early_q7_q13.mean()),
                               later_on_common_tasks=float(common[window].mean()),
                               **interval(common[window] - common.early_q7_q13, tasks, suites)))
    pd.DataFrame(paired).to_csv(output / "paired_window_comparisons.csv", index=False)
    support, alarms = [], []
    frozen = pd.read_csv(source / "frozen_first_alarms.csv").set_index("global_row").loc[b.global_row].reset_index(drop=True)
    for field in ("source", "episode", "length", "failure"):
        np.testing.assert_array_equal(frozen[field], b[field])
    for checkpoint in ("early_q7_q13", "through_50pct", "through_60pct"):
        cutoff = b.suite.map(lambda s: bounds(s, checkpoint)[1]).to_numpy(int)
        for scope, current in [("all", b)] + list(b.groupby("suite", sort=True)):
            ix = current.index.to_numpy()
            failure = current.failure.to_numpy(bool)
            active = current.length.to_numpy() > cutoff[ix]
            eligible = 0
            for _, group in current.assign(active=active).groupby("task"):
                eligible += int((group.active & group.failure).any() and (group.active & ~group.failure).any())
            support.append(dict(checkpoint=checkpoint, scope=scope, episodes=len(current),
                                failures=int(failure.sum()), successes=int((~failure).sum()),
                                active_failures=int((active & failure).sum()),
                                active_successes=int((active & ~failure).sum()), eligible_tasks=eligible))
            for method in ("v7_frozen", "v8_frozen", "v82_frozen"):
                first = frozen[method].to_numpy()[ix]
                fired = (first >= 0) & (first <= cutoff[ix])
                tp, fp = int((fired & failure).sum()), int((fired & ~failure).sum())
                alarms.append(dict(checkpoint=checkpoint, scope=scope, method=method, tp=tp, fp=fp,
                                   failures=int(failure.sum()), successes=int((~failure).sum()),
                                   recall=tp / failure.sum(), fpr=fp / (~failure).sum()))
    pd.DataFrame(support).to_csv(output / "checkpoint_support.csv", index=False)
    pd.DataFrame(alarms).to_csv(output / "frozen_cumulative_alarms.csv", index=False)
    write_json(output / "verification.json", dict(
        all_checks_passed=True, independent_pairwise_auc_checks=checked,
        original_early_estimates_and_intervals_reproduced=True, test_episodes=len(b),
        input_sha256=hashes, source_sha256=digest(__file__),
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    ))
    print(summary.loc[summary.scope.eq("all")].to_string(index=False))
    print(pd.DataFrame(support).query("scope == 'all'").to_string(index=False))
    print(pd.DataFrame(alarms).query("scope == 'all'").to_string(index=False))
    print("Independent pairwise AUC checks:", checked)


if __name__ == "__main__":
    main()
