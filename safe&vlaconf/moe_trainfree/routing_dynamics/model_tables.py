"""Shared reports for fixed models with identical episode/query support."""

import numpy as np
import pandas as pd

from guard import ALPHAS, KINDS
from run_guard_experiment import ACTION_CAPS, S05
from run_analysis import bootstrap_indices, rate_interval
from monitor import auc, union


def combined_alarm(saved, kind, alpha, method):
    if method == "v82_frozen":
        return saved["frozen_first"]
    ki, ai = KINDS.index(kind), ALPHAS.index(alpha)
    methods = list(saved["methods"])
    if method in methods:
        first = saved["first"][ki, ai, methods.index(method)]
    else:
        first = saved["control_first"][ki, ai, list(saved["control_names"]).index(method)]
    return union(saved["frozen_first"], first)


def alarm_tables(b, saved, output, reference):
    methods, controls = list(saved["methods"]), list(saved["control_names"])
    y, caps = b.failure.to_numpy(bool), b.suite.map(ACTION_CAPS).to_numpy()
    scopes = [("all", np.arange(len(b)))]
    scopes += [(str(name), part.index.to_numpy()) for field in ("suite", "task") for name, part in b.groupby(field)]
    settings = [("original", np.nan, "v82_frozen")]
    settings += [(kind, alpha, method) for kind in KINDS for alpha in ALPHAS for method in (*controls, *methods)]
    metrics, paired, cases, s05 = [], [], [], []
    for kind, alpha, method in settings:
        alarm = combined_alarm(saved, kind, alpha, method).astype(np.int64)
        for scope, ix in scopes:
            labels, hit = y[ix], alarm[ix] >= 0
            tp, fp = int((labels & hit).sum()), int((~labels & hit).sum())
            failures, successes = int(labels.sum()), int((~labels).sum())
            row = dict(calibration=kind, alpha=alpha, method=method, scope=scope, tp=tp, fp=fp,
                       failures=failures, successes=successes,
                       recall=tp / failures if failures else np.nan, fpr=fp / successes if successes else np.nan,
                       median_failure_q=float(np.median(alarm[ix][hit & labels])) if tp else np.nan)
            for fraction in (.5, .6):
                early = hit & (10 * alarm[ix] <= fraction * caps[ix])
                row[f"tp_by_cap_{int(100*fraction)}"] = int((early & labels).sum())
                row[f"fp_by_cap_{int(100*fraction)}"] = int((early & ~labels).sum())
            if scope == "all":
                row.update(rate_interval(b, alarm >= 0, y))
            metrics.append(row)
        if alpha == .01 or kind == "original":
            for row in b.loc[b.task.eq(S05)].itertuples():
                s05.append(dict(calibration=kind, method=method, global_row=row.global_row,
                                episode=row.episode, failure=row.failure, first_alarm=int(alarm[row.Index])))
        if method not in methods:
            continue
        for baseline in ("v82_frozen", "old_ac", reference):
            if method == baseline:
                continue
            old = combined_alarm(saved, kind, alpha, baseline)
            for scope, ix in scopes:
                a, o, labels = alarm[ix] >= 0, old[ix] >= 0, y[ix]
                shared = a & o & labels
                delta = alarm[ix][shared] - old[ix][shared]
                paired.append(dict(calibration=kind, alpha=alpha, method=method, baseline=baseline, scope=scope,
                                   gained_tp=int((a & ~o & labels).sum()), lost_tp=int((~a & o & labels).sum()),
                                   added_fp=int((a & ~o & ~labels).sum()), removed_fp=int((~a & o & ~labels).sum()),
                                   earlier=int((delta < 0).sum()), same=int((delta == 0).sum()), later=int((delta > 0).sum())))
        if alpha == .01:
            frozen = saved["frozen_first"]
            for i in np.flatnonzero((alarm != frozen) | ((alarm >= 0) & ~y)):
                row = b.iloc[i]
                cases.append(dict(calibration=kind, method=method, global_row=row.global_row, suite=row.suite,
                                  task=row.task, episode=row.episode, failure=row.failure,
                                  frozen_first=int(frozen[i]), combined_first=int(alarm[i]),
                                  added_new_alarm=bool(frozen[i] < 0)))
    for name, values in (("alarm_metrics.csv", metrics), ("paired_changes.csv", paired),
                         ("s05_alarms.csv", s05), ("alarm_cases_alpha01.csv", cases)):
        pd.DataFrame(values).to_csv(output / name, index=False)


def auroc_tables(b, saved, output, reference):
    methods, records = list(saved["methods"]), []
    control_index = methods.index(reference)
    for ki, kind in enumerate(KINDS):
        for score_kind, values in (("raw_score", saved["scores"]), ("calibrated_rank", -np.log(saved["tails"][ki]))):
            for (suite, task), part in b.groupby(["suite", "task"]):
                ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
                for q in range(8, values.shape[-1]):
                    for mi, method in enumerate(methods):
                        available = (np.isfinite(values[mi, ix, q]) & np.isfinite(values[control_index, ix, q])
                                     & np.isfinite(saved["old_ac_rank"][ki, ix, q])
                                     & np.isfinite(saved["raw_v82_scores"][ix, q]))
                        labels, rows = y[available], ix[available]
                        if not labels.any() or labels.all():
                            continue
                        value = auc(labels, values[mi, rows, q])
                        control = auc(labels, values[control_index, rows, q])
                        old = auc(labels, saved["old_ac_rank"][ki, rows, q])
                        records.append(dict(calibration=kind, score_kind=score_kind, method=method,
                                            suite=suite, task=task, query=q, auc=value,
                                            control_auc=control, old_ac_auc=old, delta_control=value-control,
                                            delta_old_ac=value-old, failures=int(labels.sum()), successes=int((~labels).sum())))
    table = pd.DataFrame(records)
    table.to_csv(output / "query_auroc.csv", index=False)
    summaries, task_rows = [], []
    fields = ("auc", "control_auc", "old_ac_auc", "delta_control", "delta_old_ac")
    for window, rows in (("q8_13", table.loc[table["query"].between(8, 13)]), ("all_comparable_queries", table)):
        tasks = rows.groupby(["calibration", "score_kind", "method", "suite", "task"], as_index=False).agg(
            **{name: (name, "mean") for name in fields}, queries=("query", "nunique"),
            first_query=("query", "min"), last_query=("query", "max"))
        tasks["window"] = window
        task_rows.append(tasks)
        for (kind, score_kind, method), part in tasks.groupby(["calibration", "score_kind", "method"]):
            for scope, selected in (("all", part), ("excluding_S05", part.loc[part.task.ne(S05)])):
                draws = bootstrap_indices(selected.task, selected.suite)
                for field in fields:
                    values = selected[field].to_numpy()
                    lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
                    summaries.append(dict(calibration=kind, score_kind=score_kind, method=method, window=window,
                                          scope=scope, metric=field, estimate=float(values.mean()), lo=float(lo), hi=float(hi),
                                          tasks=len(selected), first_query=int(selected.first_query.min())))
    pd.concat(task_rows, ignore_index=True).to_csv(output / "task_auroc.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "auroc_summary.csv", index=False)
