"""Outcome comparisons, matched distance controls and external alarm windows."""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from v82_knn_research import archive, first_crossing, alarm_metrics, auc, save_json
from score_v82_knn_reference import load_profile
from alarm_trajectory_study import ROOT, PREVIOUS, METHODS, physical_window, load_bank


def task_bootstrap(values, seed=2026091502):
    x = np.asarray(values, float)
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), (2000, len(x)))].mean(-1)
    return tuple(np.quantile(sample, [.025, .975]))


def main():
    _, params = load_bank()
    _, instant_params = load_profile()
    metrics, ranks, matched, windows, events, ambiguity, timings = [], [], [], [], [], [], []
    cohorts = ['B', 'current'] + (['fresh'] if (ROOT / 'fresh-scores.npz').exists() else [])
    for cohort in cohorts:
        frame = pd.read_csv(ROOT / (cohort+'-decisions.csv'))
        data = archive(ROOT / (cohort+'-scores.npz'))
        if cohort == 'fresh':
            instant = data['instantaneous']
        else:
            old = archive(PREVIOUS / ('available-B-geometry.npz' if cohort == 'B' else 'current-geometry.npz'))
            instant = old['score']
        delayed = instant.copy()
        delayed[:, :12] = np.nan
        methods = dict(v82=frame.v82_frozen.to_numpy(), instantaneous=frame.knn_available_first.to_numpy(),
                       instantaneous_from_q12=first_crossing(delayed, instant_params['threshold']))
        methods.update({name: frame[name+'_first'].to_numpy() for name in METHODS})
        subsets = [(cohort, frame)]
        if cohort != 'B':
            subsets += [(cohort+'_'+name, part) for name, part in frame.groupby('benchmark')]
        for scope, part in subsets:
            for name, first in methods.items():
                selected = first[part.index]
                for failure in (False, True):
                    included = part.failure.to_numpy() == failure
                    hit = included & (selected >= 0) & (selected < part.length.to_numpy())
                    fraction = selected[hit]*10/part.horizon_actions.to_numpy()[hit]
                    remaining = part.horizon_actions.to_numpy()[hit]-selected[hit]*10
                    timings.append(dict(cohort=scope, method=name, failure=failure, parents=int(included.sum()),
                        flagged=int(hit.sum()), median_query=float(np.median(selected[hit])) if hit.any() else np.nan,
                        median_budget_fraction=float(np.median(fraction)) if hit.any() else np.nan,
                        q25_budget_fraction=float(np.quantile(fraction, .25)) if hit.any() else np.nan,
                        q75_budget_fraction=float(np.quantile(fraction, .75)) if hit.any() else np.nan,
                        median_remaining_budget_actions=float(np.median(remaining)) if hit.any() else np.nan))
            for cutoff_name, cutoff in [('all', 10000), ('q13', 13), ('q20', 20),
                                        ('budget50', part.horizon_actions.to_numpy()*.5/10),
                                        ('budget75', part.horizon_actions.to_numpy()*.75/10)]:
                for name, first in methods.items():
                    metrics.append(dict(cohort=scope, method=name, cutoff=cutoff_name,
                                        **alarm_metrics(part, first[part.index], cutoff)))
        if cohort == 'B':
            measurements = dict(instantaneous=instant, **{key: data[key] for key in
                 ('displacement', 'endpoint', 'rematched_endpoint', 'prefix_distance', 'reference_disagreement', 'direction_cosine')})
            for task, part in frame.groupby('task'):
                for q in (13, 20, 30):
                    eligible = part[part.length > q]
                    for name, values in measurements.items():
                        score = auc(eligible.failure.to_numpy(), values[eligible.index, q])
                        if np.isfinite(score):
                            ranks.append(dict(task=task, query=q, method=name, auc=score, parents=len(eligible),
                                              failures=int(eligible.failure.sum())))
                    good, bad = eligible[~eligible.failure], eligible[eligible.failure]
                    used = set()
                    for failure in bad.itertuples():
                        candidates = [s for s in good.itertuples() if s.Index not in used
                                      and abs(instant[failure.Index, q]-instant[s.Index, q]) <= .15
                                      and abs(old['unique_neighbor_parents'][failure.Index, q]-old['unique_neighbor_parents'][s.Index, q]) <= 2]
                        if not candidates:
                            continue
                        success = min(candidates, key=lambda s: (abs(instant[failure.Index, q]-instant[s.Index, q]), s.Index))
                        used.add(success.Index)
                        for name in ('displacement', 'endpoint', 'reference_disagreement'):
                            delta = float(data[name][failure.Index, q]-data[name][success.Index, q])
                            matched.append(dict(task=task, query=q, method=name, failure_row=failure.Index,
                                                 success_row=success.Index, difference=delta, failure_higher=float(delta > 0)+.5*(delta == 0),
                                                 instantaneous_gap=float(instant[failure.Index, q]-instant[success.Index, q])))
            for method in METHODS:
                for row in frame.itertuples():
                    q = int(getattr(row, method+'_first'))
                    if q >= 0:
                        ambiguity.append(dict(cohort='B', method=method, failure=bool(row.failure), query=q,
                            score=float(data[method][row.Index, q]), prefix_distance=float(data['prefix_distance'][row.Index, q]),
                            reference_disagreement=float(data['reference_disagreement'][row.Index, q]),
                            unique_parents=int(data['unique_parents'][row.Index, q])))
        else:
            physical_dir = ROOT / ('physical' if cohort == 'current' else 'fresh-physical')
            for row in frame.itertuples():
                path = physical_dir / (row.name+'.npz')
                if not path.exists():
                    continue
                physics = archive(path)
                parent_windows = {}
                for horizon in (3, 5):
                    for q in range(row.length):
                        record = dict(cohort=cohort, name=row.name, task_id=row.task_id, benchmark=row.benchmark,
                                      failure=bool(row.failure), horizon=horizon, **physical_window(physics, q, horizon))
                        parent_windows[horizon, q] = record
                        windows.append(record)
                for method, first in methods.items():
                    q = int(first[row.Index])
                    if q < 0:
                        continue
                    for horizon in (3, 5):
                        events.append(dict(**parent_windows[horizon, q], method=method,
                                           budget_fraction=q*10/row.horizon_actions))
    pd.DataFrame(metrics).to_csv(ROOT / 'alarm-metrics.csv', index=False)
    pd.DataFrame(timings).to_csv(ROOT / 'alarm-timing.csv', index=False)
    uncertainty = []
    for record in metrics:
        if record['cohort'].startswith('fresh_') and record['cutoff'] == 'all':
            for outcome, numerator, denominator in (('recall', 'tp', 'failures'), ('fpr', 'fp', 'successes')):
                count, total = record[numerator], record[denominator]
                interval = binomtest(count, total).proportion_ci(method='wilson') if total else None
                uncertainty.append(dict(cohort=record['cohort'], method=record['method'], metric=outcome,
                    count=count, total=total, low=interval.low if interval else np.nan,
                    high=interval.high if interval else np.nan, interval='descriptive_Wilson_95'))
    pd.DataFrame(uncertainty, columns=['cohort', 'method', 'metric', 'count', 'total', 'low', 'high', 'interval']).to_csv(
        ROOT / 'fresh-uncertainty.csv', index=False)
    pd.DataFrame(ranks).to_csv(ROOT / 'same-query-ranking.csv', index=False)
    pd.DataFrame(matched).to_csv(ROOT / 'distance-matched-pairs.csv', index=False)
    pd.DataFrame(ambiguity).to_csv(ROOT / 'reference-ambiguity-at-alarm.csv', index=False)
    pd.DataFrame(windows).to_csv(ROOT / 'physical-query-windows.csv', index=False)
    event_table = pd.DataFrame(events)
    event_table.to_csv(ROOT / 'external-alarm-windows.csv', index=False)
    external_summary = event_table.groupby(['cohort', 'benchmark', 'method', 'horizon', 'failure', 'category']).size().reset_index(name='parents')
    external_summary.to_csv(ROOT / 'external-alarm-categories.csv', index=False)
    matched_table = pd.DataFrame(matched)
    match_summary = []
    for (query, method), part in matched_table.groupby(['query', 'method']):
        means = part.groupby('task').failure_higher.mean()
        low, high = task_bootstrap(means)
        match_summary.append(dict(query=query, method=method, pairs=len(part), tasks=len(means),
                                  task_mean_failure_higher=means.mean(), bootstrap_low=low, bootstrap_high=high))
    pd.DataFrame(match_summary).to_csv(ROOT / 'distance-matched-summary.csv', index=False)
    ranking_summary = pd.DataFrame(ranks).groupby(['query', 'method']).agg(auc=('auc', 'mean'), tasks=('task', 'nunique')).reset_index()
    ranking_summary.to_csv(ROOT / 'same-query-ranking-summary.csv', index=False)
    save_json(ROOT / 'trajectory-results.json', dict(metrics=[r for r in metrics if r['cutoff'] == 'all'],
              rank_summary=ranking_summary.to_dict('records'), distance_matched=match_summary,
              external_categories=external_summary.to_dict('records'), no_local_trap_ground_truth=True,
              candidate_thresholds=params['thresholds'], fresh_included='fresh' in cohorts))
    print(pd.DataFrame(metrics).query('cutoff == "all"').to_string(index=False), flush=True)
    print(pd.DataFrame(match_summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
