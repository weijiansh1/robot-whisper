"""Descriptive task/suite context of already-frozen continuation neighbors."""

import json

import numpy as np
import pandas as pd

from alarm_trajectory_study import ROOT, PREVIOUS, METHODS, load_bank
from v82_knn_research import archive, save_json, alarm_metrics


def main():
    bank, _ = load_bank()
    available = pd.read_csv(PREVIOUS/'available-index.csv')
    bank_tasks = available.iloc[bank['available_rows']].task.to_numpy()
    bank_suites = available.iloc[bank['available_rows']].suite.to_numpy()
    plans = json.loads((ROOT/'fresh-plans.json').read_text())
    task_names = {job['base_task_id']: 'libero_long/'+job['base_task_name']
                  for job in plans['plans']['plus']['jobs']}
    rows, coverage = [], []
    cohorts = ['B', 'current']+(['fresh'] if (ROOT/'fresh-scores.npz').exists() else [])
    for cohort in cohorts:
        frame = pd.read_csv(ROOT/(cohort+'-decisions.csv'))
        data = archive(ROOT/(cohort+'-scores.npz'))
        canonical = frame.task if cohort == 'B' else frame.task_id.map(task_names)
        supported = canonical.isin(bank_tasks).to_numpy()
        methods = dict(v82=frame.v82_frozen.to_numpy(), instantaneous=frame.knn_available_first.to_numpy(),
                       **{name: frame[name+'_first'].to_numpy() for name in METHODS})
        for support, mask in (('all', np.ones(len(frame), bool)), ('supported', supported), ('unsupported', ~supported)):
            for name, first in methods.items():
                coverage.append(dict(cohort=cohort, reference_task_coverage=support, method=name,
                                     **alarm_metrics(frame[mask], first[mask])))
        for i, parent in enumerate(frame.itertuples()):
            task = parent.task if cohort == 'B' else task_names[parent.task_id]
            valid = data['neighbor_ids'][i, :, 0] >= 0
            ids = data['neighbor_ids'][i, valid]
            same_task = (bank_tasks[ids] == task).mean(-1)
            same_suite = (bank_suites[ids] == parent.suite).mean(-1)
            context = dict(cohort=cohort, parent_row=i, failure=bool(parent.failure),
                           task=task, task_present_in_bank=bool((bank_tasks == task).any()))
            if len(ids):
                rows.append(dict(**context, scope='all_scored_parent_mean', query=-1,
                    same_task_neighbor_fraction=same_task.mean(), same_suite_neighbor_fraction=same_suite.mean()))
            for method in METHODS:
                q = getattr(parent, method+'_first')
                if q >= 0:
                    ids = data['neighbor_ids'][i, q]
                    rows.append(dict(**context, scope=method+'_alarm', query=q,
                        same_task_neighbor_fraction=(bank_tasks[ids] == task).mean(),
                        same_suite_neighbor_fraction=(bank_suites[ids] == parent.suite).mean()))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT/'reference-context-parents.csv', index=False)
    summary = table.groupby(['cohort', 'scope', 'failure']).agg(
        parents=('parent_row', 'size'), task_supported_parents=('task_present_in_bank', 'sum'),
        median_same_task_fraction=('same_task_neighbor_fraction', 'median'),
        median_same_suite_fraction=('same_suite_neighbor_fraction', 'median')).reset_index()
    summary.to_csv(ROOT/'reference-context-summary.csv', index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'reference-coverage-metrics.csv', index=False)
    reference_tasks = set(json.loads((PREVIOUS/'available-profile/parameters.json').read_text())['reference_tasks'])
    save_json(ROOT/'reference-context-results.json', dict(posthoc_descriptive=True,
        scoring_or_threshold_changes=False, conditioning_not_applied=True,
        reference_A_tasks=len(reference_tasks), continuation_bank_tasks=len(set(bank_tasks)),
        A_tasks_without_continuation_anchor=sorted(reference_tasks-set(bank_tasks)),
        summary=summary.to_dict('records')))
    print(summary.to_string(index=False))
    print(pd.DataFrame(coverage).query('cohort == "B"').to_string(index=False))


if __name__ == '__main__':
    main()
