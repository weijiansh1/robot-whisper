"""Analyze exact historical decisions before reconstructing any missing reference."""

import numpy as np
import pandas as pd

from v82_knn_research import (ROOT, RUN_A, RUN_B, load_frozen, combine_first,
                              alarm_metrics, alarm_groups, common_prefix_pairs, save_json)


def main():
    frame = load_frozen()
    methods = combine_first(frame.v82_frozen, frame.knn20)
    metrics, groups, timing = [], [], []
    cutoffs = [('all', 10000), ('q9', 9), ('q13', 13), ('q20', 20)]
    for fraction in (.25, .5, .75):
        cutoffs.append((f'budget_{fraction:.2f}', np.floor(frame.horizon_actions.to_numpy()*fraction/10)))
    cohorts = [('all', frame), ('A', frame[frame.run_id.eq(RUN_A)]), ('B', frame[frame.run_id.eq(RUN_B)])]
    cohorts += [(f'B_{suite}', part) for suite, part in frame[frame.run_id.eq(RUN_B)].groupby('suite')]
    for cohort, part in cohorts:
        ids = part.index.to_numpy()
        for cutoff_name, cutoff in cutoffs:
            selected_cutoff = cutoff[ids] if isinstance(cutoff, np.ndarray) else cutoff
            for name, alarms in methods.items():
                metrics.append(dict(cohort=cohort, cutoff=cutoff_name, method=name,
                                    **alarm_metrics(part, alarms[ids], selected_cutoff)))
            categories = alarm_groups(methods['v82'][ids], methods['knn'][ids], selected_cutoff)
            for category in ('neither', 'knn_only', 'v82_only', 'both'):
                selected = categories == category
                failed = int(part.failure.to_numpy()[selected].sum())
                size = int(selected.sum())
                groups.append(dict(cohort=cohort, cutoff=cutoff_name, group=category, episodes=size,
                                   failures=failed, successes=size-failed, failure_fraction=failed/size if size else None))
        for failure, outcome in ((False, 'success'), (True, 'failure')):
            shared = part[part.failure.eq(failure) & part.v82_frozen.ge(0) & part.knn20.ge(0)]
            delta = shared.v82_frozen.to_numpy() - shared.knn20.to_numpy()
            timing.append(dict(cohort=cohort, outcome=outcome, both=len(shared),
                               knn_first=int((delta > 0).sum()), v82_first=int((delta < 0).sum()), simultaneous=int((delta == 0).sum()),
                               median_v82_minus_knn=float(np.median(delta)) if len(delta) else None,
                               q25=float(np.quantile(delta, .25)) if len(delta) else None,
                               q75=float(np.quantile(delta, .75)) if len(delta) else None))
    pd.DataFrame(metrics).to_csv(ROOT / 'archived-alarm-metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(ROOT / 'archived-joint-groups.csv', index=False)
    pd.DataFrame(timing).to_csv(ROOT / 'archived-alarm-order.csv', index=False)
    frame['joint_group'] = alarm_groups(frame.v82_frozen, frame.knn20)
    frame.to_csv(ROOT / 'archived-episode-decisions.csv', index=False)
    b = frame[frame.run_id.eq(RUN_B)]
    pairs = common_prefix_pairs(b, methods)
    pairs.to_csv(ROOT / 'archived-common-prefix-pairs.csv', index=False)
    summaries = []
    for method in methods:
        per_task = pairs.groupby('task')[[method + '_failure_hit', method + '_success_hit']].mean()
        summaries.append(dict(method=method, pairs=len(pairs), tasks=len(per_task),
                              failure_hits=int(pairs[method + '_failure_hit'].sum()),
                              success_hits=int(pairs[method + '_success_hit'].sum()),
                              mean_task_failure_minus_success=float((per_task.iloc[:, 0]-per_task.iloc[:, 1]).mean())))
    pd.DataFrame(summaries).to_csv(ROOT / 'archived-common-prefix-summary.csv', index=False)
    task_rows = []
    for task, part in b.groupby('task'):
        categories = part.joint_group.to_numpy()
        for group in ('knn_only', 'v82_only', 'both', 'neither'):
            selected = part[categories == group]
            task_rows.append(dict(task=task, group=group, failures=int(selected.failure.sum()),
                                  successes=int((~selected.failure).sum())))
    pd.DataFrame(task_rows).to_csv(ROOT / 'archived-task-groups.csv', index=False)
    results = dict(full_episodes=len(frame), B_episodes=len(b),
                   B_metrics=[r for r in metrics if r['cohort'] == 'B'],
                   B_groups=[r for r in groups if r['cohort'] == 'B' and r['cutoff'] == 'all'],
                   B_timing=[r for r in timing if r['cohort'] == 'B'], common_prefix=summaries,
                   original_knn_scores_or_neighbors_reproduced=False)
    save_json(ROOT / 'archived-results.json', results)
    print(pd.DataFrame(results['B_metrics']).query('cutoff == "all"').to_string(index=False), flush=True)
    print(pd.DataFrame(results['B_groups']).to_string(index=False), flush=True)
    print(pd.DataFrame(results['B_timing']).to_string(index=False), flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
