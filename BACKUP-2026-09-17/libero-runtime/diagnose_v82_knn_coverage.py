"""Controlled same-task versus equal-size random reference deletion."""

import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v82_knn_research import (ROOT, RUN_B, archive, load_available, load_frozen, standardized,
                              ReferenceScorer, first_crossing, alarm_metrics, digest, save_json)
from score_v82_knn_reference import load_profile


def main():
    target = ROOT / 'reference-deletion-scores.npz'
    if target.exists():
        raise SystemExit('Reference-deletion output already exists')
    profile, params = load_profile()
    frame, cache = load_available()
    selected = frame.run_id.eq(RUN_B).to_numpy()
    frame = frame[selected].reset_index(drop=True)
    values, _ = standardized({k: v[selected] for k, v in cache.items()}, profile)
    original = archive(ROOT / 'available-B-geometry.npz')['score']
    bank = profile['success_dynamic']
    bank_tasks = load_frozen().iloc[profile['success_global_rows']].task.to_numpy()
    scores = np.full((2, *original.shape), np.nan, np.float32)
    records, deleted_ids = [], {}
    started = time.perf_counter()
    for task_index, (task, part) in enumerate(frame.groupby('task', sort=True)):
        remove_task = np.flatnonzero(bank_tasks == task)
        if not len(remove_task):
            continue
        rng = np.random.default_rng(20260915 + task_index)
        remove_random = rng.choice(len(bank), len(remove_task), replace=False)
        ids = part.index.to_numpy()
        x = values[ids]
        finite = np.isfinite(x).all(-1)
        for method, removed in enumerate((remove_task, remove_random)):
            keep = np.ones(len(bank), bool)
            keep[removed] = False
            distance, _ = ReferenceScorer({'success_dynamic': bank[keep]}).neighbors('success_dynamic', x[finite])
            out = np.full(finite.shape, np.nan, np.float32)
            out[finite] = distance.mean(-1)
            assert (out[finite] + 1e-6 >= original[ids][finite]).all()
            scores[method, ids] = out
            deleted_ids[f'{task_index}_{method}'] = removed
            first = first_crossing(out, params['threshold'])
            before = first_crossing(original[ids], params['threshold'])
            for failure in (False, True):
                mask = part.failure.to_numpy(bool) == failure
                records.append(dict(task=task, task_index=task_index, method=('same_task', 'random')[method],
                                    failure=failure, episodes=int(mask.sum()), removed_points=len(removed),
                                    original_hits=int((before[mask] >= 0).sum()), deletion_hits=int((first[mask] >= 0).sum()),
                                    added_hits=int(((first[mask] >= 0) & (before[mask] < 0)).sum())))
        print(f'Deleted-reference controls: {task_index+1}/39 tasks, {time.perf_counter()-started:.1f}s', flush=True)
    np.savez_compressed(target, score=scores, methods=np.asarray(['same_task', 'random']))
    np.savez_compressed(ROOT / 'reference-deletion-identities.npz', **deleted_ids)
    table = pd.DataFrame(records)
    table.to_csv(ROOT / 'reference-deletion-by-task.csv', index=False)
    totals = table.groupby(['method', 'failure'])[['episodes', 'original_hits', 'deletion_hits', 'added_hits']].sum().reset_index()
    totals.to_csv(ROOT / 'reference-deletion-summary.csv', index=False)
    save_json(ROOT / 'reference-deletion-results.json', dict(
        posthoc_mechanism_diagnostic=True, threshold_unchanged=params['threshold'],
        profile_sha256=digest(ROOT / 'available-profile/reference.npz'),
        protocol_sha256=digest(ROOT / 'COVERAGE_DIAGNOSTIC.md'),
        summaries=totals.to_dict('records'), task_identity_is_offline_only=True))
    print(totals.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
