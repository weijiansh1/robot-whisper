"""Fit continuation references on A, freeze, then score historical/current data."""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v82_knn_research import (RUN_A, RUN_B, archive, load_available, standardized,
                              first_crossing, alarm_metrics, digest, save_json)
from score_v82_knn_reference import load_profile
from alarm_trajectory_study import ROOT, PREVIOUS, METHODS, score_trajectories, load_bank


def build():
    directory = ROOT / 'profile'
    if directory.exists():
        raise SystemExit('Profile exists; no refitting')
    previous, previous_params = load_profile()
    frame, raw = load_available()
    rows = previous['reference_available_rows']
    values, _ = standardized({key: value[rows] for key, value in raw.items()}, previous)
    pairs = []
    for i, row in enumerate(frame.iloc[rows].itertuples()):
        if row.failure:
            continue
        eligible = [q for q in range(9, row.length-3) if np.isfinite(values[i, q-2:q+4]).all()]
        if eligible:
            pairs.extend((i, eligible[j]) for j in np.linspace(0, len(eligible)-1, min(8, len(eligible))).astype(int))
    pairs = np.asarray(pairs)
    if len(pairs) > 4096:
        pairs = pairs[np.random.default_rng(20260908).choice(len(pairs), 4096, replace=False)]
    i, q = pairs.T
    prefix = np.stack([values[r, t-2:t+1].reshape(-1)/np.sqrt(3) for r, t in pairs])
    bank = dict(prefix=prefix, start=values[i, q], end=values[i, q+3], delta=values[i, q+3]-values[i, q],
                available_rows=rows[i], global_rows=frame.iloc[rows[i]].global_row.to_numpy(), queries=q)
    assert not frame.iloc[rows[i]].failure.any() and frame.iloc[rows[i]].run_id.eq(RUN_A).all()
    cal = previous['calibration_available_rows']
    assert set(rows).isdisjoint(cal)
    cal_values, _ = standardized({key: value[cal] for key, value in raw.items()}, previous)
    scores = score_trajectories(cal_values, bank)
    part = frame.iloc[cal]
    groups = (part.task + '|' + part.init_state_id.astype(str)).to_numpy(str)
    success = ~part.failure.to_numpy()
    keys = sorted(set(groups[success]))
    rank = int(np.ceil((len(keys)+1)*.95))
    thresholds, peaks = {}, {}
    for method in METHODS:
        value = np.where(np.isfinite(scores[method]), scores[method], -np.inf)
        peaks[method] = np.asarray([value[success & (groups == key)].max() for key in keys])
        thresholds[method] = float(np.sort(peaks[method])[rank-1])
    assert np.isfinite(list(thresholds.values())).all()
    directory.mkdir()
    np.savez_compressed(directory / 'trajectory-bank.npz', **bank)
    np.savez_compressed(directory / 'calibration.npz', **{key: scores[key] for key in METHODS},
                        **{key+'_group_peaks': peaks[key] for key in METHODS}, available_rows=cal,
                        group_names=np.asarray(keys), groups=groups, failure=~success)
    params = dict(reference_parents=len(rows), successful_bank_parents=len(np.unique(rows[i])),
                  bank_points=len(pairs), prefix_queries=3, continuation_queries=3, earliest_query=12,
                  calibration_parents=len(cal), calibration_groups=len(keys), calibration_rank=rank,
                  alpha=.05, thresholds=thresholds, k=20, seed=20260908, B_used_for_fitting=False,
                  previous_profile_sha256=digest(PREVIOUS / 'available-profile/reference.npz'),
                  protocol_sha256=digest(ROOT / 'PROTOCOL.md'))
    save_json(directory / 'parameters.json', params)
    save_json(directory / 'frozen.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
              files={path.name: digest(path) for path in directory.iterdir() if path.is_file()}))
    print(params, flush=True)


def score(cohort):
    target = ROOT / (cohort+'-scores.npz')
    if target.exists():
        raise SystemExit('Scores already exist')
    bank, params = load_bank()
    previous, _ = load_profile()
    if cohort == 'B':
        all_frame, raw = load_available()
        chosen = all_frame.run_id.eq(RUN_B).to_numpy()
        frame = pd.read_csv(PREVIOUS / 'available-B-decisions.csv')
        np.testing.assert_array_equal(frame.global_row, all_frame[chosen].global_row)
        values, _ = standardized({key: value[chosen] for key, value in raw.items()}, previous)
    else:
        frame = pd.read_csv(PREVIOUS / 'current-decisions.csv')
        values, _ = standardized(archive(PREVIOUS / 'current-routing-inputs.npz'), previous)
    scores = score_trajectories(values, bank)
    np.savez_compressed(target, **scores)
    for method in METHODS:
        frame[method+'_first'] = first_crossing(scores[method], params['thresholds'][method])
    frame.to_csv(ROOT / (cohort+'-decisions.csv'), index=False)
    save_json(ROOT / (cohort+'-scoring.json'), dict(episodes=len(frame),
              scored_queries=int(np.isfinite(scores['displacement']).sum()),
              profile_sha256=digest(ROOT / 'profile/trajectory-bank.npz'), model_forwards=0))
    for method in METHODS:
        print(cohort, method, alarm_metrics(frame, frame[method+'_first']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'B', 'current'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        build() if args.stage == 'build' else score(args.stage)
