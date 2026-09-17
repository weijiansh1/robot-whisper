"""Freeze a separate A-only kNN profile using the original algorithm."""

from datetime import datetime, timezone
import json

import numpy as np
from threadpoolctl import threadpool_limits

from v82_knn_research import (ROOT, FROZEN, RAW, RUN_A, RUN_B, load_available, load_frozen,
                              dynamics, reference_pairs, reference_scaling, ReferenceScorer,
                              standardized, digest, save_json)


def main():
    output = ROOT / 'available-profile'
    if output.exists():
        raise SystemExit('The separate profile already exists; no refitting is allowed here')
    frame, cache = load_available()
    original = json.loads((FROZEN / 'profiles/parameters.json').read_text())
    geometry = original['geometry']
    reference = np.flatnonzero(frame.run_id.eq(RUN_A) & frame.init_state_id.isin(geometry['reference_initial_ids']))
    calibration = np.flatnonzero(frame.run_id.eq(RUN_A) & frame.init_state_id.isin(geometry['calibration_initial_ids']))
    assert set(reference).isdisjoint(calibration)
    assert not frame.iloc[np.r_[reference, calibration]].run_id.eq(RUN_B).any()
    assert set(zip(frame.iloc[reference].task, frame.iloc[reference].init_state_id)).isdisjoint(
        set(zip(frame.iloc[calibration].task, frame.iloc[calibration].init_state_id)))
    period = cache['periodicity'][reference]
    pscale = float(np.quantile(np.abs(period[np.isfinite(period)]), .75))
    dynamic, _ = dynamics(cache['mobility'][reference], cache['acceleration'][reference], period, pscale)
    dynamic[~cache['valid'][reference]] = np.nan
    normalized, scaling = reference_scaling(dynamic, np.arange(len(reference)))
    success = ~frame.iloc[reference].failure.to_numpy(bool)
    row, query = reference_pairs(dynamic, np.flatnonzero(success), cap=4096, seed=20260908).T
    profile = dict(dynamic_center=np.asarray(scaling['center']), dynamic_scale=np.asarray(scaling['scale']),
                   scaling_points=np.asarray(scaling['points']), periodicity_scale=np.asarray(pscale),
                   success_dynamic=normalized[row, query].astype(float),
                   success_global_rows=frame.iloc[reference[row]].global_row.to_numpy(int),
                   success_available_rows=reference[row], success_queries=query,
                   reference_available_rows=reference, calibration_available_rows=calibration,
                   reference_global_rows=frame.iloc[reference].global_row.to_numpy(int),
                   calibration_global_rows=frame.iloc[calibration].global_row.to_numpy(int),
                   checkpoint_set=np.asarray(sorted(frame.checkpoint.unique())))
    cal_cache = {key: value[calibration] for key, value in cache.items()}
    values, _ = standardized(cal_cache, profile)
    finite = np.isfinite(values).all(-1)
    cal_scores = np.full(finite.shape, np.nan, np.float32)
    distance, _ = ReferenceScorer(profile).neighbors('success_dynamic', values[finite])
    cal_scores[finite] = distance.mean(-1)
    part = frame.iloc[calibration]
    successful = ~part.failure.to_numpy(bool)
    keys = (part.task + '|' + part.init_state_id.astype(str)).to_numpy()
    peaks = np.where(np.isfinite(cal_scores[successful]), cal_scores[successful], -np.inf).max(1)
    unit_keys = sorted(set(keys[successful]))
    grouped = np.asarray([peaks[keys[successful] == key].max() for key in unit_keys])
    rank = int(np.ceil((len(grouped)+1) * .95))
    threshold = float(np.sort(grouped)[rank-1])
    assert np.isfinite(threshold) and (grouped > threshold).sum() <= len(grouped)+1-rank
    output.mkdir()
    np.savez_compressed(output / 'reference.npz', **profile)
    np.savez_compressed(output / 'calibration.npz', scores=cal_scores, available_rows=calibration,
                        failure=~successful, groups=keys, grouped_peaks=grouped, group_names=np.asarray(unit_keys))
    reference_tasks = sorted(frame.iloc[reference].task.unique())
    parameters = dict(schema='local.v82_knn.available_reference.v1', original_bank_reproduced=False,
                       algorithm='original 10D Euclidean success kNN, separate available-data profile',
                       seed=20260908, k=20, bank_points=len(row), reference_episodes=len(reference),
                       reference_successes=int(success.sum()), calibration_episodes=len(calibration),
                       calibration_successes=int(successful.sum()), calibration_groups=len(grouped),
                       alpha=.05, rank=rank, threshold=threshold, periodicity_scale=pscale,
                       reference_tasks=reference_tasks, b_used_for_fitting=False,
                       current_plus_pro_used_for_fitting=False,
                       original_threshold=geometry['thresholds']['knn20']['threshold'],
                       raw_input_hashes={str(RAW / (name + '_route_features.npz')): digest(RAW / (name + '_route_features.npz'))
                                         for name in ('development_main', 'external_8b')},
                       source_protocol_sha256=digest(ROOT / 'PROTOCOL.md'))
    save_json(output / 'parameters.json', parameters)
    save_json(output / 'frozen.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                                          frozen_before_B_and_current_scoring=True,
                                          files={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    frame.to_csv(ROOT / 'available-index.csv', index=False)
    missing = load_frozen().loc[lambda df: ~df.global_row.isin(frame.global_row)]
    missing.to_csv(ROOT / 'unavailable-histories.csv', index=False)
    print(json.dumps(parameters, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
