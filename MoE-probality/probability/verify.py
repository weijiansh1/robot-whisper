"""Verify sampled raw Zarr prefixes against the dataset and online API."""

from __future__ import annotations

import hashlib

import numpy as np
import zarr

from .features import PrefixFeatures
from .model import SuccessMonitor, predict_readout


def verify_raw(hub, episodes, x, ep, bundles):
    samples, total, error, probability_error = [], 0, 0.0, 0.0
    starts = np.r_[0, np.cumsum(episodes.length.to_numpy())[:-1]]
    seen_suites = set()
    for source, group in episodes.groupby("source_run", sort=True):
        rank = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % len(group)
        row = group.iloc[rank]
        offset = int(group.length.iloc[:rank].sum())
        length = int(row.length)
        store = zarr.open_group(str(hub / source / "server/routes.zarr"), mode="r")
        ids = np.asarray(store["episode_id"][:])
        steps = np.asarray(store["control_step"][:])
        np.testing.assert_array_equal(ids, np.repeat(group.episode.to_numpy(), group.length.to_numpy()))
        if len(steps) != len(ids) or not np.all(np.diff(steps) == 1):
            raise AssertionError("raw source query order mismatch")
        raw = np.asarray(store["hb_router_probs"][offset:offset+length, :, 9, 1:, :])
        online = PrefixFeatures(int(row.max_steps), int(row.replan_steps))
        actual = np.concatenate([online.update(chunk)[0] for chunk in raw])
        start = int(starts[row.episode_row])
        expected = x[start:start+length]
        np.testing.assert_array_equal(ep[start:start+length], row.episode_row)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7, equal_nan=True)
        mask = np.isfinite(actual) & np.isfinite(expected)
        error = max(error, float(np.abs(actual[mask]-expected[mask]).max()))
        if row.suite not in seen_suites:
            bundle = bundles[row.suite]
            model = SuccessMonitor(bundle, bundle["policy"]["checkpoint_sha256"])
            p = np.asarray([model.update(chunk)["success_probability"] for chunk in raw])
            offline = predict_readout(bundle, expected)["moe"]
            np.testing.assert_allclose(p, offline, atol=1e-10, rtol=1e-10)
            probability_error = max(probability_error, float(np.max(np.abs(p-offline))))
            seen_suites.add(row.suite)
        samples.append(dict(source_run=source, episode=int(row.episode), queries=length))
        total += length
        if len(samples) % 10 == 0:
            print(f"raw verification {len(samples)}/80", flush=True)
    return dict(source_runs=len(samples), queries=total, max_feature_error=error,
                max_online_probability_error=probability_error,
                source_episode_ids_and_control_steps_verified=True,
                online_model_suites_checked=sorted(seen_suites), samples=samples)
