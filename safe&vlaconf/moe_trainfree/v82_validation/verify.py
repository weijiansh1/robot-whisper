"""Independent raw-route, calibration, and cumulative-count verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from monitor import ALPHAS, HERE, METHODS, ROOT, first_alarm, conformal_threshold
from run_analysis import archive, digest, write_json, state_splits
from intrinsic_guard_monitor import IntrinsicGuardMonitor, normalize_probability


def verify_raw(frame, cache, raw):
    rng = np.random.default_rng(20260908)
    checks = []
    for source, part in frame.groupby("source", sort=True):
        row = part.iloc[int(rng.integers(len(part)))]
        path = ROOT / "VLA_MUI_HUB" / source
        summaries = sorted(json.loads((path / "client/summaries.json").read_text()), key=lambda r: r["episode_index"])
        offsets, count = {}, 0
        for entry in summaries:
            n = int(entry["inference_calls"])
            offsets[int(entry["episode_index"])] = (count, count + n)
            count += n
        store = zarr.open_group(str(path / "server/routes.zarr"), mode="r")
        assert store["hb_router_probs"].shape[0] == count
        lo, hi = offsets[int(row.episode)]
        np.testing.assert_array_equal(np.asarray(store["episode_id"][lo:hi]), int(row.episode))
        probability = np.asarray(store["hb_router_probs"][lo:hi])
        previous, history = None, []
        mobility, acceleration, periodicity, additions = [], [], [], []
        for current in probability:
            m, a, p, final = IntrinsicGuardMonitor._query_features(current, previous, history)
            mobility.append(m)
            acceleration.append(a)
            periodicity.append(p)
            previous = final
            history.append(final[4:].reshape(40, 32))
            root = np.sqrt(normalize_probability(current)[:, :, 1:])
            speed = (np.linalg.norm(np.diff(root, axis=1), axis=-1) / np.sqrt(2.)).mean(axis=-1).astype(np.float32)
            path_length = speed.sum(axis=-1)
            inversion = -np.log(np.maximum(path_length[:4].mean(), 1e-12) / np.maximum(path_length[4:].mean(), 1e-12))
            curvature = np.abs(np.diff(speed[4:][:, [0, 4, 8]], n=2, axis=-1)).mean()
            additions.append([inversion, curvature])
        record = dict(source=source, episode=int(row.episode), global_row=int(row.global_row), queries=hi - lo)
        for name, expected in (("mobility", mobility), ("acceleration", acceleration), ("periodicity", periodicity)):
            actual = cache[name][int(row.global_row), :hi - lo]
            expected = np.asarray(expected)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-6, equal_nan=True)
            record[name + "_max_abs"] = float(np.nanmax(np.abs(actual - expected)))
        actual = raw[int(row.global_row), :hi - lo]
        np.testing.assert_allclose(actual, additions, rtol=2e-4, atol=3e-6, equal_nan=True)
        record["v8_max_abs"] = float(np.max(np.abs(actual - additions)))
        checks.append(record)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/v82_validation_20260908")
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "analysis_verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert digest(HERE / name) == expected, name
    inputs = json.loads((output / "input_verification.json").read_text())
    for name, expected in inputs["inputs"].items():
        assert digest(ROOT / name) == expected, name
    frame = pd.read_csv(output / "index.csv")
    prediction = archive(output / "crossfit_predictions.npz")
    np.testing.assert_array_equal(prediction["methods"], METHODS)
    np.testing.assert_array_equal(prediction["alphas"], ALPHAS)
    rows = prediction["global_rows"]
    lengths = frame.iloc[rows].length.to_numpy(int)
    valid = np.arange(52)[None] < lengths[:, None]
    np.testing.assert_array_equal(prediction["valid"], valid)
    assert not np.isfinite(prediction["scores"][:, ~valid]).any()
    thresholds = pd.read_csv(output / "calibration_thresholds.csv")
    ownership = np.zeros(len(frame), int)
    inverse = np.full(len(frame), -1, int)
    inverse[rows] = np.arange(len(rows))
    alarm_checks = 0
    for fold, ref, cal, test in state_splits(frame):
        ownership[test] += 1
        np.testing.assert_array_equal(prediction["fold_ids"][inverse[test]], fold)
        calibration = archive(output / ("fold_%d_calibration.npz" % fold))
        np.testing.assert_array_equal(calibration["rows"], cal)
        np.testing.assert_array_equal(calibration["failure"], frame.iloc[cal].failure.to_numpy())
        success = ~calibration["failure"]
        part = frame.iloc[cal].loc[success]
        keys = (part.task + "|" + part.init_state_id.astype(str)).to_numpy()
        group, names = pd.factorize(keys, sort=True)
        for mi, method in enumerate(METHODS):
            values = calibration["scores"][mi, success]
            peaks = np.where(np.isfinite(values), values, -np.inf).max(axis=1)
            grouped = np.full(len(names), -np.inf)
            np.maximum.at(grouped, group, peaks)
            for ki, (kind, units) in enumerate(zip(("episode", "task_init"), (peaks, grouped))):
                for ai, alpha in enumerate(ALPHAS):
                    threshold, rank = conformal_threshold(units, alpha)
                    row = thresholds.loc[(thresholds.fold == fold) & thresholds.method.eq(method) &
                                         thresholds.calibration.eq(kind) & (thresholds.alpha == alpha)].iloc[0]
                    np.testing.assert_allclose(threshold, row.threshold, rtol=1e-14)
                    assert row["rank"] == rank and row.units == len(units)
                    expected = first_alarm(prediction["scores"][mi, inverse[test]], valid[inverse[test]], threshold)
                    np.testing.assert_array_equal(expected, prediction["first"][ki, ai, mi, inverse[test]])
                    alarm_checks += len(test)
    np.testing.assert_array_equal(ownership[rows], 1)
    assert ownership.sum() == 16000
    curves = pd.read_csv(output / "cumulative_curves.csv")
    groups = ["family", "method", "calibration", "alpha", "scope"]
    for _, part in curves.groupby(groups, dropna=False):
        part = part.sort_values("query")
        np.testing.assert_array_equal(part["query"], np.arange(52))
        assert (np.diff(part.tp) >= 0).all() and (np.diff(part.fp) >= 0).all()
        assert part.failures.nunique() == part.successes.nunique() == 1
        assert (np.diff(part.active_failure) <= 0).all() and (np.diff(part.active_success) <= 0).all()
    cache = archive(HERE.parent / "results/round3_safe/v7/v7_inputs.npz")
    raw = archive(HERE.parent / "results/round7_temporal_fusion/v8_inputs.npz")["raw"]
    raw_checks = verify_raw(frame, cache, raw) if not args.skip_raw else []
    pd.DataFrame(raw_checks).to_csv(output / "raw_replay_checks.csv", index=False)
    write_json(output / "independent_verification.json", dict(input_hashes=len(inputs["inputs"]),
                artifact_hashes=len(manifest["artifacts"]), independent_alarm_checks=alarm_checks,
                raw_replayed_episodes=len(raw_checks), raw_replayed_queries=sum(r["queries"] for r in raw_checks),
                cumulative_curve_rows=len(curves), all_checks_passed=True, verifier_sha256=digest(__file__)))
    print("VERIFIED %d alarm decisions and %d raw episodes" % (alarm_checks, len(raw_checks)), flush=True)


if __name__ == "__main__":
    main()
