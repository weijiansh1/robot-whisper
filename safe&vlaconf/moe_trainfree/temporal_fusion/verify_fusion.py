"""Independent calibration, direct-neighbor, padding, and raw replay checks."""

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

from fusion import (HERE, ROOT, METHODS, ALPHAS, flow_speed, raw_v8_features,
                    v8_heads, base_streams, combine_streams, dynamics)
from core import digest, write_json, trajectory_peak
from run_fusion import load_npz
sys.path.insert(0, str(HERE))
from monitor import TemporalFusionMonitor
from v7_adapter import raw_episode, extract_episode


def audit_original_v8(output):
    sys.path.insert(0, str(ROOT / "moe-v8-0906/experiments"))
    import evaluate_full_corpus as legacy
    with np.load(ROOT / "moe-v8-0906/results/development_main_flow_speed.npz") as data:
        speed = data["flow_speed"]
    valid = np.isfinite(speed).all(axis=(-2, -1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old = legacy.heads_from_flow_speed(speed)
    corrected = v8_heads(raw_v8_features(speed))
    rows = []
    for i, name in enumerate(("frontback_flowpath", "curvature_3step")):
        values = old[name]
        reference = valid & (np.arange(52)[None] >= 6)
        target = -values if i == 0 else values
        np.testing.assert_allclose(corrected[..., i][reference], target[reference], rtol=3e-5, atol=3e-6)
        level = .005 if i == 0 else .995
        rows.append(dict(head=name, quantile=level,
            finite_real_queries=int((valid & np.isfinite(values)).sum()),
            finite_padded_queries=int((~valid & np.isfinite(values)).sum()),
            original_threshold=float(np.quantile(values[np.isfinite(values)], level, method="lower")),
            valid_only_threshold=float(np.quantile(values[valid & np.isfinite(values)], level, method="lower")),
            valid_after_warmup_threshold=float(np.quantile(values[reference & np.isfinite(values)], level, method="lower"))))
    pd.DataFrame(rows).to_csv(output / "v8_padding_audit.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    hash_checks = 0
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
        hash_checks += 1
    for kind in ("inputs", "sources"):
        for name, expected in manifest[kind].items():
            assert digest(ROOT / name) == expected, name
            hash_checks += 1
    frame = pd.read_csv(output / "outcome_alignment.csv")
    v7 = load_npz(HERE.parent / "results/round3_safe/v7/v7_inputs.npz")
    raw_features = load_npz(output / "v8_inputs.npz")["raw"]
    rank_checks, oracle_checks, replays, timings = 0, 0, [], []
    rng = np.random.default_rng(20260911)
    for info in manifest["folds"]:
        fold = info["fold"]
        data = load_npz(output / "predictions" / f"{fold}.npz")
        profile_path = output / "profiles" / f"{fold}.npz"
        profile = load_npz(profile_path)
        cal = frame.iloc[data["calibration_rows"]].reset_index(drop=True)
        success = data["calibration_labels"] == 0
        ids = (cal.loc[success, "task"] + "|" + cal.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi in range(len(METHODS)):
            peaks = trajectory_peak(data["calibration_scores"][mi, success])
            units = (peaks, np.asarray([np.max(peaks[ids == key]) for key in np.unique(ids)]))
            for ki, values in enumerate(units):
                ordered = np.sort(values)
                for ai, alpha in enumerate(ALPHAS):
                    rank = int(np.ceil((len(ordered)+1)*(1-alpha)))
                    expected = ordered[rank-1] if rank <= len(ordered) else np.inf
                    assert data["thresholds"][ki, ai, mi] == expected
                    crossing = data["scores"][mi] > expected
                    first = np.where(crossing.any(1), crossing.argmax(1), -1)
                    np.testing.assert_array_equal(data["first"][ki, ai, mi], first)
                    rank_checks += 1
        points = np.argwhere(np.isfinite(data["scores"][METHODS.index("knn12_v8")]))
        chosen = points[rng.choice(len(points), 6, replace=False)]
        for position, query in chosen:
            row = int(data["test_rows"][position])
            dynamic, _ = dynamics(v7["mobility"][[row]], v7["acceleration"][[row]], v7["periodicity"][[row]], float(profile["periodicity_scale"]))
            d = (dynamic[0, query].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
            h = (v8_heads(raw_features[[row]])[0, query].astype(np.float64) - profile["v8_center"]) / profile["v8_scale"]
            point = np.r_[d, h]
            distances = np.linalg.norm(profile["success_augmented"]-point, axis=1)
            expected = np.sort(distances)[:20].mean()
            np.testing.assert_allclose(data["scores"][METHODS.index("knn12_v8"), position, query], expected, rtol=2e-6, atol=2e-7)
            oracle_checks += 1
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        positions = []
        for failure in (True, False):
            eligible = np.flatnonzero(part.failure.to_numpy() == failure)
            positions.append(int(eligible[len(eligible)//2]))
        for position in positions:
            row = int(data["test_rows"][position])
            record = frame.iloc[row]
            raw = raw_episode(record)
            m, a, p = extract_episode(raw)
            extra = raw_v8_features(flow_speed(raw))
            np.testing.assert_allclose(extra, raw_features[row, :len(raw)], rtol=2e-4, atol=3e-6)
            base = base_streams(profile, m[None], a[None], p[None], extra[None])
            scores = combine_streams(base, profile)[:, 0]
            cached_base = base_streams(profile, v7["mobility"][[row]], v7["acceleration"][[row]],
                v7["periodicity"][[row]], raw_features[[row]])
            cached_scores = combine_streams(cached_base, profile)[:, 0, :len(raw)]
            np.testing.assert_array_equal(cached_scores, data["scores"][:, position, :len(raw)])
            # Sub-micro float32 mobility differences are amplified by MAD scaling.
            np.testing.assert_allclose(scores, cached_scores, rtol=3e-4, atol=2e-4, equal_nan=True)
            raw_max_error = float(np.nanmax(np.abs(scores-cached_scores)))
            ai = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
            for method_i in range(len(METHODS)):
                hit = scores[method_i] > data["thresholds"][1, ai, method_i]
                first = int(hit.argmax()) if hit.any() else -1
                assert first == data["first"][1, ai, method_i, position]
            monitor = TemporalFusionMonitor(profile_path, str(profile["checkpoint"]))
            trace = []
            for query in raw:
                start = time.perf_counter()
                result = monitor.update(query)
                elapsed = 1000*(time.perf_counter()-start)
                if result["query"] >= 7:
                    timings.append(elapsed)
                trace.append(result["score"])
            mi = METHODS.index(monitor.method)
            ai = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
            np.testing.assert_allclose(trace, scores[mi], rtol=1e-6, atol=2e-7, equal_nan=True)
            assert monitor.first_alarm_query == data["first"][1, ai, mi, position]
            monitor.reset()
            assert monitor.update(raw[0])["first_alarm_query"] == -1
            replays.append(dict(fold=fold, global_row=row, queries=len(raw), all_methods=len(METHODS),
                                max_raw_score_error=raw_max_error))
        print(f"FUSION VERIFIED {fold}", flush=True)
    padding = audit_original_v8(output)
    write_json(output / "verification.json", dict(hash_checks=hash_checks, calibration_rank_checks=rank_checks,
        direct_knn12_score_checks=oracle_checks, complete_raw_replays=replays, raw_methods_per_replay=len(METHODS),
        online_primary="knn_and_v8", query_timing_ms=dict(count=len(timings), median=float(np.median(timings)),
        p95=float(np.quantile(timings, .95))), historical_v8_padding_audit=padding,
        verifier_sha256=digest(Path(__file__)), all_checks_passed=True))
    print("ALL FUSION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
