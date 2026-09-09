"""Reuse the implemented v7 mechanisms under the SAFE-inspired task splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import zarr

from core import ROOT, ALPHAS, SEEDS, ReferenceDistance, digest, write_json, conformal_threshold, first_alarm, trajectory_peak
from run import load_suite, make_split, calibrate

V7 = ROOT / "moe-v7-0905/method"
sys.path.insert(0, str(V7))
from intrinsic_guard_monitor import IntrinsicGuardMonitor, intrinsic_score_arrays, row_max
from unlabeled_budget_calibration import calibrate_reference, calibrate_peaks, alarms_from_scores

HERE = Path(__file__).resolve().parent
METHODS = ("v7_guard_constant", "v7_guard_timeband", "v7_freeze_constant", "v7_turbulence_constant", "v7_success_fusion_constant")


def raw_episode(frame_row):
    source = ROOT / "VLA_MUI_HUB" / frame_row.source
    summaries = sorted(json.loads((source / "client/summaries.json").read_text()), key=lambda r: r["episode_index"])
    ordinal = next(i for i, r in enumerate(summaries) if int(r["episode_index"]) == frame_row.episode)
    begin = sum(r["inference_calls"] for r in summaries[:ordinal])
    group = zarr.open_group(str(source / "server/routes.zarr"), mode="r")
    return np.asarray(group["hb_router_probs"][begin:begin + frame_row.length])


def extract_episode(raw):
    previous, history = None, []
    m, a, p = [], [], []
    for query in raw:
        mobility, acceleration, periodicity, final = IntrinsicGuardMonitor._query_features(query, previous, history)
        m.append(mobility)
        a.append(acceleration)
        p.append(periodicity)
        history.append(final[4:].reshape(40, 32))
        previous = final
    return np.asarray(m, np.float32), np.asarray(a, np.float32), np.asarray(p, np.float32)


def load_inputs(parent, output, frame):
    layer_root = ROOT / "moe-v4-0904/results/layerwise_mobility"
    route_root = ROOT / "double-selete/trainfree/results"
    pairs = (("main_reference.npz", "online_multihead_hub"),
             ("extra_reference.npz", "online_multihead_hub_external"),
             ("external_8b.npz", "online_precision_cascade_external"))
    key_to_row = {(r.task, r.run_id, int(r.episode)): i for i, r in enumerate(frame.itertuples())}
    mobility = np.full((len(frame), 52, 8), np.nan, np.float32)
    acceleration = np.full((len(frame), 52), np.nan, np.float32)
    periodicity = np.full_like(acceleration, np.nan)
    filled = np.zeros(len(frame), bool)
    inputs = {}
    for layer_name, route_directory in pairs:
        lp, rp = layer_root / layer_name, route_root / route_directory / "unlabeled_query_features.npz"
        with np.load(lp, allow_pickle=False) as archive:
            layer = {k: archive[k] for k in archive.files}
        with np.load(rp, allow_pickle=False) as archive:
            route = {k: archive[k] for k in archive.files}
        for field in ("task_names", "task_index", "episode", "init_state_id", "flow_noise_seed", "length", "valid"):
            np.testing.assert_array_equal(layer[field], route[field])
        tasks = layer["task_names"].astype(str)[layer["task_index"]]
        rows = np.asarray([key_to_row[(t, str(layer["run_id"]), int(ep))] for t, ep in zip(tasks, layer["episode"])])
        for ours, cached in (("init_state_id", "init_state_id"), ("noise_seed", "flow_noise_seed"), ("length", "length")):
            np.testing.assert_array_equal(frame.iloc[rows][ours], layer[cached])
        if filled[rows].any():
            raise AssertionError("overlapping v7 source cache")
        names = list(route["feature_names"].astype(str))
        mobility[rows] = layer["mobility"]
        acceleration[rows] = route["features"][..., names.index("route_acceleration")]
        periodicity[rows] = route["features"][..., names.index("lag_periodicity")]
        filled[rows] = True
        inputs[str(lp.relative_to(ROOT))], inputs[str(rp.relative_to(ROOT))] = digest(lp), digest(rp)
    missing = np.flatnonzero(~filled)
    for i, row in enumerate(missing):
        record = frame.loc[row]
        m, a, p = extract_episode(raw_episode(record))
        mobility[row, :record.length], acceleration[row, :record.length], periodicity[row, :record.length] = m, a, p
        if (i + 1) % 100 == 0:
            print(f"V7 RAW COMPLETION {i+1}/{len(missing)}", flush=True)
    replays = []
    for (_, _), part in frame.groupby(["suite", "run_id"]):
        for row in part.index[[0, len(part) // 2, -1]]:
            record = frame.loc[row]
            m, a, p = extract_episode(raw_episode(record))
            for expected, actual in ((mobility[row, :record.length], m), (acceleration[row, :record.length], a), (periodicity[row, :record.length], p)):
                np.testing.assert_allclose(expected, actual, rtol=2e-4, atol=3e-6, equal_nan=True)
            replays.append({"global_row": int(row), "queries": int(record.length)})
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    mobility = np.where(valid[..., None], mobility, np.nan)
    acceleration, periodicity = np.where(valid, acceleration, np.nan), np.where(valid, periodicity, np.nan)
    np.savez_compressed(output / "v7_inputs.npz", mobility=mobility, acceleration=acceleration, periodicity=periodicity, valid=valid)
    write_json(output / "v7_input_audit.json", {"sources": inputs, "reused_episodes": int(filled.sum()),
        "new_raw_episodes": len(missing), "raw_replays": replays,
        "feature_implementation": str((V7 / "intrinsic_guard_monitor.py").relative_to(ROOT))})
    return mobility, acceleration, periodicity, valid


def standardize(values, reference):
    samples = values[reference]
    finite = samples[np.isfinite(samples)]
    center = float(np.median(finite))
    scale = max(float(1.4826 * np.median(np.abs(finite - center))), 1e-6)
    return (values - center) / scale, {"center": center, "scale": scale}


def peak_prefix(values):
    return np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    parent = args.input.resolve()
    output = parent / "v7"
    output.mkdir(exist_ok=False)
    for directory in ("predictions", "profiles", "calibration", "budget_predictions"):
        (output / directory).mkdir()
    (output / "features").symlink_to("../features", target_is_directory=True)
    for name in ("index.csv", "extraction_audit.json"):
        shutil.copyfile(parent / name, output / name)
    full = pd.read_csv(parent / "index.csv")
    mobility, acceleration, periodicity, valid = load_inputs(parent, output, full)
    summaries = []
    for suite_i, suite in enumerate(sorted(full.suite.unique())):
        frame, arrays, labels = load_suite(parent, full, suite)
        global_rows = frame.global_row.to_numpy()
        m, a, p, v = mobility[global_rows], acceleration[global_rows], periodicity[global_rows], valid[global_rows]
        for seed in SEEDS:
            begin = time.perf_counter()
            name = f"{suite}_{seed}"
            reference, calibration, test, seen, unseen = make_split(frame, seed, suite_i)
            periodicity_scale = float(np.quantile(np.abs(p[reference][np.isfinite(p[reference])]), 0.75))
            original = intrinsic_score_arrays(m, a, p, periodicity_scale)
            normalized, head_stats = {}, {}
            for head in ("freeze", "acceleration_persistent", "periodicity_persistent"):
                normalized[head], head_stats[head] = standardize(original[head], reference)
            turbulence = np.minimum(peak_prefix(normalized["acceleration_persistent"]), peak_prefix(normalized["periodicity_persistent"]))
            turbulence[~np.isfinite(turbulence)] = np.nan
            guard = np.fmax(normalized["freeze"], turbulence)
            with np.load(parent / "profiles" / f"{name}.npz", allow_pickle=False) as bank:
                success = bank["stats_success"]
            success_distance = ReferenceDistance(success, success).score(arrays["stats"])[..., 1]
            normalized_guard, fusion_guard_stats = standardize(guard, reference)
            normalized_distance, fusion_distance_stats = standardize(success_distance, reference)
            fusion = np.fmax(normalized_guard, normalized_distance)
            scores = np.asarray([guard, guard, normalized["freeze"], turbulence, fusion], dtype=np.float32)
            scores = np.where(v[None], scores, np.nan)
            center, scale, counts, thresholds, first, cal_records = calibrate(scores, labels, frame, reference, calibration, test)
            success_cal = calibration[labels[calibration] == 0]
            ids = (frame.iloc[success_cal].task + "|" + frame.iloc[success_cal].init_state_id.astype(str)).to_numpy()
            for method_i, method in enumerate(METHODS):
                if method.endswith("constant"):
                    center[method_i], scale[method_i] = 0, 1
                    peaks = trajectory_peak(scores[method_i, success_cal])
                    grouped = np.asarray([peaks[ids == group].max() for group in sorted(set(ids))])
                    for kind, units in enumerate((peaks, grouped)):
                        for alpha_i, alpha in enumerate(ALPHAS):
                            threshold, rank = conformal_threshold(units, alpha)
                            thresholds[kind, alpha_i, method_i] = threshold
                            first[kind, alpha_i, method_i] = first_alarm(scores[method_i, test], threshold)
                            for record in cal_records:
                                if record["method_index"] == method_i and record["kind"] == ("episode", "task_init")[kind] and record["alpha"] == alpha:
                                    record.update(threshold=threshold, rank=rank, exceedances=int((units > threshold).sum()))
            for record in cal_records:
                record["method"] = METHODS[record.pop("method_index")]
                record["fold"] = name
            np.savez_compressed(output / "predictions" / f"{name}.npz", scores=scores[:, test], methods=np.asarray(METHODS),
                test_rows=global_rows[test], test_unseen=frame.iloc[test].task.isin(unseen).to_numpy(),
                reference_rows=global_rows[reference], calibration_rows=global_rows[calibration],
                center=center, scale=scale, reference_counts=counts, thresholds=thresholds, first=first,
                alphas=np.asarray(ALPHAS), checkpoint=np.asarray(frame.checkpoint.iloc[0]), seed=np.asarray(seed), reference_cap=np.asarray(4096))
            budget_first = np.empty((2, len(ALPHAS), len(test)), np.int16)
            budget_profiles, budget_audits = [], []
            for alpha_i, alpha in enumerate(ALPHAS):
                auto = calibrate_reference(m[reference], a[reference], p[reference], v[reference], alarm_budget=alpha)
                scores_auto = intrinsic_score_arrays(m[test], a[test], p[test], auto.profile.periodicity_scale)
                budget_first[0, alpha_i] = alarms_from_scores(scores_auto, v[test], auto.profile)["guard"]
                peaks = [row_max(original[head][success_cal]) for head in ("freeze", "acceleration", "periodicity", "acceleration_persistent", "periodicity_persistent")]
                successful = calibrate_peaks(*peaks, periodicity_scale, alarm_budget=alpha)
                budget_first[1, alpha_i] = alarms_from_scores({head: x[test] for head, x in original.items()}, v[test], successful.profile)["guard"]
                for kind, fitted in (("unlabeled_reference_budget", auto), ("success_calibration_budget", successful)):
                    budget_profiles.append({"kind": kind, "alpha": alpha, **asdict(fitted.profile)})
                    budget_audits.append({"kind": kind, "alpha": alpha, **fitted.audit})
            np.savez_compressed(output / "budget_predictions" / f"{name}.npz", first=budget_first, alphas=np.asarray(ALPHAS), test_rows=global_rows[test], test_unseen=frame.iloc[test].task.isin(unseen).to_numpy())
            pd.DataFrame(cal_records).to_csv(output / "calibration" / f"{name}.csv", index=False)
            info = {"fold": name, "suite": suite, "seed": seed, "reference_cap": 4096, "seen_tasks": seen,
                "unseen_tasks": unseen, "reference_episodes": len(reference), "calibration_episodes": len(calibration),
                "test_episodes": len(test), "seconds": time.perf_counter() - begin}
            summaries.append(info)
            write_json(output / "profiles" / f"{name}.json", dict(info, head_stats=head_stats, periodicity_scale=periodicity_scale,
                fusion_guard_stats=fusion_guard_stats, fusion_distance_stats=fusion_distance_stats,
                budget_profiles=budget_profiles, budget_audits=budget_audits))
            print(f"V7 SEALED {name}: {info['seconds']:.1f}s", flush=True)
        del arrays
    artifacts = {str(p.relative_to(output)): digest(p) for directory in ("predictions", "profiles", "calibration", "budget_predictions") for p in sorted((output / directory).iterdir())}
    artifacts.update({name: digest(output / name) for name in ("index.csv", "extraction_audit.json", "v7_inputs.npz", "v7_input_audit.json")})
    write_json(output / "sealed_manifest.json", {"folds": summaries, "artifacts": artifacts,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), HERE / "V7_ZH.md", HERE / "core.py", HERE / "run.py", V7 / "intrinsic_guard_monitor.py", V7 / "unlabeled_budget_calibration.py")},
        "new_model_training": False, "test_labels_used_in_scoring": False, "primary": None,
        "user_requested_reuse_of_v7": True, "post_result_exploration": True})


if __name__ == "__main__":
    main()
