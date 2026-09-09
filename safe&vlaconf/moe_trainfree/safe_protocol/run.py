"""Fit only reference statistics, seal SAFE-style train-free online predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from core import (ROOT, RUNS, SEEDS, ALPHAS, PRIMARY, METHODS, BASE_NAMES, VECTOR_NAMES,
                  ReferenceDistance, reference_points, aggregate, reference_band,
                  trajectory_peak, conformal_threshold, first_alarm, digest, write_json)

HERE = Path(__file__).resolve().parent


def make_split(frame, seed, suite_index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, suite_index]))
    tasks = np.asarray(sorted(frame.task.unique()))
    tasks = tasks[rng.permutation(len(tasks))]
    seen, unseen = tasks[:7], tasks[7:]
    states = rng.permutation(50)
    ref_states, cal_states, test_states = states[:30], states[30:40], states[40:]
    run = frame.run_id.to_numpy()
    in_seen = frame.task.isin(seen).to_numpy()
    is_unseen = frame.task.isin(unseen).to_numpy()
    reference = np.flatnonzero((run == RUNS[0]) & in_seen & frame.init_state_id.isin(ref_states))
    calibration = np.flatnonzero((run == RUNS[0]) & in_seen & frame.init_state_id.isin(cal_states))
    test = np.flatnonzero((run == RUNS[1]) & (is_unseen | (in_seen & frame.init_state_id.isin(test_states))))
    sets = [{(frame.iloc[i].task, int(frame.iloc[i].init_state_id)) for i in ix} for ix in (reference, calibration, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise AssertionError("task/init leakage")
    if set(frame.iloc[reference].noise_seed) & set(frame.iloc[test].noise_seed):
        raise AssertionError("noise cohorts overlap")
    return reference, calibration, test, seen.tolist(), unseen.tolist()


def load_suite(output, full_frame, suite):
    part = full_frame.loc[full_frame.suite == suite].copy()
    part["global_row"] = part.index
    part = part.reset_index(drop=True)
    arrays = {name: np.full((len(part), 52, size), np.nan, np.float32)
              for name, size in (("load", 256), ("stats", 32), ("history", 40), ("behavior", 4), ("direct", 6))}
    audits = json.loads((output / "extraction_audit.json").read_text())
    for audit in audits:
        rows = np.flatnonzero(part.source == audit["source"])
        if not len(rows):
            continue
        with np.load(output / "features" / audit["output"], allow_pickle=False) as archive:
            stored = pd.DataFrame(json.loads(str(archive["index"])))
            for field in ("episode", "length", "init_state_id", "noise_seed", "checkpoint"):
                if not np.array_equal(part.iloc[rows][field], stored[field]):
                    raise ValueError(f"feature index mismatch {field}")
            for name in arrays:
                arrays[name][rows] = archive[name]
    labels = np.full(len(part), -1, np.int8)
    for source, rows in part.loc[part.run_id == RUNS[0]].groupby("source").groups.items():
        summaries = json.loads((ROOT / "VLA_MUI_HUB" / source / "client/summaries.json").read_text())
        successes = {int(s["episode_index"]): bool(s["success"]) for s in summaries}
        labels[rows] = [int(not successes[int(part.loc[i, "episode"])]) for i in rows]
    if (labels[part.run_id == RUNS[1]] != -1).any():
        raise AssertionError("test outcome entered scorer")
    return part, arrays, labels


def build_scores(frame, arrays, labels, reference, needed, seed, cap=4096):
    base = {}
    banks = {}
    timings = []
    for name in VECTOR_NAMES:
        if cap != 4096 and name != "stats":
            continue
        values = arrays[name]
        started = time.perf_counter()
        success = reference_points(values, reference[labels[reference] == 0], cap, seed)
        failure = reference_points(values, reference[labels[reference] == 1], cap, seed + 1)
        distance = ReferenceDistance(success, failure, normalize=name != "load")
        predictions = distance.score(values[needed])
        base[f"{name}_contrast"] = predictions[..., 0]
        if name == "stats":
            base["stats_success"] = predictions[..., 1]
        banks[f"{name}_success"] = success
        banks[f"{name}_failure"] = failure
        timings.append({"representation": name, "reference_cap": cap, "success_points": len(success),
                        "failure_points": len(failure), "seconds": time.perf_counter() - started,
                        "queries": int(np.isfinite(values[needed]).all(-1).sum())})
    if cap != 4096:
        names = ("stats_contrast__current", "stats_contrast__cumsum")
        return np.stack([aggregate(base["stats_contrast"], mode) for mode in ("current", "cumsum")]), names, banks, timings
    for column, name in enumerate(BASE_NAMES[5:]):
        base[name] = arrays["direct"][needed, :, column]
    values = [aggregate(base[name], mode) for name in BASE_NAMES for mode in ("current", "cumsum")]
    valid = np.arange(52)[None] < frame.iloc[needed].length.to_numpy()[:, None]
    clock = np.broadcast_to(np.arange(52), valid.shape)
    # The same per-episode random trajectory is reused across task splits.
    random = np.empty(valid.shape, np.float32)
    for i, row in enumerate(frame.iloc[needed].global_row):
        random[i] = np.random.default_rng(np.random.SeedSequence([20260907, int(row)])).random(52)
    values += [np.where(valid, clock, np.nan), np.where(valid, random, np.nan)]
    return np.stack(values).astype(np.float32), METHODS, banks, timings


def calibrate(scores, labels, frame, reference_pos, calibration_pos, test_pos):
    m = len(scores)
    center, scale = np.empty((m, 52)), np.empty((m, 52))
    counts = np.empty((m, 52), int)
    thresholds = np.empty((2, len(ALPHAS), m))
    first = np.empty((2, len(ALPHAS), m, len(test_pos)), np.int16)
    records = []
    success_ref = reference_pos[labels[reference_pos] == 0]
    success_cal = calibration_pos[labels[calibration_pos] == 0]
    group_ids = (frame.iloc[success_cal].task + "|" + frame.iloc[success_cal].init_state_id.astype(str)).to_numpy()
    groups = [np.flatnonzero(group_ids == key) for key in sorted(set(group_ids))]
    for method_i, score in enumerate(scores):
        center[method_i], scale[method_i], counts[method_i] = reference_band(score[success_ref])
        standardized = (score - center[method_i]) / scale[method_i]
        peaks = trajectory_peak(standardized[success_cal])
        grouped = np.asarray([np.max(peaks[ix]) for ix in groups])
        for calibration_i, (kind, examples) in enumerate((("episode", peaks), ("task_init", grouped))):
            for alpha_i, alpha in enumerate(ALPHAS):
                threshold, rank = conformal_threshold(examples, alpha)
                thresholds[calibration_i, alpha_i, method_i] = threshold
                first[calibration_i, alpha_i, method_i] = first_alarm(standardized[test_pos], threshold)
                exceedances = int((examples > threshold).sum())
                if exceedances > len(examples) + 1 - rank:
                    raise AssertionError("conformal rank check failed")
                records.append({"method_index": method_i, "kind": kind, "alpha": alpha,
                                "calibration_units": len(examples), "rank": rank,
                                "threshold": threshold, "exceedances": exceedances,
                                "success_calibration_episodes": len(success_cal)})
    return center, scale, counts, thresholds, first, records


def run_fold(output, suite, suite_index, seed, frame, arrays, labels, cap=4096):
    name = f"{suite}_{seed}" + (f"_cap{cap}" if cap != 4096 else "")
    path = output / "predictions" / f"{name}.npz"
    if path.exists():
        raise FileExistsError(f"predictions already exist: {path}")
    started = time.perf_counter()
    reference, calibration, test, seen, unseen = make_split(frame, seed, suite_index)
    needed = np.unique(np.concatenate((reference, calibration, test)))
    remap = np.full(len(frame), -1, int)
    remap[needed] = np.arange(len(needed))
    scores, names, banks, timings = build_scores(frame, arrays, labels, reference, needed, seed, cap)
    center, scale, counts, thresholds, first, records = calibrate(
        scores, labels[needed], frame.iloc[needed].reset_index(drop=True), remap[reference], remap[calibration], remap[test])
    for record in records:
        record["method"] = names[record.pop("method_index")]
        record["fold"] = name
    global_rows = frame.global_row.to_numpy()
    np.savez_compressed(path, scores=scores[:, remap[test]], methods=np.asarray(names),
                        test_rows=global_rows[test], test_unseen=frame.iloc[test].task.isin(unseen).to_numpy(),
                        reference_rows=global_rows[reference], calibration_rows=global_rows[calibration],
                        center=center, scale=scale, reference_counts=counts, thresholds=thresholds,
                        first=first, alphas=np.asarray(ALPHAS), checkpoint=np.asarray(frame.checkpoint.iloc[0]),
                        seed=np.asarray(seed), reference_cap=np.asarray(cap))
    if cap == 4096:
        np.savez_compressed(output / "profiles" / f"{name}.npz", **banks,
                            center=center, scale=scale, thresholds=thresholds, methods=np.asarray(names),
                            alphas=np.asarray(ALPHAS), checkpoint=np.asarray(frame.checkpoint.iloc[0]))
    pd.DataFrame(records).to_csv(output / "calibration" / f"{name}.csv", index=False)
    info = {"fold": name, "suite": suite, "seed": seed, "reference_cap": cap,
            "seen_tasks": seen, "unseen_tasks": unseen, "reference_episodes": len(reference),
            "reference_failures": int((labels[reference] == 1).sum()),
            "calibration_episodes": len(calibration), "test_episodes": len(test),
            "timings": timings, "seconds": time.perf_counter() - started,
            "prediction_sha256": digest(path)}
    write_json(output / "profiles" / f"{name}.json", info)
    print(f"SEALED {name}: ref {len(reference)}, calibration {len(calibration)}, test {len(test)}, {info['seconds']:.1f}s", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.input.resolve()
    if (output / "sealed_manifest.json").exists():
        raise FileExistsError("already sealed")
    for directory in ("predictions", "profiles", "calibration"):
        (output / directory).mkdir(exist_ok=True)
    code = {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), HERE / "core.py", HERE / "PROTOCOL_ZH.md")}
    contract = output / "scoring_contract.json"
    if contract.exists():
        if json.loads(contract.read_text())["sources"] != code:
            raise ValueError("scoring code changed after experiment start")
    else:
        write_json(contract, {"sources": code, "primary": PRIMARY, "label_use": "reference and success calibration only"})
    full_frame = pd.read_csv(output / "index.csv")
    summaries = []
    for suite_index, suite in enumerate(sorted(full_frame.suite.unique())):
        frame, arrays, labels = load_suite(output, full_frame, suite)
        for seed in SEEDS:
            for cap in (4096, 512, 1024):
                name = f"{suite}_{seed}" + (f"_cap{cap}" if cap != 4096 else "")
                info_path = output / "profiles" / f"{name}.json"
                if args.resume and info_path.exists():
                    info = json.loads(info_path.read_text())
                    if digest(output / "predictions" / f"{name}.npz") != info["prediction_sha256"]:
                        raise ValueError("existing prediction changed")
                    summaries.append(info)
                else:
                    summaries.append(run_fold(output, suite, suite_index, seed, frame, arrays, labels, cap))
        del arrays
    artifacts = {str(p.relative_to(output)): digest(p) for directory in ("predictions", "profiles", "calibration")
                 for p in sorted((output / directory).iterdir()) if p.is_file()}
    artifacts.update({str(p.relative_to(output)): digest(p) for p in sorted((output / "features").glob("*.npz"))})
    for name in ("index.csv", "extraction_audit.json", "scoring_contract.json"):
        artifacts[name] = digest(output / name)
    write_json(output / "sealed_manifest.json", {"sources": code, "folds": summaries, "artifacts": artifacts,
               "test_outcomes_used_for_scoring": False, "new_model_training": False, "label_free": False,
               "historically_explored_data": True, "primary": PRIMARY, "primary_alpha": 0.05})
    print("ALL PREDICTIONS SEALED", flush=True)


if __name__ == "__main__":
    main()
