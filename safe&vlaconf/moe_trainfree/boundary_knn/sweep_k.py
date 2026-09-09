"""Sweep k on frozen full-corpus reference banks without refitting features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from full_corpus_knn import ROOT, HERE, CACHE, OUTPUT as PARENT, validate_split
from cosine_knn import load_npz, normalized_vectors
from diagnose_cosine import METHODS as OLD_METHODS
from core import conformal_threshold, digest, first_alarm, trajectory_peak, write_json

METHODS = OLD_METHODS[:5]
KS = np.asarray([*range(1, 11), 20], np.int16)
KINDS = ("episode", "task_init")
OUTPUT = HERE.parent / "results/round10_k_sweep"


def sorted_nearest(matrix):
    ids = np.argpartition(matrix, 19, axis=1)[:, :20]
    order = np.argsort(np.take_along_axis(matrix, ids, axis=1), axis=1, kind="stable")
    return np.take_along_axis(ids, order, axis=1)


def score_ks(x, bank, offset):
    valid = np.isfinite(x).all(-1)
    values = x[valid]
    for array in (values, bank, values + offset, bank + offset):
        assert (np.linalg.norm(array, axis=1) > 1e-12).all()
    output = np.full((len(METHODS), len(KS), *valid.shape), np.nan, np.float32)
    flat = np.empty((len(METHODS), len(KS), len(values)), np.float32)
    for start in range(0, len(values), 512):
        stop = min(start + 512, len(values))
        current = values[start:stop]
        e = cdist(current, bank, metric="euclidean")
        c = np.clip(cdist(current, bank, metric="cosine"), 0, 2)
        ei, ci = sorted_nearest(e), sorted_nearest(c)
        absolute = np.clip(cdist(current + offset, bank + offset, metric="cosine"), 0, 2)
        ai = sorted_nearest(absolute)
        for mi, matrix, ids in ((0, e, ei), (1, c, ci), (2, e, ci), (3, c, ei), (4, absolute, ai)):
            distances = np.take_along_axis(matrix, ids, axis=1)
            for ki, k in enumerate(KS):
                flat[mi, ki, start:stop] = distances[:, :k].mean(1)
    output[:, :, valid] = flat
    return output


def calibrate_ks(scores, labels, frame, ncal):
    success = labels == 0
    groups = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(METHODS), len(KS)), np.float64)
    first = np.empty((*thresholds.shape, scores.shape[2] - ncal), np.int16)
    records = []
    for mi, method in enumerate(METHODS):
        for ki, k in enumerate(KS):
            peaks = trajectory_peak(scores[mi, ki, :ncal][success])
            grouped = np.asarray([peaks[groups == key].max() for key in sorted(set(groups))])
            for kind_i, (kind, units) in enumerate(zip(KINDS, (peaks, grouped))):
                tau, rank = conformal_threshold(units, .05)
                thresholds[kind_i, mi, ki] = tau
                first[kind_i, mi, ki] = first_alarm(scores[mi, ki, ncal:], tau)
                records.append(dict(method=method, k=int(k), calibration=kind, threshold=tau,
                    units=len(units), rank=rank, exceedances=int((units > tau).sum())))
    fixed = np.stack([np.stack([first_alarm(score, thresholds[1, mi, -1]) for score in scores[mi, :, ncal:]])
                      for mi in range(len(METHODS))])
    return thresholds, first, fixed, records


def check_baseline(data, old):
    audits = []
    for mi, method in enumerate(METHODS):
        old_mi = list(old["methods"]).index(method)
        for key in ("scores", "calibration_scores"):
            actual, expected = data[key][mi, -1], old[key][old_mi]
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7, equal_nan=True)
            finite = np.isfinite(expected)
            audits.append(dict(method=method, population=key, queries=int(finite.sum()),
                max_abs_difference=float(np.max(np.abs(actual[finite] - expected[finite])))))
        np.testing.assert_allclose(data["thresholds"][:, mi, -1], old["thresholds"][:, old_mi], rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(data["first"][:, mi, -1], old["first"][:, old_mi])
    return audits


def score_all(output):
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads((PARENT / "sealed_manifest.json").read_text())
    verification = json.loads((PARENT / "verification.json").read_text())
    assert verification["passed"] and verification["sealed_manifest_sha256"] == digest(PARENT / "sealed_manifest.json")
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", PARENT)):
        for name, expected in parent[category].items():
            assert digest(root / name) == expected, name
    frame, cache = pd.read_csv(PARENT / "index.csv"), load_npz(CACHE)
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in
              (PARENT / "sealed_manifest.json", PARENT / "verification.json", PARENT / "index.csv", CACHE)}
    source_paths = (Path(__file__), HERE / "K_SWEEP_PROTOCOL_ZH.md", HERE / "full_corpus_knn.py",
        HERE / "cosine_knn.py", HERE / "knn.py", HERE.parent / "feature_geometry/analyze.py",
        HERE.parent / "safe_protocol/core.py", ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    output.mkdir(parents=True)
    for directory in ("predictions", "calibration"):
        (output / directory).mkdir()
    frame.to_csv(output / "index.csv", index=False)
    contract = dict(sources=sources, methods=METHODS, ks=KS, calibration_kinds=KINDS, alpha=.05,
        primary_method="euclidean", primary_calibration="task_init", same_reference_banks=True,
        same_test_ownership=True, test_outcomes_used_for_scoring=False, new_training=False, new_rollouts=False,
        historically_explored_data=True, fixed_k20_threshold_is_diagnostic=True)
    write_json(output / "scoring_contract.json", contract)
    folds, audits = [], []
    ownership = np.zeros(len(frame), np.int16)
    for info in parent["folds"]:
        started = time.perf_counter()
        paths = [PARENT / directory / f"{info['fold']}.npz" for directory in ("profiles", "predictions")]
        for path in paths:
            inputs[str(path.relative_to(ROOT))] = digest(path)
        profile, old = (load_npz(path) for path in paths)
        validate_split(frame, old["reference_rows"], old["calibration_rows"], old["test_rows"], info)
        needed = np.r_[old["calibration_rows"], old["test_rows"]]
        x = normalized_vectors(cache, needed, profile)
        scores = score_ks(x, profile["success_dynamic"], profile["dynamic_center"] / profile["dynamic_scale"])
        ncal = len(old["calibration_rows"])
        thresholds, first, fixed, records = calibrate_ks(scores, old["calibration_labels"],
            frame.loc[old["calibration_rows"]].reset_index(drop=True), ncal)
        data = {key: old[key] for key in ("reference_rows", "reference_labels", "calibration_rows", "calibration_labels", "test_rows", "checkpoint")}
        data.update(methods=np.asarray(METHODS), ks=KS, scores=scores[:, :, ncal:], calibration_scores=scores[:, :, :ncal],
                    thresholds=thresholds, first=first, first_fixed_k20=fixed)
        audits.extend(dict(fold=info["fold"], **row) for row in check_baseline(data, old))
        np.savez_compressed(output / "predictions" / f"{info['fold']}.npz", **data)
        pd.DataFrame(records).to_csv(output / "calibration" / f"{info['fold']}.csv", index=False)
        ownership[old["test_rows"]] += 1
        fold = dict(info, k_sweep_seconds=time.perf_counter() - started)
        folds.append(fold)
        print(f"SEALED {info['fold']}: {len(old['test_rows'])} tests, {fold['k_sweep_seconds']:.1f}s", flush=True)
    np.testing.assert_array_equal(ownership, 1)
    pd.DataFrame(audits).to_csv(output / "k20_replay_audit.csv", index=False)
    artifacts = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "sealed_manifest.json", dict(contract, inputs=inputs, artifacts=artifacts, folds=folds,
        unique_trajectories=len(frame), trajectory_configurations=len(frame) * len(METHODS) * len(KS)))
    print("ALL K SWEEP PREDICTIONS SEALED", flush=True)


def direct_ks(current, bank, offset):
    e = np.linalg.norm(bank - current, axis=1)
    c = np.clip(1 - (bank @ current) / (np.linalg.norm(bank, axis=1) * np.linalg.norm(current)), 0, 2)
    ei, ci = np.argsort(e), np.argsort(c)
    raw, ref = current + offset, bank + offset
    absolute = np.clip(1 - (ref @ raw) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(raw)), 0, 2)
    distances = (e[ei], c[ci], e[ci], c[ei], np.sort(absolute))
    return np.asarray([[values[:k].mean() for k in KS] for values in distances], np.float32)


def verify_all(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    hashes = 0
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
            hashes += 1
    frame, cache = pd.read_csv(output / "index.csv"), load_npz(CACHE)
    ownership = np.zeros(len(frame), np.int16)
    distance_points, threshold_checks, decisions, queries = 0, 0, 0, 0
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        old = load_npz(PARENT / "predictions" / f"{info['fold']}.npz")
        profile = load_npz(PARENT / "profiles" / f"{info['fold']}.npz")
        for key in ("reference_rows", "reference_labels", "calibration_rows", "calibration_labels", "test_rows", "checkpoint"):
            np.testing.assert_array_equal(data[key], old[key])
        np.testing.assert_array_equal(data["ks"], KS)
        np.testing.assert_array_equal(data["methods"], METHODS)
        validate_split(frame, data["reference_rows"], data["calibration_rows"], data["test_rows"], info)
        check_baseline(data, old)
        needed = np.r_[data["calibration_rows"], data["test_rows"]]
        x = normalized_vectors(cache, needed, profile)
        scores = np.concatenate((data["calibration_scores"], data["scores"]), axis=2)
        valid = np.isfinite(x).all(-1)
        for score in scores.reshape(-1, *valid.shape):
            np.testing.assert_array_equal(np.isfinite(score), valid)
        np.testing.assert_array_equal(valid, cache["valid"][needed] & (np.arange(52)[None] >= 7))
        for mi in (0, 1, 4):
            assert (np.diff(scores[mi][:, valid], axis=0) >= -1e-6).all()
            assert (np.diff(data["thresholds"][:, mi], axis=1) >= -1e-6).all()
        success = data["calibration_labels"] == 0
        cf = frame.loc[data["calibration_rows"]].reset_index(drop=True)
        groups = (cf.loc[success, "task"] + "|" + cf.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi in range(len(METHODS)):
            for ki, k in enumerate(KS):
                raw = data["calibration_scores"][mi, ki, success]
                peaks = np.max(np.where(np.isfinite(raw), raw, -np.inf), axis=1)
                grouped = np.asarray([peaks[groups == key].max() for key in sorted(set(groups))])
                for kind_i, units in enumerate((peaks, grouped)):
                    rank = int(np.ceil((len(units) + 1) * .95))
                    tau = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                    assert tau == data["thresholds"][kind_i, mi, ki]
                    raw_test = data["scores"][mi, ki]
                    crossed = np.isfinite(raw_test) & (raw_test > tau)
                    first = np.where(crossed.any(1), crossed.argmax(1), -1)
                    np.testing.assert_array_equal(first, data["first"][kind_i, mi, ki])
                    assert ((first == -1) | ((first >= 7) & (first < frame.loc[data["test_rows"], "length"].to_numpy()))).all()
                    threshold_checks += 1
                    decisions += len(first)
                crossed = np.isfinite(raw_test) & (raw_test > data["thresholds"][1, mi, -1])
                fixed = np.where(crossed.any(1), crossed.argmax(1), -1)
                np.testing.assert_array_equal(fixed, data["first_fixed_k20"][mi, ki])
                decisions += len(fixed)
        candidates = np.argwhere(valid)
        rng = np.random.default_rng(info["bank_seed"])
        selected = candidates[rng.choice(len(candidates), 12, replace=False)].tolist()
        selected += [list(np.unravel_index(np.nanargmax(scores[mi, ki]), valid.shape)) for mi, ki in ((0, 0), (0, -1), (1, 0))]
        for position, query in selected:
            expected = direct_ks(x[position, query], profile["success_dynamic"], profile["dynamic_center"] / profile["dynamic_scale"])
            np.testing.assert_allclose(scores[:, :, position, query], expected, rtol=3e-6, atol=3e-7)
            distance_points += 1
        ownership[data["test_rows"]] += 1
        queries += int(valid.sum())
        print(f"VERIFIED {info['fold']}", flush=True)
    np.testing.assert_array_equal(ownership, 1)
    result = dict(passed=True, hash_checks=hashes, unique_test_trajectories=len(frame),
        independently_recomputed_thresholds=threshold_checks, independently_recomputed_trajectory_decisions=decisions,
        independently_recomputed_distance_points=distance_points, configurations_per_distance_point=len(METHODS) * len(KS),
        calibration_and_test_valid_queries=queries, k20_all_scores_and_alarms_reproduced=True,
        sealed_manifest_sha256=digest(output / "sealed_manifest.json"))
    write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (score_all if args.command == "score" else verify_all)(args.output.resolve())


if __name__ == "__main__":
    main()
