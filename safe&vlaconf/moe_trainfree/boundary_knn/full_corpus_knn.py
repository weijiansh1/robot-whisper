"""Score every historical trajectory once with its entire task held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from knn import HERE, ROOT, dynamics, reference_pairs, reference_scaling
from cosine_knn import load_npz, normalized_vectors
from diagnose_cosine import METHODS, score_factorial, calibrate
from core import RUNS, digest, write_json

OUTPUT = HERE.parent / "results/round9_full_corpus"
PARENT = HERE.parent / "results/round5_knn"
CACHE = HERE.parent / "results/round3_safe/v7/v7_inputs.npz"
SEED = 20260907
KINDS = ("episode", "task_init")


def split_suite(frame, suite_index):
    tasks = np.asarray(sorted(frame.task.unique()))
    assert len(tasks) == 10 and len(frame) == 8000
    assert frame.checkpoint.nunique() == 1
    rng = np.random.default_rng(np.random.SeedSequence([SEED, suite_index]))
    tasks = tasks[rng.permutation(10)]
    for fold_index, start in enumerate((0, 3, 6, 9)):
        owned = tasks[start:start + 3].tolist()
        excluded = owned if len(owned) == 3 else owned + tasks[:2].tolist()
        seen = [task for task in tasks if task not in excluded]
        states = np.random.default_rng(np.random.SeedSequence([SEED, suite_index, fold_index, 1])).permutation(50)
        eligible = frame.run_id.eq(RUNS[0]) & frame.task.isin(seen)
        reference = frame.index[eligible & frame.init_state_id.isin(states[:30])].to_numpy()
        calibration = frame.index[eligible & frame.init_state_id.isin(states[30:40])].to_numpy()
        test = frame.index[frame.task.isin(owned)].to_numpy()
        assert (len(reference), len(calibration), len(test)) == (1680, 560, 800 * len(owned))
        info = dict(fold=f"{frame.suite.iloc[0]}_full_{fold_index}", suite=frame.suite.iloc[0],
            suite_index=suite_index, fold_index=fold_index, seed=SEED,
            bank_seed=SEED + 100 * suite_index + fold_index, test_tasks=owned,
            excluded_tasks=excluded, reference_tasks=seen,
            reference_init_states=states[:30].tolist(), calibration_init_states=states[30:40].tolist())
        yield info, reference, calibration, test


def historical_labels(frame, rows, inputs):
    labels = np.full(len(frame), -1, np.int8)
    part = frame.loc[rows]
    assert part.run_id.eq(RUNS[0]).all()
    for source, group in part.groupby("source"):
        path = ROOT / "VLA_MUI_HUB" / source / "client/summaries.json"
        inputs[str(path.relative_to(ROOT))] = digest(path)
        summaries = {int(entry["episode_index"]): entry for entry in json.loads(path.read_text())}
        for row in group.itertuples():
            entry = summaries[int(row.episode)]
            assert isinstance(entry["success"], bool)
            assert int(entry["inference_calls"]) == row.length
            labels[row.Index] = int(not entry["success"])
    assert np.isin(labels[rows], [0, 1]).all()
    return labels


def build_profile(cache, frame, reference, labels, seed):
    p = cache["periodicity"][reference]
    p_scale = float(np.quantile(np.abs(p[np.isfinite(p)]), .75))
    dynamic, _ = dynamics(cache["mobility"][reference], cache["acceleration"][reference], p, p_scale)
    dynamic[~cache["valid"][reference]] = np.nan
    normalized, scaling = reference_scaling(dynamic, np.arange(len(reference)))
    pairs = reference_pairs(dynamic, np.flatnonzero(labels[reference] == 0), seed=seed)
    row, query = pairs.T
    return dict(dynamic_center=np.asarray(scaling["center"]), dynamic_scale=np.asarray(scaling["scale"]),
        scaling_points=np.asarray(scaling["points"]), periodicity_scale=np.asarray(p_scale),
        success_dynamic=normalized[row, query].astype(np.float64),
        success_global_rows=reference[row], success_queries=query,
        checkpoint=np.asarray(frame.loc[reference, "checkpoint"].iloc[0]))


def validate_split(frame, reference, calibration, test, info):
    sets = [set(rows.tolist()) for rows in (reference, calibration, test)]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    ref_keys, cal_keys = [set(zip(frame.loc[rows, "task"], frame.loc[rows, "init_state_id"]))
                          for rows in (reference, calibration)]
    assert not ref_keys & cal_keys
    assert frame.loc[np.r_[reference, calibration], "run_id"].eq(RUNS[0]).all()
    assert set(frame.loc[test, "task"]) == set(info["test_tasks"])
    assert not set(frame.loc[test, "task"]) & set(frame.loc[np.r_[reference, calibration], "task"])
    assert set(frame.loc[reference, "task"]) == set(info["reference_tasks"])
    assert set(frame.loc[calibration, "task"]) == set(info["reference_tasks"])
    assert frame.loc[np.r_[reference, calibration, test], "checkpoint"].nunique() == 1


def score_all(output):
    if output.exists():
        raise FileExistsError(output)
    old = json.loads((PARENT / "sealed_manifest.json").read_text())
    assert digest(PARENT / "index.csv") == old["artifacts"]["index.csv"]
    assert digest(CACHE) == old["inputs"][str(CACHE.relative_to(ROOT))]
    source_paths = (Path(__file__), HERE / "FULL_CORPUS_PROTOCOL_ZH.md", HERE / "knn.py",
        HERE / "cosine_knn.py", HERE / "diagnose_cosine.py", HERE / "compare_cosine.py",
        HERE.parent / "feature_geometry/analyze.py", HERE.parent / "safe_protocol/core.py",
        ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in (PARENT / "index.csv", PARENT / "sealed_manifest.json", CACHE)}
    frame = pd.read_csv(PARENT / "index.csv")
    assert len(frame) == 32000 and not frame.duplicated(["source", "episode"]).any()
    assert set(frame.run_id) == set(RUNS)
    cache = load_npz(CACHE)
    np.testing.assert_array_equal(cache["valid"], np.arange(52)[None] < frame.length.to_numpy()[:, None])
    output.mkdir(parents=True)
    for name in ("profiles", "predictions", "calibration"):
        (output / name).mkdir()
    frame.to_csv(output / "index.csv", index=False)
    contract = dict(sources=sources, seed=SEED, methods=METHODS, calibration_kinds=KINDS,
        alpha=.05, primary_calibration="task_init", test_ownership="disjoint task groups 3,3,3,1 per suite",
        new_model_training=False, new_rollouts=False, historically_explored_data=True,
        labels_for_own_test_tasks_used_by_fold=False, historical_labels_used_in_other_folds=True,
        nominal_unseen_task_fpr_guarantee=False)
    write_json(output / "scoring_contract.json", contract)
    folds, assignments, audits = [], [], []
    tested = np.zeros(len(frame), np.int16)
    for suite_index, (_, part) in enumerate(frame.groupby("suite", sort=True)):
        for info, reference, calibration, test in split_suite(part, suite_index):
            started = time.perf_counter()
            validate_split(frame, reference, calibration, test, info)
            labels = historical_labels(frame, np.r_[reference, calibration], inputs)
            assert (labels[test] == -1).all()
            profile = build_profile(cache, frame, reference, labels, info["bank_seed"])
            needed = np.r_[calibration, test]
            x = normalized_vectors(cache, needed, profile)
            scores, neighbor_ids = score_factorial(x, profile["success_dynamic"], profile["dynamic_center"] / profile["dynamic_scale"])
            del neighbor_ids
            ncal = len(calibration)
            thresholds, first, records = calibrate(scores, labels[calibration], frame.loc[calibration].reset_index(drop=True), ncal)
            np.savez_compressed(output / "profiles" / f"{info['fold']}.npz", **profile,
                thresholds=thresholds, methods=np.asarray(METHODS))
            np.savez_compressed(output / "predictions" / f"{info['fold']}.npz", methods=np.asarray(METHODS),
                scores=scores[:, ncal:], calibration_scores=scores[:, :ncal], thresholds=thresholds, first=first,
                reference_rows=reference, reference_labels=labels[reference], calibration_rows=calibration,
                calibration_labels=labels[calibration], test_rows=test, checkpoint=profile["checkpoint"])
            pd.DataFrame(records).to_csv(output / "calibration" / f"{info['fold']}.csv", index=False)
            tested[test] += 1
            assignments.extend(dict(global_row=int(row), fold=info["fold"], task=frame.loc[row, "task"]) for row in test)
            info.update(reference_episodes=len(reference), calibration_episodes=ncal, test_episodes=len(test),
                bank_points=len(profile["success_dynamic"]), reference_successes=int((labels[reference] == 0).sum()),
                calibration_successes=int((labels[calibration] == 0).sum()),
                test_score_queries=int(np.isfinite(scores[0, ncal:]).sum()),
                calibration_score_queries=int(np.isfinite(scores[0, :ncal]).sum()), seconds=time.perf_counter() - started)
            audits.append(dict(fold=info["fold"], minimum_query_norm=float(np.linalg.norm(x[np.isfinite(x).all(-1)], axis=1).min()),
                minimum_bank_norm=float(np.linalg.norm(profile["success_dynamic"], axis=1).min())))
            folds.append(info)
            write_json(output / "profiles" / f"{info['fold']}.json", info)
            print(f"SEALED {info['fold']}: {len(test)} tests, {info['test_score_queries']} test chunks, {info['seconds']:.1f}s", flush=True)
    np.testing.assert_array_equal(tested, 1)
    pd.DataFrame(assignments).sort_values("global_row").to_csv(output / "test_assignment.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "norm_audit.csv", index=False)
    artifacts = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "sealed_manifest.json", dict(contract, inputs=inputs, artifacts=artifacts, folds=folds,
        unique_test_trajectories=len(frame), test_appearances=int(tested.sum())))
    print("SEALED ALL 32000 UNIQUE TRAJECTORIES", flush=True)


def direct_scores(current, bank, offset):
    euclidean = np.linalg.norm(bank - current, axis=1)
    cosine = np.clip(1 - (bank @ current) / (np.linalg.norm(bank, axis=1) * np.linalg.norm(current)), 0, 2)
    eids, cids = np.argsort(euclidean)[:20], np.argsort(cosine)[:20]
    raw, ref = current + offset, bank + offset
    absolute = np.clip(1 - (ref @ raw) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(raw)), 0, 2)
    return np.asarray([euclidean[eids].mean(), cosine[cids].mean(), euclidean[cids].mean(),
        cosine[eids].mean(), np.sort(absolute)[:20].mean(), np.linalg.norm(current)], np.float32)


def verify_all(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    hash_checks = 0
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
            hash_checks += 1
    frame, cache = pd.read_csv(output / "index.csv"), load_npz(CACHE)
    tested = np.zeros(len(frame), np.int16)
    expected_splits = {info["fold"]: (info, ref, cal, test) for si, (_, part) in enumerate(frame.groupby("suite", sort=True))
                       for info, ref, cal, test in split_suite(part, si)}
    calibration_checks, distance_checks, all_queries, all_decisions = 0, 0, 0, 0
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        profile = load_npz(output / "profiles" / f"{info['fold']}.npz")
        np.testing.assert_array_equal(data["methods"], METHODS)
        reference, calibration, test = (data[key] for key in ("reference_rows", "calibration_rows", "test_rows"))
        validate_split(frame, reference, calibration, test, info)
        expected_info, *expected_rows = expected_splits[info["fold"]]
        for actual, expected in zip((reference, calibration, test), expected_rows):
            np.testing.assert_array_equal(actual, expected)
        for key, expected in expected_info.items():
            assert info[key] == expected, key
        labels = historical_labels(frame, np.r_[reference, calibration], {})
        np.testing.assert_array_equal(data["reference_labels"], labels[reference])
        np.testing.assert_array_equal(data["calibration_labels"], labels[calibration])
        assert (labels[test] == -1).all()
        rebuilt = build_profile(cache, frame, reference, labels, info["bank_seed"])
        for key, expected in rebuilt.items():
            np.testing.assert_array_equal(profile[key], expected)
        assert set(profile["success_global_rows"]).issubset(set(reference))
        assert (labels[profile["success_global_rows"]] == 0).all()
        pairs = np.c_[profile["success_global_rows"], profile["success_queries"]]
        assert len(np.unique(pairs, axis=0)) == len(pairs) <= 4096
        assert pd.Series(pairs[:, 0]).value_counts().max() <= 8
        needed = np.r_[calibration, test]
        x = normalized_vectors(cache, needed, profile)
        scores = np.concatenate((data["calibration_scores"], data["scores"]), axis=1)
        valid = np.isfinite(x).all(-1)
        np.testing.assert_array_equal(valid, cache["valid"][needed] & (np.arange(52)[None] >= 7))
        for score in scores:
            np.testing.assert_array_equal(np.isfinite(score), valid)
        assert np.isnan(scores[:, :, :7]).all()
        assert (scores[2, valid] >= scores[0, valid] - 1e-6).all()
        assert (scores[3, valid] >= scores[1, valid] - 1e-6).all()
        success = data["calibration_labels"] == 0
        cal_frame = frame.loc[calibration].reset_index(drop=True)
        groups = (cal_frame.loc[success, "task"] + "|" + cal_frame.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi in range(len(METHODS)):
            raw_cal = data["calibration_scores"][mi, success]
            peaks = np.max(np.where(np.isfinite(raw_cal), raw_cal, -np.inf), axis=1)
            grouped = np.asarray([peaks[groups == group].max() for group in sorted(set(groups))])
            for ki, units in enumerate((peaks, grouped)):
                rank = int(np.ceil((len(units) + 1) * .95))
                threshold = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                assert threshold == data["thresholds"][ki, mi] == profile["thresholds"][ki, mi]
                crossed = np.isfinite(data["scores"][mi]) & (data["scores"][mi] > threshold)
                first = np.where(crossed.any(1), crossed.argmax(1), -1)
                np.testing.assert_array_equal(first, data["first"][ki, mi])
                assert ((first == -1) | ((first >= 7) & (first < frame.loc[test, "length"].to_numpy()))).all()
                assert int((units > threshold).sum()) <= len(units) + 1 - rank
                calibration_checks += 1
                all_decisions += len(test)
        candidates = np.argwhere(valid)
        rng = np.random.default_rng(info["bank_seed"])
        selected = candidates[rng.choice(len(candidates), 12, replace=False)].tolist()
        selected += [list(np.unravel_index(np.nanargmax(scores[mi]), valid.shape)) for mi in (0, 1, 5)]
        for position, query in selected:
            expected = direct_scores(x[position, query], profile["success_dynamic"], profile["dynamic_center"] / profile["dynamic_scale"])
            np.testing.assert_allclose(scores[:, position, query], expected, rtol=3e-6, atol=3e-7)
            distance_checks += 1
        # A truncated trajectory must reproduce the prefix already scored from its full recording.
        row = int(test[np.argmax(frame.loc[test, "length"].to_numpy())])
        for stop in (8, 15):
            prefix = {key: cache[key][row:row + 1, :stop] for key in cache}
            small = normalized_vectors(prefix, np.asarray([0]), profile)
            full_position = int(np.flatnonzero(needed == row)[0])
            np.testing.assert_array_equal(small[0], x[full_position, :stop])
        tested[test] += 1
        all_queries += int(valid.sum())
        print(f"VERIFIED {info['fold']}", flush=True)
    np.testing.assert_array_equal(tested, 1)
    assignment = pd.read_csv(output / "test_assignment.csv")
    np.testing.assert_array_equal(assignment.global_row, np.arange(len(frame)))
    for info in manifest["folds"]:
        rows = assignment.loc[assignment.fold.eq(info["fold"]), "global_row"].to_numpy()
        np.testing.assert_array_equal(rows, expected_splits[info["fold"]][3])
    result = dict(passed=True, hash_checks=hash_checks, unique_test_trajectories=len(frame), folds=len(manifest["folds"]),
        task_isolation_checks=len(manifest["folds"]), reference_profile_rebuilds=len(manifest["folds"]),
        independently_recomputed_thresholds=calibration_checks, independently_recomputed_trajectory_decisions=all_decisions,
        independently_recomputed_distance_points=distance_checks, methods_per_distance_point=len(METHODS),
        cal_and_test_valid_queries=all_queries, causal_prefix_checks=2 * len(manifest["folds"]),
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
