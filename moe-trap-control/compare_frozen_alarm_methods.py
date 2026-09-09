#!/usr/bin/env python3
"""Replay one frozen configuration across the complete HUB corpus, on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from compare_alarm_methods import ROOT, HERE, SAFE, RESULTS, audit, guard, metrics

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from knn import ReferenceScorer, dynamics, reference_pairs, reference_scaling
from cosine_knn import CosineReference, normalized_vectors
from kmeans_reference import fit_clusters, score_clusters

SEED = 20260908
ALPHA = 0.05
LEGACY = ("v7_frozen", "v8_frozen", "v82_frozen")
GEOMETRY = ("knn20", "c4_assigned_radius", "c32_assigned_radius",
            "c32_union_radius", "norm_only", "cosine_knn20")
METHODS = LEGACY + GEOMETRY
CHECKPOINTS = (7, 10, 13, 20, 51)


def score_geometry(x, reference, clusters):
    finite = np.isfinite(x).all(-1)
    scores = np.full((len(GEOMETRY), *finite.shape), np.nan, np.float32)
    if finite.any():
        distance, _ = ReferenceScorer(reference).neighbors("success_dynamic", x[finite])
        scores[0, finite] = distance.mean(axis=1)
        scores[4, finite] = np.linalg.norm(x[finite], axis=1)
        scores[5] = CosineReference(reference["success_dynamic"]).score(x)
    c4, _ = score_clusters(x, clusters[4])
    c32, _ = score_clusters(x, clusters[32])
    scores[1], scores[2], scores[3] = c4[1], c32[1], c32[2]
    return scores


def freeze_profiles(directory, frame, cache):
    directory.mkdir()
    ids = np.asarray(sorted(frame.init_state_id.unique()))
    assert len(ids) == 50
    ordered = np.random.default_rng(SEED).permutation(ids)
    a = frame.run_id.eq(audit.RUN_A).to_numpy()
    reference = np.flatnonzero(a & frame.init_state_id.isin(ordered[:30]).to_numpy())
    calibration = np.flatnonzero(a & frame.init_state_id.isin(ordered[30:40]).to_numpy())
    holdout = np.flatnonzero(a & frame.init_state_id.isin(ordered[40:]).to_numpy())
    assert (len(reference), len(calibration), len(holdout)) == (9600, 3200, 3200)
    assert not set(reference) & set(calibration)
    period = cache["periodicity"][reference]
    pscale = float(np.quantile(np.abs(period[np.isfinite(period)]), .75))
    dynamic, _ = dynamics(cache["mobility"][reference], cache["acceleration"][reference], period, pscale)
    dynamic[~cache["valid"][reference]] = np.nan
    normalized, scaling = reference_scaling(dynamic, np.arange(len(reference)))
    success = ~frame.iloc[reference].failure.to_numpy(bool)
    row, query = reference_pairs(dynamic, np.flatnonzero(success), cap=4096, seed=SEED).T
    profile = dict(dynamic_center=np.asarray(scaling["center"]), dynamic_scale=np.asarray(scaling["scale"]),
        scaling_points=np.asarray(scaling["points"]), periodicity_scale=np.asarray(pscale),
        success_dynamic=normalized[row, query].astype(np.float64), success_global_rows=reference[row],
        success_queries=query, checkpoint_set=np.asarray(sorted(frame.checkpoint.unique())),
        reference_rows=reference, calibration_rows=calibration, a_geometry_holdout_rows=holdout)
    np.savez_compressed(directory / "global_reference.npz", **profile)
    clusters = {count: fit_clusters(profile["success_dynamic"], count, SEED + 10000 + count)
                for count in (4, 32)}
    for count, model in clusters.items():
        np.savez_compressed(directory / f"c{count}.npz", **model)
    print("REFERENCE fixed: 9600 A episodes, 4096 successful points, one bank for all suites", flush=True)

    cal_scores = score_geometry(normalized_vectors(cache, calibration, profile), profile, clusters)
    cal_frame = frame.iloc[calibration]
    successful = ~cal_frame.failure.to_numpy(bool)
    group_keys = (cal_frame.task + "|" + cal_frame.init_state_id.astype(str)).to_numpy(dtype=str)
    group_ids, names = pd.factorize(group_keys[successful], sort=True)
    thresholds = {}
    for mi, method in enumerate(GEOMETRY):
        peaks = np.where(np.isfinite(cal_scores[mi, successful]), cal_scores[mi, successful], -np.inf).max(axis=1)
        grouped = np.full(len(names), -np.inf)
        np.maximum.at(grouped, group_ids, peaks)
        tau, rank = guard.conformal_threshold(grouped, ALPHA)
        assert np.isfinite(tau)
        thresholds[method] = dict(threshold=float(tau), rank=rank, task_init_groups=len(names),
            successful_episodes=int(successful.sum()), group_exceedances=int((grouped > tau).sum()))
    np.savez_compressed(directory / "calibration.npz", rows=calibration, scores=cal_scores,
        groups=group_keys, failure=~successful, methods=np.asarray(GEOMETRY))

    v7path = ROOT / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
    v8path = ROOT / "moe-v8-0906/results/v8_full_corpus_summary.json"
    v82path = ROOT / "moe-v8-0906/results/v82_summary.json"
    v7 = audit.archive(v7path)
    v8, v82 = (json.loads(p.read_text()) for p in (v8path, v82path))
    assert v8["thresholds"] == v82["thresholds"]
    assert v82["config"] == dict(baseline=4, width=6, confirm=2, slope=-.0015)
    legacy = dict(v7={key: float(v7[key]) for key in
        ("periodicity_scale", "freeze_threshold", "acceleration_threshold", "periodicity_threshold")},
        v8_thresholds=v8["thresholds"], v8_confirm=v8["confirm"], v82_config=v82["config"],
        earliest_v8_head_query=6, v7_boolean="freeze OR (latched_acceleration AND latched_periodicity)")
    sources = [Path(__file__), HERE / "compare_alarm_methods.py", v7path, v8path, v82path,
        ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py", SAFE / "v82_validation/monitor.py",
        SAFE / "v82_validation/run_analysis.py", SAFE / "feature_geometry/analyze.py",
        SAFE / "boundary_knn/knn.py", SAFE / "boundary_knn/cosine_knn.py",
        SAFE / "boundary_knn/kmeans_reference.py"]
    parameters = dict(schema=1, methods=METHODS, parameter_scope="one global profile per method",
        task_or_suite_specific_parameters=False, folds=0, seed=SEED, legacy=legacy,
        geometry=dict(thresholds=thresholds, alpha=ALPHA, calibration="successful task/init maxima",
            knn_neighbors=20, bank_cap=4096, dimensions=10, first_query=7, confirm=1,
            clusters=[4, 32], radius_quantile=.9, minimum_cluster_points=20,
            periodicity_scale=pscale, dynamic_center=scaling["center"], dynamic_scale=scaling["scale"],
            reference_initial_ids=ordered[:30], calibration_initial_ids=ordered[30:40],
            a_holdout_initial_ids=ordered[40:], reference_episodes=len(reference),
            reference_successes=int(success.sum()), calibration_episodes=len(calibration),
            scope="pooled 10D route dynamics across four checkpoints; no raw expert-ID pooling"),
        sources={str(p.relative_to(ROOT)): audit.digest(p) for p in sources},
        input_verification_sha256=audit.digest(directory.parent / "input_verification.json"),
        new_blind_test=False, b_used_for_fitting=False, v7_v8_refitted=False)
    audit.write_json(directory / "parameters.json", parameters)
    files = {p.name: audit.digest(p) for p in directory.iterdir() if p.is_file()}
    audit.write_json(directory / "manifest.json", dict(files=files, frozen_before_full_corpus_scoring=True))
    print("PARAMETERS FROZEN: original v7/v8/v8.2 plus six global geometry thresholds", flush=True)


def load_profiles(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert audit.digest(directory / name) == expected, name
    parameters = json.loads((directory / "parameters.json").read_text())
    for name, expected in parameters["sources"].items():
        assert audit.digest(ROOT / name) == expected, name
    assert audit.digest(output / "input_verification.json") == parameters["input_verification_sha256"]
    assert parameters["folds"] == 0 and not parameters["task_or_suite_specific_parameters"]
    assert tuple(parameters["methods"]) == METHODS
    reference = audit.archive(directory / "global_reference.npz")
    clusters = {count: audit.archive(directory / f"c{count}.npz") for count in (4, 32)}
    return parameters, reference, clusters, manifest


def replay_legacy(cache, raw, config):
    valid, p = cache["valid"], config["v7"]
    streams = guard.intrinsic_score_arrays(cache["mobility"], cache["acceleration"],
                                           cache["periodicity"], p["periodicity_scale"])
    freeze = guard.first_alarm(streams["freeze"], valid, p["freeze_threshold"])
    acceleration = guard.first_alarm(streams["acceleration_persistent"], valid, p["acceleration_threshold"])
    periodicity = guard.first_alarm(streams["periodicity_persistent"], valid, p["periodicity_threshold"])
    v7 = guard.union(freeze, guard.first_and(acceleration, periodicity))
    active = audit.historical_series(raw, valid)
    active[:, :config["earliest_v8_head_query"]] = np.nan
    thresholds = (-config["v8_thresholds"]["frontback_flowpath"], config["v8_thresholds"]["curvature_3step"])
    versions = [v7]
    for slope in (0.0, config["v82_config"]["slope"]):
        heads = [guard.first_alarm(guard.persistent(active[:, :, i] - slope * np.arange(valid.shape[1])[None],
                 config["v8_confirm"]), valid, threshold, inclusive=True) for i, threshold in enumerate(thresholds)]
        versions.append(guard.union(v7, *heads))
    return np.asarray(versions)


def verify(output, directory, frame, cache, raw, parameters, reference, clusters, scores, first, manifest):
    valid = cache["valid"]
    for row in first:
        assert ((row == -1) | ((row >= 0) & (row < frame.length.to_numpy()))).all()
    geometry_valid = valid & (np.arange(valid.shape[1])[None] >= 7)
    for mi, method in enumerate(GEOMETRY):
        np.testing.assert_array_equal(np.isfinite(scores[mi]), geometry_valid)
        tau = parameters["geometry"]["thresholds"][method]["threshold"]
        crossing = geometry_valid & np.isfinite(scores[mi]) & (scores[mi] > tau)
        expected = np.where(crossing.any(axis=1), crossing.argmax(axis=1), -1)
        np.testing.assert_array_equal(expected, first[len(LEGACY) + mi])

    anchor_path = RESULTS / "v82_validation_20260908/frozen_first_alarms.csv"
    anchor = pd.read_csv(anchor_path)
    for key in ("source", "episode", "run_id", "task", "length", "failure"):
        np.testing.assert_array_equal(anchor[key], frame[key])
    for mi, method in enumerate(LEGACY):
        np.testing.assert_array_equal(first[mi], anchor[method])

    cal = audit.archive(directory / "calibration.npz")
    successful = ~cal["failure"]
    for mi, method in enumerate(GEOMETRY):
        np.testing.assert_array_equal(scores[mi, cal["rows"]], cal["scores"][mi])
        peaks = np.where(np.isfinite(cal["scores"][mi, successful]), cal["scores"][mi, successful], -np.inf).max(axis=1)
        names = cal["groups"][successful]
        units = np.asarray([peaks[names == key].max() for key in sorted(set(names))])
        rank = int(np.ceil((len(units) + 1) * (1 - parameters["geometry"]["alpha"])))
        assert np.sort(units)[rank - 1] == parameters["geometry"]["thresholds"][method]["threshold"]
    ref, cal_rows = reference["reference_rows"], reference["calibration_rows"]
    assert frame.iloc[np.r_[ref, cal_rows]].run_id.eq(audit.RUN_A).all()
    assert set(zip(frame.iloc[ref].task, frame.iloc[ref].init_state_id)).isdisjoint(
        set(zip(frame.iloc[cal_rows].task, frame.iloc[cal_rows].init_state_id)))
    assert set(reference["success_global_rows"]).issubset(set(ref))
    assert not frame.iloc[reference["success_global_rows"]].failure.any()

    selected = np.linspace(0, len(frame) - 1, 32, dtype=int)[::-1]
    x = normalized_vectors(cache, selected, reference)
    np.testing.assert_array_equal(score_geometry(x, reference, clusters), scores[:, selected])
    bank = reference["success_dynamic"]
    pairs = np.argwhere(np.isfinite(x).all(-1))
    for position, query in pairs[np.linspace(0, len(pairs) - 1, 32, dtype=int)]:
        value = x[position, query]
        distance = np.linalg.norm(bank - value, axis=1)
        cosine = np.clip(1 - (bank @ value) / (np.linalg.norm(bank, axis=1) * np.linalg.norm(value)), 0, 2)
        expected = [np.sort(distance)[:20].mean()]
        for count in (4, 32):
            model = clusters[count]
            delta = np.linalg.norm(model["centers"] - value, axis=1)
            expected.append(delta[delta.argmin()] / model["radii"][delta.argmin()])
        expected.extend(((np.linalg.norm(clusters[32]["centers"] - value, axis=1) / clusters[32]["radii"]).min(),
                         np.linalg.norm(value), np.sort(cosine)[:20].mean()))
        np.testing.assert_allclose(scores[:, selected[position], query], expected, rtol=2e-6, atol=1e-7)

    prefix = {key: cache[key][selected].copy() for key in ("mobility", "acceleration", "periodicity", "valid")}
    for key, value in prefix.items():
        value[:, 14:] = False if key == "valid" else np.nan
    prefix_raw = raw[selected].copy()
    prefix_raw[:, 14:] = np.nan
    prefix_first = replay_legacy(prefix, prefix_raw, parameters["legacy"])
    np.testing.assert_array_equal(prefix_first, np.where(first[:3, selected] <= 13, first[:3, selected], -1))
    prefix_x = normalized_vectors(prefix, np.arange(len(selected)), reference)
    np.testing.assert_array_equal(prefix_x[:, :14], x[:, :14])
    for name, expected in manifest["files"].items():
        assert audit.digest(directory / name) == expected
    return dict(passed=True, trajectory_decisions=int(first.size), exact_legacy_replays=3 * len(frame),
        independent_distance_points=32, prefix_and_batch_order_episodes=32, invalid_padding_alarms=0,
        calibration_thresholds_verified=len(GEOMETRY), profiles_unchanged_during_evaluation=True,
        b_rows_in_reference_or_calibration=0, historical_anchor_sha256=audit.digest(anchor_path))


def export_metrics(output, frame, first, reference):
    rows, suites = [], []
    cohorts = (("all", frame), ("A", frame.loc[frame.run_id.eq(audit.RUN_A)]),
        ("B", frame.loc[frame.run_id.eq(audit.RUN_B)]),
        ("A_geometry_holdout", frame.iloc[reference["a_geometry_holdout_rows"]]))
    for mi, method in enumerate(METHODS):
        for cohort, part in cohorts:
            for cutoff in CHECKPOINTS:
                rows.append(dict(method=method, cohort=cohort, query_cutoff=cutoff,
                                 **metrics(first[mi, part.index], part, cutoff)))
            for suite, subset in part.groupby("suite", sort=True):
                suites.append(dict(method=method, cohort=cohort, suite=suite,
                                    **metrics(first[mi, subset.index], subset)))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(suites).to_csv(output / "suite_metrics.csv", index=False)
    alarms = frame[["global_row", "source", "episode", "run_id", "suite", "task", "init_state_id", "length", "failure"]].copy()
    for mi, method in enumerate(METHODS):
        alarms[method] = first[mi]
    alarms.to_csv(output / "first_alarms.csv", index=False)
    budget = []
    core = ("v7_frozen", "v82_frozen", "knn20", "c4_assigned_radius")
    for cohort, part in cohorts[:3]:
        chosen = first[[METHODS.index(method) for method in core]][:, part.index]
        states = [len(set(values[values >= 0].tolist())) for values in chosen.T]
        budget.append(dict(cohort=cohort, methods=core, main_rows=len(part),
            mains_with_any_alarm=int(np.count_nonzero(states)), unique_first_alarm_states=int(np.sum(states)),
            nine_suffixes_for_every_state=int(9 * np.sum(states))))
    audit.write_json(output / "alarm_union_budget.json", budget)


def run(output, profiles=None):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    audit.write_json(output / "contract.json", dict(methods=METHODS, folds=0,
        parameter_scope="one global configuration per method across all 32000 trajectories",
        original_v7_v8_unchanged=True, geometry_global_profile_created_once=profiles is None,
        b_used_for_fitting=False, new_blind_test=False, a_and_all_include_reference_overlap=True,
        primary_evaluation="B: 16000 trajectories; new noise repetitions of known tasks and initial states",
        alpha_is_not_a_guarantee_of_test_fpr=True, c4_previously_explored=True,
        hidden_capture=False, model_queries=0, gpu_compute=False, pro_plus_results=False,
        fpr="successful trajectories with any valid alarm / all successful trajectories",
        false_alarms_removed_by_lead_cutoff=False))
    frame, cache, raw, _ = audit.load_inputs(output)
    if profiles is None:
        profiles = output / "profiles"
        freeze_profiles(profiles, frame, cache)
    parameters, reference, clusters, manifest = load_profiles(profiles, output)
    first = np.full((len(METHODS), len(frame)), -1, np.int16)
    first[:3] = replay_legacy(cache, raw, parameters["legacy"])
    print("ORIGINAL GUARDS replayed with unchanged historical parameters", flush=True)
    scores = np.full((len(GEOMETRY), len(frame), cache["valid"].shape[1]), np.nan, np.float32)
    for start in range(0, len(frame), 2000):
        stop = min(start + 2000, len(frame))
        rows = np.arange(start, stop)
        x = normalized_vectors(cache, rows, reference)
        scores[:, rows] = score_geometry(x, reference, clusters)
        for mi, method in enumerate(GEOMETRY):
            tau = parameters["geometry"]["thresholds"][method]["threshold"]
            first[len(LEGACY) + mi, rows] = guard.first_alarm(scores[mi, rows], cache["valid"][rows], tau)
        print(f"FIXED REPLAY {stop}/{len(frame)} episodes, {time.perf_counter() - started:.1f}s", flush=True)
    verification = verify(output, profiles, frame, cache, raw, parameters, reference, clusters, scores, first, manifest)
    export_metrics(output, frame, first, reference)
    np.savez_compressed(output / "scores_and_alarms.npz", geometry_methods=np.asarray(GEOMETRY),
        scores=scores, methods=np.asarray(METHODS), first=first, valid=cache["valid"],
        thresholds=np.asarray([parameters["geometry"]["thresholds"][m]["threshold"] for m in GEOMETRY]))
    verification.update(elapsed_seconds=time.perf_counter() - started, profiles=str(profiles.resolve()),
        parameters_sha256=audit.digest(profiles / "parameters.json"),
        profile_manifest_sha256=audit.digest(profiles / "manifest.json"),
        artifacts={str(p.relative_to(output)): audit.digest(p) for p in output.rglob("*") if p.is_file()})
    audit.write_json(output / "verification.json", verification)
    print("VERIFIED: 288000 decisions, 96000 exact historical matches; parameters stayed fixed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "design/frozen_alarm_comparison_20260908")
    parser.add_argument("--profiles", type=Path, help="Reuse frozen profiles with no fitting or calibration")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.profiles)
