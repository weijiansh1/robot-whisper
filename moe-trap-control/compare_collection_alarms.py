#!/usr/bin/env python3
"""Replay frozen route alarms on audited, unintervened collection mains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from compare_frozen_alarm_methods import (
    HERE, SAFE, GEOMETRY, LEGACY, METHODS, audit, guard, load_profiles,
    metrics, normalized_vectors, replay_legacy, score_geometry,
)

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, atomic_npz, digest, records
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

sys.path.insert(0, str(SAFE / "temporal_fusion"))
from fusion import flow_speed, raw_v8_features

COMBINATIONS = ("knn_euclidean_OR_cosine", "knn_euclidean_AND_cosine_latched")
ALL_METHODS = METHODS + COMBINATIONS


def score_all(cache, raw, parameters, reference, clusters):
    vectors = normalized_vectors(cache, np.arange(len(raw)), reference)
    scores = score_geometry(vectors, reference, clusters)
    legacy = replay_legacy(cache, raw, parameters["legacy"])
    geometry = np.asarray([
        guard.first_alarm(scores[i], cache["valid"],
                          parameters["geometry"]["thresholds"][name]["threshold"])
        for i, name in enumerate(GEOMETRY)
    ])
    first = np.concatenate((legacy, geometry))
    euclidean, cosine = (first[METHODS.index(name)] for name in ("knn20", "cosine_knn20"))
    first = np.concatenate((first, [guard.union(euclidean, cosine), guard.first_and(euclidean, cosine)]))
    return vectors, scores, first


def scalar_v8_first(raw, v7_first, config, slope):
    """Check smoothing, confirmation and inclusivity without vectorized alarm helpers."""
    baseline = raw[1:5, 1].mean(dtype=np.float32)
    relative = np.log(np.maximum(raw[:, 1], 1e-12) / np.maximum(baseline, 1e-12))
    thresholds = (-config["v8_thresholds"]["frontback_flowpath"],
                  config["v8_thresholds"]["curvature_3step"])
    counts = [0, 0]
    first = v7_first
    for q in range(config["earliest_v8_head_query"], len(raw)):
        values = (raw[q - 5:q + 1, 0].mean(dtype=np.float64),
                  relative[q - 5:q + 1].mean(dtype=np.float64))
        for head, value in enumerate(values):
            counts[head] = counts[head] + 1 if value - slope * q >= thresholds[head] else 0
            if counts[head] >= config["v8_confirm"] and (first < 0 or q < first):
                first = q
    return first


def verify_scores(cache, raw, parameters, reference, clusters, vectors, scores, first):
    valid = cache["valid"]
    expected_valid = valid & (np.arange(valid.shape[1])[None] >= parameters["geometry"]["first_query"])
    np.testing.assert_array_equal(np.isfinite(vectors).all(-1), expected_valid)
    for i, name in enumerate(GEOMETRY):
        np.testing.assert_array_equal(np.isfinite(scores[i]), expected_valid)
        threshold = parameters["geometry"]["thresholds"][name]["threshold"]
        crossings = np.isfinite(scores[i]) & (scores[i] > threshold) & valid
        expected = np.where(crossings.any(1), crossings.argmax(1), -1)
        np.testing.assert_array_equal(first[len(LEGACY) + i], expected)
    for row, length in enumerate(valid.sum(1)):
        for i, slope in ((1, 0.0), (2, parameters["legacy"]["v82_config"]["slope"])):
            expected = scalar_v8_first(raw[row, :length], int(first[0, row]), parameters["legacy"], slope)
            if expected != first[i, row]:
                raise ValueError("Independent v8 confirmation replay differs")
    bank = reference["success_dynamic"]
    points = np.argwhere(expected_valid)
    selected = points[np.linspace(0, len(points) - 1, min(32, len(points)), dtype=int)]
    for row, query in selected:
        value = vectors[row, query]
        euclidean = np.linalg.norm(bank - value, axis=1)
        cosine = np.clip(1 - (bank @ value) /
                         (np.linalg.norm(bank, axis=1) * np.linalg.norm(value)), 0, 2)
        expected = [np.sort(euclidean)[:20].mean()]
        for count in (4, 32):
            model = clusters[count]
            distances = np.linalg.norm(model["centers"] - value, axis=1)
            nearest = distances.argmin()
            expected.append(distances[nearest] / model["radii"][nearest])
        expected.extend(((np.linalg.norm(clusters[32]["centers"] - value, axis=1) /
                          clusters[32]["radii"]).min(), np.linalg.norm(value), np.sort(cosine)[:20].mean()))
        np.testing.assert_allclose(scores[:, row, query], expected, rtol=2e-6, atol=1e-7)

    # Masking future queries checks that early decisions do not use the eventual outcome or suffix.
    selected_rows = np.linspace(0, len(raw) - 1, min(8, len(raw)), dtype=int)[::-1]
    _, reversed_scores, reversed_first = score_all(
        {key: value[selected_rows] for key, value in cache.items()}, raw[selected_rows],
        parameters, reference, clusters)
    np.testing.assert_array_equal(reversed_scores, scores[:, selected_rows])
    np.testing.assert_array_equal(reversed_first, first[:, selected_rows])
    for cutoff in (7, 13, 20):
        prefix = {key: value[selected_rows].copy() for key, value in cache.items()}
        prefix_raw = raw[selected_rows].copy()
        for key, value in prefix.items():
            value[:, cutoff + 1:] = False if key == "valid" else np.nan
        prefix_raw[:, cutoff + 1:] = np.nan
        _, prefix_scores, prefix_first = score_all(prefix, prefix_raw, parameters, reference, clusters)
        np.testing.assert_array_equal(prefix_scores[:, :, :cutoff + 1], scores[:, selected_rows, :cutoff + 1])
        np.testing.assert_array_equal(prefix_first, np.where(first[:, selected_rows] <= cutoff,
                                                            first[:, selected_rows], -1))
    lengths = valid.sum(1)
    if not np.all((first == -1) | ((first >= 0) & (first < lengths[None]))):
        raise ValueError("Alarm outside an observed main")
    return dict(independent_distance_points=len(selected), scalar_v8_replays=2 * len(raw),
                prefix_cutoffs=[7, 13, 20], prefix_and_order_mains=len(selected_rows))


def run(args):
    started = time.perf_counter()
    parameters, reference, clusters, manifest = load_profiles(args.source / "profiles", args.source)
    audited = json.loads(args.audit.read_text())
    if audited["status"] != "passed" or not audited["formal_collection"]:
        raise ValueError("A passed formal collection audit is required")
    tasks = audited["tasks"]
    if len(tasks) != audited["planned_mains"] or len({row["main_id"] for row in tasks}) != len(tasks):
        raise ValueError("Missing or duplicated formal mains")
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = (Path(__file__), HERE / "collection_storage.py", SAFE / "temporal_fusion/fusion.py")
    source_hashes = {str(path): digest(path) for path in source_paths}
    contract = dict(
        collection_run=audited["run"], collection_audit=str(args.audit.resolve()),
        collection_audit_sha256=digest(args.audit), frozen_source=str(args.source.resolve()),
        parameters_sha256=digest(args.source / "profiles/parameters.json"),
        profile_manifest_sha256=digest(args.source / "profiles/manifest.json"), sources=source_hashes,
        methods=ALL_METHODS, actual_online_trigger="v7_frozen", evaluation="offline_native_main_replay",
        ground_truth="eventual native main success, one observation per main_id",
        exclude_from_primary="unchanged_scene_control", reference_fit=False, threshold_fit=False,
        suite_specific_parameters=False, hidden_capture=False, model_queries=0, gpu_compute=False,
        k=20, reference_points=len(reference["success_dynamic"]),
        combinations="OR at either first alarm; latched AND at the later first alarm",
        combination_false_positive_budget_recalibrated=False,
        intervention_outcomes_for_new_alarm_states_available=False,
    )
    atomic_json(args.output / "contract.json", contract)
    length = max(task["main_queries"] for task in tasks)
    cache = dict(valid=np.zeros((len(tasks), length), bool),
                 mobility=np.full((len(tasks), length, 8), np.nan, np.float32),
                 acceleration=np.full((len(tasks), length), np.nan, np.float32),
                 periodicity=np.full((len(tasks), length), np.nan, np.float32))
    raw = np.full((len(tasks), length, 2), np.nan, np.float32)
    frame_rows, inputs = [], []
    profile = GlobalIntrinsicProfile(**parameters["legacy"]["v7"])
    checked_queries = 0
    for row, task in enumerate(tasks):
        directory = Path(task["directory"])
        result = json.loads((directory / "result.json").read_text())
        commit = json.loads((directory / "main_complete.json").read_text())
        if result["status"] != "completed" or not result["main_complete"]:
            raise ValueError("Main is not committed")
        for key, expected in (("main_id", task["main_id"]), ("queries", task["main_queries"]),
                              ("success", task["success"]), ("action_steps", task["action_steps"])):
            if result[key] != expected or commit[key] != expected:
                raise ValueError("Main identity/outcome differs from audit: " + key)
        if commit["manifest_sha256"] != digest(directory / "main/manifest.json"):
            raise ValueError("Main manifest differs from commit")
        monitor = IntrinsicGuardMonitor(profile)
        count = 0
        final_success = False
        for record in records(directory / "main"):
            query = int(record["query"])
            if query != count or final_success:
                raise ValueError("Noncontiguous query or post-success inference")
            alarm = monitor.update(record[PROBS_KEY])
            if bool(record["alarm"]) != alarm["alarm"]:
                raise ValueError("Online v7 alarm differs from replay")
            names = ("freeze_score", "acceleration_score", "periodicity_score")
            np.testing.assert_array_equal(np.asarray([alarm[name] for name in names], np.float32),
                                          record["alarm_scores"])
            cache["valid"][row, query] = True
            cache["mobility"][row, query] = alarm["layer_mobility"]
            cache["acceleration"][row, query] = alarm["route_acceleration"]
            cache["periodicity"][row, query] = alarm["lag_periodicity"]
            raw[row, query] = raw_v8_features(flow_speed(record[PROBS_KEY]))
            final_success = bool(record["success"])
            count += 1
        first = -1 if task["first_alarm_query"] is None else task["first_alarm_query"]
        if count != task["main_queries"] or final_success != task["success"] or monitor.first_alarm_query != first:
            raise ValueError("Main length, success or first alarm differs from audit")
        checked_queries += count
        frame_rows.append({key: task[key] for key in ("main_id", "benchmark", "category", "task_name", "analysis_role")})
        frame_rows[-1].update(length=count, failure=not final_success, online_v7_first=first)
        inputs.append(dict(main_id=task["main_id"], directory=str(directory),
            result_sha256=digest(directory / "result.json"), commit_sha256=digest(directory / "main_complete.json"),
            main_manifest_sha256=commit["manifest_sha256"]))
        if (row + 1) % 20 == 0:
            print(f"EXTRACTED {row + 1}/{len(tasks)} mains, {checked_queries} queries", flush=True)
    if checked_queries != audited["main_queries"]:
        raise ValueError("Audited query count differs")
    frame = pd.DataFrame(frame_rows)
    vectors, scores, first = score_all(cache, raw, parameters, reference, clusters)
    np.testing.assert_array_equal(first[0], frame.online_v7_first.to_numpy())
    checks = verify_scores(cache, raw, parameters, reference, clusters, vectors, scores, first)
    cohorts = dict(all=frame, primary=frame.loc[frame.analysis_role.eq("perturbation")],
                   unchanged_scene_controls=frame.loc[~frame.analysis_role.eq("perturbation")])
    for benchmark in sorted(frame.benchmark.unique()):
        cohorts["all_" + benchmark] = frame.loc[frame.benchmark.eq(benchmark)]
        cohorts["primary_" + benchmark] = cohorts["primary"].loc[cohorts["primary"].benchmark.eq(benchmark)]
    metric_rows, differences = [], []
    for cohort, part in cohorts.items():
        failure = part.failure.to_numpy(bool)
        baseline = first[0, part.index]
        for i, name in enumerate(ALL_METHODS):
            alarm = first[i, part.index]
            metric_rows.append(dict(cohort=cohort, method=name, **metrics(alarm, part, cutoff=length - 1)))
            extra, lost = (alarm >= 0) & (baseline < 0), (alarm < 0) & (baseline >= 0)
            differences.append(dict(cohort=cohort, method=name,
                extra_tp_vs_v7=int((extra & failure).sum()), extra_fp_vs_v7=int((extra & ~failure).sum()),
                lost_tp_vs_v7=int((lost & failure).sum()), lost_fp_vs_v7=int((lost & ~failure).sum()),
                earlier_tp_vs_v7=int(((alarm >= 0) & (baseline >= 0) & (alarm < baseline) & failure).sum())))
    first_frame = frame.copy()
    for i, name in enumerate(ALL_METHODS):
        first_frame[name] = first[i]
    first_frame.to_csv(args.output / "first_alarms.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(args.output / "differences_vs_v7.csv", index=False)
    atomic_npz(args.output / "scores_and_alarms.npz", dict(
        methods=np.asarray(ALL_METHODS), first=first, geometry_methods=np.asarray(GEOMETRY),
        geometry_scores=scores, raw_v8=raw, normalized_dynamics=vectors, **cache))
    atomic_json(args.output / "input_verification.json", dict(mains=inputs, checked_queries=checked_queries))
    atomic_json(args.output / "summary.json", dict(
        contract=contract, metrics=metric_rows, differences_vs_v7=differences,
        caveat="Descriptive Long batch, not full benchmark scores; new offline alarm states have no paired intervention results"))
    load_profiles(args.source / "profiles", args.source)
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError("Replay source changed while scoring")
    if digest(args.audit) != contract["collection_audit_sha256"]:
        raise ValueError("Collection audit changed while scoring")
    artifacts = {path.name: digest(path) for path in args.output.iterdir() if path.is_file()}
    atomic_json(args.output / "verification.json", dict(
        status="passed", mains=len(tasks), raw_queries=checked_queries,
        exact_online_v7_queries=checked_queries, exact_online_v7_first_alarms=len(tasks),
        trajectory_decisions=int(first.size), profiles_unchanged=True,
        profile_manifest=manifest, elapsed_seconds=time.perf_counter() - started,
        artifacts=artifacts, **checks))
    selected = pd.DataFrame(metric_rows).loc[lambda table: table.cohort.eq("all")]
    print(selected[["method", "tp", "fp", "fn", "tn", "precision", "recall", "fpr"]].to_string(index=False))
    print("VERIFIED: frozen profiles and native mains unchanged; no GPU inference", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=HERE / "design/experiment_long_batch1_audit_20260908.json")
    parser.add_argument("--source", type=Path, default=HERE / "design/frozen_alarm_comparison_20260908")
    parser.add_argument("--output", type=Path, default=HERE / "design/experiment_long_batch1_alarm_comparison_20260908")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
