#!/usr/bin/env python3
"""Compare route alarms on identical held-out tasks, without model inference."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAFE = ROOT / "safe&vlaconf/moe_trainfree"
RESULTS = SAFE / "results"
sys.path.insert(0, str(SAFE / "v82_validation"))
import monitor as guard
import run_analysis as audit
sys.path.insert(0, str(SAFE / "boundary_knn"))
from full_corpus_knn import validate_split

GUARDS = {
    "v7_calibrated": "v7", "v8_calibrated": "v8_fixed", "v82_calibrated": "v82",
    "clock": "clock", "eef_motion_low": "eef_motion_low",
}
GEOMETRY = ("knn20", "norm_only", "c4_assigned_radius", "c32_assigned_radius", "c32_union_radius")
METHODS = tuple(GUARDS) + GEOMETRY + ("cosine_knn20",)
ALPHAS = (0.02, 0.05)
CHECKPOINTS = (7, 10, 13, 20, 51)


def metrics(first, frame, cutoff=51):
    failure = frame.failure.to_numpy(bool)
    length = frame.length.to_numpy(int)
    hit = (first >= 0) & (first < length) & (first <= cutoff)
    tp, fp = int((hit & failure).sum()), int((hit & ~failure).sum())
    positive, negative = int(failure.sum()), int((~failure).sum())
    early_tp = int((hit & failure & (length - first - 1 >= 4)).sum())
    return dict(episodes=len(frame), failures=positive, successes=negative, tp=tp, fp=fp,
                fn=positive - tp, tn=negative - fp,
                recall=tp / positive if positive else None,
                fpr=fp / negative if negative else None,
                precision=tp / (tp + fp) if tp + fp else None,
                tp_with_four_later_queries=early_tp,
                median_first_tp=float(np.median(first[hit & failure])) if tp else None,
                median_first_fp=float(np.median(first[hit & ~failure])) if fp else None)


def calibrate(scores, rows, frame, valid, methods, fold):
    successful = ~frame.loc[rows, "failure"].to_numpy(bool)
    selected = frame.loc[rows].loc[successful]
    keys = (selected.task + "|" + selected.init_state_id.astype(str)).to_numpy()
    groups, names = pd.factorize(keys, sort=True)
    thresholds = np.empty((len(ALPHAS), len(methods)))
    records = []
    for mi, name in enumerate(methods):
        calibration = scores[mi, :len(rows)][successful]
        peaks = np.where(np.isfinite(calibration), calibration, -np.inf).max(axis=1)
        units = np.full(len(names), -np.inf)
        np.maximum.at(units, groups, peaks)
        for ai, alpha in enumerate(ALPHAS):
            tau, rank = guard.conformal_threshold(units, alpha)
            thresholds[ai, mi] = tau
            records.append(dict(fold=fold, method=name, alpha=alpha, calibration="task_init",
                                threshold=tau, rank=rank, groups=len(names)))
    first = np.asarray([
        [guard.first_alarm(scores[mi, len(rows):], valid, thresholds[ai, mi])
         for mi in range(len(methods))] for ai in range(len(ALPHAS))
    ])
    return thresholds, first, records


def verify_calibration(data):
    methods = data["methods"].astype(str)
    groups = data["calibration_group"].astype(str)
    success = ~data["calibration_failure"]
    decisions = 0
    for mi, _ in enumerate(methods):
        calibration = data["calibration_scores"][mi, success]
        maxima = np.where(np.isfinite(calibration), calibration, -np.inf).max(axis=1)
        unit = np.asarray([maxima[groups[success] == key].max() for key in sorted(set(groups[success]))])
        for ai, alpha in enumerate(ALPHAS):
            rank = int(np.ceil((len(unit) + 1) * (1 - alpha)))
            expected = np.sort(unit)[rank - 1] if rank <= len(unit) else np.inf
            assert expected == data["thresholds"][ai, mi]
            crossing = np.isfinite(data["scores"][mi]) & (data["scores"][mi] > expected) & data["valid"]
            first = np.where(crossing.any(axis=1), crossing.argmax(axis=1), -1)
            np.testing.assert_array_equal(first, data["first"][ai, mi])
            decisions += len(first)
    return decisions


def conditional_early_auc(frame, scores):
    rows = []
    for cohort, subset in (("all", frame), ("B", frame.loc[frame.run_id.eq(audit.RUN_B)])):
        for mi, method in enumerate(METHODS):
            per_task = []
            for task, part in subset.groupby("task", sort=True):
                values = []
                for query in range(7, 14):
                    eligible = part.loc[part.length > query]
                    value = guard.auc(eligible.failure.to_numpy(), scores[mi, eligible.index, query])
                    if np.isfinite(value):
                        values.append(value)
                if values:
                    per_task.append(float(np.mean(values)))
            rows.append(dict(cohort=cohort, method=method, query_start=7, query_end=13,
                             comparable_tasks=len(per_task),
                             mean_task_query_auc=float(np.mean(per_task)) if per_task else None))
    return rows


def export_comparisons(output, frame, first, scores):
    rows, suite_rows = [], []
    for ai, alpha in enumerate(ALPHAS):
        for mi, method in enumerate(METHODS):
            for cohort, part in (("all", frame), ("A", frame.loc[frame.run_id.eq(audit.RUN_A)]),
                                  ("B", frame.loc[frame.run_id.eq(audit.RUN_B)])):
                for cutoff in CHECKPOINTS:
                    rows.append(dict(method=method, alpha=alpha, cohort=cohort, query_cutoff=cutoff,
                                     **metrics(first[ai, mi, part.index], part, cutoff)))
            for suite, part in frame.groupby("suite", sort=True):
                suite_rows.append(dict(method=method, alpha=alpha, suite=suite,
                                       **metrics(first[ai, mi, part.index], part)))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(suite_rows).to_csv(output / "suite_metrics.csv", index=False)
    pd.DataFrame(conditional_early_auc(frame, scores)).to_csv(output / "early_conditional_auc.csv", index=False)

    comparisons = []
    chosen = ("v7_calibrated", "v8_calibrated", "v82_calibrated", "knn20", "c4_assigned_radius")
    for ai, alpha in enumerate(ALPHAS):
        for left, right in itertools.combinations(chosen, 2):
            a, b = (first[ai, METHODS.index(name)] >= 0 for name in (left, right))
            failure = frame.failure.to_numpy()
            comparisons.append(dict(alpha=alpha, left=left, right=right,
                left_only_tp=int((a & ~b & failure).sum()), right_only_tp=int((b & ~a & failure).sum()),
                left_only_fp=int((a & ~b & ~failure).sum()), right_only_fp=int((b & ~a & ~failure).sum()),
                both_tp=int((a & b & failure).sum()), both_fp=int((a & b & ~failure).sum())))
    pd.DataFrame(comparisons).to_csv(output / "paired_alarm_counts.csv", index=False)

    core = ("v7_calibrated", "v82_calibrated", "knn20", "c4_assigned_radius")
    union_rows = []
    for ai, alpha in enumerate(ALPHAS):
        alarms = first[ai, [METHODS.index(name) for name in core]]
        for cohort, part in (("all", frame), ("B", frame.loc[frame.run_id.eq(audit.RUN_B)])):
            count = np.asarray([len(set(alarms[:, row][alarms[:, row] >= 0].tolist())) for row in part.index])
            union_rows.append(dict(alpha=alpha, cohort=cohort, methods=list(core), main_rows=len(part),
                mains_with_any_alarm=int((count > 0).sum()), unique_first_alarm_states=int(count.sum()),
                states_per_main_max=int(count.max()), nine_suffixes_for_every_state=int(count.sum() * 9)))
    audit.write_json(output / "alarm_union_budget.json", union_rows)

    frozen_path = RESULTS / "v82_validation_20260908/frozen_first_alarms.csv"
    frozen = pd.read_csv(frozen_path)
    for key in ("source", "episode", "run_id", "task", "length", "failure"):
        np.testing.assert_array_equal(frozen[key], frame[key])
    anchors = []
    for name in ("v7_frozen", "v8_frozen", "v82_frozen", "v8_padding_corrected", "v82_padding_corrected"):
        for cohort, part in (("all", frame), ("B", frame.loc[frame.run_id.eq(audit.RUN_B)])):
            alarms = frozen.loc[part.index, name].to_numpy(int)
            assert ((alarms == -1) | ((alarms >= 0) & (alarms < part.length))).all()
            for cutoff in CHECKPOINTS:
                anchors.append(dict(method=name, cohort=cohort, query_cutoff=cutoff,
                                    protocol="historical_global_profile_not_heldout_task_fit",
                                    **metrics(alarms, part, cutoff)))
    pd.DataFrame(anchors).to_csv(output / "historical_frozen_metrics.csv", index=False)
    return audit.digest(frozen_path)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "folds").mkdir()
    started = time.perf_counter()
    contract = dict(methods=METHODS, alphas=ALPHAS, calibration="task_init_success_maxima",
        split="Round 9 frozen 16 folds, held-out tasks, A reference/calibration, A+B test once each",
        guard_versions="reference-scaled success-recalibrated guard variants, not original frozen v7/v8",
        kmeans_c4="exploratory configuration chosen in previously inspected data",
        nominal_alpha_is_not_unseen_task_fpr_guarantee=True, new_blind_test=False,
        hidden_capture=False, model_queries=0, gpu_compute=False,
        successful_trajectory_false_alarms_never_removed_by_lead_cutoff=True,
        four_later_queries="length - first_alarm_query - 1 >= 4; supplementary TP only",
        pro_plus_results=False)
    audit.write_json(output / "contract.json", contract)
    frame, cache, raw, eef = audit.load_inputs(output)
    parent = RESULTS / "round9_full_corpus"
    cluster = RESULTS / "round11_kmeans"
    manifests = {path: json.loads((path / "sealed_manifest.json").read_text()) for path in (parent, cluster)}
    for path in manifests:
        pd.testing.assert_frame_equal(pd.read_csv(path / "index.csv"), frame.drop(columns=[
            "global_row", "failure", "action_steps", "primary_failure_reason"]))
    first_all = np.full((len(ALPHAS), len(METHODS), len(frame)), -1, np.int16)
    scores_all = np.full((len(METHODS), len(frame), 52), np.nan, np.float32)
    tested = np.zeros(len(frame), int)
    threshold_rows, provenance = [], {}
    decisions = 0
    for info in manifests[parent]["folds"]:
        fold = info["fold"]
        files = [path / "predictions" / (fold + ".npz") for path in (parent, cluster)]
        for path in files:
            digest = audit.digest(path)
            assert digest == manifests[path.parent.parent]["artifacts"][str(path.relative_to(path.parent.parent))]
            provenance[str(path.relative_to(ROOT))] = digest
        old, km = (audit.archive(path) for path in files)
        reference, calibration, test = (old[key] for key in ("reference_rows", "calibration_rows", "test_rows"))
        validate_split(frame, reference, calibration, test, info)
        for key in ("reference_rows", "calibration_rows", "test_rows", "reference_labels", "calibration_labels"):
            np.testing.assert_array_equal(old[key], km[key])
        np.testing.assert_array_equal(old["calibration_labels"].astype(bool), frame.loc[calibration, "failure"])
        needed = np.r_[reference, calibration, test]
        subset = {key: value[needed] for key, value in cache.items() if key in ("mobility", "acceleration", "periodicity", "valid")}
        streams, profile = guard.build_budget_streams(subset, raw[needed], eef[needed], np.arange(len(reference)))
        values = {name: streams[guard.METHODS.index(source), len(reference):] for name, source in GUARDS.items()}
        for name in GEOMETRY:
            mi = list(km["methods"].astype(str)).index(name)
            values[name] = np.concatenate((km["calibration_scores"][mi], km["scores"][mi]))
        mi = list(old["methods"].astype(str)).index("cosine")
        values["cosine_knn20"] = np.concatenate((old["calibration_scores"][mi], old["scores"][mi]))
        scores = np.asarray([values[name] for name in METHODS], np.float32)
        rows = np.r_[calibration, test]
        assert np.isnan(scores[:, ~cache["valid"][rows]]).all()
        thresholds, first, records = calibrate(scores, calibration, frame, cache["valid"][test], METHODS, fold)
        for name in GEOMETRY:
            mi, source = METHODS.index(name), list(km["methods"].astype(str)).index(name)
            for ai, alpha in enumerate(ALPHAS):
                source_alpha = int(np.flatnonzero(np.isclose(km["alphas"], alpha))[0])
                assert thresholds[ai, mi] == km["thresholds"][1, source_alpha, source]
                np.testing.assert_array_equal(first[ai, mi], km["first"][1, source_alpha, source])
        # Fit scales only once, then verify a future change cannot alter q0..q13.
        mobility = subset["mobility"].copy()
        acceleration = subset["acceleration"].copy()
        periodicity = subset["periodicity"].copy()
        for value in (mobility, acceleration, periodicity):
            value[:, 14:] = np.nan
        original_intrinsic = guard.intrinsic_score_arrays(subset["mobility"], subset["acceleration"],
            subset["periodicity"], profile["periodicity_scale"])
        truncated = guard.intrinsic_score_arrays(mobility, acceleration, periodicity, profile["periodicity_scale"])
        for key in original_intrinsic:
            np.testing.assert_allclose(original_intrinsic[key][:, :14], truncated[key][:, :14], equal_nan=True)
        np.testing.assert_allclose(guard.v8_streams(raw[needed], subset["valid"])[:, :14],
                                  guard.v8_streams(raw[needed, :14], subset["valid"][:, :14]), equal_nan=True)
        ncal = len(calibration)
        data = dict(methods=np.asarray(METHODS), alphas=np.asarray(ALPHAS), thresholds=thresholds,
            first=first, scores=scores[:, ncal:], calibration_scores=scores[:, :ncal],
            valid=cache["valid"][test], test_rows=test, reference_rows=reference,
            calibration_rows=calibration, calibration_failure=frame.loc[calibration, "failure"].to_numpy(),
            calibration_group=(frame.loc[calibration, "task"] + "|" + frame.loc[calibration, "init_state_id"].astype(str)).to_numpy(dtype=str))
        decisions += verify_calibration(data)
        np.savez_compressed(output / "folds" / (fold + ".npz"), **data)
        audit.write_json(output / "folds" / (fold + ".json"), dict(split=info, guard_profile=profile))
        first_all[:, :, test] = first
        scores_all[:, test] = scores[:, ncal:]
        tested[test] += 1
        threshold_rows.extend(records)
        print("COMPARED %s: %d held-out episodes" % (fold, len(test)), flush=True)
    np.testing.assert_array_equal(tested, 1)
    pd.DataFrame(threshold_rows).to_csv(output / "calibration_thresholds.csv", index=False)
    frozen_hash = export_comparisons(output, frame, first_all, scores_all)
    np.savez_compressed(output / "first_alarms.npz", methods=np.asarray(METHODS), alphas=np.asarray(ALPHAS), first=first_all)
    sources = [Path(__file__), SAFE / "v82_validation/monitor.py", SAFE / "v82_validation/run_analysis.py",
               SAFE / "boundary_knn/full_corpus_knn.py", ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py"]
    report = dict(contract, episodes=len(frame), queries=int(cache["valid"].sum()), folds=int(tested.sum() == 32000) * 16,
        independent_alarm_decisions=decisions, existing_geometry_predictions_reproduced=True,
        prefix_invariance_through_q13=True, source_sha256={str(path.relative_to(ROOT)): audit.digest(path) for path in sources},
        prediction_input_sha256=provenance, frozen_anchor_sha256=frozen_hash, seconds=time.perf_counter() - started)
    report["artifacts"] = {str(path.relative_to(output)): audit.digest(path) for path in output.rglob("*") if path.is_file()}
    audit.write_json(output / "verification.json", report)
    selected = pd.read_csv(output / "metrics.csv")
    print(selected.loc[(selected.alpha == .05) & selected.cohort.eq("all") & (selected.query_cutoff == 51),
                       ["method", "tp", "fp", "recall", "fpr", "precision"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "design/alarm_comparison_20260908")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output.resolve())
