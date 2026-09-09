"""Audit VLAConf-inspired success support with only scalar outcome calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import zarr

from .audit_budget import TEST_SPLITS, load_inputs, task_folds
from .data import json_records, sha256, write_json
from .evaluation import calibration_bins, cluster_draws, metrics
from .features import DETECTOR_PATH, FEATURE_NAMES
from .progress_statistics import matched_auc
from .pure_moe import LOCAL_NAMES, WINDOW, select_local_features
from .scalar_confidence import (AGGREGATIONS, NEIGHBORS, SCALAR_MODELS, SCHEMA, SCORE_WINDOW,
                                MoEScalarMonitor, MonotoneSigmoid, SuccessSupport,
                                aggregate_scores, freeze_risk)


HERE = Path(__file__).resolve().parents[1]
MAIN_MODELS = ["prior", *SCALAR_MODELS, "supervised_moe", "clock_cap", "prior_unseen_task",
               *(name + "_unseen_task" for name in SCALAR_MODELS), "supervised_moe_unseen_task"]


def verified_artifact(directory, name):
    expected = json.loads((directory / "artifact_manifest.json").read_text())[name]
    if sha256(directory / name) != expected:
        raise ValueError(f"previous experiment artifact changed: {directory / name}")
    return expected


def attach_controls(frame, budget_dir, physical_dir):
    hashes = {}
    hashes[str(budget_dir / "predictions.csv.gz")] = verified_artifact(budget_dir, "predictions.csv.gz")
    names = dict(moe_local_global="supervised_moe", clock_cap_global="clock_cap",
                 moe_local_unseen_task="supervised_moe_unseen_task")
    old = pd.read_csv(budget_dir / "predictions.csv.gz", usecols=["episode_row", "query", *names])
    np.testing.assert_array_equal(frame[["episode_row", "query"]], old[["episode_row", "query"]])
    for source, target in names.items():
        frame[target] = old[source].to_numpy()
    hashes[str(physical_dir / "labels.csv.gz")] = verified_artifact(physical_dir, "labels.csv.gz")
    # Only the causal physical stage is read, never future-progress labels or eligibility.
    stage = pd.read_csv(physical_dir / "labels.csv.gz", usecols=["episode_row", "query", "stage"])
    result = frame.merge(stage, on=["episode_row", "query"], how="left", validate="one_to_one")
    np.testing.assert_array_equal(frame[["episode_row", "query"]], result[["episode_row", "query"]])
    if result.stage.isna().any():
        raise ValueError("missing physical evaluation stage")
    return result, hashes


def score_in_batches(model, features, label, batch_size=8192):
    blocks = []
    started = time.perf_counter()
    for start in range(0, len(features), batch_size):
        blocks.append(model.score(features[start:start+batch_size]))
        if start // batch_size % 4 == 0 or start+batch_size >= len(features):
            print(f"{label}: {min(start+batch_size, len(features)):,}/{len(features):,} prefixes, "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    return np.concatenate(blocks)


def fit_models(episodes, frame, x, output):
    local = select_local_features(x)
    train = frame.split.eq("train").to_numpy()
    cal = frame.split.eq("calibration").to_numpy()
    test = frame.split.isin(TEST_SPLITS).to_numpy()
    folds = task_folds(episodes)
    frame["task_fold"] = frame.task.map(folds)
    y = frame.success.to_numpy(int)
    freeze = freeze_risk(x[:, FEATURE_NAMES.index("freeze_score")])
    for name in SCALAR_MODELS:
        for suffix in ("", "_unseen_task"):
            frame[name + suffix] = np.nan
            frame[name + suffix + "_score"] = np.nan
    frame["prior"], frame["prior_unseen_task"] = np.nan, np.nan
    coefficients, audits, bundle = [], [], None
    for fold in (None, 0, 1, 2, 3, 4):
        label = "shared" if fold is None else f"held_task_fold{fold}"
        held = np.zeros(len(frame), bool) if fold is None else frame.task_fold.eq(fold).to_numpy()
        references, calibration = train & (y == 1) & ~held, cal & ~held
        evaluation = calibration | (test if fold is None else (test & held))
        destinations = evaluation if fold is None else test & held
        ref_tasks = set(frame.loc[references, "task"])
        held_tasks = set(frame.loc[held, "task"])
        if ref_tasks & held_tasks or set(frame.loc[calibration, "task"]) & held_tasks:
            raise ValueError("held tasks reached the reference library or calibration")
        if set(frame.loc[references, "cluster"]) & set(frame.loc[calibration, "cluster"]):
            raise ValueError("reference/calibration initial-state leakage")
        primary = destinations & frame.split.eq("test_unseen_init").to_numpy()
        if set(frame.loc[references | calibration, "cluster"]) & set(frame.loc[primary, "cluster"]):
            raise ValueError("primary test initial-state leakage")
        print(f"build {label}: {references.sum():,} success references, {calibration.sum():,} calibration prefixes", flush=True)
        support = SuccessSupport().fit(local[references])
        selected = frame.loc[evaluation]
        score = score_in_batches(support, local[evaluation], label)
        aggregated = aggregate_scores(score, selected.episode_row.to_numpy(), selected["query"].to_numpy())
        aggregated["freeze"] = freeze[evaluation]
        local_cal, local_dest = calibration[evaluation], destinations[evaluation]
        suffix = "" if fold is None else "_unseen_task"
        frame.loc[destinations, "prior" + suffix] = y[calibration].mean()
        calibrators = {}
        for name, values in aggregated.items():
            calibrator = MonotoneSigmoid.fit(values[local_cal], y[calibration])
            calibrators[name] = calibrator
            frame.loc[destinations, name + suffix] = calibrator.predict(values[local_dest])
            frame.loc[destinations, name + suffix + "_score"] = values[local_dest]
            coefficients.append(dict(scope=label, model=name, **asdict(calibrator)))
        reference_keys = frame.loc[references, ["episode_row", "query", "task", "cluster"]]
        reference_keys.to_csv(output / "models" / f"{label}_references.csv.gz", index=False)
        current_bundle = dict(schema=SCHEMA, feature_names=LOCAL_NAMES, window=WINDOW,
                              score_window=SCORE_WINDOW, neighbors=NEIGHBORS, support=support,
                              calibrators=calibrators, detector_sha256=sha256(DETECTOR_PATH),
                              protocol_sha256=sha256(HERE / "VLACONF_PROTOCOL.md"),
                              online_task_suite_clock_budget_inputs=False,
                              estimand="P(original_deadline_success | recent_MoE_support,active,q>=7)",
                              policy_by_suite=json.loads((HERE / "results/data_audit.json").read_text())["policy_by_suite"])
        joblib.dump(current_bundle, output / "models" / f"{label}.joblib", compress=3)
        if fold is None:
            bundle = current_bundle
        audits.append(dict(scope=label, fold=fold, held_tasks=sorted(held_tasks),
                          reference_tasks=sorted(ref_tasks), calibration_tasks=sorted(set(frame.loc[calibration, "task"])),
                          reference_queries=int(references.sum()), reference_episodes=int(reference_keys.episode_row.nunique()),
                          calibration_queries=int(calibration.sum()), test_queries=int((destinations & test).sum()),
                          reference_sha256=sha256(output / "models" / f"{label}_references.csv.gz")))
    if frame.loc[test, MAIN_MODELS].isna().any().any():
        raise ValueError("incomplete test probabilities")
    pd.DataFrame(coefficients).to_csv(output / "calibrators.csv", index=False)
    write_json(output / "fit_audit.json", audits)
    write_json(output / "task_folds.json", folds)
    return bundle


def evaluate(frame, episodes, output):
    rows, reliability = [], []
    for split in TEST_SPLITS:
        for suite in ("all", *sorted(episodes.suite.unique())):
            base = frame[(frame.split == split) & ((frame.suite == suite) if suite != "all" else True)]
            for selection in ("all_q7plus", "q7", "q15", "q25", "half_k4_alarm"):
                block = base if selection == "all_q7plus" else (
                    base[base.half_k4_alarm] if selection == "half_k4_alarm" else base[base["query"] == int(selection[1:])])
                if block.empty:
                    continue
                for name in MAIN_MODELS:
                    score = metrics(block.success.to_numpy(), block[name].to_numpy())
                    raw_auc = None
                    if name + "_score" in block and block.success.nunique() == 2:
                        raw_auc = float(roc_auc_score(block.success, -block[name + "_score"]))
                    rows.append(dict(split=split, suite=suite, selection=selection, model=name,
                                     raw_score_auroc=raw_auc, episodes=int(block.episode_row.nunique()), **score))
                    if suite == "all" and selection == "all_q7plus":
                        reliability.extend(dict(split=split, model=name, **row) for row in
                                           calibration_bins(block.success.to_numpy(), block[name].to_numpy()))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "calibration.csv", index=False)
    primary = frame[frame.split == "test_unseen_init"].copy()
    population = episodes[episodes.split == "test_unseen_init"]
    # The reusable matched-pair routine names its binary target 'progress'.
    primary["progress"] = primary.success
    matched = []
    for selection, strata in (("task_query", ["task", "query"]), ("task_query_stage", ["task", "query", "stage"])):
        print(f"matched final-success discrimination: {selection}", flush=True)
        result, coverage, per_task = matched_auc(primary, MAIN_MODELS, strata, population)
        result.insert(0, "selection", selection)
        matched.append(result)
        coverage.to_csv(output / f"support_{selection}.csv", index=False)
        per_task.to_csv(output / f"per_task_{selection}.csv", index=False)
    pd.concat(matched).to_csv(output / "matched_auroc.csv", index=False)
    comparisons = [("support_window4", "prior"), ("support_window4", "supervised_moe"),
                   ("support_window4", "clock_cap"), ("support_window4", "support_prefix_max"),
                   ("support_window4", "support_prefix_mean"), ("support_window4", "freeze"),
                   ("support_window4_unseen_task", "prior_unseen_task"),
                   ("support_window4_unseen_task", "supervised_moe_unseen_task")]
    intervals = []
    for left, right in comparisons:
        losses = primary[["task", "cluster"]].copy()
        y = primary.success.to_numpy()
        for name in (left, right):
            p = np.clip(primary[name].to_numpy(), 1e-6, 1-1e-6)
            losses[name+"_brier"] = (p-y)**2
            losses[name+"_log_loss"] = -(y*np.log(p)+(1-y)*np.log1p(-p))
        losses["count"] = 1
        columns = [left+"_brier", right+"_brier", left+"_log_loss", right+"_log_loss", "count"]
        draws = cluster_draws(losses, columns, population=population)
        for metric, i in (("brier", 0), ("log_loss", 2)):
            low, high = np.quantile((draws[:, i]-draws[:, i+1])/draws[:, -1], [0.025, 0.975])
            intervals.append(dict(left=left, right=right, metric=metric,
                                  difference=float((losses[columns[i]]-losses[columns[i+1]]).mean()),
                                  low=float(low), high=float(high)))
    pd.DataFrame(intervals).to_csv(output / "cluster_intervals.csv", index=False)


def describe_dynamics(frame, episodes, output):
    rows, alarm_rows = [], []
    index = episodes.set_index("episode_row")
    for split in TEST_SPLITS:
        block = frame[frame.split == split].copy()
        first_q = index.loc[block.episode_row, "first_q_freeze_back_half_k4"].to_numpy()
        block["after_alarm"] = (first_q >= 0) & (block["query"].to_numpy() > first_q)
        for name in ["support_" + agg for agg in AGGREGATIONS]:
            delta = block.groupby("episode_row", sort=False)[name].diff()
            for selection in ("all", "successful_after_alarm", "failed_after_alarm"):
                take = delta.notna()
                if selection != "all":
                    take &= block.after_alarm & block.success.eq(int(selection.startswith("successful")))
                values = delta[take]
                rows.append(dict(split=split, model=name, selection=selection, transitions=len(values),
                                 episodes=int(block.loc[take, "episode_row"].nunique()),
                                 increases=int((values > 1e-10).sum()),
                                 increase_rate=float((values > 1e-10).mean()) if len(values) else None,
                                 median_delta=float(values.median()) if len(values) else None))
        alarms = block[block.half_k4_alarm]
        for row in alarms.itertuples():
            later = block[(block.episode_row == row.episode_row) & (block["query"] > row.query)
                          & (block["query"] <= row.query + 5)]
            for name in ["support_" + agg for agg in AGGREGATIONS]:
                alarm_rows.append(dict(split=split, episode_row=row.episode_row, query=row.query,
                                       success=row.success, model=name, probability_at_alarm=getattr(row, name),
                                       observed_later_queries=len(later),
                                       max_later_increase=float(later[name].max()-getattr(row, name)) if len(later) else None))
    pd.DataFrame(rows).to_csv(output / "probability_dynamics.csv", index=False)
    pd.DataFrame(alarm_rows).to_csv(output / "first_alarm_probabilities.csv", index=False)


def verify_online(source, hub, episodes, frame, bundle):
    samples = json.loads((source / "raw_verification.json").read_text())["samples"]
    test_bundle = bundle
    saved = frame.set_index(["episode_row", "query"])
    count, max_score_error, max_probability_error = 0, 0.0, 0.0
    raw_queries, timings = 0, []
    for sample in samples:
        group = episodes[episodes.source_run == sample["source_run"]].sort_values("episode")
        local = int(np.flatnonzero(group.episode.to_numpy() == sample["episode"])[0])
        row = group.iloc[local]
        offset, length = int(group.length.iloc[:local].sum()), int(row.length)
        store = zarr.open_group(str(hub / row.source_run / "server/routes.zarr"), mode="r")
        raw = np.asarray(store["hb_router_probs"][offset:offset+length, :, 9, 1:, :])
        monitor = MoEScalarMonitor(test_bundle)
        raw_queries += len(raw)
        # Training rows are deliberately not evaluated against their own library.
        # Batch parity on those sampled episodes is a numerical check only.
        roots = []
        batch_features = []
        from .pure_moe import features_from_window
        from .features import detector
        for chunk in raw:
            roots.append(detector.root_action_routes(chunk))
            if len(roots) >= WINDOW:
                batch_features.append(features_from_window(roots[-WINDOW:])[0])
        if batch_features:
            batch_scores = bundle["support"].score(np.asarray(batch_features))
            expected_scores = aggregate_scores(batch_scores, np.full(len(batch_scores), row.episode_row),
                                               np.arange(WINDOW-1, len(raw)))
        for q, chunk in enumerate(raw):
            started = time.perf_counter()
            result = monitor.update(chunk)
            if q < WINDOW-1:
                assert result["ready"] is False and result["success_probability"] is None
                continue
            timings.append(time.perf_counter()-started)
            for agg in AGGREGATIONS:
                name = "support_" + agg
                expected_score = expected_scores[name][q-(WINDOW-1)]
                expected_p = float(bundle["calibrators"][name].predict(expected_score))
                if row.split != "train":
                    stored = saved.loc[(row.episode_row, q)]
                    np.testing.assert_allclose(expected_score, stored[name+"_score"], rtol=1e-8, atol=1e-8)
                    np.testing.assert_allclose(expected_p, stored[name], rtol=1e-8, atol=1e-8)
                max_score_error = max(max_score_error, abs(result["aggregated_scores"][agg]-expected_score))
                max_probability_error = max(max_probability_error, abs(result["probabilities"][agg]-expected_p))
                np.testing.assert_allclose(result["probabilities"][agg], expected_p, rtol=1e-8, atol=1e-8)
            assert result["success_probability"] == result["probabilities"]["window4"]
            count += 1
    return dict(source_runs=len(samples), raw_queries=raw_queries, ready_queries=count,
                aggregation_probabilities_checked=count*len(AGGREGATIONS), samples=samples,
                max_score_error=max_score_error, max_probability_error=max_probability_error,
                latency_median_ms=float(np.median(timings)*1000), latency_p95_ms=float(np.quantile(timings, .95)*1000),
                latency_scope="CPU synchronous scalar monitor, four threads, excludes VLA inference",
                training_sample_checks="numerical parity only; no in-library training performance reported")


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    if not args.render_only:
        episodes, frame, x, hashes = load_inputs(args.source)
        frame, other_hashes = attach_controls(frame, args.budget_source, args.physical_source)
        bundle = fit_models(episodes, frame, x, output)
        evaluate(frame, episodes, output)
        describe_dynamics(frame, episodes, output)
        print("verify raw-route online probabilities", flush=True)
        raw = verify_online(args.source, args.hub, episodes, frame, bundle)
        write_json(output / "raw_verification.json", raw)
        frame.loc[frame.split.ne("train")].to_csv(output / "predictions.csv.gz", index=False, float_format="%.12g")
        counts = frame.groupby("split").agg(queries=("query", "size"), episodes=("episode_row", "nunique"),
                                           successes=("success", "sum")).reset_index()
        counts.to_csv(output / "row_counts.csv", index=False)
        write_json(output / "summary.json", dict(source_hashes=hashes, control_hashes=other_hashes,
                   protocol_sha256=sha256(HERE / "VLACONF_PROTOCOL.md"), source_episodes=len(episodes),
                   source_queries=int(episodes.length.sum()), ready_queries=len(frame),
                   primary_method="support_window4", target="original_deadline_success",
                   feature_names=LOCAL_NAMES, neighbors=NEIGHBORS, router_window=WINDOW, score_window=SCORE_WINDOW,
                   future_or_metadata_online_inputs=False, true_trap_and_escape_not_identified=True,
                   main_metrics=json_records(pd.read_csv(output / "metrics.csv").query(
                       "split == 'test_unseen_init' and suite == 'all' and selection == 'all_q7plus'"))))
    from .scalar_report import report
    report(output)
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".log"}
    for path in [HERE / "VLACONF_PROTOCOL.md", *sorted((HERE / "probability").glob("*.py")),
                 *sorted((HERE / "tests").glob("*.py"))]:
        manifest[str(path.relative_to(HERE))] = sha256(path)
    write_json(output / "artifact_manifest.json", manifest)
    print(f"completed {output / 'REPORT.zh.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--budget-source", type=Path, default=HERE / "results_no_budget")
    parser.add_argument("--physical-source", type=Path, default=HERE / "results_progress")
    parser.add_argument("--hub", type=Path, default=HERE.parent / "VLA_MUI_HUB")
    parser.add_argument("--output", type=Path, default=HERE / "results_vlaconf")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
