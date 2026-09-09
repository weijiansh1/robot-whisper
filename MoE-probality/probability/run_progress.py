"""Fit and evaluate a fixed-horizon, MoE-only physical progress readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import zarr

from .audit_budget import load_inputs, task_folds, TEST_SPLITS
from .data import json_records, sha256, write_json
from .evaluation import calibration_bins, cluster_draws, metrics
from .features import DETECTOR_PATH
from .model import MODEL_PARAMS
from .progress_labels import HORIZON, STAGE_NAMES, episode_labels
from .progress_model import MoEProgressMonitor, SCHEMA
from .progress_statistics import matched_auc
from .pure_moe import LOCAL_NAMES, WINDOW, fit_calibrated, probabilities, select_local_features


HERE = Path(__file__).resolve().parents[1]
MAIN_MODELS = ["prior", "task_clock", "moe", "physical", "physical_moe",
               "physical_task_clock", "physical_task_clock_moe", "prior_unseen_task",
               "moe_unseen_task", "physical_unseen_task", "physical_moe_unseen_task"]


def build_labels(episodes, physical_dir, output):
    manifest = json.loads((physical_dir / "manifest.json").read_text())
    if manifest["partial"] or (manifest["episodes"], manifest["queries"], manifest["tasks"]) != (32000, 508023, 40):
        raise ValueError("complete physical extraction required for both natural cohorts")
    if set(a["source_run"] for a in manifest["runs"]) != set(episodes.source_run):
        raise ValueError("physical and MoE cohorts differ")
    blocks, checks = [], []
    for audit in manifest["runs"]:
        cache_path = physical_dir / audit["cache_file"]
        if sha256(cache_path) != audit["cache_sha256"] or sha256(HERE / "probability/extract_progress.py") != audit["extractor_sha256"]:
            raise ValueError("physical cache or extractor changed")
        if audit["robot_alignment_outliers"] or audit["predicate_api_mismatches"]:
            raise ValueError("physical alignment requires review before fitting")
        if audit["input_hash"] != sha256(HERE / "results/episodes.csv"):
            raise ValueError("physical labels used another episode index")
        group = episodes[episodes.source_run == audit["source_run"]].sort_values("episode")
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {name: archive[name] for name in archive.files}
            offset = 0
            for row in group.itertuples():
                sl = slice(offset, offset+row.length)
                np.testing.assert_array_equal(cache["episode_row"][sl], row.episode_row)
                np.testing.assert_array_equal(cache["query"][sl], np.arange(row.length))
                result = episode_labels(**{name: cache[name][sl] for name in ("goal", "grasp", "position", "distance")},
                                         subject_goal=cache["subject_goal"], success=bool(row.success),
                                         action_steps=row.action_steps, max_steps=row.max_steps)
                physical = result.pop("physical")
                block = pd.DataFrame(result)
                block.insert(0, "episode_row", row.episode_row)
                for i, name in enumerate(STAGE_NAMES):
                    block[name] = physical[:, i]
                blocks.append(block)
                offset += row.length
            if offset != len(cache["query"]):
                raise ValueError("unaccounted physical checkpoint rows")
        checks.append({k: audit[k] for k in ("source_run", "episodes", "queries", "max_robot_state_error",
                      "max_eef_position_error_m", "max_eef_rotation_error_rad", "max_gripper_error_m",
                      "robot_alignment_outliers", "restored_completed_checkpoints", "layout_verified", "cache_sha256")})
    labels = pd.concat(blocks, ignore_index=True).sort_values(["episode_row", "query"]).reset_index(drop=True)
    if labels.duplicated(["episode_row", "query"]).any():
        raise ValueError("duplicate physical checkpoint")
    labels.to_csv(output / "labels.csv.gz", index=False, float_format="%.9g")
    write_json(output / "physical_audit.json", dict(
        runs=checks, episodes=32000, checkpoints=len(labels), eligible_queries=int(labels.eligible.sum()),
        restored_completed_checkpoints=int(labels.already_complete.sum()),
        restored_complete_q7plus=int((labels.already_complete & (labels["query"] >= 7)).sum()),
        source_manifest_sha256=sha256(physical_dir / "manifest.json"),
        alignment_thresholds=manifest["runs"][0]["alignment_thresholds"],
        sampled_or_executed_actions=0, physics_source_sha256=manifest["runs"][0]["physics_source_sha256"]))
    print(f"physical labels: {len(labels):,} checkpoints, {labels.eligible.sum():,} eligible", flush=True)
    return labels


def fit_models(episodes, frame, local, output):
    physical = frame[STAGE_NAMES].to_numpy(np.float32)
    task_ids = pd.Categorical(frame.task, categories=sorted(episodes.task.unique())).codes
    clock = np.column_stack([frame["query"].to_numpy(), np.eye(episodes.task.nunique())[task_ids]]).astype(np.float32)
    designs = dict(moe=local, task_clock=clock, physical=physical,
                   physical_moe=np.column_stack([physical, local]),
                   physical_task_clock=np.column_stack([physical, clock]),
                   physical_task_clock_moe=np.column_stack([physical, clock, local]))
    y = frame.progress.to_numpy(int)
    train, cal = frame.split.eq("train").to_numpy(), frame.split.eq("calibration").to_numpy()
    frame["prior"] = float(y[train].mean())
    for name, design in designs.items():
        print(f"fit progress {name}: train={train.sum()}, calibration={cal.sum()}", flush=True)
        model = fit_calibrated(design[train], y[train], design[cal], y[cal])
        raw, calibrated = probabilities(model, design)
        frame[name], frame[name + "_raw"] = calibrated, raw
        joblib.dump(model, output / "models" / (name + ".joblib"), compress=3)
        if name == "moe":
            bundle = dict(schema=SCHEMA, model=model, feature_names=LOCAL_NAMES, window=WINDOW,
                          horizon_chunks=HORIZON, replan_steps=10, detector_sha256=sha256(DETECTOR_PATH),
                          model_params=MODEL_PARAMS, online_metadata_inputs=False,
                          estimand="P(observed_physical_milestone_within_50_actions | last_8_MoE,eligible_active_prefix)",
                          label_protocol_sha256=sha256(HERE / "PROGRESS_PROTOCOL.md"),
                          policy_by_suite=json.loads((HERE / "results/data_audit.json").read_text())["policy_by_suite"])
            joblib.dump(bundle, output / "models/moe_progress_shared.joblib", compress=3)
    folds = task_folds(episodes)
    for name in ("prior_unseen_task", "moe_unseen_task", "physical_unseen_task", "physical_moe_unseen_task"):
        frame[name] = np.nan
        if name != "prior_unseen_task":
            frame[name + "_raw"] = np.nan
    splits = []
    for fold in range(5):
        held = frame.task.map(folds).eq(fold).to_numpy()
        testing = held & frame.split.isin(TEST_SPLITS).to_numpy()
        fitting, calibration = train & ~held, cal & ~held
        held_tasks = sorted(frame.loc[held, "task"].unique())
        assert set(held_tasks).isdisjoint(frame.loc[fitting | calibration, "task"])
        frame.loc[testing, "prior_unseen_task"] = y[fitting].mean()
        for name in ("moe", "physical", "physical_moe"):
            print(f"fit progress unseen task {fold+1}/5 {name}", flush=True)
            design = designs[name]
            model = fit_calibrated(design[fitting], y[fitting], design[calibration], y[calibration])
            raw, calibrated = probabilities(model, design[testing])
            frame.loc[testing, name + "_unseen_task"] = calibrated
            frame.loc[testing, name + "_unseen_task_raw"] = raw
            joblib.dump(model, output / "models" / f"fold{fold}_{name}.joblib", compress=3)
        splits.append(dict(fold=fold, held_tasks=held_tasks,
                           training_tasks=sorted(frame.loc[fitting, "task"].unique()),
                           training_queries=int(fitting.sum()), calibration_queries=int(calibration.sum()),
                           testing_queries=int(testing.sum())))
    write_json(output / "task_folds.json", splits)
    return bundle


def evaluate(frame, episodes, output):
    rows, reliability = [], []
    names = MAIN_MODELS + [name + "_raw" for name in MAIN_MODELS if not name.startswith("prior")]
    for split in TEST_SPLITS:
        for suite in ("all", *sorted(episodes.suite.unique())):
            block = frame[(frame.split == split) & ((frame.suite == suite) if suite != "all" else True)]
            for name in names:
                score = metrics(block.progress.to_numpy(), block[name].to_numpy())
                score["positives"], score["positive_rate"] = score.pop("successes"), score.pop("success_rate")
                rows.append(dict(split=split, suite=suite, model=name, **score))
                if suite == "all":
                    reliability.extend(dict(split=split, model=name, **r)
                                       for r in calibration_bins(block.progress.to_numpy(), block[name].to_numpy()))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "calibration.csv", index=False)
    matched = []
    primary = frame[frame.split == "test_unseen_init"].copy()
    population = episodes[episodes.split == "test_unseen_init"]
    for selection, strata in (("task_query", ["task", "query"]), ("task_query_stage", ["task", "query", "stage"])):
        print(f"matched progress discrimination: {selection}", flush=True)
        result, coverage, task_results = matched_auc(primary, MAIN_MODELS, strata, population)
        result.insert(0, "selection", selection)
        matched.append(result)
        coverage.to_csv(output / f"support_{selection}.csv", index=False)
        task_results.to_csv(output / f"per_task_{selection}.csv", index=False)
    pd.concat(matched).to_csv(output / "matched_auroc.csv", index=False)
    comparisons = [("moe", "prior"), ("moe", "task_clock"), ("physical_moe", "physical"),
                   ("physical_task_clock_moe", "physical_task_clock"),
                   ("moe_unseen_task", "prior_unseen_task"),
                   ("physical_moe_unseen_task", "physical_unseen_task")]
    intervals = []
    for left, right in comparisons:
        losses = primary[["task", "cluster"]].copy()
        y = primary.progress.to_numpy()
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


def describe_labels(labels, frame, episodes, output):
    index = episodes.set_index("episode_row")
    all_rows = labels.copy()
    for name in ("suite", "task", "split", "success", "source_run", "episode"):
        all_rows[name] = index.loc[all_rows.episode_row, name].to_numpy()
    counts = []
    for split in ("all", "train", "calibration", *TEST_SPLITS):
        block = all_rows if split == "all" else all_rows[all_rows.split == split]
        selected = block[block.eligible]
        counts.append(dict(split=split, raw_queries=len(block), eligible_queries=len(selected),
                           eligible_episodes=int(selected.episode_row.nunique()), positives=int(selected.progress.sum()),
                           positive_rate=float(selected.progress.mean()), success_h5=int(selected.success_h5.sum()),
                           goal_advance_h5=int(selected.goal_advance_h5.sum()), grasp_lift_h5=int(selected.grasp_lift_h5.sum()),
                           progress_without_success_h5=int((selected.progress & ~selected.success_h5).sum()),
                           excluded_warmup=int((block["query"] < 7).sum()),
                           excluded_tail_after_warmup=int(((block["query"] >= 7) &
                               ((block["query"]+HORIZON)*10 >= index.loc[block.episode_row, "max_steps"].to_numpy())).sum()),
                           already_complete=int(block.already_complete.sum())))
    pd.DataFrame(counts).to_csv(output / "label_counts.csv", index=False)
    first_q = index.loc[all_rows.episode_row, "first_q_freeze_back_half_k4"].to_numpy()
    alarms = all_rows[all_rows["query"].to_numpy() == first_q].copy()
    alarms = alarms.merge(frame[["episode_row", "query", "moe", "moe_unseen_task"]],
                          on=["episode_row", "query"], how="left", validate="one_to_one")
    alarms.to_csv(output / "first_alarm_physics.csv", index=False)
    # Every known alarm is retained, even if its future window is unsupported.
    if len(alarms) != int((episodes.first_q_freeze_back_half_k4 >= 0).sum()):
        raise ValueError("an alarm checkpoint was lost during physical labeling")
    alarm_summary = []
    for outcome, block in alarms.groupby("success"):
        supported = block[block.eligible]
        alarm_summary.append(dict(final_success=bool(outcome), alarms=len(block), supported=len(supported),
                                  progress_h5=int(supported.progress.sum()), success_h5=int(supported.success_h5.sum()),
                                  goal_advance_h5=int(supported.goal_advance_h5.sum()),
                                  grasp_lift_h5=int(supported.grasp_lift_h5.sum())))
    write_json(output / "alarm_summary.json", alarm_summary)


def verify_online(source, hub, episodes, frame, bundle):
    samples = json.loads((source / "raw_verification.json").read_text())["samples"]
    saved = frame.set_index(["episode_row", "query"])
    count, raw_queries, max_error = 0, 0, 0.0
    for sample in samples:
        group = episodes[episodes.source_run == sample["source_run"]].sort_values("episode")
        local = int(np.flatnonzero(group.episode.to_numpy() == sample["episode"])[0])
        row = group.iloc[local]
        offset, length = int(group.length.iloc[:local].sum()), int(row.length)
        store = zarr.open_group(str(hub / row.source_run / "server/routes.zarr"), mode="r")
        raw = np.asarray(store["hb_router_probs"][offset:offset+length, :, 9, 1:, :])
        monitor = MoEProgressMonitor(bundle)
        for q, chunk in enumerate(raw):
            prediction = monitor.update(chunk)
            raw_queries += 1
            if q < WINDOW-1:
                assert not prediction["ready"] and prediction["progress_probability"] is None
            elif (row.episode_row, q) in saved.index:
                expected = float(saved.loc[(row.episode_row, q), "moe"])
                max_error = max(max_error, abs(expected-prediction["progress_probability"]))
                np.testing.assert_allclose(prediction["progress_probability"], expected, atol=1e-12, rtol=1e-12)
                count += 1
    return dict(raw_episodes=len(samples), raw_queries=raw_queries, probabilities_checked=count,
                max_error=max_error, physical_inputs_to_online_model=False)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    if not args.render_only:
        episodes, original, x, hashes = load_inputs(args.source)
        labels = build_labels(episodes, output / "physical", output)
        frame = original.drop(columns=["original_suite_clock", "original_suite_moe_clock", "half_k4_alarm"])
        frame = frame.merge(labels, on=["episode_row", "query"], how="left", validate="one_to_one", sort=False)
        np.testing.assert_array_equal(frame[["episode_row", "query"]], original[["episode_row", "query"]])
        if frame.eligible.isna().any():
            raise ValueError("missing physical labels")
        keep = frame.eligible.to_numpy(bool)
        frame = frame[keep].reset_index(drop=True)
        local = select_local_features(x[keep])
        bundle = fit_models(episodes, frame, local, output)
        evaluate(frame, episodes, output)
        describe_labels(labels, frame, episodes, output)
        raw = verify_online(args.source, args.hub, episodes, frame, bundle)
        write_json(output / "raw_verification.json", raw)
        frame.to_csv(output / "predictions.csv.gz", index=False, float_format="%.12g")
        write_json(output / "summary.json", dict(source_hashes=hashes, horizon_chunks=HORIZON, window=WINDOW,
                   feature_names=LOCAL_NAMES, model_params=MODEL_PARAMS, source_episodes=len(episodes),
                   eligible_queries=len(frame), task_folds_disjoint=True, future_or_physical_online_inputs=False,
                   main_metrics=json_records(pd.read_csv(output / "metrics.csv").query(
                       "split == 'test_unseen_init' and suite == 'all'")),
                   true_trap_and_escape_not_identified=True))
    from .progress_report import report
    report(output)
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".log"}
    for path in [HERE / "PROGRESS_PROTOCOL.md", HERE / "extract_progress.sh", *sorted((HERE / "probability").glob("*.py")),
                 *sorted((HERE / "tests").glob("*.py"))]:
        manifest[str(path.relative_to(HERE))] = sha256(path)
    write_json(output / "artifact_manifest.json", manifest)
    print(f"completed {output / 'REPORT.zh.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--hub", type=Path, default=HERE.parent / "VLA_MUI_HUB")
    parser.add_argument("--output", type=Path, default=HERE / "results_progress")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
