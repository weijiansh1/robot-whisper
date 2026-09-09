"""Evaluate frozen LIBERO-10 scores before a shared termination boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from kmeans_reference import OUTPUT as PARENT, load_npz
from core import conformal_threshold, digest, first_alarm, trajectory_peak, write_json

OUTPUT = PARENT.parent / "round12_common_horizon"
METHODS = ("knn20", "c32_centroid_distance", "c4_centroid_distance", "c4_assigned_radius")
MODES = ("frozen_full", "prefix_calibration")
ALPHA = .05
CHUNK = 10
WARMUP = 7


def last_query_before(action_steps):
    steps = np.asarray(action_steps, dtype=np.int64)
    if np.any(steps <= 0):
        raise ValueError("termination must follow at least one action")
    return (steps - 1) // CHUNK


def prefix_scores(scores, last_query):
    scores = np.asarray(scores)
    cutoffs = np.broadcast_to(np.asarray(last_query, dtype=np.int64), (len(scores),))
    if np.any(cutoffs < 0) or np.any(cutoffs >= scores.shape[1]):
        raise ValueError("query cutoff outside recorded array")
    mask = np.arange(scores.shape[1])[None] <= cutoffs[:, None]
    clipped = np.where(mask, scores, np.nan)
    return trajectory_peak(clipped), scores[np.arange(len(scores)), cutoffs], clipped


def grouped_threshold(scores, labels, frame, last_query=None):
    if last_query is not None:
        scores = scores[:, :last_query + 1]
    success = np.asarray(labels) == 0
    peaks = trajectory_peak(scores[success])
    units = frame.loc[success, ["task", "init_state_id"]].copy()
    units["peak"] = peaks
    maxima = units.groupby(["task", "init_state_id"], sort=True).peak.max().to_numpy()
    threshold, rank = conformal_threshold(maxima, ALPHA)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("no finite positive calibrated threshold")
    return threshold, dict(units=len(maxima), rank=rank, successful_episodes=int(success.sum()),
                           exceedances=int((maxima > threshold).sum()))


def ranking(labels, scores):
    if np.unique(labels).size < 2:
        return np.nan, np.nan
    finite = np.isfinite(scores)
    if not finite.all():
        raise ValueError("ranking cannot silently drop missing observations")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def metrics(labels, peak, endpoint, thresholds, first):
    alarm = first >= 0
    tp, fp = int((alarm & labels).sum()), int((alarm & ~labels).sum())
    failures, successes = int(labels.sum()), int((~labels).sum())
    auc, ap = ranking(labels, peak / thresholds)
    raw_auc, raw_ap = ranking(labels, peak)
    endpoint_auc, _ = ranking(labels, endpoint / thresholds)
    raw_endpoint_auc, _ = ranking(labels, endpoint)
    return dict(episodes=len(labels), failures=failures, successes=successes, tp=tp, fp=fp,
                fn=failures - tp, tn=successes - fp,
                recall=tp / failures if failures else np.nan,
                fpr=fp / successes if successes else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                auc_peak_ratio=auc, ap_peak_ratio=ap, auc_peak_raw=raw_auc, ap_peak_raw=raw_ap,
                auc_endpoint_ratio=endpoint_auc, auc_endpoint_raw=raw_endpoint_auc)


def plot_comparison(output, table, last_fixed):
    windows = ("full", "task_common", last_fixed)
    labels = ("Full trajectory", "Task-common prefix", "All at 130 actions")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for method, color, name in (("knn20", "#555555", "kNN-20"),
                                ("c4_assigned_radius", "#168579", "C4 scaled")):
        for mode, style, marker in (("frozen_full", "-", "o"), ("prefix_calibration", "--", "s")):
            rows = table.loc[table.method.eq(method) & table.threshold_mode.eq(mode)].set_index("window").loc[list(windows)]
            for ax, field in zip(axes, ("auc_peak_ratio", "recall", "precision")):
                ax.plot(np.arange(3), 100 * rows[field], linestyle=style, marker=marker, color=color,
                        label=f"{name}, {'frozen' if mode == 'frozen_full' else 'prefix'} threshold")
    for ax, title in zip(axes, ("AUROC (%)", "Failure recall (%)", "Alarm precision (%)")):
        ax.set(xticks=np.arange(3), xticklabels=labels, ylabel=title, ylim=(0, 102))
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(alpha=.2)
    axes[0].axhline(50, color="#999999", linestyle=":", linewidth=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("LIBERO-10 | 8,000 held-out trajectories | shared pre-termination windows")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"length_control.{suffix}", dpi=180)
    plt.close(fig)


def evaluate(parent, output):
    if output.exists():
        raise FileExistsError(output)
    manifest_path = parent / "sealed_manifest.json"
    sealed = json.loads(manifest_path.read_text())
    verified = json.loads((parent / "verification.json").read_text())
    reported = json.loads((parent / "report_verification.json").read_text())
    assert verified["passed"] and reported["passed"]
    assert verified["sealed_manifest_sha256"] == digest(manifest_path)
    inputs = {name: digest(parent / name) for name in
              ("sealed_manifest.json", "verification.json", "report_verification.json")}

    def checked(name):
        path = parent / name
        expected = sealed["artifacts"].get(name, reported["artifacts"].get(name))
        actual = digest(path)
        assert expected is not None and actual == expected, name
        inputs[name] = actual
        return path

    columns = ["global_row", "source", "episode", "run_id", "cohort", "suite", "task", "fold",
               "failure", "actual_action_steps", "length", "init_state_id", "noise_seed", "checkpoint"]
    master = pd.read_csv(checked("trajectory_results.csv"), usecols=columns).set_index("global_row", drop=False)
    np.testing.assert_array_equal(master.global_row, np.arange(32000))
    frame = master.loc[master.suite.eq("libero_long")].copy().reset_index(drop=True)
    assert len(frame) == 8000 and frame.failure.sum() == 541
    assert not frame.duplicated(["source", "episode"]).any()
    np.testing.assert_array_equal(frame.length, (frame.actual_action_steps + CHUNK - 1) // CHUNK)
    positions = pd.Series(np.arange(len(frame)), index=frame.global_row)
    task_cutoffs = frame.groupby("task").agg(episodes=("episode", "size"), failures=("failure", "sum"),
        first_termination_action=("actual_action_steps", "min"), last_termination_action=("actual_action_steps", "max"))
    task_cutoffs["last_query"] = last_query_before(task_cutoffs.first_termination_action)
    task_cutoffs["last_observed_action"] = CHUNK * task_cutoffs.last_query
    assert len(task_cutoffs) == 10 and task_cutoffs.episodes.eq(800).all()
    for task, row in task_cutoffs.iterrows():
        first = frame.loc[frame.task.eq(task) & frame.actual_action_steps.eq(row.first_termination_action)]
        assert not first.failure.any(), task
    last_common = int(last_query_before(frame.actual_action_steps.min()))
    assert last_common == 13
    windows = ("full", "task_common") + tuple(f"fixed_q{q:02}" for q in range(WARMUP, last_common + 1))
    limits = np.stack([frame.length.to_numpy() - 1, frame.task.map(task_cutoffs.last_query).to_numpy(),
                      *[np.full(len(frame), q, dtype=int) for q in range(WARMUP, last_common + 1)]])
    assert ((limits * CHUNK) < frame.actual_action_steps.to_numpy()[None]).all()
    for task, ids in frame.groupby("task").groups.items():
        assert np.unique(limits[1, ids]).size == 1
    assert (limits >= WARMUP).all()
    assert (limits < frame.length.to_numpy()[None]).all()

    peaks = np.full((len(windows), len(METHODS), len(frame)), np.nan, dtype=np.float64)
    endpoints = np.full_like(peaks, np.nan)
    thresholds = np.full((len(windows), len(MODES), len(METHODS), len(frame)), np.nan, dtype=np.float64)
    firsts = np.full(thresholds.shape, -1, dtype=np.int16)
    ownership = np.zeros(len(frame), dtype=np.int16)
    calibration_records = []
    for info in sealed["folds"]:
        if info["suite"] != "libero_long":
            continue
        data = load_npz(checked(f"predictions/{info['fold']}.npz"))
        ai = int(np.flatnonzero(np.isclose(data["alphas"], ALPHA))[0])
        mis = [list(data["methods"]).index(method) for method in METHODS]
        rows, cal_rows, ref_rows = (data[key] for key in ("test_rows", "calibration_rows", "reference_rows"))
        ids = positions.loc[rows].to_numpy()
        test, cal = master.loc[rows], master.loc[cal_rows].reset_index(drop=True)
        assert not set(test.task) & set(master.loc[np.r_[cal_rows, ref_rows], "task"])
        assert set(cal.run_id) == {"right-50x8-20260903"}
        assert master.loc[np.r_[rows, cal_rows, ref_rows], "checkpoint"].nunique() == 1
        np.testing.assert_array_equal(data["calibration_labels"].astype(bool), cal.failure)
        np.testing.assert_array_equal(data["reference_labels"].astype(bool), master.loc[ref_rows, "failure"])
        score, cal_score = data["scores"][mis], data["calibration_scores"][mis]
        valid = (np.arange(52)[None] >= WARMUP) & (np.arange(52)[None] < test.length.to_numpy()[:, None])
        cal_valid = (np.arange(52)[None] >= WARMUP) & (np.arange(52)[None] < cal.length.to_numpy()[:, None])
        for mi, method in enumerate(METHODS):
            np.testing.assert_array_equal(np.isfinite(score[mi]), valid)
            np.testing.assert_array_equal(np.isfinite(cal_score[mi]), cal_valid)
            original_tau = float(data["thresholds"][1, ai, mis[mi]])
            rebuilt_tau, _ = grouped_threshold(cal_score[mi], data["calibration_labels"], cal)
            assert rebuilt_tau == original_tau
            original_first = data["first"][1, ai, mis[mi]]
            np.testing.assert_array_equal(first_alarm(score[mi], original_tau), original_first)
            prefix_thresholds = {}
            for wi, window in enumerate(windows):
                cutoff = limits[wi, ids]
                peak, endpoint, clipped = prefix_scores(score[mi], cutoff)
                assert np.isfinite(peak).all() and np.isfinite(endpoint).all()
                peaks[wi, mi, ids], endpoints[wi, mi, ids] = peak, endpoint
                thresholds[wi, 0, mi, ids] = original_tau
                firsts[wi, 0, mi, ids] = first_alarm(clipped, original_tau)
                expected_first = np.where((original_first >= 0) & (original_first <= cutoff), original_first, -1)
                np.testing.assert_array_equal(firsts[wi, 0, mi, ids], expected_first)
                if window == "full":
                    thresholds[wi, 1, mi, ids] = original_tau
                else:
                    for last in np.unique(cutoff):
                        last = int(last)
                        if last not in prefix_thresholds:
                            tau, details = grouped_threshold(cal_score[mi], data["calibration_labels"], cal, last)
                            assert tau <= original_tau
                            prefix_thresholds[last] = tau
                            calibration_records.append(dict(fold=info["fold"], method=method, last_query=last,
                                action_steps=CHUNK * last, threshold=tau, original_threshold=original_tau, **details))
                        thresholds[wi, 1, mi, ids[cutoff == last]] = prefix_thresholds[last]
                firsts[wi, 1, mi, ids] = first_alarm(clipped, thresholds[wi, 1, mi, ids][:, None])
                window_first = firsts[wi, :, mi][:, ids]
                assert np.all((window_first < 0) | (window_first <= cutoff[None]))
        ownership[ids] += 1
        print(f"CHECKED {info['fold']}: {len(ids)} tests", flush=True)
    np.testing.assert_array_equal(ownership, 1)
    assert np.isfinite(peaks).all() and np.isfinite(endpoints).all() and np.isfinite(thresholds).all()
    assert (thresholds > 0).all()
    labels = frame.failure.to_numpy()
    baseline = {"knn20": (430, 159), "c32_centroid_distance": (450, 292),
                "c4_centroid_distance": (390, 70), "c4_assigned_radius": (405, 82)}
    previous = pd.read_csv(checked("suite_metrics.csv"))
    for mi, method in enumerate(METHODS):
        alarm = firsts[0, 0, mi] >= 0
        assert (int((alarm & labels).sum()), int((alarm & ~labels).sum())) == baseline[method]
        before = previous.loc[previous.suite.eq("libero_long") & previous.cohort.eq("all") &
            previous.calibration.eq("task_init") & previous.alpha.eq(ALPHA) & previous.method.eq(method)].iloc[0]
        assert (before.tp, before.fp) == baseline[method]

    populations = [("all", "all", np.arange(len(frame)))]
    populations += [(level, name, np.asarray(ids)) for level in ("task", "fold", "cohort")
                    for name, ids in frame.groupby(level).groups.items()]
    records = []
    clocks = []
    for wi, window in enumerate(windows):
        clock = frame.actual_action_steps.to_numpy() if wi == 0 else CHUNK * limits[wi]
        for level, name, ids in populations:
            clock_auc, clock_ap = ranking(labels[ids], clock[ids])
            clocks.append(dict(window=window, level=level, group=name, episodes=len(ids),
                               auc=clock_auc, ap=clock_ap))
            if wi > 1 or (wi == 1 and level == "task"):
                assert clock_auc == .5
            for ki, mode in enumerate(MODES):
                for mi, method in enumerate(METHODS):
                    record = metrics(labels[ids], peaks[wi, mi, ids], endpoints[wi, mi, ids],
                                     thresholds[wi, ki, mi, ids], firsts[wi, ki, mi, ids])
                    records.append(dict(window=window, threshold_mode=mode, method=method, alpha=ALPHA,
                                        level=level, group=name, **record))
    grouped = pd.DataFrame(records)
    overall = grouped.loc[grouped.level.eq("all")].copy()
    macro = grouped.loc[grouped.level.eq("task")].groupby(["window", "threshold_mode", "method"]).agg(
        task_macro_auc=("auc_peak_ratio", "mean"), task_macro_endpoint_auc=("auc_endpoint_ratio", "mean"),
        task_macro_ap=("ap_peak_ratio", "mean"), evaluated_tasks=("group", "size")).reset_index()
    overall = overall.merge(macro, on=["window", "threshold_mode", "method"], validate="one_to_one")
    windows_frame = pd.DataFrame([dict(window=window, min_last_query=int(limits[wi].min()),
        max_last_query=int(limits[wi].max()), min_last_action=int(CHUNK * limits[wi].min()),
        max_last_action=int(CHUNK * limits[wi].max()), episodes=len(frame),
        any_query_at_or_after_own_termination=bool(((CHUNK * limits[wi]) >= frame.actual_action_steps.to_numpy()).any()),
        any_success_terminated_globally_by_last_query=bool(CHUNK * limits[wi].max() >= frame.loc[~labels, "actual_action_steps"].min()))
        for wi, window in enumerate(windows)])
    false_ids = np.flatnonzero((firsts[0, 0, METHODS.index("c4_assigned_radius")] >= 0) & ~labels)
    false_alarms = frame.loc[false_ids].copy()
    assert len(false_alarms) == 82
    for wi, window in enumerate(windows):
        if window not in ("full", "task_common", windows[-1]):
            continue
        false_alarms[f"{window}_last_query"] = limits[wi, false_ids]
        for ki, mode in enumerate(MODES):
            mi = METHODS.index("c4_assigned_radius")
            false_alarms[f"{window}_{mode}_first_query"] = firsts[wi, ki, mi, false_ids]

    output.mkdir(parents=True)
    overall.to_csv(output / "metrics.csv", index=False)
    grouped.to_csv(output / "group_metrics.csv", index=False)
    pd.DataFrame(clocks).to_csv(output / "clock_metrics.csv", index=False)
    task_cutoffs.to_csv(output / "task_cutoffs.csv")
    windows_frame.to_csv(output / "windows.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(output / "calibration_thresholds.csv", index=False)
    false_alarms.to_csv(output / "original_82_false_alarms.csv", index=False)
    frame.to_csv(output / "index.csv", index=False)
    np.savez_compressed(output / "window_results.npz", global_rows=frame.global_row.to_numpy(),
        windows=np.asarray(windows), methods=np.asarray(METHODS), threshold_modes=np.asarray(MODES),
        last_queries=limits, peak_scores=peaks, endpoint_scores=endpoints, thresholds=thresholds,
        first=firsts, failure=labels)
    plot_comparison(output, overall, windows[-1])
    write_json(output / "verification.json", dict(passed=True, suite="libero_long", episodes=8000,
        successes=7459, failures=541, reference_refitted=False, new_rollouts=False, alpha=ALPHA,
        unique_test_coverage=True, original_false_alarms_reproduced=82, original_detections_reproduced=405,
        all_fixed_windows_keep_all_episodes=True, strict_pretermination_queries=True,
        earliest_success_action=140, last_universal_query=last_common, last_universal_action=CHUNK * last_common,
        retrospectively_selected_horizons=True, test_labels_used_for_threshold_fitting=False,
        source_sha256=digest(Path(__file__)),
        protocol_sha256=digest(Path(__file__).with_name("COMMON_HORIZON_PROTOCOL_ZH.md")),
        inputs=inputs, artifacts={path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}))
    chosen = overall.loc[overall.window.isin(("full", "task_common", windows[-1])) &
                         overall.method.isin(("knn20", "c4_assigned_radius"))]
    print(chosen[["window", "threshold_mode", "method", "auc_peak_ratio", "task_macro_auc",
                  "tp", "fp", "recall", "fpr", "precision"]].to_string(index=False))
    print("COMMON HORIZON EVALUATION VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    evaluate(arguments.parent.resolve(), arguments.output.resolve())
