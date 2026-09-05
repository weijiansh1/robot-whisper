#!/usr/bin/env python3
"""Evaluate sealed route-derivative alarms after outcome reveal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_online_closed_loop_v2 as common
import online_multihead_alarm as v1
from route_derivative_monitor import BRANCHES, PRIMARY_DETECTOR


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results/online_route_derivative"
DEFAULT_BASELINE = HERE / "results/online_single_rollout"
DEFAULT_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
DEFAULT_ONSETS = HERE / "results/online_multihead_hub/physical_proxy_onsets.csv"
PROTOCOL = HERE / "ONLINE_ROUTE_DERIVATIVE_PROTOCOL.md"
PRIMARY_QUANTILE = 0.95
BASELINE_DETECTOR = "persistent_level"
SEED = 20260904


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify_seal(result_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (result_dir / "sealed_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "himoe.route_derivative.manifest.v1":
        raise ValueError("unknown route-derivative manifest schema")
    forbidden = (
        "runtime_cross_rollout_access",
        "runtime_future_access",
        "runtime_final_length_access",
        "duration_detector_feature",
        "termination_or_censoring_filter",
        "state_or_action_used",
        "sim_state_used",
    )
    if any(manifest.get(name) for name in forbidden):
        raise ValueError("route-derivative seal violates the runtime contract")
    if manifest.get("labels_used") != [] or not manifest.get("query_causal"):
        raise ValueError("scorer was not sealed outcome-blind and causal")
    artifacts = manifest["artifacts"]
    paths = {
        "protocol_sha256": PROTOCOL,
        "scorer_sha256": HERE / "online_route_derivative_alarm.py",
        "runtime_monitor_sha256": HERE / "route_derivative_monitor.py",
        "route_feature_cache_sha256": HERE
        / "results/online_multihead_hub/unlabeled_query_features.npz",
        "sealed_scores_sha256": result_dir / "sealed_online_scores.npz",
        "thresholds_sha256": result_dir / "unlabeled_thresholds.csv",
        "episode_alarms_sha256": result_dir / "sealed_episode_alarms.csv",
        "deployment_profiles_sha256": result_dir / "deployment_profiles.npz",
    }
    for key, path in paths.items():
        if sha256(path) != artifacts[key]:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def comparison_table(
    sealed: dict[str, np.ndarray],
    labels: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].to_numpy(dtype=str)
    lengths = sealed["length"].astype(int)
    baseline_alarm, baseline_first = common.detector_alarm(
        sealed, BASELINE_DETECTOR, PRIMARY_QUANTILE
    )
    baseline_lead = lengths - 1 - baseline_first
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for candidate in sealed["detector_names"].astype(str):
        if candidate == BASELINE_DETECTOR:
            continue
        candidate_alarm, candidate_first = common.detector_alarm(
            sealed, candidate, PRIMARY_QUANTILE
        )
        candidate_lead = lengths - 1 - candidate_first
        metrics = [
            (
                "failure_recall",
                candidate_alarm & failure,
                baseline_alarm & failure,
                failure,
            ),
            (
                "success_fpr",
                candidate_alarm & success,
                baseline_alarm & success,
                success,
            ),
        ]
        for early in common.EARLY_LEADS:
            metrics.append(
                (
                    f"early{early}_recall",
                    candidate_alarm & failure & (candidate_lead >= early),
                    baseline_alarm & failure & (baseline_lead >= early),
                    failure,
                )
            )
        for metric, candidate_num, baseline_num, denominator in metrics:
            point = common.ratio(
                candidate_num.sum() - baseline_num.sum(), denominator.sum()
            )
            low, high, p_value = common.bootstrap_difference(
                tasks,
                candidate_num,
                baseline_num,
                denominator,
                draws,
                rng,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": BASELINE_DETECTOR,
                    "metric": metric,
                    "quantile": PRIMARY_QUANTILE,
                    "delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "two_sided_p": p_value,
                    "candidate_only": int(np.sum(candidate_num & ~baseline_num)),
                    "baseline_only": int(np.sum(baseline_num & ~candidate_num)),
                    "bootstrap_unit": "task",
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def aligned_onsets(
    sealed: dict[str, np.ndarray], onset_path: Path
) -> pd.DataFrame:
    index = common.build_index(sealed)[["sealed_row", "task", "episode"]]
    source = pd.read_csv(onset_path).drop(columns="sealed_row")
    aligned = source.merge(
        index, on=["task", "episode"], how="left", validate="one_to_one"
    )
    if aligned["sealed_row"].isna().any():
        raise ValueError("physical onset table does not align to derivative cohort")
    return aligned


def onset_table(
    sealed: dict[str, np.ndarray], onsets: pd.DataFrame
) -> pd.DataFrame:
    usable = onsets[onsets["onset_query"] >= 0].copy()
    rows: list[dict[str, Any]] = []
    for behavior in ("all", *sorted(usable["primary_behavior"].unique())):
        subset = (
            usable
            if behavior == "all"
            else usable[usable["primary_behavior"] == behavior]
        )
        index = subset["sealed_row"].to_numpy(dtype=int)
        onset = subset["onset_query"].to_numpy(dtype=int)
        for detector in sealed["detector_names"].astype(str):
            alarm, first_all = common.detector_alarm(
                sealed, detector, PRIMARY_QUANTILE
            )
            first = first_all[index]
            relative = first - onset
            alerted = alarm[index]
            rows.append(
                {
                    "primary_behavior": behavior,
                    "detector": detector,
                    "quantile": PRIMARY_QUANTILE,
                    "onset_n": int(len(index)),
                    "alarm_any_rate": float(alerted.mean()),
                    "alarm_before_onset_rate": float(
                        np.mean(alerted & (relative <= -1))
                    ),
                    "strict_precursor_m4_m1_rate": float(
                        np.mean(alerted & (relative >= -4) & (relative <= -1))
                    ),
                    "timely_m2_p2_rate": float(
                        np.mean(alerted & (relative >= -2) & (relative <= 2))
                    ),
                    "reaction_p0_p4_rate": float(
                        np.mean(alerted & (relative >= 0) & (relative <= 4))
                    ),
                    "first_alarm_relative_median": (
                        float(np.median(relative[alerted]))
                        if alerted.any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def subtype_table(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame
) -> pd.DataFrame:
    failure = labels[labels["failure"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for detector in sealed["detector_names"].astype(str):
        alarm, first = common.detector_alarm(sealed, detector, PRIMARY_QUANTILE)
        for behavior, subset in failure.groupby("primary_behavior", sort=True):
            index = subset["sealed_row"].to_numpy(dtype=int)
            rows.append(
                {
                    "primary_behavior": behavior,
                    "detector": detector,
                    "quantile": PRIMARY_QUANTILE,
                    "failure_n": int(len(index)),
                    "alarm_n": int(alarm[index].sum()),
                    "alarm_rate": float(alarm[index].mean()),
                    "first_alarm_query_median": (
                        float(np.median(first[index][alarm[index]]))
                        if alarm[index].any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def branch_table(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame
) -> pd.DataFrame:
    alarm, first = common.detector_alarm(
        sealed, PRIMARY_DETECTOR, PRIMARY_QUANTILE
    )
    failure = labels["failure"].to_numpy(dtype=bool)
    branch = sealed["winning_branch"].astype(int)
    head = sealed["winning_head"].astype(int)
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, np.ndarray]] = [
        ("failure/all", failure),
        ("success/all", ~failure),
    ]
    for behavior in sorted(labels.loc[failure, "primary_behavior"].unique()):
        groups.append(
            (
                f"failure/{behavior}",
                failure
                & (labels["primary_behavior"].to_numpy(dtype=str) == behavior),
            )
        )
    for group, member in groups:
        index = np.flatnonzero(alarm & member)
        query = first[index]
        first_branch = branch[index, query] if len(index) else np.empty(0, int)
        first_head = head[index, query] if len(index) else np.empty(0, int)
        for branch_position, branch_name in enumerate(BRANCHES):
            for head_position, head_name in enumerate(v1.HEADS):
                count = int(
                    np.sum(
                        (first_branch == branch_position)
                        & (first_head == head_position)
                    )
                )
                rows.append(
                    {
                        "group": group,
                        "alarm_n": int(len(index)),
                        "branch": branch_name,
                        "route_head": head_name,
                        "count": count,
                        "rate_among_group_alarms": common.ratio(count, len(index)),
                    }
                )
    return pd.DataFrame(rows)


def assert_baseline_identity(
    sealed: dict[str, np.ndarray], baseline_path: Path
) -> dict[str, Any]:
    baseline = common.load_scores(
        baseline_path / "sealed_online_scores.npz",
        "himoe.single_rollout.sealed.v1",
    )
    baseline = common.align_scores(common.build_index(sealed), baseline)
    new_position = sealed["detector_names"].astype(str).tolist().index(
        BASELINE_DETECTOR
    )
    old_position = baseline["detector_names"].astype(str).tolist().index(
        "persistent_route"
    )
    score_equal = np.array_equal(
        sealed["scores"][:, :, new_position],
        baseline["scores"][:, :, old_position],
        equal_nan=True,
    )
    threshold_equal = np.array_equal(
        sealed["thresholds"][:, new_position],
        baseline["thresholds"][:, old_position],
    )
    alarm_equal = np.array_equal(
        sealed["alarms"][:, :, new_position],
        baseline["alarms"][:, :, old_position],
    )
    if not (score_equal and threshold_equal and alarm_equal):
        raise ValueError("persistent baseline changed in derivative experiment")
    return {
        "score_equal": score_equal,
        "threshold_equal": threshold_equal,
        "alarm_equal": alarm_equal,
    }


def main() -> None:
    args = parse_args()
    manifest = verify_seal(args.result_dir)
    sealed = common.load_scores(
        args.result_dir / "sealed_online_scores.npz",
        "himoe.route_derivative.sealed.v1",
    )
    baseline_identity = assert_baseline_identity(sealed, args.baseline_dir)
    labels = common.join_labels(sealed, args.labels)
    metrics, by_task, by_suite = common.outcome_tables(
        sealed, labels, args.bootstrap, args.seed
    )
    comparisons = comparison_table(
        sealed, labels, args.bootstrap, args.seed + 1
    )
    onsets = aligned_onsets(sealed, args.onsets)
    onset_metrics = onset_table(sealed, onsets)
    subtypes = subtype_table(sealed, labels)
    branches = branch_table(sealed, labels)

    metrics.to_csv(args.result_dir / "outcome_metrics.csv", index=False)
    by_task.to_csv(args.result_dir / "outcome_metrics_by_task.csv", index=False)
    by_suite.to_csv(args.result_dir / "outcome_metrics_by_suite.csv", index=False)
    comparisons.to_csv(
        args.result_dir / "paired_detector_comparisons.csv", index=False
    )
    onset_metrics.to_csv(args.result_dir / "onset_event_metrics.csv", index=False)
    subtypes.to_csv(
        args.result_dir / "failure_subtype_alarm_rates.csv", index=False
    )
    branches.to_csv(
        args.result_dir / "first_alarm_branch_composition.csv", index=False
    )

    q95 = metrics[np.isclose(metrics["quantile"], PRIMARY_QUANTILE)]
    primary = q95[q95["detector"] == PRIMARY_DETECTOR].iloc[0]
    baseline = q95[q95["detector"] == BASELINE_DETECTOR].iloc[0]
    delta = comparisons[
        (comparisons["candidate"] == PRIMARY_DETECTOR)
        & (comparisons["metric"].isin(["failure_recall", "success_fpr"]))
    ].set_index("metric")
    recall_delta = delta.loc["failure_recall"]
    fpr_delta = delta.loc["success_fpr"]
    passes = bool(
        recall_delta["delta"] > 0
        and recall_delta["ci_low"] > 0
        and fpr_delta["delta"] <= 0.01
    )
    summary = {
        "schema": "himoe.route_derivative.evaluation.v1",
        "manifest_sha256": sha256(args.result_dir / "sealed_manifest.json"),
        "labels_sha256": sha256(args.labels),
        "onsets_sha256": sha256(args.onsets),
        "episodes": int(len(labels)),
        "failures": int(labels["failure"].sum()),
        "successes": int((~labels["failure"]).sum()),
        "primary_q95": plain(primary.to_dict()),
        "persistent_baseline_q95": plain(baseline.to_dict()),
        "primary_vs_persistent_recall": plain(recall_delta.to_dict()),
        "primary_vs_persistent_fpr": plain(fpr_delta.to_dict()),
        "primary_passes_frozen_decision_rule": passes,
        "baseline_identity": plain(baseline_identity),
        "exploratory": True,
        "independent_validation": False,
        "bootstrap_unit": "task",
        "bootstrap_draws": args.bootstrap,
        "sealed_manifest": manifest,
    }
    (args.result_dir / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "route-derivative evaluation: "
        f"fusion recall={primary.failure_recall:.4f} "
        f"FPR={primary.success_fpr:.4f}; "
        f"persistent recall={baseline.failure_recall:.4f} "
        f"FPR={baseline.success_fpr:.4f}; "
        f"decision={'pass' if passes else 'fail'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
