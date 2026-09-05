#!/usr/bin/env python3
"""Evaluate sealed strict single-rollout alarms after outcome reveal."""

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
from single_rollout_monitor import DETECTORS, PRIMARY_DETECTOR, TYPED_STATES


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results/online_single_rollout"
DEFAULT_V1 = HERE / "results/online_multihead_hub"
DEFAULT_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
DEFAULT_ONSETS = DEFAULT_V1 / "physical_proxy_onsets.csv"
PROTOCOL = HERE / "ONLINE_SINGLE_ROLLOUT_PROTOCOL.md"
PRIMARY_QUANTILE = 0.95
SEED = 20260904


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--v1-result-dir", type=Path, default=DEFAULT_V1)
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
    if manifest.get("schema") != "himoe.single_rollout.manifest.v1":
        raise ValueError("unknown single-rollout manifest schema")
    required_false = (
        "runtime_cross_rollout_access",
        "runtime_future_access",
        "runtime_final_length_access",
        "duration_detector_feature",
        "termination_or_censoring_filter",
        "sim_state_used",
    )
    if any(manifest.get(name) for name in required_false):
        raise ValueError("single-rollout seal violates the runtime contract")
    if manifest.get("labels_used") != [] or not manifest.get("query_causal"):
        raise ValueError("scorer was not sealed outcome-blind and causal")
    artifacts = manifest["artifacts"]
    paths = {
        "protocol_sha256": PROTOCOL,
        "scorer_sha256": HERE / "online_single_rollout_alarm.py",
        "runtime_monitor_sha256": HERE / "single_rollout_monitor.py",
        "route_feature_cache_sha256": DEFAULT_V1 / "unlabeled_query_features.npz",
        "physical_feature_cache_sha256": HERE
        / "results/online_closed_loop_v2/unlabeled_physical_features.npz",
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
    v1: dict[str, np.ndarray],
    labels: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    tasks = labels["task"].to_numpy(dtype=str)
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    lengths = sealed["length"].astype(int)
    comparisons = (
        (PRIMARY_DETECTOR, sealed, "instant_route", "stream/instant_route"),
        (PRIMARY_DETECTOR, sealed, "persistent_route", "stream/persistent_route"),
        (PRIMARY_DETECTOR, v1, "multi_max", "v1/multi_max"),
        ("persistent_route", sealed, "instant_route", "stream/instant_route"),
        ("persistent_route", v1, "multi_max", "v1/multi_max"),
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for candidate_name, baseline_source, baseline_name, baseline_label in comparisons:
        candidate_alarm, candidate_first = common.detector_alarm(
            sealed, candidate_name, PRIMARY_QUANTILE
        )
        baseline_alarm, baseline_first = common.detector_alarm(
            baseline_source, baseline_name, PRIMARY_QUANTILE
        )
        candidate_lead = lengths - 1 - candidate_first
        baseline_lead = lengths - 1 - baseline_first
        metrics = [
            ("failure_recall", candidate_alarm & failure, baseline_alarm & failure, failure),
            ("success_fpr", candidate_alarm & success, baseline_alarm & success, success),
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
            low, high, p = common.bootstrap_difference(
                tasks,
                candidate_num,
                baseline_num,
                denominator,
                draws,
                rng,
            )
            rows.append(
                {
                    "candidate": f"stream/{candidate_name}",
                    "baseline": baseline_label,
                    "metric": metric,
                    "quantile": PRIMARY_QUANTILE,
                    "delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "two_sided_p": p,
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
    aligned = source.merge(index, on=["task", "episode"], how="left", validate="one_to_one")
    if aligned["sealed_row"].isna().any():
        raise ValueError("physical onset table does not align to streaming cohort")
    return aligned


def onset_tables(
    sealed: dict[str, np.ndarray], onsets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = onsets[onsets["onset_query"] >= 0].copy()
    rows: list[dict[str, Any]] = []
    for behavior in ("all", *sorted(usable["primary_behavior"].unique())):
        subset = usable if behavior == "all" else usable[usable["primary_behavior"] == behavior]
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
                    "onset_n": int(len(subset)),
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
                        float(np.median(relative[alerted])) if alerted.any() else np.nan
                    ),
                }
            )

    primary_alarm, primary_first = common.detector_alarm(
        sealed, PRIMARY_DETECTOR, PRIMARY_QUANTILE
    )
    winner = sealed["winning_state"].astype(int)
    composition: list[dict[str, Any]] = []
    for behavior, subset in usable.groupby("primary_behavior", sort=True):
        index = subset["sealed_row"].to_numpy(dtype=int)
        onset = subset["onset_query"].to_numpy(dtype=int)
        first = primary_first[index]
        timely = primary_alarm[index] & (first - onset >= -2) & (first - onset <= 2)
        timely_rows = index[timely]
        timely_query = first[timely]
        winning = winner[timely_rows, timely_query] if len(timely_rows) else np.empty(0, int)
        for state_position, state in enumerate(TYPED_STATES):
            composition.append(
                {
                    "primary_behavior": behavior,
                    "state": state,
                    "onset_n": int(len(subset)),
                    "timely_primary_alarm_n": int(timely.sum()),
                    "winning_state_n": int(np.sum(winning == state_position)),
                    "winning_state_rate_among_timely": common.ratio(
                        np.sum(winning == state_position), len(winning)
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(composition)


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


def main() -> None:
    args = parse_args()
    manifest = verify_seal(args.result_dir)
    sealed = common.load_scores(
        args.result_dir / "sealed_online_scores.npz",
        "himoe.single_rollout.sealed.v1",
    )
    labels = common.join_labels(sealed, args.labels)
    v1_raw = common.load_scores(
        args.v1_result_dir / "sealed_online_scores.npz",
        "himoe.online_multihead.sealed.v1",
    )
    v1 = common.align_scores(common.build_index(sealed), v1_raw)
    metrics, by_task, by_suite = common.outcome_tables(
        sealed, labels, args.bootstrap, args.seed
    )
    comparisons = comparison_table(
        sealed, v1, labels, args.bootstrap, args.seed + 1
    )
    onsets = aligned_onsets(sealed, args.onsets)
    onset_metrics, onset_composition = onset_tables(sealed, onsets)
    subtypes = subtype_table(sealed, labels)

    metrics.to_csv(args.result_dir / "outcome_metrics.csv", index=False)
    by_task.to_csv(args.result_dir / "outcome_metrics_by_task.csv", index=False)
    by_suite.to_csv(args.result_dir / "outcome_metrics_by_suite.csv", index=False)
    comparisons.to_csv(args.result_dir / "paired_detector_comparisons.csv", index=False)
    onset_metrics.to_csv(args.result_dir / "onset_event_metrics.csv", index=False)
    onset_composition.to_csv(
        args.result_dir / "onset_winning_state_composition.csv", index=False
    )
    subtypes.to_csv(args.result_dir / "failure_subtype_alarm_rates.csv", index=False)

    q95 = metrics[np.isclose(metrics["quantile"], PRIMARY_QUANTILE)]
    primary = q95[q95["detector"] == PRIMARY_DETECTOR].iloc[0]
    persistent = q95[q95["detector"] == "persistent_route"].iloc[0]
    summary = {
        "schema": "himoe.single_rollout.evaluation.v1",
        "manifest_sha256": sha256(args.result_dir / "sealed_manifest.json"),
        "labels_sha256": sha256(args.labels),
        "onsets_sha256": sha256(args.onsets),
        "episodes": int(len(labels)),
        "failures": int(labels["failure"].sum()),
        "successes": int((~labels["failure"]).sum()),
        "primary_q95": plain(primary.to_dict()),
        "best_predeclared_q95": plain(persistent.to_dict()),
        "primary_passes_decision_rule": False,
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
        "single-rollout evaluation: "
        f"primary recall={primary.failure_recall:.4f} FPR={primary.success_fpr:.4f}; "
        f"persistent-route recall={persistent.failure_recall:.4f} "
        f"FPR={persistent.success_fpr:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
