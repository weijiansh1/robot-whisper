#!/usr/bin/env python3
"""Evaluate the sealed v2 alarm after outcome labels are revealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results/online_closed_loop_v2"
DEFAULT_V1_RESULT = HERE / "results/online_multihead_hub"
DEFAULT_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
PROTOCOL = HERE / "ONLINE_CLOSED_LOOP_V2_PROTOCOL.md"
SEED = 20260904
PRIMARY_DETECTOR = "delay_confirmed"
PRIMARY_QUANTILE = 0.95
EARLY_LEADS = (2, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--v1-result-dir", type=Path, default=DEFAULT_V1_RESULT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
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
    manifest_path = result_dir / "sealed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "himoe.online_closed_loop.manifest.v2":
        raise ValueError("unknown v2 manifest schema")
    if manifest.get("labels_used") != []:
        raise ValueError("sealed scorer already used outcome labels")
    if not manifest.get("query_causal") or manifest.get(
        "future_queries_used_by_online_update"
    ):
        raise ValueError("sealed scorer does not assert causal updates")
    if manifest.get("held_out_eventual_length_used_by_online_update"):
        raise ValueError("held-out eventual length leaked into online update")

    artifacts = manifest["artifacts"]
    paths = {
        "protocol_sha256": PROTOCOL,
        "scorer_sha256": HERE / "online_closed_loop_alarm_v2.py",
        "v1_feature_code_sha256": HERE / "online_multihead_alarm.py",
        "route_feature_cache_sha256": DEFAULT_V1_RESULT / "unlabeled_query_features.npz",
        "physical_feature_cache_sha256": result_dir / "unlabeled_physical_features.npz",
        "sealed_scores_sha256": result_dir / "sealed_online_scores.npz",
        "thresholds_sha256": result_dir / "unlabeled_thresholds.csv",
        "episode_alarms_sha256": result_dir / "sealed_episode_alarms.csv",
    }
    for key, path in paths.items():
        observed = sha256(path)
        if observed != artifacts[key]:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def load_scores(path: Path, schema: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    if str(values["schema"]) != schema:
        raise ValueError(f"unexpected score schema in {path}")
    return values


def build_index(sealed: dict[str, np.ndarray]) -> pd.DataFrame:
    task_names = sealed["task_names"].astype(str)
    return pd.DataFrame(
        {
            "sealed_row": np.arange(len(sealed["episode"]), dtype=np.int64),
            "task": task_names[sealed["task_index"].astype(int)],
            "episode": sealed["episode"].astype(int),
            "init_state_id": sealed["init_state_id"].astype(int),
            "flow_noise_seed": sealed["flow_noise_seed"].astype(int),
            "sealed_length": sealed["length"].astype(int),
        }
    )


def join_labels(sealed: dict[str, np.ndarray], label_path: Path) -> pd.DataFrame:
    index = build_index(sealed)
    labels = pd.read_csv(label_path)
    if labels.duplicated(["task", "episode"]).any():
        raise ValueError("duplicate label key")
    merged = index.merge(
        labels,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    ).sort_values("sealed_row")
    if len(merged) != len(index) or merged["failure"].isna().any():
        raise ValueError("sealed episodes do not match label table")
    if not np.array_equal(
        merged["sealed_length"].to_numpy(), merged["episode_length"].to_numpy()
    ):
        raise ValueError("episode length changed after sealing")
    for name in ("init_state_id", "flow_noise_seed"):
        if not np.array_equal(
            merged[name].to_numpy(), merged[f"{name}_label"].to_numpy()
        ):
            raise ValueError(f"metadata mismatch after sealing: {name}")
    return merged.reset_index(drop=True)


def align_scores(
    reference_index: pd.DataFrame, other: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    other_index = build_index(other)
    key = ["task", "episode", "init_state_id", "flow_noise_seed", "sealed_length"]
    order = reference_index[key].merge(
        other_index[key + ["sealed_row"]], on=key, how="left", validate="one_to_one"
    )["sealed_row"]
    if order.isna().any():
        raise ValueError("v1 and v2 cohorts do not align")
    take = order.to_numpy(dtype=int)
    output = dict(other)
    for name in ("task_index", "episode", "init_state_id", "flow_noise_seed", "length", "valid"):
        output[name] = other[name][take]
    for name in ("scores", "thresholds", "alarms", "winning_head"):
        if name in other:
            output[name] = other[name][take]
    output["task_names"] = np.asarray(reference_index["task"].unique())
    return output


def first_alarm_query(alarm: np.ndarray) -> np.ndarray:
    alarm = np.asarray(alarm, dtype=bool)
    any_alarm = alarm.any(axis=1)
    first = np.argmax(alarm, axis=1).astype(np.int16)
    first[~any_alarm] = -1
    return first


def detector_alarm(
    sealed: dict[str, np.ndarray], detector: str, quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    names = sealed["detector_names"].astype(str).tolist()
    detector_index = names.index(detector)
    quantile_index = int(np.argmin(np.abs(sealed["quantiles"].astype(float) - quantile)))
    alarm = sealed["alarms"][:, :, detector_index, quantile_index].astype(bool)
    return alarm.any(axis=1), first_alarm_query(alarm)


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def clustered_interval(
    tasks: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    names = np.unique(tasks)
    task_num = np.asarray([numerator[tasks == name].sum() for name in names], dtype=float)
    task_den = np.asarray([denominator[tasks == name].sum() for name in names], dtype=float)
    sample = rng.integers(0, len(names), size=(draws, len(names)))
    values = task_num[sample].sum(axis=1) / np.maximum(task_den[sample].sum(axis=1), 1.0)
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def outcome_tables(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].to_numpy(dtype=str)
    suites = np.asarray([task.split("/", 1)[0] for task in tasks])
    lengths = sealed["length"].astype(int)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for quantile in sealed["quantiles"].astype(float):
        for detector in sealed["detector_names"].astype(str):
            alarm, first = detector_alarm(sealed, detector, quantile)
            lead = lengths - 1 - first
            tp = int(np.sum(alarm & failure))
            fp = int(np.sum(alarm & success))
            row: dict[str, Any] = {
                "detector": detector,
                "quantile": quantile,
                "failure_n": int(failure.sum()),
                "success_n": int(success.sum()),
                "tp": tp,
                "fp": fp,
                "fn": int(np.sum(~alarm & failure)),
                "tn": int(np.sum(~alarm & success)),
                "failure_recall": ratio(tp, failure.sum()),
                "success_fpr": ratio(fp, success.sum()),
                "precision": ratio(tp, tp + fp),
            }
            for metric, numerator, denominator in (
                ("failure_recall", alarm & failure, failure),
                ("success_fpr", alarm & success, success),
                ("precision", alarm & failure, alarm),
            ):
                low, high = clustered_interval(tasks, numerator, denominator, draws, rng)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            for early in EARLY_LEADS:
                detected_early = alarm & (lead >= early)
                metric = f"early{early}_recall"
                row[metric] = ratio(np.sum(detected_early & failure), failure.sum())
                low, high = clustered_interval(
                    tasks, detected_early & failure, failure, draws, rng
                )
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            detected_lead = lead[alarm & failure]
            row["detected_failure_lead_mean"] = (
                float(np.mean(detected_lead)) if len(detected_lead) else float("nan")
            )
            for name, value in zip(
                ("detected_failure_lead_q25", "detected_failure_lead_median", "detected_failure_lead_q75"),
                np.quantile(detected_lead, (0.25, 0.5, 0.75)) if len(detected_lead) else [np.nan] * 3,
            ):
                row[name] = float(value)
            rows.append(row)

            for grouping, values, destination in (
                ("task", tasks, task_rows),
                ("suite", suites, suite_rows),
            ):
                for group in np.unique(values):
                    member = values == group
                    fail_group = failure & member
                    success_group = success & member
                    alarm_group = alarm & member
                    lead_group = lead[alarm & fail_group]
                    item = {
                        "detector": detector,
                        "quantile": quantile,
                        grouping: group,
                        "failure_n": int(fail_group.sum()),
                        "success_n": int(success_group.sum()),
                        "tp": int((alarm & fail_group).sum()),
                        "fp": int((alarm & success_group).sum()),
                        "failure_recall": ratio((alarm & fail_group).sum(), fail_group.sum()),
                        "success_fpr": ratio((alarm & success_group).sum(), success_group.sum()),
                        "precision": ratio((alarm & fail_group).sum(), alarm_group.sum()),
                        "detected_failure_lead_median": (
                            float(np.median(lead_group)) if len(lead_group) else float("nan")
                        ),
                    }
                    for early in EARLY_LEADS:
                        item[f"early{early}_recall"] = ratio(
                            np.sum(alarm & fail_group & (lead >= early)), fail_group.sum()
                        )
                    destination.append(item)
    return pd.DataFrame(rows), pd.DataFrame(task_rows), pd.DataFrame(suite_rows)


def bootstrap_difference(
    tasks: np.ndarray,
    candidate_num: np.ndarray,
    baseline_num: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    names = np.unique(tasks)
    candidate_task = np.asarray(
        [candidate_num[tasks == name].sum() for name in names], dtype=float
    )
    baseline_task = np.asarray(
        [baseline_num[tasks == name].sum() for name in names], dtype=float
    )
    denominator_task = np.asarray(
        [denominator[tasks == name].sum() for name in names], dtype=float
    )
    sample = rng.integers(0, len(names), size=(draws, len(names)))
    den = np.maximum(denominator_task[sample].sum(axis=1), 1.0)
    difference = (
        candidate_task[sample].sum(axis=1) - baseline_task[sample].sum(axis=1)
    ) / den
    low, high = np.quantile(difference, (0.025, 0.975))
    p = min(1.0, 2.0 * min(np.mean(difference <= 0.0), np.mean(difference >= 0.0)))
    return float(low), float(high), float(p)


def comparison_tables(
    v2: dict[str, np.ndarray],
    v1: dict[str, np.ndarray],
    labels: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].to_numpy(dtype=str)
    lengths = v2["length"].astype(int)
    candidate_alarm, candidate_first = detector_alarm(v2, PRIMARY_DETECTOR, PRIMARY_QUANTILE)
    candidate_lead = lengths - 1 - candidate_first
    baselines = (
        ("v2/completion_delay", v2, "completion_delay"),
        ("v2/route_only", v2, "route_only"),
        ("v2/mechanism_only", v2, "mechanism_only"),
        ("v1/multi_max", v1, "multi_max"),
        ("v1/clock", v1, "clock"),
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for label, source, detector in baselines:
        baseline_alarm, baseline_first = detector_alarm(source, detector, PRIMARY_QUANTILE)
        baseline_lead = lengths - 1 - baseline_first
        metrics = (
            ("failure_recall", candidate_alarm & failure, baseline_alarm & failure, failure),
            ("success_fpr", candidate_alarm & success, baseline_alarm & success, success),
            *tuple(
                (
                    f"early{early}_recall",
                    candidate_alarm & failure & (candidate_lead >= early),
                    baseline_alarm & failure & (baseline_lead >= early),
                    failure,
                )
                for early in EARLY_LEADS
            ),
        )
        for metric, candidate_num, baseline_num, denominator in metrics:
            point = ratio(candidate_num.sum() - baseline_num.sum(), denominator.sum())
            low, high, p = bootstrap_difference(
                tasks, candidate_num, baseline_num, denominator, draws, rng
            )
            rows.append(
                {
                    "candidate": f"v2/{PRIMARY_DETECTOR}",
                    "baseline": label,
                    "quantile": PRIMARY_QUANTILE,
                    "metric": metric,
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

        both_failure = candidate_alarm & baseline_alarm & failure
        shift = baseline_first[both_failure] - candidate_first[both_failure]
        lead_rows.append(
            {
                "candidate": f"v2/{PRIMARY_DETECTOR}",
                "baseline": label,
                "both_detected_failures": int(both_failure.sum()),
                "candidate_earlier": int(np.sum(shift > 0)),
                "same_query": int(np.sum(shift == 0)),
                "candidate_later": int(np.sum(shift < 0)),
                "candidate_only_failures": int(np.sum(candidate_alarm & ~baseline_alarm & failure)),
                "baseline_only_failures": int(np.sum(~candidate_alarm & baseline_alarm & failure)),
                "first_alarm_advance_median": float(np.median(shift)) if len(shift) else np.nan,
                "first_alarm_advance_mean": float(np.mean(shift)) if len(shift) else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(lead_rows)


def provenance_table(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame
) -> pd.DataFrame:
    primary_alarm, primary_first = detector_alarm(
        sealed, PRIMARY_DETECTOR, PRIMARY_QUANTILE
    )
    delay_alarm, delay_first = detector_alarm(sealed, "completion_delay", PRIMARY_QUANTILE)
    confirmation_names = sealed["confirmation_names"].astype(str)
    winner = np.full(len(primary_alarm), "none", dtype="<U32")
    alarmed = np.flatnonzero(primary_alarm)
    winner_index = sealed["winning_confirmation"][alarmed, primary_first[alarmed]].astype(int)
    has_winner = winner_index >= 0
    winner[alarmed[has_winner]] = confirmation_names[winner_index[has_winner]]
    source = np.full(len(primary_alarm), "no_alarm", dtype="<U32")
    source[primary_alarm & delay_alarm & (primary_first == delay_first)] = "completion_delay"
    source[primary_alarm & (~delay_alarm | (primary_first < delay_first))] = "mechanism_advanced"
    source[primary_alarm & delay_alarm & (primary_first > delay_first)] = "later_than_delay"

    failure = labels["failure"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for outcome, mask in (("failure", failure), ("success", ~failure)):
        for source_name in np.unique(source):
            source_mask = mask & (source == source_name)
            for phenotype in np.unique(winner[source_mask]):
                rows.append(
                    {
                        "outcome": outcome,
                        "alarm_source": source_name,
                        "winning_confirmation": phenotype,
                        "count": int(np.sum(source_mask & (winner == phenotype))),
                        "outcome_n": int(mask.sum()),
                        "fraction_of_outcome": ratio(
                            np.sum(source_mask & (winner == phenotype)), mask.sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def structural_audit(labels: pd.DataFrame) -> dict[str, Any]:
    limits = {
        "libero_spatial": 22,
        "libero_object": 28,
        "libero_goal": 30,
        "libero_long": 52,
    }
    suite = labels["task"].str.split("/").str[0]
    at_limit = labels["episode_length"].eq(suite.map(limits)).to_numpy()
    failure = labels["failure"].to_numpy(dtype=bool)
    cell = labels.groupby(["task", "init_state_id"])["failure"].agg(["sum", "count"])
    mixed = cell[(cell["sum"] > 0) & (cell["sum"] < cell["count"])]
    return {
        "outcome_censoring_mismatches": int(np.sum(failure != at_limit)),
        "failure_at_query_limit": int(np.sum(failure & at_limit)),
        "success_before_query_limit": int(np.sum(~failure & ~at_limit)),
        "task_init_cells": int(len(cell)),
        "mixed_outcome_cells": int(len(mixed)),
        "failures_in_mixed_cells": int(mixed["sum"].sum()),
    }


def main() -> None:
    args = parse_args()
    manifest = verify_seal(args.result_dir)
    v2 = load_scores(
        args.result_dir / "sealed_online_scores.npz",
        "himoe.online_closed_loop.sealed.v2",
    )
    labels = join_labels(v2, args.labels)
    reference_index = build_index(v2)
    v1_raw = load_scores(
        args.v1_result_dir / "sealed_online_scores.npz",
        "himoe.online_multihead.sealed.v1",
    )
    v1 = align_scores(reference_index, v1_raw)

    metrics, by_task, by_suite = outcome_tables(
        v2, labels, args.bootstrap, args.seed
    )
    comparisons, lead = comparison_tables(
        v2, v1, labels, args.bootstrap, args.seed + 1
    )
    provenance = provenance_table(v2, labels)
    metrics.to_csv(args.result_dir / "outcome_metrics.csv", index=False)
    by_task.to_csv(args.result_dir / "outcome_metrics_by_task.csv", index=False)
    by_suite.to_csv(args.result_dir / "outcome_metrics_by_suite.csv", index=False)
    comparisons.to_csv(args.result_dir / "paired_detector_comparisons.csv", index=False)
    lead.to_csv(args.result_dir / "first_alarm_lead_comparisons.csv", index=False)
    provenance.to_csv(args.result_dir / "primary_alarm_provenance.csv", index=False)

    primary = metrics[
        (metrics["detector"] == PRIMARY_DETECTOR)
        & np.isclose(metrics["quantile"], PRIMARY_QUANTILE)
    ].iloc[0]
    delay = metrics[
        (metrics["detector"] == "completion_delay")
        & np.isclose(metrics["quantile"], PRIMARY_QUANTILE)
    ].iloc[0]
    summary = {
        "schema": "himoe.online_closed_loop.evaluation.v2",
        "manifest_sha256": sha256(args.result_dir / "sealed_manifest.json"),
        "label_table_sha256": sha256(args.labels),
        "episodes": int(len(labels)),
        "failures": int(labels["failure"].sum()),
        "successes": int((~labels["failure"]).sum()),
        "structural_audit": structural_audit(labels),
        "primary_q95": plain(primary.to_dict()),
        "completion_delay_q95": plain(delay.to_dict()),
        "exploratory": True,
        "independent_validation": False,
        "censoring_is_outcome_proxy_on_this_cohort": True,
        "bootstrap_unit": "task",
        "bootstrap_draws": args.bootstrap,
        "sealed_manifest": manifest,
    }
    (args.result_dir / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "evaluated v2: "
        f"TP={int(primary.tp)} FP={int(primary.fp)} "
        f"recall={primary.failure_recall:.4f} FPR={primary.success_fpr:.4f} "
        f"early8={primary.early8_recall:.4f} median_lead={primary.detected_failure_lead_median:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
