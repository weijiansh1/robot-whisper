#!/usr/bin/env python3
"""Reveal outcomes after verifying the external precision-cascade seal."""

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
from online_precision_cascade_alarm import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MAIN_CACHE,
    DEFAULT_MAIN_EXTERNAL_CACHE,
    DEFAULT_OUTPUT,
    PROTOCOL,
    TEST_RUN_ID,
)
from precision_cascade_monitor import PRIMARY_DETECTOR, PRIMARY_QUANTILE


HERE = Path(__file__).resolve().parent
SEED = 20260904


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
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
    if manifest.get("schema") != "himoe.precision_cascade.manifest.v1":
        raise ValueError("unknown precision-cascade manifest schema")
    if manifest.get("labels_used") != []:
        raise ValueError("external scorer used labels before sealing")
    if manifest.get("test_episodes_used_for_calibration") != 0:
        raise ValueError("external test trajectories entered calibration")
    if not manifest.get("query_causal") or manifest.get("runtime_future_access"):
        raise ValueError("external scorer does not satisfy causal runtime contract")
    if manifest.get("external_outcomes_read_before_seal"):
        raise ValueError("external outcomes were read before sealing")

    artifacts = manifest["artifacts"]
    paths = {
        "protocol_sha256": PROTOCOL,
        "scorer_sha256": HERE / "online_precision_cascade_alarm.py",
        "runtime_monitor_sha256": HERE / "precision_cascade_monitor.py",
        "runtime_extractor_sha256": HERE / "route_derivative_monitor.py",
        "runtime_base_extractor_sha256": HERE / "single_rollout_monitor.py",
        "feature_extractor_sha256": HERE / "online_multihead_alarm.py",
        "main_reference_cache_sha256": DEFAULT_MAIN_CACHE,
        "main_external_reference_cache_sha256": DEFAULT_MAIN_EXTERNAL_CACHE,
        "external_feature_cache_sha256": result_dir / "unlabeled_query_features.npz",
        "sealed_scores_sha256": result_dir / "sealed_online_scores.npz",
        "thresholds_sha256": result_dir / "unlabeled_thresholds.csv",
        "episode_alarms_sha256": result_dir / "sealed_episode_alarms.csv",
        "deployment_profiles_sha256": result_dir / "deployment_profiles.npz",
    }
    for key, path in paths.items():
        if sha256(path) != artifacts[key]:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def load_sealed(result_dir: Path) -> dict[str, np.ndarray]:
    path = result_dir / "sealed_online_scores.npz"
    with np.load(path, allow_pickle=False) as archive:
        sealed = {name: np.asarray(archive[name]) for name in archive.files}
    if str(sealed["schema"]) != "himoe.precision_cascade.sealed.v1":
        raise ValueError("unknown precision-cascade score schema")
    return sealed


def reveal_labels(
    cache_root: Path, sealed: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for task in sealed["task_names"].astype(str):
        path = cache_root / task / TEST_RUN_ID / "client/summaries.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        hashes[str(path)] = sha256(path)
        for item in raw:
            rows.append(
                {
                    "task": task,
                    "episode": int(item["episode_index"]),
                    "init_state_id_label": int(item["init_state_id"]),
                    "flow_noise_seed_label": int(item["flow_noise_seed"]),
                    "episode_length": int(item["inference_calls"]),
                    "failure": not bool(item["success"]),
                }
            )
    labels = pd.DataFrame(rows)
    index = common.build_index(sealed)
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("sealed_row")
    if len(merged) != len(index) or merged["failure"].isna().any():
        raise ValueError("external outcome rows do not match sealed index")
    if not np.array_equal(
        merged["sealed_length"].to_numpy(), merged["episode_length"].to_numpy()
    ):
        raise ValueError("external episode length changed after seal")
    for name in ("init_state_id", "flow_noise_seed"):
        if not np.array_equal(
            merged[name].to_numpy(), merged[f"{name}_label"].to_numpy()
        ):
            raise ValueError(f"external metadata changed after seal: {name}")
    return merged.reset_index(drop=True), hashes


def paired_comparisons(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].to_numpy(dtype=str)
    lengths = sealed["length"].astype(int)
    comparisons = (
        ("mobility_w4_k4", "mobility_w8_k3"),
        ("mobility_w4_k4", "mobility_w4_k1"),
        ("typed_confirmed", "mobility_w4_k4"),
        ("static_confirmed", "mobility_w4_k4"),
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for quantile in sealed["quantiles"].astype(float):
        for candidate, baseline in comparisons:
            ca, cf = common.detector_alarm(sealed, candidate, quantile)
            ba, bf = common.detector_alarm(sealed, baseline, quantile)
            clead = lengths - 1 - cf
            blead = lengths - 1 - bf
            metrics = [
                ("failure_recall", ca & failure, ba & failure, failure),
                ("success_fpr", ca & success, ba & success, success),
            ]
            for early in common.EARLY_LEADS:
                metrics.append(
                    (
                        f"early{early}_recall",
                        ca & failure & (clead >= early),
                        ba & failure & (blead >= early),
                        failure,
                    )
                )
            for metric, candidate_num, baseline_num, denominator in metrics:
                delta = common.ratio(
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
                        "candidate": candidate,
                        "baseline": baseline,
                        "quantile": quantile,
                        "metric": metric,
                        "delta": delta,
                        "ci_low": low,
                        "ci_high": high,
                        "two_sided_p": p,
                        "candidate_only": int((candidate_num & ~baseline_num).sum()),
                        "baseline_only": int((baseline_num & ~candidate_num).sum()),
                    }
                )
    return pd.DataFrame(rows)


def false_alarm_diagnostics(
    sealed: dict[str, np.ndarray], labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = ~labels["failure"].to_numpy(dtype=bool)
    lengths = sealed["length"].astype(int)
    tasks = labels["task"].to_numpy(dtype=str)
    alarm, first = common.detector_alarm(
        sealed, PRIMARY_DETECTOR, PRIMARY_QUANTILE
    )
    fp = alarm & success
    tn = ~alarm & success
    lead = lengths - 1 - first
    phase = np.full(len(first), np.nan, dtype=float)
    phase[alarm] = first[alarm] / np.maximum(lengths[alarm] - 1, 1)
    summary = pd.DataFrame(
        [
            {
                "population": "false_positive_success",
                "n": int(fp.sum()),
                "length_median": float(np.median(lengths[fp])) if fp.any() else np.nan,
                "first_alarm_query_median": float(np.median(first[fp])) if fp.any() else np.nan,
                "lead_to_success_median": float(np.median(lead[fp])) if fp.any() else np.nan,
                "alarm_phase_median": float(np.median(phase[fp])) if fp.any() else np.nan,
            },
            {
                "population": "true_negative_success",
                "n": int(tn.sum()),
                "length_median": float(np.median(lengths[tn])) if tn.any() else np.nan,
                "first_alarm_query_median": np.nan,
                "lead_to_success_median": np.nan,
                "alarm_phase_median": np.nan,
            },
        ]
    )
    task_rows: list[dict[str, Any]] = []
    for task in np.unique(tasks):
        member = tasks == task
        task_success = member & success
        task_fp = member & fp
        task_rows.append(
            {
                "task": task,
                "success_n": int(task_success.sum()),
                "false_alarms": int(task_fp.sum()),
                "success_fpr": common.ratio(task_fp.sum(), task_success.sum()),
                "median_fp_lead_to_success": (
                    float(np.median(lead[task_fp])) if task_fp.any() else np.nan
                ),
            }
        )
    return summary, pd.DataFrame(task_rows)


def main() -> None:
    args = parse_args()
    manifest = verify_seal(args.result_dir)
    sealed = load_sealed(args.result_dir)
    labels, label_hashes = reveal_labels(args.cache_root, sealed)
    outcome, by_task, by_suite = common.outcome_tables(
        sealed, labels, args.bootstrap, args.seed
    )
    paired = paired_comparisons(
        sealed, labels, args.bootstrap, args.seed + 1
    )
    false_alarm_summary, false_alarm_by_task = false_alarm_diagnostics(
        sealed, labels
    )

    outcome.to_csv(args.result_dir / "outcome_metrics.csv", index=False)
    by_task.to_csv(args.result_dir / "outcome_metrics_by_task.csv", index=False)
    by_suite.to_csv(args.result_dir / "outcome_metrics_by_suite.csv", index=False)
    paired.to_csv(args.result_dir / "paired_detector_comparisons.csv", index=False)
    false_alarm_summary.to_csv(
        args.result_dir / "false_alarm_summary.csv", index=False
    )
    false_alarm_by_task.to_csv(
        args.result_dir / "false_alarm_by_task.csv", index=False
    )

    primary = outcome[
        (outcome["detector"] == PRIMARY_DETECTOR)
        & np.isclose(outcome["quantile"], PRIMARY_QUANTILE)
    ].iloc[0]
    balanced = outcome[
        (outcome["detector"] == PRIMARY_DETECTOR)
        & np.isclose(outcome["quantile"], 0.975)
    ].iloc[0]
    summary = {
        "schema": "himoe.precision_cascade.evaluation.v1",
        "seal_verified_before_labels": True,
        "sealed_manifest_sha256": sha256(args.result_dir / "sealed_manifest.json"),
        "labels_used_by_scorer": manifest["labels_used"],
        "labels_used_by_evaluator": ["success/failure"],
        "label_files": label_hashes,
        "tasks": int(len(sealed["task_names"])),
        "episodes": int(len(labels)),
        "failures": int(labels["failure"].sum()),
        "successes": int((~labels["failure"]).sum()),
        "primary_detector": PRIMARY_DETECTOR,
        "primary_quantile": PRIMARY_QUANTILE,
        "primary": primary.to_dict(),
        "balanced_quantile": 0.975,
        "balanced": balanced.to_dict(),
        "bootstrap_unit": "task",
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
    }
    (args.result_dir / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    columns = [
        "detector",
        "quantile",
        "tp",
        "fp",
        "failure_recall",
        "success_fpr",
        "precision",
        "early8_recall",
        "detected_failure_lead_median",
    ]
    print(outcome[columns].to_string(index=False))


if __name__ == "__main__":
    main()
