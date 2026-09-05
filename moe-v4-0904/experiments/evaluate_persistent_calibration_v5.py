#!/usr/bin/env python3
"""Evaluate direct calibration of the exact dual-regime persistent scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_dual_regime_v4 as v4
import evaluate_layerwise_alarm_development as dev
import evaluate_layerwise_persistence_v3 as v3


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUNDLE = HERE.parent
CANONICAL = WORKSPACE / "double-selete/trainfree"
PROTOCOL = BUNDLE / "method/ONLINE_PERSISTENT_CALIBRATION_V5_PROTOCOL.md"
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = CANONICAL / "results/timeout_extension_plus10"
DEFAULT_V4_RESULT = BUNDLE / "results/cache_new_v4"
DEFAULT_OUTPUT = BUNDLE / "results/persistent_calibration_v5"

QUANTILES = np.asarray((*dev.QUANTILES, 1.0), dtype=np.float64)
EARLY_LEADS = (2, 4, 8, 12)
OVERALL_FPR_LIMIT = 0.005
SUITE_FPR_LIMIT = 0.01
TASK_FPR_LIMIT = 0.025
MIN_PRECISION = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--v4-result", type=Path, default=DEFAULT_V4_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_peak(values: np.ndarray) -> np.ndarray:
    """Maximum deployed score, retaining too-short trajectories as -inf."""
    values = np.asarray(values, dtype=np.float32)
    return np.max(np.where(np.isfinite(values), values, -np.inf), axis=1)


def higher_quantiles(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    if not len(ordered):
        raise ValueError("empty calibration population")
    positions = np.ceil(QUANTILES * (len(ordered) - 1)).astype(int)
    return ordered[positions]


def crossfit_thresholds(cache: dict[str, np.ndarray], peaks: np.ndarray) -> np.ndarray:
    task_index = cache["task_index"].astype(int)
    init_state = cache["init_state_id"].astype(int)
    output = np.full((len(peaks), len(QUANTILES)), np.nan, dtype=np.float32)
    for task_position in np.unique(task_index):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"task {task_position} has {len(take)} episodes")
        task_init = init_state[take]
        for held_out in np.unique(task_init):
            test = task_init == held_out
            reference = ~test
            if int(test.sum()) != 8 or int(reference.sum()) != 392:
                raise ValueError("expected an eight-route held-out initial state")
            output[take[test]] = higher_quantiles(peaks[take[reference]])
    if np.isnan(output).any():
        raise ValueError("incomplete development threshold assignment")
    return output


def first_alarm_grid(
    persistent: np.ndarray, valid: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    output = np.full((thresholds.shape[1], len(persistent)), -1, dtype=np.int16)
    for position in range(thresholds.shape[1]):
        output[position] = v4.first_from_threshold(
            persistent, valid, thresholds[:, position]
        )
    return output


def development_head(
    cache: dict[str, np.ndarray], head: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, persistent = v4.head_scores(cache, head)
    thresholds = crossfit_thresholds(cache, trajectory_peak(persistent))
    first = first_alarm_grid(persistent, cache["valid"].astype(bool), thresholds)
    return first, thresholds, persistent


def external_head(
    external: dict[str, np.ndarray],
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
    head: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    _, persistent = v4.head_scores(external, head)
    thresholds = np.full(
        (len(persistent), len(QUANTILES)), np.nan, dtype=np.float32
    )
    task_index = external["task_index"].astype(int)
    rows: list[dict[str, Any]] = []
    peak_cache: dict[int, np.ndarray] = {}

    for task_position, task in enumerate(external["task_names"].astype(str)):
        test = np.flatnonzero(task_index == task_position)
        reference, take = v3.reference_for_task(
            task, main_reference, extra_reference
        )
        cache_key = id(reference)
        if cache_key not in peak_cache:
            _, reference_persistent = v4.head_scores(reference, head)
            peak_cache[cache_key] = trajectory_peak(reference_persistent)
        task_thresholds = higher_quantiles(peak_cache[cache_key][take])
        thresholds[test] = task_thresholds
        for quantile, threshold in zip(QUANTILES, task_thresholds, strict=True):
            rows.append(
                {
                    "task": task,
                    "head": head,
                    "quantile": quantile,
                    "threshold": threshold,
                    "reference_episodes": len(take),
                    "calibration_statistic": "trajectory_max_persistent_score",
                    "outcomes_used": False,
                }
            )
    if np.isnan(thresholds).any():
        raise ValueError(f"incomplete external thresholds for {head}")
    return (
        first_alarm_grid(persistent, external["valid"].astype(bool), thresholds),
        thresholds,
        rows,
    )


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metric_row(
    cohort: str,
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    group: str = "all",
) -> dict[str, Any]:
    alarm = np.asarray(first) >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    timely = ~risk
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    persistent_failure = labels["failure"].to_numpy(dtype=bool)
    lead = labels["length"].to_numpy(dtype=int) - 1 - np.asarray(first)
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    row: dict[str, Any] = {
        "cohort": cohort,
        "group": group,
        "detector": detector,
        "episodes": len(labels),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "late_n": int(late.sum()),
        "persistent_n": int(persistent_failure.sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "late_recall": ratio((alarm & late).sum(), late.sum()),
        "persistent_recall": ratio(
            (alarm & persistent_failure).sum(), persistent_failure.sum()
        ),
        "detected_risk_lead_median": (
            float(np.median(lead[alarm & risk])) if tp else float("nan")
        ),
    }
    for value in EARLY_LEADS:
        row[f"early{value}_risk_recall"] = ratio(
            (alarm & risk & (lead >= value)).sum(), risk.sum()
        )
    return row


def grouped_rows(
    cohort: str,
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = labels[column].to_numpy(dtype=str)
    for group in sorted(np.unique(values)):
        take = values == group
        rows.append(
            metric_row(
                cohort,
                detector,
                np.asarray(first)[take],
                labels.loc[take].reset_index(drop=True),
                group,
            )
        )
    return rows


def candidate_row(
    lock_quantile: float,
    instability_quantile: float,
    first: np.ndarray,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    row = metric_row("development_main", "candidate", first, labels)
    suites = grouped_rows(
        "development_main", "candidate", first, labels, "suite"
    )
    tasks = grouped_rows(
        "development_main", "candidate", first, labels, "task"
    )
    task_recalls = [
        item["risk_recall"]
        for item in tasks
        if item["risk_n"] >= 5 and np.isfinite(item["risk_recall"])
    ]
    row.update(
        {
            "lock_quantile": lock_quantile,
            "instability_quantile": instability_quantile,
            "max_suite_fpr": max(item["timely_fpr"] for item in suites),
            "max_task_fpr": max(item["timely_fpr"] for item in tasks),
            "worst_task_recall_min_risk5": min(task_recalls),
        }
    )
    row["eligible"] = bool(
        row["timely_fpr"] <= OVERALL_FPR_LIMIT
        and row["max_suite_fpr"] <= SUITE_FPR_LIMIT
        and row["max_task_fpr"] <= TASK_FPR_LIMIT
        and row["precision"] >= MIN_PRECISION
    )
    return row


def select_candidate(frame: pd.DataFrame) -> pd.Series | None:
    eligible = frame[frame["eligible"]].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        [
            "early4_risk_recall",
            "early8_risk_recall",
            "risk_recall",
            "worst_task_recall_min_risk5",
            "max_suite_fpr",
            "timely_fpr",
            "lock_quantile",
            "instability_quantile",
        ],
        ascending=[False, False, False, False, True, True, False, False],
    ).iloc[0]


def quantile_position(value: float) -> int:
    positions = np.flatnonzero(np.isclose(QUANTILES, value))
    if len(positions) != 1:
        raise KeyError(f"quantile absent or ambiguous: {value}")
    return int(positions[0])


def detector_tables(
    cohort: str,
    labels: pd.DataFrame,
    detectors: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall: list[dict[str, Any]] = []
    suites: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for detector, first in detectors.items():
        overall.append(metric_row(cohort, detector, first, labels))
        suites.extend(grouped_rows(cohort, detector, first, labels, "suite"))
        tasks.extend(grouped_rows(cohort, detector, first, labels, "task"))
    return pd.DataFrame(overall), pd.DataFrame(suites), pd.DataFrame(tasks)


def main() -> None:
    args = parse_args()
    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")

    main_lock, main_lock_thresholds, _ = development_head(main_cache, "lock")
    main_instability, main_instability_thresholds, _ = development_head(
        main_cache, "instability"
    )
    external_lock, external_lock_thresholds, lock_rows = external_head(
        external_cache, main_cache, extra_cache, "lock"
    )
    external_instability, external_instability_thresholds, instability_rows = (
        external_head(external_cache, main_cache, extra_cache, "instability")
    )

    development_labels = v3.aligned_labels(
        main_cache,
        args.label_root / "development_main_clean_labels.csv",
        "development_main",
    )
    candidate_rows: list[dict[str, Any]] = []
    for lock_position, lock_quantile in enumerate(QUANTILES):
        for instability_position, instability_quantile in enumerate(QUANTILES):
            first = v4.first_or(
                main_lock[lock_position], main_instability[instability_position]
            )
            candidate_rows.append(
                candidate_row(
                    float(lock_quantile),
                    float(instability_quantile),
                    first,
                    development_labels,
                )
            )
    candidates = pd.DataFrame(candidate_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output / "development_candidates.csv", index=False)
    selected = select_candidate(candidates)
    max_position = quantile_position(1.0)
    main_max = v4.first_or(main_lock[max_position], main_instability[max_position])
    external_max = v4.first_or(
        external_lock[max_position], external_instability[max_position]
    )
    if selected is not None:
        selected_lock_position = quantile_position(float(selected["lock_quantile"]))
        selected_instability_position = quantile_position(
            float(selected["instability_quantile"])
        )
        main_selected = v4.first_or(
            main_lock[selected_lock_position],
            main_instability[selected_instability_position],
        )
        external_selected = v4.first_or(
            external_lock[selected_lock_position],
            external_instability[selected_instability_position],
        )

    threshold_frame = pd.DataFrame(lock_rows + instability_rows)
    threshold_frame.to_csv(
        args.output / "external_outcome_blind_threshold_grid.csv", index=False
    )
    selection = {
        "schema": "himoe.persistent_calibration_v5.selection.v1",
        "status": (
            "selected" if selected is not None else "rejected_no_eligible_candidate"
        ),
        "external_outcomes_read": False,
        "candidate_count": len(candidates),
        "eligible_count": int(candidates["eligible"].sum()),
        "constraint_pass_counts": {
            "overall_fpr": int(
                (candidates["timely_fpr"] <= OVERALL_FPR_LIMIT).sum()
            ),
            "suite_fpr": int(
                (candidates["max_suite_fpr"] <= SUITE_FPR_LIMIT).sum()
            ),
            "task_fpr": int(
                (candidates["max_task_fpr"] <= TASK_FPR_LIMIT).sum()
            ),
            "precision": int((candidates["precision"] >= MIN_PRECISION).sum()),
        },
        "constraints": {
            "overall_fpr_max": OVERALL_FPR_LIMIT,
            "suite_fpr_max": SUITE_FPR_LIMIT,
            "task_fpr_max": TASK_FPR_LIMIT,
            "precision_min": MIN_PRECISION,
        },
        "ranking": [
            "early4_risk_recall desc",
            "early8_risk_recall desc",
            "risk_recall desc",
            "worst_task_recall_min_risk5 desc",
            "max_suite_fpr asc",
            "timely_fpr asc",
        ],
        "selected": selected.to_dict() if selected is not None else None,
    }
    (args.output / "selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Seal the fixed control and any eligible selection before loading outcomes.
    alarm_payload: dict[str, np.ndarray] = {
        "schema": np.asarray("himoe.persistent_calibration_v5.alarms.v1"),
        "quantiles": QUANTILES,
        "main_max_reference_lock": main_lock[max_position],
        "main_max_reference_instability": main_instability[max_position],
        "main_max_reference_dual": main_max,
        "external_max_reference_lock": external_lock[max_position],
        "external_max_reference_instability": external_instability[max_position],
        "external_max_reference_dual": external_max,
        "main_max_reference_lock_threshold": main_lock_thresholds[:, max_position],
        "main_max_reference_instability_threshold": main_instability_thresholds[
            :, max_position
        ],
        "external_max_reference_lock_threshold": external_lock_thresholds[
            :, max_position
        ],
        "external_max_reference_instability_threshold": (
            external_instability_thresholds[:, max_position]
        ),
    }
    if selected is not None:
        alarm_payload.update(
            selected_lock_quantile=np.asarray(selected["lock_quantile"]),
            selected_instability_quantile=np.asarray(
                selected["instability_quantile"]
            ),
            main_selected_lock=main_lock[selected_lock_position],
            main_selected_instability=main_instability[
                selected_instability_position
            ],
            main_selected_dual=main_selected,
            external_selected_lock=external_lock[selected_lock_position],
            external_selected_instability=external_instability[
                selected_instability_position
            ],
            external_selected_dual=external_selected,
            main_selected_lock_threshold=main_lock_thresholds[
                :, selected_lock_position
            ],
            main_selected_instability_threshold=main_instability_thresholds[
                :, selected_instability_position
            ],
            external_selected_lock_threshold=external_lock_thresholds[
                :, selected_lock_position
            ],
            external_selected_instability_threshold=(
                external_instability_thresholds[:, selected_instability_position]
            ),
        )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        **alarm_payload,
    )

    task_index = external_cache["task_index"].astype(int)
    first_per_task = np.asarray(
        [np.flatnonzero(task_index == position)[0] for position in range(len(external_cache["task_names"]))]
    )
    np.savez_compressed(
        args.output / "max_reference_profiles.npz",
        schema=np.asarray("himoe.persistent_calibration_v5.max_profile.v1"),
        task_names=external_cache["task_names"],
        lock_thresholds=external_lock_thresholds[
            first_per_task, max_position
        ],
        instability_thresholds=external_instability_thresholds[
            first_per_task, max_position
        ],
        lock_quantile=np.asarray(1.0),
        instability_quantile=np.asarray(1.0),
        calibration_statistic=np.asarray("trajectory_max_persistent_score"),
    )

    external_labels = v3.aligned_labels(
        external_cache,
        args.label_root / "external_8b_clean_labels.csv",
        "external_8b",
    )
    with np.load(args.v4_result / "sealed_first_alarms.npz", allow_pickle=False) as archive:
        v4_main = np.asarray(archive["main_dual"])
        v4_external = np.asarray(archive["external_dual"])

    detector_names = {
        "v4_dual_regime_or": (v4_main, v4_external),
        "persistent_max_reference_lock": (
            main_lock[max_position],
            external_lock[max_position],
        ),
        "persistent_max_reference_instability": (
            main_instability[max_position],
            external_instability[max_position],
        ),
        "persistent_max_reference_dual": (main_max, external_max),
    }
    if selected is not None:
        detector_names.update(
            {
                "persistent_lead_selected_lock": (
                    main_lock[selected_lock_position],
                    external_lock[selected_lock_position],
                ),
                "persistent_lead_selected_instability": (
                    main_instability[selected_instability_position],
                    external_instability[selected_instability_position],
                ),
                "persistent_lead_selected_dual": (
                    main_selected,
                    external_selected,
                ),
            }
        )
    overall_frames: list[pd.DataFrame] = []
    suite_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    for cohort, labels, position in (
        ("development_main", development_labels, 0),
        ("external_8b", external_labels, 1),
    ):
        detectors = {
            name: values[position] for name, values in detector_names.items()
        }
        overall, suites, tasks = detector_tables(cohort, labels, detectors)
        overall_frames.append(overall)
        suite_frames.append(suites)
        task_frames.append(tasks)
    overall = pd.concat(overall_frames, ignore_index=True)
    by_suite = pd.concat(suite_frames, ignore_index=True)
    by_task = pd.concat(task_frames, ignore_index=True)
    overall.to_csv(args.output / "outcome_metrics.csv", index=False)
    by_suite.to_csv(args.output / "outcome_metrics_by_suite.csv", index=False)
    by_task.to_csv(args.output / "outcome_metrics_by_task.csv", index=False)

    episodes = pd.concat(
        [
            development_labels.assign(
                first_v4_query=v4_main,
                first_persistent_max_lock_query=main_lock[max_position],
                first_persistent_max_instability_query=main_instability[max_position],
                first_persistent_max_query=main_max,
            ),
            external_labels.assign(
                first_v4_query=v4_external,
                first_persistent_max_lock_query=external_lock[max_position],
                first_persistent_max_instability_query=external_instability[
                    max_position
                ],
                first_persistent_max_query=external_max,
            ),
        ],
        ignore_index=True,
    )
    episodes.to_csv(args.output / "episode_alarms.csv", index=False)

    summary = {
        "schema": "himoe.persistent_calibration_v5.evaluation.v1",
        "status": (
            "rejected on development constraints; fixed max-reference control "
            "replayed externally; external 8B is not pristine"
            if selected is None
            else "development-selected post-hoc external replay; external 8B is not pristine"
        ),
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_rule_frozen_before_outcome_join": True,
        "calibration_statistic": "trajectory_max_persistent_score",
        "max_reference_union_bound": {
            "development": 2.0 / 393.0,
            "external": 2.0 / 401.0,
            "assumption": "within-task exchangeability; strict crossing; Bonferroni over two heads",
        },
        "selection": selection,
        "metrics": overall.to_dict(orient="records"),
        "artifacts": {
            "evaluator_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(PROTOCOL),
            "selection_sha256": sha256(args.output / "selection.json"),
            "sealed_first_alarms_sha256": sha256(
                args.output / "sealed_first_alarms.npz"
            ),
            "max_reference_profiles_sha256": sha256(
                args.output / "max_reference_profiles.npz"
            ),
            "outcome_metrics_sha256": sha256(
                args.output / "outcome_metrics.csv"
            ),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
