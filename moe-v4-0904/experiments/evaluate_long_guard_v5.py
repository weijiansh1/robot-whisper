#!/usr/bin/env python3
"""Select a lead-aware long-suite lock guard and replay it externally."""

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
PROTOCOL = BUNDLE / "method/ONLINE_LONG_GUARD_V5_PROTOCOL.md"
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = CANONICAL / "results/timeout_extension_plus10"
DEFAULT_V4_RESULT = BUNDLE / "results/cache_new_v4"
DEFAULT_OUTPUT = BUNDLE / "results/long_guard_v5"

REPRESENTATIONS = ("all_median", "L12", "back_min")
WIDTHS = (2, 4)
CONFIRMATIONS = (4, 6, 8, 10, 12)
QUANTILES = dev.QUANTILES
LONG_SUITE = "libero_long"
EARLY_LEADS = (2, 4, 8, 12)
OVERALL_FPR_LIMIT = 0.005
LONG_FPR_LIMIT = 0.01
LONG_TASK_FPR_LIMIT = 0.025
LONG_PRECISION_MIN = 0.75


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
    remaining = labels["failure"].to_numpy(dtype=bool)
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
        "persistent_n": int(remaining.sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "late_recall": ratio((alarm & late).sum(), late.sum()),
        "persistent_recall": ratio((alarm & remaining).sum(), remaining.sum()),
        "detected_risk_lead_median": (
            float(np.median(lead[alarm & risk])) if tp else float("nan")
        ),
    }
    for value in EARLY_LEADS:
        row[f"early{value}_risk_recall"] = ratio(
            (alarm & risk & (lead >= value)).sum(), risk.sum()
        )
    return row


def group_rows(
    cohort: str,
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    values = labels[column].to_numpy(dtype=str)
    rows: list[dict[str, Any]] = []
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


def oriented_instant(
    cache: dict[str, np.ndarray], representation: str, width: int
) -> np.ndarray:
    values, family = dev.representations(cache)[representation]
    if family != "mobility":
        raise ValueError(f"expected a mobility representation: {representation}")
    instant = -dev.trailing_mean(values, width)
    instant[~cache["valid"].astype(bool)] = np.nan
    return instant


def development_lock_grid(
    cache: dict[str, np.ndarray], representation: str, width: int
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    instant = oriented_instant(cache, representation, width)
    thresholds = dev.crossfit_thresholds(
        dev.row_max(instant),
        cache["task_index"].astype(int),
        cache["init_state_id"].astype(int),
    )
    first_by_confirmation: dict[int, np.ndarray] = {}
    valid = cache["valid"].astype(bool)
    for confirmation in CONFIRMATIONS:
        persistent = dev.persistent_score(instant, confirmation)
        first = np.full((len(QUANTILES), len(instant)), -1, dtype=np.int16)
        for position in range(len(QUANTILES)):
            first[position] = v4.first_from_threshold(
                persistent, valid, thresholds[:, position]
            )
        first_by_confirmation[confirmation] = first
    return first_by_confirmation, thresholds


def candidate_row(
    representation: str,
    width: int,
    confirmations: int,
    quantile: float,
    first: np.ndarray,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    overall = metric_row("development_main", "candidate", first, labels)
    long_take = labels["suite"].eq(LONG_SUITE).to_numpy()
    long_labels = labels.loc[long_take].reset_index(drop=True)
    long_row = metric_row(
        "development_main",
        "candidate",
        np.asarray(first)[long_take],
        long_labels,
        LONG_SUITE,
    )
    tasks = group_rows(
        "development_main",
        "candidate",
        np.asarray(first)[long_take],
        long_labels,
        "task",
    )
    task_recalls = [
        row["risk_recall"]
        for row in tasks
        if row["risk_n"] >= 5 and np.isfinite(row["risk_recall"])
    ]
    overall.update(
        {
            "representation": representation,
            "width": width,
            "confirmations": confirmations,
            "quantile": quantile,
            "long_risk_n": long_row["risk_n"],
            "long_tp": long_row["tp"],
            "long_fp": long_row["fp"],
            "long_recall": long_row["risk_recall"],
            "long_early4_recall": long_row["early4_risk_recall"],
            "long_early8_recall": long_row["early8_risk_recall"],
            "long_fpr": long_row["timely_fpr"],
            "long_precision": long_row["precision"],
            "long_max_task_fpr": max(row["timely_fpr"] for row in tasks),
            "long_worst_task_recall_min_risk5": min(task_recalls),
        }
    )
    overall["eligible"] = bool(
        overall["timely_fpr"] <= OVERALL_FPR_LIMIT
        and overall["long_fpr"] <= LONG_FPR_LIMIT
        and overall["long_max_task_fpr"] <= LONG_TASK_FPR_LIMIT
        and overall["long_precision"] >= LONG_PRECISION_MIN
    )
    return overall


def select_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    eligible = candidates[candidates["eligible"]]
    if eligible.empty:
        return None
    return eligible.sort_values(
        [
            "early4_risk_recall",
            "early8_risk_recall",
            "long_early4_recall",
            "long_early8_recall",
            "risk_recall",
            "long_worst_task_recall_min_risk5",
            "long_fpr",
            "timely_fpr",
            "quantile",
            "confirmations",
        ],
        ascending=[False, False, False, False, False, False, True, True, False, True],
    ).iloc[0]


def quantile_position(value: float) -> int:
    positions = np.flatnonzero(np.isclose(np.asarray(QUANTILES), value))
    if len(positions) != 1:
        raise KeyError(value)
    return int(positions[0])


def reference_instant(
    reference: dict[str, np.ndarray], representation: str, width: int
) -> np.ndarray:
    return oriented_instant(reference, representation, width)


def external_selected_lock(
    external: dict[str, np.ndarray],
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
    representation: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    instant = oriented_instant(external, representation, width)
    persistent = dev.persistent_score(instant, confirmations)
    valid = external["valid"].astype(bool)
    task_index = external["task_index"].astype(int)
    first = np.full(len(instant), -1, dtype=np.int16)
    thresholds = np.full(len(instant), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    instant_cache: dict[int, np.ndarray] = {}
    for task_position, task in enumerate(external["task_names"].astype(str)):
        test = np.flatnonzero(task_index == task_position)
        reference, take = v3.reference_for_task(task, main_reference, extra_reference)
        cache_key = id(reference)
        if cache_key not in instant_cache:
            instant_cache[cache_key] = reference_instant(
                reference, representation, width
            )
        calibration_peak = dev.row_max(instant_cache[cache_key][take])
        threshold = dev.quantile_higher(calibration_peak, quantile)
        thresholds[test] = threshold
        first[test] = v4.first_from_threshold(
            persistent[test], valid[test], threshold
        )
        rows.append(
            {
                "task": task,
                "representation": representation,
                "width": width,
                "confirmations": confirmations,
                "quantile": quantile,
                "threshold": threshold,
                "reference_episodes": len(take),
                "outcomes_used": False,
            }
        )
    if not np.isfinite(thresholds).all():
        raise ValueError("incomplete external threshold assignment")
    return first, thresholds, rows


def tables(
    cohort: str,
    labels: pd.DataFrame,
    detectors: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall: list[dict[str, Any]] = []
    suites: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for name, first in detectors.items():
        overall.append(metric_row(cohort, name, first, labels))
        suites.extend(group_rows(cohort, name, first, labels, "suite"))
        tasks.extend(group_rows(cohort, name, first, labels, "task"))
    return pd.DataFrame(overall), pd.DataFrame(suites), pd.DataFrame(tasks)


def main() -> None:
    args = parse_args()
    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")
    development_labels = v3.aligned_labels(
        main_cache,
        args.label_root / "development_main_clean_labels.csv",
        "development_main",
    )
    with np.load(
        args.v4_result / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        v4_main_lock = np.asarray(archive["main_lock"])
        v4_main_instability = np.asarray(archive["main_instability"])
        v4_main_dual = np.asarray(archive["main_dual"])
        v4_external_lock = np.asarray(archive["external_lock"])
        v4_external_instability = np.asarray(archive["external_instability"])
        v4_external_dual = np.asarray(archive["external_dual"])

    long_main = development_labels["suite"].eq(LONG_SUITE).to_numpy()
    candidates: list[dict[str, Any]] = []
    first_cache: dict[tuple[str, int, int], np.ndarray] = {}
    threshold_cache: dict[tuple[str, int], np.ndarray] = {}
    for representation in REPRESENTATIONS:
        for width in WIDTHS:
            first_by_confirmation, thresholds = development_lock_grid(
                main_cache, representation, width
            )
            threshold_cache[(representation, width)] = thresholds
            for confirmations, first_grid in first_by_confirmation.items():
                first_cache[(representation, width, confirmations)] = first_grid
                for position, quantile in enumerate(QUANTILES):
                    hybrid = v4_main_dual.copy()
                    long_alarm = v4.first_or(
                        first_grid[position, long_main],
                        v4_main_instability[long_main],
                    )
                    hybrid[long_main] = long_alarm
                    candidates.append(
                        candidate_row(
                            representation,
                            width,
                            confirmations,
                            quantile,
                            hybrid,
                            development_labels,
                        )
                    )

    candidate_frame = pd.DataFrame(candidates)
    selected = select_candidate(candidate_frame)
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(args.output / "development_candidates.csv", index=False)
    selection = {
        "schema": "himoe.long_guard_v5.selection.v1",
        "status": "selected" if selected is not None else "rejected_no_eligible_candidate",
        "external_outcomes_read": False,
        "candidate_count": len(candidate_frame),
        "eligible_count": int(candidate_frame["eligible"].sum()),
        "constraints": {
            "overall_fpr_max": OVERALL_FPR_LIMIT,
            "long_fpr_max": LONG_FPR_LIMIT,
            "long_task_fpr_max": LONG_TASK_FPR_LIMIT,
            "long_precision_min": LONG_PRECISION_MIN,
        },
        "selected": selected.to_dict() if selected is not None else None,
    }
    (args.output / "selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if selected is None:
        raise RuntimeError("no long-guard candidate satisfied the frozen constraints")

    representation = str(selected["representation"])
    width = int(selected["width"])
    confirmations = int(selected["confirmations"])
    quantile = float(selected["quantile"])
    position = quantile_position(quantile)
    main_selected_lock = first_cache[(representation, width, confirmations)][position]
    main_selected_threshold = threshold_cache[(representation, width)][:, position]
    main_deployed_lock = v4_main_lock.copy()
    main_deployed_lock[long_main] = main_selected_lock[long_main]
    main_hybrid = v4_main_dual.copy()
    main_hybrid[long_main] = v4.first_or(
        main_selected_lock[long_main], v4_main_instability[long_main]
    )

    external_lock, external_threshold, threshold_rows = external_selected_lock(
        external_cache,
        main_cache,
        extra_cache,
        representation,
        width,
        confirmations,
        quantile,
    )
    external_task = external_cache["task_names"].astype(str)[
        external_cache["task_index"].astype(int)
    ]
    long_external = np.char.startswith(external_task, f"{LONG_SUITE}/")
    external_deployed_lock = v4_external_lock.copy()
    external_deployed_lock[long_external] = external_lock[long_external]
    external_hybrid = v4_external_dual.copy()
    external_hybrid[long_external] = v4.first_or(
        external_lock[long_external], v4_external_instability[long_external]
    )

    pd.DataFrame(threshold_rows).to_csv(
        args.output / "external_outcome_blind_thresholds.csv", index=False
    )
    with np.load(
        args.v4_result / "deployment_profiles.npz", allow_pickle=False
    ) as archive:
        profile_tasks = archive["task_names"].astype(str)
        external_tasks = external_cache["task_names"].astype(str)
        if not np.array_equal(profile_tasks, external_tasks):
            raise ValueError(
                "v4 deployment-profile task order does not match external cache"
            )
        base_lock_thresholds = np.asarray(
            archive["lock_thresholds"], dtype=np.float32
        )
        instability_thresholds = np.asarray(
            archive["instability_thresholds"], dtype=np.float32
        )
    selected_thresholds = np.asarray(
        [row["threshold"] for row in threshold_rows], dtype=np.float32
    )
    long_profile = np.char.startswith(profile_tasks, f"{LONG_SUITE}/")
    profile_lock_thresholds = np.where(
        long_profile, selected_thresholds, base_lock_thresholds
    ).astype(np.float32)
    profile_count = len(profile_tasks)
    np.savez_compressed(
        args.output / "deployment_profiles.npz",
        schema=np.asarray("himoe.long_guard_v5.profile.v1"),
        task_names=profile_tasks,
        lock_thresholds=profile_lock_thresholds,
        instability_thresholds=instability_thresholds,
        lock_representations=np.where(
            long_profile, representation, "all_median"
        ),
        lock_widths=np.where(long_profile, width, v4.LOCK_WIDTH).astype(np.int16),
        lock_confirmations=np.where(
            long_profile, confirmations, v4.LOCK_CONFIRMATIONS
        ).astype(np.int16),
        lock_quantiles=np.where(
            long_profile, quantile, v4.LOCK_QUANTILE
        ).astype(np.float32),
        instability_layers=np.full(profile_count, v4.INSTABILITY_LAYER),
        instability_widths=np.full(
            profile_count, v4.INSTABILITY_WIDTH, dtype=np.int16
        ),
        instability_confirmations=np.full(
            profile_count, v4.INSTABILITY_CONFIRMATIONS, dtype=np.int16
        ),
        instability_quantiles=np.full(
            profile_count, v4.INSTABILITY_QUANTILE, dtype=np.float32
        ),
    )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.long_guard_v5.alarms.v1"),
        representation=np.asarray(representation),
        width=np.asarray(width),
        confirmations=np.asarray(confirmations),
        quantile=np.asarray(quantile),
        main_selected_lock=main_selected_lock,
        main_deployed_lock=main_deployed_lock,
        main_hybrid=main_hybrid,
        main_selected_threshold=main_selected_threshold,
        external_selected_lock=external_lock,
        external_deployed_lock=external_deployed_lock,
        external_hybrid=external_hybrid,
        external_selected_threshold=external_threshold,
    )

    external_labels = v3.aligned_labels(
        external_cache,
        args.label_root / "external_8b_clean_labels.csv",
        "external_8b",
    )
    frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for cohort, labels, baseline, candidate in (
        ("development_main", development_labels, v4_main_dual, main_hybrid),
        ("external_8b", external_labels, v4_external_dual, external_hybrid),
    ):
        frames.append(
            tables(
                cohort,
                labels,
                {
                    "v4_dual_regime_or": baseline,
                    "long_guard_v5": candidate,
                },
            )
        )
    overall = pd.concat([frame[0] for frame in frames], ignore_index=True)
    by_suite = pd.concat([frame[1] for frame in frames], ignore_index=True)
    by_task = pd.concat([frame[2] for frame in frames], ignore_index=True)
    overall.to_csv(args.output / "outcome_metrics.csv", index=False)
    by_suite.to_csv(args.output / "outcome_metrics_by_suite.csv", index=False)
    by_task.to_csv(args.output / "outcome_metrics_by_task.csv", index=False)

    episodes = pd.concat(
        [
            development_labels.assign(
                first_v4_query=v4_main_dual,
                first_long_guard_v5_query=main_hybrid,
            ),
            external_labels.assign(
                first_v4_query=v4_external_dual,
                first_long_guard_v5_query=external_hybrid,
            ),
        ],
        ignore_index=True,
    )
    episodes.to_csv(args.output / "episode_alarms.csv", index=False)
    summary = {
        "schema": "himoe.long_guard_v5.evaluation.v1",
        "status": "development-selected post-hoc external replay; external 8B is not pristine",
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_rule_frozen_before_outcome_join": True,
        "selection": selection,
        "metrics": overall.to_dict(orient="records"),
        "artifacts": {
            "evaluator_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(PROTOCOL),
            "selection_sha256": sha256(args.output / "selection.json"),
            "sealed_first_alarms_sha256": sha256(
                args.output / "sealed_first_alarms.npz"
            ),
            "deployment_profiles_sha256": sha256(
                args.output / "deployment_profiles.npz"
            ),
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
