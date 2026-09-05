#!/usr/bin/env python3
"""Select a lead-aware non-long lock and replay the final task-conditional rule."""

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
import evaluate_long_guard_v5 as long_guard


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUNDLE = HERE.parent
CANONICAL = WORKSPACE / "double-selete/trainfree"
PROTOCOL = BUNDLE / "method/ONLINE_LEAD_GUARD_V5_PROTOCOL.md"
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = CANONICAL / "results/timeout_extension_plus10"
DEFAULT_V4_RESULT = BUNDLE / "results/cache_new_v4"
DEFAULT_LONG_RESULT = BUNDLE / "results/long_guard_v5"
DEFAULT_OUTPUT = BUNDLE / "results/lead_guard_v5"

REPRESENTATIONS = ("all_median", "L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
OVERALL_FPR_LIMIT = 0.005
SUITE_FPR_LIMIT = 0.01
TASK_FPR_LIMIT = 0.025
PRECISION_MIN = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--v4-result", type=Path, default=DEFAULT_V4_RESULT)
    parser.add_argument("--long-result", type=Path, default=DEFAULT_LONG_RESULT)
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


def group_fpr(
    alarm: np.ndarray,
    timely: np.ndarray,
    group_index: np.ndarray,
    group_count: int,
) -> np.ndarray:
    denominator = np.bincount(group_index[timely], minlength=group_count)
    numerator = np.bincount(group_index[alarm & timely], minlength=group_count)
    return np.divide(
        numerator,
        denominator,
        out=np.full(group_count, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def candidate_row(
    representation: str,
    width: int,
    confirmations: int,
    quantile: float,
    first: np.ndarray,
    labels: pd.DataFrame,
    task_index: np.ndarray,
    suite_index: np.ndarray,
) -> dict[str, Any]:
    row = long_guard.metric_row("development_main", "candidate", first, labels)
    alarm = np.asarray(first) >= 0
    timely = ~labels["original_failure"].to_numpy(dtype=bool)
    suite_rates = group_fpr(
        alarm, timely, suite_index, int(suite_index.max()) + 1
    )
    task_rates = group_fpr(alarm, timely, task_index, int(task_index.max()) + 1)
    row.update(
        {
            "representation": representation,
            "direction": "low",
            "width": width,
            "confirmations": confirmations,
            "quantile": quantile,
            "max_suite_fpr": float(np.nanmax(suite_rates)),
            "max_task_fpr": float(np.nanmax(task_rates)),
        }
    )
    row["eligible"] = bool(
        row["timely_fpr"] <= OVERALL_FPR_LIMIT
        and row["max_suite_fpr"] <= SUITE_FPR_LIMIT
        and row["max_task_fpr"] <= TASK_FPR_LIMIT
        and row["precision"] >= PRECISION_MIN
    )
    return row


def select_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    eligible = candidates[candidates["eligible"]]
    if eligible.empty:
        return None
    return eligible.sort_values(
        [
            "early4_risk_recall",
            "early8_risk_recall",
            "risk_recall",
            "timely_fpr",
            "precision",
            "quantile",
            "confirmations",
            "width",
        ],
        ascending=[False, False, False, True, False, False, False, False],
    ).iloc[0]


def development_lock_components(
    cache: dict[str, np.ndarray], representation: str, width: int
) -> tuple[np.ndarray, np.ndarray]:
    instant = long_guard.oriented_instant(cache, representation, width)
    thresholds = dev.crossfit_thresholds(
        dev.row_max(instant),
        cache["task_index"].astype(int),
        cache["init_state_id"].astype(int),
    )
    return instant, thresholds


def development_first(
    cache: dict[str, np.ndarray],
    representation: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    instant, thresholds = development_lock_components(cache, representation, width)
    persistent = dev.persistent_score(instant, confirmations)
    position = long_guard.quantile_position(quantile)
    selected_threshold = thresholds[:, position]
    first = v4.first_from_threshold(
        persistent, cache["valid"].astype(bool), selected_threshold
    )
    return first, selected_threshold


def scan_development(
    cache: dict[str, np.ndarray],
    labels: pd.DataFrame,
    long_hybrid: np.ndarray,
    instability: np.ndarray,
) -> pd.DataFrame:
    task_index = cache["task_index"].astype(int)
    suite_index = pd.factorize(labels["suite"], sort=True)[0]
    non_long = ~labels["suite"].eq(long_guard.LONG_SUITE).to_numpy()
    valid = cache["valid"].astype(bool)
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for width in dev.WIDTHS:
            instant, thresholds = development_lock_components(
                cache, representation, width
            )
            for confirmations in dev.CONFIRMATIONS:
                persistent = dev.persistent_score(instant, confirmations)
                for position, quantile in enumerate(dev.QUANTILES):
                    lock = v4.first_from_threshold(
                        persistent, valid, thresholds[:, position]
                    )
                    hybrid = long_hybrid.copy()
                    hybrid[non_long] = v4.first_or(
                        lock[non_long], instability[non_long]
                    )
                    rows.append(
                        candidate_row(
                            representation,
                            width,
                            confirmations,
                            quantile,
                            hybrid,
                            labels,
                            task_index,
                            suite_index,
                        )
                    )
    return pd.DataFrame(rows)


def save_deployment_profile(
    output: Path,
    long_result: Path,
    external_cache: dict[str, np.ndarray],
    threshold_rows: list[dict[str, Any]],
    representation: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> None:
    with np.load(
        long_result / "deployment_profiles.npz", allow_pickle=False
    ) as archive:
        base = {name: np.asarray(archive[name]) for name in archive.files}
    tasks = external_cache["task_names"].astype(str)
    if not np.array_equal(tasks, base["task_names"].astype(str)):
        raise ValueError("long-guard profile task order does not match external cache")
    candidate_thresholds = np.asarray(
        [row["threshold"] for row in threshold_rows], dtype=np.float32
    )
    non_long = ~np.char.startswith(tasks, f"{long_guard.LONG_SUITE}/")
    lock_thresholds = base["lock_thresholds"].astype(np.float32, copy=True)
    lock_representations = base["lock_representations"].astype(str, copy=True)
    lock_widths = base["lock_widths"].astype(np.int16, copy=True)
    lock_confirmations = base["lock_confirmations"].astype(np.int16, copy=True)
    lock_quantiles = base["lock_quantiles"].astype(np.float32, copy=True)
    lock_thresholds[non_long] = candidate_thresholds[non_long]
    lock_representations[non_long] = representation
    lock_widths[non_long] = width
    lock_confirmations[non_long] = confirmations
    lock_quantiles[non_long] = quantile
    np.savez_compressed(
        output / "deployment_profiles.npz",
        schema=np.asarray("himoe.lead_guard_v5.profile.v1"),
        task_names=tasks,
        lock_thresholds=lock_thresholds,
        instability_thresholds=base["instability_thresholds"],
        lock_representations=lock_representations,
        lock_widths=lock_widths,
        lock_confirmations=lock_confirmations,
        lock_quantiles=lock_quantiles,
        instability_layers=base["instability_layers"],
        instability_widths=base["instability_widths"],
        instability_confirmations=base["instability_confirmations"],
        instability_quantiles=base["instability_quantiles"],
    )


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
        main_instability = np.asarray(archive["main_instability"])
        external_instability = np.asarray(archive["external_instability"])
        main_v4 = np.asarray(archive["main_dual"])
        external_v4 = np.asarray(archive["external_dual"])
    with np.load(
        args.long_result / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        main_long = np.asarray(archive["main_hybrid"])
        external_long = np.asarray(archive["external_hybrid"])
        main_long_lock = np.asarray(archive["main_deployed_lock"])
        external_long_lock = np.asarray(archive["external_deployed_lock"])

    candidates = scan_development(
        main_cache, development_labels, main_long, main_instability
    )
    selected = select_candidate(candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output / "development_candidates.csv", index=False)
    selection = {
        "schema": "himoe.lead_guard_v5.selection.v1",
        "status": "selected" if selected is not None else "rejected_no_eligible_candidate",
        "external_outcomes_read_by_selection": False,
        "candidate_count": len(candidates),
        "eligible_count": int(candidates["eligible"].sum()),
        "constraints": {
            "overall_fpr_max": OVERALL_FPR_LIMIT,
            "suite_fpr_max": SUITE_FPR_LIMIT,
            "task_fpr_max": TASK_FPR_LIMIT,
            "precision_min": PRECISION_MIN,
        },
        "selected": selected.to_dict() if selected is not None else None,
    }
    (args.output / "selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if selected is None:
        raise RuntimeError("no lead-guard candidate satisfied the frozen constraints")

    representation = str(selected["representation"])
    width = int(selected["width"])
    confirmations = int(selected["confirmations"])
    quantile = float(selected["quantile"])
    main_lock, main_threshold = development_first(
        main_cache, representation, width, confirmations, quantile
    )
    non_long_main = ~development_labels["suite"].eq(
        long_guard.LONG_SUITE
    ).to_numpy()
    main_deployed_lock = main_long_lock.copy()
    main_deployed_lock[non_long_main] = main_lock[non_long_main]
    main_hybrid = main_long.copy()
    main_hybrid[non_long_main] = v4.first_or(
        main_lock[non_long_main], main_instability[non_long_main]
    )

    external_lock, external_threshold, threshold_rows = (
        long_guard.external_selected_lock(
            external_cache,
            main_cache,
            extra_cache,
            representation,
            width,
            confirmations,
            quantile,
        )
    )
    external_episode_tasks = external_cache["task_names"].astype(str)[
        external_cache["task_index"].astype(int)
    ]
    non_long_external = ~np.char.startswith(
        external_episode_tasks, f"{long_guard.LONG_SUITE}/"
    )
    external_deployed_lock = external_long_lock.copy()
    external_deployed_lock[non_long_external] = external_lock[non_long_external]
    external_hybrid = external_long.copy()
    external_hybrid[non_long_external] = v4.first_or(
        external_lock[non_long_external], external_instability[non_long_external]
    )

    for row in threshold_rows:
        row["active_in_profile"] = not str(row["task"]).startswith(
            f"{long_guard.LONG_SUITE}/"
        )
    pd.DataFrame(threshold_rows).to_csv(
        args.output / "external_outcome_blind_thresholds.csv", index=False
    )
    save_deployment_profile(
        args.output,
        args.long_result,
        external_cache,
        threshold_rows,
        representation,
        width,
        confirmations,
        quantile,
    )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.lead_guard_v5.alarms.v1"),
        representation=np.asarray(representation),
        width=np.asarray(width),
        confirmations=np.asarray(confirmations),
        quantile=np.asarray(quantile),
        main_selected_lock=main_lock,
        main_deployed_lock=main_deployed_lock,
        main_hybrid=main_hybrid,
        main_selected_threshold=main_threshold,
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
    for cohort, labels, v4_alarm, long_alarm, lead_alarm in (
        (
            "development_main",
            development_labels,
            main_v4,
            main_long,
            main_hybrid,
        ),
        (
            "external_8b",
            external_labels,
            external_v4,
            external_long,
            external_hybrid,
        ),
    ):
        frames.append(
            long_guard.tables(
                cohort,
                labels,
                {
                    "v4_dual_regime_or": v4_alarm,
                    "long_guard_v5": long_alarm,
                    "lead_guard_v5": lead_alarm,
                },
            )
        )
    overall = pd.concat([frame[0] for frame in frames], ignore_index=True)
    by_suite = pd.concat([frame[1] for frame in frames], ignore_index=True)
    by_task = pd.concat([frame[2] for frame in frames], ignore_index=True)
    overall.to_csv(args.output / "outcome_metrics.csv", index=False)
    by_suite.to_csv(args.output / "outcome_metrics_by_suite.csv", index=False)
    by_task.to_csv(args.output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(
        [
            development_labels.assign(
                first_v4_query=main_v4,
                first_long_guard_v5_query=main_long,
                first_lead_guard_v5_query=main_hybrid,
            ),
            external_labels.assign(
                first_v4_query=external_v4,
                first_long_guard_v5_query=external_long,
                first_lead_guard_v5_query=external_hybrid,
            ),
        ],
        ignore_index=True,
    ).to_csv(args.output / "episode_alarms.csv", index=False)

    summary = {
        "schema": "himoe.lead_guard_v5.evaluation.v1",
        "status": (
            "post-hoc iteration on previously inspected external 8B; "
            "requires pristine confirmation"
        ),
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_outcomes_loaded_after_alarm_seal_in_evaluator": True,
        "external_cohort_pristine_holdout": False,
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
