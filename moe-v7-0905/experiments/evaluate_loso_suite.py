#!/usr/bin/env python3
"""Leave-one-suite-out validation of the task-agnostic intrinsic guard."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import loso_folds  # noqa: E402
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_FEATURE,
    EXTERNAL_LAYER,
    EXTRA_FEATURE,
    EXTRA_LAYER,
    LABEL_ROOT,
    MAIN_FEATURE,
    MAIN_LAYER,
    aligned_labels,
    assert_aligned,
    combine_reference,
    feature,
    load_npz,
    metric_row,
    plain,
    sha256,
)


DEFAULT_OUTPUT = BUNDLE / "results/loso_validation"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
FIRST_FREEZE_QUERY = 6
FIRST_TURBULENCE_QUERY = 10
LEVELS = ("published", "loso_l1", "loso_l2")
ALARM_NAMES = ("freeze", "acceleration", "periodicity", "turbulence", "guard")
DRIFT_CONSTANTS = (
    "freeze_threshold",
    "acceleration_threshold",
    "periodicity_threshold",
    "periodicity_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def eligible_counts(length: np.ndarray) -> dict[str, int]:
    """Episodes long enough for each branch to be able to fire at all."""
    return {
        "eligible_freeze_n": int((length >= FIRST_FREEZE_QUERY + 1).sum()),
        "eligible_turbulence_n": int((length >= FIRST_TURBULENCE_QUERY + 1).sum()),
    }


def macro_summary(metrics: pd.DataFrame, level: str) -> dict[str, float | int]:
    """Equal weight per fold. Infeasible folds are excluded and counted."""
    block = metrics[(metrics["level"] == level) & metrics["feasible"].astype(bool)]
    if block.empty:
        return {"folds": 0}
    return {
        "folds": int(len(block)),
        "risk_recall": float(block["risk_recall"].mean()),
        "precision": float(block["precision"].mean()),
        "timely_fpr": float(block["timely_fpr"].mean()),
        "early4_risk_recall": float(block["early4_risk_recall"].mean()),
    }


def micro_summary(metrics: pd.DataFrame, level: str) -> dict[str, float | int]:
    """Episode-weighted pooling across folds."""
    block = metrics[(metrics["level"] == level) & metrics["feasible"].astype(bool)]
    if block.empty:
        return {"folds": 0}
    tp = int(block["tp"].sum())
    fp = int(block["fp"].sum())
    risk = int(block["risk_n"].sum())
    timely = int(block["timely_n"].sum())
    return {
        "folds": int(len(block)),
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / risk if risk else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "timely_fpr": fp / timely if timely else float("nan"),
    }


def write_metrics(
    args: argparse.Namespace,
    output: Path,
    fold_state: dict[str, dict[str, Any]],
    external_layer: dict[str, np.ndarray],
) -> None:
    """Open held-out outcomes and score the three levels side by side."""
    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []

    for held_out, state in fold_state.items():
        evaluation = state["evaluation"]
        labels = external_labels.loc[evaluation].reset_index(drop=True)
        counts = eligible_counts(labels["length"].to_numpy(int))
        frame = labels.copy()
        frame["held_out_suite"] = held_out
        for level in LEVELS:
            if level not in state["alarms"]:
                rows.append(
                    {
                        "held_out_suite": held_out,
                        "level": level,
                        "feasible": False,
                        "episodes": int(len(labels)),
                        **counts,
                    }
                )
                continue
            guard = state["alarms"][level]["guard"]
            row = metric_row(
                "external_8b",
                f"{held_out}__{level}",
                guard,
                labels,
                args.bootstrap,
                rng,
                group=held_out,
            )
            row.update({"held_out_suite": held_out, "level": level, "feasible": True})
            row.update(counts)
            if level in state["constants"]:
                constants = state["constants"][level]
                row.update(
                    {
                        "freeze_quantile": constants.freeze_quantile,
                        "acceleration_quantile": constants.acceleration_quantile,
                        "periodicity_quantile": constants.periodicity_quantile,
                    }
                )
            rows.append(row)
            frame[f"first_{level}_query"] = guard

            tasks = labels["task"].to_numpy(str)
            for task in np.unique(tasks):
                take = tasks == task
                task_row = metric_row(
                    "external_8b",
                    f"{held_out}__{level}",
                    guard[take],
                    labels.loc[take].reset_index(drop=True),
                    args.bootstrap,
                    rng,
                    group=task,
                    intervals=False,
                )
                task_row.update(
                    {"held_out_suite": held_out, "level": level, "task": task}
                )
                task_rows.append(task_row)
        episode_frames.append(frame)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "loso_metrics.csv", index=False)
    pd.DataFrame(task_rows).to_csv(output / "loso_metrics_by_task.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(
        output / "episode_alarms.csv", index=False
    )

    # The drift table carries the metric consequence of each drift, so a
    # threshold move can be read against what it actually cost.
    drift = pd.read_csv(output / "threshold_drift.csv")
    keyed = metrics[metrics["feasible"].astype(bool)].set_index(
        ["held_out_suite", "level"]
    )
    baseline = keyed.xs("published", level="level")
    for column in ("risk_recall", "precision", "timely_fpr"):
        drift[f"{column}_vs_published"] = [
            float(keyed.loc[(suite, level), column] - baseline.loc[suite, column])
            if (suite, level) in keyed.index
            else float("nan")
            for suite, level in zip(drift["held_out_suite"], drift["level"], strict=True)
        ]
    drift.to_csv(output / "threshold_drift.csv", index=False)

    summary = {
        "schema": "himoe.intrinsic_guard_v7.loso.evaluation.v1",
        "macro": {level: macro_summary(metrics, level) for level in LEVELS},
        "micro": {level: micro_summary(metrics, level) for level in LEVELS},
        "infeasible_folds": sorted(
            metrics.loc[~metrics["feasible"].astype(bool), "held_out_suite"]
            .unique()
            .tolist()
        ),
    }
    (output / "loso_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = output / "loso_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["threshold_drift_sha256"] = sha256(
        output / "threshold_drift.csv"
    )
    manifest["artifacts"]["loso_metrics_sha256"] = sha256(output / "loso_metrics.csv")
    manifest_path.write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        metrics[
            [
                "held_out_suite",
                "level",
                "feasible",
                "tp",
                "fp",
                "risk_recall",
                "precision",
                "timely_fpr",
                "early4_risk_recall",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    external_layer = load_npz(EXTERNAL_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    extra_feature = load_npz(EXTRA_FEATURE)
    external_feature = load_npz(EXTERNAL_FEATURE)
    assert_aligned(main_layer, main_feature, "main")
    assert_aligned(extra_layer, extra_feature, "extra")
    assert_aligned(external_layer, external_feature, "external")

    ref_mobility, ref_acceleration, ref_periodicity = combine_reference(
        main_layer, extra_layer, main_feature, extra_feature
    )
    ref_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    is_main = np.arange(len(ref_suite)) < len(main_layer["task_index"])
    main_suite = loso_folds.suite_of(main_layer)
    main_valid = main_layer["valid"].astype(bool)
    external_suite = loso_folds.suite_of(external_layer)
    external_valid = external_layer["valid"].astype(bool)
    external_mobility = external_layer["mobility"]
    external_acceleration = feature(external_feature, "route_acceleration")
    external_periodicity = feature(external_feature, "lag_periodicity")

    # Development outcomes for the calibration suites only; each fold slices
    # this frame before it is used and never reads the held-out rows.
    main_labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    )
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        published_alarms = {
            name: np.asarray(sealed[f"external_{name}"]) for name in ALARM_NAMES
        }
    with np.load(PUBLISHED / "global_profile.npz", allow_pickle=False) as published:
        baseline_constants = {
            name: float(published[name]) for name in DRIFT_CONSTANTS
        }

    profiles: dict[str, np.ndarray] = {}
    fold_alarms: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {}
    drift_rows: list[dict[str, Any]] = []
    fold_state: dict[str, dict[str, Any]] = {}

    for held_out in loso_folds.SUITES:
        calibration = ref_suite != held_out
        scale = loso_folds.periodicity_scale_of(ref_periodicity[calibration])
        calibration_scores = loso_folds.cohort_scores(
            ref_mobility[calibration],
            ref_acceleration[calibration],
            ref_periodicity[calibration],
            scale,
        )
        peaks = loso_folds.peaks_from_scores(calibration_scores)

        # Development rows are exactly the calibration rows that came from main,
        # in main order, so they are sliced rather than recomputed.
        development_rows = is_main[calibration]
        development_scores = {
            name: values[development_rows]
            for name, values in calibration_scores.items()
        }
        development_keep = main_suite != held_out
        development_labels = main_labels.loc[development_keep].reset_index(drop=True)
        candidates, primary = loso_folds.select_fold_operating_point(
            peaks,
            development_scores,
            main_valid[development_keep],
            development_labels,
        )
        candidates.to_csv(output / f"fold_candidates_{held_out}.csv", index=False)

        constants = {
            "loso_l1": loso_folds.constants_from_peaks(
                peaks, scale, *loso_folds.PUBLISHED_QUANTILES
            )
        }
        if primary is not None:
            constants["loso_l2"] = loso_folds.constants_from_peaks(
                peaks,
                scale,
                float(primary["freeze_quantile"]),
                float(primary["acceleration_quantile"]),
                float(primary["periodicity_quantile"]),
            )
        selection[held_out] = {
            "development_episodes": int(len(development_labels)),
            "calibration_episodes": int(calibration.sum()),
            "calibration_suites": sorted(set(loso_folds.SUITES) - {held_out}),
            "candidate_count": int(len(candidates)),
            "feasible": primary is not None,
            "primary": None if primary is None else plain(primary.to_dict()),
        }

        evaluation = external_suite == held_out
        alarms: dict[str, dict[str, np.ndarray]] = {
            "published": {
                name: values[evaluation] for name, values in published_alarms.items()
            }
        }
        for level, fold_constants in constants.items():
            scores = loso_folds.cohort_scores(
                external_mobility[evaluation],
                external_acceleration[evaluation],
                external_periodicity[evaluation],
                fold_constants.periodicity_scale,
            )
            alarms[level] = loso_folds.guard_alarms(
                scores, external_valid[evaluation], fold_constants
            )
            for field, value in asdict(fold_constants).items():
                profiles[f"{held_out}__{level}__{field}"] = np.asarray(
                    value, dtype=np.float64
                )
        for level, level_alarms in alarms.items():
            for name, values in level_alarms.items():
                fold_alarms[f"{held_out}__{level}__{name}"] = values

        fold_state[held_out] = {
            "evaluation": evaluation,
            "alarms": alarms,
            "constants": constants,
        }

        for level, fold_constants in constants.items():
            for field, value in baseline_constants.items():
                fold_value = float(getattr(fold_constants, field))
                drift_rows.append(
                    {
                        "held_out_suite": held_out,
                        "level": level,
                        "constant": field,
                        "published_value": value,
                        "fold_value": fold_value,
                        "absolute_drift": fold_value - value,
                        "relative_drift_percent": 100.0 * (fold_value - value) / value
                        if value != 0.0
                        else float("nan"),
                    }
                )
        print(
            f"fold {held_out}: calibrated on {int(calibration.sum()):,} trajectories, "
            f"feasible={primary is not None}",
            flush=True,
        )

    np.savez_compressed(
        output / "fold_profiles.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loso.profiles.v1"),
        **profiles,
    )
    np.savez_compressed(
        output / "fold_first_alarms.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loso.alarms.v1"),
        **fold_alarms,
    )
    pd.DataFrame(drift_rows).to_csv(output / "threshold_drift.csv", index=False)
    (output / "fold_selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.loso.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "folds": list(loso_folds.SUITES),
        "held_out_outcomes_used_for_calibration": False,
        "held_out_outcomes_used_for_selection": False,
        "external_outcomes_loaded_after_seal": True,
        "external_cohort_pristine_holdout": False,
        "artifacts": {
            "fold_profiles_sha256": sha256(output / "fold_profiles.npz"),
            "fold_first_alarms_sha256": sha256(output / "fold_first_alarms.npz"),
            "threshold_drift_sha256": sha256(output / "threshold_drift.csv"),
            "fold_selection_sha256": sha256(output / "fold_selection.json"),
            "folds_module_sha256": sha256(HERE / "loso_folds.py"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "loso_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("sealed all folds; opening held-out outcomes now", flush=True)

    write_metrics(args, output, fold_state, external_layer)


if __name__ == "__main__":
    main()
