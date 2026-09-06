#!/usr/bin/env python3
"""Leave-one-task-out validation of the task-agnostic intrinsic guard.

Companion to evaluate_loso_suite.py. LOSO answers "unseen suite"; this answers
"unseen task, seen suite" — the weaker but more common deployment case.

The natural LOTO estimator is pooled: every held-out fold contributes its own
400 external episodes, each scored by a profile calibrated without that task.
Concatenating the folds gives a full-cohort number directly comparable to the
published 439 TP / 80 FP.
"""

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


DEFAULT_OUTPUT = BUNDLE / "results/loto_validation"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
LEVELS = ("published", "loto_l1", "loto_l2")
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
    ref_task = np.concatenate(
        (loso_folds.task_of(main_layer), loso_folds.task_of(extra_layer))
    )
    is_main = np.arange(len(ref_task)) < len(main_layer["task_index"])
    main_task = loso_folds.task_of(main_layer)
    main_valid = main_layer["valid"].astype(bool)
    external_task = loso_folds.task_of(external_layer)
    external_valid = external_layer["valid"].astype(bool)
    external_mobility = external_layer["mobility"]
    external_acceleration = feature(external_feature, "route_acceleration")
    external_periodicity = feature(external_feature, "lag_periodicity")

    main_labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    )
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        published_alarms = {
            name: np.asarray(sealed[f"external_{name}"]) for name in ALARM_NAMES
        }
    with np.load(PUBLISHED / "global_profile.npz", allow_pickle=False) as published:
        baseline_constants = {name: float(published[name]) for name in DRIFT_CONSTANTS}

    tasks = sorted(set(ref_task.tolist()))
    profiles: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {}
    drift_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    infeasible: list[str] = []

    # Pooled held-out predictions, one slot per external episode.
    pooled = {
        level: np.full(len(external_task), -1, dtype=np.int16)
        for level in LEVELS
    }
    covered = {level: np.zeros(len(external_task), dtype=bool) for level in LEVELS}

    for held_out in tasks:
        calibration = ref_task != held_out
        scale = loso_folds.periodicity_scale_of(ref_periodicity[calibration])
        calibration_scores = loso_folds.cohort_scores(
            ref_mobility[calibration],
            ref_acceleration[calibration],
            ref_periodicity[calibration],
            scale,
        )
        peaks = loso_folds.peaks_from_scores(calibration_scores)

        development_rows = is_main[calibration]
        development_scores = {
            name: values[development_rows]
            for name, values in calibration_scores.items()
        }
        development_keep = main_task != held_out
        development_labels = main_labels.loc[development_keep].reset_index(drop=True)
        _, primary = loso_folds.select_fold_operating_point(
            peaks,
            development_scores,
            main_valid[development_keep],
            development_labels,
        )

        constants = {
            "loto_l1": loso_folds.constants_from_peaks(
                peaks, scale, *loso_folds.PUBLISHED_QUANTILES
            )
        }
        if primary is not None:
            constants["loto_l2"] = loso_folds.constants_from_peaks(
                peaks,
                scale,
                float(primary["freeze_quantile"]),
                float(primary["acceleration_quantile"]),
                float(primary["periodicity_quantile"]),
            )
        else:
            infeasible.append(held_out)

        evaluation = external_task == held_out
        selection[held_out] = {
            "calibration_episodes": int(calibration.sum()),
            "development_episodes": int(len(development_labels)),
            "evaluation_episodes": int(evaluation.sum()),
            "held_out_in_calibration": bool((ref_task[calibration] == held_out).any()),
            "feasible": primary is not None,
            "primary": None if primary is None else plain(primary.to_dict()),
        }
        for level, fold_constants in constants.items():
            for field, value in asdict(fold_constants).items():
                profiles[f"{held_out}__{level}__{field}"] = np.asarray(
                    value, dtype=np.float64
                )
            for field, value in baseline_constants.items():
                fold_value = float(getattr(fold_constants, field))
                drift_rows.append(
                    {
                        "held_out_task": held_out,
                        "level": level,
                        "constant": field,
                        "published_value": value,
                        "fold_value": fold_value,
                        "relative_drift_percent": 100.0 * (fold_value - value) / value
                        if value != 0.0
                        else float("nan"),
                    }
                )

        if not evaluation.any():
            # The one reference task with no external replay. Logged, not hidden.
            skipped.append(held_out)
            continue

        pooled["published"][evaluation] = published_alarms["guard"][evaluation]
        covered["published"][evaluation] = True
        for level, fold_constants in constants.items():
            scores = loso_folds.cohort_scores(
                external_mobility[evaluation],
                external_acceleration[evaluation],
                external_periodicity[evaluation],
                fold_constants.periodicity_scale,
            )
            alarms = loso_folds.guard_alarms(
                scores, external_valid[evaluation], fold_constants
            )
            pooled[level][evaluation] = alarms["guard"]
            covered[level][evaluation] = True

    print(
        f"{len(tasks)} folds; {len(skipped)} without external replay; "
        f"{len(infeasible)} infeasible",
        flush=True,
    )

    np.savez_compressed(
        output / "fold_profiles.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loto.profiles.v1"),
        **profiles,
    )
    np.savez_compressed(
        output / "pooled_first_alarms.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.loto.alarms.v1"),
        **{f"{level}_guard": pooled[level] for level in LEVELS},
        **{f"{level}_covered": covered[level] for level in LEVELS},
    )
    pd.DataFrame(drift_rows).to_csv(output / "threshold_drift.csv", index=False)
    (output / "fold_selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.loto.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "reference_tasks": len(tasks),
        "folds_without_external_replay": skipped,
        "infeasible_folds": infeasible,
        "held_out_outcomes_used_for_calibration": False,
        "held_out_outcomes_used_for_selection": False,
        "external_outcomes_loaded_after_seal": True,
        "external_cohort_pristine_holdout": False,
        "artifacts": {
            "fold_profiles_sha256": sha256(output / "fold_profiles.npz"),
            "pooled_first_alarms_sha256": sha256(output / "pooled_first_alarms.npz"),
            "threshold_drift_sha256": sha256(output / "threshold_drift.csv"),
            "fold_selection_sha256": sha256(output / "fold_selection.json"),
            "folds_module_sha256": sha256(HERE / "loso_folds.py"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "loto_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("sealed all folds; opening held-out outcomes now", flush=True)

    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        take = covered[level]
        block = external_labels.loc[take].reset_index(drop=True)
        row = metric_row(
            "external_8b", level, pooled[level][take], block, args.bootstrap, rng
        )
        row.update(
            {
                "level": level,
                "covered_episodes": int(take.sum()),
                "covered_tasks": int(len(np.unique(external_task[take]))),
            }
        )
        rows.append(row)
        names = external_task[take]
        for task in np.unique(names):
            inner = names == task
            task_row = metric_row(
                "external_8b",
                level,
                pooled[level][take][inner],
                block.loc[inner].reset_index(drop=True),
                args.bootstrap,
                rng,
                group=task,
                intervals=False,
            )
            task_row["level"] = level
            task_rows.append(task_row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "loto_metrics.csv", index=False)
    task_frame = pd.DataFrame(task_rows)
    task_frame.to_csv(output / "loto_metrics_by_task.csv", index=False)

    summary = {
        "schema": "himoe.intrinsic_guard_v7.loto.evaluation.v1",
        "pooled": {
            level: plain(metrics[metrics["level"] == level].iloc[0].to_dict())
            for level in LEVELS
        },
        "macro_over_tasks": {
            level: {
                "tasks": int((task_frame["level"] == level).sum()),
                "risk_recall": float(
                    task_frame.loc[task_frame["level"] == level, "risk_recall"].mean()
                ),
                "precision": float(
                    task_frame.loc[task_frame["level"] == level, "precision"].mean()
                ),
                "timely_fpr": float(
                    task_frame.loc[task_frame["level"] == level, "timely_fpr"].mean()
                ),
            }
            for level in LEVELS
        },
        "folds_without_external_replay": skipped,
        "infeasible_folds": infeasible,
    }
    (output / "loto_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        metrics[
            [
                "level",
                "covered_episodes",
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


if __name__ == "__main__":
    main()
