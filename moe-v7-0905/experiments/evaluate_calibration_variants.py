#!/usr/bin/env python3
"""Compare decision-line variants under the same leave-one-suite-out protocol.

LOSO established that v7's pooled trajectory-peak cuts move ~20% std when the
calibration corpus horizon mix changes, and LOTO established they barely move
when it does not. This script asks whether a different cut removes that
sensitivity without giving up detection.

Every variant is scored through the identical pipeline: the same score streams,
the same four folds, the same 640-point selection protocol on calibration-suite
development outcomes, and the same held-out external metrics. Only the shape of
the decision line differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import calibration_variants as variants  # noqa: E402
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
from intrinsic_guard_monitor import first_and, first_or  # noqa: E402
from select_operating_point import (  # noqa: E402
    ACCELERATION_QUANTILES,
    FREEZE_QUANTILES,
    PERIODICITY_QUANTILES,
    choose,
    metrics,
)


DEFAULT_OUTPUT = BUNDLE / "results/calibration_variants"
FREEZE_DEVIATIONS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
ACCELERATION_DEVIATIONS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
PERIODICITY_DEVIATIONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

# Which stream feeds calibration and which stream the line is applied to. v7
# calibrates on the smoothed stream and fires on the persistent stream; that
# asymmetry is preserved so the comparison isolates the cut.
STREAMS = {
    "freeze": ("freeze", "freeze"),
    "acceleration": ("acceleration", "acceleration_persistent"),
    "periodicity": ("periodicity", "periodicity_persistent"),
}
LEVEL_GRIDS = {
    "pooled_peak": (FREEZE_QUANTILES, ACCELERATION_QUANTILES, PERIODICITY_QUANTILES),
    "query_indexed": (FREEZE_QUANTILES, ACCELERATION_QUANTILES, PERIODICITY_QUANTILES),
    "self_normalized": (
        FREEZE_DEVIATIONS,
        ACCELERATION_DEVIATIONS,
        PERIODICITY_DEVIATIONS,
    ),
    "query_normalized_peak": (
        FREEZE_QUANTILES,
        ACCELERATION_QUANTILES,
        PERIODICITY_QUANTILES,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def prepare_stream(
    variant: str,
    calibration_scores: dict[str, np.ndarray],
    target_scores: dict[str, np.ndarray],
    stream: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the variant's score transform.

    Returns the corpus series the line is calibrated on, the target series the
    line fires against, and the target's smoothed series (which self_normalized
    needs for its prefix location and scale).

    v7 calibrates on the smoothed stream and fires on the persistent stream;
    that asymmetry is preserved for every variant so the comparison isolates
    the cut rather than the streams.
    """
    calibration_stream, target_stream = STREAMS[stream]
    corpus = calibration_scores[calibration_stream]
    target = target_scores[target_stream]
    smoothed = target_scores[calibration_stream]
    if variant == "query_normalized_peak":
        location, scale = variants.query_reference_location_scale(corpus)
        corpus = variants.query_normalized(corpus, location, scale)
        target = variants.query_normalized(target, location, scale)
        smoothed = variants.query_normalized(smoothed, location, scale)
    return corpus, target, smoothed


def build_lines(
    variant: str,
    corpus: np.ndarray,
    smoothed: np.ndarray,
    levels: tuple[float, ...],
) -> dict[float, Any]:
    """Decision line for every grid level of one prepared stream."""
    if variant in ("pooled_peak", "query_normalized_peak"):
        return {
            level: variants.pooled_peak_threshold(corpus, level) for level in levels
        }
    if variant == "query_indexed":
        return {
            level: variants.query_indexed_threshold(corpus, level) for level in levels
        }
    if variant == "self_normalized":
        centre, spread = variants.prefix_location_scale(smoothed)
        return {
            level: variants.self_normalized_threshold(centre, spread, level)
            for level in levels
        }
    raise ValueError(f"unknown variant: {variant}")


def variant_alarms(
    variant: str,
    calibration_scores: dict[str, np.ndarray],
    target_scores: dict[str, np.ndarray],
    valid: np.ndarray,
    levels: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """First-alarm arrays for one fully specified operating point."""
    firsts: dict[str, np.ndarray] = {}
    for stream, level in zip(STREAMS, levels, strict=True):
        corpus, target, smoothed = prepare_stream(
            variant, calibration_scores, target_scores, stream
        )
        line = build_lines(variant, corpus, smoothed, (level,))[level]
        firsts[stream] = variants.first_alarm(target, line, valid)
    turbulence = first_and(firsts["acceleration"], firsts["periodicity"])
    firsts["turbulence"] = turbulence
    firsts["guard"] = first_or(firsts["freeze"], turbulence)
    return firsts


def select_variant_operating_point(
    variant: str,
    calibration_scores: dict[str, np.ndarray],
    development_scores: dict[str, np.ndarray],
    valid: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Same 640-point protocol as v7, on the calibration suites only."""
    grids = LEVEL_GRIDS[variant]
    firsts: dict[str, dict[float, np.ndarray]] = {}
    for stream, grid in zip(STREAMS, grids, strict=True):
        corpus, target, smoothed = prepare_stream(
            variant, calibration_scores, development_scores, stream
        )
        firsts[stream] = {
            level: variants.first_alarm(target, line, valid)
            for level, line in build_lines(variant, corpus, smoothed, grid).items()
        }
    rows: list[dict[str, float | int]] = []
    for freeze_level, freeze_alarm in firsts["freeze"].items():
        for acceleration_level, acceleration_alarm in firsts["acceleration"].items():
            for periodicity_level, periodicity_alarm in firsts["periodicity"].items():
                guard = first_or(
                    freeze_alarm, first_and(acceleration_alarm, periodicity_alarm)
                )
                rows.append(
                    {
                        "freeze_level": freeze_level,
                        "acceleration_level": acceleration_level,
                        "periodicity_level": periodicity_level,
                        **metrics(guard, labels),
                    }
                )
    candidates = pd.DataFrame(rows)
    try:
        primary = choose(candidates, conservative=False)
    except RuntimeError:
        primary = None
    return candidates, primary


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

    main_labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    )

    sealed: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {}
    fold_state: list[dict[str, Any]] = []

    for held_out in loso_folds.SUITES:
        calibration = ref_suite != held_out
        scale = loso_folds.periodicity_scale_of(ref_periodicity[calibration])
        calibration_scores = loso_folds.cohort_scores(
            ref_mobility[calibration],
            ref_acceleration[calibration],
            ref_periodicity[calibration],
            scale,
        )
        development_rows = is_main[calibration]
        development_scores = {
            name: values[development_rows]
            for name, values in calibration_scores.items()
        }
        development_keep = main_suite != held_out
        development_labels = main_labels.loc[development_keep].reset_index(drop=True)
        development_valid = main_valid[development_keep]

        evaluation = external_suite == held_out
        evaluation_scores = loso_folds.cohort_scores(
            external_mobility[evaluation],
            external_acceleration[evaluation],
            external_periodicity[evaluation],
            scale,
        )

        for variant in variants.VARIANTS:
            candidates, primary = select_variant_operating_point(
                variant,
                calibration_scores,
                development_scores,
                development_valid,
                development_labels,
            )
            candidates.to_csv(
                output / f"candidates_{variant}_{held_out}.csv", index=False
            )
            feasible = primary is not None
            selection[f"{held_out}__{variant}"] = {
                "feasible": feasible,
                "candidate_count": int(len(candidates)),
                "development_episodes": int(len(development_labels)),
                "primary": None if primary is None else plain(primary.to_dict()),
            }
            if not feasible:
                continue
            levels = (
                float(primary["freeze_level"]),
                float(primary["acceleration_level"]),
                float(primary["periodicity_level"]),
            )
            alarms = variant_alarms(
                variant,
                calibration_scores,
                evaluation_scores,
                external_valid[evaluation],
                levels,
            )
            sealed[f"{held_out}__{variant}__guard"] = alarms["guard"]
            fold_state.append(
                {
                    "held_out": held_out,
                    "variant": variant,
                    "evaluation": evaluation,
                    "guard": alarms["guard"],
                    "levels": levels,
                }
            )
        print(f"fold {held_out}: {len(variants.VARIANTS)} variants scored", flush=True)

    np.savez_compressed(
        output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.variants.alarms.v1"),
        **sealed,
    )
    (output / "variant_selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.variants.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "variants": list(variants.VARIANTS),
        "held_out_outcomes_used_for_calibration": False,
        "held_out_outcomes_used_for_selection": False,
        "external_outcomes_loaded_after_seal": True,
        "artifacts": {
            "sealed_first_alarms_sha256": sha256(output / "sealed_first_alarms.npz"),
            "variant_selection_sha256": sha256(output / "variant_selection.json"),
            "variants_module_sha256": sha256(HERE / "calibration_variants.py"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "variants_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("sealed all variants; opening held-out outcomes now", flush=True)

    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for state in fold_state:
        labels = external_labels.loc[state["evaluation"]].reset_index(drop=True)
        row = metric_row(
            "external_8b",
            f"{state['held_out']}__{state['variant']}",
            state["guard"],
            labels,
            args.bootstrap,
            rng,
            group=state["held_out"],
        )
        row.update(
            {
                "held_out_suite": state["held_out"],
                "variant": state["variant"],
                "freeze_level": state["levels"][0],
                "acceleration_level": state["levels"][1],
                "periodicity_level": state["levels"][2],
            }
        )
        rows.append(row)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(output / "variant_fold_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants.VARIANTS:
        block = fold_metrics[fold_metrics["variant"] == variant]
        if block.empty:
            summary_rows.append({"variant": variant, "folds": 0})
            continue
        tp, fp = int(block["tp"].sum()), int(block["fp"].sum())
        risk, timely = int(block["risk_n"].sum()), int(block["timely_n"].sum())
        summary_rows.append(
            {
                "variant": variant,
                "folds": int(len(block)),
                "macro_risk_recall": float(block["risk_recall"].mean()),
                "macro_precision": float(block["precision"].mean()),
                "macro_timely_fpr": float(block["timely_fpr"].mean()),
                "macro_early4": float(block["early4_risk_recall"].mean()),
                "micro_tp": tp,
                "micro_fp": fp,
                "micro_risk_recall": tp / risk if risk else float("nan"),
                "micro_precision": tp / (tp + fp) if tp + fp else float("nan"),
                "micro_timely_fpr": fp / timely if timely else float("nan"),
                "worst_fold_precision": float(block["precision"].min()),
                "worst_fold_timely_fpr": float(block["timely_fpr"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "variant_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
