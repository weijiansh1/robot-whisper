#!/usr/bin/env python3
"""Does length-stratified reweighting close the LOSO gap?

Four calibration levels on the same four folds, same scores, same published
quantile levels (0.975 / 0.70 / 0.65). Only the reference population changes.

  published        full 16,000-trajectory corpus, unweighted. Contains the
                   held-out suite, so this is the leaky reference point.
  loso_l1          the 12,000-trajectory complement, unweighted. The honest
                   unseen-suite baseline established by the LOSO report.
  loso_reweighted  the same 12,000, reweighted so their rollout length
                   histogram matches the held-out suite's.
  oracle_same      the held-out suite's own 4,000 reference trajectories. What
                   calibration would give if same-distribution reference were
                   available. An upper bound, not a deployable option.

The question is how much of the loso_l1 -> oracle_same gap reweighting closes
using only unlabeled length information.
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

import length_reweighting as reweight  # noqa: E402
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
from intrinsic_guard_monitor import quantile_higher  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/length_reweighting"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
LEVELS = (
    "published",
    "loso_l1",
    "loso_reweighted",
    "loso_reweighted_fa",
    "oracle_same",
)

# Reweighting by length can only help a stream whose cut is length sensitive.
# LOSO measured that sensitivity: freeze and acceleration drift 18.7% and 20.8%
# std across folds, periodicity only 9.0%. The _fa level applies weights to the
# first two and leaves periodicity on the plain complement quantile.
LENGTH_SENSITIVE_STREAMS = frozenset({"freeze", "acceleration"})
STREAM_QUANTILES = dict(
    zip(loso_folds.PEAK_STREAMS, loso_folds.PUBLISHED_QUANTILES, strict=True)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--target-sample",
        type=int,
        default=0,
        help="if >0, estimate the target length histogram from this many "
        "randomly drawn held-out rollouts instead of all of them",
    )
    return parser.parse_args()


def constants_from_thresholds(
    thresholds: dict[str, float], scale: float
) -> loso_folds.FoldConstants:
    return loso_folds.FoldConstants(
        freeze_threshold=thresholds["freeze"],
        acceleration_threshold=thresholds["acceleration"],
        periodicity_threshold=thresholds["periodicity"],
        periodicity_scale=scale,
        freeze_quantile=STREAM_QUANTILES["freeze"],
        acceleration_quantile=STREAM_QUANTILES["acceleration"],
        periodicity_quantile=STREAM_QUANTILES["periodicity"],
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
    ref_length = np.concatenate(
        (main_layer["length"].astype(int), extra_layer["length"].astype(int))
    )
    external_suite = loso_folds.suite_of(external_layer)
    external_length = external_layer["length"].astype(int)
    external_valid = external_layer["valid"].astype(bool)
    external_mobility = external_layer["mobility"]
    external_acceleration = feature(external_feature, "route_acceleration")
    external_periodicity = feature(external_feature, "lag_periodicity")

    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        published_guard = np.asarray(sealed["external_guard"])

    rng = np.random.default_rng(args.seed)
    fold_state: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    threshold_rows: list[dict[str, Any]] = []

    for held_out in loso_folds.SUITES:
        complement = ref_suite != held_out
        same_suite = ref_suite == held_out
        evaluation = external_suite == held_out

        target_length = external_length[evaluation]
        if args.target_sample > 0:
            draw = rng.choice(
                len(target_length),
                size=min(args.target_sample, len(target_length)),
                replace=False,
            )
            target_length = target_length[draw]

        weights, weight_diagnostics = reweight.length_weights(
            ref_length[complement], target_length
        )
        weight_diagnostics["target_sample_size"] = int(len(target_length))
        diagnostics[held_out] = weight_diagnostics

        # One scale per fold, shared by every level so the comparison isolates
        # the reference population rather than the score definition.
        scale = loso_folds.periodicity_scale_of(ref_periodicity[complement])

        populations = {
            "loso_l1": (complement, None, frozenset()),
            "loso_reweighted": (
                complement,
                weights,
                frozenset(loso_folds.PEAK_STREAMS),
            ),
            "loso_reweighted_fa": (complement, weights, LENGTH_SENSITIVE_STREAMS),
            "oracle_same": (same_suite, None, frozenset()),
        }
        peaks_by_mask: dict[int, dict[str, np.ndarray]] = {}
        constants: dict[str, loso_folds.FoldConstants] = {}
        for level, (mask, weight, weighted_streams) in populations.items():
            key = int(mask.sum()) if mask is complement else -int(mask.sum())
            if key not in peaks_by_mask:
                peaks_by_mask[key] = loso_folds.peaks_from_scores(
                    loso_folds.cohort_scores(
                        ref_mobility[mask],
                        ref_acceleration[mask],
                        ref_periodicity[mask],
                        scale,
                    )
                )
            peaks = peaks_by_mask[key]
            thresholds = {
                stream: (
                    reweight.weighted_quantile_higher(
                        peaks[stream], weight, STREAM_QUANTILES[stream]
                    )
                    if weight is not None and stream in weighted_streams
                    else quantile_higher(peaks[stream], STREAM_QUANTILES[stream])
                )
                for stream in loso_folds.PEAK_STREAMS
            }
            constants[level] = constants_from_thresholds(thresholds, scale)
            for stream, value in thresholds.items():
                threshold_rows.append(
                    {
                        "held_out_suite": held_out,
                        "level": level,
                        "stream": stream,
                        "threshold": value,
                    }
                )

        evaluation_scores = loso_folds.cohort_scores(
            external_mobility[evaluation],
            external_acceleration[evaluation],
            external_periodicity[evaluation],
            scale,
        )
        guards = {"published": published_guard[evaluation]}
        for level, fold_constants in constants.items():
            guards[level] = loso_folds.guard_alarms(
                evaluation_scores, external_valid[evaluation], fold_constants
            )["guard"]
        fold_state.append(
            {"held_out": held_out, "evaluation": evaluation, "guards": guards}
        )
        print(
            f"fold {held_out}: ESS {weight_diagnostics['effective_sample_size']:.0f}, "
            f"uncovered target mass {weight_diagnostics['uncovered_target_mass']:.4f}",
            flush=True,
        )

    thresholds_frame = pd.DataFrame(threshold_rows)
    thresholds_frame.to_csv(output / "fold_thresholds.csv", index=False)
    np.savez_compressed(
        output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.intrinsic_guard_v7.reweight.alarms.v1"),
        **{
            f"{state['held_out']}__{level}": guard
            for state in fold_state
            for level, guard in state["guards"].items()
        },
    )
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.reweight.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "levels": list(LEVELS),
        "held_out_outcomes_used": False,
        "held_out_lengths_used": True,
        "held_out_lengths_note": (
            "loso_reweighted matches the held-out suite's rollout length "
            "histogram, so it is a weaker claim than plain LOSO. Lengths carry "
            "no outcome but are held-out cohort information."
        ),
        "target_sample_size": args.target_sample or "all",
        "weight_diagnostics": diagnostics,
        "artifacts": {
            "fold_thresholds_sha256": sha256(output / "fold_thresholds.csv"),
            "sealed_first_alarms_sha256": sha256(output / "sealed_first_alarms.npz"),
            "reweighting_module_sha256": sha256(HERE / "length_reweighting.py"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "reweight_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("sealed all folds; opening held-out outcomes now", flush=True)

    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    metric_rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for state in fold_state:
        labels = external_labels.loc[state["evaluation"]].reset_index(drop=True)
        for level in LEVELS:
            row = metric_row(
                "external_8b",
                f"{state['held_out']}__{level}",
                state["guards"][level],
                labels,
                args.bootstrap,
                metric_rng,
                group=state["held_out"],
            )
            row.update({"held_out_suite": state["held_out"], "level": level})
            rows.append(row)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(output / "reweight_fold_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        block = fold_metrics[fold_metrics["level"] == level]
        tp, fp = int(block["tp"].sum()), int(block["fp"].sum())
        risk, timely = int(block["risk_n"].sum()), int(block["timely_n"].sum())
        summary_rows.append(
            {
                "level": level,
                "macro_risk_recall": float(block["risk_recall"].mean()),
                "macro_precision": float(block["precision"].mean()),
                "macro_timely_fpr": float(block["timely_fpr"].mean()),
                "macro_early4": float(block["early4_risk_recall"].mean()),
                "micro_tp": tp,
                "micro_fp": fp,
                "micro_risk_recall": tp / risk,
                "micro_precision": tp / (tp + fp),
                "micro_timely_fpr": fp / timely,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "reweight_summary.csv", index=False)
    (output / "reweight_summary.json").write_text(
        json.dumps(plain(summary.to_dict(orient="records")), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
