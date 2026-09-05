#!/usr/bin/env python3
"""Audit the fixed global-quantile grid using development outcomes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
METHOD = BUNDLE / "method"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(METHOD))

from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTRA_FEATURE,
    EXTRA_LAYER,
    LABEL_ROOT,
    MAIN_FEATURE,
    MAIN_LAYER,
    PERIODICITY_SCALE_QUANTILE,
    aligned_labels,
    assert_aligned,
    combine_reference,
    feature,
    load_npz,
)
from intrinsic_guard_monitor import (  # noqa: E402
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)


DEFAULT_OUTPUT = BUNDLE / "results/intrinsic_guard_v7"
FREEZE_QUANTILES = (0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995)
ACCELERATION_QUANTILES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975)
PERIODICITY_QUANTILES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(first: np.ndarray, labels: pd.DataFrame) -> dict[str, float | int]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(bool)
    timely = ~risk
    lead = labels["length"].to_numpy(int) - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    recall = tp / int(risk.sum())
    precision = tp / max(tp + fp, 1)
    fpr = fp / int(timely.sum())
    f1 = 2.0 * recall * precision / max(recall + precision, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "risk_recall": recall,
        "precision": precision,
        "timely_fpr": fpr,
        "early4_risk_recall": float((alarm & risk & (lead >= 4)).sum() / risk.sum()),
        "f1": f1,
    }


def choose(candidates: pd.DataFrame, conservative: bool) -> pd.Series:
    if conservative:
        eligible = candidates[
            (candidates["precision"] >= 0.90)
            & (candidates["timely_fpr"] <= 0.003)
            & (candidates["risk_recall"] >= 0.65)
        ]
        order = ["risk_recall", "early4_risk_recall", "f1"]
    else:
        eligible = candidates[
            (candidates["precision"] >= 0.85)
            & (candidates["timely_fpr"] <= 0.005)
            & (candidates["risk_recall"] >= 0.70)
        ]
        order = ["f1", "early4_risk_recall", "risk_recall"]
    if eligible.empty:
        raise RuntimeError("no operating point satisfies the predeclared constraints")
    return eligible.sort_values(order, ascending=False, kind="stable").iloc[0]


def row_dict(row: pd.Series) -> dict[str, float | int]:
    output = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in row.to_dict().items()
    }
    output["tp"] = int(output["tp"])
    output["fp"] = int(output["fp"])
    return output


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    extra_feature = load_npz(EXTRA_FEATURE)
    assert_aligned(main_layer, main_feature, "main")
    assert_aligned(extra_layer, extra_feature, "extra")
    reference = combine_reference(main_layer, extra_layer, main_feature, extra_feature)

    ref_mobility, ref_acceleration, ref_periodicity = reference
    finite_periodicity = np.abs(ref_periodicity[np.isfinite(ref_periodicity)])
    periodicity_scale = float(
        np.quantile(finite_periodicity, PERIODICITY_SCALE_QUANTILE, method="linear")
    )
    reference_scores = intrinsic_score_arrays(
        ref_mobility, ref_acceleration, ref_periodicity, periodicity_scale
    )
    development_scores = intrinsic_score_arrays(
        main_layer["mobility"],
        feature(main_feature, "route_acceleration"),
        feature(main_feature, "lag_periodicity"),
        periodicity_scale,
    )
    valid = main_layer["valid"].astype(bool)

    first_freeze = {
        quantile: first_from_score(
            development_scores["freeze"],
            quantile_higher(row_max(reference_scores["freeze"]), quantile),
            valid,
        )
        for quantile in FREEZE_QUANTILES
    }
    first_acceleration = {
        quantile: first_from_score(
            development_scores["acceleration_persistent"],
            quantile_higher(row_max(reference_scores["acceleration"]), quantile),
            valid,
        )
        for quantile in ACCELERATION_QUANTILES
    }
    first_periodicity = {
        quantile: first_from_score(
            development_scores["periodicity_persistent"],
            quantile_higher(row_max(reference_scores["periodicity"]), quantile),
            valid,
        )
        for quantile in PERIODICITY_QUANTILES
    }

    # This is the only outcome access in this script; no external cache is opened.
    labels = aligned_labels(
        main_layer,
        LABEL_ROOT / "development_main_clean_labels.csv",
        "development_main",
    )
    rows: list[dict[str, float | int]] = []
    for freeze_q, freeze_alarm in first_freeze.items():
        for acceleration_q, acceleration_alarm in first_acceleration.items():
            for periodicity_q, periodicity_alarm in first_periodicity.items():
                turbulence = first_and(acceleration_alarm, periodicity_alarm)
                guard = first_or(freeze_alarm, turbulence)
                rows.append(
                    {
                        "freeze_quantile": freeze_q,
                        "acceleration_quantile": acceleration_q,
                        "periodicity_quantile": periodicity_q,
                        **metrics(guard, labels),
                    }
                )
    candidates = pd.DataFrame(rows).sort_values(
        ["freeze_quantile", "acceleration_quantile", "periodicity_quantile"],
        kind="stable",
    )
    candidates_path = args.output / "development_threshold_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    primary = choose(candidates, conservative=False)
    conservative = choose(candidates, conservative=True)
    result = {
        "schema": "himoe.intrinsic_guard_v7.selection.v1",
        "method": "fixed Boolean mechanism and global unlabeled quantile grid",
        "model_training": False,
        "learned_weights": False,
        "task_conditioning": False,
        "external_data_loaded": False,
        "development_outcomes_used": True,
        "candidate_count": len(candidates),
        "primary_constraints": {
            "precision_at_least": 0.85,
            "timely_fpr_at_most": 0.005,
            "risk_recall_at_least": 0.70,
            "ranking": ["f1", "early4_risk_recall", "risk_recall"],
        },
        "primary": row_dict(primary),
        "conservative_constraints": {
            "precision_at_least": 0.90,
            "timely_fpr_at_most": 0.003,
            "risk_recall_at_least": 0.65,
            "ranking": ["risk_recall", "early4_risk_recall", "f1"],
        },
        "conservative": row_dict(conservative),
        "artifacts": {
            "candidate_csv_sha256": sha256(candidates_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    selection_path = args.output / "development_threshold_selection.json"
    selection_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
