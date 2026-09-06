#!/usr/bin/env python3
"""How much of the guard's precision is detection, and how much is base rate?

Risk is defined as failing to finish before the horizon cap, so a rollout that
is still running is already more likely to be a failure -- increasingly so as
the cap approaches. That gives a trivial competing detector which reads no MoE
state at all:

    survival baseline: at chunk q, P(risk | still running at q)

For every alarm the guard raised at chunk q, this script looks up the matched
survival prior at that same chunk and compares it against whether the alarm was
actually right. The ratio is what MoE routing contributed beyond knowing how
long the rollout has been going.

The prior is estimated on the same cohort being scored, which flatters the
baseline slightly. The effect sizes here are far too large for that to change
the conclusion, but a deployed survival detector would need its own reference.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    EXTERNAL_LAYER,
    LABEL_ROOT,
    aligned_labels,
    load_npz,
    plain,
    sha256,
)


DEFAULT_OUTPUT = BUNDLE / "results/survival_baseline"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
LOSO = BUNDLE / "results/loso_validation"
PRIOR_BANDS = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def survival_prior(labels: pd.DataFrame) -> dict[str, dict[int, float]]:
    """P(risk | still running at chunk q), estimated within each suite."""
    priors: dict[str, dict[int, float]] = {}
    for suite, block in labels.groupby("suite"):
        risk = block["original_failure"].to_numpy(bool)
        length = block["length"].to_numpy(int)
        priors[str(suite)] = {
            chunk: float(risk[length > chunk].mean())
            for chunk in range(int(length.max()))
        }
    return priors


def attach_prior(
    labels: pd.DataFrame, first: np.ndarray, priors: dict[str, dict[int, float]]
) -> pd.DataFrame:
    fired = labels.loc[first >= 0].copy()
    fired["chunk"] = first[first >= 0]
    fired["prior"] = [
        priors[suite][chunk]
        for suite, chunk in zip(fired["suite"], fired["chunk"], strict=True)
    ]
    fired["correct"] = fired["original_failure"].astype(bool)
    return fired


def summarise(fired: pd.DataFrame, detector: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(group: str, block: pd.DataFrame) -> None:
        if block.empty:
            return
        precision = float(block["correct"].mean())
        prior = float(block["prior"].mean())
        rows.append(
            {
                "detector": detector,
                "group": group,
                "alarms": int(len(block)),
                "precision": precision,
                "matched_prior": prior,
                "net_gain": precision - prior,
                "lift": precision / prior if prior > 0 else float("nan"),
            }
        )

    record("all", fired)
    for suite, block in fired.groupby("suite"):
        record(str(suite), block)
    for low, high in PRIOR_BANDS:
        band = fired[(fired["prior"] >= low) & (fired["prior"] < high)]
        record(f"prior[{low:.2f},{high:.2f})", band)
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    layer = load_npz(EXTERNAL_LAYER)
    labels = aligned_labels(
        layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    labels["suite"] = loso_folds.suite_of(layer)
    priors = survival_prior(labels)

    detectors: dict[str, np.ndarray] = {}
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        detectors["published"] = np.asarray(sealed["external_guard"], dtype=int)
    loso_path = LOSO / "fold_first_alarms.npz"
    if loso_path.exists():
        with np.load(loso_path, allow_pickle=False) as fold:
            pooled = np.full(len(labels), -1, dtype=int)
            covered = np.zeros(len(labels), bool)
            for suite in loso_folds.SUITES:
                key = f"{suite}__loso_l2__guard"
                if key in fold.files:
                    take = labels["suite"].to_numpy(str) == suite
                    pooled[take] = np.asarray(fold[key], dtype=int)
                    covered[take] = True
            if covered.all():
                detectors["loso_l2"] = pooled

    rows: list[dict[str, Any]] = []
    per_alarm: list[pd.DataFrame] = []
    for detector, first in detectors.items():
        fired = attach_prior(labels, first, priors)
        fired["detector"] = detector
        per_alarm.append(fired)
        rows.extend(summarise(fired, detector))

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "survival_baseline.csv", index=False)
    pd.concat(per_alarm, ignore_index=True)[
        ["detector", "suite", "task", "episode", "chunk", "prior", "correct"]
    ].to_csv(args.output / "alarm_priors.csv", index=False)

    summary = {
        "schema": "himoe.intrinsic_guard_v7.survival_baseline.v1",
        "prior_definition": "P(risk | still running at chunk q), per suite",
        "prior_estimated_in_sample": True,
        "detectors": list(detectors),
        "rows": plain(table.to_dict(orient="records")),
        "artifacts": {
            "survival_baseline_sha256": sha256(args.output / "survival_baseline.csv"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (args.output / "survival_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for detector in detectors:
        print(f"=== {detector} ===")
        print(
            table[table["detector"] == detector][
                ["group", "alarms", "precision", "matched_prior", "net_gain", "lift"]
            ].to_string(index=False, float_format="%.3f"),
            flush=True,
        )
        print()


if __name__ == "__main__":
    main()
