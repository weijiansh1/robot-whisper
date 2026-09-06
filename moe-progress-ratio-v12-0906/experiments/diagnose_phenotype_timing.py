#!/usr/bin/env python3
"""Why the single-threshold R rule misses the low-prior region.

`evaluate_phenotype_2x2.py` reports the four-cell occupancy pooled over all
chunks, and finds the circling cell (high mobility, low R) *depleted* in risk
episodes -- 0.30x to 0.61x.  That pooled comparison is confounded: a risk
episode is by definition one that ran to the horizon cap, so risk episodes
contribute disproportionately many late chunks, and late chunks have a
different phenotype mix.  It is the same survival-time confound that
`moe-v7-0905/docs/SURVIVAL_BASELINE_REPORT_ZH.md` was written to handle,
reappearing one level down.

This script repeats the comparison chunk-matched within suite, then splits the
matched cells by the survival prior at that chunk.

RETRACTION (2026-09-06).  The first version of this script aggregated by a
weighted mean of per-cell ratios and reported that the circling cell lifts
2.84x-5.60x in the low-prior band while freeze lifts only 1.47x-1.81x, i.e.
that circling is the early signal and freeze the late one.  That conclusion is
an artefact of the aggregator.  A mean of per-cell ratios is dominated by cells
whose denominator is nearly empty -- one libero_goal cell carries 9 circling
timely episodes out of 1546 and contributes a 40x per-cell ratio.

Pooling the same cells under the same gates by observed over expected events
reverses it: in the low-prior band circling is *depleted* at 0.58x-0.84x while
freeze is enriched at 1.28x-1.29x.  Both estimators are now emitted side by
side and the pooled one is authoritative.  The `*_mean_of_ratios` columns are
retained only so the retracted numbers stay reproducible.

Run:  python experiments/diagnose_phenotype_timing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

CACHE = BUNDLE / "results/progress_cache/development_main.npz"
LABELS = (
    WORKSPACE
    / "double-selete/trainfree/results/timeout_extension_plus10"
    / "development_main_clean_labels.csv"
)
DEFAULT_OUTPUT = BUNDLE / "results/phenotype"

POINTS = ((2, "all"), (4, "all"), (6, "all"), (10, "all"), (4, "front"), (6, "back"))
EPS_LENGTH = 1e-3
PRIOR_CUT = 0.25
MIN_CELL = 5


def load_cohort() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    cache = dict(np.load(CACHE, allow_pickle=False))
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
        }
    )
    labels = pd.read_csv(LABELS)[["task", "episode", "original_failure"]]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError("development labels do not align with the progress cache")
    risk = merged["original_failure"].to_numpy(bool)
    suite = pd.Series(task).str.split("/", n=1).str[0].to_numpy()
    return cache, risk, suite, cache["length"].astype(int)


def survival_prior(
    risk: np.ndarray, suite: np.ndarray, length: np.ndarray
) -> dict[str, dict[int, float]]:
    return {
        str(name): {
            chunk: float(risk[(suite == name) & (length > chunk)].mean())
            for chunk in range(int(length[suite == name].max()))
        }
        for name in np.unique(suite)
    }


def cell_masks(cache: dict, window: int, group: str) -> tuple[np.ndarray, ...]:
    adjacent = cache["lag_distance"][:, :, 0, :]
    length = pr.path_length(adjacent, window)
    displacement = cache["lag_distance"][:, :, pr.LAGS.index(window), :]
    ratio = pr.group_ratio(
        pr.progress_ratio(displacement, length, EPS_LENGTH), group
    )
    grouped_length = pr.group_ratio(length, group)
    usable = cache["valid"] & np.isfinite(ratio) & np.isfinite(grouped_length)
    high_mobility = grouped_length >= np.median(grouped_length[usable])
    low_ratio = ratio < np.median(ratio[usable])
    return usable, ~high_mobility & low_ratio, high_mobility & low_ratio


def matched_lift(
    usable: np.ndarray,
    cells: dict[str, np.ndarray],
    risk: np.ndarray,
    suite: np.ndarray,
    prior: dict[str, dict[int, float]],
) -> list[dict]:
    """Risk-vs-timely occupancy ratio within each (suite, chunk) cell."""
    records: list[dict] = []
    for name in np.unique(suite):
        for chunk in range(usable.shape[1]):
            if chunk not in prior[str(name)]:
                continue
            rows = (suite == name) & usable[:, chunk]
            risky, timely = rows & risk, rows & ~risk
            if risky.sum() < MIN_CELL or timely.sum() < MIN_CELL:
                continue
            record = {
                "suite": str(name),
                "chunk": chunk,
                "prior": prior[str(name)][chunk],
                "n_risk": int(risky.sum()),
            }
            for label, mask in cells.items():
                timely_share = mask[timely, chunk].mean()
                record[label] = (
                    float(mask[risky, chunk].mean() / timely_share)
                    if timely_share > 0
                    else float("nan")
                )
                # Pooled numerator and denominator, kept per cell so the
                # stratified ratio can be formed over any subset of cells.
                record[f"{label}_observed"] = float(mask[risky, chunk].sum())
                record[f"{label}_expected"] = float(timely_share * risky.sum())
            records.append(record)
    return records


def weighted(frame: pd.DataFrame, column: str) -> float:
    """Mean of per-cell ratios.  RETRACTED as an effect estimate -- see module docstring."""
    usable = frame[np.isfinite(frame[column])]
    if usable.empty:
        return float("nan")
    return float(
        (usable[column] * usable["n_risk"]).sum() / usable["n_risk"].sum()
    )


def pooled(frame: pd.DataFrame, column: str) -> float:
    """Stratified observed-over-expected across the same cells.  Authoritative.

    Unlike a mean of per-cell ratios this cannot be dominated by a cell whose
    denominator is nearly empty, because every cell contributes events rather
    than a ratio.
    """
    expected = frame[f"{column}_expected"].sum()
    if not expected > 0:
        return float("nan")
    return float(frame[f"{column}_observed"].sum() / expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cache, risk, suite, length = load_cohort()
    prior = survival_prior(risk, suite, length)

    rows: list[dict] = []
    summary: list[dict] = []
    for window, group in POINTS:
        usable, freeze, circling = cell_masks(cache, window, group)
        cells = {"freeze": freeze, "circling": circling}

        unmatched = {}
        for label, mask in cells.items():
            risk_share = mask[risk].sum() / usable[risk].sum()
            timely_share = mask[~risk].sum() / usable[~risk].sum()
            unmatched[label] = float(risk_share / timely_share)

        matched = pd.DataFrame(matched_lift(usable, cells, risk, suite, prior))
        matched["window"] = window
        matched["layer_group"] = group
        rows.append(matched)

        low = matched[matched["prior"] < PRIOR_CUT]
        high = matched[matched["prior"] >= PRIOR_CUT]
        summary.append(
            {
                "window": window,
                "layer_group": group,
                "unmatched_freeze_lift": unmatched["freeze"],
                "unmatched_circling_lift": unmatched["circling"],
                "low_prior_freeze_pooled": pooled(low, "freeze"),
                "low_prior_circling_pooled": pooled(low, "circling"),
                "high_prior_freeze_pooled": pooled(high, "freeze"),
                "high_prior_circling_pooled": pooled(high, "circling"),
                "matched_freeze_mean_of_ratios": weighted(matched, "freeze"),
                "matched_circling_mean_of_ratios": weighted(matched, "circling"),
                "low_prior_freeze_mean_of_ratios": weighted(low, "freeze"),
                "low_prior_circling_mean_of_ratios": weighted(low, "circling"),
                "high_prior_freeze_mean_of_ratios": weighted(high, "freeze"),
                "high_prior_circling_mean_of_ratios": weighted(high, "circling"),
                "low_prior_cells": int(len(low)),
                "high_prior_cells": int(len(high)),
            }
        )

    pd.concat(rows).to_csv(args.output / "phenotype_timing_cells.csv", index=False)
    frame = pd.DataFrame(summary)
    frame.to_csv(args.output / "phenotype_timing.csv", index=False)
    (args.output / "phenotype_timing.json").write_text(
        json.dumps(
            {
                "prior_cut": PRIOR_CUT,
                "min_cell": MIN_CELL,
                "eps_length": EPS_LENGTH,
                "rows": summary,
            },
            indent=2,
        )
    )

    pd.set_option("display.width", 200)
    print(
        frame[
            [
                "window",
                "layer_group",
                "low_prior_freeze_pooled",
                "low_prior_circling_pooled",
                "high_prior_freeze_pooled",
                "high_prior_circling_pooled",
                "low_prior_circling_mean_of_ratios",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
