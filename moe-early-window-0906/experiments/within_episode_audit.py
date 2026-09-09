#!/usr/bin/env python3
"""Within-episode information for every quantity, before any of them is used.

A channel that is bit-constant inside an episode carries no within-trajectory
information at all.  Such a channel can still pass a frozen selection rule,
because a causal trailing mean in float32 accumulates rounding along the chunk
axis and a strict `>` test against a per-task quantile flips on those last bits;
`moe-unused-channels-0906/experiments/task_matched_lift.py` documents one head
that reached external lift 4.40 that way.  So this audit runs first and any
quantity it marks constant is excluded from the candidate set and kept only as a
negative control.

Reported per (cohort, quantity, layer), over the valid chunks of each episode:

  zero_range_fraction   share of episodes with max == min exactly
  median_within_sd      median over episodes of the within-episode SD
  between_sd            SD across episodes of the per-episode mean
  within_between_ratio  median_within_sd / between_sd
  median_distinct       median number of distinct float32 values per episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


COHORTS = ("development_main", "external_8b", "legacy_main16x32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def audit(frame: dict) -> list[dict]:
    values = frame["values"]  # (n, chunk, layer, quantity), NaN outside the episode
    valid = frame["valid"]
    n_ep = frame["n_ep"]
    rows: list[dict] = []
    with np.errstate(invalid="ignore"):
        vmax = np.nanmax(values, axis=1)
        vmin = np.nanmin(values, axis=1)
        vsd = np.nanstd(values, axis=1)
        vmean = np.nanmean(values, axis=1)
    rng = vmax - vmin
    counts = valid.sum(axis=1)
    for qi, quantity in enumerate(frame["quantities"]):
        for li, layer in enumerate(C.LAYER_NAMES):
            r = rng[:, li, qi]
            sd = vsd[:, li, qi]
            mu = vmean[:, li, qi]
            finite = np.isfinite(r)
            distinct = np.array(
                [
                    len(np.unique(values[i, : counts[i], li, qi]))
                    for i in range(0, n_ep, max(1, n_ep // 400))
                ]
            )
            rows.append(
                {
                    "cohort": frame["cohort"],
                    "quantity": quantity,
                    "layer": layer,
                    "episodes": int(finite.sum()),
                    "zero_range_fraction": float((r[finite] == 0).mean()),
                    "median_range": float(np.median(r[finite])),
                    "median_within_sd": float(np.median(sd[finite])),
                    "between_sd": float(np.std(mu[np.isfinite(mu)])),
                    "within_between_ratio": float(
                        np.median(sd[finite]) / max(float(np.std(mu[np.isfinite(mu)])), 1e-30)
                    ),
                    "median_distinct_sampled": float(np.median(distinct)),
                    "is_control": quantity in C.CONTROL_QUANTITIES,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cohort in COHORTS:
        frame = C.build_cohort(cohort)
        rows += audit(frame)
        del frame
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "within_episode_information.csv", index=False)

    # A quantity is excluded if it is bit-constant inside every episode in any
    # cohort where it exists.
    grouped = table.groupby("quantity")["zero_range_fraction"].max()
    excluded = sorted(grouped[grouped >= 1.0].index.tolist())
    marginal = sorted(grouped[(grouped >= 0.01) & (grouped < 1.0)].index.tolist())
    C.write_json(
        args.output / "within_episode_verdict.json",
        {
            "rule": "exclude any quantity whose within-episode range is exactly 0 for every episode",
            "excluded_constant": excluded,
            "partially_constant": {
                q: float(grouped[q]) for q in marginal
            },
            "retained": sorted(
                q for q in grouped.index if q not in excluded and q not in C.CONTROL_QUANTITIES
            ),
            "controls": sorted(q for q in grouped.index if q in C.CONTROL_QUANTITIES),
        },
    )
    print(table.to_string(max_rows=40))
    print("excluded:", excluded)


if __name__ == "__main__":
    main()
