#!/usr/bin/env python3
"""What does a monitor that reads nothing but the clock already achieve?

The survival baseline has been used throughout this repository as a yardstick,
with an explicit note that it is not a deployable detector. That left a gap: it
was never scored as one. Filling it changes what the other numbers mean.

`risk` is defined as failing to finish before the horizon cap, so every risk
episode has length exactly equal to the cap. The rule "alarm if this rollout is
still running at chunk T" therefore catches every risk episode for any
T <= cap, and its only cost is the timely rollouts that happen to run past T:

    recall(T)      = 1.0                       for every T <= cap
    timely FPR(T)  = P(timely length > T)

If that holds, recall and precision cannot separate any two monitors here, and
the single quantity that distinguishes them is **how early they fire at matched
false-alarm rate**. This script checks the identity and then does the comparison
that survives it: for each risk episode, does the routing monitor fire strictly
before the clock rule would have, at the same timely FPR.

Per-suite thresholds use suite identity, which is task-side information; the
global threshold does not. Both are reported.

Run:  python experiments/verify_clock_ceiling.py
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
sys.path.insert(0, str(BUNDLE / "experiments"))

V7 = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
LEDGER = BUNDLE / "results/ledger/episodes.csv"
DEFAULT_OUTPUT = BUNDLE / "results/clock_ceiling"

TARGET_FPR = (0.005, 0.01, 0.02)
CHANNELS = ("v4", "v7_freeze", "v7_turbulence", "v7_guard", "progress_ratio")
DRAWS = 2000
SEED = 20260906


def clock_threshold(lengths: np.ndarray, target: float) -> int:
    """Smallest T whose timely false-alarm rate is at or below the target."""
    for threshold in range(int(lengths.max()) + 2):
        if float((lengths > threshold).mean()) <= target:
            return threshold
    return int(lengths.max()) + 1


def clustered_interval(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    names = np.unique(clusters)
    index = {name: np.flatnonzero(clusters == name) for name in names}
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = values[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv(LEDGER)
    records, comparisons = [], []
    for cohort, block in ledger.groupby("cohort"):
        risk = block[block["risk"].astype(bool)]
        timely = block[block["timely"].astype(bool)]

        # The identity: are all risk lengths pinned at the suite cap?
        caps = block.groupby("suite")["length"].max()
        pinned = float(
            np.mean(
                [
                    row["length"] == caps[row["suite"]]
                    for _, row in risk.iterrows()
                ]
            )
        )

        for target in TARGET_FPR:
            for mode in ("global", "per_suite"):
                if mode == "global":
                    threshold = {
                        str(suite): clock_threshold(timely["length"].to_numpy(), target)
                        for suite in block["suite"].unique()
                    }
                else:
                    threshold = {
                        str(suite): clock_threshold(group["length"].to_numpy(), target)
                        for suite, group in timely.groupby("suite")
                    }
                fires = risk["suite"].map(threshold).to_numpy()
                false_alarms = int(
                    (timely["length"].to_numpy() > timely["suite"].map(threshold).to_numpy()).sum()
                )
                records.append(
                    {
                        "cohort": cohort,
                        "target_fpr": target,
                        "mode": mode,
                        "thresholds": {k: int(v) for k, v in threshold.items()},
                        "risk_caught": int((fires <= risk["length"].to_numpy() - 1).sum()),
                        "risk_total": int(len(risk)),
                        "recall": float((fires <= risk["length"].to_numpy() - 1).mean()),
                        "false_alarms": false_alarms,
                        "timely_fpr": float(false_alarms / len(timely)),
                        "median_clock_alarm_chunk": float(np.median(fires)),
                    }
                )

                if mode != "per_suite" or target != 0.005:
                    continue
                rng = np.random.default_rng(SEED)
                for channel in CHANNELS:
                    column = f"{channel}_chunk"
                    if column not in risk.columns:
                        continue
                    alarm = risk[column].to_numpy()
                    detected = alarm >= 0
                    earlier = detected & (alarm < fires)
                    lead = np.where(earlier, fires - alarm, 0)
                    low, high = clustered_interval(
                        earlier.astype(float), risk["task"].to_numpy(), rng
                    )
                    comparisons.append(
                        {
                            "cohort": cohort,
                            "channel": channel,
                            "risk_total": int(len(risk)),
                            "detected": int(detected.sum()),
                            "earlier_than_clock": int(earlier.sum()),
                            "share_earlier": float(earlier.mean()),
                            "share_earlier_ci95": [low, high],
                            "median_lead_chunks_when_earlier": float(
                                np.median(lead[earlier]) if earlier.any() else np.nan
                            ),
                            "median_lead_env_steps_when_earlier": float(
                                10 * np.median(lead[earlier]) if earlier.any() else np.nan
                            ),
                        }
                    )
        records[-1]["risk_length_pinned_at_cap"] = pinned

    ceiling = pd.DataFrame(records)
    ceiling.to_csv(args.output / "clock_ceiling.csv", index=False)
    table = pd.DataFrame(comparisons)
    table.to_csv(args.output / "earlier_than_clock.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps({"ceiling": records, "comparisons": comparisons}, indent=2, default=str)
    )

    pd.set_option("display.width", 220)
    print("时钟规则：跑到第 T 个 chunk 仍未结束即报警")
    print(
        ceiling[
            [
                "cohort",
                "target_fpr",
                "mode",
                "recall",
                "false_alarms",
                "timely_fpr",
                "median_clock_alarm_chunk",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n各通道相对时钟规则（per_suite, FPR 0.5%）是否报得更早")
    print(
        table[
            [
                "cohort",
                "channel",
                "detected",
                "earlier_than_clock",
                "share_earlier",
                "median_lead_env_steps_when_earlier",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
