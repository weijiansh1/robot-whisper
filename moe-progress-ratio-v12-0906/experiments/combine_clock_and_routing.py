#!/usr/bin/env python3
"""Recall, precision and lead at once, by treating the clock as a channel.

The clock-ceiling measurement made these look like competitors: a rule that
alarms on anything still running at chunk T gets recall 1.000 at timely FPR
0.42%, beating every routing monitor on both axes, while the routing monitors
win only on when they fire. Read that way, one has to choose.

They are complementary instead. The clock never misses but fires late; routing
fires early on part of the corpus but misses the rest. Their union should inherit
both properties, and the cost is whatever false alarms routing adds that the
clock did not already have.

There is reason to think that cost is small. A routing false alarm is a timely
rollout that looked stuck, and the survival-baseline work found those are the
slow successes, with median length 32-44 against a timely median of 24. The
clock's false alarms are timely rollouts that ran past T. Those are largely the
same episodes, so the union's false-alarm set may be close to the clock's alone.

This sweeps the routing threshold against a fixed clock and reports the whole
frontier: recall, precision, timely FPR, median alarm chunk, and the share of
risk episodes the union catches strictly earlier than the clock alone would.

Run:  python experiments/combine_clock_and_routing.py
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
sys.path.insert(0, str(HERE))

from fuse_v7_branches import (  # noqa: E402
    CLOCK_T,
    STEPS_PER_CHUNK,
    aligned_labels,
    calibrated_cut,
    candidate_alarms,
    earliest_masks,
    guard_alarms,
    load_cohort,
    PROFILE,
)
from intrinsic_guard_monitor import first_from_score, first_or  # noqa: E402

DEFAULT_OUTPUT = BUNDLE / "results/combined"
QUANTILES = (0.80, 0.90, 0.95, 0.9725, 0.98, 0.99, 0.995)
DRAWS = 2000
SEED = 20260906


def clock_alarms(cohort: dict) -> np.ndarray:
    """Alarm at T if the rollout is still running there, else never."""
    threshold = pd.Series(cohort["suite"]).map(CLOCK_T[cohort["name"]]).to_numpy()
    return np.where(cohort["length"] > threshold, threshold, -1).astype(np.int16)


def clustered_share(
    values: np.ndarray, tasks: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    names = np.unique(tasks)
    index = {name: np.flatnonzero(tasks == name) for name in names}
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = values[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def measure(cohort: dict, alarms: np.ndarray, labels: pd.DataFrame, reference: np.ndarray) -> dict:
    risk = labels["original_failure"].to_numpy(bool)
    fired = alarms >= 0
    true_positive = fired & risk
    false_positive = fired & ~risk
    earlier = true_positive & ((reference < 0) | (alarms < reference))
    lead = np.where(earlier & (reference >= 0), reference - alarms, 0)
    rng = np.random.default_rng(SEED)
    low, high = clustered_share(earlier[risk].astype(float), cohort["task"][risk], rng)
    return {
        "tp": int(true_positive.sum()),
        "fp": int(false_positive.sum()),
        "recall": float(true_positive.sum() / risk.sum()),
        "precision": float(true_positive.sum() / max(fired.sum(), 1)),
        "timely_fpr": float(false_positive.sum() / (~risk).sum()),
        "median_alarm_chunk": float(np.median(alarms[true_positive])),
        "share_earlier_than_clock": float(earlier[risk].mean()),
        "share_earlier_ci95": [low, high],
        "median_lead_env_steps": float(
            STEPS_PER_CHUNK * np.median(lead[earlier & (reference >= 0)])
            if (earlier & (reference >= 0)).any()
            else np.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profile = dict(np.load(PROFILE, allow_pickle=False))
    base = {
        "freeze": float(profile["freeze_threshold"]),
        "acceleration": float(profile["acceleration_threshold"]),
        "periodicity": float(profile["periodicity_threshold"]),
    }
    cohorts = {
        name: load_cohort(name)
        for name in ("development_main", "development_extra", "external_8b")
    }
    reference_pool = [cohorts["development_main"], cohorts["development_extra"]]
    labels = {name: aligned_labels(cohorts[name]) for name in ("development_main", "external_8b")}

    rows = []
    for name in ("development_main", "external_8b"):
        cohort = cohorts[name]
        clock = clock_alarms(cohort)
        guard = guard_alarms(cohort, base)

        rows.append(
            {"channel": "clock_only", "quantile": None, "cohort": name,
             **measure(cohort, clock, labels[name], clock)}
        )
        rows.append(
            {"channel": "v7_guard_only", "quantile": None, "cohort": name,
             **measure(cohort, guard, labels[name], clock)}
        )
        rows.append(
            {"channel": "clock_OR_v7guard", "quantile": None, "cohort": name,
             **measure(cohort, first_or(clock, guard), labels[name], clock)}
        )

        # Overlap of the two false-alarm sets, which is what decides the cost.
        risk = labels[name]["original_failure"].to_numpy(bool)
        clock_fp = (clock >= 0) & ~risk
        guard_fp = (guard >= 0) & ~risk
        rows[-1]["clock_fp"] = int(clock_fp.sum())
        rows[-1]["guard_fp"] = int(guard_fp.sum())
        rows[-1]["shared_fp"] = int((clock_fp & guard_fp).sum())
        rows[-1]["guard_only_fp"] = int((guard_fp & ~clock_fp).sum())

        for quantile in QUANTILES:
            cut = calibrated_cut(reference_pool, "freeze_only", float(quantile), base)
            freeze = first_from_score(
                cohort["streams"]["freeze"], cut, earliest_masks(cohort)["freeze"]
            )
            rows.append(
                {"channel": "clock_OR_freeze", "quantile": float(quantile), "cohort": name,
                 **measure(cohort, first_or(clock, freeze), labels[name], clock)}
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "combined.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2))

    pd.set_option("display.width", 240)
    columns = [
        "cohort", "channel", "quantile", "tp", "fp", "recall", "precision",
        "timely_fpr", "median_alarm_chunk", "share_earlier_than_clock",
        "median_lead_env_steps",
    ]
    print(table[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n误报重叠（决定并集的代价）：")
    print(
        table[table["channel"] == "clock_OR_v7guard"][
            ["cohort", "clock_fp", "guard_fp", "shared_fp", "guard_only_fp"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
