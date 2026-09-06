#!/usr/bin/env python3
"""Two anchors that must hold before any number in this bundle is believed.

1. The step-9 slice of the new per-step mobility must reproduce the frozen v4
   `mobility` cache, which is defined at the final flow step.  If it does not,
   the step axis was mis-indexed.
2. The copied detector protocol must reproduce the published external mobility
   heads exactly: L2 low q0.70 -> 272/57 under per_task thresholds, and L12 low
   q0.975 -> 195/17 under a single global threshold.

Run: python tests/test_anchors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
PROFILES = BUNDLE / "results/step_profiles"
ALARM = BUNDLE / "results/step_alarm"

CACHES = {
    "development_main": "main_reference.npz",
    "development_extra": "extra_reference.npz",
    "external_8b": "external_8b.npz",
}
PUBLISHED = {
    "per_task": ("L2", "low", 0.70, 272, 57),
    "global": ("L12", "low", 0.975, 195, 17),
}


def check_step9_matches_v4() -> None:
    for cohort, cache_name in CACHES.items():
        with np.load(V4 / cache_name, allow_pickle=False) as cache:
            reference = np.asarray(cache["mobility"])
            valid = cache["valid"].astype(bool)
        valid[:, 0] = False
        block = np.load(PROFILES / f"{cohort}_mobility.npy", mmap_mode="r")
        observed = np.asarray(block[:, :, :, 9])
        gap = np.abs(observed[valid] - reference[valid])
        assert np.isfinite(gap).all(), f"{cohort}: non-finite step-9 mobility"
        assert gap.max() < 1e-5, f"{cohort}: step-9 gap {gap.max()}"
        assert not np.isfinite(observed[~valid]).any(), f"{cohort}: mask mismatch"
        print(f"  {cohort}: step-9 mobility matches v4 to {gap.max():.2e}")


def check_published_anchor() -> None:
    external = pd.read_csv(ALARM / "external_detectors.csv")
    for mode, (representation, direction, quantile, tp, fp) in PUBLISHED.items():
        row = external[
            (external["quantity"] == "mobility_s9") & (external["mode"] == mode)
        ].iloc[0]
        assert row["representation"] == representation, row["representation"]
        assert row["direction"] == direction, row["direction"]
        assert abs(float(row["quantile"]) - quantile) < 1e-9, row["quantile"]
        assert int(row["tp"]) == tp and int(row["fp"]) == fp, (row["tp"], row["fp"])
        print(f"  {mode}: reproduced {representation} {direction} q{quantile} -> {tp}/{fp}")
    summary = json.loads((ALARM / "summary.json").read_text())
    assert all(entry["matches"] for entry in summary["published_anchor"].values())


def main() -> int:
    print("step-9 mobility vs the frozen v4 cache")
    check_step9_matches_v4()
    print("published external mobility heads")
    check_published_anchor()
    print("all anchors hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
