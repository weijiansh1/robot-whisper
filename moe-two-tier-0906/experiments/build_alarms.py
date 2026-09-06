#!/usr/bin/env python3
"""Rebuild first-alarm arrays for the 24 frame-survey heads on BOTH splits.

The frame survey published external alarms only. Selecting a two-tier rule
honestly needs the same heads on development, so this script re-runs the survey's
own head definitions -- same representation, direction, quantile and threshold
mode as `external_detectors.csv` records -- against development_main, and
re-derives the external arrays from scratch as a reproduction check.

If any rebuilt external array disagrees with the published one, this exits
non-zero. That check is the licence to trust the development twins.

The v7 intrinsic guard is carried through as a fifth head option; its sealed
npz already contains matched development and external arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import twotier_lib as lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lib.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohorts = {
        name: lib.load_cohort(name)
        for name in ("development_main", "development_extra", "external_8b")
    }
    published = np.load(lib.EXTERNAL_ALARMS, allow_pickle=True)
    detectors = pd.read_csv(lib.EXTERNAL_DETECTORS)
    detectors = detectors[detectors["feasible"].fillna(False)]

    development: dict[str, np.ndarray] = {}
    external: dict[str, np.ndarray] = {}
    report: list[dict[str, object]] = []

    for _, row in detectors.iterrows():
        key = f"{row['quantity']}|{row['mode']}"
        spec = dict(
            quantity=str(row["quantity"]),
            representation=str(row["representation"]),
            direction=str(row["direction"]),
            quantile=float(row["quantile"]),
            mode=str(row["mode"]),
        )
        dev_first = lib.head_alarms(cohorts, "development_main", **spec)
        ext_first = lib.head_alarms(cohorts, "external_8b", **spec)
        development[key] = dev_first.astype(np.int16)
        external[key] = ext_first.astype(np.int16)

        reference = np.asarray(published[key], dtype=np.int16)
        matches = bool(np.array_equal(reference, ext_first.astype(np.int16)))
        dev_risk = cohorts["development_main"]["risk"]
        ext_risk = cohorts["external_8b"]["risk"]
        report.append(
            {
                "head": key,
                "frame": lib.head_frame(key),
                **spec,
                "external_reproduces_published": matches,
                "external_tp": int(((ext_first >= 0) & ext_risk).sum()),
                "external_fp": int(((ext_first >= 0) & ~ext_risk).sum()),
                "development_tp": int(((dev_first >= 0) & dev_risk).sum()),
                "development_fp": int(((dev_first >= 0) & ~dev_risk).sum()),
            }
        )
        flag = "ok " if matches else "MISMATCH"
        print(
            f"  {flag} {key:<42s} dev {report[-1]['development_tp']:>4d}/"
            f"{report[-1]['development_fp']:<4d} ext {report[-1]['external_tp']:>4d}/"
            f"{report[-1]['external_fp']:<4d}",
            flush=True,
        )

    # v7 intrinsic guard, already sealed on both splits.
    guard = np.load(lib.V7_ALARMS, allow_pickle=True)
    for mode_key, source in (("v7_guard|global", "guard"),):
        development[mode_key] = np.asarray(guard[f"main_{source}"], dtype=np.int16)
        external[mode_key] = np.asarray(guard[f"external_{source}"], dtype=np.int16)
        dev_risk = cohorts["development_main"]["risk"]
        ext_risk = cohorts["external_8b"]["risk"]
        report.append(
            {
                "head": mode_key,
                "frame": lib.V7_FRAME,
                "quantity": "v7_guard",
                "representation": "v7_intrinsic_guard_bundle",
                "direction": "n/a",
                "quantile": float("nan"),
                "mode": "global",
                "external_reproduces_published": True,
                "external_tp": int(((external[mode_key] >= 0) & ext_risk).sum()),
                "external_fp": int(((external[mode_key] >= 0) & ~ext_risk).sum()),
                "development_tp": int(((development[mode_key] >= 0) & dev_risk).sum()),
                "development_fp": int(((development[mode_key] >= 0) & ~dev_risk).sum()),
            }
        )
        print(
            f"  ok  {mode_key:<42s} dev {report[-1]['development_tp']:>4d}/"
            f"{report[-1]['development_fp']:<4d} ext {report[-1]['external_tp']:>4d}/"
            f"{report[-1]['external_fp']:<4d}",
            flush=True,
        )

    table = pd.DataFrame(report)
    table.to_csv(args.output / "head_inventory.csv", index=False)

    bad = table[~table["external_reproduces_published"].astype(bool)]
    if len(bad):
        raise SystemExit(
            "external rebuild does not reproduce the published survey alarms:\n"
            + bad[["head"]].to_string(index=False)
        )

    for split, block, cohort in (
        ("development", development, cohorts["development_main"]),
        ("external", external, cohorts["external_8b"]),
    ):
        modes = lib.physical_modes(cohort)
        np.savez_compressed(
            args.output / f"{split}_first_alarms.npz",
            schema=np.asarray("himoe.two_tier.alarms.v1"),
            risk=cohort["risk"],
            suite=cohort["suite"].astype("U16"),
            length=cohort["length"],
            physical_mode=modes.astype("U48"),
            **block,
        )

    counts = {
        split: lib.mode_counts(
            lib.physical_modes(cohorts[key]), cohorts[key]["risk"]
        )
        for split, key in (
            ("development", "development_main"),
            ("external", "external_8b"),
        )
    }
    (args.output / "alarm_build.json").write_text(
        json.dumps(
            {
                "schema": "himoe.two_tier.alarm_build.v1",
                "heads": sorted(development),
                "external_reproduces_published_survey": True,
                "width": lib.WIDTH,
                "confirmations": lib.CONFIRMATIONS,
                "low_prior_threshold": lib.LOW_PRIOR,
                "mode_floor": lib.MODE_FLOOR,
                "physical_mode_counts_on_risk": counts,
                "modes_scored": {
                    split: lib.scored_modes(block) for split, block in counts.items()
                },
                "suite_horizon": {
                    split: {
                        str(s): int(cohorts[key]["length"][cohorts[key]["suite"] == s].max())
                        for s in sorted(set(cohorts[key]["suite"]))
                    }
                    for split, key in (
                        ("development", "development_main"),
                        ("external", "external_8b"),
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nall 24 survey heads reproduce the published external alarms", flush=True)
    for split, block in counts.items():
        print(f"{split} modes scored: {lib.scored_modes(block)}", flush=True)


if __name__ == "__main__":
    main()
