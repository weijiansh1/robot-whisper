#!/usr/bin/env python3
"""Score the whole predeclared rule grid on development and freeze a shortlist.

This script never opens the external cohort. It writes

    results/development_rules.csv     every rule in the grid
    results/development_baselines.csv singles and pairwise ANDs, same scoring
    results/shortlist.json            the frozen shortlist that external sees

The selection is the one written in results/PREREGISTRATION.json: for each
threshold mode, pool, objective and false-alarm cap, take the single best
development rule under the predeclared constraints and tie-breaks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import combination_core as core


DEFAULT_OUTPUT = core.BUNDLE / "results"
LEVEL_ORDER = {level: position for position, level in enumerate(core.LEVELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def baseline_rows(cohort: core.Cohort) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in core.MODES:
        for level in core.LEVELS:
            for quantity in core.POOLS["all12"]:
                first = cohort.alarms[f"{quantity}|{mode}|{level}"]
                rows.append(
                    {
                        "rule": f"single|{quantity}|{mode}|{level}",
                        "family": "single",
                        "mode": mode,
                        "pool": "single",
                        "level": level,
                        "level_loose": level,
                        "k": 1,
                        "window": -1,
                        "frames": 1,
                        "members": quantity,
                        **cohort.score(first),
                        "alarms": int((first >= 0).sum()),
                    }
                )
        for left, right in itertools.combinations(core.POOLS["all12"], 2):
            first = core.pairwise_and(
                cohort.alarms[f"{left}|{mode}|selected"],
                cohort.alarms[f"{right}|{mode}|selected"],
            )
            rows.append(
                {
                    "rule": f"and|{left}+{right}|{mode}|selected",
                    "family": "pairwise_and",
                    "mode": mode,
                    "pool": "pair",
                    "level": "selected",
                    "level_loose": "selected",
                    "k": 2,
                    "window": -1,
                    "frames": 1,
                    "members": f"{left}+{right}",
                    **cohort.score(first),
                    "alarms": int((first >= 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def rank(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["level_index"] = frame["level"].map(LEVEL_ORDER)
    frame["level_loose_index"] = frame["level_loose"].map(LEVEL_ORDER)
    return frame.sort_values(
        [
            objective,
            "low_prior_precision",
            "precision",
            "mean_alarm_prior",
            "k",
            "window",
            "frames",
            "level_index",
            "level_loose_index",
            "pool",
        ],
        ascending=[False, False, False, True, True, True, True, True, True, True],
        kind="stable",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohort = core.Cohort("development_main")
    rules = core.enumerate_quorum_rules() + core.enumerate_cascade_rules()
    print(f"development cohort: {cohort.episodes} episodes, {int(cohort.risk.sum())} risks")
    print(f"scoring {len(rules):,} predeclared combination rules", flush=True)

    table, _ = core.evaluate_rules(cohort, rules)
    table.to_csv(args.output / "development_rules.csv", index=False)

    baselines = baseline_rows(cohort)
    baselines.to_csv(args.output / "development_baselines.csv", index=False)

    eligible = table[
        (table["low_prior_precision"] >= core.MIN_LOW_PRIOR_PRECISION)
        & (table["tp"] > 0)
    ]
    shortlist: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mode in core.MODES:
        for pool in core.POOLS:
            for objective, column in core.OBJECTIVES.items():
                for cap in core.FPR_CAPS:
                    block = eligible[
                        (eligible["mode"] == mode)
                        & (eligible["pool"] == pool)
                        & (eligible["timely_fpr"] <= cap)
                    ]
                    if block.empty:
                        continue
                    best = rank(block, column).iloc[0]
                    record = {
                        "selected_by": f"{mode}|{pool}|{objective}|fpr<={cap:g}",
                        "objective": objective,
                        "cap": cap,
                        "family": str(best["family"]),
                        "mode": str(best["mode"]),
                        "pool": str(best["pool"]),
                        "level": str(best["level"]),
                        "level_loose": str(best["level_loose"]),
                        "k": int(best["k"]),
                        "window": int(best["window"]),
                        "frames": int(best["frames"]),
                        "development": {
                            key: (
                                value.item()
                                if isinstance(value, np.generic)
                                else value
                            )
                            for key, value in best.drop(
                                labels=["level_index", "level_loose_index"]
                            ).items()
                        },
                    }
                    shortlist.append(record)
                    seen.add(str(best["rule"]))

    # Reference operating points, replayed on external alongside the shortlist.
    references = [
        {"kind": "single", "mode": mode, "quantity": quantity, "level": "selected"}
        for mode in core.MODES
        for quantity in core.POOLS["all12"]
    ]
    dev_pairs = baselines[baselines["family"] == "pairwise_and"]
    for mode in core.MODES:
        block = dev_pairs[
            (dev_pairs["mode"] == mode)
            & (dev_pairs["low_prior_precision"] >= core.MIN_LOW_PRIOR_PRECISION)
            & (dev_pairs["timely_fpr"] <= 0.0015)
        ]
        for objective, column in core.OBJECTIVES.items():
            if block.empty:
                continue
            best = block.sort_values(
                [column, "low_prior_precision", "precision"],
                ascending=False,
                kind="stable",
            ).iloc[0]
            references.append(
                {
                    "kind": "pairwise_and",
                    "mode": mode,
                    "members": str(best["members"]),
                    "level": "selected",
                    "selected_by": f"{mode}|pairwise_and|{objective}|fpr<=0.0015",
                }
            )

    payload = {
        "schema": "himoe.combination_rules.shortlist.v1",
        "prereg": "results/PREREGISTRATION.json",
        "cohort": "development_main",
        "episodes": int(cohort.episodes),
        "risks": int(cohort.risk.sum()),
        "rules_scored": int(len(table)),
        "rules_eligible": int(len(eligible)),
        "unique_shortlisted_rules": int(len(seen)),
        "shortlist": shortlist,
        "references": references,
        "external_not_opened_by_this_script": True,
    }
    (args.output / "shortlist.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    columns = [
        "family", "mode", "pool", "level", "level_loose", "k", "window", "frames",
        "tp", "fp", "timely_fpr", "low_prior_tp", "low_prior_fp",
        "low_prior_precision", "union_coverage_share", "median_confirmation_delay",
    ]
    print(f"\n=== development shortlist ({len(shortlist)} picks, {len(seen)} unique) ===")
    print(
        pd.DataFrame(
            [{"selected_by": r["selected_by"], **{c: r["development"][c] for c in columns}}
             for r in shortlist]
        ).to_string(index=False, float_format="%.4f"),
        flush=True,
    )


if __name__ == "__main__":
    main()
