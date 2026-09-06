#!/usr/bin/env python3
"""A task-matched survival prior, and what it does to every head in this project.

The `as_entropy` head found in `analyse_as_router.py` is built from a channel
that is bit-for-bit constant inside every episode: its within-episode range is
exactly 0.0. It nevertheless passes the frozen selection rule and reaches an
external lift of 4.40, higher than any published head. The mechanism, verified
in that script, is that `trailing_mean` accumulates float32 rounding across the
chunk axis, the strict `>` test against a per-task quantile flips on those 8
ULPs, and the resulting rule is exactly "still running at chunk 15, in one of
six libero_goal tasks".

That rule is a length-only detector. It scores a high lift because `lift` is
defined against a **per-suite** survival prior, and the per-suite prior does not
absorb the large between-task spread in risk rate inside a suite. A task-matched
prior does absorb it. This script recomputes every head's lift both ways so the
size of the loophole is on the record, and so any claim in this bundle is made
against the stricter denominator.

Task-matched prior of an alarm at chunk q in task t: P(risk | task t, length > q),
estimated on the same cohort, exactly as the suite-matched prior is.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import protocol as P
from protocol import dev
import analyse_as_router as A
import evaluate_discrete_churn as D


DEFAULT_OUTPUT = P.RESULTS
MIN_TASK_STRATUM = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def task_prior(frame: dict[str, Any]) -> dict[str, dict[int, float]]:
    priors: dict[str, dict[int, float]] = {}
    risk, length = frame["risk"], frame["length"]
    for name in np.unique(frame["task"]):
        take = frame["task"] == name
        priors[str(name)] = {
            chunk: float(risk[take][length[take] > chunk].mean())
            for chunk in range(int(length[take].max()))
        }
    return priors


def scored(frame: dict[str, Any], first: np.ndarray, priors_task) -> dict[str, Any]:
    suite_prior = P.prior_of(first, frame["suite"], frame["priors"])
    task_prior_values = P.prior_of(first, frame["task"], priors_task)
    base = P.score_candidate(first, frame["risk"], suite_prior)
    fired = first >= 0
    mean_task = float(np.nanmean(task_prior_values[fired])) if fired.any() else np.nan
    return {
        **base,
        "mean_alarm_task_prior": mean_task,
        "lift_task_matched": (
            base["precision"] / mean_task if np.isfinite(mean_task) and mean_task > 0 else np.nan
        ),
        "distinct_alarm_chunks": int(len(np.unique(first[fired]))),
        "tasks_firing": int(len(np.unique(frame["task"][fired]))),
    }


def main() -> None:
    args = parse_args()
    frames = P.cohort_frames()
    P.assert_anchors(frames)
    external = frames["external_8b"]
    priors_task = task_prior(external)
    values = D.assemble(frames)

    heads: dict[str, np.ndarray] = {}

    # the three published anchors
    for quantity, representation, direction, quantile, mode, *_ in P.ANCHORS:
        reference = [
            (
                frames[c],
                {
                    n: b
                    for n, (b, _) in dev.representations(
                        P.layer_cache(frames[c], P.quantity_values(frames[c], quantity))
                    ).items()
                }[representation],
            )
            for c in ("development_main", "development_extra")
        ]
        target = {
            n: b
            for n, (b, _) in dev.representations(
                P.layer_cache(external, P.quantity_values(external, quantity))
            ).items()
        }[representation]
        heads[f"anchor|{quantity}|{representation}|{direction}|q{quantile:g}|{mode}"] = (
            P.external_alarm(external, target, reference, direction, quantile, mode)
        )

    # this bundle's discrete heads, replayed from the recorded selections
    selections = pd.read_csv(args.output / "discrete_external_detectors.csv")
    for _, row in selections[selections["feasible"].fillna(False)].iterrows():
        quantity, mode = row["quantity"], row["mode"]
        reference = [
            (
                frames[c],
                {
                    n: b
                    for n, (b, _) in dev.representations(
                        P.layer_cache(frames[c], values[c][quantity])
                    ).items()
                }[row["representation"]],
            )
            for c in ("development_main", "development_extra")
        ]
        target = {
            n: b
            for n, (b, _) in dev.representations(
                P.layer_cache(external, values["external_8b"][quantity])
            ).items()
        }[row["representation"]]
        heads[f"discrete|{quantity}|{mode}"] = P.external_alarm(
            external, target, reference, row["direction"], float(row["quantile"]), mode
        )

    # the two as_* heads, which carry no within-episode information at all
    as_selections = pd.read_csv(args.output / "as_external_detectors.csv")
    for _, row in as_selections[as_selections["feasible"].fillna(False)].iterrows():
        scalar = row["quantity"]
        reprs = {c: A.as_representations(A.as_series(frames[c], scalar)) for c in frames}
        heads[f"as_router|{scalar}|{row['mode']}"] = P.external_alarm(
            external,
            reprs["external_8b"][row["representation"]],
            [(frames[c], reprs[c][row["representation"]]) for c in
             ("development_main", "development_extra")],
            row["direction"],
            float(row["quantile"]),
            row["mode"],
        )

    # the causal length-only baseline
    crossing = {"libero_goal": 18, "libero_long": 26, "libero_object": 17, "libero_spatial": 13}
    for label, chunk_of in (
        ("length_only|prior025_crossing", crossing),
        ("length_only|chunk15", {s: 15 for s in crossing}),
        ("length_only|chunk15_libero_goal_only", {"libero_goal": 15}),
    ):
        first = np.full(len(external["risk"]), -1, dtype=np.int16)
        for suite, chunk in chunk_of.items():
            take = (external["suite"] == suite) & (external["length"] > chunk)
            first[take] = chunk
        heads[label] = first

    rows = [
        {"head": name, **scored(external, first, priors_task)}
        for name, first in heads.items()
    ]
    table = pd.DataFrame(rows).sort_values("lift", ascending=False)
    table.to_csv(args.output / "task_matched_lift.csv", index=False)

    pd.set_option("display.width", 260)
    print(
        table[
            [
                "head", "tp", "fp", "precision", "risk_recall",
                "mean_alarm_prior", "lift", "mean_alarm_task_prior", "lift_task_matched",
                "low_prior_tp", "low_prior_fp", "tasks_firing", "distinct_alarm_chunks",
            ]
        ].to_string(index=False, float_format="%.4f")
    )

    within_suite_spread = {}
    for suite in np.unique(external["suite"]):
        take = external["suite"] == suite
        rates = [
            float(external["risk"][external["task"] == t].mean())
            for t in np.unique(external["task"][take])
        ]
        within_suite_spread[str(suite)] = {
            "tasks": len(rates),
            "min_task_risk_rate": min(rates),
            "max_task_risk_rate": max(rates),
            "suite_risk_rate": float(external["risk"][take].mean()),
        }
    P.write_json(
        args.output / "task_matched_lift.json",
        {
            "schema": "himoe.unused_channels.task_matched_lift.v1",
            "definition": {
                "suite_matched": "precision / mean_q P(risk | suite, length > q), the published lift",
                "task_matched": "precision / mean_q P(risk | task, length > q)",
            },
            "why": (
                "the as_entropy head carries exactly zero within-episode information "
                "yet scores the highest suite-matched lift in this bundle; the gap "
                "between the two denominators is the size of the loophole"
            ),
            "within_suite_task_risk_spread": within_suite_spread,
            "heads": table.to_dict("records"),
        },
    )
    print("\n=== between-task spread of the risk rate inside each suite (external) ===")
    print(pd.DataFrame(within_suite_spread).T.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
