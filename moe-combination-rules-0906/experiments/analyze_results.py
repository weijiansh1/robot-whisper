#!/usr/bin/env python3
"""Post-evaluation analysis: premise checks, frontiers, suite splits, timing.

Nothing here selects a rule. It reads the frozen shortlist scores and the
reference/context blocks and answers the questions the report has to answer:

  * does the stated premise hold -- does a four detector union really cover
    525 of the 564 external risks
  * where does each frozen rule sit relative to the reference operating points
    and relative to the plain-OR sensitivity curve
  * is the gain recall or is it just firing later, per suite and per chunk
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
HORIZON_CAP = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}
HEADLINE = (
    "global|all12|O2_recall|fpr<=0.001",
    "global|all12|O2_recall|fpr<=0.0005",
    "global|all12|O2_recall|fpr<=0.003",
    "per_task|all12|O2_recall|fpr<=0.0015",
    "per_task|all12|O2_recall|fpr<=0.003",
    "per_task|dedup11|O1_early|fpr<=0.003",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def frontier(frame: pd.DataFrame, cost: str, gain: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost, gain], ascending=[True, False], kind="stable")
    best = -1
    keep = []
    for position, row in enumerate(ordered.itertuples()):
        value = getattr(row, gain)
        if value > best:
            best = value
            keep.append(position)
    return ordered.iloc[keep]


def main() -> None:
    args = parse_args()
    external = core.Cohort("external_8b")
    development = core.Cohort("development_main")
    shortlist = pd.read_csv(args.output / "external_shortlist.csv")
    references = pd.read_csv(args.output / "reference_operating_points.csv")
    context = pd.read_csv(args.output / "union_coverage_context.csv")

    report: dict[str, Any] = {"schema": "himoe.combination_rules.analysis.v1"}

    # 1. Premise check: how large is the union of the strongest existing detectors.
    premise: dict[str, Any] = {}
    quartet = (
        "expert_load_effective_rank|per_task|selected",
        "flow_settling_log_ratio|per_task|selected",
        "mobility|per_task|selected",
        "mobility|global|selected",
    )
    for name, keys in {
        "four_strongest_mixed_mode": quartet,
        "all12_per_task_selected": tuple(
            f"{q}|per_task|selected" for q in core.POOLS["all12"]
        ),
        "all12_global_selected": tuple(
            f"{q}|global|selected" for q in core.POOLS["all12"]
        ),
        "all24_selected": tuple(
            f"{q}|{m}|selected" for q in core.POOLS["all12"] for m in core.MODES
        ),
    }.items():
        fired = np.zeros(external.episodes, dtype=bool)
        for key in keys:
            fired |= external.alarms[key] >= 0
        premise[name] = {
            "detectors": len(keys),
            "tp": int((fired & external.risk).sum()),
            "fp": int((fired & ~external.risk).sum()),
            "risk_recall": float((fired & external.risk).sum() / external.risk.sum()),
            "risks_missed": int((~fired & external.risk).sum()),
        }
    # Best four-detector union by external coverage, reported as an upper bound
    # on the premise, not as a selection.
    keys = [f"{q}|{m}|selected" for q in core.POOLS["all12"] for m in core.MODES]
    fired = {key: external.alarms[key] >= 0 for key in keys}
    best = max(
        itertools.combinations(keys, 4),
        key=lambda combo: int(
            (np.logical_or.reduce([fired[key] for key in combo]) & external.risk).sum()
        ),
    )
    union = np.logical_or.reduce([fired[key] for key in best])
    premise["best_possible_four_detector_union"] = {
        "members": list(best),
        "tp": int((union & external.risk).sum()),
        "fp": int((union & ~external.risk).sum()),
        "risk_recall": float((union & external.risk).sum() / external.risk.sum()),
        "risks_missed": int((~union & external.risk).sum()),
        "note": "chosen by external coverage, so this is post-hoc and only quoted as the premise's upper bound",
    }
    report["premise"] = premise

    # 2. Survival prior geometry.
    prior_geometry = {}
    for cohort_name, cohort in (("development_main", development), ("external_8b", external)):
        block = {}
        for suite in sorted(cohort.priors):
            curve = cohort.priors[suite]
            crossing = next(
                (chunk for chunk in sorted(curve) if curve[chunk] >= core.LOW_PRIOR), None
            )
            block[suite] = {
                "max_chunk_observed": int(max(curve)),
                "declared_horizon_cap": HORIZON_CAP.get(suite),
                "first_chunk_with_prior_at_least_0.25": crossing,
                "prior_at_chunk_0": float(curve[0]),
            }
        prior_geometry[cohort_name] = block
    report["survival_prior_geometry"] = prior_geometry

    # 3. Frontiers: frozen rules against reference operating points and the OR curve.
    frontier_rows: list[dict[str, Any]] = []
    for mode in core.MODES:
        pieces = [
            shortlist[shortlist["mode"] == mode].assign(
                block="frozen_rule", label=shortlist["selected_by"]
            ),
            references[
                (references["cohort"] == "external_8b") & (references["mode"] == mode)
            ].assign(label=lambda f: f["block"] + ":" + f["detector"]),
            context[
                (context["cohort"] == "external_8b")
                & (context["mode"] == mode)
                & (context["pool"] == "all12")
            ].assign(label=lambda f: "or_all12@" + f["level"].astype(str)),
        ]
        merged = pd.concat(
            [
                piece[
                    [
                        "block", "label", "tp", "fp", "precision", "risk_recall",
                        "timely_fpr", "low_prior_tp", "low_prior_fp",
                        "low_prior_precision", "mean_alarm_prior", "lift",
                    ]
                ]
                for piece in pieces
            ],
            ignore_index=True,
        )
        merged["mode"] = mode
        for cost, gain, name in (
            ("fp", "tp", "overall"),
            ("low_prior_fp", "low_prior_tp", "early"),
        ):
            edge = frontier(merged, cost, gain).assign(frontier=name)
            frontier_rows.append(edge)
    frontiers = pd.concat(frontier_rows, ignore_index=True)
    frontiers.to_csv(args.output / "external_frontiers.csv", index=False)

    # 4. Headline rules: suite split and alarm timing against the references.
    rules = {pick["selected_by"]: pick for _, pick in shortlist.iterrows()}
    timing_rows: list[dict[str, Any]] = []
    for label in HEADLINE:
        pick = rules[label]
        rule = {
            "family": pick["family"], "mode": pick["mode"], "pool": pick["pool"],
            "level": str(pick["level"]), "level_loose": str(pick["level_loose"]),
            "k": int(pick["k"]), "window": int(pick["window"]),
            "frames": int(pick["frames"]),
        }
        _, firsts = core.evaluate_rules(external, [rule])
        first = next(iter(firsts.values()))
        timing_rows.append(
            {
                "label": label,
                "kind": "frozen_rule",
                **alarm_timing(external, first),
            }
        )
    for mode, detector in (
        ("global", "mobility"),
        ("global", "expert_load_effective_rank"),
        ("per_task", "expert_load_effective_rank"),
        ("per_task", "mobility"),
    ):
        timing_rows.append(
            {
                "label": f"single:{detector}|{mode}",
                "kind": "reference",
                **alarm_timing(external, external.alarms[f"{detector}|{mode}|selected"]),
            }
        )
    for mode, left, right in (
        ("per_task", "expert_load_effective_rank", "mobility"),
        ("per_task", "flow_settling_log_ratio", "mobility"),
        ("global", "conditional_query_d1", "mobility"),
    ):
        first = core.pairwise_and(
            external.alarms[f"{left}|{mode}|selected"],
            external.alarms[f"{right}|{mode}|selected"],
        )
        timing_rows.append(
            {
                "label": f"and:{left}+{right}|{mode}",
                "kind": "reference",
                **alarm_timing(external, first),
            }
        )
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(args.output / "external_alarm_timing.csv", index=False)

    (args.output / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 250)
    print("=== premise check on external ===")
    print(json.dumps(premise, indent=2, sort_keys=True, default=str))
    print("\n=== external overall frontier, global mode ===")
    print(
        frontiers[(frontiers["mode"] == "global") & (frontiers["frontier"] == "overall")][
            ["label", "block", "tp", "fp", "precision", "low_prior_tp", "low_prior_fp", "mean_alarm_prior", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )
    print("\n=== external early frontier, global mode ===")
    print(
        frontiers[(frontiers["mode"] == "global") & (frontiers["frontier"] == "early")][
            ["label", "block", "low_prior_tp", "low_prior_fp", "tp", "fp", "low_prior_precision", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )
    print("\n=== external overall frontier, per_task mode ===")
    print(
        frontiers[(frontiers["mode"] == "per_task") & (frontiers["frontier"] == "overall")][
            ["label", "block", "tp", "fp", "precision", "low_prior_tp", "low_prior_fp", "mean_alarm_prior", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )
    print("\n=== external early frontier, per_task mode ===")
    print(
        frontiers[(frontiers["mode"] == "per_task") & (frontiers["frontier"] == "early")][
            ["label", "block", "low_prior_tp", "low_prior_fp", "tp", "fp", "low_prior_precision", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )
    print("\n=== alarm timing ===")
    print(timing.to_string(index=False, float_format="%.3f"))


def alarm_timing(cohort: core.Cohort, first: np.ndarray) -> dict[str, Any]:
    fired = first >= 0
    risk = cohort.risk
    out: dict[str, Any] = {
        "tp": int((fired & risk).sum()),
        "fp": int((fired & ~risk).sum()),
        "median_tp_chunk": float(np.median(first[fired & risk])) if (fired & risk).any() else float("nan"),
        "median_tp_lead_chunks": float(
            np.median(cohort.length[fired & risk] - 1 - first[fired & risk])
        )
        if (fired & risk).any()
        else float("nan"),
        "mean_alarm_prior": float(np.nanmean(cohort.prior_of(first)[fired])),
    }
    for suite in sorted(set(cohort.suite.tolist())):
        take = cohort.suite == suite
        out[f"{suite}_tp"] = int((fired & risk & take).sum())
        out[f"{suite}_fp"] = int((fired & ~risk & take).sum())
        out[f"{suite}_risks"] = int((risk & take).sum())
    return out


if __name__ == "__main__":
    main()
