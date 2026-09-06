#!/usr/bin/env python3
"""Replay the frozen development shortlist on the external cohort, once.

Nothing is chosen here. The shortlist comes from results/shortlist.json, which
select_on_development.py wrote without opening the external cache. Three blocks
are scored:

    shortlist   the forty development winners (thirty-two distinct rules)
    reference   the twenty-four frame-survey singles and the best pairwise ANDs
    context     plain OR over the pool at every sensitivity level, so the union
                coverage that any rule is trying to recover is on the same page

The context block is reported, never selected from; it is the denominator of
union_coverage_share and the answer to "how much of the union is reachable".
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def suite_block(cohort: core.Cohort, first: np.ndarray) -> dict[str, Any]:
    prior = cohort.prior_of(first)
    out: dict[str, Any] = {}
    for suite in sorted(set(cohort.suite.tolist())):
        take = cohort.suite == suite
        block = core.score_candidate(first[take], cohort.risk[take], prior[take])
        out[f"{suite}__tp"] = block["tp"]
        out[f"{suite}__fp"] = block["fp"]
        out[f"{suite}__low_prior_tp"] = block["low_prior_tp"]
        out[f"{suite}__low_prior_fp"] = block["low_prior_fp"]
    return out


def main() -> None:
    args = parse_args()
    shortlist = json.loads((args.output / "shortlist.json").read_text(encoding="utf-8"))

    development = core.Cohort("development_main")
    external = core.Cohort("external_8b")
    print(
        f"external cohort: {external.episodes} episodes, "
        f"{int(external.risk.sum())} risks",
        flush=True,
    )

    rules: list[dict[str, Any]] = []
    labels: list[str] = []
    for pick in shortlist["shortlist"]:
        rules.append(
            {
                "family": pick["family"],
                "mode": pick["mode"],
                "pool": pick["pool"],
                "level": pick["level"],
                "level_loose": pick["level_loose"],
                "k": pick["k"],
                "window": pick["window"],
                "frames": pick["frames"],
            }
        )
        labels.append(pick["selected_by"])

    table, firsts = core.evaluate_rules(external, rules)
    table.insert(0, "selected_by", labels)
    table["block"] = "shortlist"
    development_view = pd.DataFrame(
        [
            {
                "selected_by": pick["selected_by"],
                "dev_tp": pick["development"]["tp"],
                "dev_fp": pick["development"]["fp"],
                "dev_timely_fpr": pick["development"]["timely_fpr"],
                "dev_low_prior_tp": pick["development"]["low_prior_tp"],
                "dev_low_prior_fp": pick["development"]["low_prior_fp"],
                "dev_union_coverage_share": pick["development"]["union_coverage_share"],
            }
            for pick in shortlist["shortlist"]
        ]
    )
    table = table.merge(development_view, on="selected_by", validate="one_to_one")
    for position, first in enumerate(firsts.values()):
        for key, value in suite_block(external, first).items():
            table.loc[position, key] = value

    reference_rows: list[dict[str, Any]] = []
    for mode in core.MODES:
        for quantity in core.POOLS["all12"]:
            for cohort_name, cohort in (
                ("development_main", development),
                ("external_8b", external),
            ):
                first = cohort.alarms[f"{quantity}|{mode}|selected"]
                reference_rows.append(
                    {
                        "block": "reference_single",
                        "cohort": cohort_name,
                        "mode": mode,
                        "detector": quantity,
                        **cohort.score(first),
                    }
                )
        for left, right in itertools.combinations(core.POOLS["all12"], 2):
            for cohort_name, cohort in (
                ("development_main", development),
                ("external_8b", external),
            ):
                first = core.pairwise_and(
                    cohort.alarms[f"{left}|{mode}|selected"],
                    cohort.alarms[f"{right}|{mode}|selected"],
                )
                reference_rows.append(
                    {
                        "block": "reference_pairwise_and",
                        "cohort": cohort_name,
                        "mode": mode,
                        "detector": f"{left}+{right}",
                        **cohort.score(first),
                    }
                )
    references = pd.DataFrame(reference_rows)

    context_rows: list[dict[str, Any]] = []
    for mode in core.MODES:
        for pool in core.POOLS:
            for level in core.LEVELS:
                for cohort_name, cohort in (
                    ("development_main", development),
                    ("external_8b", external),
                ):
                    votes = cohort.votes(pool, mode, level)
                    first = core.pool_or(votes)
                    context_rows.append(
                        {
                            "block": "context_union",
                            "cohort": cohort_name,
                            "mode": mode,
                            "pool": pool,
                            "level": level,
                            "members_firing": int((votes >= 0).any(axis=0).sum()),
                            **cohort.score(first),
                        }
                    )
    context = pd.DataFrame(context_rows)

    table.to_csv(args.output / "external_shortlist.csv", index=False)
    references.to_csv(args.output / "reference_operating_points.csv", index=False)
    context.to_csv(args.output / "union_coverage_context.csv", index=False)

    # How many external risks are unreachable by the whole pool at any level.
    reachable = np.zeros(external.episodes, dtype=bool)
    for mode in core.MODES:
        for level in core.LEVELS:
            reachable |= core.pool_or(external.votes("all12", mode, level)) >= 0
    global_reachable = np.zeros(external.episodes, dtype=bool)
    for level in core.LEVELS:
        global_reachable |= core.pool_or(external.votes("all12", "global", level)) >= 0

    summary = {
        "schema": "himoe.combination_rules.external.v1",
        "prereg": "results/PREREGISTRATION.json",
        "external_episodes": int(external.episodes),
        "external_risks": int(external.risk.sum()),
        "external_timely": int((~external.risk).sum()),
        "risks_reachable_by_any_pool_member_any_level_any_mode": int(
            (reachable & external.risk).sum()
        ),
        "risks_reachable_global_mode_only": int(
            (global_reachable & external.risk).sum()
        ),
        "timely_flagged_by_any_pool_member_any_level_any_mode": int(
            (reachable & ~external.risk).sum()
        ),
        "shortlist_rules": int(len(table)),
        "artifacts": {
            "shortlist": "external_shortlist.csv",
            "references": "reference_operating_points.csv",
            "union_context": "union_coverage_context.csv",
        },
    }
    (args.output / "external_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    columns = [
        "selected_by", "family", "mode", "pool", "level", "level_loose",
        "k", "window", "frames",
        "dev_tp", "dev_fp", "tp", "fp", "timely_fpr", "precision",
        "low_prior_tp", "low_prior_fp", "low_prior_precision",
        "mean_alarm_prior", "lift", "union_coverage_share",
    ]
    print("\n=== external replay of the frozen development shortlist ===")
    print(table[columns].to_string(index=False, float_format="%.4f"), flush=True)
    print("\n=== external union coverage context (global mode, all12) ===")
    print(
        context[
            (context["cohort"] == "external_8b")
            & (context["mode"] == "global")
            & (context["pool"] == "all12")
        ][["level", "tp", "fp", "risk_recall", "timely_fpr", "low_prior_tp", "low_prior_fp"]]
        .to_string(index=False, float_format="%.4f"),
        flush=True,
    )
    print("\n" + json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
