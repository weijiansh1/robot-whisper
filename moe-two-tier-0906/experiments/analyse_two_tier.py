#!/usr/bin/env python3
"""Post-external analysis: reproduction checks, uncertainty, budget frontier.

Three jobs, all clearly separated from selection:

1. Reproduce the AND-degeneracy facts the controller reported, as a check that
   this bundle's k-of-n combiner behaves the way theirs did.
2. Put an interval on the one number that would otherwise be over-read: ACT's
   lift, which lands above the 1.764 single-head anchor. Stratified episode
   bootstrap, paired against `mobility|global`.
3. Report which frozen rules actually fit the external false-alarm budget, and
   what worst-mode coverage costs at each budget. Rules are ranked here on
   external, so this table is labelled post-hoc: it describes the frontier, it
   does not select a rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import twotier_lib as lib
from select_early_lock import prior_of, score_candidate


BOOTSTRAP = 4000
SEED = 20260906
SKIP = {"schema", "risk", "suite", "length", "physical_mode"}
SINGLE_HEAD_LIFT_ANCHOR = 1.7640674271437853

# Facts the controller reported on external, restated as checks.
CONTROLLER_CLAIMS = {
    "cross_frame_or_3": {
        "heads": ("mobility|global", "flow_path|global", "expert_load_effective_rank|global"),
        "k": 1,
        "expect": {"tp": 381, "fp": 54},
    },
    "cross_frame_and_3": {
        "heads": ("mobility|global", "flow_path|global", "expert_load_effective_rank|global"),
        "k": 3,
        "expect": {"tp": 0},
    },
    "mobility_and_flow_path": {
        "heads": ("mobility|global", "flow_path|global"),
        "k": 2,
        "expect": {"tp": 0},
    },
    "mobility_and_expert_load": {
        "heads": ("mobility|global", "expert_load_effective_rank|global"),
        "k": 2,
        "expect": {"tp": 69, "fp": 3},
    },
    "mobility_and_flow_settling_per_task": {
        "heads": ("mobility|per_task", "flow_settling_log_ratio|per_task"),
        "k": 2,
        "expect": {"tp": 187, "fp": 8},
    },
    "same_frame_or_3_adjacent_query": {
        "heads": (
            "mobility|global",
            "conditional_query_d1|global",
            "partial_query_d1|global",
        ),
        "k": 1,
        "expect": {"tp": 277, "fp": 207},
    },
}
# Mode statistics the controller also quoted, checked to three decimals.
CONTROLLER_MODE_CLAIMS = {
    "cross_frame_or_3": {"worst_mode_coverage": 0.357, "mode_cv": 0.215},
    "same_frame_or_3_adjacent_query": {"worst_mode_coverage": 0.240, "mode_cv": 0.354},
    "mobility_and_expert_load": {"worst_mode_coverage": 0.000},
}
HEADLINE = {
    "primary|global": {"watch": "watch_a3", "act": "act_b", "single": "mobility|global"},
    "primary|per_task": {"watch": "watch_a3", "act": "act_b", "single": "mobility|global"},
    "secondary_with_v7|global": {
        "watch": "watch_a3",
        "act": "act_b",
        "single": "mobility|global",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lib.RESULTS)
    return parser.parse_args()


def lift_of(first: np.ndarray, risk: np.ndarray, prior: np.ndarray) -> tuple[float, float, float]:
    fired = first >= 0
    tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
    if tp + fp == 0:
        return float("nan"), float("nan"), float("nan")
    precision = tp / (tp + fp)
    base = float(np.nanmean(prior[fired]))
    return precision, base, precision / base


def main() -> None:
    args = parse_args()
    cohort = lib.load_cohort("external_8b")
    data = np.load(args.output / "external_first_alarms.npz", allow_pickle=True)
    alarms = {k: np.asarray(data[k], np.int16) for k in data.files if k not in SKIP}
    modes = data["physical_mode"].astype(str)
    counts = lib.mode_counts(modes, cohort["risk"])
    scored = lib.scored_modes(counts)
    risk = cohort["risk"]
    fp_cap = 0.005 * int((~risk).sum())

    # ---- 1. controller reproduction checks -------------------------------
    checks: dict[str, Any] = {}
    for name, claim in CONTROLLER_CLAIMS.items():
        first = lib.combine([alarms[h] for h in claim["heads"]], claim["k"])
        got = score_candidate(first, risk, prior_of(first, cohort["suite"], cohort["priors"]))
        agree = all(int(got[key]) == int(value) for key, value in claim["expect"].items())
        checks[name] = {
            "heads": list(claim["heads"]),
            "k": claim["k"],
            "expected": claim["expect"],
            "observed_tp": int(got["tp"]),
            "observed_fp": int(got["fp"]),
            "precision": float(got["precision"]),
            "lift": float(got["lift"]) if got["tp"] + got["fp"] else float("nan"),
            "reproduces": bool(agree),
        }
        if not agree:
            print(f"  NOTE {name}: expected {claim['expect']}, got "
                  f"tp={got['tp']} fp={got['fp']}", flush=True)
        else:
            print(f"  ok  {name}: tp={got['tp']} fp={got['fp']}", flush=True)

    for name, want in CONTROLLER_MODE_CLAIMS.items():
        claim = CONTROLLER_CLAIMS[name]
        record = lib.evaluate(
            lib.combine([alarms[h] for h in claim["heads"]], claim["k"]),
            cohort,
            modes,
            scored,
            counts,
        )
        for key, value in want.items():
            checks[name][key] = float(record[key])
            checks[name][f"{key}_reproduces"] = bool(abs(record[key] - value) < 5e-4)
            if not checks[name][f"{key}_reproduces"]:
                print(f"  NOTE {name}.{key}: expected {value}, got {record[key]:.4f}", flush=True)

    # ---- 2. bootstrap on the tiers ---------------------------------------
    external = pd.read_csv(args.output / "external_rules.csv")
    rng = np.random.default_rng(SEED)
    suite = cohort["suite"]
    blocks = [np.flatnonzero(suite == s) for s in np.unique(suite)]
    draws = [
        np.concatenate([rng.choice(b, size=len(b), replace=True) for b in blocks])
        for _ in range(BOOTSTRAP)
    ]

    bootstrap: dict[str, Any] = {}
    for tag, roles in HEADLINE.items():
        pool, mode = tag.split("|")
        block = external[(external["pool"] == pool) & (external["mode"] == mode)]
        entry: dict[str, Any] = {}
        rules: dict[str, np.ndarray] = {"single_mobility_global": alarms[roles["single"]]}
        for role in ("watch", "act"):
            row = block[block["role"] == roles[role]]
            if row.empty:
                continue
            row = row.iloc[0]
            rules[role] = lib.combine(
                [alarms[h] for h in str(row["heads"]).split("+")], int(row["k"])
            )
        priors = {k: prior_of(v, suite, cohort["priors"]) for k, v in rules.items()}
        point = {k: lift_of(rules[k], risk, priors[k]) for k in rules}
        samples = {k: np.empty(BOOTSTRAP) for k in rules}
        paired = np.empty(BOOTSTRAP)
        for draw, index in enumerate(draws):
            for k in rules:
                samples[k][draw] = lift_of(rules[k][index], risk[index], priors[k][index])[2]
            paired[draw] = samples["act"][draw] - samples["single_mobility_global"][draw]
        for k in rules:
            good = np.isfinite(samples[k])
            entry[k] = {
                "precision": point[k][0],
                "mean_matched_prior": point[k][1],
                "lift": point[k][2],
                "lift_ci95": [
                    float(np.percentile(samples[k][good], 2.5)),
                    float(np.percentile(samples[k][good], 97.5)),
                ],
                "p_lift_above_single_head_anchor": float(
                    (samples[k][good] > SINGLE_HEAD_LIFT_ANCHOR).mean()
                ),
            }
        good = np.isfinite(paired)
        entry["act_minus_single_lift"] = {
            "point": float(point["act"][2] - point["single_mobility_global"][2]),
            "ci95": [
                float(np.percentile(paired[good], 2.5)),
                float(np.percentile(paired[good], 97.5)),
            ],
            "p_positive": float((paired[good] > 0).mean()),
        }
        bootstrap[tag] = entry
        print(
            f"  {tag}: act lift {entry['act']['lift']:.3f} "
            f"[{entry['act']['lift_ci95'][0]:.3f}, {entry['act']['lift_ci95'][1]:.3f}] "
            f"vs single {entry['single_mobility_global']['lift']:.3f}; "
            f"paired diff {entry['act_minus_single_lift']['point']:+.3f} "
            f"p(>0)={entry['act_minus_single_lift']['p_positive']:.3f}",
            flush=True,
        )

    # ---- 3. external budget frontier (post-hoc, descriptive) -------------
    watch_like = external[external["role"].str.startswith(("watch", "reference", "same_frame"))]
    frontier = watch_like.assign(fits=lambda f: f["fp"] <= fp_cap).sort_values(
        ["fits", "worst_mode_coverage", "fp"], ascending=[False, False, True]
    )
    frontier[
        [
            "pool",
            "mode",
            "role",
            "heads",
            "n_frames",
            "tp",
            "fp",
            "fits",
            "precision",
            "lift",
            "low_prior_tp",
            "low_prior_fp",
            "worst_mode_coverage",
            "mode_cv",
        ]
    ].to_csv(args.output / "external_watch_frontier.csv", index=False)

    feasible = frontier[frontier["fits"]]
    best_feasible = (
        feasible.iloc[0][["pool", "mode", "role", "heads", "tp", "fp", "worst_mode_coverage", "mode_cv", "lift"]].to_dict()
        if len(feasible)
        else None
    )

    summary = {
        "schema": "himoe.two_tier.analysis.v1",
        "post_hoc": {
            "budget_frontier_ranked_on_external": True,
            "note": "the frontier table ranks frozen rules on external; it describes "
            "the trade-off, it does not select a rule",
        },
        "controller_reproduction": checks,
        "external_fp_cap": fp_cap,
        "bootstrap": bootstrap,
        "bootstrap_draws": BOOTSTRAP,
        "seed": SEED,
        "best_budget_feasible_watch_on_external": lib.plain(best_feasible),
        "single_head_lift_anchor": SINGLE_HEAD_LIFT_ANCHOR,
    }
    (args.output / "analysis.json").write_text(
        json.dumps(lib.plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nwrote analysis.json and external_watch_frontier.csv", flush=True)


if __name__ == "__main__":
    main()
