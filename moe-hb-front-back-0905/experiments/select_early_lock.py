#!/usr/bin/env python3
"""Re-select the lock head against early detection instead of overall precision.

The survival-baseline addendum showed that most of every layer rule's precision
is base rate, and that only 17-26% of detections land where the survival prior
is still low. Overall precision is therefore the wrong selection objective: it
rewards firing late.

This script keeps v4's machinery -- same mobility representations, same causal
trailing mean, same consecutive-confirmation logic, same outcome-blind
same-task quantile thresholds -- and changes only what the grid is ranked by:

    maximise TP in the low-prior band, subject to a v4-level false alarm cap

Selection reads development outcomes only. The winner is rebuilt on external
with same-task reference thresholds and scored there once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import (  # noqa: E402
    LABEL_PATHS,
    MOBILITY_PATHS,
    calibrated_external_alarm,
    load_npz,
    profile_task_map,
)


DEFAULT_OUTPUT = BUNDLE / "results/early_lock"
CANDIDATE_REPRESENTATIONS = (
    "L2",
    "L3",
    "L4",
    "L5",
    "front_median",
    "front_mean",
    "front_min",
    "front_max",
    "back_median",
    "all_median",
    "front_back_mean_gap",
)
LOW_PRIOR = 0.25

# Predeclared before any external cache is opened.
MAX_TIMELY_FPR = 0.005
MIN_RISK_RECALL = 0.40

# Objective A was declared before any run. It maximises early detections and
# ignores their cost, which turned out to be the wrong trade: it buys 8 extra
# early true positives for 35 extra early false positives.
#
# Objective B was declared after seeing that, and asks for the cleanest early
# band that still matches v4's early detection count. It is a second look at
# the same development cohort and is reported as such, never as an independent
# confirmation. Both objectives are scored and both winners are replayed.
OBJECTIVES = {
    "A_max_early_tp": ("low_prior_tp", "low_prior_precision", "risk_recall"),
    "B_max_early_precision": ("low_prior_precision", "low_prior_tp", "risk_recall"),
}
V4_DEVELOPMENT_LOW_PRIOR_TP = 45

# v4's own numbers under the same decomposition, as the bar to clear.
V4_BASELINE = {"low_prior_tp": 76, "low_prior_fp": 54, "tp": 410, "fp": 81}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def suite_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    return np.asarray([name.split("/", 1)[0] for name in task])


def survival_prior(
    suite: np.ndarray, length: np.ndarray, risk: np.ndarray
) -> dict[str, dict[int, float]]:
    priors: dict[str, dict[int, float]] = {}
    for name in np.unique(suite):
        take = suite == name
        priors[str(name)] = {
            chunk: float(risk[take][length[take] > chunk].mean())
            for chunk in range(int(length[take].max()))
        }
    return priors


def prior_of(
    first: np.ndarray, suite: np.ndarray, priors: dict[str, dict[int, float]]
) -> np.ndarray:
    out = np.full(len(first), np.nan)
    fired = first >= 0
    out[fired] = [
        priors[s][int(c)] for s, c in zip(suite[fired], first[fired], strict=True)
    ]
    return out


def score_candidate(
    first: np.ndarray,
    risk: np.ndarray,
    prior: np.ndarray,
) -> dict[str, Any]:
    fired = first >= 0
    early = fired & (prior < LOW_PRIOR)
    tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
    etp, efp = int((early & risk).sum()), int((early & ~risk).sum())
    return {
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / int(risk.sum()),
        "timely_fpr": fp / int((~risk).sum()),
        "precision": tp / max(tp + fp, 1),
        "low_prior_tp": etp,
        "low_prior_fp": efp,
        "low_prior_precision": etp / max(etp + efp, 1),
        "low_prior_share": etp / max(tp, 1),
        "mean_alarm_prior": float(np.nanmean(prior[fired])) if tp + fp else float("nan"),
        "lift": (tp / max(tp + fp, 1)) / float(np.nanmean(prior[fired]))
        if (tp + fp) and np.isfinite(np.nanmean(prior[fired]))
        else float("nan"),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main_cache = load_npz(MOBILITY_PATHS["development_main"])
    extra_cache = load_npz(MOBILITY_PATHS["development_extra"])
    external_cache = load_npz(MOBILITY_PATHS["external_8b"])
    main_repr = {n: v for n, (v, _) in dev.representations(main_cache).items()}
    extra_repr = {n: v for n, (v, _) in dev.representations(extra_cache).items()}
    external_repr = {n: v for n, (v, _) in dev.representations(external_cache).items()}

    labels = pd.read_csv(LABEL_PATHS["development_main"])
    risk = labels["original_failure"].to_numpy(bool)
    suite = suite_of(main_cache)
    length = main_cache["length"].astype(int)
    priors = survival_prior(suite, length, risk)

    task_index = main_cache["task_index"].astype(int)
    init_state = main_cache["init_state_id"].astype(int)
    valid = main_cache["valid"].astype(bool)

    # The frozen v4 L5 instability head, so single-head and two-head candidates
    # compete on equal structural footing. v4 is a lock OR this; ranking a
    # single head against it would confound the representation with the OR.
    l5_oriented = dev.trailing_mean(main_repr["L5"], 4)
    l5_first = dev.first_query(
        np.isfinite(dev.persistent_score(l5_oriented, 8))
        & (
            dev.persistent_score(l5_oriented, 8)
            > dev.crossfit_thresholds(
                dev.row_max(l5_oriented), task_index, init_state
            )[:, dev.QUANTILES.index(0.80)][:, None]
        )
        & valid
    )

    def combine(lock: np.ndarray) -> np.ndarray:
        return np.where(
            lock < 0,
            l5_first,
            np.where(l5_first < 0, lock, np.minimum(lock, l5_first)),
        ).astype(np.int16)

    rows: list[dict[str, Any]] = []
    for representation in CANDIDATE_REPRESENTATIONS:
        values = main_repr[representation]
        for direction in ("low", "high"):
            for width in dev.WIDTHS:
                oriented = dev.trailing_mean(values, width)
                if direction == "low":
                    oriented = -oriented
                thresholds = dev.crossfit_thresholds(
                    dev.row_max(oriented), task_index, init_state
                )
                for confirmations in dev.CONFIRMATIONS:
                    persistent = dev.persistent_score(oriented, confirmations)
                    finite = np.isfinite(persistent)
                    for position, quantile in enumerate(dev.QUANTILES):
                        trigger = (
                            finite
                            & (persistent > thresholds[:, position][:, None])
                            & valid
                        )
                        lock = dev.first_query(trigger)
                        for structure, first in (
                            ("single", lock),
                            ("or_l5", combine(lock)),
                        ):
                            rows.append(
                                {
                                    "representation": representation,
                                    "direction": direction,
                                    "width": width,
                                    "confirmations": confirmations,
                                    "quantile": quantile,
                                    "structure": structure,
                                    **score_candidate(
                                        first, risk, prior_of(first, suite, priors)
                                    ),
                                }
                            )
    candidates = pd.DataFrame(rows)
    candidates.to_csv(args.output / "development_candidates.csv", index=False)
    print(f"scored {len(candidates):,} development candidates", flush=True)

    eligible = candidates[
        (candidates["timely_fpr"] <= MAX_TIMELY_FPR)
        & (candidates["risk_recall"] >= MIN_RISK_RECALL)
    ]
    if eligible.empty:
        raise RuntimeError("no candidate satisfies the predeclared constraints")
    ranked_by: dict[str, pd.DataFrame] = {
        "A_max_early_tp": eligible.sort_values(
            list(OBJECTIVES["A_max_early_tp"]), ascending=False, kind="stable"
        ),
        "B_max_early_precision": eligible[
            eligible["low_prior_tp"] >= V4_DEVELOPMENT_LOW_PRIOR_TP
        ].sort_values(
            list(OBJECTIVES["B_max_early_precision"]), ascending=False, kind="stable"
        ),
    }
    winners = {name: frame.iloc[0] for name, frame in ranked_by.items()}
    ranked = pd.concat(
        [frame.head(args.top).assign(objective=name) for name, frame in ranked_by.items()]
    ).drop_duplicates(
        subset=["representation", "direction", "width", "confirmations", "quantile", "structure"]
    )
    winner = winners["A_max_early_tp"]
    print("\n=== development shortlist (both objectives) ===")
    print(
        ranked[
            [
                "representation",
                "direction",
                "width",
                "confirmations",
                "quantile",
                "structure",
                "tp",
                "fp",
                "low_prior_tp",
                "low_prior_fp",
                "risk_recall",
                "precision",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )

    reference_map = profile_task_map([main_cache, extra_cache])
    reference_representations = {id(main_cache): main_repr, id(extra_cache): extra_repr}
    external_labels = pd.read_csv(LABEL_PATHS["external_8b"])
    external_risk = external_labels["original_failure"].to_numpy(bool)
    external_suite = suite_of(external_cache)
    external_priors = survival_prior(
        external_suite, external_cache["length"].astype(int), external_risk
    )

    external_l5 = calibrated_external_alarm(
        external_cache,
        external_repr["L5"],
        reference_map,
        reference_representations,
        "L5",
        "high",
        4,
        8,
        0.80,
    )

    def external_combine(lock: np.ndarray) -> np.ndarray:
        return np.where(
            lock < 0,
            external_l5,
            np.where(external_l5 < 0, lock, np.minimum(lock, external_l5)),
        ).astype(np.int16)

    external_rows: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        lock = calibrated_external_alarm(
            external_cache,
            external_repr[row["representation"]],
            reference_map,
            reference_representations,
            row["representation"],
            row["direction"],
            int(row["width"]),
            int(row["confirmations"]),
            float(row["quantile"]),
        )
        first = lock if row["structure"] == "single" else external_combine(lock)
        prior = prior_of(first, external_suite, external_priors)
        record = {
            "representation": row["representation"],
            "direction": row["direction"],
            "width": int(row["width"]),
            "confirmations": int(row["confirmations"]),
            "quantile": float(row["quantile"]),
            "structure": row["structure"],
            "selected_by": ",".join(
                name
                for name, pick in winners.items()
                if all(
                    row[key] == pick[key]
                    for key in (
                        "representation",
                        "direction",
                        "width",
                        "confirmations",
                        "quantile",
                        "structure",
                    )
                )
            ),
            **score_candidate(first, external_risk, prior),
        }
        for name in np.unique(external_suite):
            take = external_suite == name
            block = score_candidate(first[take], external_risk[take], prior[take])
            record[f"{name}__low_prior_tp"] = block["low_prior_tp"]
            record[f"{name}__low_prior_fp"] = block["low_prior_fp"]
        external_rows.append(record)

    external_table = pd.DataFrame(external_rows)
    external_table.to_csv(args.output / "external_shortlist.csv", index=False)
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "schema": "himoe.hb_front_back.early_lock.v1",
                "objective": "maximise low-prior TP under a v4-level false alarm cap",
                "low_prior_threshold": LOW_PRIOR,
                "constraints": {
                    "timely_fpr_at_most": MAX_TIMELY_FPR,
                    "risk_recall_at_least": MIN_RISK_RECALL,
                },
                "objectives": {k: list(v) for k, v in OBJECTIVES.items()},
                "objective_B_declared_after_objective_A": True,
                "v4_external_baseline": V4_BASELINE,
                "development_candidates": int(len(candidates)),
                "eligible": int(len(eligible)),
                "winners": {
                    name: {
                        key: (value.item() if isinstance(value, np.generic) else value)
                        for key, value in pick.to_dict().items()
                    }
                    for name, pick in winners.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== external replay of the development shortlist ===")
    print(
        external_table[
            [
                "representation",
                "direction",
                "width",
                "confirmations",
                "quantile",
                "structure",
                "selected_by",
                "tp",
                "fp",
                "low_prior_tp",
                "low_prior_fp",
                "risk_recall",
                "precision",
                "lift",
            ]
        ].to_string(index=False, float_format="%.3f"),
        flush=True,
    )
    print(
        f"\nv4 external baseline: {V4_BASELINE['tp']} TP / {V4_BASELINE['fp']} FP, "
        f"low-prior {V4_BASELINE['low_prior_tp']} TP / {V4_BASELINE['low_prior_fp']} FP",
        flush=True,
    )


if __name__ == "__main__":
    main()
