#!/usr/bin/env python3
"""The single external evaluation declared in results/PREREG.md.

Nothing in the detector protocol is re-implemented: `protocol.py` and the
sweep / external-replay drivers of `detect_step_alarm.py` are imported verbatim
from `moe-flow-semantics-0906/experiments` (see `sweep_lib.py`).

Order of operations, and it matters:

1. reproduce the three published anchors from scratch.  Abort if any differs.
2. sweep the seventeen declared quantities on development_main, select one head
   per (quantity, threshold mode) by the fixed rule.
3. open external once.  Score every selected head, both anchors, the survival
   priors, the per-suite breakdown, the Wilson lift intervals and the
   false-alarm dependence against the published mobility head.
4. write everything, including the heads that failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep_lib import ANCHORS, COHORTS, Cohort, FAMILY, QUANTITIES, D, P  # noqa: E402

DEFAULT_OUTPUT = HERE.parent / "results/detectors"

# Frozen published values.  Source: moe-hb-front-back-0905 frame survey and
# moe-flow-semantics-0906/results/step_alarm/external_detectors.csv.
PUBLISHED_ANCHORS = {
    ("mobility_s9", "global"): dict(
        representation="L12", direction="low", quantile=0.975, tp=195, fp=17,
        precision=0.9198, lift=1.7641,
    ),
    ("mobility_s9", "per_task"): dict(
        representation="L2", direction="low", quantile=0.700, tp=272, fp=57,
    ),
    ("load_entropy_s9", "per_task"): dict(
        representation="L3", direction="low", quantile=0.850, tp=370, fp=93,
        lift=1.5479,
    ),
}
HORIZON_CAP = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}
EARLY_CROSSING = {"libero_goal": 18, "libero_long": 26, "libero_object": 17, "libero_spatial": 13}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Copied from moe-circuit-analogy-0906/experiments/summarise_results.py."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def check_survival_priors(suite: np.ndarray, length: np.ndarray, risk: np.ndarray) -> dict:
    priors = P.survival_prior(suite, length, risk)
    observed = {}
    for name, table in priors.items():
        crossing = next((q for q, p in sorted(table.items()) if p >= P.LOW_PRIOR), None)
        observed[name] = {
            "horizon_cap_observed": int(length[suite == name].max()),
            "horizon_cap_expected": HORIZON_CAP.get(name),
            "prior_crosses_0.25_at": crossing,
            "prior_crosses_expected": EARLY_CROSSING.get(name),
            "n_episodes": int((suite == name).sum()),
            "n_risk": int(risk[suite == name].sum()),
        }
    return observed


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohorts = {name: Cohort(name) for name in COHORTS}
    main_cohort, extra_cohort, external_cohort = (
        cohorts["development_main"], cohorts["development_extra"], cohorts["external_8b"]
    )

    dev_labels = pd.read_csv(P.LABEL_PATHS["development_main"])
    ext_labels = pd.read_csv(P.LABEL_PATHS["external_8b"])
    for cohort, labels in ((main_cohort, dev_labels), (external_cohort, ext_labels)):
        if not np.array_equal(labels["task"].to_numpy(str), cohort.task) or not np.array_equal(
            labels["episode"].to_numpy(int), cohort.episode
        ):
            raise ValueError(f"{cohort.name}: labels are not row aligned")
    dev_risk = dev_labels["original_failure"].to_numpy(bool)
    ext_risk = ext_labels["original_failure"].to_numpy(bool)
    dev_suite = P.suite_of(main_cohort.task_names, main_cohort.task_index)
    ext_suite = P.suite_of(external_cohort.task_names, external_cohort.task_index)
    dev_priors = P.survival_prior(dev_suite, main_cohort.length, dev_risk)
    ext_priors = P.survival_prior(ext_suite, external_cohort.length, ext_risk)

    prior_audit = {
        "development_main": check_survival_priors(dev_suite, main_cohort.length, dev_risk),
        "external_8b": check_survival_priors(ext_suite, external_cohort.length, ext_risk),
    }
    print(json.dumps(prior_audit["external_8b"], indent=2), flush=True)

    order = list(ANCHORS) + list(QUANTITIES)
    development_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    stored: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for quantity in order:
        reprs = {
            name: P.representations(cohorts[name].quantity(quantity), cohorts[name].valid)
            for name in COHORTS
        }
        stored[quantity] = reprs
        rows = D.sweep(
            main_cohort, extra_cohort,
            reprs["development_main"], reprs["development_extra"],
            dev_risk, dev_suite, dev_priors,
        )
        family = FAMILY.get(quantity, "anchor")
        for row in rows:
            row["quantity"], row["family"] = quantity, family
        development_rows.extend(rows)

        frame = pd.DataFrame(rows)
        for mode in P.MODES:
            eligible = frame[
                (frame["mode"] == mode)
                & (frame["timely_fpr"] <= P.MAX_TIMELY_FPR)
                & (frame["low_prior_precision"] >= P.MIN_LOW_PRIOR_PRECISION)
            ]
            if eligible.empty:
                selections.append(
                    {"quantity": quantity, "family": family, "mode": mode, "feasible": False}
                )
                continue
            best = eligible.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]
            selections.append(
                {
                    "quantity": quantity, "family": family, "mode": mode, "feasible": True,
                    "representation": str(best["representation"]),
                    "direction": str(best["direction"]),
                    "quantile": float(best["quantile"]),
                    "dev_tp": int(best["tp"]), "dev_fp": int(best["fp"]),
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    "dev_low_prior_fp": int(best["low_prior_fp"]),
                    "dev_precision": float(best["precision"]),
                    "dev_lift": float(best["lift"]),
                }
            )
        print(f"  development sweep done: {quantity}", flush=True)

    pd.DataFrame(development_rows).to_csv(args.output / "development_grid.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output / "development_selection.csv", index=False)

    # ------------------------------------------------------------------ external
    references = [main_cohort, extra_cohort]
    alarms: dict[str, np.ndarray] = {}
    external_rows: list[dict[str, Any]] = []
    for record in selections:
        if not record.get("feasible"):
            external_rows.append(dict(record))
            continue
        first = D.external_alarm(
            external_cohort, references, stored[record["quantity"]],
            record["representation"], record["direction"], record["quantile"], record["mode"],
        )
        key = f"{record['quantity']}|{record['mode']}"
        alarms[key] = first
        prior = P.prior_of(first, ext_suite, ext_priors)
        scored = P.score_candidate(first, ext_risk, prior)
        fired = first >= 0
        low, high = wilson(scored["tp"], scored["tp"] + scored["fp"])
        row = {
            **record, **scored,
            "precision_ci_low": low, "precision_ci_high": high,
            "lift_ci_low": low / scored["mean_alarm_prior"] if scored["tp"] + scored["fp"] else float("nan"),
            "lift_ci_high": high / scored["mean_alarm_prior"] if scored["tp"] + scored["fp"] else float("nan"),
            "median_alarm_chunk": float(np.median(first[fired])) if fired.any() else float("nan"),
        }
        for name in np.unique(ext_suite):
            take = ext_suite == name
            block = P.score_candidate(first[take], ext_risk[take], prior[take])
            row[f"{name}__tp"] = block["tp"]
            row[f"{name}__fp"] = block["fp"]
            row[f"{name}__low_prior_tp"] = block["low_prior_tp"]
            row[f"{name}__low_prior_fp"] = block["low_prior_fp"]
        external_rows.append(row)

    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_detectors.csv", index=False)

    # ------------------------------------------------------------------- anchors
    anchor_report = {}
    for (quantity, mode), published in PUBLISHED_ANCHORS.items():
        row = external[(external["quantity"] == quantity) & (external["mode"] == mode)].iloc[0]
        reproduced = {
            "representation": str(row["representation"]), "direction": str(row["direction"]),
            "quantile": float(row["quantile"]), "tp": int(row["tp"]), "fp": int(row["fp"]),
            "precision": float(row["precision"]), "lift": float(row["lift"]),
        }
        ok = (
            reproduced["representation"] == published["representation"]
            and reproduced["direction"] == published["direction"]
            and abs(reproduced["quantile"] - published["quantile"]) < 1e-9
            and reproduced["tp"] == published["tp"]
            and reproduced["fp"] == published["fp"]
            and (("precision" not in published) or abs(reproduced["precision"] - published["precision"]) < 5e-4)
            and (("lift" not in published) or abs(reproduced["lift"] - published["lift"]) < 5e-4)
        )
        anchor_report[f"{quantity}|{mode}"] = {
            "published": published, "reproduced": reproduced, "matches": bool(ok)
        }
    print(json.dumps(anchor_report, indent=2), flush=True)
    if not all(entry["matches"] for entry in anchor_report.values()):
        raise RuntimeError("published anchors did NOT reproduce; every number below is void")

    # ------------------------------------------------ false-alarm dependence
    timely = ~ext_risk
    dependence_rows: list[dict[str, Any]] = []
    for mode in P.MODES:
        baseline_key = f"mobility_s9|{mode}"
        if baseline_key not in alarms:
            continue
        baseline = alarms[baseline_key] >= 0
        for key, value in alarms.items():
            if not key.endswith(f"|{mode}"):
                continue
            head = value >= 0
            n = int(timely.sum())
            expected = head[timely].sum() * baseline[timely].sum() / n
            both = int((head & baseline & timely).sum())
            dependence_rows.append(
                {
                    "mode": mode, "head": key.split("|", 1)[0],
                    "head_fp": int((head & timely).sum()),
                    "baseline_fp": int((baseline & timely).sum()),
                    "both_fp": both,
                    "expected_both_if_independent": float(expected),
                    "false_alarm_dependence": float(both / expected) if expected > 0 else float("nan"),
                    "tp": int((head & ext_risk).sum()),
                    "tp_missed_by_mobility": int((head & ext_risk & ~baseline).sum()),
                    "risk_caught_only_by_mobility": int((~head & ext_risk & baseline).sum()),
                    "or_tp": int(((head | baseline) & ext_risk).sum()),
                    "or_fp": int(((head | baseline) & timely).sum()),
                }
            )
    pd.DataFrame(dependence_rows).to_csv(args.output / "false_alarm_dependence.csv", index=False)

    np.savez_compressed(
        args.output / "external_first_alarms.npz",
        schema=np.asarray("himoe.state_channel.alarms.v1"),
        risk=ext_risk, suite=ext_suite.astype("U32"),
        length=external_cohort.length, task=external_cohort.task.astype("U160"),
        episode=external_cohort.episode,
        **alarms,
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.state_channel.detectors.v1",
                "protocol": "imported verbatim from moe-flow-semantics-0906/experiments/protocol.py",
                "lock_parameters": {"width": P.WIDTH, "confirmations": P.CONFIRMATIONS},
                "selection_rule": {
                    "timely_fpr_at_most": P.MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": P.MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                    "selection_reads_development_outcomes_only": True,
                },
                "declared_quantities": list(QUANTITIES),
                "anchors": anchor_report,
                "survival_prior_audit": prior_audit,
                "external_heads_scored": int(len(external)),
                "multiplicity": "17 declared quantities x 2 threshold modes, scored on external in one pass",
            },
            indent=2, sort_keys=True, default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 260)
    columns = ["quantity", "family", "mode", "representation", "direction", "quantile",
               "tp", "fp", "precision", "risk_recall", "low_prior_tp", "low_prior_fp",
               "mean_alarm_prior", "lift", "lift_ci_low", "lift_ci_high", "median_alarm_chunk"]
    for family in ("anchor", "A_primary_formation_slope", "A_secondary_endpoint",
                   "A_negative_control", "B_state_channel"):
        block = external[external["family"] == family]
        if block.empty:
            continue
        print(f"\n=== external :: {family} ===")
        print(block[[c for c in columns if c in block.columns]].to_string(index=False, float_format="%.3f"))
    print("\n=== false-alarm dependence against the published mobility head ===")
    print(pd.DataFrame(dependence_rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
