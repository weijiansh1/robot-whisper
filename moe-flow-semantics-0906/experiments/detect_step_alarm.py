#!/usr/bin/env python3
"""Does a non-final denoising step carry detection signal the final step misses?

Every alarm head published in this project reads the routing tensor at flow step
9.  This script re-runs the identical detector protocol -- same causal trailing
mean (W=4), same consecutive confirmation (K=4), same outcome-blind quantile
grid, same per_task and global threshold modes, same fixed selection rule -- once
per denoising step, so the only thing that changes is which step is read.

Declared before any external cache is opened
--------------------------------------------
primary      cross-query mobility at each of the ten steps.  Ten quantities.
             The object of interest is the profile over steps, not a winner.
derived      two step-axis summaries of the same cross-query mobility:
               mobility_settling_log_ratio  log(mean over steps 7-9 / steps 0-2)
               mobility_step_range          max over steps minus min over steps
             Both are causal: at query q the whole flow of query q has already
             run before the action is emitted.
combination  OR and AND of the selected step-0 head with the selected step-9
             head, per mode.  Built mechanically from the primary selection, no
             extra search.
exploratory  three within-query quantities at four probe steps, to check whether
             the step axis matters for quantities other than mobility.

Selection is on development only, by the fixed rule the layer work used:
    timely_fpr <= 0.005 and low_prior_precision >= 0.60,
    ranked by low_prior_tp then low_prior_precision.
External is opened once, at the end, and every declared head is scored there.
The multiplicity is stated in the summary: nothing here is a single hypothesis.
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
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402


BUNDLE = HERE.parent
PROFILES = BUNDLE / "results/step_profiles"
DEFAULT_OUTPUT = BUNDLE / "results/step_alarm"
COHORTS = ("development_main", "development_extra", "external_8b")
EPSILON = 1e-8

PRIMARY = tuple(f"mobility_s{step}" for step in range(10))
DERIVED = ("mobility_settling_log_ratio", "mobility_step_range")
EXPLORATORY = tuple(
    f"{name}_s{step}"
    for name in ("token_differentiation", "conditional_effective_rank", "load_entropy")
    for step in (0, 3, 6, 9)
) + tuple(f"flow_speed_s{step}" for step in (1, 3, 6, 9))
QUANTITIES = PRIMARY + DERIVED + EXPLORATORY
FAMILY = (
    {name: "primary_step_mobility" for name in PRIMARY}
    | {name: "derived_step_summary" for name in DERIVED}
    | {name: "exploratory_within_query" for name in EXPLORATORY}
)

# The published external mobility heads, reproduced here as the anchor before
# any new number is believed.  Source: moe-hb-front-back-0905 frame survey.
PUBLISHED = {
    "per_task": {"representation": "L2", "direction": "low", "quantile": 0.70, "tp": 272, "fp": 57},
    "global": {"representation": "L12", "direction": "low", "quantile": 0.975, "tp": 195, "fp": 17},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class Cohort:
    def __init__(self, profiles: Path, name: str) -> None:
        index = load_npz(profiles / f"{name}_index.npz")
        self.name = name
        self.task_names = index["task_names"].astype(str)
        self.task_index = index["task_index"].astype(int)
        self.task = self.task_names[self.task_index]
        self.init_state = index["init_state_id"].astype(int)
        self.length = index["length"].astype(int)
        self.valid = index["valid"].astype(bool)
        self.metric_names = index["metric_names"].astype(str).tolist()
        self._mobility = np.load(profiles / f"{name}_mobility.npy", mmap_mode="r")
        self._metrics = np.load(profiles / f"{name}_metrics.npy", mmap_mode="r")
        self._plane: dict[str, np.ndarray] = {}

    def plane(self, metric: str) -> np.ndarray:
        """[n, q, 8, 10] for one metric, cached."""
        if metric not in self._plane:
            if metric == "mobility":
                self._plane[metric] = np.asarray(self._mobility, dtype=np.float32)
            else:
                position = self.metric_names.index(metric)
                self._plane[metric] = np.ascontiguousarray(
                    self._metrics[:, :, :, :, position], dtype=np.float32
                )
        return self._plane[metric]

    def quantity(self, name: str) -> np.ndarray:
        """[n, q, 8] for one declared quantity."""
        if name == "mobility_settling_log_ratio":
            block = self.plane("mobility")
            return np.log(block[..., 7:10].mean(axis=-1) + EPSILON) - np.log(
                block[..., 0:3].mean(axis=-1) + EPSILON
            )
        if name == "mobility_step_range":
            block = self.plane("mobility")
            return block.max(axis=-1) - block.min(axis=-1)
        metric, step = name.rsplit("_s", 1)
        return self.plane(metric)[..., int(step)]

    def release(self) -> None:
        self._plane.clear()


def sweep(
    cohort: Cohort,
    extra: Cohort,
    reprs: dict[str, np.ndarray],
    extra_reprs: dict[str, np.ndarray],
    risk: np.ndarray,
    suite: np.ndarray,
    priors: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in P.REPRESENTATIONS:
        for direction in ("low", "high"):
            oriented = P.oriented_score(reprs[representation], direction)
            persistent = P.persistent_score(oriented, P.CONFIRMATIONS)
            finite = np.isfinite(persistent)
            crossfit = P.crossfit_thresholds(
                P.row_max(oriented), cohort.task_index, cohort.init_state
            )
            pooled = np.concatenate(
                [
                    P.row_max(oriented),
                    P.row_max(P.oriented_score(extra_reprs[representation], direction)),
                ]
            )
            for position, quantile in enumerate(P.QUANTILES):
                for mode in P.MODES:
                    if mode == "per_task":
                        line = crossfit[:, position][:, None]
                    else:
                        line = P.quantile_higher(pooled, quantile)
                        if not np.isfinite(line):
                            continue
                    first = P.first_query(
                        finite & (persistent > line) & cohort.valid
                    )
                    rows.append(
                        {
                            "representation": representation,
                            "direction": direction,
                            "quantile": quantile,
                            "mode": mode,
                            **P.score_candidate(
                                first, risk, P.prior_of(first, suite, priors)
                            ),
                        }
                    )
    return rows


def external_alarm(
    external: Cohort,
    references: list[Cohort],
    reprs: dict[str, dict[str, np.ndarray]],
    representation: str,
    direction: str,
    quantile: float,
    mode: str,
) -> np.ndarray:
    ext_repr = reprs["external_8b"][representation]
    persistent = P.persistent_score(P.oriented_score(ext_repr, direction), P.CONFIRMATIONS)
    finite = np.isfinite(persistent)
    reference_peak = {
        reference.name: P.row_max(
            P.oriented_score(reprs[reference.name][representation], direction)
        )
        for reference in references
    }
    if mode == "global":
        line = P.quantile_higher(
            np.concatenate([reference_peak[r.name] for r in references]), quantile
        )
        return P.first_query(finite & (persistent > line) & external.valid)

    first = np.full(len(persistent), -1, dtype=np.int16)
    for task in np.unique(external.task):
        take = np.flatnonzero(external.task == task)
        peaks = [
            reference_peak[reference.name][reference.task == task]
            for reference in references
            if (reference.task == task).any()
        ]
        line = P.quantile_higher(np.concatenate(peaks), quantile)
        first[take] = P.first_query(
            finite[take] & (persistent[take] > line) & external.valid[take]
        )
    return first


def combine(left: np.ndarray, right: np.ndarray, operation: str) -> np.ndarray:
    fired_left, fired_right = left >= 0, right >= 0
    if operation == "or":
        both = np.minimum(np.where(fired_left, left, 1 << 14), np.where(fired_right, right, 1 << 14))
        return np.where(fired_left | fired_right, both, -1).astype(np.int16)
    both = np.maximum(left, right)
    return np.where(fired_left & fired_right, both, -1).astype(np.int16)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cohorts = {name: Cohort(args.profiles, name) for name in COHORTS}
    main_cohort = cohorts["development_main"]
    extra_cohort = cohorts["development_extra"]
    external_cohort = cohorts["external_8b"]

    dev_labels = pd.read_csv(P.LABEL_PATHS["development_main"])
    ext_labels = pd.read_csv(P.LABEL_PATHS["external_8b"])
    for cohort, labels in ((main_cohort, dev_labels), (external_cohort, ext_labels)):
        if not np.array_equal(labels["task"].to_numpy(str), cohort.task) or not np.array_equal(
            labels["episode"].to_numpy(int), load_npz(args.profiles / f"{cohort.name}_index.npz")["episode"].astype(int)
        ):
            raise ValueError(f"{cohort.name}: labels are not row aligned with the profile")
    dev_risk = dev_labels["original_failure"].to_numpy(bool)
    ext_risk = ext_labels["original_failure"].to_numpy(bool)
    dev_suite = P.suite_of(main_cohort.task_names, main_cohort.task_index)
    ext_suite = P.suite_of(external_cohort.task_names, external_cohort.task_index)
    dev_priors = P.survival_prior(dev_suite, main_cohort.length, dev_risk)
    ext_priors = P.survival_prior(ext_suite, external_cohort.length, ext_risk)

    development_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    stored_values: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for quantity in QUANTITIES:
        reprs = {
            name: P.representations(cohorts[name].quantity(quantity), cohorts[name].valid)
            for name in COHORTS
        }
        stored_values[quantity] = reprs
        rows = sweep(
            main_cohort,
            extra_cohort,
            reprs["development_main"],
            reprs["development_extra"],
            dev_risk,
            dev_suite,
            dev_priors,
        )
        for row in rows:
            row["quantity"] = quantity
            row["family"] = FAMILY[quantity]
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
                    {"quantity": quantity, "family": FAMILY[quantity], "mode": mode, "feasible": False}
                )
                continue
            best = eligible.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]
            selections.append(
                {
                    "quantity": quantity,
                    "family": FAMILY[quantity],
                    "mode": mode,
                    "feasible": True,
                    "representation": str(best["representation"]),
                    "direction": str(best["direction"]),
                    "quantile": float(best["quantile"]),
                    "dev_tp": int(best["tp"]),
                    "dev_fp": int(best["fp"]),
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    "dev_low_prior_fp": int(best["low_prior_fp"]),
                    "dev_lift": float(best["lift"]),
                }
            )
        print(f"  development sweep done: {quantity}", flush=True)

    pd.DataFrame(development_rows).to_csv(
        args.output / "development_grid.csv", index=False
    )
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(args.output / "development_selection.csv", index=False)

    # ---------------------------------------------------------------- external
    references = [main_cohort, extra_cohort]
    alarms: dict[str, np.ndarray] = {}
    external_rows: list[dict[str, Any]] = []
    for record in selections:
        if not record.get("feasible"):
            external_rows.append({**record})
            continue
        first = external_alarm(
            external_cohort,
            references,
            stored_values[record["quantity"]],
            record["representation"],
            record["direction"],
            record["quantile"],
            record["mode"],
        )
        key = f"{record['quantity']}|{record['mode']}"
        alarms[key] = first
        external_rows.append(
            {
                **record,
                **P.score_candidate(
                    first, ext_risk, P.prior_of(first, ext_suite, ext_priors)
                ),
                "median_alarm_chunk": float(np.median(first[first >= 0]))
                if (first >= 0).any()
                else float("nan"),
            }
        )

    # Anchor: the step-9 mobility head must reproduce the published numbers.
    anchors = {}
    for mode, published in PUBLISHED.items():
        record = next(
            r for r in external_rows if r["quantity"] == "mobility_s9" and r["mode"] == mode
        )
        anchors[mode] = {
            "published": published,
            "reproduced": {
                "representation": record["representation"],
                "direction": record["direction"],
                "quantile": record["quantile"],
                "tp": record["tp"],
                "fp": record["fp"],
            },
            "matches": (
                record["representation"] == published["representation"]
                and record["direction"] == published["direction"]
                and abs(record["quantile"] - published["quantile"]) < 1e-9
                and record["tp"] == published["tp"]
                and record["fp"] == published["fp"]
            ),
        }
    print(json.dumps(anchors, indent=2, default=str), flush=True)

    # ------------------------------------------------------- combination heads
    for mode in P.MODES:
        early_key, late_key = f"mobility_s0|{mode}", f"mobility_s9|{mode}"
        if early_key not in alarms or late_key not in alarms:
            continue
        for operation in ("or", "and"):
            first = combine(alarms[early_key], alarms[late_key], operation)
            key = f"mobility_s0_{operation}_s9|{mode}"
            alarms[key] = first
            external_rows.append(
                {
                    "quantity": f"mobility_s0_{operation}_s9",
                    "family": "combination",
                    "mode": mode,
                    "feasible": True,
                    "representation": "combined",
                    "direction": "combined",
                    "quantile": float("nan"),
                    **P.score_candidate(
                        first, ext_risk, P.prior_of(first, ext_suite, ext_priors)
                    ),
                    "median_alarm_chunk": float(np.median(first[first >= 0]))
                    if (first >= 0).any()
                    else float("nan"),
                }
            )

    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_detectors.csv", index=False)

    # ---------------------------------------------- added value over the final step
    added_rows: list[dict[str, Any]] = []
    for mode in P.MODES:
        late_key = f"mobility_s9|{mode}"
        if late_key not in alarms:
            continue
        late = alarms[late_key] >= 0
        for key in [k for k in alarms if k.endswith(f"|{mode}")]:
            early = alarms[key] >= 0
            timely = ~ext_risk
            expected = early[timely].sum() * late[timely].sum() / int(timely.sum())
            added_rows.append(
                {
                    "mode": mode,
                    "head": key.split("|", 1)[0],
                    "tp": int((early & ext_risk).sum()),
                    "fp": int((early & timely).sum()),
                    "tp_missed_by_step9": int((early & ext_risk & ~late).sum()),
                    "fp_added_over_step9": int((early & timely & ~late).sum()),
                    "risk_caught_only_by_step9": int((~early & ext_risk & late).sum()),
                    "false_alarm_dependence": float(
                        (early & late & timely).sum() / expected
                    )
                    if expected > 0
                    else float("nan"),
                    "median_alarm_chunk": float(np.median(alarms[key][early]))
                    if early.any()
                    else float("nan"),
                }
            )
    added = pd.DataFrame(added_rows)
    added.to_csv(args.output / "step_added_value.csv", index=False)

    np.savez_compressed(
        args.output / "external_first_alarms.npz",
        schema=np.asarray("himoe.flow_semantics.step_alarms.v1"),
        **alarms,
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.flow_semantics.step_alarm.v1",
                "protocol": "verbatim copy of the layer-work detector protocol; see experiments/protocol.py",
                "lock_parameters": {"width": P.WIDTH, "confirmations": P.CONFIRMATIONS},
                "selection_rule": {
                    "timely_fpr_at_most": P.MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": P.MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                    "applied_identically_to_every_quantity": True,
                    "selection_reads_development_outcomes_only": True,
                },
                "declared_quantities": {
                    "primary": list(PRIMARY),
                    "derived": list(DERIVED),
                    "exploratory": list(EXPLORATORY),
                    "combination": ["mobility_s0_or_s9", "mobility_s0_and_s9"],
                },
                "external_heads_scored": int(len(external)),
                "multiplicity_note": (
                    "28 declared quantities x 2 threshold modes plus 4 combination "
                    "heads were scored on external in one pass. The primary "
                    "pre-declared comparison is the ten-step mobility profile; the "
                    "single best head over all 60 is optimistically biased and is "
                    "not claimed as a tuned detector."
                ),
                "published_anchor": anchors,
                "baselines_to_beat_on_external": {
                    "global_L12_mobility_step9": {"tp": 195, "fp": 17, "lift": 1.764},
                    "per_task_expert_load_effective_rank": {"tp": 370, "fp": 93, "lift": 1.548},
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 240)
    columns = [
        "quantity",
        "mode",
        "representation",
        "direction",
        "quantile",
        "tp",
        "fp",
        "precision",
        "risk_recall",
        "low_prior_tp",
        "low_prior_fp",
        "mean_alarm_prior",
        "lift",
    ]
    for family in ("primary_step_mobility", "derived_step_summary", "combination", "exploratory_within_query"):
        block = external[external["family"] == family]
        if block.empty:
            continue
        print(f"\n=== external, {family} ===")
        print(block[columns].to_string(index=False, float_format="%.3f"))
    print("\n=== added value over the step-9 head ===")
    print(added.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
