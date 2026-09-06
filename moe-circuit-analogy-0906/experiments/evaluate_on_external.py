#!/usr/bin/env python3
"""Replay the frozen development selection on external_8b.  Run once.

Thresholds are always fitted on development, never on external:
    global    one pooled quantile of the development_main + development_extra
              per-episode peak distribution.  No task identity anywhere.
    per_task  the same-task development peak quantile.

Nothing is selected here.  Every configuration comes from
results/detectors/development_selection.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import protocol as P


DEFAULT_OUTPUT = P.BUNDLE / "results/detectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def conjunction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both = (left >= 0) & (right >= 0)
    out = np.full(len(left), -1, dtype=np.int16)
    out[both] = np.maximum(left[both], right[both])
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((args.output / "development_selection.json").read_text())
    if frozen["external_opened"]:
        raise ValueError("development selection is not frozen")

    cohorts = {name: P.load_cohort(name) for name in P.COHORTS}
    quantities = P.quantity_list(cohorts["external_8b"])
    lookup = {(family, name): index for family, name, index in quantities}

    ext = cohorts["external_8b"]["circuit"]
    labels = pd.read_csv(P.LABEL_PATHS["external_8b"])
    ext_task = ext["task_names"].astype(str)[ext["task_index"].astype(int)]
    if not np.array_equal(ext_task, labels["task"].to_numpy(str)):
        raise ValueError("external labels are not aligned")
    if not np.array_equal(ext["episode"].astype(int), labels["episode"].to_numpy(int)):
        raise ValueError("external episode ids are not aligned")
    ext_risk = labels["original_failure"].to_numpy(bool)
    ext_suite = P.suite_of(ext)
    ext_priors = P.survival_prior(ext_suite, ext["length"].astype(int), ext_risk)
    ext_valid = ext["valid"].astype(bool)

    reference_task = {
        name: cohorts[name]["circuit"]["task_names"].astype(str)[
            cohorts[name]["circuit"]["task_index"].astype(int)
        ]
        for name in ("development_main", "development_extra")
    }
    covered = set(reference_task["development_main"]) | set(reference_task["development_extra"])
    missing = sorted(set(ext_task) - covered)
    if missing:
        raise ValueError(f"external tasks with no development reference: {missing}")

    crossing = {
        suite: next((q for q, p in sorted(table.items()) if p >= P.LOW_PRIOR), None)
        for suite, table in ext_priors.items()
    }
    horizon = {
        suite: int(ext["length"][ext_suite == suite].max()) for suite in ext_priors
    }

    cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}

    def repr_values(cohort_name: str, family: str, name: str, representation: str) -> np.ndarray:
        key = (cohort_name, family, name)
        if key not in cache:
            cache.clear()
            built = P.dev.representations(
                P.quantity_cache(cohorts[cohort_name], family, lookup[(family, name)])
            )
            cache[key] = {k: v for k, (v, _) in built.items()}
        return cache[key][representation]

    def external_alarm(family: str, name: str, representation: str,
                       direction: str, quantile: float, mode: str) -> np.ndarray:
        values = P.oriented(repr_values("external_8b", family, name, representation), direction)
        persistent = P.dev.persistent_score(values, P.CONFIRMATIONS)
        first = np.full(len(persistent), -1, dtype=np.int16)
        if mode == "global":
            pooled = np.concatenate([
                P.dev.row_max(P.oriented(repr_values(c, family, name, representation), direction))
                for c in ("development_main", "development_extra")
            ])
            line = P.dev.quantile_higher(pooled, quantile)
            if not np.isfinite(line):
                raise ValueError("global threshold is not finite")
            first = P.dev.first_query(np.isfinite(persistent) & (persistent > line) & ext_valid)
        else:
            for task in np.unique(ext_task):
                take = np.flatnonzero(ext_task == task)
                peaks = [
                    P.dev.row_max(P.oriented(
                        repr_values(c, family, name, representation)[reference_task[c] == task],
                        direction))
                    for c in ("development_main", "development_extra")
                    if (reference_task[c] == task).any()
                ]
                line = P.dev.quantile_higher(np.concatenate(peaks), quantile)
                if not np.isfinite(line):
                    raise ValueError(f"per-task threshold is not finite for {task}")
                first[take] = P.dev.first_query(
                    np.isfinite(persistent[take]) & (persistent[take] > line) & ext_valid[take]
                )
        return first

    rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    # deterministic order, and grouped by quantity so the representation cache hits
    order = sorted(
        [s for s in frozen["selection"] if s["feasible"]],
        key=lambda s: (s["family"], s["quantity"], s["mode"]),
    )
    for record in order:
        first = external_alarm(record["family"], record["quantity"],
                               record["representation"], record["direction"],
                               record["quantile"], record["mode"])
        key = f"{record['family']}|{record['quantity']}|{record['mode']}"
        alarms[key] = first
        fired = first >= 0
        alarm_chunk = first[fired]
        rows.append({
            "family": record["family"], "quantity": record["quantity"],
            "mode": record["mode"], "primary": record["primary"],
            "representation": record["representation"], "direction": record["direction"],
            "quantile": record["quantile"],
            "dev_low_prior_tp": record["dev_low_prior_tp"],
            "dev_lift": record["dev_lift"],
            "median_alarm_chunk": float(np.median(alarm_chunk)) if fired.any() else float("nan"),
            **P.score_candidate(first, ext_risk, P.prior_of(first, ext_suite, ext_priors)),
        })
        print(f"  external {key}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["mode", "low_prior_tp"], ascending=[True, False])
    frame.to_csv(args.output / "external_detectors.csv", index=False)

    # ---- pre-declared conjunction --------------------------------------
    declared = frozen["predeclared_conjunction"]
    lead, partner = declared["circuit_lead"], declared["mobility_partner"]
    conj_row = None
    if lead is not None and partner is not None:
        left = external_alarm(lead["family"], lead["quantity"], lead["representation"],
                              lead["direction"], lead["quantile"], "global")
        right = external_alarm(partner["family"], partner["quantity"],
                               partner["representation"], partner["direction"],
                               partner["quantile"], "global")
        first = conjunction(left, right)
        alarms["conjunction|circuit_and_mobility|global"] = first
        conj_row = {
            "left": f"{lead['quantity']}|{lead['representation']}|{lead['direction']}|{lead['quantile']}",
            "right": f"{partner['quantity']}|{partner['representation']}|{partner['direction']}|{partner['quantile']}",
            **P.score_candidate(first, ext_risk, P.prior_of(first, ext_suite, ext_priors)),
        }
        timely = ~ext_risk
        a, b = (left >= 0) & timely, (right >= 0) & timely
        expected = a.sum() * b.sum() / int(timely.sum())
        conj_row["false_alarm_dependence"] = (
            float((a & b).sum() / expected) if expected > 0 else float("nan")
        )
        conj_row["left_only_fp"] = int((a & ~b).sum())
        conj_row["right_only_fp"] = int((b & ~a).sum())

    np.savez_compressed(
        args.output / "external_first_alarms.npz",
        schema=np.asarray("himoe.circuit_analogy.external_alarms.v1"),
        **alarms,
    )
    payload = {
        "schema": "himoe.circuit_analogy.external_evaluation.v1",
        "cohort": "external_8b",
        "evaluations": int(len(frame)),
        "risk_episodes": int(ext_risk.sum()),
        "episodes": int(len(ext_risk)),
        "horizon_cap_by_suite": horizon,
        "prior_crosses_0.25_at_chunk": crossing,
        "prior_estimated_in_sample": True,
        "predeclared_conjunction": conj_row,
        "best_global_circuit": frame[(frame["mode"] == "global") & (frame["family"] == "circuit")]
        .sort_values("low_prior_tp", ascending=False).head(3).to_dict("records"),
        "best_global_published": frame[(frame["mode"] == "global") & (frame["family"] == "published")]
        .sort_values("low_prior_tp", ascending=False).head(3).to_dict("records"),
    }
    (args.output / "external_evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
