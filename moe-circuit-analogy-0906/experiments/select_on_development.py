#!/usr/bin/env python3
"""Score every circuit quantity as a causal alarm on development_main only.

The external cohort is never opened here.  Two arms are scored under identical
code so the comparison is apples to apples:

    circuit    the 31 electrical-network quantities of this bundle
    published  the 11 HB layer-graph metrics plus layerwise mobility, i.e. the
               quantities the existing baselines are built from (control arm)

Selection rule is byte-identical to the published frame survey:
    timely_fpr <= 0.005 and low_prior_precision >= 0.60,
    then rank by (low_prior_tp, low_prior_precision) descending.

One conjunction is pre-declared here, before external is opened: the best
primary circuit detector in `global` mode AND the best `mobility` detector in
`global` mode, firing at max(first_a, first_b) when both fire.  It tests
whether the circuit view adds anything the published view does not already
have.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import protocol as P


DEFAULT_OUTPUT = P.BUNDLE / "results/detectors"

_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def prepare(cohort_name: str) -> dict[str, Any]:
    cohort = P.load_cohort(cohort_name)
    circuit = cohort["circuit"]
    labels = pd.read_csv(P.LABEL_PATHS[cohort_name])
    task = circuit["task_names"].astype(str)[circuit["task_index"].astype(int)]
    if not np.array_equal(task, labels["task"].to_numpy(str)):
        raise ValueError("label rows are not aligned with the feature cache")
    if not np.array_equal(circuit["episode"].astype(int), labels["episode"].to_numpy(int)):
        raise ValueError("episode ids are not aligned")
    risk = labels["original_failure"].to_numpy(bool)
    suite = P.suite_of(circuit)
    return {
        "cohort": cohort,
        "risk": risk,
        "suite": suite,
        "priors": P.survival_prior(suite, circuit["length"].astype(int), risk),
        "task_index": circuit["task_index"].astype(int),
        "init_state": circuit["init_state_id"].astype(int),
        "valid": circuit["valid"].astype(bool),
    }


def init_worker() -> None:
    _STATE.update(prepare("development_main"))


def score_quantity(payload: tuple[str, str, int]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    family, name, index = payload
    reprs = {k: v for k, (v, _) in
             P.dev.representations(P.quantity_cache(_STATE["cohort"], family, index)).items()}
    rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    for representation in P.REPRESENTATIONS:
        for direction in ("low", "high"):
            values = P.oriented(reprs[representation], direction)
            persistent = P.dev.persistent_score(values, P.CONFIRMATIONS)
            peak = P.dev.row_max(values)
            crossfit = P.dev.crossfit_thresholds(peak, _STATE["task_index"], _STATE["init_state"])
            for position, quantile in enumerate(P.dev.QUANTILES):
                for mode in ("per_task", "global"):
                    if mode == "per_task":
                        line = crossfit[:, position][:, None]
                    else:
                        line = P.dev.quantile_higher(peak, quantile)
                        if not np.isfinite(line):
                            continue
                    first = P.dev.first_query(
                        np.isfinite(persistent) & (persistent > line) & _STATE["valid"]
                    )
                    rows.append({
                        "family": family,
                        "quantity": name,
                        "representation": representation,
                        "direction": direction,
                        "quantile": quantile,
                        "mode": mode,
                        "primary": family == "circuit" and name in P.PRIMARY_FEATURES,
                        **P.score_candidate(
                            first, _STATE["risk"],
                            P.prior_of(first, _STATE["suite"], _STATE["priors"]),
                        ),
                    })
                    alarms[f"{family}|{name}|{representation}|{direction}|{quantile}|{mode}"] = (
                        first.astype(np.int16)
                    )
    return rows, alarms


def conjunction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both = (left >= 0) & (right >= 0)
    out = np.full(len(left), -1, dtype=np.int16)
    out[both] = np.maximum(left[both], right[both])
    return out


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = P.load_cohort("development_main")
    jobs = P.quantity_list(cohort)

    collected: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        for rows, block in pool.imap_unordered(score_quantity, jobs, chunksize=1):
            collected.extend(rows)
            alarms.update(block)
            print(f"  scored {rows[0]['family']}/{rows[0]['quantity']}", flush=True)

    frame = pd.DataFrame(collected).sort_values(
        ["family", "quantity", "representation", "direction", "mode", "quantile"], kind="stable"
    )
    frame.to_csv(args.output / "development_candidates.csv", index=False)

    def pick(subset: pd.DataFrame) -> dict[str, Any] | None:
        eligible = subset[
            (subset["timely_fpr"] <= P.MAX_TIMELY_FPR)
            & (subset["low_prior_precision"] >= P.MIN_LOW_PRIOR_PRECISION)
        ]
        if eligible.empty:
            return None
        best = eligible.sort_values(
            ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
        ).iloc[0]
        return {
            "family": str(best["family"]),
            "quantity": str(best["quantity"]),
            "representation": str(best["representation"]),
            "direction": str(best["direction"]),
            "quantile": float(best["quantile"]),
            "dev_tp": int(best["tp"]), "dev_fp": int(best["fp"]),
            "dev_precision": float(best["precision"]),
            "dev_lift": float(best["lift"]),
            "dev_risk_recall": float(best["risk_recall"]),
            "dev_low_prior_tp": int(best["low_prior_tp"]),
            "dev_low_prior_precision": float(best["low_prior_precision"]),
        }

    selection: list[dict[str, Any]] = []
    for family, name, _ in jobs:
        for mode in ("per_task", "global"):
            chosen = pick(frame[(frame["family"] == family) & (frame["quantity"] == name)
                                & (frame["mode"] == mode)])
            record = {"family": family, "quantity": name, "mode": mode,
                      "primary": family == "circuit" and name in P.PRIMARY_FEATURES,
                      "feasible": chosen is not None}
            if chosen is not None:
                record.update(chosen)
            selection.append(record)

    # ---- pre-declared conjunction --------------------------------------
    circuit_lead = pick(frame[(frame["family"] == "circuit") & frame["primary"]
                              & (frame["mode"] == "global")])
    mobility_global = pick(frame[(frame["quantity"] == "mobility") & (frame["mode"] == "global")])
    conj = None
    if circuit_lead is not None and mobility_global is not None:
        key_a = ("{family}|{quantity}|{representation}|{direction}|{quantile}|global"
                 .format(**circuit_lead))
        key_b = ("{family}|{quantity}|{representation}|{direction}|{quantile}|global"
                 .format(**mobility_global))
        first = conjunction(alarms[key_a], alarms[key_b])
        state = prepare("development_main")
        conj = {
            "left": key_a, "right": key_b,
            **P.score_candidate(first, state["risk"],
                                P.prior_of(first, state["suite"], state["priors"])),
        }

    payload = {
        "schema": "himoe.circuit_analogy.development_selection.v2",
        "cohort": "development_main",
        "external_opened": False,
        "arms": {"circuit": 31, "published": 12},
        "width": P.WIDTH,
        "confirmations": P.CONFIRMATIONS,
        "quantiles": list(P.dev.QUANTILES),
        "representations": list(P.REPRESENTATIONS),
        "primary_features": list(P.PRIMARY_FEATURES),
        "selection_rule": {
            "timely_fpr_at_most": P.MAX_TIMELY_FPR,
            "low_prior_precision_at_least": P.MIN_LOW_PRIOR_PRECISION,
            "ranking": ["low_prior_tp", "low_prior_precision"],
        },
        "candidate_rows": int(len(frame)),
        "selection": selection,
        "predeclared_conjunction": {
            "definition": "fires at max(first_left, first_right) when both fire",
            "circuit_lead": circuit_lead,
            "mobility_partner": mobility_global,
            "development": conj,
        },
    }
    (args.output / "development_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidate_rows": len(frame),
                      "feasible": sum(s["feasible"] for s in selection),
                      "of": len(selection),
                      "circuit_lead": circuit_lead,
                      "conjunction_dev": conj}, indent=2))


if __name__ == "__main__":
    main()
