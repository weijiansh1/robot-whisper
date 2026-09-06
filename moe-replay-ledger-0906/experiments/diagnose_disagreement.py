#!/usr/bin/env python3
"""Locate the partial_edge_std|global reproducibility failure.

moe-hb-front-back-0905 publishes `partial_edge_std | global` as L3 / high /
q0.90 -> 99 TP / 147 FP. moe-circuit-analogy-0906 publishes the same quantity in
the same mode under the same frozen selection rule as L3 / high / q0.95 ->
42 TP / 35 FP. This script asks which half of the pipeline the two bundles stop
agreeing in: the development candidate table that drives the selection, or the
external scoring that follows it.

The test: rebuild the frame-survey pipeline at the *other* bundle's quantile and
see whether it lands on the other bundle's external numbers. If it does, the
external scoring path is shared and the disagreement is upstream, in the feature
values that build the development table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_core import (  # noqa: E402
    CIRCUIT,
    PROJECT,
    CONFIRMATIONS,
    HB,
    MOBILITY_PATHS,
    PROFILE_ROOT,
    Cohort,
    _npz,
    dev,
    load_npz,
    oriented,
    quantity_values,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUSPECTS = ("partial_edge_std", "conditional_effective_rank", "partial_query_d1")


def main() -> None:
    cohort = Cohort("external_8b")
    graphs = {
        n: load_npz(PROFILE_ROOT / f"{n}.npz")
        for n in ("development_main", "development_extra")
    }
    mobs = {n: load_npz(MOBILITY_PATHS[n]) for n in graphs}
    graphs["external_8b"] = cohort.graph_cache
    mobs["external_8b"] = cohort.mobility_cache

    survey = pd.read_csv(HB / "results/frame_survey/external_detectors.csv")
    circ = pd.read_csv(CIRCUIT / "results/detectors/external_detectors.csv")
    circ_arrays = _npz(CIRCUIT / "results/detectors/external_first_alarms.npz")
    fs_arrays = _npz(HB / "results/frame_survey/external_first_alarms.npz")

    report = {
        "schema": "himoe.replay_ledger.bundle_disagreement.v1",
        "question": (
            "two bundles publish different operating points for the same quantity "
            "under the same frozen selection rule; where do they diverge?"
        ),
        "heads": [],
    }

    for quantity in SUSPECTS:
        fs_row = survey[(survey["quantity"] == quantity) & (survey["mode"] == "global")]
        c_row = circ[(circ["quantity"] == quantity) & (circ["mode"] == "global")]
        if len(fs_row) != 1 or len(c_row) != 1:
            continue
        rep = str(fs_row.iloc[0]["representation"])
        direction = str(fs_row.iloc[0]["direction"])
        reprs = {
            n: {
                k: v
                for k, (v, _) in dev.representations(
                    quantity_values(graphs[n], mobs[n], quantity)
                ).items()
            }
            for n in graphs
        }
        pooled = np.concatenate(
            [
                dev.row_max(oriented(reprs[n][rep], direction))
                for n in ("development_main", "development_extra")
            ]
        )
        persistent = dev.persistent_score(
            oriented(reprs["external_8b"][rep], direction), CONFIRMATIONS
        )
        sweep = []
        for quantile in dev.QUANTILES:
            line = dev.quantile_higher(pooled, quantile)
            first = dev.first_query(
                np.isfinite(persistent) & (persistent > line) & cohort.valid
            )
            fired = first >= 0
            sweep.append(
                {
                    "quantile": quantile,
                    "threshold": float(line),
                    "tp": int((fired & cohort.risk).sum()),
                    "fp": int((fired & ~cohort.risk).sum()),
                }
            )
        c_quantile = float(c_row.iloc[0]["quantile"])
        at_circuit_q = [s for s in sweep if abs(s["quantile"] - c_quantile) < 1e-9]
        entry = {
            "quantity": quantity,
            "frame_survey": {
                "representation": rep,
                "direction": direction,
                "quantile": float(fs_row.iloc[0]["quantile"]),
                "tp": int(fs_row.iloc[0]["tp"]),
                "fp": int(fs_row.iloc[0]["fp"]),
            },
            "circuit_analogy": {
                "representation": str(c_row.iloc[0]["representation"]),
                "direction": str(c_row.iloc[0]["direction"]),
                "quantile": c_quantile,
                "tp": int(c_row.iloc[0]["tp"]),
                "fp": int(c_row.iloc[0]["fp"]),
            },
            "same_representation_and_direction": (
                rep == str(c_row.iloc[0]["representation"])
                and direction == str(c_row.iloc[0]["direction"])
            ),
            "frame_survey_pipeline_at_circuits_quantile": at_circuit_q[0]
            if at_circuit_q
            else None,
            "arrays_identical": bool(
                np.array_equal(
                    fs_arrays[f"{quantity}|global"],
                    circ_arrays.get(f"published|{quantity}|global", np.zeros(1)),
                )
            ),
            "external_quantile_sweep": sweep,
        }
        # is the circuit array reproduced by the frame-survey feature values at
        # any quantile at all?
        circ_arr = circ_arrays.get(f"published|{quantity}|global")
        if circ_arr is not None:
            fired = circ_arr >= 0
            entry["circuit_array_assembled"] = {
                "tp": int((fired & cohort.risk).sum()),
                "fp": int((fired & ~cohort.risk).sum()),
            }
            match = [
                s["quantile"]
                for s in sweep
                if s["tp"] == entry["circuit_array_assembled"]["tp"]
                and s["fp"] == entry["circuit_array_assembled"]["fp"]
            ]
            entry["frame_survey_quantile_reproducing_circuit_counts"] = match
        # which development reference pool reproduces which bundle's table?
        dev_graph, dev_mob = graphs["development_main"], mobs["development_main"]
        dev_valid = dev_graph["valid"].astype(bool)
        dev_labels = pd.read_csv(
            PROJECT
            / "double-selete/trainfree/results/timeout_extension_plus10"
            / "development_main_clean_labels.csv"
        )
        dev_risk = dev_labels["original_failure"].to_numpy(bool)
        dev_oriented = oriented(reprs["development_main"][rep], direction)
        dev_persistent = dev.persistent_score(dev_oriented, CONFIRMATIONS)
        peak_main = dev.row_max(dev_oriented)
        peak_extra = dev.row_max(oriented(reprs["development_extra"][rep], direction))
        pools = {
            "development_main + development_extra (16,000)": np.concatenate(
                [peak_main, peak_extra]
            ),
            "development_main only (14,800)": peak_main,
        }
        pool_tables = {}
        for label, values in pools.items():
            table = []
            for quantile in dev.QUANTILES:
                line = dev.quantile_higher(values, quantile)
                f = dev.first_query(
                    np.isfinite(dev_persistent) & (dev_persistent > line) & dev_valid
                )
                fired = f >= 0
                table.append(
                    [quantile, int((fired & dev_risk).sum()), int((fired & ~dev_risk).sum())]
                )
            pool_tables[label] = table
        entry["development_table_by_reference_pool"] = pool_tables
        fs_dev = pd.read_csv(HB / "results/frame_survey/development_candidates.csv")
        c_dev = pd.read_csv(CIRCUIT / "results/detectors/development_candidates.csv")

        def published_table(frame):
            block = frame[
                (frame["quantity"] == quantity)
                & (frame["mode"] == "global")
                & (frame["representation"] == rep)
                & (frame["direction"] == direction)
            ].sort_values("quantile")
            return [
                [float(r["quantile"]), int(r["tp"]), int(r["fp"])]
                for _, r in block.iterrows()
            ]

        entry["published_development_table"] = {
            "moe-hb-front-back-0905": published_table(fs_dev),
            "moe-circuit-analogy-0906": published_table(c_dev),
        }
        entry["root_cause"] = {
            "frame_survey_pool": next(
                label
                for label, table in pool_tables.items()
                if table == entry["published_development_table"]["moe-hb-front-back-0905"]
            ),
            "circuit_pool": next(
                label
                for label, table in pool_tables.items()
                if table
                == entry["published_development_table"]["moe-circuit-analogy-0906"]
            ),
        }
        report["heads"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "external_quantile_sweep"}, indent=1), flush=True)

    # which of the 12 global heads agree across the two bundles at all?
    agreement = []
    for _, row in survey.iterrows():
        if row["mode"] != "global":
            continue
        q = row["quantity"]
        c = circ[(circ["quantity"] == q) & (circ["mode"] == "global")]
        if len(c) != 1:
            continue
        agreement.append(
            {
                "quantity": q,
                "frame_survey": f"{row['representation']}/{row['direction']}/q{float(row['quantile']):g}"
                f" -> {int(row['tp'])}/{int(row['fp'])}",
                "circuit_analogy": f"{c.iloc[0]['representation']}/{c.iloc[0]['direction']}"
                f"/q{float(c.iloc[0]['quantile']):g} -> {int(c.iloc[0]['tp'])}/{int(c.iloc[0]['fp'])}",
                "agree": bool(
                    int(row["tp"]) == int(c.iloc[0]["tp"])
                    and int(row["fp"]) == int(c.iloc[0]["fp"])
                ),
            }
        )
    report["all_twelve_global_heads"] = agreement
    n_agree = sum(a["agree"] for a in agreement)
    report["summary"] = (
        f"{n_agree} of {len(agreement)} global heads agree exactly between "
        "moe-hb-front-back-0905 and moe-circuit-analogy-0906"
    )
    print(report["summary"], flush=True)
    for a in agreement:
        if not a["agree"]:
            print("  DISAGREE", a, flush=True)

    (RESULTS / "bundle_disagreement.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
