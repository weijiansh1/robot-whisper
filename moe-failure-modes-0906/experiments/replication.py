#!/usr/bin/env python3
"""Q2: which (routing quantity -> physical mode) cells replicate.

A single-cohort BH-significant cell is a discovery, not a fact. The strong test
is a two-stage one that needs no extra multiplicity correction on the second
stage: discover on development, confirm on external.

    discovered   BH(q=0.05) significant on development under stratification S
    replicated   discovered, AND same sign on external, AND raw permutation
                 p < 0.05 on external under the same stratification S

Development is the selection cohort throughout this project, so this is the
same discipline used for thresholds, applied to the interpretation claim.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"


def main() -> None:
    table = pd.read_csv(OUT / "mode_recall.csv")
    dev = table[(table.cohort == "development_main") & (table.population == "all_risks")]
    ext = table[(table.cohort == "external_8b") & (table.population == "all_risks")]
    merged = dev.merge(ext, on=["detector", "mode", "mode_short"],
                       suffixes=("_dev", "_ext"), validate="one_to_one")

    rows = []
    for stratum in ("suite", "task"):
        for _, r in merged.iterrows():
            sel_d, sel_e = r[f"sel_{stratum}_dev"], r[f"sel_{stratum}_ext"]
            same_sign = bool(np.isfinite(sel_d) and np.isfinite(sel_e)
                             and np.sign(sel_d) == np.sign(sel_e))
            if not np.isfinite(sel_d) and not np.isfinite(sel_e):
                # both -inf: detector catches zero of this mode in both cohorts
                same_sign = bool(sel_d == sel_e == -np.inf)
            discovered = bool(r[f"bh_{stratum}_dev"])
            rows.append({
                "stratification": stratum,
                "detector": r["detector"],
                "mode_short": r["mode_short"],
                "n_dev": int(r["n_mode_dev"]),
                "n_ext": int(r["n_mode_ext"]),
                "recall_dev": r["recall_dev"],
                "recall_ext": r["recall_ext"],
                "expected_dev": r[f"expected_recall_{stratum}_dev"],
                "expected_ext": r[f"expected_recall_{stratum}_ext"],
                "sel_dev": sel_d,
                "sel_ext": sel_e,
                "p_dev": r[f"p_{stratum}_dev"],
                "p_ext": r[f"p_{stratum}_ext"],
                "discovered_on_development": discovered,
                "same_sign": same_sign,
                "replicated": bool(discovered and same_sign and r[f"p_{stratum}_ext"] < 0.05),
                "small_cell": bool(r["n_mode_dev"] < 30 or r["n_mode_ext"] < 30),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "replication.csv", index=False)

    summary = {"schema": "himoe.failure_modes_0906.replication.v1"}
    for stratum in ("suite", "task"):
        block = frame[frame.stratification == stratum]
        disc = block[block.discovered_on_development]
        summary[stratum] = {
            "cells": int(len(block)),
            "discovered_on_development": int(len(disc)),
            "same_sign_on_external": int(disc.same_sign.sum()),
            "replicated": int(disc.replicated.sum()),
            "replicated_non_small": int((disc.replicated & ~disc.small_cell).sum()),
            "expected_replications_if_null": float(0.05 * len(disc)),
            "replicated_list": [
                {"detector": r.detector, "mode": r.mode_short,
                 "n_dev": r.n_dev, "n_ext": r.n_ext,
                 "recall_dev": round(float(r.recall_dev), 3),
                 "expected_dev": round(float(r.expected_dev), 3),
                 "recall_ext": round(float(r.recall_ext), 3),
                 "expected_ext": round(float(r.expected_ext), 3),
                 "sel_dev": round(float(r.sel_dev), 3) if np.isfinite(r.sel_dev) else "-inf",
                 "sel_ext": round(float(r.sel_ext), 3) if np.isfinite(r.sel_ext) else "-inf",
                 "p_ext": float(r.p_ext), "small_cell": bool(r.small_cell)}
                for r in disc[disc.replicated].itertuples()
            ],
        }
    (OUT / "replication.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
