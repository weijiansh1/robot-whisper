#!/usr/bin/env python3
"""Reproduce every published anchor before anything new is reported.

  v7 whole corpus        1078 TP / 172 FP over 1358 risks
  v7 sealed external     439 TP / 80 FP;  development 382 TP / 67 FP
  v7 in-window           472 TP / 127 FP; long 424/712, goal 39/250,
                         object 9/81, spatial 0/315

The in-window anchor pins down the window definition.  ``(q+1)/cap <= 0.65``
gives 390/122 and long 342, not 424; ``q/cap <= 0.65`` gives 482/130 and object
15, not 9.  The published constants are ``q <= round(0.65*cap - 1)`` under
banker's rounding = {goal 18, long 33, object 17, spatial 13}, which reproduces
all six numbers exactly.  That window is adopted for the whole bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as P

V7_SEALED = P.PROJECT / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
V7_LEGACY = (
    P.PROJECT
    / "moe-v7-legacy16x32-0906/results/legacy16x32/legacy16x32_first_alarms.npz"
)
LEGACY_ALARMS = P.PROJECT / "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv"

TARGET = {
    "corpus_tp": 1078,
    "corpus_fp": 172,
    "corpus_risks": 1358,
    "dev_tp": 382,
    "dev_fp": 67,
    "ext_tp": 439,
    "ext_fp": 80,
    "iw_tp": 472,
    "iw_fp": 127,
    "iw_libero_long": 424,
    "iw_libero_goal": 39,
    "iw_libero_object": 9,
    "iw_libero_spatial": 0,
    "risk_libero_long": 712,
    "risk_libero_goal": 250,
    "risk_libero_object": 81,
    "risk_libero_spatial": 315,
}


def v7_first_alarms() -> dict[str, np.ndarray]:
    sealed = np.load(V7_SEALED, allow_pickle=False)
    legacy = np.load(V7_LEGACY, allow_pickle=False)
    return {
        "development_main": sealed["main_guard"].astype(np.int64),
        "external_8b": sealed["external_guard"].astype(np.int64),
        "legacy_main16x32": legacy["legacy_guard"].astype(np.int64),
    }


def check(frames: dict[str, dict]) -> dict:
    first = v7_first_alarms()
    got: dict[str, int] = {}
    tot = {"tp": 0, "fp": 0, "iw_tp": 0, "iw_fp": 0, "n_risk": 0}
    per_suite = {s: [0, 0] for s in P.SUITES}
    rows = []
    for cohort, frame in frames.items():
        assert len(first[cohort]) == len(frame["risk"]), cohort
        got_c = P.evaluate(first[cohort], frame)
        rows.append({"cohort": cohort, **got_c})
        for key in tot:
            tot[key] += got_c[key]
        for s in P.SUITES:
            per_suite[s][0] += got_c[f"iw_tp_{s}"]
            per_suite[s][1] += got_c[f"n_risk_{s}"]
    got.update(
        corpus_tp=tot["tp"], corpus_fp=tot["fp"], corpus_risks=tot["n_risk"],
        iw_tp=tot["iw_tp"], iw_fp=tot["iw_fp"],
        dev_tp=rows[0]["tp"], dev_fp=rows[0]["fp"],
        ext_tp=[r for r in rows if r["cohort"] == "external_8b"][0]["tp"],
        ext_fp=[r for r in rows if r["cohort"] == "external_8b"][0]["fp"],
    )
    for s in P.SUITES:
        got[f"iw_{s}"] = per_suite[s][0]
        got[f"risk_{s}"] = per_suite[s][1]
    mismatch = {k: (TARGET[k], got[k]) for k in TARGET if TARGET[k] != got[k]}
    return {
        "target": TARGET,
        "measured": got,
        "mismatch": mismatch,
        "reproduced": not mismatch,
        "per_cohort": rows,
        "window_end": P.IN_WINDOW_END,
        "window_end_strict_phase": P.IN_WINDOW_END_STRICT,
    }


def window_variants(frames: dict[str, dict]) -> pd.DataFrame:
    """What each candidate in-window definition would have given for v7."""
    first = v7_first_alarms()
    rows = []
    variants = {
        "published_anchor  q<=round(0.65*cap-1)": P.IN_WINDOW_END,
        "strict  (q+1)/cap<=0.65": P.IN_WINDOW_END_STRICT,
        "loose   q/cap<=0.65": {s: int(np.floor(0.65 * c)) for s, c in P.CAPS.items()},
    }
    for label, ends in variants.items():
        tp = fp = 0
        suite_tp = {s: 0 for s in P.SUITES}
        for cohort, frame in frames.items():
            end = np.asarray([ends[s] for s in frame["suite"]])
            f = first[cohort]
            inw = (f >= 0) & (f <= end)
            tp += int((inw & frame["risk"]).sum())
            fp += int((inw & ~frame["risk"]).sum())
            for s in P.SUITES:
                suite_tp[s] += int((inw & frame["risk"] & (frame["suite"] == s)).sum())
        rows.append({"variant": label, "ends": str(ends), "iw_tp": tp, "iw_fp": fp,
                     **{f"iw_{s}": v for s, v in suite_tp.items()}})
    return pd.DataFrame(rows)
