#!/usr/bin/env python3
"""Does targeting the two modes separately beat one undifferentiated detector?

Three things are compared at a matched false-alarm count on external:

  single      one detector selected on development for *all* risks at budget B;
  union       the two mode-targeted detectors, each selected at budget B/2 and
              OR-ed (earliest alarm wins), so the pair costs at most B on
              development;
  published   what already exists - `v7_guard`, and the union of every
              published task-agnostic alarm vector, per mode, for context.

The union is the operationally relevant form of "target the modes separately":
a deployed monitor does not know which mode it is about to see, so it must run
both heads and act on whichever fires.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import arms as A
import modes_common as M
import capfree_protocol as CF
from select_and_score import (HEADLINE_BUDGET, load_table, pick,
                              baseline_frame, baseline_at, alarm_vector, union,
                              Q0_PROTOCOL)

# A denser budget grid than the selection table's, so the single arm and the
# union arm can be read off at a matched *external* false-alarm count rather
# than only at a matched development one.
UNION_BUDGETS = (10, 19, 28, 38, 47, 57, 76, 100, 150, 220)


def row_summary(first: np.ndarray, frame: dict, base: dict, label: str,
                extra: dict) -> dict:
    got = M.score_modes(first, frame)
    fp4 = got["fp_lead4"]
    out = {"detector": label, **extra,
           "n_drop": got["n_drop"], "n_grasp": got["n_grasp"],
           "n_risk": int(frame["risk"].sum()),
           "alarms": got["alarms"], "median_lead_at_lead4":
               got["median_lead_at_headline"]}
    for lead in M.LEADS:
        out[f"tp_lead{lead}"] = got[f"tp_lead{lead}"]
        out[f"fp_lead{lead}"] = got[f"fp_lead{lead}"]
        out[f"tp_drop_lead{lead}"] = got[f"tp_drop_lead{lead}"]
        out[f"tp_grasp_lead{lead}"] = got[f"tp_grasp_lead{lead}"]
        out[f"tp_other_lead{lead}"] = got[f"tp_other_lead{lead}"]
    for sweep, b in base.items():
        for key in ("tp_all", "tp_drop", "tp_grasp"):
            got_b = baseline_at(b, fp4, key)
            out[f"base_{sweep}_{key}"] = got_b
            k = key if key != "tp_all" else "tp"
            mine = out[f"{k}_lead4"] if key != "tp_all" else out["tp_lead4"]
            out[f"excess_{sweep}_{key}"] = mine - got_b
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=M.CACHE)
    p.add_argument("--out", type=Path, default=M.RESULTS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = {c: M.load(c) for c in ("development_main", "external_8b")}
    dev = load_table(args.cache / "development_main_ops.npz")
    extR = load_table(args.cache / "external_8b_ops_ratematched.npz")
    calib = pickle.loads((args.cache / "development_main_calibration.pkl").read_bytes())

    # Only the protocol sweep (q0 < 40) is a legitimate baseline: for
    # q0 >= cap - 4 the rule is identical to the risk label by construction.
    base = {c: {"protocol": baseline_frame(f, Q0_PROTOCOL)}
            for c, f in frames.items()}

    ops = dev["ops"]
    arm_of = ops["arm"].to_numpy()
    pool = A.pool_masks(ops)["headline"]
    all_dev = np.ones(len(dev["tasks"]), bool)

    rows = []
    for budget in UNION_BUDGETS:
        for arm in list(A.ARMS) + ["any"]:
            mask = pool if arm == "any" else (pool & (arm_of == arm))
            single = pick(dev, mask, all_dev, "all", budget)
            half = max(1, budget // 2)
            op_d = pick(dev, mask, all_dev, "drop", half)
            op_g = pick(dev, mask, all_dev, "grasp", half)
            # Control for "two heads are better than one" as such: the same
            # pair-of-heads construction, both heads selected for *all* risks,
            # forced onto different channels.  If the mode-targeted pair does
            # no better than this, the gain is ensembling, not mode targeting.
            first_all = pick(dev, mask, all_dev, "all", half)
            if first_all is None:
                continue
            r0 = ops.iloc[first_all]
            other = mask & ~((ops["quantity"] == r0["quantity"]).to_numpy()
                             & (ops["layer"] == r0["layer"]).to_numpy())
            second_all = pick(dev, other, all_dev, "all", half)
            if single is None or op_d is None or op_g is None or second_all is None:
                continue
            for cohort, table in (("development_main", dev), ("external_8b", extR)):
                f = frames[cohort]
                vec = {j: alarm_vector(f, calib, table["ops"].iloc[j])
                       for j in (single, op_d, op_g, first_all, second_all)}
                common = {"cohort": cohort, "budget_dev_fp_lead4": budget, "arm": arm}
                rows.append(row_summary(
                    vec[single], f, base[cohort], "single_undifferentiated",
                    {**common, "config": _cfg(table, single)}))
                rows.append(row_summary(
                    vec[op_d], f, base[cohort], "drop_head_only",
                    {**common, "config": _cfg(table, op_d)}))
                rows.append(row_summary(
                    vec[op_g], f, base[cohort], "grasp_head_only",
                    {**common, "config": _cfg(table, op_g)}))
                rows.append(row_summary(
                    union(vec[op_d], vec[op_g]), f, base[cohort],
                    "union_of_two_mode_heads",
                    {**common, "config": _cfg(table, op_d) + " OR " + _cfg(table, op_g)}))
                rows.append(row_summary(
                    union(vec[first_all], vec[second_all]), f, base[cohort],
                    "union_of_two_undifferentiated_heads",
                    {**common, "config": _cfg(table, first_all) + " OR "
                     + _cfg(table, second_all)}))

    # published detectors, for context
    for cohort in frames:
        f = frames[cohort]
        det = CF.load_detectors(cohort, len(f["risk"]))
        guard = next(k for k in det if "v7_guard|global" in k)
        rows.append(row_summary(det[guard], f, base[cohort], "v7_guard",
                                {"cohort": cohort, "budget_dev_fp_lead4": -1,
                                 "arm": "published", "config": guard}))
        rows.append(row_summary(union(*det.values()), f, base[cohort],
                                "union_of_all_published",
                                {"cohort": cohort, "budget_dev_fp_lead4": -1,
                                 "arm": "published",
                                 "config": f"{len(det)} task-agnostic vectors"}))

    frame = pd.DataFrame(rows)
    M.write_csv(args.out / "union_vs_single.csv", frame)

    show = frame[(frame.cohort == "external_8b")
                 & (frame.budget_dev_fp_lead4.isin((HEADLINE_BUDGET, -1)))]
    cols = ["detector", "arm", "tp_lead4", "fp_lead4", "tp_drop_lead4",
            "n_drop", "tp_grasp_lead4", "n_grasp", "excess_protocol_tp_all"]
    print("== external_8b, lead >= 4, development budget %d ==" % HEADLINE_BUDGET)
    print(show[cols].to_string(index=False))


def _cfg(table: dict, op: int) -> str:
    r = table["ops"].iloc[op]
    return (f"{r['quantity']}|{r['layer']}|sign{int(r['sign']):+d}|"
            f"{r['stat']}|k{int(r['grid_k'])}")


if __name__ == "__main__":
    main()
