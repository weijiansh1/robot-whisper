#!/usr/bin/env python3
"""The real evidence: within-task, fixed-chunk, survivors-only AUC, per mode.

Only episodes of the same task, at the same chunk, all still running, are ever
compared.  Task difficulty, horizon and elapsed time are constant by
construction, so a constant channel scores exactly 0.5 - which is asserted, not
assumed.  Positives are the risk episodes *of one physical mode*; negatives are
the non-risk survivors of the same task; risk episodes of any other mode are
dropped from the comparison so the two modes never score each other.

This is also the cleanest test of the hypothesis that does not depend on any
detector design.  If a drop is a change point and a never-formed grasp is a
persistent anomaly, then the AUC of the grasp mode should already be away from
0.5 at the earliest chunks, while the AUC of the drop mode should start near
0.5 and separate later.  That prediction is read straight off the per-chunk
curves.

The null is a white-noise sweep of the same size: the same number of columns,
over the same (suite, chunk) grid, drawn fresh at every (episode, chunk).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import modes_common as M   # noqa: F401  (puts moe-early-window-0906 on the path)
import common as EW        # noqa: E402

SEED = M.SEED
NULL_COLUMNS = 216   # matched to the routing column count

# A cell in which only one task carries the mode is not a within-task estimate
# of anything general: it is that one task's AUC.  For the grasp mode this is
# the rule rather than the exception - 307 of 355 long cases and 73 of 77
# spatial cases sit in a single task each - so the headline is restricted to
# cells with at least two contributing tasks and a substantive pair count, and
# the white-noise ceiling is recomputed on exactly the same restricted grid.
MIN_TASKS_HEADLINE = 2
MIN_PAIRS_HEADLINE = 2000


def sweep(frame: dict, columns: np.ndarray, names: list[str], layers: list[str],
          positives: np.ndarray, tag: str, cohort: str) -> pd.DataFrame:
    """AUC over (suite, chunk) for every column, positives given explicitly."""
    risk = frame["risk"]
    other_risk = risk & ~positives
    rows = []
    for suite in sorted(set(frame["suite"])):
        for q in EW.suite_chunks(suite):
            alive = (frame["suite"] == suite) & (frame["length"] > q) & ~other_risk
            take = np.flatnonzero(alive)
            if take.size < EW.MIN_STRATUM:
                continue
            pos = positives[take]
            if pos.sum() < EW.MIN_POS:
                continue
            cell = EW.stratified_cell(columns[take, q, :], pos, frame["task"][take])
            lo, hi = EW.logit_ci(cell["auc"], cell["se"])
            good = np.isfinite(cell["auc"])
            for j in np.flatnonzero(good):
                rows.append({
                    "cohort": cohort, "target": tag, "suite": suite, "chunk": q,
                    "quantity": names[j], "layer": layers[j],
                    "auc": float(cell["auc"][j]), "se": float(cell["se"][j]),
                    "ci_lo": float(lo[j]), "ci_hi": float(hi[j]),
                    "effect": float(cell["auc"][j] - 0.5),
                    "abs_effect": float(abs(cell["auc"][j] - 0.5)),
                    "n_tasks": int(cell["n_tasks"][j]),
                    "pairs": float(cell["pairs"][j]),
                    "n_alive": int(take.size),
                    "n_pos": int(pos.sum()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=M.RESULTS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows, null_rows = [], []
    for cohort in ("development_main", "external_8b"):
        frame = M.load(cohort)
        names, layers = EW.column_index(frame["quantities"])
        block = EW.flatten(frame["values"])
        keep = [j for j, q in enumerate(names)
                if q in frame["routing_quantities"]]
        cols = np.ascontiguousarray(block[:, :, keep])
        cnames = [names[j] for j in keep]
        clayers = [layers[j] for j in keep]

        risk, mode = frame["risk"], frame["mode"]
        targets = {
            "all_risk": risk,
            "drop": risk & (mode == M.MODE_DROP),
            "grasp": risk & (mode == M.MODE_GRASP),
        }
        for tag, pos in targets.items():
            all_rows.append(sweep(frame, cols, cnames, clayers, pos, tag, cohort))

        # constant column must be exactly 0.5, in this exact code path
        const = np.zeros((len(risk), frame["n_chunk"], 1))
        chk = sweep(frame, const, ["ctrl_zero"], ["L2"], risk, "all_risk", cohort)
        assert (chk["auc"] == 0.5).all(), chk[chk.auc != 0.5].head()
        assert (chk["se"] == 0.0).all()

        rng = np.random.default_rng(SEED + (0 if cohort == "development_main" else 1))
        noise = rng.standard_normal(
            (len(risk), frame["n_chunk"], NULL_COLUMNS), dtype=np.float32)
        noise[~frame["valid"]] = np.nan
        nn = [f"white_{i}" for i in range(NULL_COLUMNS)]
        nl = ["L0"] * NULL_COLUMNS
        for tag, pos in targets.items():
            null_rows.append(sweep(frame, noise, nn, nl, pos, tag, cohort))

    auc = pd.concat(all_rows, ignore_index=True)
    nul = pd.concat(null_rows, ignore_index=True)
    for f in (auc, nul):
        f["headline_cell"] = ((f.n_tasks >= MIN_TASKS_HEADLINE)
                              & (f.pairs >= MIN_PAIRS_HEADLINE))
    auc.to_csv(args.out / "within_task_auc.csv.gz", index=False)
    nul.to_csv(args.out / "within_task_auc_null.csv.gz", index=False)
    ceiling = pd.concat([
        (nul.groupby(["cohort", "target"]).abs_effect.max()
         .rename("null_ceiling").reset_index().assign(grid="all_cells")),
        (nul[nul.headline_cell].groupby(["cohort", "target"]).abs_effect.max()
         .rename("null_ceiling").reset_index().assign(grid="headline_cells")),
    ], ignore_index=True)
    ceiling["n_null_cells"] = ceiling.apply(
        lambda r: int(((nul.cohort == r.cohort) & (nul.target == r.target)
                       & (nul.headline_cell if r.grid == "headline_cells"
                          else True)).sum()), axis=1)
    M.write_csv(args.out / "within_task_null_ceiling.csv", ceiling)

    head_ceiling = (ceiling[ceiling.grid == "headline_cells"]
                    [["cohort", "target", "null_ceiling"]])
    merged = auc.merge(head_ceiling, on=["cohort", "target"])
    merged["above_null"] = merged.abs_effect > merged.null_ceiling

    # replication across cohorts, on the same (target, suite, chunk, channel)
    key = ["target", "suite", "chunk", "quantity", "layer"]
    d = merged[merged.cohort == "development_main"].set_index(key)
    e = merged[merged.cohort == "external_8b"].set_index(key)
    both = d.join(e, lsuffix="_d", rsuffix="_e", how="inner").reset_index()
    both["same_sign"] = np.sign(both.effect_d) == np.sign(both.effect_e)
    both["headline_cell"] = both.headline_cell_d & both.headline_cell_e
    both["replicated"] = both.same_sign & both.above_null_d & both.above_null_e
    both["min_abs"] = np.minimum(both.abs_effect_d, both.abs_effect_e)
    both.sort_values("min_abs", ascending=False).to_csv(
        args.out / "within_task_replicated.csv.gz", index=False)

    # separation as a function of chunk - the detector-free form of the hypothesis
    early = []
    for (cohort, target, suite), grp in merged.groupby(["cohort", "target", "suite"]):
        for q, sub in grp.groupby("chunk"):
            head = sub[sub.headline_cell]
            early.append({"cohort": cohort, "target": target, "suite": suite,
                          "chunk": int(q),
                          "max_abs_effect": float(sub.abs_effect.max()),
                          "max_abs_effect_headline":
                              float(head.abs_effect.max()) if len(head) else np.nan,
                          "n_above_null": int(sub.above_null.sum()),
                          "n_above_null_headline": int(head.above_null.sum()),
                          "n_cells": int(len(sub)),
                          "n_headline_cells": int(len(head)),
                          "null_ceiling": float(sub.null_ceiling.iloc[0]),
                          "n_pos": int(sub.n_pos.iloc[0]),
                          "n_tasks_max": int(sub.n_tasks.max())})
    early = pd.DataFrame(early).sort_values(["cohort", "target", "suite", "chunk"])
    M.write_csv(args.out / "within_task_by_chunk.csv", early)

    print("== null ceiling (|AUC-0.5|, white noise, same sweep size) ==")
    print(ceiling.to_string(index=False))
    print("\n== cells by number of contributing tasks ==")
    print(pd.crosstab([auc.cohort, auc.target], auc.n_tasks).to_string())
    print("\n== replicated cells above the null in both cohorts, by target ==")
    print(both.groupby(["target", "headline_cell"])
          .replicated.agg(["sum", "size"]).to_string())
    print("\n== strongest replicated cell per (target, suite), headline cells only ==")
    top = (both[both.replicated & both.headline_cell]
           .sort_values("min_abs", ascending=False)
           .groupby(["target", "suite"]).head(2))
    print(top[["target", "suite", "chunk", "quantity", "layer", "auc_d", "auc_e",
               "n_tasks_d", "n_pos_d", "n_pos_e"]].to_string(index=False)
          if len(top) else "   none")
    print("\n== separation by chunk (external_8b, headline cells) ==")
    ex = early[early.cohort == "external_8b"]
    for suite in sorted(set(ex.suite)):
        sub = ex[ex.suite == suite]
        print(f"-- {suite}")
        piv = sub.pivot_table(index="chunk", columns="target",
                              values="max_abs_effect_headline")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
