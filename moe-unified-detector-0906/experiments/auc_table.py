#!/usr/bin/env python3
"""Replicated within-task fixed-chunk AUC, on development only.

For every (suite, chunk q, column, normalisation):

    keep the episodes of that suite still running at q  (length > q)
    inside each task, Mann-Whitney AUC of the value against eventual risk
    pool the per-task AUCs with Mann-Whitney weights n+ * n-

Every episode inside a stratum shares a task *and* a chunk index, so neither
task identity nor elapsed time can contribute to the statistic.  This is the same
estimator as ``moe-early-window-0906/experiments/common.stratified_cell``; the
only addition is that the per-task AUCs are kept so that *replication across
tasks* can be measured, which is what the gating rule needs.

`raw` and `pop` are the same monotone order inside a fixed chunk, so their
within-task fixed-chunk AUC is identical by construction.  Only `raw` is
computed here; ``tests/test_protocol.py`` asserts the identity.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import protocol as P
import arms
from common import _rank_auc_columns, MIN_POS, MIN_NEG, MIN_STRATUM  # noqa: E402

MIN_TASKS = 4
REPLICATION_FLOOR = 0.75
EFFECT_FLOOR = 0.10

_G: dict = {}


def per_task_auc(
    x: np.ndarray, risk: np.ndarray, task: np.ndarray
) -> dict[str, np.ndarray]:
    """Pooled AUC, DeLong SE and cross-task replication for every column of x."""
    m = x.shape[1]
    num = np.zeros(m)
    den = np.zeros(m)
    var_num = np.zeros(m)
    n_tasks = np.zeros(m)
    n_same = np.zeros(m)
    eq_sum = np.zeros(m)
    task_auc: list[tuple[np.ndarray, np.ndarray]] = []
    for name in np.unique(task):
        take = task == name
        sub = x[take]
        r = risk[take]
        n1, n0 = int(r.sum()), int((~r).sum())
        if n1 < MIN_POS or n0 < MIN_NEG or (n1 + n0) < MIN_STRATUM:
            continue
        good = np.isfinite(sub).all(axis=0)
        if not good.any():
            continue
        auc, var = _rank_auc_columns(sub[:, good], r)
        w = float(n1 * n0)
        num[good] += auc * w
        den[good] += w
        var_num[good] += var * (w**2)
        eq_sum[good] += auc
        n_tasks[good] += 1
        task_auc.append((good, auc))
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        se = np.where(den > 0, np.sqrt(var_num) / np.maximum(den, 1e-12), np.nan)
        equal = np.where(n_tasks > 0, eq_sum / np.maximum(n_tasks, 1), np.nan)
    sign = np.sign(pooled - 0.5)
    for good, auc in task_auc:
        agree = np.sign(auc - 0.5) == sign[good]
        n_same[good] += agree
    with np.errstate(invalid="ignore", divide="ignore"):
        replication = np.where(n_tasks > 0, n_same / np.maximum(n_tasks, 1), np.nan)
    return {
        "auc": pooled,
        "se": se,
        "auc_equal": equal,
        "pairs": den,
        "n_tasks": n_tasks,
        "replication": replication,
    }


def _init(frame: dict, blocks: dict[str, np.ndarray], columns: list[str]) -> None:
    _G["frame"] = frame
    _G["blocks"] = blocks
    _G["columns"] = columns


def _cell(job: tuple[str, int]) -> pd.DataFrame:
    suite, chunk = job
    frame = _G["frame"]
    columns = _G["columns"]
    take = (frame["suite"] == suite) & (frame["length"] > chunk)
    risk = frame["risk"][take]
    task = frame["task"][take]
    rows = []
    for rep, block in _G["blocks"].items():
        got = per_task_auc(block[take, chunk, :], risk, task)
        rows.append(
            pd.DataFrame(
                {
                    "suite": suite,
                    "chunk": chunk,
                    "rep": rep,
                    "column": columns,
                    **got,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out["effect"] = out["auc"] - 0.5
    out["abs_effect"] = out["effect"].abs()
    out["n_alive"] = int(take.sum())
    out["n_alive_risk"] = int(risk.sum())
    out["prior"] = float(risk.mean())
    return out


def build(
    frame: dict,
    values: np.ndarray,
    columns: list[str],
    workers: int = 48,
) -> pd.DataFrame:
    blocks = {"raw": values, "self": P.apply_self(values)}
    jobs = [
        (suite, chunk)
        for suite in P.SUITES
        for chunk in range(P.FIRST_SCORED_CHUNK, P.IN_WINDOW_END[suite] + 1)
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(frame, blocks, columns)) as pool:
        parts = pool.map(_cell, jobs, chunksize=1)
    table = pd.concat(parts, ignore_index=True)
    meta = P.column_frame(columns).set_index("column")
    for col in ("quantity", "layer", "step", "base", "is_control", "is_portable"):
        table[col] = meta.loc[table["column"], col].to_numpy()
    table["replicated"] = (
        (table["n_tasks"] >= MIN_TASKS)
        & (table["replication"] >= REPLICATION_FLOOR)
        & (table["abs_effect"] >= EFFECT_FLOOR)
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="real", choices=arms.ARMS)
    parser.add_argument("--scope", default="extended", choices=("portable", "extended"))
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    frame = P.load_cohort(P.FIT_COHORT)
    meta = P.column_frame(frame["columns"])
    keep = meta["is_portable"] | meta["is_control"] if args.scope == "portable" else None
    columns = (
        meta.loc[keep, "column"].tolist() if keep is not None else list(frame["columns"])
    )
    values = arms.arm_values(frame, args.arm, columns, arms.arm_seed(args.arm, P.FIT_COHORT))
    table = build(frame, values, columns, workers=args.workers)

    out = P.RESULTS / "auc"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"development_{args.arm}_{args.scope}.csv.gz"
    table.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(table)}")

    ctrl = table[table["quantity"] == "ctrl_const_elapsed"]
    assert len(ctrl) > 0
    bad = ctrl[np.abs(ctrl["auc"] - 0.5) > 1e-12]
    print(f"ctrl_const_elapsed cells={len(ctrl)}  off-0.5={len(bad)}  "
          f"max|auc-0.5|={np.nanmax(np.abs(ctrl['auc'] - 0.5)):.3e}")


if __name__ == "__main__":
    main()
