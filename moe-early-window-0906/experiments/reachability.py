#!/usr/bin/env python3
"""What in-window recall the measured information actually supports.

The information map says how well a quantity separates eventual risk inside a
task at a fixed chunk.  It does not say what a detector gets, because the
operating point is brutal: a timely FPR of 0.005 over ~15 000 episodes is ~75
false alarms in total.  This script converts information into recall under that
budget.

Alarm rule, deliberately the most sensitive one that is still deployable:

  * score = +/- the quantity value at chunk q (direction is part of the sweep)
  * threshold = the (1 - alpha) quantile of the score over *all* episodes of the
    same task still running at q.  This reads no outcome, so it could be
    shipped; at a 1-3% risk rate it is within noise of the non-risk quantile.
  * fire on score >= threshold.  `>=` and not `>`: several of these channels are
    discrete, and a strict test silently drops the entire top tie group.  The
    length leak is the check -- under `>` it scored 0.34 recall, under `>=` it
    scores what it must, ~1.0.
  * timely FPR is measured, never assumed, so ties inflating the alarm count are
    paid for in the FPR column.

Three arms:

  single_chunk    fire at one chunk only.  The right arm when the information is
                  chunk-localised, because the whole FPR budget goes to one test.
  window_union    fire at any chunk in the window.  More chances, but the budget
                  is split across them.
  multivariate    every quantity at once: a within-task-standardised L2 logistic
                  per chunk, cross-fitted by init_state_id so no episode is
                  scored by a model that saw its own initial state.

and three readings of each, which must be read together:

  upper_bound   best cell chosen on the same cohort it is scored on.  Not an
                estimate -- if this cannot reach the target, nothing can.
  transfer      cell chosen on development_main, replayed on external_8b with
                external's own task quantiles.  The deployable number.
  control       the identical sweep on the negative-control channels.  Whatever
                it reaches is what the sweep buys from noise alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import warnings

import common as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALPHAS = np.array(
    [
        0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008,
        0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10,
        0.15, 0.20, 0.30,
    ]
)
FPR_BUDGET = 0.005
BASELINE_COSTS = (0, 2, 4, 6, 8)
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def task_thresholds(x: np.ndarray, task_code: np.ndarray, n_task: int) -> np.ndarray:
    """(n_alpha, n_task, n_col) per-task (1-alpha) quantiles."""
    out = np.full((len(ALPHAS), n_task, x.shape[1]), np.inf, np.float32)
    for t in range(n_task):
        m = task_code == t
        if m.any():
            out[:, t, :] = np.quantile(x[m], 1.0 - ALPHAS, axis=0).astype(np.float32)
    return out


def chunk_crossings(
    block: np.ndarray, rows: np.ndarray, q: int, task_code: np.ndarray, n_task: int
) -> np.ndarray:
    """(n_alpha, 2, n_row, n_col) bool crossings at one chunk for both directions."""
    x0 = block[rows, q, :]
    finite = np.isfinite(x0).all(axis=0)
    out = np.zeros((len(ALPHAS), 2, len(rows), x0.shape[1]), bool)
    for d, sign in enumerate((1.0, -1.0)):
        x = np.where(finite[None, :], sign * x0, -np.inf).astype(np.float32)
        thr = task_thresholds(x, task_code, n_task)
        out[:, d] = x[None, :, :] >= thr[:, task_code, :]
        out[:, d] &= finite[None, None, :]
    return out


def tally(fired: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fired is (..., n_episode, n_col) -> tp, fp with the episode axis summed."""
    return fired[..., risk, :].sum(axis=-2), fired[..., ~risk, :].sum(axis=-2)


def frame_rows(tp, fp, n_pos, n_neg, names, layers, **extra) -> pd.DataFrame:
    n_alpha, n_dir, n_col = tp.shape
    a = np.repeat(ALPHAS, n_dir * n_col)
    d = np.tile(np.repeat(np.array(["high", "low"]), n_col), n_alpha)
    table = pd.DataFrame(
        {
            "alpha": a,
            "direction": d,
            "quantity": np.tile(names, n_alpha * n_dir),
            "layer": np.tile(layers, n_alpha * n_dir),
            "tp": tp.ravel(),
            "fp": fp.ravel(),
        }
    )
    table["recall"] = table["tp"] / max(n_pos, 1)
    table["fpr"] = table["fp"] / max(n_neg, 1)
    for key, value in extra.items():
        table[key] = value
    return table


def best_under_budget(table: pd.DataFrame, budget: float = FPR_BUDGET):
    ok = table[table["fpr"] <= budget]
    if not len(ok):
        return None
    return ok.sort_values(["recall", "fpr"], ascending=[False, True]).iloc[0]


# --------------------------------------------------------------------------- #


def standardise_within_task(x: np.ndarray, task_code: np.ndarray, n_task: int) -> np.ndarray:
    out = np.array(x, np.float64, copy=True)
    for t in range(n_task):
        m = task_code == t
        if m.sum() < 2:
            out[m] = 0.0
            continue
        mu, sd = out[m].mean(axis=0), out[m].std(axis=0)
        out[m] = (out[m] - mu) / np.where(sd > 1e-12, sd, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def crossfit_scores(x, risk, groups, seed):
    out = np.full(len(risk), np.nan)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    if n_splits < 2 or risk.sum() < n_splits:
        return out, None
    full = LogisticRegression(penalty="l2", C=0.02, max_iter=4000, random_state=seed)
    full.fit(x, risk)
    for train, test in GroupKFold(n_splits=n_splits).split(x, risk, groups):
        if risk[train].sum() < 2 or (~risk[train]).sum() < 2:
            continue
        model = LogisticRegression(penalty="l2", C=0.02, max_iter=4000, random_state=seed)
        model.fit(x[train], risk[train])
        out[test] = model.decision_function(x[test])
    return out, full


def score_to_alarms(score, task_code, n_task):
    ok = np.isfinite(score)
    s = np.where(ok, score, -np.inf).astype(np.float32)[:, None]
    thr = task_thresholds(s, task_code, n_task)
    return (s[None, :, 0] >= thr[:, task_code, 0]) & ok[None, :]


# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = {c: C.build_cohort(c) for c in ("development_main", "external_8b")}
    blocks = {c: C.flatten(f["values"]) for c, f in frames.items()}
    quantities = frames["development_main"]["quantities"]
    names, layers = C.column_index(quantities)
    names, layers = np.asarray(names), np.asarray(layers)
    is_control = np.isin(names, list(C.CONTROL_QUANTITIES))
    keep_real = np.flatnonzero(~is_control)
    keep_ctrl = np.flatnonzero(is_control)

    single: list[pd.DataFrame] = []
    union: list[pd.DataFrame] = []
    multi: list[dict] = []

    for suite in sorted(C.CAPS):
        q_max = C.WINDOW_65[suite]
        chunks = list(range(C.SWEEP_START, q_max + 1))
        for cohort, frame in frames.items():
            rows_suite = np.flatnonzero(frame["suite"] == suite)
            length = frame["length"][rows_suite]
            _, task_code_all = np.unique(frame["task"][rows_suite], return_inverse=True)
            n_task = int(task_code_all.max()) + 1
            risk = frame["risk"][rows_suite]
            n_risk, n_safe = int(risk.sum()), int((~risk).sum())
            block = blocks[cohort][rows_suite]

            # single chunk + running unions from every allowed start
            ever = {
                b: np.zeros((len(ALPHAS), 2, len(rows_suite), block.shape[2]), bool)
                for b in BASELINE_COSTS
            }
            for q in chunks:
                alive = length > q
                if alive.sum() < C.MIN_STRATUM:
                    continue
                rows = np.flatnonzero(alive)
                cross = chunk_crossings(block, rows, q, task_code_all[rows], n_task)
                tp, fp = tally(cross, risk[rows])
                single.append(
                    frame_rows(
                        tp, fp, n_risk, n_safe, names, layers,
                        cohort=cohort, suite=suite, chunk=q,
                        phase=(q + 1) / C.CAPS[suite],
                        n_risk=n_risk, n_safe=n_safe, n_alive=int(alive.sum()),
                    )
                )
                for b in BASELINE_COSTS:
                    if q >= max(C.SWEEP_START, b):
                        ever[b][:, :, rows, :] |= cross
            for b in BASELINE_COSTS:
                tp, fp = tally(ever[b], risk)
                start = max(C.SWEEP_START, b)
                union.append(
                    frame_rows(
                        tp, fp, n_risk, n_safe, names, layers,
                        cohort=cohort, suite=suite, baseline_chunks=b,
                        window=f"{start}-{q_max}", n_chunks=q_max - start + 1,
                        n_risk=n_risk, n_safe=n_safe,
                    )
                )
            del ever

        # ---- multivariate
        dev_models: dict[int, LogisticRegression] = {}
        for cohort in ("development_main", "external_8b"):
            frame = frames[cohort]
            rows_suite = np.flatnonzero(frame["suite"] == suite)
            length = frame["length"][rows_suite]
            _, task_code_all = np.unique(frame["task"][rows_suite], return_inverse=True)
            n_task = int(task_code_all.max()) + 1
            risk = frame["risk"][rows_suite]
            groups = frame["init_state_id"][rows_suite]
            block = blocks[cohort][rows_suite]
            n_risk, n_safe = int(risk.sum()), int((~risk).sum())
            unions = {
                arm: np.zeros((len(ALPHAS), len(rows_suite)), bool)
                for arm in ("crossfit", "control", "transfer")
            }
            for q in chunks:
                alive = length > q
                if alive.sum() < C.MIN_STRATUM or risk[alive].sum() < 5:
                    continue
                rows = np.flatnonzero(alive)
                tcode = task_code_all[rows]
                for arm, keep in (("crossfit", keep_real), ("control", keep_ctrl)):
                    x = standardise_within_task(
                        block[rows, q, :][:, keep], tcode, n_task
                    )
                    score, full = crossfit_scores(x, risk[rows], groups[rows], C.SEED + q)
                    if arm == "crossfit" and cohort == "development_main" and full is not None:
                        dev_models[q] = full
                    fired = score_to_alarms(score, tcode, n_task)
                    unions[arm][:, rows] |= fired
                    tp = fired[:, risk[rows]].sum(axis=1)
                    fp = fired[:, ~risk[rows]].sum(axis=1)
                    for a, alpha in enumerate(ALPHAS):
                        multi.append(
                            dict(
                                suite=suite, cohort=cohort, arm=arm, scope="single_chunk",
                                chunk=q, phase=(q + 1) / C.CAPS[suite], alpha=float(alpha),
                                tp=int(tp[a]), fp=int(fp[a]),
                                recall=tp[a] / max(n_risk, 1), fpr=fp[a] / max(n_safe, 1),
                                n_risk=n_risk, n_safe=n_safe,
                            )
                        )
                if cohort == "external_8b" and q in dev_models:
                    x = standardise_within_task(
                        block[rows, q, :][:, keep_real], tcode, n_task
                    )
                    fired = score_to_alarms(dev_models[q].decision_function(x), tcode, n_task)
                    unions["transfer"][:, rows] |= fired
                    tp = fired[:, risk[rows]].sum(axis=1)
                    fp = fired[:, ~risk[rows]].sum(axis=1)
                    for a, alpha in enumerate(ALPHAS):
                        multi.append(
                            dict(
                                suite=suite, cohort=cohort, arm="dev_fitted_transfer",
                                scope="single_chunk", chunk=q,
                                phase=(q + 1) / C.CAPS[suite], alpha=float(alpha),
                                tp=int(tp[a]), fp=int(fp[a]),
                                recall=tp[a] / max(n_risk, 1), fpr=fp[a] / max(n_safe, 1),
                                n_risk=n_risk, n_safe=n_safe,
                            )
                        )
            for arm, fired in unions.items():
                if not fired.any():
                    continue
                label = "dev_fitted_transfer" if arm == "transfer" else arm
                if arm == "transfer" and cohort != "external_8b":
                    continue
                tp = fired[:, risk].sum(axis=1)
                fp = fired[:, ~risk].sum(axis=1)
                for a, alpha in enumerate(ALPHAS):
                    multi.append(
                        dict(
                            suite=suite, cohort=cohort, arm=label, scope="window_union",
                            chunk=-1, phase=(q_max + 1) / C.CAPS[suite], alpha=float(alpha),
                            tp=int(tp[a]), fp=int(fp[a]),
                            recall=tp[a] / max(n_risk, 1), fpr=fp[a] / max(n_safe, 1),
                            n_risk=n_risk, n_safe=n_safe,
                        )
                    )
        print(f"{suite}: done", flush=True)

    single_table = pd.concat(single, ignore_index=True)
    union_table = pd.concat(union, ignore_index=True)
    multi_table = pd.DataFrame(multi)
    single_table["is_control"] = single_table["quantity"].isin(C.CONTROL_QUANTITIES)
    union_table["is_control"] = union_table["quantity"].isin(C.CONTROL_QUANTITIES)

    # keep the files small: only rows that are at all competitive
    single_table[single_table["fpr"] <= 0.05].to_csv(
        args.output / "reachability_single_chunk.csv.gz", index=False, float_format="%.6g"
    )
    union_table[union_table["fpr"] <= 0.05].to_csv(
        args.output / "reachability_window_union.csv.gz", index=False, float_format="%.6g"
    )
    multi_table.to_csv(args.output / "reachability_multivariate.csv", index=False)
    print("single", single_table.shape, "union", union_table.shape, "multi", multi_table.shape)

    # sanity: the definitional length leak must be recovered
    leak = single_table[
        (single_table["quantity"] == "leak_full_length")
        & (single_table["direction"] == "high")
        & (single_table["fpr"] <= FPR_BUDGET)
    ]
    check = leak.groupby("suite")["recall"].max().to_dict()
    C.write_json(
        args.output / "reachability_checks.json",
        {
            "length_leak_recall_at_fpr_0.005": check,
            "note": (
                "risk is defined as reaching the horizon cap, so a threshold on total "
                "episode length must recover essentially every risk at negligible FPR. "
                "It is a definitional leak, not a detector: the value is unknown until "
                "the episode ends."
            ),
        },
    )
    print("length-leak recall check:", check)


if __name__ == "__main__":
    main()
