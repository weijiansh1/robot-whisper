#!/usr/bin/env python3
"""The information map: fixed-chunk, task-stratified, survival-conditioned AUC.

One cell = (cohort, representation, suite, chunk q, quantity, layer).  Its value
is the task-stratified AUC for eventual risk among the episodes still running at
q, pooled over tasks with Mann-Whitney weights.  Every episode inside a stratum
shares a task and an elapsed time, so neither can contribute.

Reported per cell, never a count of p < 0.05:

  auc, se, ci_lo, ci_hi          pair-weighted pool, DeLong variance, logit CI
  auc_equal, se_equal            task-equal pool
  n_tasks, pairs                 how much of the suite actually contributes
  effect = auc - 0.5             signed, so direction is visible
  n_alive, n_alive_risk, prior   the survival stratum itself

Estimator checks asserted at run time:

  * ctrl_const_elapsed is constant inside every stratum, so it must score
    exactly 0.5 with variance exactly 0.  Elapsed length is therefore verified
    to contribute nothing under this estimator.
  * leak_full_length is the definitional leak (risk <=> length == cap) and must
    score close to 1.0.  It is not a baseline, it is the thing the estimator has
    to be blind to; it is reported to show what a leaking channel looks like.

Multiplicity is handled by permutation, not by a significance count.  A
replicate permutes the risk labels inside every (task, alive-at-q) stratum, which
preserves the task composition and the survival prior exactly, and the maximum
|auc - 0.5| over the whole sweep is recorded.  Permuting independently per chunk
destroys the across-chunk dependence, which can only inflate the maximum, so the
resulting critical value is conservative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common as C


COHORTS = ("development_main", "external_8b", "legacy_main16x32")
N_PERM = 1000
MIN_TASKS = 4      # admissibility of a cell, declared before the null was drawn
MIN_PAIRS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    parser.add_argument("--permutations", type=int, default=N_PERM)
    return parser.parse_args()


def representations(frame: dict) -> dict[str, np.ndarray]:
    raw = frame["values"]
    return {
        "raw": C.flatten(raw),
        f"tm{C.WIDTH}": C.flatten(C.trailing_mean(raw, C.WIDTH)),
    }


def perm_null_cell(
    x: np.ndarray, risk: np.ndarray, task: np.ndarray, rng: np.random.Generator, n_perm: int
) -> np.ndarray:
    """(n_perm, n_col) pooled AUC under within-(task, alive) label permutation.

    Ranks do not depend on the labels, so each replicate is a single matmul of a
    0/1 membership matrix against the per-task rank matrix.
    """
    m = x.shape[1]
    num = np.zeros((n_perm, m), np.float64)
    den = np.zeros(m, np.float64)
    for name in np.unique(task):
        take = task == name
        sub = x[take]
        r = risk[take]
        n1, n0 = int(r.sum()), int((~r).sum())
        n = n1 + n0
        if n1 < C.MIN_POS or n0 < C.MIN_NEG or n < C.MIN_STRATUM:
            continue
        good = np.isfinite(sub).all(axis=0)
        if not good.any():
            continue
        ranks = rankdata(sub[:, good], axis=0).astype(np.float32)
        pick = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
        sel = np.zeros((n_perm, n), np.float32)
        np.put_along_axis(sel, pick, 1.0, axis=1)
        rank_sum = sel @ ranks  # (n_perm, n_good)
        auc = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        w = float(n1 * n0)
        num[:, good] += auc * w
        den[good] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def run_cohort(frame: dict, n_perm: int) -> tuple[list[dict], list[dict], list[dict]]:
    quantities = frame["quantities"]
    col_quantity, col_layer = C.column_index(quantities)
    col_quantity = np.asarray(col_quantity)
    col_layer = np.asarray(col_layer)
    reps = representations(frame)
    risk_all = frame["risk"]
    task_all = frame["task"]
    suite_all = frame["suite"]
    length = frame["length"]
    rng = np.random.default_rng(C.SEED)

    rows: list[dict] = []
    null_rows: list[dict] = []
    checks: list[dict] = []
    ci_elapsed = quantities.index("ctrl_const_elapsed") * 8
    ci_leak = quantities.index("leak_full_length") * 8

    for suite in sorted(set(suite_all)):
        cap = C.CAPS[suite]
        chunks = C.suite_chunks(suite)
        suite_mask = suite_all == suite
        for rep_name, block in reps.items():
            null_max = np.zeros(n_perm) if rep_name == "raw" else None
            null_se_probe: list[dict] = []
            for q in chunks:
                if q >= frame["n_chunk"]:
                    continue
                alive = suite_mask & (length > q)
                if alive.sum() < C.MIN_STRATUM:
                    continue
                x = block[alive, q, :]
                r = risk_all[alive]
                t = task_all[alive]
                cell = C.stratified_cell(x, r, t)
                lo, hi = C.logit_ci(cell["auc"], cell["se"])
                lo_e, hi_e = C.logit_ci(cell["auc_equal"], cell["se_equal"])

                # estimator checks
                a_el = cell["auc"][ci_elapsed]
                s_el = cell["se"][ci_elapsed]
                if np.isfinite(a_el):
                    assert abs(a_el - 0.5) < 1e-12, (
                        f"constant-within-stratum control is not 0.5: {a_el}"
                    )
                    assert s_el == 0.0, f"constant control has non-zero variance: {s_el}"
                checks.append(
                    {
                        "cohort": frame["cohort"],
                        "representation": rep_name,
                        "suite": suite,
                        "chunk": q,
                        "auc_const_elapsed": float(a_el),
                        "se_const_elapsed": float(s_el),
                        "auc_full_length_leak": float(cell["auc"][ci_leak]),
                    }
                )

                prior = float(r.mean())
                for j in range(x.shape[1]):
                    if not np.isfinite(cell["auc"][j]):
                        continue
                    rows.append(
                        {
                            "cohort": frame["cohort"],
                            "representation": rep_name,
                            "suite": suite,
                            "cap": cap,
                            "chunk": q,
                            "phase": (q + 1) / cap,
                            "quantity": col_quantity[j],
                            "layer": col_layer[j],
                            "auc": cell["auc"][j],
                            "se": cell["se"][j],
                            "ci_lo": lo[j],
                            "ci_hi": hi[j],
                            "effect": cell["auc"][j] - 0.5,
                            "auc_equal": cell["auc_equal"][j],
                            "se_equal": cell["se_equal"][j],
                            "ci_equal_lo": lo_e[j],
                            "ci_equal_hi": hi_e[j],
                            "n_tasks": int(cell["n_tasks"][j]),
                            "pairs": cell["pairs"][j],
                            "n_alive": int(alive.sum()),
                            "n_alive_risk": int(r.sum()),
                            "prior": prior,
                            "in_window_65": q <= C.WINDOW_65[suite],
                            "in_window_cross": q <= C.WINDOW_CROSS[suite],
                        }
                    )

                if null_max is not None and np.isfinite(cell["auc"]).any():
                    null = perm_null_cell(x, r, t, rng, n_perm)
                    # A cell only enters the family-wise null if it is admissible:
                    # at least MIN_TASKS contributing tasks and MIN_PAIRS comparable
                    # pairs.  Late-window strata where one task with three survivors
                    # carries the whole pooled weight are excluded from both the
                    # null and every headline, because their permutation
                    # distribution reaches 0 and 1 by construction.
                    usable = (
                        np.isfinite(null).all(axis=0)
                        & (cell["n_tasks"] >= MIN_TASKS)
                        & (cell["pairs"] >= MIN_PAIRS)
                    )
                    if usable.any():
                        dev = np.max(np.abs(null[:, usable] - 0.5), axis=1)
                        null_max = np.maximum(null_max, dev)
                        null_se_probe.append(
                            {
                                "cohort": frame["cohort"],
                                "suite": suite,
                                "chunk": q,
                                "median_delong_se": float(np.nanmedian(cell["se"][usable])),
                                "median_perm_sd": float(
                                    np.median(np.std(null[:, usable], axis=0))
                                ),
                            }
                        )
            if null_max is not None:
                null_rows.append(
                    {
                        "cohort": frame["cohort"],
                        "suite": suite,
                        "scope": "suite_sweep",
                        "n_permutations": n_perm,
                        "null_max_effect_p50": float(np.percentile(null_max, 50)),
                        "null_max_effect_p95": float(np.percentile(null_max, 95)),
                        "null_max_effect_p99": float(np.percentile(null_max, 99)),
                        "se_probe": null_se_probe,
                        "_draws": null_max,
                    }
                )
    return rows, null_rows, checks


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_null: list[dict] = []
    all_checks: list[dict] = []
    for cohort in COHORTS:
        frame = C.build_cohort(cohort)
        rows, null_rows, checks = run_cohort(frame, args.permutations)
        all_rows += rows
        all_null += null_rows
        all_checks += checks
        print(f"{cohort}: {len(rows)} cells", flush=True)
        del frame

    table = pd.DataFrame(all_rows)
    table.to_csv(args.output / "information_map.csv.gz", index=False, float_format="%.6g")

    checks_table = pd.DataFrame(all_checks)
    checks_table.to_csv(args.output / "estimator_checks.csv", index=False)

    # family-wise null, pooled over the whole sweep of a cohort
    per_cohort: dict[str, np.ndarray] = {}
    for entry in all_null:
        draws = entry.pop("_draws")
        per_cohort.setdefault(entry["cohort"], np.zeros_like(draws))
        per_cohort[entry["cohort"]] = np.maximum(per_cohort[entry["cohort"]], draws)
    cohort_null = [
        {
            "cohort": cohort,
            "scope": "cohort_sweep",
            "n_permutations": args.permutations,
            "null_max_effect_p50": float(np.percentile(draws, 50)),
            "null_max_effect_p95": float(np.percentile(draws, 95)),
            "null_max_effect_p99": float(np.percentile(draws, 99)),
        }
        for cohort, draws in per_cohort.items()
    ]
    C.write_json(
        args.output / "permutation_null.json",
        {"per_suite": all_null, "per_cohort": cohort_null},
    )
    print(pd.DataFrame(cohort_null).to_string(index=False))


if __name__ == "__main__":
    main()
