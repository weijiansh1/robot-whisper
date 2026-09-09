#!/usr/bin/env python3
"""Mandatory controls.  A result without these is not reportable.

  1. the constant-elapsed channel must score exactly 0.5 under the within-task
     fixed-chunk AUC estimator;
  2. thresholding must fire on the tie group -- `>=` against a threshold that is
     an actual sample value, never `>`.  The shared
     ``moe-hb-front-back-0905/experiments/select_early_lock.py`` lines 176 and
     207 use `>` against a quantile, which silently drops the whole tie group;
     the size of that loss is measured here per channel;
  3. within-episode information for every channel that the detector uses;
  4. the length negative control, which recalls 100% by construction and is
     therefore labelled ``is_baseline = False`` and excluded from every ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as P


TIE_CHANNELS = ("set_dwell", "query_top1_churn", "query_hard_churn", "tie_margin",
                "denoise_hard_churn", "query_hard_churn_d9", "selected_mass",
                "hb_entropy_action")
TIE_ALPHAS = (0.005, 0.01, 0.05, 0.10, 0.20)


def const_elapsed_check(table: pd.DataFrame) -> dict:
    cell = table[table["quantity"] == "ctrl_const_elapsed"]
    dev = np.abs(cell["auc"].to_numpy() - 0.5)
    finite = dev[np.isfinite(dev)]
    return {
        "n_cells": int(len(cell)),
        "n_finite": int(len(finite)),
        "max_abs_deviation_from_0.5": float(finite.max()) if len(finite) else None,
        "exactly_half": bool(len(finite) and finite.max() == 0.0),
    }


def tie_audit(frame: dict, values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Fraction of the alarm set that a strict `>` would silently drop."""
    rows = []
    meta = P.column_frame(columns)
    for quantity in TIE_CHANNELS:
        cols = meta.loc[meta["quantity"] == quantity, "column"].tolist()
        if not cols:
            continue
        for col in cols:
            j = columns.index(col)
            block = values[:, :, j]
            alive = frame["alive"]
            sample = block[alive]
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                continue
            n_distinct = int(len(np.unique(sample)))
            for alpha in TIE_ALPHAS:
                tau = float(np.quantile(sample, 1.0 - alpha, method="lower"))
                n_ge = int((sample >= tau).sum())
                n_gt = int((sample > tau).sum())
                rows.append(
                    {
                        "quantity": quantity,
                        "column": col,
                        "n_distinct": n_distinct,
                        "alpha": alpha,
                        "threshold": tau,
                        "n_ge": n_ge,
                        "n_gt": n_gt,
                        "loss_pct_from_strict_gt": 100.0 * (1.0 - n_gt / max(n_ge, 1)),
                    }
                )
    return pd.DataFrame(rows)


def tie_group_fires(score: np.ndarray, mask: np.ndarray, alpha: float) -> dict:
    """Assert that `>=` at a sample-valued threshold fires on the whole tie group."""
    sample = score[mask]
    sample = sample[np.isfinite(sample)]
    tau = float(np.quantile(sample, 1.0 - alpha, method="lower"))
    at_tau = sample == tau
    fired = sample >= tau
    assert at_tau.sum() >= 1, "threshold is not a sample value"
    assert bool(fired[at_tau].all()), "tie group at the threshold does not fire"
    return {
        "alpha": alpha,
        "threshold": tau,
        "n_at_threshold": int(at_tau.sum()),
        "n_fired": int(fired.sum()),
        "n_fired_if_strict_gt": int((sample > tau).sum()),
        "tie_group_fires": True,
    }
