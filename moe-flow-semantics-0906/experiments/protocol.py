#!/usr/bin/env python3
"""Verbatim copy of the frozen v4 / front-back detector protocol.

Every function below is copied unchanged from

    moe-v4-0904/experiments/evaluate_layerwise_alarm_development.py
    moe-hb-front-back-0905/experiments/select_early_lock.py
    moe-hb-front-back-0905/experiments/compare_layers_early.py
    moe-hb-front-back-0905/experiments/survey_reference_frames.py

so that anything scored in this bundle is directly comparable to the published
baselines.  The copy exists only so that importing does not write bytecode into
those directories; there are no behavioural changes.  ``tests/test_protocol.py``
re-derives the published external mobility detector as the check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

LABEL_PATHS = {
    "development_main": PROJECT
    / "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
    "external_8b": PROJECT
    / "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
}

QUANTILES = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.96,
    0.975,
    0.98,
    0.99,
    0.995,
)
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
REPRESENTATIONS = LAYER_NAMES + ("front_median", "back_median", "all_median")
WIDTH, CONFIRMATIONS = 4, 4
MAX_TIMELY_FPR = 0.005
MIN_LOW_PRIOR_PRECISION = 0.60
LOW_PRIOR = 0.25
MODES = ("per_task", "global")


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.pad(np.cumsum(filled, axis=1), ((0, 0), (1, 0)))
    count = np.pad(np.cumsum(finite, axis=1), ((0, 0), (1, 0)))
    output[:, width - 1 :] = (csum[:, width:] - csum[:, :-width]) / width
    complete = count[:, width:] - count[:, :-width] == width
    output[:, width - 1 :][~complete] = np.nan
    return output


def persistent_score(oriented: np.ndarray, count: int) -> np.ndarray:
    """Minimum oriented anomaly score in each causal window of `count` queries."""
    oriented = np.asarray(oriented, dtype=np.float32)
    output = np.full_like(oriented, np.nan)
    for query in range(count - 1, oriented.shape[1]):
        window = oriented[:, query - count + 1 : query + 1]
        complete = np.isfinite(window).all(axis=1)
        output[complete, query] = window[complete].min(axis=1)
    return output


def row_max(values: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmax(values[good], axis=1)
    return output


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 32:
        return float("nan")
    return float(np.quantile(finite, quantile, method="higher"))


def first_query(trigger: np.ndarray) -> np.ndarray:
    fired = trigger.any(axis=1)
    output = np.argmax(trigger, axis=1).astype(np.int16)
    output[~fired] = -1
    return output


def crossfit_thresholds(
    calibration_peak: np.ndarray, task_index: np.ndarray, init_state: np.ndarray
) -> np.ndarray:
    output = np.full((len(calibration_peak), len(QUANTILES)), np.nan, np.float32)
    for task_position in np.unique(task_index):
        take = np.flatnonzero(task_index == task_position)
        task_init = init_state[take]
        if len(take) != 400:
            raise ValueError(f"task {task_position} has {len(take)} episodes")
        for held_out in np.unique(task_init):
            test = task_init == held_out
            reference = ~test
            if int(reference.sum()) != 392 or int(test.sum()) != 8:
                raise ValueError("expected 50 initial states with eight seeds each")
            finite = calibration_peak[take[reference]]
            finite = np.sort(finite[np.isfinite(finite)])
            if len(finite) < 32:
                continue
            positions = [
                int(np.ceil(quantile * (len(finite) - 1))) for quantile in QUANTILES
            ]
            output[take[test]] = finite[positions]
    if not np.isfinite(output).all():
        raise ValueError("non-finite cross-fitted threshold")
    return output


def representations(values: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """The eleven representations the layer work scores, from a [n, q, 8] array."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    filled = np.where(valid[:, :, None], values, 0.0)
    output: dict[str, np.ndarray] = {}
    for position, name in enumerate(LAYER_NAMES):
        output[name] = filled[:, :, position]
    for group, indices in (
        ("front", slice(0, 4)),
        ("back", slice(4, 8)),
        ("all", slice(0, 8)),
    ):
        output[f"{group}_median"] = np.median(filled[:, :, indices], axis=2)
    for name, block in output.items():
        block = block.astype(np.float32, copy=True)
        block[~valid] = np.nan
        output[name] = block
    return output


def suite_of(task_names: np.ndarray, task_index: np.ndarray) -> np.ndarray:
    task = np.asarray(task_names, dtype=str)[np.asarray(task_index, dtype=int)]
    return np.asarray([name.split("/", 1)[0] for name in task])


def survival_prior(
    suite: np.ndarray, length: np.ndarray, risk: np.ndarray
) -> dict[str, dict[int, float]]:
    priors: dict[str, dict[int, float]] = {}
    for name in np.unique(suite):
        take = suite == name
        priors[str(name)] = {
            chunk: float(risk[take][length[take] > chunk].mean())
            for chunk in range(int(length[take].max()))
        }
    return priors


def prior_of(
    first: np.ndarray, suite: np.ndarray, priors: dict[str, dict[int, float]]
) -> np.ndarray:
    out = np.full(len(first), np.nan)
    fired = first >= 0
    out[fired] = [
        priors[s][int(c)] for s, c in zip(suite[fired], first[fired], strict=True)
    ]
    return out


def score_candidate(
    first: np.ndarray, risk: np.ndarray, prior: np.ndarray
) -> dict[str, Any]:
    fired = first >= 0
    early = fired & (prior < LOW_PRIOR)
    tp, fp = int((fired & risk).sum()), int((fired & ~risk).sum())
    etp, efp = int((early & risk).sum()), int((early & ~risk).sum())
    return {
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / int(risk.sum()),
        "timely_fpr": fp / int((~risk).sum()),
        "precision": tp / max(tp + fp, 1),
        "low_prior_tp": etp,
        "low_prior_fp": efp,
        "low_prior_precision": etp / max(etp + efp, 1),
        "low_prior_share": etp / max(tp, 1),
        "mean_alarm_prior": float(np.nanmean(prior[fired])) if tp + fp else float("nan"),
        "lift": (tp / max(tp + fp, 1)) / float(np.nanmean(prior[fired]))
        if (tp + fp) and np.isfinite(np.nanmean(prior[fired]))
        else float("nan"),
    }


def oriented_score(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed
