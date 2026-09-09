#!/usr/bin/env python3
"""Shared loading, channel assembly and mode-aware scoring.

Nothing here is new machinery where old machinery exists:

  * the cohort frames, the per-chunk routing block and the within-task
    fixed-chunk survivors-only AUC come from
    ``moe-early-window-0906/experiments/common.py`` (imported, not copied);
  * the cap-free metric, the fixed-chunk baseline sweep and the rate-matched
    null come from ``moe-capfree-0906/experiments/capfree_protocol.py``
    (imported, not copied).

What is new is only (a) the join to the physical failure-mode annotations and
(b) the three detector arms.

Nothing outside moe-failure-mode-detectors-0906/ is written.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

sys.path.insert(0, str(PROJECT / "moe-early-window-0906" / "experiments"))
sys.path.insert(0, str(PROJECT / "moe-capfree-0906" / "experiments"))

import common as EW  # noqa: E402  (moe-early-window-0906)
import capfree_protocol as CF  # noqa: E402  (moe-capfree-0906)

RESULTS = BUNDLE / "results"
CACHE = BUNDLE / "cache"

LABEL_CSV = PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"

# (run_id in the annotation table) for each replay cohort.
COHORT_RUN_ID = {
    "development_main": "right-50x8-20260903",
    "external_8b": "right-50x8b-20260903",
    "legacy_main16x32": "right-16x32",
}

# The two dominant physical failure modes.  Both carry
# failure_reason_confidence == "medium": they are failure-mode *annotations*
# produced by a replay heuristic, not proof of an internal mechanism.
MODE_DROP = "object_released_or_dropped_before_goal"
MODE_GRASP = "stable_grasp_not_observed"
MODES = (MODE_DROP, MODE_GRASP)
MODE_SHORT = {MODE_DROP: "drop", MODE_GRASP: "grasp"}

LEADS = CF.LEADS               # (0, 2, 4, 8, 12)
HEADLINE_LEAD = CF.HEADLINE_LEAD  # 4
SEED = 20260906

LAYER_NAMES = EW.LAYER_NAMES

# `expert_load_effective_rank` = exp(A + B) / 32 with A = token_entropy and
# B = token_differentiation.  A and B point in opposite directions and cancel,
# so the sum is offered nowhere; A and B are offered separately.
BANNED_QUANTITIES = ("expert_load_effective_rank",)
# Length is the definitional leak (risk <=> length == cap) and needs the cap.
# It is a negative control, never a baseline.
NEGATIVE_CONTROL = ("leak_full_length",)


# --------------------------------------------------------------------------- #
# io helpers


def plain(value: Any) -> Any:
    return EW.plain(value)


def write_json(path: Path, payload: Any) -> None:
    EW.write_json(path, payload)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# cohort + physical-mode labels


def physical_modes(cohort: str, task: np.ndarray, episode: np.ndarray,
                   risk: np.ndarray) -> np.ndarray:
    """Join the physical failure-mode annotation on (run_id, task, episode).

    `task` is "<suite>/<task_name>" and `episode` the episode index, which is
    the key the annotation table stores as (suite, task_name, episode_index).
    """
    labels = pd.read_csv(LABEL_CSV)
    run = labels[labels["run_id"] == COHORT_RUN_ID[cohort]].copy()
    run["task"] = run["suite"] + "/" + run["task_name"]
    merged = pd.DataFrame({"task": task, "episode": episode}).merge(
        run[["task", "episode_index", "primary_failure_reason",
             "recorded_success", "failure_reason_confidence"]],
        left_on=["task", "episode"], right_on=["task", "episode_index"],
        how="left", validate="one_to_one",
    )
    if merged["episode_index"].isna().any():
        raise ValueError(f"{cohort}: {int(merged.episode_index.isna().sum())} "
                         "episodes have no physical-failure annotation")
    mode = merged["primary_failure_reason"].fillna("unlabelled").to_numpy()
    success = merged["recorded_success"].to_numpy(bool)

    # Asserted, not assumed: the annotation's own success flag is the risk
    # label, and no successful episode carries a mode.
    if not np.array_equal(~success, risk):
        raise ValueError(f"{cohort}: recorded_success disagrees with risk")
    if (mode[~risk] != "unlabelled").any():
        raise ValueError(f"{cohort}: a non-risk episode carries a mode label")
    conf = merged["failure_reason_confidence"].to_numpy()
    for m in MODES:
        take = mode == m
        if take.any() and set(pd.unique(conf[take])) != {"medium"}:
            raise ValueError(f"{cohort}/{m}: unexpected confidence values")
    return mode


def _late_step_block(cohort: str, n_ep: int, n_chunk: int
                     ) -> dict[str, np.ndarray] | None:
    """Last-denoising-step slices of token_entropy (A) and token_differentiation (B).

    `moe-early-window-0906` only carries the step-0 slice of these two.  The
    published reference cell `long chunk 14 token_entropy|L3` (0.348 dev /
    0.352 ext) is the *step-9* slice, so it has to be available for the anchor
    to be reproducible.  A and B are carried separately and their sum
    (`expert_load_effective_rank`) is banned - see BANNED_QUANTITIES.
    """
    idx_path = EW.STEP_ROOT / f"{cohort}_index.npz"
    if not idx_path.exists():
        return None
    names = EW.load_npz(idx_path)["metric_names"].astype(str).tolist()
    met = np.load(EW.STEP_ROOT / f"{cohort}_metrics.npy", mmap_mode="r")
    out = {}
    for label, metric in (("token_entropy_d9", "token_entropy"),
                          ("token_differentiation_d9", "token_differentiation")):
        out[label] = np.asarray(met[:, :, :, 9, names.index(metric)], np.float32)
        assert out[label].shape == (n_ep, n_chunk, 8), label
    return out


def load(cohort: str) -> dict[str, Any]:
    """Cohort frame + routing block + physical mode, with the block filtered."""
    frame = EW.build_cohort(cohort)
    frame["mode"] = physical_modes(
        cohort, frame["task"], frame["episode"], frame["risk"])

    late = _late_step_block(cohort, frame["n_ep"], frame["n_chunk"])
    if late is not None:
        extra = np.empty((frame["n_ep"], frame["n_chunk"], 8, len(late)), np.float32)
        for i, key in enumerate(late):
            extra[:, :, :, i] = late[key]
        extra[~frame["valid"]] = np.nan
        frame["values"] = np.concatenate([frame["values"], extra], axis=3)
        frame["quantities"] = frame["quantities"] + tuple(late)

    keep = [i for i, q in enumerate(frame["quantities"])
            if q not in BANNED_QUANTITIES]
    frame["values"] = np.ascontiguousarray(frame["values"][:, :, :, keep])
    frame["quantities"] = tuple(frame["quantities"][i] for i in keep)
    frame["controls"] = tuple(q for q in frame["quantities"]
                              if q in EW.CONTROL_QUANTITIES)
    frame["routing_quantities"] = tuple(
        q for q in frame["quantities"]
        if q not in EW.CONTROL_QUANTITIES)
    return frame


def channel(frame: dict[str, Any], quantity: str, layer: str) -> np.ndarray:
    qi = frame["quantities"].index(quantity)
    li = LAYER_NAMES.index(layer)
    return frame["values"][:, :, li, qi]


def channel_names(frame: dict[str, Any], include_controls: bool = False
                  ) -> list[tuple[str, str]]:
    qs = frame["quantities"] if include_controls else frame["routing_quantities"]
    return [(q, l) for q in qs for l in LAYER_NAMES
            if q not in NEGATIVE_CONTROL]


# --------------------------------------------------------------------------- #
# statistics for the three arms
#
# All three read the same per-chunk channel.  They differ only in what they
# ask of it, which is exactly the hypothesis under test:
#
#   threshold   : is the value large *now*?              (no temporal shape)
#   change_point: did the value *shift* at some chunk?   (two regimes)
#   persistence : has the value been large *repeatedly*? (population baseline)
#
# Every statistic is causal: the value at chunk q uses chunks 0..q only.
# None of them sees the cap, the phase, the task or the suite.

MIN_PRE = 3   # chunks before a candidate change point
MIN_POST = 2  # chunks after it
PERSIST_MIN_CHUNKS = 6
SELF_BASELINE_W = 4


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.where(valid, values, np.nan).astype(np.float64)
    return out


def stat_threshold(u: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The control arm: the standardised value itself."""
    return np.where(valid, u, -np.inf)


def stat_change_point(u: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """max over split points of a before/after mean contrast (binary-segmentation
    CUSUM statistic), computed causally at every chunk.

    At chunk q the statistic is

        max_{s}  sqrt(n1 n2 / (n1 + n2)) * (mean(u[s..q]) - mean(u[0..s-1]))

    with n1 = s, n2 = q - s + 1, s ranging over [MIN_PRE, q - MIN_POST + 1].
    The baseline is the episode's *own* early chunks - the self-baseline the
    hypothesis says should work for a drop and fail for a never-formed grasp.
    """
    n, T = u.shape
    x = np.where(valid, u, 0.0).astype(np.float64)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(x, axis=1)], axis=1)
    out = np.full((n, T), -np.inf)
    for q in range(MIN_PRE + MIN_POST - 1, T):
        best = np.full(n, -np.inf)
        tot = cs[:, q + 1]
        for s in range(MIN_PRE, q - MIN_POST + 2):
            n1, n2 = float(s), float(q - s + 1)
            pre = cs[:, s] / n1
            post = (tot - cs[:, s]) / n2
            scale = math.sqrt(n1 * n2 / (n1 + n2))
            best = np.maximum(best, scale * (post - pre))
        out[:, q] = np.where(valid[:, q], best, -np.inf)
    return out


def stat_self_baseline(u: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Deviation from the episode's own first SELF_BASELINE_W chunks.

    A pure self-baseline: no population reference at all.  Included to test the
    hypothesis' second half directly - a self baseline should fail when the
    baseline window is itself already abnormal.
    """
    n, T = u.shape
    w = SELF_BASELINE_W
    x = np.where(valid, u, np.nan)
    with np.errstate(invalid="ignore"):
        base = np.nanmean(x[:, :w], axis=1)
        spread = np.nanstd(x[:, :w], axis=1)
    spread = np.where(np.isfinite(spread) & (spread > 1e-6), spread, 1.0)
    out = (x - base[:, None]) / spread[:, None]
    out[:, :w] = -np.inf
    return np.where(valid & np.isfinite(out), out, -np.inf)


def _state(u: np.ndarray, valid: np.ndarray, gate: float) -> np.ndarray:
    return valid & (u >= gate)


def stat_persist_fraction(u: np.ndarray, valid: np.ndarray, gate: float
                          ) -> np.ndarray:
    """Fraction of chunks so far spent in the anomalous state.

    The state is defined by a *population* gate (a frozen quantile of the
    development distribution at the same chunk), never by the episode's own
    early chunks.
    """
    a = _state(u, valid, gate).astype(np.float64)
    seen = np.cumsum(valid.astype(np.float64), axis=1)
    frac = np.cumsum(a, axis=1) / np.maximum(seen, 1.0)
    enough = seen >= PERSIST_MIN_CHUNKS
    return np.where(valid & enough, frac, -np.inf)


def stat_persist_run(u: np.ndarray, valid: np.ndarray, gate: float
                     ) -> np.ndarray:
    """Longest run of consecutive anomalous chunks so far."""
    a = _state(u, valid, gate)
    n, T = a.shape
    cur = np.zeros(n)
    best = np.zeros(n)
    out = np.full((n, T), -np.inf)
    for q in range(T):
        cur = np.where(a[:, q], cur + 1.0, 0.0)
        best = np.maximum(best, cur)
        out[:, q] = np.where(valid[:, q], best, -np.inf)
    return out


def stat_persist_cycles(u: np.ndarray, valid: np.ndarray, gate: float
                        ) -> np.ndarray:
    """Number of entries into the anomalous state so far - repetition, not level.

    "Repeated failed attempts" is a count of re-entries, so a channel that goes
    up and down many times scores high even when its peak is unremarkable.
    """
    a = _state(u, valid, gate)
    prev = np.concatenate([np.zeros((a.shape[0], 1), bool), a[:, :-1]], axis=1)
    enter = (a & ~prev).astype(np.float64)
    out = np.cumsum(enter, axis=1)
    seen = np.cumsum(valid.astype(np.float64), axis=1)
    return np.where(valid & (seen >= PERSIST_MIN_CHUNKS), out, -np.inf)


# --------------------------------------------------------------------------- #
# alarm extraction
#
# Every statistic above is turned into a first-alarm chunk by a single rule:
# alarm at the first chunk whose statistic is >= the threshold.  Because the
# running maximum of the statistic is non-decreasing in q, the whole threshold
# sweep is one searchsorted per episode instead of one pass per threshold.
# Ties use >= throughout: a strict > silently drops whole tie groups.


def running_max(stat: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(stat, axis=1)


def first_alarm(stat_runmax: np.ndarray, threshold: float) -> np.ndarray:
    """First chunk with statistic >= threshold, or -1.  `>=`, never `>`."""
    idx = np.argmax(stat_runmax >= threshold, axis=1)
    hit = (stat_runmax >= threshold).any(axis=1)
    return np.where(hit, idx, -1).astype(np.int64)


def first_alarm_sweep(stat_runmax: np.ndarray, thresholds: np.ndarray
                      ) -> np.ndarray:
    """(n_threshold, n_episode) first-alarm chunks, via searchsorted."""
    n, T = stat_runmax.shape
    out = np.empty((len(thresholds), n), np.int64)
    finite = np.where(np.isneginf(stat_runmax), -np.inf, stat_runmax)
    for j, h in enumerate(thresholds):
        mask = finite >= h
        idx = np.argmax(mask, axis=1)
        out[j] = np.where(mask.any(axis=1), idx, -1)
    return out


# --------------------------------------------------------------------------- #
# mode-aware cap-free scoring


def score_modes(first: np.ndarray, frame: dict[str, Any],
                leads: tuple[int, ...] = LEADS) -> dict[str, Any]:
    """`capfree_protocol.score` plus a per-mode true-positive breakdown.

    False alarms carry no mode - a successful episode has none - so the false
    alarm count is shared across modes and reported once.
    """
    risk, length, mode = frame["risk"], frame["length"], frame["mode"]
    row = dict(CF.score(first, risk, length))
    fired = first >= 0
    lead = np.where(fired, length - first, -1)
    hit = fired & risk
    row["median_lead_at_headline"] = (
        float(np.median(lead[hit & (lead >= HEADLINE_LEAD)]))
        if (hit & (lead >= HEADLINE_LEAD)).any() else np.nan)
    for m in MODES:
        tag = MODE_SHORT[m]
        is_m = risk & (mode == m)
        row[f"n_{tag}"] = int(is_m.sum())
        row[f"tp_{tag}"] = int((fired & is_m).sum())
        for b in leads:
            timely = fired & (lead >= b)
            row[f"tp_{tag}_lead{b}"] = int((timely & is_m).sum())
    other = risk & ~np.isin(mode, MODES)
    row["n_other"] = int(other.sum())
    for b in leads:
        timely = fired & (lead >= b)
        row[f"tp_other_lead{b}"] = int((timely & other).sum())
    return row


def mode_mask(frame: dict[str, Any], mode: str) -> np.ndarray:
    if mode == "all":
        return frame["risk"]
    return frame["risk"] & (frame["mode"] == mode)


# --------------------------------------------------------------------------- #
# per-chunk population standardisation (frozen on development)
#
# Admissible under the cap-free protocol: it uses the chunk index and routing
# only.  A prior ablation showed that within a fixed chunk a population rank
# and the raw value are the same order, so this is a rescaling, not a new
# representation; the raw value is what is being ranked.


def chunk_reference(values: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-chunk median and IQR of a channel over every episode alive at q."""
    T = values.shape[1]
    med = np.full(T, np.nan)
    iqr = np.full(T, np.nan)
    for q in range(T):
        col = values[valid[:, q], q]
        col = col[np.isfinite(col)]
        if col.size < 20:
            continue
        lo, m, hi = np.quantile(col, (0.25, 0.5, 0.75))
        med[q] = m
        iqr[q] = max(hi - lo, 1e-9)
    return {"median": med, "iqr": iqr}


def standardise(values: np.ndarray, valid: np.ndarray,
                ref: dict[str, np.ndarray], sign: int) -> np.ndarray:
    med, iqr = ref["median"], ref["iqr"]
    out = sign * (values - med[None, :]) / iqr[None, :]
    return np.where(valid & np.isfinite(out), out, np.nan)
