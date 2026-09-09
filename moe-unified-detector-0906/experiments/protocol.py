#!/usr/bin/env python3
"""Frozen protocol for the single task-agnostic unified early-warning detector.

Everything the build is allowed to know is declared here, once.

What may be used
----------------
* the routing channels themselves, at the current chunk and earlier;
* the *suite horizon cap* (30 / 52 / 28 / 22).  The cap is a rollout
  configuration value fixed before the episode starts.  It is not an outcome and
  it is not task identity.  Phase is therefore ``(q + 1) / cap``, and the gated
  chunks and the alarm window are allowed to depend on the cap.
* a frozen *reference corpus* (development_main) for population ranking.

What may NOT be used
--------------------
* task identity, anywhere: no per-task threshold, no per-task normalisation, no
  per-task reference.  The only stratification allowed at *fit* time is the
  within-task AUC estimator, which is a measurement device, not a detector.
* the outcome of any episode outside development_main.
* any future chunk.

In-window definition
--------------------
The published v7 anchor ("in-window recall 472/1358 with 127 FP") is reproduced
bit-for-bit by ``q <= round(0.65 * cap - 1)`` under Python's banker's rounding,
which gives {goal 18, long 33, object 17, spatial 13}.  That is *not* the same as
``(q + 1) / cap <= 0.65`` for libero_long (which would give 32 and 342 long TP,
not 424).  The anchor's constants are adopted verbatim so that every number in
this bundle is on the same axis as the anchor; both windows are reported in
``results/window_definition.json``.
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
RESULTS = BUNDLE / "results"
CACHE = BUNDLE / "cache"

sys.path.insert(0, str(PROJECT / "moe-early-window-0906" / "experiments"))
sys.path.insert(0, str(PROJECT / "moe-prior-correction-0906" / "experiments"))

import common as ew  # noqa: E402  (shared loaders + the fixed-chunk AUC estimator)

STEP_ROOT = PROJECT / "moe-flow-semantics-0906/results/step_profiles"

CAPS = dict(ew.CAPS)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
COHORTS = ("development_main", "external_8b", "legacy_main16x32")
FIT_COHORT = "development_main"
SEALED_COHORTS = ("external_8b", "legacy_main16x32")

PHASE_TARGET = 0.65
# reproduces the published anchor exactly; see module docstring
IN_WINDOW_END = {s: int(round(PHASE_TARGET * c - 1.0)) for s, c in CAPS.items()}
# the strict (q+1)/cap <= 0.65 reading, reported alongside
IN_WINDOW_END_STRICT = {s: math.floor(PHASE_TARGET * c) - 1 for s, c in CAPS.items()}

SELF_BASELINE_CHUNKS = 4          # self = value - mean(own chunks 0..3)
SELF_BASELINE_MIN_FINITE = 3
FIRST_SCORED_CHUNK = SELF_BASELINE_CHUNKS  # the self form costs the first 4 chunks
MAX_GATED_CHUNKS = 6
SEED = 20260906

LAYER_NAMES = tuple(ew.LAYER_NAMES)

# ---- channel universe ------------------------------------------------------ #
# portable: present in all three cohorts (v4 layerwise mobility + hb layer graphs)
PORTABLE_QUANTITIES = ("mobility",) + tuple(ew.GRAPH_METRICS)
# development / external only
DISCRETE_QUANTITIES = tuple(ew.DISCRETE_QUANTITIES)
STEP_METRICS = (
    "token_entropy",
    "load_entropy",
    "token_differentiation",
    "action_consensus",
    "state_action_alignment",
    "conditional_energy",
    "conditional_effective_rank",
    "flow_speed",
)
STEP_GRID = (0, 2, 4, 6, 9)
CONTROL_QUANTITIES = tuple(ew.CONTROL_QUANTITIES)

NORMALISATIONS = ("raw", "self", "pop")


def plain(value: Any) -> Any:
    return ew.plain(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", "utf-8")


# --------------------------------------------------------------------------- #
# cohort assembly


def _step_block(cohort: str, n_ep: int, n_chunk: int) -> dict[str, np.ndarray]:
    """Per-denoising-step slices on the frozen STEP_GRID.  {} when unavailable."""
    idx_path = STEP_ROOT / f"{cohort}_index.npz"
    if not idx_path.exists():
        return {}
    with np.load(idx_path, allow_pickle=False) as arc:
        names = arc["metric_names"].astype(str).tolist()
        assert np.array_equal(arc["valid"].shape, (n_ep, n_chunk))
    met = np.load(STEP_ROOT / f"{cohort}_metrics.npy", mmap_mode="r")
    mob = np.load(STEP_ROOT / f"{cohort}_mobility.npy", mmap_mode="r")
    smob = np.load(STEP_ROOT / f"{cohort}_state_mobility.npy", mmap_mode="r")
    out: dict[str, np.ndarray] = {}
    for step in STEP_GRID:
        for metric in STEP_METRICS:
            block = np.asarray(met[:, :, :, step, names.index(metric)], np.float32)
            if not np.isfinite(block).any():
                continue  # flow_speed is undefined at step 0
            out[f"{metric}@s{step}"] = block
        out[f"mobility_step@s{step}"] = np.asarray(mob[:, :, :, step], np.float32)
        out[f"state_mobility_step@s{step}"] = np.asarray(smob[:, :, :, step], np.float32)
    for key, arr in out.items():
        assert arr.shape == (n_ep, n_chunk, 8), (key, arr.shape)
    return out


def load_cohort(cohort: str, include_step: bool = True) -> dict[str, Any]:
    """Metadata plus a (n_ep, n_chunk, n_column) float32 block in column order."""
    frame = ew.build_cohort(cohort)
    base_names = list(frame["quantities"])
    values = frame.pop("values")
    n_ep, n_chunk = frame["n_ep"], frame["n_chunk"]

    keep = [q for q in base_names if q not in ew.STEP0_QUANTITIES]  # replaced by STEP_GRID
    blocks = {q: values[:, :, :, base_names.index(q)] for q in keep}
    if include_step:
        blocks.update(_step_block(cohort, n_ep, n_chunk))

    quantities = tuple(blocks)
    stack = np.empty((n_ep, n_chunk, 8, len(quantities)), np.float32)
    for i, name in enumerate(quantities):
        stack[:, :, :, i] = blocks[name]
    stack[~frame["valid"]] = np.nan

    columns = [f"{q}|{lay}" for q in quantities for lay in LAYER_NAMES]
    frame["quantities"] = quantities
    frame["columns"] = columns
    frame["values"] = ew.flatten(stack)  # (n, chunk, quantity*layer)
    frame["cap"] = np.asarray([CAPS[s] for s in frame["suite"]], np.int64)
    frame["window_end"] = np.asarray([IN_WINDOW_END[s] for s in frame["suite"]], np.int64)
    frame["alive"] = frame["valid"]
    return frame


def column_frame(columns: list[str]) -> pd.DataFrame:
    quantity = [c.split("|")[0] for c in columns]
    return pd.DataFrame(
        {
            "column": columns,
            "quantity": quantity,
            "layer": [c.split("|")[1] for c in columns],
            "step": [
                int(q.split("@s")[1]) if "@s" in q else -1 for q in quantity
            ],
            "base": [q.split("@s")[0] for q in quantity],
            "is_control": [q in CONTROL_QUANTITIES for q in quantity],
            "is_portable": [q in PORTABLE_QUANTITIES for q in quantity],
        }
    )


# --------------------------------------------------------------------------- #
# the three normalisations


def self_baseline(values: np.ndarray) -> np.ndarray:
    """Per-episode mean over chunks 0..3, NaN when fewer than 3 chunks are finite."""
    head = values[:, :SELF_BASELINE_CHUNKS]
    finite = np.isfinite(head)
    count = finite.sum(axis=1)
    total = np.where(finite, head, 0.0).sum(axis=1, dtype=np.float64)
    out = np.where(count >= SELF_BASELINE_MIN_FINITE, total / np.maximum(count, 1), np.nan)
    return out.astype(np.float32)


def apply_self(values: np.ndarray) -> np.ndarray:
    """value minus the episode's own chunk 0..3 mean; undefined before chunk 4."""
    out = values - self_baseline(values)[:, None, :]
    out[:, :FIRST_SCORED_CHUNK] = np.nan
    return out.astype(np.float32)


def pop_reference(values: np.ndarray, alive: np.ndarray) -> list[list[np.ndarray]]:
    """Sorted reference values per (chunk, column) over the alive reference corpus."""
    n_chunk, n_col = values.shape[1], values.shape[2]
    ref: list[list[np.ndarray]] = []
    for q in range(n_chunk):
        take = alive[:, q]
        block = values[take, q, :]
        ref.append(
            [np.sort(block[np.isfinite(block[:, j]), j]) for j in range(n_col)]
        )
    return ref


def apply_pop(values: np.ndarray, ref: list[list[np.ndarray]]) -> np.ndarray:
    """Mid-rank fraction of each value inside the frozen reference distribution.

    Ties are handled by the mean of the left and right insertion points, so a
    value that is exactly equal to a block of reference values receives the
    centre of that block, never its edge.
    """
    out = np.full(values.shape, np.nan, np.float32)
    for q in range(values.shape[1]):
        for j in range(values.shape[2]):
            base = ref[q][j]
            if base.size == 0:
                continue
            col = values[:, q, j]
            good = np.isfinite(col)
            if not good.any():
                continue
            v = col[good].astype(np.float64)
            lo = np.searchsorted(base, v, side="left")
            hi = np.searchsorted(base, v, side="right")
            out[good, q, j] = ((lo + hi) / (2.0 * base.size)).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# chunk-wise standardisation (task-agnostic; fitted on the reference corpus)


def chunk_moments(values: np.ndarray, alive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per (chunk, column) mean and sd over the alive reference episodes."""
    n_chunk, n_col = values.shape[1], values.shape[2]
    mu = np.full((n_chunk, n_col), np.nan, np.float64)
    sd = np.full((n_chunk, n_col), np.nan, np.float64)
    import warnings
    for q in range(n_chunk):
        block = values[alive[:, q], q, :]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clean = np.where(np.isfinite(block), block, np.nan)
            if clean.shape[0] == 0:
                continue
            mu[q] = np.nanmean(clean, axis=0)
            sd[q] = np.nanstd(clean, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, np.nan)
    return mu, sd


def standardise(values: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((values - mu[None, :, :]) / sd[None, :, :]).astype(np.float32)


# --------------------------------------------------------------------------- #
# alarms and metrics


def first_alarm(fire: np.ndarray) -> np.ndarray:
    """First chunk at which `fire` is True, -1 when it never is."""
    any_fire = fire.any(axis=1)
    out = np.where(any_fire, fire.argmax(axis=1), -1)
    return out.astype(np.int64)


def evaluate(first: np.ndarray, frame: dict[str, Any]) -> dict[str, Any]:
    """TP/FP, in-window TP/FP, recall, FPR, precision, per suite and overall."""
    risk = frame["risk"]
    suite = frame["suite"]
    fired = first >= 0
    in_window = fired & (first <= frame["window_end"])
    n_risk = int(risk.sum())
    n_safe = int((~risk).sum())
    out = {
        "n_episodes": int(len(risk)),
        "n_risk": n_risk,
        "n_safe": n_safe,
        "tp": int((fired & risk).sum()),
        "fp": int((fired & ~risk).sum()),
        "iw_tp": int((in_window & risk).sum()),
        "iw_fp": int((in_window & ~risk).sum()),
    }
    out["recall"] = out["tp"] / n_risk if n_risk else float("nan")
    out["fpr"] = out["fp"] / n_safe if n_safe else float("nan")
    out["iw_recall"] = out["iw_tp"] / n_risk if n_risk else float("nan")
    out["iw_fpr"] = out["iw_fp"] / n_safe if n_safe else float("nan")
    denom = out["iw_tp"] + out["iw_fp"]
    out["iw_precision"] = out["iw_tp"] / denom if denom else float("nan")
    denom = out["tp"] + out["fp"]
    out["precision"] = out["tp"] / denom if denom else float("nan")
    for s in SUITES:
        m = suite == s
        nr = int((risk & m).sum())
        ns = int((~risk & m).sum())
        out[f"iw_tp_{s}"] = int((in_window & risk & m).sum())
        out[f"iw_fp_{s}"] = int((in_window & ~risk & m).sum())
        out[f"n_risk_{s}"] = nr
        out[f"iw_recall_{s}"] = out[f"iw_tp_{s}"] / nr if nr else float("nan")
        out[f"iw_fpr_{s}"] = out[f"iw_fp_{s}"] / ns if ns else float("nan")
    return out


def in_window_first(first: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    """First alarm censored at the window end (-1 when it fired too late)."""
    return np.where((first >= 0) & (first <= frame["window_end"]), first, -1)


def pareto_front(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Upper-left Pareto front: maximise y for each x, keep only non-dominated."""
    ordered = points.sort_values([x, y], ascending=[True, False]).reset_index(drop=True)
    best = -np.inf
    keep = []
    for i, row in ordered.iterrows():
        if row[y] > best:
            keep.append(i)
            best = row[y]
    return ordered.loc[keep].reset_index(drop=True)
