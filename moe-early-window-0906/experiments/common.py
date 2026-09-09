#!/usr/bin/env python3
"""Shared loaders, quantity assembly and the fixed-chunk task-stratified estimator.

The estimator implemented here is the one the brief specifies, and nothing else:

    for each (cohort, suite, chunk q, quantity, layer)
        keep only episodes still running at q  (length > q)
        inside each task, compute the Mann-Whitney AUC of the quantity value
        against eventual risk
        pool the per-task AUCs with Mann-Whitney weights n+ * n-

Every episode in a stratum shares a task and a chunk index, so neither task
identity nor elapsed time can contribute to the statistic.  Two summaries are
produced for every cell: the pair-weighted pool (which the Mann-Whitney weights
give) and the task-equal pool (each contributing task counted once).

Variance is DeLong's, computed per task and combined as a weighted sum of
independent strata.  Confidence intervals are logit intervals so they stay in
[0, 1].

Nothing outside moe-early-window-0906/ is written.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

MOBILITY_ROOT = PROJECT / "moe-v4-0904/results/layerwise_mobility"
GRAPH_ROOT = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
DISCRETE_ROOT = PROJECT / "moe-unused-channels-0906/results/channels"
STEP_ROOT = PROJECT / "moe-flow-semantics-0906/results/step_profiles"
LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
LEGACY_ROOT = PROJECT / "moe-v4-0904/results/cache16x32_v4"

RESULTS = BUNDLE / "results"
CACHE = BUNDLE / "cache"

CAPS = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}

# Operational window.  phase of an alarm at chunk q in an episode of length n is
# (q + 1) / n -- the fraction of the episode consumed once the alarm chunk is
# done.  Risk episodes always have n == cap, so "phase <= 0.65" is
# q <= ceil(0.65 * cap) - 1.  This reproduces the sweep ends the brief gives for
# spatial (13), goal (18) and object (17).
PHASE_TARGET = 0.65
WINDOW_65 = {s: int(math.floor(PHASE_TARGET * c - 1.0)) for s, c in CAPS.items()}
# Chunk at which the per-suite survival prior P(risk | alive at q) crosses 0.25,
# quoted in the brief as phase 51.0 / 62.1 / 63.0 / 61.9 %.
WINDOW_CROSS = {
    "libero_spatial": 13,
    "libero_goal": 18,
    "libero_object": 17,
    "libero_long": 26,
}
# Sweep goes past the crossing for contrast.
SWEEP_END = {"libero_spatial": 18, "libero_goal": 24, "libero_object": 22, "libero_long": 34}
SWEEP_START = 4

MIN_POS = 3
MIN_NEG = 3
MIN_STRATUM = 30  # same floor moe-audit-0906 / moe-token-geometry-0906 use
WIDTH = 4  # frozen causal trailing-mean width
SEED = 20260906

LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")

GRAPH_METRICS = (
    "flow_path",
    "flow_settling_log_ratio",
    "flow_endpoint",
    "action_consensus",
    "state_action_alignment",
    "conditional_energy",
    "conditional_effective_rank",
    "partial_edge_std",
    "expert_load_effective_rank",
    "conditional_query_d1",
    "partial_query_d1",
)
DISCRETE_QUANTITIES = (
    "query_hard_churn",
    "query_hard_churn_d9",
    "query_top1_churn",
    "denoise_hard_churn",
    "set_dwell",
    "tie_margin",
    "selected_mass",
    "hb_entropy_action",
)
# early-denoising-step axis; step 0 of the flow-semantics profiles
STEP0_QUANTITIES = (
    "mobility_d0",
    "state_mobility_d0",
    "mobility_d0_minus_d9",
    "token_entropy_d0",
    "token_differentiation_d0",
    "flow_speed_d1",  # flow_speed is a step-to-step difference, undefined at step 0
)
CONTROL_QUANTITIES = (
    "ctrl_const_elapsed",       # constant inside every stratum -> must be exactly 0.5
    "ctrl_episode_const_rand",  # random, drawn once per episode, constant inside it
    "ctrl_episode_const_rand2",
    "ctrl_flow_noise_seed",     # outcome-blind metadata, constant inside the episode
    "ctrl_white_noise",         # fresh iid noise at every (episode, chunk)
    "leak_full_length",         # the definitional leak: risk <=> length == cap
)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", "utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


# --------------------------------------------------------------------------- #
# cohort assembly


def _base(cohort: str) -> dict[str, Any]:
    if cohort == "legacy_main16x32":
        mob = load_npz(LEGACY_ROOT / "layerwise_mobility.npz")
        graph = load_npz(GRAPH_ROOT / "legacy_main16x32.npz")
        alarms = pd.read_csv(LEGACY_ROOT / "episode_alarms.csv")
        risk = alarms["failure"].to_numpy(bool)
        assert np.array_equal(alarms["length"].to_numpy(int), mob["length"].astype(int))
    else:
        name = {"development_main": "main_reference", "external_8b": "external_8b"}[cohort]
        mob = load_npz(MOBILITY_ROOT / f"{name}.npz")
        graph = load_npz(GRAPH_ROOT / f"{cohort}.npz")
        labels = pd.read_csv(LABEL_ROOT / f"{cohort}_clean_labels.csv")
        risk = labels["original_failure"].to_numpy(bool)
        assert np.array_equal(labels["episode"].to_numpy(int), mob["episode"].astype(int))

    for key in ("episode", "task_index", "length"):
        assert np.array_equal(graph[key], mob[key]), f"{cohort}: graph/mobility {key}"

    task = mob["task_names"].astype(str)[mob["task_index"].astype(int)]
    suite = np.asarray([t.split("/", 1)[0] for t in task])
    length = mob["length"].astype(int)
    valid = mob["valid"].astype(bool)
    n_ep, n_chunk = valid.shape
    # valid must be exactly "chunk index below the episode length"
    assert np.array_equal(valid, np.arange(n_chunk)[None, :] < length[:, None])
    at_cap = np.asarray([length[i] == CAPS[suite[i]] for i in range(n_ep)])
    return {
        "cohort": cohort,
        "n_ep": n_ep,
        "n_chunk": n_chunk,
        "task": task,
        "suite": suite,
        "length": length,
        "valid": valid,
        "risk": risk,
        "at_cap": at_cap,
        "episode": mob["episode"].astype(int),
        "init_state_id": mob["init_state_id"].astype(int),
        "flow_noise_seed": mob["flow_noise_seed"].astype(np.int64),
        "task_index": mob["task_index"].astype(int),
        "task_names": mob["task_names"].astype(str),
        "_mobility": mob["mobility"],
        "_graph": graph,
    }


def _step0_block(cohort: str, n_ep: int, n_chunk: int) -> dict[str, np.ndarray] | None:
    """step-0 slices of the flow-semantics per-denoising-step profiles."""
    cache_path = CACHE / f"{cohort}_step0.npz"
    if cache_path.exists():
        return load_npz(cache_path)
    idx_path = STEP_ROOT / f"{cohort}_index.npz"
    if not idx_path.exists():
        return None
    idx = load_npz(idx_path)
    valid = idx["valid"].astype(bool)
    names = idx["metric_names"].astype(str).tolist()
    mob = np.load(STEP_ROOT / f"{cohort}_mobility.npy", mmap_mode="r")
    smob = np.load(STEP_ROOT / f"{cohort}_state_mobility.npy", mmap_mode="r")
    met = np.load(STEP_ROOT / f"{cohort}_metrics.npy", mmap_mode="r")
    out = {
        "mobility_d0": np.asarray(mob[:, :, :, 0], dtype=np.float32),
        "state_mobility_d0": np.asarray(smob[:, :, :, 0], dtype=np.float32),
        "mobility_d0_minus_d9": np.asarray(
            mob[:, :, :, 0] - mob[:, :, :, 9], dtype=np.float32
        ),
    }
    for label, metric, step in (
        ("token_entropy_d0", "token_entropy", 0),
        ("token_differentiation_d0", "token_differentiation", 0),
        ("flow_speed_d1", "flow_speed", 1),
    ):
        out[label] = np.asarray(met[:, :, :, step, names.index(metric)], dtype=np.float32)
    for key, arr in out.items():
        assert arr.shape == (n_ep, n_chunk, 8), (key, arr.shape)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    return out


def build_cohort(cohort: str) -> dict[str, Any]:
    """Metadata plus one (n_ep, n_chunk, n_layer, n_quantity) float32 block."""
    frame = _base(cohort)
    n_ep, n_chunk = frame["n_ep"], frame["n_chunk"]
    graph = frame.pop("_graph")
    gnames = graph["metric_names"].astype(str).tolist()

    blocks: dict[str, np.ndarray] = {"mobility": np.asarray(frame.pop("_mobility"), np.float32)}
    for metric in GRAPH_METRICS:
        blocks[metric] = np.asarray(graph["metrics"][:, :, :, gnames.index(metric)], np.float32)

    disc_index = DISCRETE_ROOT / f"{cohort}_index.npz"
    if disc_index.exists():
        di = load_npz(disc_index)
        assert np.array_equal(di["episode"].astype(int), frame["episode"])
        assert np.array_equal(di["length"].astype(int), frame["length"])
        qnames = di["quantity_names"].astype(str).tolist()
        arr = np.load(DISCRETE_ROOT / f"{cohort}_quantities.npy", mmap_mode="r")
        for quantity in DISCRETE_QUANTITIES:
            blocks[quantity] = np.asarray(arr[:, :, :, qnames.index(quantity)], np.float32)

    step0 = _step0_block(cohort, n_ep, n_chunk)
    if step0 is not None:
        for quantity in STEP0_QUANTITIES:
            blocks[quantity] = np.asarray(step0[quantity], np.float32)

    rng = np.random.default_rng(SEED + abs(hash(cohort)) % 10_000)
    chunk_ix = np.arange(n_chunk, dtype=np.float32)[None, :, None]
    ones = np.ones((n_ep, n_chunk, 8), np.float32)
    controls = {
        "ctrl_const_elapsed": np.broadcast_to(chunk_ix, (n_ep, n_chunk, 8)).astype(np.float32),
        "ctrl_episode_const_rand": (
            rng.standard_normal(n_ep, dtype=np.float32)[:, None, None] * ones
        ),
        "ctrl_episode_const_rand2": (
            rng.standard_normal(n_ep, dtype=np.float32)[:, None, None] * ones
        ),
        "ctrl_flow_noise_seed": (
            frame["flow_noise_seed"].astype(np.float32)[:, None, None] * ones
        ),
        "ctrl_white_noise": rng.standard_normal((n_ep, n_chunk, 8), dtype=np.float32),
        "leak_full_length": frame["length"].astype(np.float32)[:, None, None] * ones,
    }
    blocks.update(controls)

    quantities = tuple(blocks)
    stack = np.empty((n_ep, n_chunk, 8, len(quantities)), np.float32)
    for i, name in enumerate(quantities):
        stack[:, :, :, i] = blocks[name]
    # everything outside an episode is undefined
    stack[~frame["valid"]] = np.nan
    frame["quantities"] = quantities
    frame["controls"] = tuple(k for k in quantities if k in CONTROL_QUANTITIES)
    frame["values"] = stack
    frame["layer_names"] = list(LAYER_NAMES)
    return frame


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Causal trailing mean over the chunk axis; NaN until `width` observations exist.

    Identical in behaviour to the frozen `dev.trailing_mean`, but written over the
    4-D (episode, chunk, layer, quantity) block.  Uses float64 accumulation so the
    result does not depend on chunk-axis rounding -- the failure mode that let a
    within-episode-constant channel pass a frozen selection rule.
    """
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0).astype(np.float64)
    cs = np.cumsum(filled, axis=1)
    ct = np.cumsum(finite, axis=1)
    n_chunk = values.shape[1]
    out = np.full(values.shape, np.nan, np.float32)
    for q in range(n_chunk):
        lo = q - width
        s = cs[:, q] - (cs[:, lo] if lo >= 0 else 0.0)
        c = ct[:, q] - (ct[:, lo] if lo >= 0 else 0)
        ok = c == width
        block = np.where(ok, s / np.maximum(c, 1), np.nan)
        out[:, q] = block.astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# the estimator


def _rank_auc_columns(x: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC and DeLong variance for every column of x, positives given by `pos`.

    x is (n, m) and must be finite.  Ties get midranks, so a column that is
    constant scores exactly 0.5 with variance 0.
    """
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    r_all = rankdata(x, axis=0)
    r_pos = rankdata(x[pos], axis=0)
    r_neg = rankdata(x[~pos], axis=0)
    auc = (r_all[pos].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    v10 = (r_all[pos] - r_pos) / n0
    v01 = 1.0 - (r_all[~pos] - r_neg) / n1
    var = v10.var(axis=0, ddof=1) / n1 + v01.var(axis=0, ddof=1) / n0
    return auc, var


def stratified_cell(
    x: np.ndarray, risk: np.ndarray, task: np.ndarray
) -> dict[str, np.ndarray]:
    """Task-stratified pooled AUC for every column of x.

    Columns are handled together; a task contributes to a column only when that
    column is finite for every alive episode of the task (the blocks here are
    finite-or-all-NaN per chunk, so this is a per-task decision, not per-episode).
    """
    m = x.shape[1]
    num = np.zeros(m)
    den = np.zeros(m)
    var_num = np.zeros(m)
    eq_sum = np.zeros(m)
    eq_var = np.zeros(m)
    eq_n = np.zeros(m)
    tasks = np.unique(task)
    for name in tasks:
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
        var_num[good] += var * (w ** 2)
        eq_sum[good] += auc
        eq_var[good] += var
        eq_n[good] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        pooled_se = np.where(den > 0, np.sqrt(var_num) / np.maximum(den, 1e-12), np.nan)
        equal = np.where(eq_n > 0, eq_sum / np.maximum(eq_n, 1), np.nan)
        equal_se = np.where(eq_n > 0, np.sqrt(eq_var) / np.maximum(eq_n, 1), np.nan)
    return {
        "auc": pooled,
        "se": pooled_se,
        "pairs": den,
        "auc_equal": equal,
        "se_equal": equal_se,
        "n_tasks": eq_n,
    }


def logit_ci(auc: np.ndarray, se: np.ndarray, z: float = 1.959963985) -> tuple[np.ndarray, np.ndarray]:
    a = np.clip(auc, 1e-9, 1 - 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.log(a / (1 - a))
        se_eta = se / (a * (1 - a))
        lo = 1.0 / (1.0 + np.exp(-(eta - z * se_eta)))
        hi = 1.0 / (1.0 + np.exp(-(eta + z * se_eta)))
    lo = np.where(np.isfinite(se) & (se > 0), lo, np.nan)
    hi = np.where(np.isfinite(se) & (se > 0), hi, np.nan)
    # a constant column has se == 0 and auc == 0.5 exactly
    exact = np.isfinite(se) & (se == 0)
    lo = np.where(exact, auc, lo)
    hi = np.where(exact, auc, hi)
    return lo, hi


def suite_chunks(suite: str, extended: bool = True) -> list[int]:
    end = SWEEP_END[suite] if extended else WINDOW_CROSS[suite]
    return list(range(SWEEP_START, end + 1))


def column_index(quantities: tuple[str, ...]) -> tuple[list[str], list[str]]:
    names, layers = [], []
    for q in quantities:
        for layer in LAYER_NAMES:
            names.append(q)
            layers.append(layer)
    return names, layers


def flatten(values: np.ndarray) -> np.ndarray:
    """(n, chunk, layer, quantity) -> (n, chunk, layer*quantity) in (quantity, layer) order."""
    n, c, nl, nq = values.shape
    return np.ascontiguousarray(values.transpose(0, 1, 3, 2).reshape(n, c, nq * nl))
