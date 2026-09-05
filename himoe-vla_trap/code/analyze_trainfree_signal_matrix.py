#!/usr/bin/env python3
"""Train-free signal matrix for physical Trap dynamics in HiMoE-VLA.

Feature definitions never use success/failure or Trap labels. Labels enter
only after the scalar time series have been frozen, for grouped AUCs,
permutation tests, and onset-aligned comparisons.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

# The analysis is memory-bandwidth bound; large implicit BLAS pools only exhaust
# worker threads on the shared evaluation host.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
OUTPUT_ROOT = PACKAGE_ROOT / "results/trainfree_signal_matrix"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
INTERMEDIATE_DIR = OUTPUT_ROOT / "intermediate"
A_ROOT = (
    WORKSPACE_ROOT
    / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
)
B_ROOT = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
SSM_CACHE = WORKSPACE_ROOT / "analysis_ssm/cache"

LAYERS = slice(4, 8)  # stored HB layers 12-15
ACTION_TOKENS = slice(1, 11)
STATE_TOKEN = 0
DENOISE_FINAL = 9
FLOW_LATE_START = 6
N_EXPERTS = 32
WINDOW = 8
ALIGN_WINDOW = 4
LAGS = (1, 2, 3, 4)
FIXED_TIMES = (20, 25, 30)
ALIGN_REL = tuple(range(-8, 5))
TEST_REL = (-8, -6, -4, -2, 0)
SEED = 20260903
SCHEMA = "himoe.trainfree_signal_matrix.v1"

PRIMARY_SIGNALS = (
    "weighted_recurrence",
    "lag_periodicity",
    "late_flow_volatility",
    "route_acceleration",
    "gate_entropy",
    "top12_margin",
    "state_action_gap",
    "macro_stickiness",
)

DISPLAY = {
    "weighted_recurrence": "Weighted recurrence",
    "lag_periodicity": "Lag periodicity",
    "late_flow_volatility": "Late-flow volatility",
    "route_acceleration": "Route acceleration",
    "gate_entropy": "Gate entropy",
    "top12_margin": "Top1-Top2 margin",
    "state_action_gap": "State-action gap",
    "macro_stickiness": "Macro-state stickiness",
    "route_mobility": "d9 Hellinger mobility",
    "action_change": "Action change",
    "action_recurrence": "Action recurrence",
    "action_magnitude": "Action magnitude",
    "physical_progress": "Physical progress",
}


@dataclass
class Corpus:
    tag: str
    meta: pd.DataFrame
    valid: np.ndarray
    rowidx: np.ndarray
    actions: np.ndarray
    match_group: np.ndarray
    trap_onset: np.ndarray
    loop_onset: np.ndarray
    static_onset: np.ndarray
    trap_proxy: np.ndarray
    goal_distance: np.ndarray
    physical_progress: np.ndarray
    eef_motion: np.ndarray
    loop_ratio: np.ndarray


def normalize_probability(p: np.ndarray) -> np.ndarray:
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    bc = np.sqrt(np.maximum(p, 0.0) * np.maximum(q, 0.0)).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def weighted_jaccard(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    numerator = np.minimum(p, q).sum(axis=-1)
    denominator = np.maximum(p, q).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing mean on time axis; require a complete finite window."""
    out = np.full(x.shape, np.nan, np.float64)
    for t in range(width - 1, x.shape[1]):
        segment = x[:, t - width + 1 : t + 1]
        good = np.isfinite(segment).all(axis=1)
        out[good, t] = segment[good].mean(axis=1)
    return out


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.ptp(a[good]) == 0 or np.ptp(b[good]) == 0:
        return np.nan
    return float(spearmanr(a[good], b[good]).statistic)


def route_store(tag: str) -> Path:
    if tag == "A":
        return A_ROOT / "formal/server/routes.zarr"
    return B_ROOT / "server/routes.zarr"


def extract_row_features(tag: str, rebuild: bool) -> tuple[dict[str, np.ndarray], dict]:
    """Stream full-flow routes and reduce every row to scalar descriptors."""
    cache = INTERMEDIATE_DIR / f"corpus_{tag}_route_features.npz"
    if cache.exists() and not rebuild:
        with np.load(cache) as data:
            schema = str(data["schema"].item())
            if schema == SCHEMA:
                arrays = {
                    key: np.asarray(data[key])
                    for key in data.files
                    if key not in {"schema", "state_flow_max_deviation"}
                }
                audit = {
                    "state_flow_max_deviation": float(data["state_flow_max_deviation"]),
                    "cache_reused": True,
                }
                return arrays, audit

    group = zarr.open_group(str(route_store(tag)), mode="r")
    router = group["hb_router_probs"]
    total = int(router.shape[0])
    arrays = {
        name: np.empty(total, np.float32)
        for name in (
            "gate_entropy",
            "top12_margin",
            "state_action_gap",
            "late_flow_volatility",
            "route_acceleration",
            "token_disagreement",
        )
    }
    state_flow_max_deviation = 0.0
    block = 64
    for start in range(0, total, block):
        stop = min(total, start + block)
        p = np.asarray(router[start:stop, LAYERS, :, :, :], np.float32)
        p = normalize_probability(p)
        action = p[:, :, :, ACTION_TOKENS, :]
        final_action = action[:, :, DENOISE_FINAL]
        final_state = p[:, :, DENOISE_FINAL, STATE_TOKEN]

        ent = -(final_action * np.log(np.maximum(final_action, 1e-12))).sum(-1)
        arrays["gate_entropy"][start:stop] = (
            ent.mean((1, 2)) / np.log(N_EXPERTS)
        )
        ordered = np.partition(final_action, -2, axis=-1)
        arrays["top12_margin"][start:stop] = (
            ordered[..., -1] - ordered[..., -2]
        ).mean((1, 2))

        action_mean = normalize_probability(final_action.mean(axis=2))
        arrays["state_action_gap"][start:stop] = hellinger(
            final_state, action_mean
        ).mean(axis=1)
        arrays["token_disagreement"][start:stop] = hellinger(
            final_action, action_mean[:, :, None, :]
        ).mean((1, 2))

        late = action[:, :, FLOW_LATE_START:]
        arrays["late_flow_volatility"][start:stop] = (
            1.0 - weighted_jaccard(late[:, :, 1:], late[:, :, :-1])
        ).mean((1, 2, 3))

        root = np.sqrt(action)
        acceleration = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        arrays["route_acceleration"][start:stop] = (
            np.linalg.norm(acceleration, axis=-1).mean((1, 2, 3)) / np.sqrt(2.0)
        )
        state_flow_max_deviation = max(
            state_flow_max_deviation,
            float(np.abs(p[:, :, :, STATE_TOKEN] - p[:, :, :1, STATE_TOKEN]).max()),
        )
        if stop % 2048 < block or stop == total:
            print(f"[{tag}] full-flow rows {stop}/{total}", flush=True)

    np.savez_compressed(
        cache,
        schema=np.asarray(SCHEMA),
        state_flow_max_deviation=np.asarray(state_flow_max_deviation),
        **arrays,
    )
    return arrays, {
        "state_flow_max_deviation": state_flow_max_deviation,
        "cache_reused": False,
    }


def scatter_rows(values: np.ndarray, rowidx: np.ndarray) -> np.ndarray:
    out = np.full(rowidx.shape, np.nan, np.float64)
    valid = rowidx >= 0
    out[valid] = values[rowidx[valid]]
    return out


def periodicity_from_lags(lag_similarity: np.ndarray, width: int) -> np.ndarray:
    smoothed = np.stack(
        [rolling_mean(lag_similarity[..., i], width) for i in range(len(LAGS))],
        axis=-1,
    )
    later = smoothed[..., 1:]
    later_valid = np.isfinite(later).any(axis=-1)
    later_max = np.where(np.isfinite(later), later, -np.inf).max(axis=-1)
    out = later_max - smoothed[..., 0]
    out[~later_valid | ~np.isfinite(smoothed[..., 0])] = np.nan
    return out


def route_temporal_features(
    tag: str, rowidx: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    source = np.load(SSM_CACHE / f"{tag}_full.npy", mmap_mode="r")
    shape = rowidx.shape
    change = np.full(shape, np.nan, np.float64)
    lag_similarity = np.full((*shape, len(LAGS)), np.nan, np.float64)
    for b in range(shape[0]):
        rows = rowidx[b, valid[b]]
        p = np.asarray(source[rows, LAYERS, ACTION_TOKENS, :], np.float32)
        p = normalize_probability(p).reshape(len(rows), 40, N_EXPERTS)
        if len(rows) > 1:
            change[b, 1 : len(rows)] = hellinger(p[1:], p[:-1]).mean(axis=1)
        for ki, lag in enumerate(LAGS):
            if len(rows) > lag:
                lag_similarity[b, lag : len(rows), ki] = weighted_jaccard(
                    p[lag:], p[:-lag]
                ).mean(axis=1)
    lag_valid = np.isfinite(lag_similarity).any(axis=-1)
    recurrence = np.where(np.isfinite(lag_similarity), lag_similarity, -np.inf).max(
        axis=-1
    )
    recurrence[~lag_valid] = np.nan
    return {
        "route_change": change,
        "route_mobility_w8": rolling_mean(change, WINDOW),
        "route_mobility_w4": rolling_mean(change, ALIGN_WINDOW),
        "weighted_recurrence_w8": rolling_mean(recurrence, WINDOW),
        "weighted_recurrence_w4": rolling_mean(recurrence, ALIGN_WINDOW),
        "lag_periodicity_w8": periodicity_from_lags(lag_similarity, WINDOW),
        "lag_periodicity_w4": periodicity_from_lags(lag_similarity, ALIGN_WINDOW),
        "lag_similarity": lag_similarity,
    }


def action_features(
    actions: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    scaled = actions.astype(np.float64).copy()
    early = valid.copy()
    early[:, 13:] = False
    samples = scaled[early]
    scale = np.nanmedian(np.abs(samples), axis=(0, 1))
    scale = np.maximum(scale, 0.01)
    scaled /= scale[None, None, None, :]
    shape = valid.shape
    change = np.full(shape, np.nan)
    magnitude = np.full(shape, np.nan)
    lag_distance = np.full((*shape, len(LAGS)), np.nan)
    magnitude[valid] = np.sqrt(np.mean(scaled[valid] ** 2, axis=(1, 2)))
    for lag_i, lag in enumerate(LAGS):
        for t in range(lag, shape[1]):
            good = valid[:, t] & valid[:, t - lag]
            if good.any():
                delta = scaled[good, t] - scaled[good, t - lag]
                distance = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
                lag_distance[good, t, lag_i] = distance
                if lag == 1:
                    change[good, t] = distance
    reference = np.nanmedian(lag_distance[:, 1:13, 0])
    lag_valid = np.isfinite(lag_distance).any(axis=-1)
    closest = np.where(np.isfinite(lag_distance), lag_distance, np.inf).min(axis=-1)
    closest[~lag_valid] = np.nan
    recurrence = np.exp(-closest / max(reference, 1e-12))
    periodicity = np.full(shape, np.nan)
    periodic_valid = np.isfinite(lag_distance[..., 1:]).any(axis=-1)
    periodicity[periodic_valid] = lag_distance[..., 0][periodic_valid] - np.nanmin(
        lag_distance[..., 1:][periodic_valid], axis=-1
    )
    return {
        "action_change_w8": rolling_mean(change, WINDOW),
        "action_change_w4": rolling_mean(change, ALIGN_WINDOW),
        "action_recurrence_w8": rolling_mean(recurrence, WINDOW),
        "action_recurrence_w4": rolling_mean(recurrence, ALIGN_WINDOW),
        "action_periodicity_w8": rolling_mean(periodicity, WINDOW),
        "action_magnitude_w8": rolling_mean(magnitude, WINDOW),
        "action_magnitude_w4": rolling_mean(magnitude, ALIGN_WINDOW),
        "action_scale": scale,
    }


def goal_distance(objects: np.ndarray, references: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(
        objects[:, :, None, :] - references[None, None, :, :], axis=-1
    ).min(axis=-1)
    return distance.max(axis=1)


def physical_onsets(
    eef: np.ndarray,
    objects: np.ndarray,
    gripper: np.ndarray,
    references: np.ndarray,
) -> tuple[int, int, int]:
    """Frozen loop rule plus validated query-resolution 80-action stasis proxy."""
    goal = goal_distance(objects, references)
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    loop = -1
    for right in range(3, len(eef)):
        for left in range(0, right - 2):
            if (
                np.linalg.norm(eef[right] - eef[left]) <= 0.045
                and np.linalg.norm(objects[right] - objects[left], axis=1).max()
                <= 0.030
                and abs(gripper[right] - gripper[left]) <= 0.012
                and cumulative[right] - cumulative[left] >= 0.120
                and goal[left] - goal[right] <= 0.035
            ):
                loop = right
                break
        if loop >= 0:
            break

    object_step = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    grip_step = np.abs(np.diff(gripper))
    width = 2
    if len(step) >= width:
        kernel = np.ones(width)
        static_window = (
            (np.convolve(step, kernel, mode="valid") <= 0.020)
            & (np.convolve(object_step, kernel, mode="valid") <= 0.005)
            & (np.convolve(grip_step, kernel, mode="valid") <= 0.001)
        )
    else:
        static_window = np.zeros(0, bool)
    run = 0
    static = -1
    for window_start, value in enumerate(static_window):
        run = run + 1 if value else 0
        if run >= 7:  # seven 2-query windows span 8 query intervals = 80 actions
            static = window_start + width
            break
    candidates = [value for value in (loop, static) if value >= 0]
    return loop, static, min(candidates, default=-1)


def trailing_physical(
    values: np.ndarray, valid: np.ndarray, width: int
) -> np.ndarray:
    out = np.full(valid.shape, np.nan, np.float64)
    for t in range(width, valid.shape[1]):
        good = valid[:, t] & valid[:, t - width]
        out[good, t] = (values[good, t - width] - values[good, t]) / width
    return out


def load_corpus(tag: str) -> tuple[Corpus, pd.DataFrame]:
    meta = pd.read_csv(SSM_CACHE / f"{tag}_meta.csv")
    meta["success"] = meta["success"].astype(bool)
    rowidx = np.load(SSM_CACHE / f"{tag}_rowidx.npy")
    valid = rowidx >= 0
    actions = np.load(SSM_CACHE / f"{tag}_act.npy", mmap_mode="r")
    bsz, tmax = valid.shape
    goal = np.full((bsz, tmax), np.nan)
    eef = np.full((bsz, tmax, 3), np.nan)
    objects = np.full((bsz, tmax, 2, 3), np.nan)
    gripper = np.full((bsz, tmax), np.nan)
    loop_proxy = np.full(bsz, -1, np.int16)
    static_proxy = np.full(bsz, -1, np.int16)
    trap_proxy = np.full(bsz, -1, np.int16)

    definitions = json.loads(
        (A_ROOT / "analysis_dryrun_v10/physical_label_definitions.json").read_text()
    )
    references = np.asarray(
        definitions["success_terminal_references"]["moka_pot_1_joint0"],
        np.float64,
    )

    if tag == "A":
        labels = pd.read_csv(A_ROOT / "analysis/candidate_physical_labels.csv")
        by_episode = labels.set_index("episode_id")
        match_group = []
        for b, row in meta.iterrows():
            item = by_episode.loc[int(row.episode_id)]
            path = (
                A_ROOT
                / f"formal/worker{int(item.worker)}/snapshot_{int(item.snapshot):03d}/"
                f"candidate_{int(item.candidate):02d}.npz"
            )
            with np.load(path) as data:
                state = np.asarray(data["policy_state"], np.float64)
                sim = np.asarray(data["sim_state"], np.float64)
            length = int(row["T"])
            eef[b, :length] = state[:length, :3]
            objects[b, :length, 0] = sim[:length, 10:13]
            objects[b, :length, 1] = sim[:length, 17:20]
            gripper[b, :length] = state[:length, 6:8].mean(axis=1)
            match_group.append(str(item.snapshot_key))
            lo, st, tr = physical_onsets(
                eef[b, :length], objects[b, :length], gripper[b, :length], references
            )
            loop_proxy[b], static_proxy[b], trap_proxy[b] = lo, st, tr
        strict = pd.read_csv(
            A_ROOT / "analysis_trap_event_moe_20260829/event_onsets.csv"
        ).set_index("episode_id")
        aligned = strict.reindex(meta.episode_id)
        loop_onset = aligned.loop_onset.to_numpy(np.int16)
        static_onset = aligned.static_onset.to_numpy(np.int16)
        trap_onset = aligned.trap_onset.to_numpy(np.int16)
        match_group = np.asarray(match_group)
    else:
        summaries = json.loads((B_ROOT / "client/summaries.json").read_text())
        match_group = np.empty(bsz, object)
        for b, row in meta.iterrows():
            episode = int(row.episode_id)
            path = B_ROOT / f"client/episode_{episode:02d}.npz"
            with np.load(path, allow_pickle=True) as data:
                state = np.asarray(data["state"], np.float64)
                sim = np.asarray(data["sim_state"], np.float64)
            length = int(row["T"])
            eef[b, :length] = state[:length, :3]
            objects[b, :length, 0] = sim[:length, 10:13]
            objects[b, :length, 1] = sim[:length, 17:20]
            gripper[b, :length] = state[:length, 6:8].mean(axis=1)
            match_group[b] = str(summaries[episode]["init_state_id"])
            lo, st, tr = physical_onsets(
                eef[b, :length], objects[b, :length], gripper[b, :length], references
            )
            loop_proxy[b], static_proxy[b], trap_proxy[b] = lo, st, tr
        loop_onset = loop_proxy.copy()
        static_onset = static_proxy.copy()
        trap_onset = trap_proxy.copy()

    for b in range(bsz):
        length = int(meta.iloc[b]["T"])
        goal[b, :length] = goal_distance(objects[b, :length], references)
    progress = trailing_physical(goal, valid, ALIGN_WINDOW)
    eef_step = np.full((bsz, tmax), np.nan)
    for t in range(1, tmax):
        good = valid[:, t]
        eef_step[good, t] = np.linalg.norm(eef[good, t] - eef[good, t - 1], axis=1)
    eef_motion = rolling_mean(eef_step, ALIGN_WINDOW)
    loop_ratio = np.full((bsz, tmax), np.nan)
    for t in range(ALIGN_WINDOW, tmax):
        good = valid[:, t] & valid[:, t - ALIGN_WINDOW]
        path = np.zeros(good.sum())
        for j in range(t - ALIGN_WINDOW + 1, t + 1):
            path += eef_step[good, j]
        net = np.linalg.norm(eef[good, t] - eef[good, t - ALIGN_WINDOW], axis=1)
        loop_ratio[good, t] = 1.0 - net / np.maximum(path, 1e-12)

    inventory_rows = []
    for kind, proxy, strict_values in (
        ("loop", loop_proxy, loop_onset),
        ("static", static_proxy, static_onset),
        ("trap", trap_proxy, trap_onset),
    ):
        yp = proxy >= 0
        ys = strict_values >= 0
        row = dict(
            corpus=tag,
            event=kind,
            ground_truth="dense" if tag == "A" else "query_proxy",
            n_event=int(ys.sum()),
            n_failure_event=int((ys & ~meta.success.to_numpy()).sum()),
            n_success_event=int((ys & meta.success.to_numpy()).sum()),
        )
        if tag == "A":
            both = yp & ys
            delta = proxy[both] - strict_values[both]
            row.update(
                proxy_n=int(yp.sum()),
                proxy_sensitivity=float((yp & ys).sum() / max(ys.sum(), 1)),
                proxy_specificity=float((~yp & ~ys).sum() / max((~ys).sum(), 1)),
                proxy_agreement=float((yp == ys).mean()),
                proxy_onset_delta_median=float(np.median(delta)) if len(delta) else np.nan,
            )
        inventory_rows.append(row)

    corpus = Corpus(
        tag=tag,
        meta=meta,
        valid=valid,
        rowidx=rowidx,
        actions=np.asarray(actions),
        match_group=match_group,
        trap_onset=trap_onset,
        loop_onset=loop_onset,
        static_onset=static_onset,
        trap_proxy=trap_proxy,
        goal_distance=goal,
        physical_progress=progress,
        eef_motion=eef_motion,
        loop_ratio=loop_ratio,
    )
    return corpus, pd.DataFrame(inventory_rows)


def macro_features(
    tag: str,
    route_change: np.ndarray,
    gate_entropy: np.ndarray,
    late_flow: np.ndarray,
    state_action_gap: np.ndarray,
    corpus: Corpus,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    sources = [route_change, gate_entropy, late_flow, state_action_gap]
    edges = np.asarray([np.nanmedian(x[:, 8:13]) for x in sources])
    code = np.full(corpus.valid.shape, -1, np.int16)
    good = corpus.valid & np.logical_and.reduce([np.isfinite(x) for x in sources])
    bits = np.zeros(corpus.valid.shape, np.int16)
    for bit, (x, edge) in enumerate(zip(sources, edges)):
        bits |= ((x > edge).astype(np.int16) << bit)
    code[good] = bits[good]

    counts = np.zeros((16, 16), np.int64)
    for b in range(len(corpus.meta)):
        for t in range(8, min(30, int(corpus.meta.iloc[b]["T"]) - 1)):
            if code[b, t] >= 0 and code[b, t + 1] >= 0:
                counts[code[b, t], code[b, t + 1]] += 1
    transition = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    pself = np.diag(transition)
    pii = np.full(code.shape, np.nan)
    pii[code >= 0] = pself[code[code >= 0]]
    same = np.full(code.shape, np.nan)
    same[:, 1:] = np.where(
        (code[:, 1:] >= 0) & (code[:, :-1] >= 0),
        (code[:, 1:] == code[:, :-1]).astype(float),
        np.nan,
    )
    stick_w8 = rolling_mean(same, WINDOW)
    stick_w4 = rolling_mean(same, ALIGN_WINDOW)

    success = corpus.meta.success.to_numpy(bool)
    state_rows = []
    common = np.zeros(code.shape, bool)
    common[:, 8:31] = corpus.valid[:, 8:31]
    for state in range(16):
        visits = common & (code == state)
        progress = np.abs(corpus.physical_progress[visits])
        state_rows.append(
            dict(
                corpus=tag,
                state=state,
                bit_route_change=int(bool(state & 1)),
                bit_gate_entropy=int(bool(state & 2)),
                bit_late_flow=int(bool(state & 4)),
                bit_state_action_gap=int(bool(state & 8)),
                visits=int(visits.sum()),
                transitions=int(counts[state].sum()),
                p_self=float(pself[state]),
                expected_dwell=float(1.0 / max(1.0 - pself[state], 1e-12)),
                occupancy_success=float(visits[success].mean()),
                occupancy_failure=float(visits[~success].mean()),
                mean_abs_goal_progress=float(np.nanmean(progress))
                if np.isfinite(progress).any()
                else np.nan,
            )
        )
    state_frame = pd.DataFrame(state_rows)
    eligible = state_frame.visits >= 20
    sticky_cut = state_frame.loc[eligible, "p_self"].quantile(0.75)
    progress_cut = state_frame.loc[eligible, "mean_abs_goal_progress"].quantile(0.25)
    state_frame["descriptive_basin"] = (
        eligible
        & (state_frame.p_self >= sticky_cut)
        & (state_frame.mean_abs_goal_progress <= progress_cut)
    )
    transition_rows = [
        dict(
            corpus=tag,
            from_state=i,
            to_state=j,
            count=int(counts[i, j]),
            probability=float(transition[i, j]),
        )
        for i in range(16)
        for j in range(16)
    ]
    return (
        {
            "macro_state": code,
            "macro_pii": pii,
            "macro_stickiness_w8": stick_w8,
            "macro_stickiness_w4": stick_w4,
            "macro_edges": edges,
        },
        state_frame,
        pd.DataFrame(transition_rows),
    )


def build_signals(
    corpus: Corpus, row_features: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict]:
    row_series = {
        name: scatter_rows(values, corpus.rowidx) for name, values in row_features.items()
    }
    route = route_temporal_features(corpus.tag, corpus.rowidx, corpus.valid)
    action = action_features(corpus.actions, corpus.valid)
    macro, macro_states, macro_transition = macro_features(
        corpus.tag,
        route["route_change"],
        row_series["gate_entropy"],
        row_series["late_flow_volatility"],
        row_series["state_action_gap"],
        corpus,
    )
    signals = {
        "weighted_recurrence_w8": route["weighted_recurrence_w8"],
        "weighted_recurrence_w4": route["weighted_recurrence_w4"],
        "lag_periodicity_w8": route["lag_periodicity_w8"],
        "lag_periodicity_w4": route["lag_periodicity_w4"],
        "late_flow_volatility_w8": rolling_mean(
            row_series["late_flow_volatility"], WINDOW
        ),
        "late_flow_volatility_w4": rolling_mean(
            row_series["late_flow_volatility"], ALIGN_WINDOW
        ),
        "route_acceleration_w8": rolling_mean(
            row_series["route_acceleration"], WINDOW
        ),
        "route_acceleration_w4": rolling_mean(
            row_series["route_acceleration"], ALIGN_WINDOW
        ),
        "gate_entropy_w8": rolling_mean(row_series["gate_entropy"], WINDOW),
        "gate_entropy_w4": rolling_mean(row_series["gate_entropy"], ALIGN_WINDOW),
        "top12_margin_w8": rolling_mean(row_series["top12_margin"], WINDOW),
        "top12_margin_w4": rolling_mean(row_series["top12_margin"], ALIGN_WINDOW),
        "state_action_gap_w8": rolling_mean(row_series["state_action_gap"], WINDOW),
        "state_action_gap_w4": rolling_mean(
            row_series["state_action_gap"], ALIGN_WINDOW
        ),
        "macro_stickiness_w8": macro["macro_stickiness_w8"],
        "macro_stickiness_w4": macro["macro_stickiness_w4"],
        "macro_pii": macro["macro_pii"],
        "route_mobility_w8": route["route_mobility_w8"],
        "route_mobility_w4": route["route_mobility_w4"],
        "action_change_w8": action["action_change_w8"],
        "action_change_w4": action["action_change_w4"],
        "action_recurrence_w8": action["action_recurrence_w8"],
        "action_recurrence_w4": action["action_recurrence_w4"],
        "action_magnitude_w8": action["action_magnitude_w8"],
        "action_magnitude_w4": action["action_magnitude_w4"],
        "physical_progress_w4": corpus.physical_progress,
        "eef_motion_w4": corpus.eef_motion,
        "physical_loop_ratio_w4": corpus.loop_ratio,
    }
    audit = {
        "macro_edges": macro["macro_edges"].tolist(),
        "action_scale": action["action_scale"].tolist(),
    }
    return signals, macro_states, macro_transition, audit


def within_group_auc(
    score: np.ndarray, positive: np.ndarray, groups: np.ndarray
) -> tuple[float, int]:
    score = np.asarray(score, np.float64)
    positive = np.asarray(positive, bool)
    groups = np.asarray(groups)
    concordant = 0.0
    pairs = 0
    for group in np.unique(groups):
        good = (groups == group) & np.isfinite(score)
        a = score[good & positive]
        b = score[good & ~positive]
        if not len(a) or not len(b):
            continue
        concordant += (a[:, None] > b).sum() + 0.5 * (a[:, None] == b).sum()
        pairs += len(a) * len(b)
    return (float(concordant / pairs) if pairs else np.nan), pairs


def rank_residual(
    score: np.ndarray, baseline: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    out = np.full(len(score), np.nan)
    for group in np.unique(groups):
        idx = np.where(
            (groups == group) & np.isfinite(score) & np.isfinite(baseline)
        )[0]
        if len(idx) < 4:
            continue
        y = rankdata(score[idx], method="average")
        x = rankdata(baseline[idx], method="average")
        y -= y.mean()
        x -= x.mean()
        beta = float(x @ y / max(x @ x, 1e-12))
        out[idx] = y - beta * x
    return out


def fixed_family_permutation(
    scores: np.ndarray,
    positive: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not np.isfinite(scores).all():
        raise ValueError("fixed-time family contains non-finite values")
    ranks = np.empty_like(scores, np.float64)
    valid_groups = []
    correction = 0.0
    denominator = 0
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        for column in range(scores.shape[1]):
            ranks[idx, column] = rankdata(scores[idx, column], method="average")
        npos = int(positive[idx].sum())
        nneg = len(idx) - npos
        if npos and nneg:
            valid_groups.append((idx, npos))
            correction += npos * (npos + 1) / 2
            denominator += npos * nneg
    observed = np.asarray(
        [max(within_group_auc(scores[:, j], positive, groups)[0],
             1.0 - within_group_auc(scores[:, j], positive, groups)[0])
         for j in range(scores.shape[1])]
    )
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, scores.shape[1]))
    for draw in range(n_permutations):
        rank_sum = np.zeros(scores.shape[1])
        for idx, npos in valid_groups:
            selected = rng.choice(idx, npos, replace=False)
            rank_sum += ranks[selected].sum(axis=0)
        auc = (rank_sum - correction) / denominator
        null[draw] = np.maximum(auc, 1.0 - auc)
    null_max = null.max(axis=1)
    point = (1 + (null >= observed).sum(axis=0)) / (n_permutations + 1)
    family = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (
        n_permutations + 1
    )
    return point, family, float(np.quantile(null_max, 0.95))


def signal_array(signals: dict[str, np.ndarray], signal: str, width: int) -> np.ndarray:
    key = {
        "weighted_recurrence": f"weighted_recurrence_w{width}",
        "lag_periodicity": f"lag_periodicity_w{width}",
        "late_flow_volatility": f"late_flow_volatility_w{width}",
        "route_acceleration": f"route_acceleration_w{width}",
        "gate_entropy": f"gate_entropy_w{width}",
        "top12_margin": f"top12_margin_w{width}",
        "state_action_gap": f"state_action_gap_w{width}",
        "macro_stickiness": f"macro_stickiness_w{width}",
    }[signal]
    return signals[key]


def fixed_time_experiment(
    corpus: Corpus,
    signals: dict[str, np.ndarray],
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = corpus.meta.success.to_numpy(bool)
    groups = corpus.meta.group.to_numpy()
    baseline = signals["route_mobility_w8"]
    rows = []
    raw_family = []
    residual_family = []
    keys = []
    for signal in PRIMARY_SIGNALS:
        series = signal_array(signals, signal, WINDOW)
        for t in FIXED_TIMES:
            score = series[:, t]
            auc, pairs = within_group_auc(score, success, groups)
            residual = rank_residual(score, baseline[:, t], groups)
            residual_auc, _ = within_group_auc(residual, success, groups)
            rows.append(
                dict(
                    corpus=corpus.tag,
                    family="routing_primary",
                    signal=signal,
                    t=t,
                    auc_success=auc,
                    det_auc=max(auc, 1.0 - auc),
                    direction="success_high" if auc >= 0.5 else "failure_high",
                    residual_auc_success=residual_auc,
                    residual_det_auc=max(residual_auc, 1.0 - residual_auc),
                    rho_route_mobility=safe_spearman(score, baseline[:, t]),
                    npairs=pairs,
                )
            )
            raw_family.append(score)
            residual_family.append(residual)
            keys.append((signal, t))

    raw = np.stack(raw_family, axis=1)
    residual = np.stack(residual_family, axis=1)
    pp, pf, q95 = fixed_family_permutation(
        raw, success, groups, n_permutations, SEED + ord(corpus.tag)
    )
    rp, rf, rq95 = fixed_family_permutation(
        residual, success, groups, n_permutations, SEED + 100 + ord(corpus.tag)
    )
    frame = pd.DataFrame(rows)
    for i, (signal, t) in enumerate(keys):
        mask = (frame.signal == signal) & (frame.t == t)
        frame.loc[mask, "p_point"] = pp[i]
        frame.loc[mask, "p_maxT"] = pf[i]
        frame.loc[mask, "null_max_q95"] = q95
        frame.loc[mask, "residual_p_point"] = rp[i]
        frame.loc[mask, "residual_p_maxT"] = rf[i]
        frame.loc[mask, "residual_null_max_q95"] = rq95

    comparison = {
        "route_mobility": signals["route_mobility_w8"],
        "action_change": signals["action_change_w8"],
        "action_recurrence": signals["action_recurrence_w8"],
        "action_magnitude": signals["action_magnitude_w8"],
        "macro_pii": signals["macro_pii"],
    }
    comparison_rows = []
    for signal, series in comparison.items():
        family = "route_baseline" if signal == "route_mobility" else (
            "macro_descriptive" if signal == "macro_pii" else "action_baseline"
        )
        for t in FIXED_TIMES:
            auc, pairs = within_group_auc(series[:, t], success, groups)
            comparison_rows.append(
                dict(
                    corpus=corpus.tag,
                    family=family,
                    signal=signal,
                    t=t,
                    auc_success=auc,
                    det_auc=max(auc, 1.0 - auc),
                    direction="success_high" if auc >= 0.5 else "failure_high",
                    npairs=pairs,
                )
            )
    return frame, pd.DataFrame(comparison_rows)


def stage_rank_residual(
    score: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    out = np.full(score.shape, np.nan)
    for t in range(score.shape[1]):
        for group in np.unique(groups):
            idx = np.where(
                (groups == group)
                & np.isfinite(score[:, t])
                & np.isfinite(baseline[:, t])
            )[0]
            if len(idx) < 4:
                continue
            y = rankdata(score[idx, t], method="average")
            x = rankdata(baseline[idx, t], method="average")
            y -= y.mean()
            x -= x.mean()
            beta = float(x @ y / max(x @ x, 1e-12))
            out[idx, t] = y - beta * x
    return out


def stage_rank_score(
    score: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Label-free, bounded within-group rank used only for aligned plots."""
    out = np.full(score.shape, np.nan)
    for t in range(score.shape[1]):
        for group in np.unique(groups):
            idx = np.where((groups == group) & np.isfinite(score[:, t]))[0]
            if len(idx) < 3:
                continue
            values = score[idx, t]
            out[idx, t] = rankdata(values, method="average") / (len(idx) + 1) - 0.5
    return out


def matched_group_statistics(
    score: np.ndarray,
    corpus: Corpus,
    relative: int,
    control_kind: str,
    onset_values: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    onset = corpus.trap_onset if onset_values is None else onset_values
    event = onset >= 0
    if control_kind in {"success_no_trap", "success_no_event"}:
        control = corpus.meta.success.to_numpy(bool) & (onset < 0)
    elif control_kind in {"all_no_trap", "all_no_event"}:
        control = onset < 0
    else:
        raise ValueError(control_kind)
    group_rows = []
    event_rows = []
    for group in np.unique(corpus.match_group):
        event_idx = np.where(event & (corpus.match_group == group))[0]
        control_idx = np.where(control & (corpus.match_group == group))[0]
        concordant = 0.0
        pairs = 0
        n_events = 0
        for i in event_idx:
            q = int(onset[i]) + relative
            if q < 0 or q >= score.shape[1] or not np.isfinite(score[i, q]):
                continue
            eligible = control_idx[
                (corpus.meta["T"].to_numpy()[control_idx] > q)
                & np.isfinite(score[control_idx, q])
            ]
            if not len(eligible):
                continue
            x = float(score[i, q])
            values = score[eligible, q]
            wins = float((x > values).sum() + 0.5 * (x == values).sum())
            concordant += wins
            pairs += len(values)
            n_events += 1
            event_rows.append(
                dict(
                    corpus=corpus.tag,
                    group=group,
                    relative=relative,
                    episode_id=int(corpus.meta.iloc[i].episode_id),
                    query=q,
                    event_value=x,
                    control_mean=float(values.mean()),
                    n_controls=len(values),
                )
            )
        if pairs:
            group_rows.append(
                dict(
                    corpus=corpus.tag,
                    group=group,
                    relative=relative,
                    concordant=concordant,
                    pairs=pairs,
                    n_events=n_events,
                    auc_trap_high=concordant / pairs,
                )
            )
    return group_rows, event_rows


def bootstrap_group_auc(
    groups: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    if groups.empty:
        return np.nan, np.nan
    concordant = groups.concordant.to_numpy(float)
    pairs = groups.pairs.to_numpy(float)
    draws = rng.integers(0, len(groups), size=(n_bootstrap, len(groups)))
    auc = concordant[draws].sum(axis=1) / np.maximum(pairs[draws].sum(axis=1), 1)
    return float(np.quantile(auc, 0.025)), float(np.quantile(auc, 0.975))


def onset_signflip_maxT(
    group_frame: pd.DataFrame,
    signals: Iterable[str],
    variant: str,
    n_permutations: int,
    seed: int,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    cells = [(signal, relative) for signal in signals for relative in TEST_REL]
    all_groups = np.asarray(sorted(group_frame.group.astype(str).unique()))
    effects = np.full((len(all_groups), len(cells)), np.nan)
    weights = np.zeros((len(all_groups), len(cells)))
    group_index = {group: i for i, group in enumerate(all_groups)}
    for column, (signal, relative) in enumerate(cells):
        part = group_frame[
            (group_frame.signal == signal)
            & (group_frame.variant == variant)
            & (group_frame.relative == relative)
        ]
        for row in part.itertuples():
            i = group_index[str(row.group)]
            effects[i, column] = float(row.auc_trap_high) - 0.5
            weights[i, column] = float(row.pairs)
    denominator = weights.sum(axis=0)
    observed_signed = np.nansum(effects * weights, axis=0) / np.maximum(denominator, 1)
    observed = np.abs(observed_signed)
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, len(cells)))
    for draw in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(all_groups))
        null[draw] = np.abs(
            np.nansum(effects * weights * signs[:, None], axis=0)
            / np.maximum(denominator, 1)
        )
    null_max = null.max(axis=1)
    q95 = float(np.quantile(null_max, 0.95))
    point = (1 + (null >= observed).sum(axis=0)) / (n_permutations + 1)
    family = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (
        n_permutations + 1
    )
    return {
        cell: (float(point[i]), float(family[i]), q95)
        for i, cell in enumerate(cells)
    }


def onset_experiment(
    corpus: Corpus,
    signals: dict[str, np.ndarray],
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = signals["route_mobility_w4"]
    rng = np.random.default_rng(SEED + 500 + ord(corpus.tag))
    aggregate_rows = []
    group_rows_all = []
    aligned_event_rows = []

    primary = {signal: signal_array(signals, signal, ALIGN_WINDOW) for signal in PRIMARY_SIGNALS}
    primary_residual = {
        signal: stage_rank_residual(series, baseline, corpus.match_group)
        for signal, series in primary.items()
    }
    extras = {
        "route_mobility": baseline,
        "action_change": signals["action_change_w4"],
        "action_recurrence": signals["action_recurrence_w4"],
        "action_magnitude": signals["action_magnitude_w4"],
        "physical_progress": signals["physical_progress_w4"],
    }

    for control_kind in ("all_no_trap", "success_no_trap"):
        variants = [
            ("raw", primary),
            ("residual_route_mobility", primary_residual),
            ("comparator", extras),
        ]
        for variant, mapping in variants:
            for signal, series in mapping.items():
                for relative in ALIGN_REL:
                    group_rows, event_rows = matched_group_statistics(
                        series, corpus, relative, control_kind
                    )
                    for row in group_rows:
                        row.update(
                            signal=signal,
                            variant=variant,
                            control_kind=control_kind,
                        )
                    group_rows_all.extend(group_rows)
                    group_frame = pd.DataFrame(group_rows)
                    concordant = group_frame.concordant.sum() if len(group_frame) else 0
                    pairs = group_frame.pairs.sum() if len(group_frame) else 0
                    auc = float(concordant / pairs) if pairs else np.nan
                    lo, hi = bootstrap_group_auc(group_frame, n_permutations, rng)
                    aggregate_rows.append(
                        dict(
                            corpus=corpus.tag,
                            signal=signal,
                            variant=variant,
                            control_kind=control_kind,
                            relative=relative,
                            auc_trap_high=auc,
                            det_auc=max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                            direction="trap_high" if auc >= 0.5 else "trap_low",
                            ci95_low=lo,
                            ci95_high=hi,
                            n_groups=len(group_frame),
                            n_events=int(group_frame.n_events.sum())
                            if len(group_frame)
                            else 0,
                            npairs=int(pairs),
                        )
                    )

    group_frame = pd.DataFrame(group_rows_all)
    aggregate = pd.DataFrame(aggregate_rows)
    tests = (
        ("all_no_trap", "raw", PRIMARY_SIGNALS, 0),
        ("all_no_trap", "residual_route_mobility", PRIMARY_SIGNALS, 100),
        ("all_no_trap", "comparator", tuple(extras), 200),
        ("success_no_trap", "raw", PRIMARY_SIGNALS, 300),
        ("success_no_trap", "residual_route_mobility", PRIMARY_SIGNALS, 400),
        ("success_no_trap", "comparator", tuple(extras), 500),
    )
    for control_kind, variant, family_signals, offset in tests:
        result = onset_signflip_maxT(
            group_frame[
                (group_frame.control_kind == control_kind)
                & (group_frame.variant == variant)
            ],
            family_signals,
            variant,
            n_permutations,
            SEED + 700 + offset + ord(corpus.tag),
        )
        for (signal, relative), (point, family, q95) in result.items():
            mask = (
                (aggregate.control_kind == control_kind)
                & (aggregate.variant == variant)
                & (aggregate.signal == signal)
                & (aggregate.relative == relative)
            )
            aggregate.loc[mask, "p_point"] = point
            aggregate.loc[mask, "p_maxT"] = family
            aggregate.loc[mask, "null_max_effect_q95"] = q95

    # Two-line onset plots use bounded stage ranks with all non-Trap controls.
    aligned_mapping = {
        "weighted_recurrence": primary["weighted_recurrence"],
        "late_flow_volatility": primary["late_flow_volatility"],
        "state_action_gap": primary["state_action_gap"],
        "physical_progress": extras["physical_progress"],
    }
    for signal, series in aligned_mapping.items():
        standardized = stage_rank_score(series, corpus.match_group)
        for relative in ALIGN_REL:
            _, event_rows = matched_group_statistics(
                standardized, corpus, relative, "all_no_trap"
            )
            for row in event_rows:
                row["signal"] = signal
            aligned_event_rows.extend(event_rows)
    return aggregate, group_frame, pd.DataFrame(aligned_event_rows)


def stratified_onset_signflip_maxT(
    group_frame: pd.DataFrame,
    event_kinds: Iterable[str],
    signals: Iterable[str],
    variant: str,
    n_permutations: int,
    seed: int,
) -> dict[tuple[str, str, int], tuple[float, float, float]]:
    """One maxT family over event kind, signal, and pre-registered lead."""
    cells = [
        (event_kind, signal, relative)
        for event_kind in event_kinds
        for signal in signals
        for relative in TEST_REL
    ]
    all_groups = np.asarray(sorted(group_frame.group.astype(str).unique()))
    group_index = {group: i for i, group in enumerate(all_groups)}
    effects = np.full((len(all_groups), len(cells)), np.nan)
    weights = np.zeros((len(all_groups), len(cells)))
    for column, (event_kind, signal, relative) in enumerate(cells):
        part = group_frame[
            (group_frame.event_kind == event_kind)
            & (group_frame.signal == signal)
            & (group_frame.variant == variant)
            & (group_frame.relative == relative)
        ]
        for row in part.itertuples():
            i = group_index[str(row.group)]
            effects[i, column] = float(row.auc_trap_high) - 0.5
            weights[i, column] = float(row.pairs)
    denominator = weights.sum(axis=0)
    if np.any(denominator <= 0):
        missing = [cells[i] for i in np.where(denominator <= 0)[0]]
        raise ValueError(f"stratified onset cells have no matched pairs: {missing}")
    observed_signed = np.nansum(effects * weights, axis=0) / denominator
    observed = np.abs(observed_signed)
    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, len(cells)))
    for draw in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=len(all_groups))
        null[draw] = np.abs(
            np.nansum(effects * weights * signs[:, None], axis=0) / denominator
        )
    null_max = null.max(axis=1)
    q95 = float(np.quantile(null_max, 0.95))
    point = (1 + (null >= observed).sum(axis=0)) / (n_permutations + 1)
    family = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (
        n_permutations + 1
    )
    return {
        cell: (float(point[i]), float(family[i]), q95)
        for i, cell in enumerate(cells)
    }


def stratified_onset_experiment(
    corpus: Corpus,
    signals: dict[str, np.ndarray],
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loop/static specificity, with multiplicity correction across both types."""
    baseline = signals["route_mobility_w4"]
    primary = {
        signal: signal_array(signals, signal, ALIGN_WINDOW)
        for signal in PRIMARY_SIGNALS
    }
    primary_residual = {
        signal: stage_rank_residual(series, baseline, corpus.match_group)
        for signal, series in primary.items()
    }
    extras = {
        "route_mobility": baseline,
        "action_change": signals["action_change_w4"],
        "action_recurrence": signals["action_recurrence_w4"],
        "action_magnitude": signals["action_magnitude_w4"],
        "physical_progress": signals["physical_progress_w4"],
    }
    event_onsets = {
        "loop": corpus.loop_onset,
        "static": corpus.static_onset,
    }
    rng = np.random.default_rng(SEED + 1100 + ord(corpus.tag))
    aggregate_rows = []
    group_rows_all = []
    for event_kind, onset_values in event_onsets.items():
        for variant, mapping in (
            ("raw", primary),
            ("residual_route_mobility", primary_residual),
            ("comparator", extras),
        ):
            for signal, series in mapping.items():
                for relative in ALIGN_REL:
                    group_rows, _ = matched_group_statistics(
                        series,
                        corpus,
                        relative,
                        "all_no_event",
                        onset_values=onset_values,
                    )
                    for row in group_rows:
                        row.update(
                            event_kind=event_kind,
                            signal=signal,
                            variant=variant,
                            control_kind="all_no_event",
                        )
                    group_rows_all.extend(group_rows)
                    group_part = pd.DataFrame(group_rows)
                    concordant = (
                        float(group_part.concordant.sum()) if len(group_part) else 0.0
                    )
                    pairs = int(group_part.pairs.sum()) if len(group_part) else 0
                    auc = concordant / pairs if pairs else np.nan
                    lo, hi = bootstrap_group_auc(group_part, n_permutations, rng)
                    aggregate_rows.append(
                        dict(
                            corpus=corpus.tag,
                            event_kind=event_kind,
                            signal=signal,
                            variant=variant,
                            control_kind="all_no_event",
                            relative=relative,
                            auc_event_high=auc,
                            det_auc=max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                            direction="event_high" if auc >= 0.5 else "event_low",
                            ci95_low=lo,
                            ci95_high=hi,
                            n_groups=len(group_part),
                            n_events=int(group_part.n_events.sum())
                            if len(group_part)
                            else 0,
                            npairs=pairs,
                        )
                    )

    groups = pd.DataFrame(group_rows_all)
    aggregate = pd.DataFrame(aggregate_rows)
    tests = (
        ("raw", PRIMARY_SIGNALS, 0),
        ("residual_route_mobility", PRIMARY_SIGNALS, 100),
        ("comparator", tuple(extras), 200),
    )
    for variant, family_signals, offset in tests:
        result = stratified_onset_signflip_maxT(
            groups[groups.variant == variant],
            event_onsets,
            family_signals,
            variant,
            n_permutations,
            SEED + 1200 + offset + ord(corpus.tag),
        )
        for (event_kind, signal, relative), (point, family, q95) in result.items():
            mask = (
                (aggregate.event_kind == event_kind)
                & (aggregate.variant == variant)
                & (aggregate.signal == signal)
                & (aggregate.relative == relative)
            )
            aggregate.loc[mask, "p_point"] = point
            aggregate.loc[mask, "p_maxT"] = family
            aggregate.loc[mask, "null_max_effect_q95"] = q95
    return aggregate, groups


def bootstrap_aligned_means(
    events: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (corpus, signal, relative), part in events.groupby(
        ["corpus", "signal", "relative"]
    ):
        group_names = part.group.astype(str).unique()
        per_group = []
        for group in group_names:
            x = part[part.group.astype(str) == group]
            per_group.append(
                (len(x), float(x.event_value.mean()), float(x.control_mean.mean()))
            )
        weights = np.asarray([x[0] for x in per_group], float)
        event_mean = np.asarray([x[1] for x in per_group])
        control_mean = np.asarray([x[2] for x in per_group])
        draws = rng.integers(0, len(per_group), size=(n_bootstrap, len(per_group)))
        denom = weights[draws].sum(axis=1)
        event_boot = (event_mean[draws] * weights[draws]).sum(axis=1) / denom
        control_boot = (control_mean[draws] * weights[draws]).sum(axis=1) / denom
        for kind, values, center in (
            ("trap", event_boot, np.average(event_mean, weights=weights)),
            ("no_trap_control", control_boot, np.average(control_mean, weights=weights)),
        ):
            rows.append(
                dict(
                    corpus=corpus,
                    signal=signal,
                    relative=int(relative),
                    trajectory=kind,
                    mean=float(center),
                    ci95_low=float(np.quantile(values, 0.025)),
                    ci95_high=float(np.quantile(values, 0.975)),
                    n_groups=len(per_group),
                    n_events=len(part),
                )
            )
    return pd.DataFrame(rows)


def make_plots(
    aligned: pd.DataFrame,
    onset: pd.DataFrame,
    type_onset: pd.DataFrame,
    fixed: pd.DataFrame,
) -> None:
    colors = {"trap": "#c6473a", "no_trap_control": "#36738c"}
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True)
    plot_signals = (
        "weighted_recurrence",
        "late_flow_volatility",
        "state_action_gap",
        "physical_progress",
    )
    for row, corpus in enumerate(("A", "B")):
        for col, signal in enumerate(plot_signals):
            ax = axes[row, col]
            part = aligned[(aligned.corpus == corpus) & (aligned.signal == signal)]
            for trajectory in ("trap", "no_trap_control"):
                x = part[part.trajectory == trajectory].sort_values("relative")
                ax.plot(
                    x.relative,
                    x["mean"],
                    color=colors[trajectory],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.5,
                    label="Trap" if trajectory == "trap" else "no-Trap control",
                )
                ax.fill_between(
                    x.relative,
                    x.ci95_low,
                    x.ci95_high,
                    color=colors[trajectory],
                    alpha=0.14,
                    linewidth=0,
                )
            ax.axvline(0, color="#222222", linewidth=0.8)
            ax.axhline(0, color="#777777", linewidth=0.6, linestyle=":")
            ax.set_title(f"{corpus}: {DISPLAY[signal]}", fontsize=10)
            ax.set_xlabel("query relative to physical Trap onset")
            if col == 0:
                ax.set_ylabel("centered within-stage rank")
            ax.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "onset_aligned_signals.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for row, corpus in enumerate(("A", "B")):
        for col, variant in enumerate(("raw", "residual_route_mobility")):
            ax = axes[row, col]
            part = onset[
                (onset.corpus == corpus)
                & (onset.control_kind == "all_no_trap")
                & (onset.variant == variant)
                & (onset.signal.isin(PRIMARY_SIGNALS))
            ]
            matrix = part.pivot(index="signal", columns="relative", values="det_auc")
            matrix = matrix.reindex(PRIMARY_SIGNALS).reindex(columns=ALIGN_REL)
            im = ax.imshow(matrix, aspect="auto", vmin=0.5, vmax=0.95, cmap="viridis")
            ax.set_xticks(range(len(ALIGN_REL)), ALIGN_REL)
            ax.set_yticks(
                range(len(PRIMARY_SIGNALS)),
                [DISPLAY[x] for x in PRIMARY_SIGNALS],
                fontsize=8,
            )
            ax.set_xlabel("query relative to onset")
            ax.set_title(f"{corpus}: {'raw' if variant == 'raw' else 'residual to d9+W4'}")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="direction-free AUC")
            tested = onset[
                (onset.corpus == corpus)
                & (onset.control_kind == "all_no_trap")
                & (onset.variant == variant)
                & (onset.signal.isin(PRIMARY_SIGNALS))
                & (onset.relative.isin(TEST_REL))
                & (onset.p_maxT < 0.05)
            ]
            for result in tested.itertuples():
                ax.text(
                    ALIGN_REL.index(int(result.relative)),
                    PRIMARY_SIGNALS.index(result.signal),
                    "*",
                    ha="center",
                    va="center",
                    color="white" if result.det_auc < 0.82 else "black",
                    fontsize=12,
                    fontweight="bold",
                )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "trainfree_signal_matrix.png", dpi=180)
    plt.close(fig)

    # Compact fixed-t30 comparison for quick inspection.
    t30 = fixed[fixed.t == 30]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(PRIMARY_SIGNALS))
    width = 0.34
    for offset, corpus, color in ((-width / 2, "A", "#2b7189"),
                                  (width / 2, "B", "#c7503e")):
        part = t30[t30.corpus == corpus].set_index("signal").reindex(PRIMARY_SIGNALS)
        ax.bar(x + offset, part.det_auc, width, label=corpus, color=color)
    ax.axhline(0.5, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [DISPLAY[s] for s in PRIMARY_SIGNALS], rotation=24, ha="right")
    ax.set_ylabel("within-group direction-free AUC")
    ax.set_title("Fixed t=30 outcome separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fixed_time_t30_auc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharex=True, sharey=True)
    loop_lines = (
        ("late_flow_volatility", "residual_route_mobility", "#c6473a", "o", "MoE late-flow volatility | d9"),
        ("route_acceleration", "residual_route_mobility", "#7b4e9d", "s", "MoE route acceleration | d9"),
        ("route_mobility", "comparator", "#777777", "^", "d9 mobility"),
        ("action_change", "comparator", "#36738c", "D", "action change"),
        ("action_recurrence", "comparator", "#39855b", "v", "action recurrence"),
    )
    for ax, corpus in zip(axes, ("A", "B")):
        for signal, variant, color, marker, label in loop_lines:
            part = type_onset[
                (type_onset.corpus == corpus)
                & (type_onset.event_kind == "loop")
                & (type_onset.signal == signal)
                & (type_onset.variant == variant)
                & (type_onset.relative <= 0)
            ].sort_values("relative")
            ax.plot(
                part.relative,
                part.det_auc,
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.7,
                label=label,
            )
            significant = part[
                part.relative.isin(TEST_REL) & (part.p_maxT < 0.05)
            ]
            ax.scatter(
                significant.relative,
                significant.det_auc,
                marker="*",
                s=90,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        ax.axhline(0.5, color="#222222", linewidth=0.8)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(f"{corpus}: loop precursor")
        ax.set_xlabel("query relative to loop onset")
        ax.set_ylim(0.48, 0.96)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("matched direction-free AUC")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Train-free loop signals (stars: familywise p < 0.05)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loop_precursor_vs_action_baselines.png", dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    inventory: pd.DataFrame,
    fixed: pd.DataFrame,
    comparisons: pd.DataFrame,
    onset: pd.DataFrame,
    type_onset: pd.DataFrame,
    macro_states: pd.DataFrame,
    audits: dict,
    n_permutations: int,
) -> None:
    def significant_cells(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
        return frame[
            (frame.variant == variant)
            & (frame.relative.isin(TEST_REL))
            & (frame.p_maxT < 0.05)
        ].copy()

    def replicated_cells(
        frame: pd.DataFrame, keys: list[str], direction: str
    ) -> pd.DataFrame:
        selected = frame[frame.p_maxT < 0.05]
        left = selected[selected.corpus == "A"]
        right = selected[selected.corpus == "B"]
        joined = left.merge(right, on=keys, suffixes=("_A", "_B"))
        return joined[joined[f"{direction}_A"] == joined[f"{direction}_B"]]

    aggregate_primary = onset[
        (onset.control_kind == "all_no_trap")
        & (onset.variant == "residual_route_mobility")
        & (onset.signal.isin(PRIMARY_SIGNALS))
    ]
    aggregate_sig = significant_cells(
        aggregate_primary, "residual_route_mobility"
    )
    aggregate_rep = replicated_cells(
        aggregate_sig, ["signal", "relative"], "direction"
    )
    type_primary = type_onset[
        (type_onset.variant == "residual_route_mobility")
        & (type_onset.signal.isin(PRIMARY_SIGNALS))
    ]
    type_sig = significant_cells(type_primary, "residual_route_mobility")
    type_rep = replicated_cells(
        type_sig, ["event_kind", "signal", "relative"], "direction"
    )
    type_rep_pre = type_rep[type_rep.relative < 0]
    replicated_names = ", ".join(
        f"{row.event_kind}/{row.signal}@{int(row.relative)}"
        for row in type_rep.sort_values(
            ["event_kind", "relative", "signal"]
        ).itertuples()
    )
    if not replicated_names:
        replicated_names = "none"
    success_sig = significant_cells(
        onset[
            (onset.control_kind == "success_no_trap")
            & (onset.variant == "residual_route_mobility")
            & (onset.signal.isin(PRIMARY_SIGNALS))
        ],
        "residual_route_mobility",
    )

    lines = [
        "# HiMoE-VLA Train-free Trap 信号矩阵",
        "",
        f"> 正式运行：{n_permutations} 次组内标签置换 / 组级 sign-flip；没有训练 outcome predictor、PCA、k-means 或 probe。",
        "",
        "## 核心结论",
        "",
        "1. `weighted_recurrence` 在固定 t=30 复现了旧 d9 结果，但与 d9 Hellinger mobility 的 Spearman 相关接近 -1；去掉 d9 后没有在 A/B 同时成立，因此它主要是旧变化率的反向表达。",
        f"2. 聚合 Trap 的 d9-residual 预注册单元中，A/B 分别有 "
        f"{int((aggregate_sig.corpus == 'A').sum())}/{int((aggregate_sig.corpus == 'B').sum())} 个通过 maxT；"
        f"严格同 signal、同 lead、同方向的跨语料复现为 {len(aggregate_rep)} 个。",
        f"3. loop/static 分层后，跨语料复现单元为 {replicated_names}。其中真正位于 onset 前的有 "
        f"{len(type_rep_pre)} 个：loop 的 late-flow volatility 与 route acceleration 都在 lead=-2 升高。",
        "4. 所以当前支持的是“type-specific loop precursor”，不是通用 Trap 早预警。聚合任务上 action change/recurrence 在 lead=-4 已很强；但 loop/lead=-2 的动作对照未通过校正。",
        "5. static 的跨语料 periodicity 只在 onset=0 成立，是检测信号而非前兆。现有离线数据也不能估计 recovery 的因果收益；Snapshot-Fork 仍需新增 rollout。",
        "",
        "## 数据与 Ground Truth",
        "",
        "- A：352 条 rolling-star 分支；使用稠密 520-step 物理轨迹定义 loop 或持续 80-action static onset。",
        "- B：512 条 right-16x32 轨迹；使用 query-resolution 代理。该代理先在 A 上验证，再迁移到 B。",
        "- 路由：HB layers 12–15，完整 10 flow；state token=0，action tokens=1–10，逐 token 的 32-way full softmax。",
        "- `gate_entropy`/`top12_margin` 基于 full softmax；它们不是实际 Top-4 combine weights。combine weights 可由 `hb_selected_prob` 归一化恢复。",
        "- 所有 feature 定义不读取成功/失败或 Trap 标签；标签只在评估阶段使用。",
        "",
        "| corpus | event | ground truth | events | failure / success | proxy sensitivity | specificity | agreement |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in inventory.itertuples():
        lines.append(
            f"| {row.corpus} | {row.event} | {row.ground_truth} | {row.n_event} | "
            f"{row.n_failure_event} / {row.n_success_event} | "
            f"{fmt(getattr(row, 'proxy_sensitivity', np.nan))} | "
            f"{fmt(getattr(row, 'proxy_specificity', np.nan))} | "
            f"{fmt(getattr(row, 'proxy_agreement', np.nan))} |"
        )

    lines += [
        "",
        "A 上综合 Trap 代理的验证决定 B-onset 结果是否可解释；B 仍明确标为 proxy，不与 A 的稠密真值等同。",
        "",
        "## 固定 t=30：Outcome 分离",
        "",
        "AUC 是组内成功–失败配对；det AUC 只表示可分性，不预设 Trap 一定高或低。Residual 是组内秩回归去掉 `d9 Hellinger + W8` 后的剩余信号。",
        "",
        "| signal | A raw / residual / p | B raw / residual / p | same raw direction |",
        "|---|---:|---:|---:|",
    ]
    for signal in PRIMARY_SIGNALS:
        a = fixed.query("corpus == 'A' and signal == @signal and t == 30").iloc[0]
        b = fixed.query("corpus == 'B' and signal == @signal and t == 30").iloc[0]
        lines.append(
            f"| `{signal}` | {a.det_auc:.3f} / {a.residual_det_auc:.3f} / {a.residual_p_maxT:.4f} | "
            f"{b.det_auc:.3f} / {b.residual_det_auc:.3f} / {b.residual_p_maxT:.4f} | "
            f"{'yes' if a.direction == b.direction else 'no'} |"
        )

    lines += [
        "",
        "### Action-only 对照",
        "",
        "| signal @ t30 | A det AUC | B det AUC |",
        "|---|---:|---:|",
    ]
    for signal in ("route_mobility", "action_change", "action_recurrence", "action_magnitude"):
        a = comparisons.query("corpus == 'A' and signal == @signal and t == 30").iloc[0]
        b = comparisons.query("corpus == 'B' and signal == @signal and t == 30").iloc[0]
        lines.append(f"| `{signal}` | {a.det_auc:.3f} | {b.det_auc:.3f} |")

    lines += [
        "",
        "## Trap 特异性 Onset 对齐",
        "",
        "主检验把每个 Trap episode 与同 snapshot/init-state、同绝对 query 的全部 no-Trap 轨迹比较，包括其他失败；这样不会把一般 success/failure 差异冒充 Trap 信号。",
        "",
        "下表给出 lead=-4；`p` 对 8 signals × 5 预注册 lead 共同做 maxT。Residual 是去掉同 query、同组 d9 Hellinger mobility 后的秩残差。",
        "",
        "| signal | A raw / residual / p | B raw / residual / p | replicated residual? |",
        "|---|---:|---:|---:|",
    ]
    for signal in PRIMARY_SIGNALS:
        ar = onset.query(
            "corpus == 'A' and signal == @signal and variant == 'raw' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        az = onset.query(
            "corpus == 'A' and signal == @signal and variant == 'residual_route_mobility' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        br = onset.query(
            "corpus == 'B' and signal == @signal and variant == 'raw' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        bz = onset.query(
            "corpus == 'B' and signal == @signal and variant == 'residual_route_mobility' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        replicated = (
            az.p_maxT < 0.05
            and bz.p_maxT < 0.05
            and az.direction == bz.direction
        )
        lines.append(
            f"| `{signal}` | {ar.det_auc:.3f} / {az.det_auc:.3f} / {fmt(az.p_maxT, 4)} | "
            f"{br.det_auc:.3f} / {bz.det_auc:.3f} / {fmt(bz.p_maxT, 4)} | "
            f"{'yes' if replicated else 'no'} |"
        )

    lines += [
        "",
        "### 通过聚合 Trap 主检验的 residual 单元",
        "",
        "| corpus | signal | lead | direction | det AUC | maxT p |",
        "|---|---|---:|---|---:|---:|",
    ]
    if aggregate_sig.empty:
        lines.append("| - | none | - | - | - | - |")
    else:
        for row in aggregate_sig.sort_values(["corpus", "relative", "signal"]).itertuples():
            lines.append(
                f"| {row.corpus} | `{row.signal}` | {int(row.relative)} | {row.direction} | "
                f"{row.det_auc:.3f} | {row.p_maxT:.4f} |"
            )

    lines += [
        "",
        "### 同 lead 的 train-free 对照",
        "",
        "这些 p 值在 comparator 自己的 5 signals × 5 leads 族内校正。physical progress 与 Trap 定义相邻，只作 sanity check。",
        "",
        "| comparator @ lead=-4 | A det / p | B det / p |",
        "|---|---:|---:|",
    ]
    for signal in (
        "route_mobility",
        "action_change",
        "action_recurrence",
        "action_magnitude",
        "physical_progress",
    ):
        a = onset.query(
            "corpus == 'A' and signal == @signal and variant == 'comparator' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        b = onset.query(
            "corpus == 'B' and signal == @signal and variant == 'comparator' and "
            "control_kind == 'all_no_trap' and relative == -4"
        ).iloc[0]
        lines.append(
            f"| `{signal}` | {a.det_auc:.3f} / {fmt(a.p_maxT, 4)} | "
            f"{b.det_auc:.3f} / {fmt(b.p_maxT, 4)} |"
        )

    lines += [
        "",
        "## Loop / Static 分层",
        "",
        "每类事件都与同组、同 query 的 all-no-that-event 控制比较；maxT 同时覆盖 2 event types × 8 signals × 5 leads。",
        "",
        "| corpus | event | signal | lead | direction | residual det AUC | maxT p |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    if type_sig.empty:
        lines.append("| - | - | none | - | - | - | - |")
    else:
        for row in type_sig.sort_values(
            ["corpus", "event_kind", "relative", "signal"]
        ).itertuples():
            lines.append(
                f"| {row.corpus} | {row.event_kind} | `{row.signal}` | "
                f"{int(row.relative)} | {row.direction} | {row.det_auc:.3f} | "
                f"{row.p_maxT:.4f} |"
            )

    lines += [
        "",
        f"严格跨语料复现共 {len(type_rep)} 个，其中 onset 前 {len(type_rep_pre)} 个。"
        "分层解释了 aggregate 的方向冲突；只有通过跨类型 maxT 的单元计入证据。",
        "",
        "### 跨语料复现单元",
        "",
        "| event | signal | lead | direction | A det / p | B det / p |",
        "|---|---|---:|---|---:|---:|",
    ]
    if type_rep.empty:
        lines.append("| - | none | - | - | - | - |")
    else:
        for row in type_rep.sort_values(
            ["event_kind", "relative", "signal"]
        ).itertuples():
            lines.append(
                f"| {row.event_kind} | `{row.signal}` | {int(row.relative)} | "
                f"{row.direction_A} | {row.det_auc_A:.3f} / {row.p_maxT_A:.4f} | "
                f"{row.det_auc_B:.3f} / {row.p_maxT_B:.4f} |"
            )

    lines += [
        "",
        "### Loop 在 lead=-2 的对照",
        "",
        "对照 p 值共同校正 2 event types × 5 comparators × 5 leads。",
        "",
        "| comparator | A det / p | B det / p |",
        "|---|---:|---:|",
    ]
    for signal in (
        "route_mobility",
        "action_change",
        "action_recurrence",
        "action_magnitude",
        "physical_progress",
    ):
        a = type_onset.query(
            "corpus == 'A' and event_kind == 'loop' and signal == @signal and "
            "variant == 'comparator' and relative == -2"
        ).iloc[0]
        b = type_onset.query(
            "corpus == 'B' and event_kind == 'loop' and signal == @signal and "
            "variant == 'comparator' and relative == -2"
        ).iloc[0]
        lines.append(
            f"| `{signal}` | {a.det_auc:.3f} / {a.p_maxT:.4f} | "
            f"{b.det_auc:.3f} / {b.p_maxT:.4f} |"
        )

    lines += [
        "",
        "![loop precursor vs actions](figures/loop_precursor_vs_action_baselines.png)",
        "",
        "## Success-only 敏感性分析",
        "",
        f"与 success/no-Trap 控制比较时，d9-residual 通过的预注册单元为 {len(success_sig)} 个。"
        "这回答的是偏离健康成功轨迹，不是 Trap 特异性；完整结果保存在 `onset_alignment.csv`。",
        "",
        "![onset aligned](figures/onset_aligned_signals.png)",
        "",
        "![signal matrix](figures/trainfree_signal_matrix.png)",
        "",
        "## Macro-state stickiness",
        "",
    ]
    for tag in ("A", "B"):
        d = macro_states[(macro_states.corpus == tag) & macro_states.descriptive_basin]
        states = ", ".join(str(int(x)) for x in d.state) if len(d) else "none"
        max_state = macro_states[macro_states.corpus == tag].sort_values(
            "p_self", ascending=False
        ).iloc[0]
        lines.append(
            f"- {tag}: highest `Pii={max_state.p_self:.3f}` at state {int(max_state.state)}; "
            f"descriptive high-stickiness/low-progress states: {states}."
        )
    lines += [
        "",
        "16 个状态由四个早期无标签中位数位组成：route change、gate entropy、late-flow volatility、state-action gap。"
        "所谓 descriptive basin 使用同批物理 progress，只作机制描述，不当作独立预测证据。",
        "",
        "## 证据边界",
        "",
        f"- state token 跨 flow 的最大概率偏差：A={audits['A']['state_flow_max_deviation']:.3g}，"
        f"B={audits['B']['state_flow_max_deviation']:.3g}；因此 flow volatility/acceleration 只定义在 action tokens。",
        "- full softmax 接近均匀是已知 capture 性质；hard Top-4 边界受 fp16 近平局影响。本轮主时序距离使用 soft probability，避免把 ID 翻转直接解释成动力学跳变。",
        "- 当前 capture 没有每个 flow step 的中间 action trajectory，不能在同一 A/B 上计算真正的 flow-action acceleration baseline。",
        "- SAFE/VLA-FAIL 属于 supervised probe，不纳入本轮 train-free 主检验；action change/recurrence/magnitude 是可用的无训练对照。",
        "- Snapshot-fork 需要新增仿真 rollout。本轮不能从观察数据估计 intervention ATE；旧 q32 与 q19 都是 3/8。",
        "- B 的 static onset 是经过 A 验证的 query proxy，不能冒充 dense-contact ground truth。",
        "",
        "## 产物",
        "",
        "- `fixed_time_auc.csv`: 八信号固定时点评估和 maxT。",
        "- `onset_alignment.csv`: onset-relative AUC、残差与多重校正。",
        "- `onset_group_effects.csv`: 每个 snapshot/init-state 的方向。",
        "- `type_onset_alignment.csv` / `type_onset_group_effects.csv`: loop/static 分层。",
        "- `aligned_means.csv`: 四张时间对齐曲线的数据。",
        "- `macro_states.csv` / `macro_transition.csv`: 确定性 macro-state 动力学。",
        "- `onset_inventory.csv`: A 稠密真值与 B 代理审计。",
    ]
    (OUTPUT_ROOT / "offline_report.generated.zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    global OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, INTERMEDIATE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--nperm", type=int, default=2000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="result root; defaults to himoe-vla_trap/results/trainfree_signal_matrix",
    )
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_dir.resolve()
    TABLE_DIR = OUTPUT_ROOT / "tables"
    FIGURE_DIR = OUTPUT_ROOT / "figures"
    INTERMEDIATE_DIR = OUTPUT_ROOT / "intermediate"
    for directory in (OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, INTERMEDIATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    inventories = []
    fixed_frames = []
    comparison_frames = []
    onset_frames = []
    onset_group_frames = []
    type_onset_frames = []
    type_onset_group_frames = []
    aligned_events = []
    macro_state_frames = []
    macro_transition_frames = []
    audits = {}

    for tag in ("A", "B"):
        print(f"[{tag}] physical data and onset audit", flush=True)
        corpus, inventory = load_corpus(tag)
        inventories.append(inventory)
        print(f"[{tag}] full-flow scalar extraction", flush=True)
        row_features, route_audit = extract_row_features(tag, args.rebuild_cache)
        print(f"[{tag}] temporal and macro-state signals", flush=True)
        signals, macro_states, macro_transition, feature_audit = build_signals(
            corpus, row_features
        )
        macro_state_frames.append(macro_states)
        macro_transition_frames.append(macro_transition)
        audits[tag] = {**route_audit, **feature_audit}
        print(f"[{tag}] fixed-time tests", flush=True)
        fixed, comparisons = fixed_time_experiment(corpus, signals, args.nperm)
        fixed_frames.append(fixed)
        comparison_frames.append(comparisons)
        print(f"[{tag}] onset alignment", flush=True)
        onset, onset_groups, event_rows = onset_experiment(corpus, signals, args.nperm)
        onset_frames.append(onset)
        onset_group_frames.append(onset_groups)
        aligned_events.append(event_rows)
        print(f"[{tag}] loop/static stratification", flush=True)
        type_onset, type_onset_groups = stratified_onset_experiment(
            corpus, signals, args.nperm
        )
        type_onset_frames.append(type_onset)
        type_onset_group_frames.append(type_onset_groups)

    inventory = pd.concat(inventories, ignore_index=True)
    fixed = pd.concat(fixed_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    onset = pd.concat(onset_frames, ignore_index=True)
    onset_groups = pd.concat(onset_group_frames, ignore_index=True)
    type_onset = pd.concat(type_onset_frames, ignore_index=True)
    type_onset_groups = pd.concat(type_onset_group_frames, ignore_index=True)
    aligned_event_frame = pd.concat(aligned_events, ignore_index=True)
    aligned = bootstrap_aligned_means(aligned_event_frame, args.nperm, SEED + 900)
    macro_states = pd.concat(macro_state_frames, ignore_index=True)
    macro_transition = pd.concat(macro_transition_frames, ignore_index=True)

    outputs = {
        "onset_inventory.csv": inventory,
        "fixed_time_auc.csv": fixed,
        "comparison_auc.csv": comparisons,
        "onset_alignment.csv": onset,
        "onset_group_effects.csv": onset_groups,
        "type_onset_alignment.csv": type_onset,
        "type_onset_group_effects.csv": type_onset_groups,
        "aligned_event_values.csv": aligned_event_frame,
        "aligned_means.csv": aligned,
        "macro_states.csv": macro_states,
        "macro_transition.csv": macro_transition,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLE_DIR / name, index=False)
    summary = {
        "schema": SCHEMA,
        "seed": SEED,
        "n_permutations": args.nperm,
        "training": False,
        "feature_labels_used": False,
        "evaluation_labels_used": True,
        "primary_signals": list(PRIMARY_SIGNALS),
        "fixed_times": list(FIXED_TIMES),
        "onset_test_relative_queries": list(TEST_REL),
        "stratified_event_kinds": ["loop", "static"],
        "audits": audits,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    make_plots(aligned, onset, type_onset, fixed)
    write_report(
        inventory,
        fixed,
        comparisons,
        onset,
        type_onset,
        macro_states,
        audits,
        args.nperm,
    )
    print(f"wrote results to {OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
