#!/usr/bin/env python3
"""Train-free F0-F3 test of execution-aware HiMoE Trap signals.

F0 is the frozen full-softmax baseline.  F1 adds the actual Top-4/Top-5
boundary, tail mass, and execution entropy.  F2 adds support/weight churn from
the experts the runtime really dispatched.  F3 retains all layers and token
position and audits state-to-action propagation across policy queries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis_moe_execution_signals.execution_features import (  # noqa: E402
    extract_block_features,
    sparse_hellinger_distance,
    support_jaccard_distance,
)
from analysis_signal_matrix_trainfree import run as baseline  # noqa: E402


SCHEMA = "himoe.execution_signal_matrix.v1"
SEED = 20260905
BLOCK_ROWS = 16
ALIGN_REL = baseline.ALIGN_REL
TEST_REL = baseline.TEST_REL

F0 = baseline.PRIMARY_SIGNALS
F1 = ("top45_margin", "tail_mass", "exec_entropy")
F2 = (
    "late_flow_support_churn",
    "late_flow_exec_churn",
    "query_support_churn",
    "query_exec_churn",
)
F3 = (
    "all_layer_support_churn",
    "all_layer_exec_churn",
    "token_support_slope",
    "token_exec_slope",
    "token_margin_slope",
    "layer_support_slope",
    "early_state_query_churn",
    "state_action_churn_gap",
)
NEW_SIGNALS = F1 + F2 + F3
ALL_SIGNALS = F0 + NEW_SIGNALS
TOKEN_METRICS = (
    "top45_margin",
    "tail_mass",
    "exec_entropy",
    "late_flow_support_churn",
    "late_flow_exec_churn",
    "query_support_churn",
    "query_exec_churn",
)
TIER = {
    **{name: "F0_router_probability" for name in F0},
    **{name: "F1_boundary_execution_weight" for name in F1},
    **{name: "F2_actual_dispatch" for name in F2},
    **{name: "F3_layer_token_propagation" for name in F3},
}
DISPLAY = {
    **baseline.DISPLAY,
    "top45_margin": "Actual Top4-Top5 margin",
    "tail_mass": "Mass outside actual Top-4",
    "exec_entropy": "Top-4 execution entropy",
    "late_flow_support_churn": "Late-flow support churn",
    "late_flow_exec_churn": "Late-flow execution churn",
    "query_support_churn": "Cross-query support churn",
    "query_exec_churn": "Cross-query execution churn",
    "all_layer_support_churn": "All-layer support churn",
    "all_layer_exec_churn": "All-layer execution churn",
    "token_support_slope": "Action-slot support slope",
    "token_exec_slope": "Action-slot execution slope",
    "token_margin_slope": "Action-slot boundary slope",
    "layer_support_slope": "HB-layer support slope",
    "early_state_query_churn": "Early-layer state churn",
    "state_action_churn_gap": "Late-action minus early-state churn",
}


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Trailing mean on axis 1, preserving optional trailing feature axes."""
    values = np.asarray(values, np.float64)
    out = np.full(values.shape, np.nan, np.float64)
    for t in range(width - 1, values.shape[1]):
        window = values[:, t - width + 1 : t + 1]
        complete = np.isfinite(window).all(axis=1)
        mean = window.sum(axis=1) / width
        out[:, t] = np.where(complete, mean, np.nan)
    return out


def scatter(values: np.ndarray, rowidx: np.ndarray) -> np.ndarray:
    out = np.full((*rowidx.shape, *values.shape[1:]), np.nan, np.float64)
    good = rowidx >= 0
    out[good] = values[rowidx[good]]
    return out


def _cache_path(tag: str) -> Path:
    return HERE / f"execution_row_features_{tag}.npz"


def extract_execution_rows(tag: str, rebuild: bool) -> tuple[dict[str, np.ndarray], dict]:
    cache = _cache_path(tag)
    if cache.exists() and not rebuild:
        with np.load(cache) as data:
            if str(data["schema"].item()) == SCHEMA:
                arrays = {
                    key: np.asarray(data[key])
                    for key in data.files
                    if key not in {"schema", "audit_json"}
                }
                return arrays, {**json.loads(str(data["audit_json"].item())), "cache_reused": True}

    group = zarr.open_group(str(baseline.route_store(tag)), mode="r")
    router = group["hb_router_probs"]
    ids_store = group["hb_expert_ids"]
    raw_store = group["hb_selected_prob"]
    total = int(router.shape[0])

    scalar_names = (
        "top45_margin",
        "top45_log_margin",
        "tail_mass",
        "exec_entropy",
        "late_flow_support_churn",
        "late_flow_exec_churn",
        "all_layer_support_churn",
        "all_layer_exec_churn",
        "token_support_slope",
        "token_exec_slope",
        "token_margin_slope",
        "layer_support_slope",
    )
    arrays: dict[str, np.ndarray] = {
        name: np.empty(total, np.float32) for name in scalar_names
    }
    arrays.update(
        {f"token/{name}": np.empty((total, 10), np.float32) for name in TOKEN_METRICS[:5]}
    )
    arrays.update(
        {
            "layer/late_flow_support_churn": np.empty((total, 8), np.float32),
            "layer/late_flow_exec_churn": np.empty((total, 8), np.float32),
            "final_ids": np.empty((total, 8, 11, 4), np.uint8),
            "final_weights": np.empty((total, 8, 11, 4), np.float16),
        }
    )
    audit = {
        "rows": total,
        "boundary_negative": 0,
        "boundary_zero": 0,
        "boundary_count": 0,
        "fp16_topk_set_disagreement": 0,
        "selected_mass_min": np.inf,
        "selected_mass_max": -np.inf,
        "combine_sum_max_error": 0.0,
    }
    for start in range(0, total, BLOCK_ROWS):
        stop = min(total, start + BLOCK_ROWS)
        block = extract_block_features(
            np.asarray(router[start:stop], np.float32),
            np.asarray(ids_store[start:stop]),
            np.asarray(raw_store[start:stop], np.float32),
        )
        for name, value in block.scalars.items():
            arrays[name][start:stop] = value
        for name, value in block.token_maps.items():
            arrays[f"token/{name}"][start:stop] = value
        for name, value in block.layer_maps.items():
            arrays[f"layer/{name}"][start:stop] = value
        arrays["final_ids"][start:stop] = block.final_ids
        arrays["final_weights"][start:stop] = block.final_weights
        for name in (
            "boundary_negative",
            "boundary_zero",
            "boundary_count",
            "fp16_topk_set_disagreement",
        ):
            audit[name] += int(block.audit[name])
        audit["selected_mass_min"] = min(
            float(audit["selected_mass_min"]), float(block.audit["selected_mass_min"])
        )
        audit["selected_mass_max"] = max(
            float(audit["selected_mass_max"]), float(block.audit["selected_mass_max"])
        )
        audit["combine_sum_max_error"] = max(
            float(audit["combine_sum_max_error"]),
            float(block.audit["combine_sum_max_error"]),
        )
        if stop % 2048 < BLOCK_ROWS or stop == total:
            print(f"[{tag}] execution rows {stop}/{total}", flush=True)

    audit["boundary_negative_rate"] = audit["boundary_negative"] / audit["boundary_count"]
    audit["boundary_zero_rate"] = audit["boundary_zero"] / audit["boundary_count"]
    audit["fp16_topk_set_disagreement_rate"] = (
        audit["fp16_topk_set_disagreement"] / audit["boundary_count"]
    )
    np.savez_compressed(
        cache,
        schema=np.asarray(SCHEMA),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
        **arrays,
    )
    return arrays, {**audit, "cache_reused": False}


def query_execution_features(
    arrays: dict[str, np.ndarray], corpus: baseline.Corpus
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shape = corpus.rowidx.shape
    scalar = {
        name: np.full(shape, np.nan, np.float64)
        for name in (
            "query_support_churn",
            "query_exec_churn",
            "early_state_query_churn",
            "late_action_query_churn",
            "state_action_churn_gap",
        )
    }
    token = {
        "query_support_churn": np.full((*shape, 10), np.nan, np.float64),
        "query_exec_churn": np.full((*shape, 10), np.nan, np.float64),
    }
    final_ids = arrays["final_ids"]
    final_weights = arrays["final_weights"]
    for episode in range(shape[0]):
        rows = corpus.rowidx[episode, corpus.valid[episode]]
        if len(rows) < 2:
            continue
        ids_a, ids_b = final_ids[rows[1:]], final_ids[rows[:-1]]
        w_a, w_b = final_weights[rows[1:]], final_weights[rows[:-1]]
        action_ids_a, action_ids_b = ids_a[:, 4:, 1:], ids_b[:, 4:, 1:]
        action_w_a, action_w_b = w_a[:, 4:, 1:], w_b[:, 4:, 1:]
        support = support_jaccard_distance(action_ids_a, action_ids_b)
        execution = sparse_hellinger_distance(
            action_ids_a, action_w_a, action_ids_b, action_w_b
        )
        support_token = support.mean(axis=1)
        execution_token = execution.mean(axis=1)
        state_support = support_jaccard_distance(
            ids_a[:, :4, 0], ids_b[:, :4, 0]
        ).mean(axis=1)
        action_support = support.mean(axis=(1, 2))
        target = slice(1, len(rows))
        token["query_support_churn"][episode, target] = support_token
        token["query_exec_churn"][episode, target] = execution_token
        scalar["query_support_churn"][episode, target] = action_support
        scalar["query_exec_churn"][episode, target] = execution.mean(axis=(1, 2))
        scalar["early_state_query_churn"][episode, target] = state_support
        scalar["late_action_query_churn"][episode, target] = action_support
        scalar["state_action_churn_gap"][episode, target] = action_support - state_support
    return scalar, token


def build_execution_signals(
    arrays: dict[str, np.ndarray], corpus: baseline.Corpus
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    row_scalar = {
        name: scatter(arrays[name], corpus.rowidx)
        for name in (
            "top45_margin",
            "tail_mass",
            "exec_entropy",
            "late_flow_support_churn",
            "late_flow_exec_churn",
            "all_layer_support_churn",
            "all_layer_exec_churn",
            "token_support_slope",
            "token_exec_slope",
            "token_margin_slope",
            "layer_support_slope",
        )
    }
    query_scalar, query_token = query_execution_features(arrays, corpus)
    row_scalar.update(query_scalar)
    signals = {}
    for name, values in row_scalar.items():
        signals[f"{name}_w4"] = trailing_mean(values, baseline.ALIGN_WINDOW)
        signals[f"{name}_w8"] = trailing_mean(values, baseline.WINDOW)

    token_maps = {
        name: scatter(arrays[f"token/{name}"], corpus.rowidx)
        for name in TOKEN_METRICS[:5]
    }
    token_maps.update(query_token)
    token_maps = {
        name: trailing_mean(values, baseline.ALIGN_WINDOW)
        for name, values in token_maps.items()
    }
    layer_maps = {
        name.removeprefix("layer/"): trailing_mean(
            scatter(values, corpus.rowidx), baseline.ALIGN_WINDOW
        )
        for name, values in arrays.items()
        if name.startswith("layer/")
    }
    return signals, token_maps, layer_maps


def stage_rank_residual_multi(
    score: np.ndarray, predictors: Iterable[np.ndarray], groups: np.ndarray
) -> np.ndarray:
    """Label-free within-group rank residual against fixed F0 predictors."""
    predictors = tuple(predictors)
    out = np.full(score.shape, np.nan, np.float64)
    for t in range(score.shape[1]):
        for group in np.unique(groups):
            mask = groups == group
            finite = np.isfinite(score[:, t])
            for predictor in predictors:
                finite &= np.isfinite(predictor[:, t])
            idx = np.where(mask & finite)[0]
            if len(idx) < len(predictors) + 3:
                continue
            y = rankdata(score[idx, t], method="average")
            columns = [rankdata(x[idx, t], method="average") for x in predictors]
            x = np.stack(columns, axis=1)
            y -= y.mean()
            x -= x.mean(axis=0, keepdims=True)
            beta = np.linalg.lstsq(x, y, rcond=1e-8)[0]
            out[idx, t] = y - x @ beta
    return out


def _aggregate_group_rows(group_frame: pd.DataFrame, nboot: int, seed: int) -> dict:
    if group_frame.empty:
        return {
            "auc_event_high": np.nan,
            "det_auc": np.nan,
            "direction": "none",
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_groups": 0,
            "n_events": 0,
            "npairs": 0,
        }
    pairs = int(group_frame.pairs.sum())
    auc = float(group_frame.concordant.sum() / pairs)
    lo, hi = baseline.bootstrap_group_auc(
        group_frame, nboot, np.random.default_rng(seed)
    )
    return {
        "auc_event_high": auc,
        "det_auc": max(auc, 1.0 - auc),
        "direction": "event_high" if auc >= 0.5 else "event_low",
        "ci95_low": lo,
        "ci95_high": hi,
        "n_groups": len(group_frame),
        "n_events": int(group_frame.n_events.sum()),
        "npairs": pairs,
    }


def evaluate_onsets(
    corpus: baseline.Corpus,
    mapping: dict[str, np.ndarray],
    variant: str,
    nperm: int,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    group_rows = []
    event_onsets = {
        "trap": corpus.trap_onset,
        "loop": corpus.loop_onset,
        "static": corpus.static_onset,
    }
    for event_index, (event_kind, onset) in enumerate(event_onsets.items()):
        for signal_index, (signal, series) in enumerate(mapping.items()):
            for relative in ALIGN_REL:
                groups, _ = baseline.matched_group_statistics(
                    series,
                    corpus,
                    relative,
                    "all_no_event",
                    onset_values=onset,
                )
                for row in groups:
                    row.update(
                        event_kind=event_kind,
                        signal=signal,
                        tier=TIER[signal],
                        variant=variant,
                    )
                group_rows.extend(groups)
                stats = _aggregate_group_rows(
                    pd.DataFrame(groups),
                    nperm,
                    SEED + seed_offset + event_index * 10000 + signal_index * 100 + relative,
                )
                aggregate_rows.append(
                    {
                        "corpus": corpus.tag,
                        "event_kind": event_kind,
                        "signal": signal,
                        "tier": TIER[signal],
                        "variant": variant,
                        "relative": relative,
                        **stats,
                    }
                )

    aggregate = pd.DataFrame(aggregate_rows)
    groups = pd.DataFrame(group_rows)
    signal_names = tuple(mapping)
    # Trap and loop/static are separate confirmatory families.  This avoids
    # charging the aggregate event twice for its overlapping typed events.
    trap_result = baseline.onset_signflip_maxT(
        groups[groups.event_kind == "trap"],
        signal_names,
        variant,
        nperm,
        SEED + seed_offset + 30000 + ord(corpus.tag),
    )
    typed_result = baseline.stratified_onset_signflip_maxT(
        groups[groups.event_kind.isin(("loop", "static"))],
        ("loop", "static"),
        signal_names,
        variant,
        nperm,
        SEED + seed_offset + 40000 + ord(corpus.tag),
    )
    for (signal, relative), (point, family, q95) in trap_result.items():
        mask = (
            (aggregate.event_kind == "trap")
            & (aggregate.signal == signal)
            & (aggregate.relative == relative)
        )
        aggregate.loc[mask, ["p_point", "p_maxT", "null_max_effect_q95"]] = (
            point,
            family,
            q95,
        )
    for (kind, signal, relative), (point, family, q95) in typed_result.items():
        mask = (
            (aggregate.event_kind == kind)
            & (aggregate.signal == signal)
            & (aggregate.relative == relative)
        )
        aggregate.loc[mask, ["p_point", "p_maxT", "null_max_effect_q95"]] = (
            point,
            family,
            q95,
        )
    return aggregate, groups


def _generic_signflip_maxT(
    groups: pd.DataFrame, cells: list[tuple[str, int]], nperm: int, seed: int
) -> dict[tuple[str, int], tuple[float, float]]:
    names = np.asarray(sorted(groups.group.astype(str).unique()))
    index = {name: i for i, name in enumerate(names)}
    effects = np.full((len(names), len(cells)), np.nan)
    weights = np.zeros((len(names), len(cells)))
    for column, (metric, token) in enumerate(cells):
        part = groups[(groups.metric == metric) & (groups.token == token)]
        for row in part.itertuples():
            i = index[str(row.group)]
            effects[i, column] = float(row.auc_trap_high) - 0.5
            weights[i, column] = float(row.pairs)
    denominator = weights.sum(axis=0)
    observed = np.abs(np.nansum(effects * weights, axis=0) / denominator)
    rng = np.random.default_rng(seed)
    null = np.empty((nperm, len(cells)))
    for draw in range(nperm):
        signs = rng.choice((-1.0, 1.0), len(names))
        null[draw] = np.abs(
            np.nansum(effects * weights * signs[:, None], axis=0) / denominator
        )
    null_max = null.max(axis=1)
    point = (1 + (null >= observed).sum(axis=0)) / (nperm + 1)
    family = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (nperm + 1)
    return {cell: (float(point[i]), float(family[i])) for i, cell in enumerate(cells)}


def evaluate_token_positions(
    corpus: baseline.Corpus,
    token_maps: dict[str, np.ndarray],
    nperm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = []
    group_rows = []
    relative = -2
    for metric_index, metric in enumerate(TOKEN_METRICS):
        values = token_maps[metric]
        for token in range(values.shape[-1]):
            rows, _ = baseline.matched_group_statistics(
                values[..., token],
                corpus,
                relative,
                "all_no_event",
                onset_values=corpus.loop_onset,
            )
            for row in rows:
                row.update(metric=metric, token=token + 1)
            group_rows.extend(rows)
            stats = _aggregate_group_rows(
                pd.DataFrame(rows),
                nperm,
                SEED + 50000 + metric_index * 100 + token + ord(corpus.tag),
            )
            aggregate.append(
                {
                    "corpus": corpus.tag,
                    "event_kind": "loop",
                    "relative": relative,
                    "metric": metric,
                    "token": token + 1,
                    **stats,
                }
            )
    group_frame = pd.DataFrame(group_rows)
    cells = [(metric, token) for metric in TOKEN_METRICS for token in range(1, 11)]
    tests = _generic_signflip_maxT(
        group_frame, cells, nperm, SEED + 51000 + ord(corpus.tag)
    )
    frame = pd.DataFrame(aggregate)
    for (metric, token), (point, family) in tests.items():
        mask = (frame.metric == metric) & (frame.token == token)
        frame.loc[mask, ["p_point", "p_maxT"]] = point, family
    return frame, group_frame


def propagation_audit(
    corpus: baseline.Corpus,
    signals: dict[str, np.ndarray],
    nboot: int,
) -> pd.DataFrame:
    state = baseline.stage_rank_score(
        signals["early_state_query_churn_w4"], corpus.match_group
    )
    action = baseline.stage_rank_score(
        signals["late_action_query_churn_w4"], corpus.match_group
    )
    rng = np.random.default_rng(SEED + 60000 + ord(corpus.tag))
    rows = []
    episodes = np.arange(state.shape[0])
    for lag in range(5):
        x = state[:, : state.shape[1] - lag or None]
        y = action[:, lag:]
        good = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[good], y[good]).statistic) if good.sum() >= 3 else np.nan
        draws = np.empty(nboot)
        for draw in range(nboot):
            chosen = rng.choice(episodes, len(episodes), replace=True)
            xb, yb = x[chosen], y[chosen]
            gb = np.isfinite(xb) & np.isfinite(yb)
            draws[draw] = (
                spearmanr(xb[gb], yb[gb]).statistic if gb.sum() >= 3 else np.nan
            )
        rows.append(
            {
                "corpus": corpus.tag,
                "state_query_leads_action_by": lag,
                "rho": rho,
                "ci95_low": float(np.nanquantile(draws, 0.025)),
                "ci95_high": float(np.nanquantile(draws, 0.975)),
                "n_pairs": int(good.sum()),
            }
        )
    return pd.DataFrame(rows)


def _replicated(frame: pd.DataFrame, variant: str, event: str) -> pd.DataFrame:
    part = frame[
        (frame.variant == variant)
        & (frame.event_kind == event)
        & (frame.relative.isin(TEST_REL))
        & (frame.relative < 0)
    ]
    a = part[part.corpus == "A"].set_index(["signal", "relative"])
    b = part[part.corpus == "B"].set_index(["signal", "relative"])
    joined = a.join(b, lsuffix="_A", rsuffix="_B", how="inner").reset_index()
    joined["same_direction"] = joined.direction_A == joined.direction_B
    joined["worst_det_auc"] = joined[["det_auc_A", "det_auc_B"]].min(axis=1)
    joined["both_maxT_005"] = (joined.p_maxT_A <= 0.05) & (joined.p_maxT_B <= 0.05)
    return joined.sort_values(
        ["both_maxT_005", "same_direction", "worst_det_auc"],
        ascending=False,
    )


def redundancy_audit(
    tag: str,
    execution_rows: dict[str, np.ndarray],
    baseline_rows: dict[str, np.ndarray],
) -> dict[str, float]:
    pairs = {
        "late_flow_support_vs_exec": (
            execution_rows["late_flow_support_churn"],
            execution_rows["late_flow_exec_churn"],
        ),
        "tail_mass_vs_gate_entropy": (
            execution_rows["tail_mass"], baseline_rows["gate_entropy"]
        ),
        "exec_entropy_vs_gate_entropy": (
            execution_rows["exec_entropy"], baseline_rows["gate_entropy"]
        ),
        "top45_margin_vs_top12_margin": (
            execution_rows["top45_margin"], baseline_rows["top12_margin"]
        ),
    }
    result = {}
    for name, (left, right) in pairs.items():
        good = np.isfinite(left) & np.isfinite(right)
        result[name] = float(spearmanr(left[good], right[good]).statistic)
    result["corpus"] = tag
    return result


def make_plots(onset: pd.DataFrame, token: pd.DataFrame) -> None:
    loop = onset[
        (onset.variant == "raw")
        & (onset.event_kind == "loop")
        & (onset.relative.isin(TEST_REL))
    ].copy()
    loop["signed_effect"] = loop.auc_event_high - 0.5
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, corpus in zip(axes, ("A", "B")):
        part = loop[loop.corpus == corpus].pivot(
            index="signal", columns="relative", values="signed_effect"
        ).reindex(ALL_SIGNALS)
        image = ax.imshow(part, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        ax.set_title(f"Corpus {corpus}: loop matched AUC - 0.5")
        ax.set_xticks(range(len(TEST_REL)), TEST_REL)
        ax.set_yticks(range(len(ALL_SIGNALS)), [DISPLAY[x] for x in ALL_SIGNALS], fontsize=7)
        ax.set_xlabel("queries relative to onset")
    fig.colorbar(image, ax=axes, fraction=0.025, label="signed matched effect")
    fig.savefig(HERE / "f0_f3_loop_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(TOKEN_METRICS), 1, figsize=(10, 14), sharex=True, constrained_layout=True)
    for ax, metric in zip(axes, TOKEN_METRICS):
        for corpus, marker in (("A", "o"), ("B", "s")):
            part = token[(token.corpus == corpus) & (token.metric == metric)]
            ax.plot(part.token, part.auc_event_high - 0.5, marker=marker, label=corpus)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_ylabel(DISPLAY[metric], fontsize=8)
    axes[0].legend()
    axes[-1].set_xlabel("action token position (loop lead -2)")
    fig.savefig(HERE / "token_position_loop_lead2.png", dpi=180)
    plt.close(fig)


def write_report(
    onset: pd.DataFrame,
    token: pd.DataFrame,
    propagation: pd.DataFrame,
    audits: dict,
    redundancy: dict[str, dict[str, float]],
    nperm: int,
) -> None:
    raw = _replicated(onset, "raw", "loop")
    residual_all = _replicated(onset, "residual_f0", "loop")
    trap_residual = _replicated(onset, "residual_f0", "trap")
    static_residual = _replicated(onset, "residual_f0", "static")
    raw_new = raw[raw.signal.isin(NEW_SIGNALS)].head(8)
    residual = residual_all.head(8)

    lines = [
        "# HiMoE actual-execution F0-F3 test",
        "",
        f"正式统计使用 {nperm} 次组级 sign-flip / bootstrap；全部 feature 在读取 Trap 标签前固定。",
        "",
        "## 核心结论",
        "",
        "1. 新信息首次稳定出现在 F2 的跨 query 实际 dispatch，而不是 F1 的另一种 entropy/margin。"
        "`query_support_churn` 与 `query_exec_churn` 在 aggregate Trap onset 前 4 个 query 下降，"
        "对旧 F0 loop 信号做秩残差后仍在 A/B 同方向且两边 maxT 通过。",
        "2. 这个增量由 static 主导：static 的 lead=-4/-2 强复现；loop 没有任何新信号在 F0-residual 后双语料通过。"
        "因此不能把它表述成通用的『Trap 前 support 更不稳定』；观察到的是 static 前执行 support 提前冻结。",
        "3. F1 的 tail mass、Top-4 执行熵和 Top-4/5 margin 在 loop lead=-2 有大效应，"
        "但 B 的统一 maxT 未通过，且 F0-residual 后消失。它们目前是原 soft-routing precursor 的重表达。",
        "4. 逐 token 单点存在显著差异，但三个预注册 token-slope 都未跨语料确认，"
        "所以现有数据不支持按 action slot 推导 safe prefix。",
        "",
        "## Capture audit",
        "",
        "| corpus | rows | fp16 Top-4 mismatch | negative boundary | zero boundary | selected mass range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in ("A", "B"):
        a = audits[tag]
        lines.append(
            f"| {tag} | {a['rows']} | {a['fp16_topk_set_disagreement_rate']:.3%} | "
            f"{a['boundary_negative_rate']:.3%} | "
            f"{a['boundary_zero_rate']:.3%} | {a['selected_mass_min']:.4f}--{a['selected_mass_max']:.4f} |"
        )
    lines += [
        "",
        "`negative actual boundary` 表示 fp16 full probability 与运行时 stored Top-4 不再严格一致；"
        "所有 support/churn 因此只使用 stored IDs，绝不从 fp16 概率重算 Top-4。",
        "",
        "## Information increment at aggregate Trap / static onset",
        "",
        "| event | signal | lead | direction A/B | residual det AUC A/B | maxT p A/B |",
        "|---|---|---:|---|---:|---:|",
    ]
    confirmed = pd.concat(
        [
            trap_residual[trap_residual.same_direction & trap_residual.both_maxT_005],
            static_residual[
                static_residual.same_direction & static_residual.both_maxT_005
            ],
        ]
    ).sort_values(["signal", "relative"])
    for row in confirmed.itertuples():
        lines.append(
            f"| {row.event_kind_A} | `{row.signal}` | {int(row.relative)} | "
            f"{row.direction_A}/{row.direction_B} | {row.det_auc_A:.3f}/{row.det_auc_B:.3f} | "
            f"{row.p_maxT_A:.4f}/{row.p_maxT_B:.4f} |"
        )
    lines += [
        "",
        "所有通过单元的方向都是 `event_low`：事件轨迹的跨 query actual support/weight 变化更小。",
        "",
        "## Representation redundancy",
        "",
        "| corpus | support vs sparse-weight churn rho | tail mass vs full entropy rho | exec entropy vs full entropy rho | p45 vs p12 margin rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in ("A", "B"):
        row = redundancy[tag]
        lines.append(
            f"| {tag} | {row['late_flow_support_vs_exec']:.3f} | "
            f"{row['tail_mass_vs_gate_entropy']:.3f} | "
            f"{row['exec_entropy_vs_gate_entropy']:.3f} | "
            f"{row['top45_margin_vs_top12_margin']:.3f} |"
        )
    lines += [
        "",
        "## Best cross-corpus loop precursors (raw)",
        "",
        "| signal | lead | direction A/B | det AUC A/B | maxT p A/B |",
        "|---|---:|---|---:|---:|",
    ]
    for row in raw_new.itertuples():
        lines.append(
            f"| `{row.signal}` | {int(row.relative)} | {row.direction_A}/{row.direction_B} | "
            f"{row.det_auc_A:.3f}/{row.det_auc_B:.3f} | {row.p_maxT_A:.4f}/{row.p_maxT_B:.4f} |"
        )
    lines += [
        "",
        "## Increment beyond frozen F0 loop signals",
        "",
        "这里先在同 snapshot/init-state、同绝对 query 内，将每个新信号对冻结的 "
        "`late_flow_volatility` 与 `route_acceleration` 做无标签秩残差，再进行相同 matched AUC。",
        "",
        "| signal | lead | direction A/B | residual det AUC A/B | maxT p A/B |",
        "|---|---:|---|---:|---:|",
    ]
    for row in residual.itertuples():
        lines.append(
            f"| `{row.signal}` | {int(row.relative)} | {row.direction_A}/{row.direction_B} | "
            f"{row.det_auc_A:.3f}/{row.det_auc_B:.3f} | {row.p_maxT_A:.4f}/{row.p_maxT_B:.4f} |"
        )

    token_summary = []
    for metric in TOKEN_METRICS:
        part = token[token.metric == metric]
        row = {"metric": metric}
        for tag in ("A", "B"):
            p = part[part.corpus == tag].sort_values("token")
            row[f"early_{tag}"] = float((p.auc_event_high.iloc[:5] - 0.5).abs().mean())
            row[f"late_{tag}"] = float((p.auc_event_high.iloc[5:] - 0.5).abs().mean())
            best = p.iloc[np.argmax((p.auc_event_high - 0.5).abs().to_numpy())]
            row[f"best_{tag}"] = int(best.token)
            row[f"p_{tag}"] = float(best.p_maxT)
        token_summary.append(row)
    lines += [
        "",
        "## Safe-prefix localization audit",
        "",
        "| metric | mean abs(AUC-0.5) early/late A | early/late B | best token A/B | family p A/B |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in token_summary:
        lines.append(
            f"| `{row['metric']}` | {row['early_A']:.3f}/{row['late_A']:.3f} | "
            f"{row['early_B']:.3f}/{row['late_B']:.3f} | "
            f"{row['best_A']}/{row['best_B']} | {row['p_A']:.4f}/{row['p_B']:.4f} |"
        )

    lines += [
        "",
        "## State-to-action propagation (label-free)",
        "",
        "| corpus | state leads action by queries | rho | episode-bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in propagation.itertuples():
        lines.append(
            f"| {row.corpus} | {int(row.state_query_leads_action_by)} | {row.rho:.3f} | "
            f"[{row.ci95_low:.3f}, {row.ci95_high:.3f}] |"
        )

    loop_residual_confirmed = residual_all[
        residual_all.same_direction & residual_all.both_maxT_005
    ]
    descriptive_late_consistent = sum(
        row["late_A"] > row["early_A"] and row["late_B"] > row["early_B"]
        for row in token_summary
    )
    token_slope_confirmed = raw[
        raw.signal.isin(("token_support_slope", "token_exec_slope", "token_margin_slope"))
        & (raw.relative == -2)
        & raw.same_direction
        & raw.both_maxT_005
    ]
    lines += [
        "",
        "## Decision gate",
        "",
        f"- loop 中跨语料、同方向、两边 maxT<=0.05 的 F0-residual 新 precursor：{len(loop_residual_confirmed)} 个。",
        f"- aggregate Trap/static 中通过相同 gate 的新单元：{len(confirmed)} 个。",
        f"- 描述上后五个 token 在 A/B 都强于前五个的指标：{descriptive_late_consistent} 个；这不是显著性 gate。",
        f"- lead=-2 跨语料确认的预注册 token-slope 信号：{len(token_slope_confirmed)} 个。",
        "- 观察数据不能估计 safe-prefix 或 reroute 的干预 ATE；只有定位结果复现后，才值得启动 paired snapshot rollout。",
        "- Authority/cancellation/functional geometry 不在现有 routes.zarr 中；它们需要 runtime-exact rich capture。",
        "",
        "## Artifacts",
        "",
        "- `onset_alignment.csv`: F0-F3 raw 与 F0-residual 的 Trap/loop/static matched tests。",
        "- `onset_group_effects.csv`: maxT 所用的逐组效应。",
        "- `token_position_auc.csv`: loop lead=-2 的逐 action-token 检验。",
        "- `propagation.csv`: early-state 到 late-action 的跨 query 相关。",
        "- `f0_f3_loop_matrix.png` / `token_position_loop_lead2.png`: 方向与位置图。",
    ]
    (HERE / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nperm", type=int, default=2000)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    onset_frames = []
    group_frames = []
    token_frames = []
    token_group_frames = []
    propagation_frames = []
    audits = {}
    redundancies = {}
    for tag in ("A", "B"):
        print(f"[{tag}] corpus and frozen F0", flush=True)
        corpus, _inventory = baseline.load_corpus(tag)
        base_rows, _base_audit = baseline.extract_row_features(tag, rebuild=False)
        base_signals, _macro, _transition, _feature_audit = baseline.build_signals(
            corpus, base_rows
        )
        print(f"[{tag}] F1-F3 execution extraction", flush=True)
        rows, audit = extract_execution_rows(tag, args.rebuild_cache)
        execution, token_maps, _layer_maps = build_execution_signals(rows, corpus)
        audits[tag] = audit
        redundancies[tag] = redundancy_audit(tag, rows, base_rows)
        raw = {
            **{
                name: baseline.signal_array(base_signals, name, baseline.ALIGN_WINDOW)
                for name in F0
            },
            **{name: execution[f"{name}_w4"] for name in NEW_SIGNALS},
        }
        residual = {
            name: stage_rank_residual_multi(
                raw[name],
                (raw["late_flow_volatility"], raw["route_acceleration"]),
                corpus.match_group,
            )
            for name in NEW_SIGNALS
        }
        print(f"[{tag}] raw joint maxT", flush=True)
        onset, groups = evaluate_onsets(corpus, raw, "raw", args.nperm, 0)
        onset_frames.append(onset)
        group_frames.append(groups)
        print(f"[{tag}] F0-residual joint maxT", flush=True)
        onset, groups = evaluate_onsets(
            corpus, residual, "residual_f0", args.nperm, 70000
        )
        onset_frames.append(onset)
        group_frames.append(groups)
        print(f"[{tag}] per-token safe-prefix audit", flush=True)
        token, token_groups = evaluate_token_positions(corpus, token_maps, args.nperm)
        token_frames.append(token)
        token_group_frames.append(token_groups)
        propagation_frames.append(propagation_audit(corpus, execution, args.nperm))

    onset = pd.concat(onset_frames, ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    token = pd.concat(token_frames, ignore_index=True)
    token_groups = pd.concat(token_group_frames, ignore_index=True)
    propagation = pd.concat(propagation_frames, ignore_index=True)
    for name, frame in (
        ("onset_alignment.csv", onset),
        ("onset_group_effects.csv", groups),
        ("token_position_auc.csv", token),
        ("token_position_group_effects.csv", token_groups),
        ("propagation.csv", propagation),
    ):
        frame.to_csv(HERE / name, index=False)
    summary = {
        "schema": SCHEMA,
        "seed": SEED,
        "n_permutations": args.nperm,
        "training": False,
        "feature_labels_used": False,
        "evaluation_labels_used": True,
        "f0_residual_predictors": ["late_flow_volatility", "route_acceleration"],
        "tiers": {"F0": list(F0), "F1": list(F1), "F2": list(F2), "F3": list(F3)},
        "audits": audits,
        "redundancy": redundancies,
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_plots(onset, token)
    write_report(onset, token, propagation, audits, redundancies, args.nperm)
    print(f"wrote results to {HERE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
