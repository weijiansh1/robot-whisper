#!/usr/bin/env python3
"""Kolmogorov-inspired, train-free analyses of HiMoE-VLA routing dynamics.

The new scores in this file never fit an outcome predictor.  Labels are used
only after scores have been frozen, for grouped AUCs and permutation tests.
Quantiles and alarm thresholds are calibration constants, not learned model
weights.  The fixed routing slice is the established baseline:

    deep HB layers 12-15 x action tokens 1-10 x denoise d9 x 32 experts.

Experiments:
  1. A deterministic four-state Markov view of W8 route mobility.
  2. Finite-window symbolic entropy rate and recurrence statistics.
  3. A normalized DEFLATE-length proxy for algorithmic compressibility.
  4. Hellinger-space quantization and layer/token measurement budgets.
  5. A B-only random-coordinate hidden-state budget control.
  6. Cross-layer zero-lag vs lagged correlations (cascade check).
  7. Leave-one-group-out calibrated online operating points.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path
import sys
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis_agg_axis"))
import lib as agglib  # noqa: E402


TS = (20, 25, 30)
T_ONLINE_LO = 8
T_ONLINE_HI = 35
WINDOW = 8
LAYERS = (12, 13, 14, 15)
TOKENS = tuple(range(1, 11))
EXPERTS = 32
TARGET_FA = {"A": 0.137, "B": 0.051}
SEED = 20260903


def cell_hellinger(P: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cellwise Hellinger distance to the previous chunk, shape (B,T,40)."""
    B, T, C, _ = P.shape
    out = np.full((B, T, C), np.nan, np.float32)
    for t in range(1, T):
        ok = valid[:, t]
        if not ok.any():
            continue
        bc = np.sqrt(P[ok, t] * P[ok, t - 1]).sum(-1)
        out[ok, t] = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
    return out


def rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing mean along axis 1; emit NaN unless the full window exists."""
    B, T = x.shape[:2]
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(width - 1, T):
        seg = x[:, t - width + 1:t + 1]
        axes = tuple(range(1, seg.ndim))
        full = np.isfinite(seg).all(axis=axes)
        if seg.ndim == 2:
            out[full, t] = seg[full].mean(1)
        else:
            out[full, t] = seg[full].mean(1)
    return out


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def symbolic_features(
    dcell: np.ndarray,
    valid: np.ndarray,
    width: int = WINDOW,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fixed four-symbol dynamics from early, label-free cellwise quartiles."""
    B, T, C = dcell.shape
    edges = np.nanquantile(dcell[:, 1:13], (0.25, 0.50, 0.75), axis=(0, 1))
    symbols = (dcell[..., None] > edges.T).sum(-1).astype(np.uint8)
    names = (
        "symbol_marginal_H",
        "symbol_conditional_H",
        "symbol_entropy_ratio",
        "symbol_novelty",
        "symbol_nonperiod2",
    )
    out = {name: np.full((B, T), np.nan, np.float64) for name in names}
    for t in range(width, min(T, T_ONLINE_HI)):
        lo = t - width + 1
        alive = np.where(valid[:, t] & valid[:, lo])[0]
        for b in alive:
            x = symbols[b, lo:t + 1]
            hm = entropy_from_counts(np.bincount(x.ravel(), minlength=4))
            trans = np.zeros((4, 4), np.int64)
            np.add.at(trans, (x[:-1].ravel(), x[1:].ravel()), 1)
            total = trans.sum()
            hc = 0.0
            for state in range(4):
                n = trans[state].sum()
                if n:
                    hc += (n / total) * entropy_from_counts(trans[state])
            periodic2 = ((x[2:] == x[:-2]) & (x[2:] != x[1:-1])).mean()
            out["symbol_marginal_H"][b, t] = hm
            out["symbol_conditional_H"][b, t] = hc
            out["symbol_entropy_ratio"][b, t] = hc / hm if hm > 0 else 0.0
            out["symbol_novelty"][b, t] = (x[1:] != x[:-1]).mean()
            out["symbol_nonperiod2"][b, t] = 1.0 - periodic2
    return out, edges


def rank_residualise_multi(
    scores: np.ndarray,
    bases: list[np.ndarray],
    groups: np.ndarray,
) -> np.ndarray:
    """Within-group rank OLS residual against one or more fixed controls."""
    out = np.full(len(scores), np.nan, np.float64)
    for group in np.unique(groups):
        mask = (groups == group) & np.isfinite(scores)
        for base in bases:
            mask &= np.isfinite(base)
        idx = np.where(mask)[0]
        if len(idx) < len(bases) + 2:
            continue
        y = agglib._avg_rank(scores[idx])
        y -= y.mean()
        X = np.stack([agglib._avg_rank(base[idx]) for base in bases], axis=1)
        X -= X.mean(0, keepdims=True)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        r[np.abs(r) < 1e-8 * len(idx)] = 0.0
        out[idx] = r
    return out


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, or NaN when either finite input is constant."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.ptp(a[good]) == 0 or np.ptp(b[good]) == 0:
        return np.nan
    return float(spearmanr(a[good], b[good]).statistic)


def score_summary(
    scores: np.ndarray,
    success: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray | None = None,
    matched_bases: list[np.ndarray] | None = None,
) -> dict[str, float | int]:
    auc, npairs = agglib.within_group_auc(scores, success, groups)
    row: dict[str, float | int] = {
        "auc_success": float(auc),
        "det_auc": float(max(auc, 1.0 - auc)),
        "npairs": int(npairs),
    }
    if baseline is not None:
        res = agglib.rank_residualise(scores, baseline, groups)
        ar, _ = agglib.within_group_auc(res, success, groups)
        rho = safe_spearman(scores, baseline)
        row.update(
            auc_resid_w8=float(ar),
            det_resid_w8=float(max(ar, 1.0 - ar)),
            rho_w8=float(rho),
        )
    if matched_bases:
        res = rank_residualise_multi(scores, matched_bases, groups)
        ar, _ = agglib.within_group_auc(res, success, groups)
        row.update(
            auc_resid_matched=float(ar),
            det_resid_matched=float(max(ar, 1.0 - ar)),
        )
    return row


def group_robustness_rows(
    tag: str,
    family: str,
    metric: str,
    window: int,
    t: int,
    scores: np.ndarray,
    matched_bases: list[np.ndarray],
    meta: pd.DataFrame,
) -> list[dict[str, object]]:
    """Per-group raw and matched-residual AUCs for heterogeneity checks."""
    success = meta["success"].to_numpy(bool)
    groups = meta["group"].to_numpy()
    residual = rank_residualise_multi(scores, matched_bases, groups)
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        n_success = int(success[mask].sum())
        n_failure = int((~success[mask]).sum())
        if not n_success or not n_failure:
            continue
        local_group = np.zeros(mask.sum(), np.int8)
        auc_raw, npairs = agglib.within_group_auc(
            scores[mask], success[mask], local_group
        )
        auc_residual, _ = agglib.within_group_auc(
            residual[mask], success[mask], local_group
        )
        rows.append(
            dict(
                corpus=tag,
                family=family,
                metric=metric,
                window=window,
                t=t,
                group=group,
                n_success=n_success,
                n_failure=n_failure,
                npairs=int(npairs),
                auc_raw=float(auc_raw),
                auc_residual=float(auc_residual),
            )
        )
    return rows


def grouped_ranks(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    ranks = np.full_like(scores, np.nan, dtype=np.float64)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        for j in range(scores.shape[1]):
            good = idx[np.isfinite(scores[idx, j])]
            ranks[good, j] = agglib._avg_rank(scores[good, j])
    return ranks


def family_permutation(
    scores: np.ndarray,
    success: np.ndarray,
    groups: np.ndarray,
    nperm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pointwise and maxT p-values for direction-free grouped AUCs."""
    ranks = grouped_ranks(scores, groups)
    obs = np.array(
        [max(agglib.within_group_auc(scores[:, j], success, groups)[0],
             1.0 - agglib.within_group_auc(scores[:, j], success, groups)[0])
         for j in range(scores.shape[1])]
    )
    rng = np.random.default_rng(seed)
    null = np.empty((nperm, scores.shape[1]), np.float64)
    valid_groups = []
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        npos = int(success[idx].sum())
        nneg = len(idx) - npos
        if npos and nneg:
            valid_groups.append((idx, npos, nneg))
    correction = sum(npos * (npos + 1) / 2 for _, npos, _ in valid_groups)
    denom = sum(npos * nneg for _, npos, nneg in valid_groups)
    for p in range(nperm):
        rank_sum = np.zeros(scores.shape[1], np.float64)
        for idx, npos, _ in valid_groups:
            chosen = rng.choice(idx, npos, replace=False)
            rank_sum += np.nansum(ranks[chosen], axis=0)
        auc = (rank_sum - correction) / denom
        null[p] = np.maximum(auc, 1.0 - auc)
    null_max = null.max(1)
    p_point = (1 + (null >= obs).sum(0)) / (nperm + 1)
    p_family = (1 + (null_max[:, None] >= obs).sum(0)) / (nperm + 1)
    return p_point, p_family, float(np.quantile(null_max, 0.95))


def markov_experiment(
    tag: str,
    mobility_w8: np.ndarray,
    valid: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Four fixed mobility bins, transition structure, and trap hitting score."""
    success = meta["success"].to_numpy(bool)
    groups = meta["group"].to_numpy()
    edges = np.nanquantile(mobility_w8[:, 8:13], (0.25, 0.50, 0.75))
    state = (mobility_w8[..., None] > edges).sum(-1).astype(np.uint8)
    counts = np.zeros((4, 4), np.int64)
    counts_by_outcome = {"success": np.zeros((4, 4), np.int64),
                         "failure": np.zeros((4, 4), np.int64)}
    for b in range(len(meta)):
        outcome = "success" if success[b] else "failure"
        top = min(T_ONLINE_HI - 1, int(meta.iloc[b]["T"]) - 1)
        for t in range(8, top):
            i, j = int(state[b, t]), int(state[b, t + 1])
            counts[i, j] += 1
            counts_by_outcome[outcome][i, j] += 1
    P = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    trans_rows = []
    for i in range(4):
        for j in range(4):
            trans_rows.append(
                dict(corpus=tag, from_state=i, to_state=j, count=int(counts[i, j]),
                     probability=float(P[i, j]))
            )

    struct_rows = []
    for outcome, mask in (("all", np.ones(len(meta), bool)),
                          ("success", success), ("failure", ~success)):
        C = counts if outcome == "all" else counts_by_outcome[outcome]
        PP = C / np.maximum(C.sum(1, keepdims=True), 1)
        state_window = state[mask, 8:T_ONLINE_HI]
        valid_window = valid[mask, 8:T_ONLINE_HI]
        observed = state_window[valid_window]
        for s in range(4):
            ps = float(PP[s, s])
            struct_rows.append(
                dict(
                    corpus=tag,
                    outcome=outcome,
                    state=s,
                    edge_lo=float(-np.inf if s == 0 else edges[s - 1]),
                    edge_hi=float(np.inf if s == 3 else edges[s]),
                    self_transition=ps,
                    expected_geometric_dwell=float(1.0 / max(1.0 - ps, 1e-12)),
                    occupancy=float((observed == s).mean()),
                )
            )

    auc_rows = []
    hit = np.zeros(4, np.float64)
    hit[0] = 1.0
    for horizon in range(1, 11):
        nxt = hit.copy()
        nxt[1:] = P[1:, 0] + P[1:, 1:] @ hit[1:]
        hit = nxt
        if horizon not in (1, 3, 5, 10):
            continue
        for t in TS:
            risk = hit[state[:, t]]
            scores = -risk  # high score means lower basin-hitting risk / success
            row = dict(corpus=tag, horizon=horizon, t=t)
            row.update(score_summary(scores, success, groups, mobility_w8[:, t]))
            row["hit_prob_state0"] = float(hit[0])
            row["hit_prob_state1"] = float(hit[1])
            row["hit_prob_state2"] = float(hit[2])
            row["hit_prob_state3"] = float(hit[3])
            auc_rows.append(row)
    return pd.DataFrame(trans_rows), pd.DataFrame(struct_rows), pd.DataFrame(auc_rows)


def deflate_length(payload: bytes) -> int:
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
    return len(compressor.compress(payload) + compressor.flush())


def compression_features(
    P: np.ndarray,
    valid: np.ndarray,
) -> dict[tuple[int, str], np.ndarray]:
    """Normalized compressed lengths of 8-bit Hellinger-embedded route streams."""
    B, T, C, E = P.shape
    code = np.rint(np.sqrt(P) * 255.0).astype(np.uint8)
    out: dict[tuple[int, str], np.ndarray] = {}
    targets = {8: tuple(range(T_ONLINE_LO, min(T, T_ONLINE_HI))), 20: TS}
    for width, times in targets.items():
        raw = np.full((B, T), np.nan, np.float64)
        delta = np.full((B, T), np.nan, np.float64)
        for t in times:
            if t < width:
                continue
            alive = np.where(valid[:, t] & valid[:, t - width])[0]
            for b in alive:
                # Coordinate-major order makes each temporal trace contiguous.
                q = code[b, t - width:t + 1].transpose(1, 2, 0)
                raw[b, t] = deflate_length(q.tobytes()) / q.size
                dq = ((q[..., 1:].astype(np.int16) - q[..., :-1].astype(np.int16))
                      & 0xFF).astype(np.uint8)
                delta[b, t] = deflate_length(dq.tobytes()) / dq.size
        out[(width, "raw_ncl")] = raw
        out[(width, "delta_ncl")] = delta
    return out


def quantized_probability(
    root_p: np.ndarray,
    bits: int,
) -> np.ndarray:
    levels = (1 << bits) - 1
    code = np.rint(root_p * levels).astype(np.uint16)
    q = code.astype(np.float32)
    q *= q
    norm = q.sum(-1, keepdims=True)
    zero = norm <= 0
    q /= np.maximum(norm, 1e-12)
    if zero.any():
        q = np.where(zero, np.float32(1.0 / EXPERTS), q)
    return q


def reconstruction_hellinger(P: np.ndarray, Q: np.ndarray, valid: np.ndarray) -> float:
    total = 0.0
    count = 0
    for t in range(P.shape[1]):
        ok = valid[:, t]
        if not ok.any():
            continue
        d = np.sqrt(np.clip(1.0 - np.sqrt(P[ok, t] * Q[ok, t]).sum(-1), 0.0, 1.0))
        total += float(d.sum())
        count += d.size
    return total / count


def route_budget_experiment(
    tag: str,
    P: np.ndarray,
    valid: np.ndarray,
    dcell: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    success = meta["success"].to_numpy(bool)
    groups = meta["group"].to_numpy()
    full_score = dcell[:, 23:31].mean((1, 2))
    root_p = np.sqrt(P)
    quant_rows = []
    d6 = None
    for bits in (1, 2, 3, 4, 5, 6, 8, 16):
        if bits == 16:
            score = full_score
            recon = 0.0
        else:
            Q = quantized_probability(root_p, bits)
            dq = cell_hellinger(Q, valid)
            score = dq[:, 23:31].mean((1, 2))
            recon = reconstruction_hellinger(P, Q, valid)
            if bits == 6:
                d6 = dq.copy()
            del Q, dq
            gc.collect()
        auc, npairs = agglib.within_group_auc(score, success, groups)
        rho = safe_spearman(score, full_score)
        quant_rows.append(
            dict(
                corpus=tag,
                bits_per_probability=bits,
                bits_per_chunk=40 * EXPERTS * bits,
                auc_success=float(auc),
                det_auc=float(max(auc, 1 - auc)),
                rho_full=float(rho) if np.isfinite(rho) else np.nan,
                score_mae=float(np.mean(np.abs(score - full_score))),
                reconstruction_hellinger=float(recon),
                npairs=int(npairs),
            )
        )
    assert d6 is not None

    cell_score = d6[:, 23:31].mean(1)
    variants: list[tuple[str, str, np.ndarray]] = []
    for i, layer in enumerate(LAYERS):
        for j, token in enumerate(TOKENS):
            variants.append(("one_cell", f"L{layer}_T{token}", np.array([i * 10 + j])))
    for j, token in enumerate(TOKENS):
        variants.append(("one_token", f"T{token}_all_layers", np.arange(4) * 10 + j))
    for i, layer in enumerate(LAYERS):
        variants.append(("one_layer", f"L{layer}_all_tokens", np.arange(i * 10, (i + 1) * 10)))
    for i, j in itertools.combinations(range(4), 2):
        idx = np.r_[np.arange(i * 10, (i + 1) * 10), np.arange(j * 10, (j + 1) * 10)]
        variants.append(("two_layers", f"L{LAYERS[i]}+L{LAYERS[j]}", idx))
    variants.append(("all", "all_40", np.arange(40)))
    cell_rows = []
    for family, variant, idx in variants:
        score = cell_score[:, idx].mean(1)
        row = dict(
            corpus=tag,
            family=family,
            variant=variant,
            n_cells=len(idx),
            bits_per_chunk=len(idx) * EXPERTS * 6,
        )
        row.update(score_summary(score, success, groups, full_score))
        cell_rows.append(row)

    # A separate telemetry question: after online Hellinger computation, how
    # many bits are needed to retain the scalar per-chunk movement signal?
    movement = np.full(dcell.shape[:2], np.nan, np.float64)
    for t in range(1, dcell.shape[1]):
        ok = valid[:, t]
        movement[ok, t] = dcell[ok, t].mean(1)
    lo, hi = np.nanquantile(movement[:, 1:13], (0.001, 0.999))
    telemetry_rows = []
    for bits in range(1, 9):
        levels = (1 << bits) - 1
        q = np.rint(np.clip((movement - lo) / (hi - lo), 0.0, 1.0) * levels)
        q = q / levels * (hi - lo) + lo
        q[~np.isfinite(movement)] = np.nan
        score = q[:, 23:31].mean(1)
        auc, npairs = agglib.within_group_auc(score, success, groups)
        telemetry_rows.append(
            dict(
                corpus=tag,
                bits_per_chunk=bits,
                calibration_lo=float(lo),
                calibration_hi=float(hi),
                auc_success=float(auc),
                det_auc=float(max(auc, 1 - auc)),
                rho_full=safe_spearman(score, full_score),
                npairs=int(npairs),
            )
        )
    del root_p, d6
    gc.collect()
    return pd.DataFrame(quant_rows), pd.DataFrame(cell_rows), pd.DataFrame(telemetry_rows)


def hidden_budget_experiment(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """B-only, label-free random coordinate subsets of d9 hidden movement."""
    import zarr

    run = (ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
           "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
    hidden_path = run / "server/hidden.zarr"
    route_path = run / "server/routes.zarr"
    hidden = zarr.open_group(str(hidden_path), mode="r")["hb_hidden"]
    rows = np.concatenate(
        [np.arange(int(row.row0) + 22, int(row.row0) + 31) for _, row in meta.iterrows()]
    )
    X = np.asarray(hidden.oindex[rows, 4:8, 9, 1:11, :], np.float32)
    X = X.reshape(len(meta), 9, 40, 1024)
    current, previous = X[:, 1:], X[:, :-1]
    success = meta["success"].to_numpy(bool)
    groups = meta["group"].to_numpy()
    dims = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    rng = np.random.default_rng(SEED)
    rows_out = []
    for repeat in range(4):
        permutation = rng.permutation(1024)
        numerator = np.zeros(current.shape[:-1], np.float32)
        denominator = np.zeros_like(numerator)
        for count, coord in enumerate(permutation, start=1):
            delta = current[..., coord] - previous[..., coord]
            numerator += delta * delta
            denominator += previous[..., coord] * previous[..., coord]
            if count not in dims:
                continue
            score = np.sqrt(numerator / np.maximum(denominator, 1e-12)).mean((1, 2))
            auc, npairs = agglib.within_group_auc(score, success, groups)
            rows_out.append(
                dict(
                    corpus="B",
                    metric="relative_l2",
                    repeat=repeat,
                    coordinates_per_cell=count,
                    scalar_coordinates_per_chunk=40 * count,
                    bits_per_chunk=40 * count * 16,
                    auc_success=float(auc),
                    det_auc=float(max(auc, 1 - auc)),
                    npairs=int(npairs),
                )
            )
    # The final repeat has visited every coordinate, so these accumulators are
    # already the exact full-dimensional squared norms.
    rel = np.sqrt(numerator / np.maximum(denominator, 1e-12)).mean((1, 2))
    cos = 1.0 - (current * previous).sum(-1) / np.sqrt(
        np.maximum((current ** 2).sum(-1) * (previous ** 2).sum(-1), 1e-12)
    )
    auc_cos, npairs = agglib.within_group_auc(cos.mean((1, 2)), success, groups)
    rows_out.append(
        dict(corpus="B", metric="cosine_full", repeat=-1, coordinates_per_cell=1024,
             scalar_coordinates_per_chunk=40960, bits_per_chunk=655360,
             auc_success=float(auc_cos), det_auc=float(max(auc_cos, 1 - auc_cos)),
             npairs=int(npairs))
    )
    auc_rel, _ = agglib.within_group_auc(rel, success, groups)
    assert abs(auc_rel - pd.DataFrame(rows_out).query(
        "metric == 'relative_l2' and coordinates_per_cell == 1024"
    )["auc_success"].iloc[0]) < 1e-12

    def directory_bytes(path: Path) -> int:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    disk = {"hidden_zarr_bytes": directory_bytes(hidden_path),
            "routes_zarr_bytes": directory_bytes(route_path)}
    del X, current, previous, numerator, denominator, rel, cos
    gc.collect()
    return pd.DataFrame(rows_out), disk


def cascade_experiment(
    tag: str,
    dcell: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = meta["success"].to_numpy(bool)
    groups = meta["group"].to_numpy()
    layer_raw = dcell.reshape(dcell.shape[0], dcell.shape[1], 4, 10).mean(-1)
    layer_w8 = rolling_mean(layer_raw, WINDOW)
    auc_rows = []
    for t in range(T_ONLINE_LO, min(layer_w8.shape[1], T_ONLINE_HI)):
        for li, layer in enumerate(LAYERS):
            auc, npairs = agglib.within_group_auc(layer_w8[:, t, li], success, groups)
            auc_rows.append(dict(corpus=tag, t=t, layer=layer, auc_success=float(auc),
                                 det_auc=float(max(auc, 1 - auc)), npairs=int(npairs)))
    lag_rows = []
    for i, j in itertools.combinations(range(4), 2):
        correlations = []
        for lag in range(-3, 4):
            if lag >= 0:
                a = layer_w8[:, 8:35 - lag, i].ravel()
                b = layer_w8[:, 8 + lag:35, j].ravel()
            else:
                a = layer_w8[:, 8 - lag:35, i].ravel()
                b = layer_w8[:, 8:35 + lag, j].ravel()
            good = np.isfinite(a) & np.isfinite(b)
            correlations.append(float(np.corrcoef(a[good], b[good])[0, 1]))
        best = int(np.arange(-3, 4)[np.argmax(correlations)])
        for lag, corr in zip(range(-3, 4), correlations):
            lag_rows.append(dict(corpus=tag, layer_a=LAYERS[i], layer_b=LAYERS[j],
                                 lag=lag, correlation=corr, best_lag=best))
    return pd.DataFrame(auc_rows), pd.DataFrame(lag_rows)


def k_consecutive_statistic(metric: np.ndarray, k: int = 3) -> np.ndarray:
    """A crossing statistic for k consecutive low metric values."""
    out = np.full_like(metric, np.nan, dtype=np.float64)
    for t in range(k - 1, metric.shape[1]):
        seg = metric[:, t - k + 1:t + 1]
        ok = np.isfinite(seg).all(1)
        out[ok, t] = -seg[ok].max(1)
    return out


def alarm_times(stat: np.ndarray, lengths: np.ndarray, bound: float) -> np.ndarray:
    alarm = np.full(len(lengths), -1, np.int16)
    for b, length in enumerate(lengths):
        hi = min(T_ONLINE_HI, int(length))
        good = np.flatnonzero(
            np.isfinite(stat[b, T_ONLINE_LO:hi])
            & (stat[b, T_ONLINE_LO:hi] >= bound)
        )
        if len(good):
            alarm[b] = T_ONLINE_LO + int(good[0])
    return alarm


def calibrate_peak_bound(
    stat: np.ndarray,
    lengths: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> float:
    peaks = []
    for b in np.where(mask)[0]:
        hi = min(T_ONLINE_HI, int(lengths[b]))
        x = stat[b, T_ONLINE_LO:hi]
        x = x[np.isfinite(x)]
        if len(x):
            peaks.append(x.max())
    if not peaks:
        return np.inf
    peaks = np.sort(np.asarray(peaks))[::-1]
    k = int(np.floor(alpha * len(peaks)))
    if k >= len(peaks):
        return -np.inf
    return float(np.nextafter(peaks[k], np.inf))


def logo_alarm(
    metric: np.ndarray,
    meta: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, dict[int, float]]:
    stat = k_consecutive_statistic(metric, 3)
    groups = meta["group"].to_numpy()
    success = meta["success"].to_numpy(bool)
    lengths = meta["T"].to_numpy(int)
    alarm = np.full(len(meta), -1, np.int16)
    bounds = {}
    for group in np.unique(groups):
        train_success = (groups != group) & success
        bound = calibrate_peak_bound(stat, lengths, train_success, alpha)
        all_alarm = alarm_times(stat, lengths, bound)
        test = groups == group
        alarm[test] = all_alarm[test]
        bounds[int(group)] = bound
    return alarm, bounds


def online_experiment(
    tag: str,
    metrics: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    success = meta["success"].to_numpy(bool)
    target = TARGET_FA[tag]
    onset = None
    if tag == "A":
        path = (ROOT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/"
                "analysis_trap_onset/trap_onset.csv")
        table = pd.read_csv(path).set_index("episode_id")
        onset = table["loop_onset_query"].reindex(meta["episode_id"]).to_numpy(float)
        onset[onset < 0] = np.nan
    rows = []
    for name, metric in metrics.items():
        best = None
        for alpha in np.arange(0.005, 0.350, 0.005):
            alarm, bounds = logo_alarm(metric, meta, float(alpha))
            fa = float((alarm[success] >= 0).mean())
            key = (abs(fa - target), fa > target, alpha)
            if best is None or key < best[0]:
                best = (key, alpha, alarm, bounds, fa)
        assert best is not None
        _, alpha, alarm, bounds, fa = best
        fired_failure = (alarm >= 0) & ~success
        times = alarm[fired_failure]
        row: dict[str, object] = dict(
            corpus=tag,
            metric=name,
            nominal_alpha=float(alpha),
            target_fa=target,
            achieved_fa=fa,
            detection=float(fired_failure.sum() / (~success).sum()),
            median_alarm_t=float(np.median(times)) if len(times) else np.nan,
            bounds=json.dumps(bounds, sort_keys=True),
        )
        if onset is not None:
            known = fired_failure & np.isfinite(onset)
            delay = alarm[known] - onset[known]
            row.update(
                n_known_onset_alarm=int(known.sum()),
                fraction_before_onset=float((delay < 0).mean()) if len(delay) else np.nan,
                median_delay=float(np.median(delay)) if len(delay) else np.nan,
                delay_p25=float(np.percentile(delay, 25)) if len(delay) else np.nan,
                delay_p75=float(np.percentile(delay, 75)) if len(delay) else np.nan,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(
    timecourse: pd.DataFrame,
    quant: pd.DataFrame,
    cells: pd.DataFrame,
    hidden: pd.DataFrame | None,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"A": "#1f6f8b", "B": "#c54f3d"}
    for tag in ("A", "B"):
        d = timecourse[(timecourse.corpus == tag) &
                       (timecourse.metric.isin(["route_mobility_w8",
                                                "symbol_conditional_H",
                                                "compression_delta_w8"]))]
        for metric, ls in (("route_mobility_w8", "-"),
                           ("symbol_conditional_H", "--"),
                           ("compression_delta_w8", ":")):
            x = d[d.metric == metric]
            axes[0].plot(x.t, x.auc_success, color=colors[tag], linestyle=ls,
                         label=f"{tag} {metric.replace('_w8', '')}")
    axes[0].axhline(0.5, color="#555555", linewidth=0.8)
    axes[0].set(xlabel="query t", ylabel="within-group AUC (success high)",
                title="Signal arrives late")
    axes[0].legend(fontsize=7, ncol=2)

    for tag in ("A", "B"):
        d = quant[quant.corpus == tag]
        axes[1].plot(d.bits_per_chunk, d.auc_success, "o-", color=colors[tag], label=tag)
    axes[1].set_xscale("log", base=2)
    axes[1].set(xlabel="routing payload bits/chunk", ylabel="AUC",
                title="Hellinger-space quantization")
    axes[1].legend()

    summary = cells.groupby(["corpus", "family"], as_index=False).agg(
        bits_per_chunk=("bits_per_chunk", "first"), auc_success=("auc_success", "median")
    )
    order = ["one_cell", "one_token", "one_layer", "two_layers", "all"]
    for tag in ("A", "B"):
        d = summary[summary.corpus == tag].set_index("family").reindex(order).dropna()
        axes[2].plot(d.bits_per_chunk, d.auc_success, "o-", color=colors[tag], label=tag)
    if hidden is not None and len(hidden):
        h = hidden[hidden.metric == "relative_l2"].groupby(
            "bits_per_chunk", as_index=False
        ).auc_success.median()
        axes[2].plot(h.bits_per_chunk, h.auc_success, "s--", color="#6f5a7e",
                     label="B hidden random coords")
    axes[2].set_xscale("log", base=2)
    axes[2].set(xlabel="input bits/chunk", ylabel="AUC",
                title="Measurement budget (descriptive)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "overview.png", dpi=180)
    plt.close(fig)


def fmt(x: float, digits: int = 3) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{digits}f}"


def write_report(
    dynamics: pd.DataFrame,
    markov_struct: pd.DataFrame,
    markov_auc: pd.DataFrame,
    compression: pd.DataFrame,
    quant: pd.DataFrame,
    cells: pd.DataFrame,
    telemetry: pd.DataFrame,
    group_robustness: pd.DataFrame,
    cascade_lag: pd.DataFrame,
    online: pd.DataFrame,
    hidden: pd.DataFrame | None,
    disk: dict[str, int] | None,
    nperm: int,
) -> None:
    def dyn(tag: str, metric: str, t: int = 30) -> pd.Series:
        return dynamics[(dynamics.corpus == tag) & (dynamics.metric == metric) &
                        (dynamics.t == t)].iloc[0]

    def mk(tag: str, outcome: str, state: int) -> pd.Series:
        return markov_struct[(markov_struct.corpus == tag) &
                             (markov_struct.outcome == outcome) &
                             (markov_struct.state == state)].iloc[0]

    lines = [
        "# Kolmogorov 启发的 MoE 动力学实验（train-free）",
        "",
        "> 运行日期：2026-09-03。新统计量不训练分类器、不拟合 PCA/k-means、也不使用结局标签构造特征；",
        "> 标签只在特征冻结后用于组内 AUC 与置换检验。分位点和报警阈值属于校准。",
        "",
        "## 口径",
        "",
        "- A：rolling-star，352 条分支，117 成功 / 235 失败，4 个 worker（3 个混合结局组）。",
        "- B：SCENE8 right-16x32，512 条轨迹，296 成功 / 216 失败，16 个初态（13 个混合结局组）。",
        "- 路由切片：深层 12–15、动作 token 1–10、`d9`（从 0 编号的最后一个去噪步）、32 专家软概率。",
        "- 基线：相邻 chunk 逐 `(layer, token)` 算 Hellinger，再对 40 格和最近 8 chunk 取均值。",
        "- AUC 全部按组内成功–失败配对数合并；`t=30` 时两批轨迹全部仍在场。",
        "",
        "## 1. Markov / trap basin",
        "",
        "把 W8 路由变化率按早期 `t=8..12` 的无标签四分位点离散成 4 状态，state 0 是低变化，state 3 是高变化。",
        "",
        "| corpus | P(low→low) | P(high→high) | low 占比 success / failure | P(low→low) success / failure |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in ("A", "B"):
        a0, a3 = mk(tag, "all", 0), mk(tag, "all", 3)
        s0, f0 = mk(tag, "success", 0), mk(tag, "failure", 0)
        lines.append(
            f"| {tag} | {a0.self_transition:.3f} | {a3.self_transition:.3f} | "
            f"{s0.occupancy:.3f} / {f0.occupancy:.3f} | "
            f"{s0.self_transition:.3f} / {f0.self_transition:.3f} |"
        )
    lines += [
        "",
        "低变化状态确实在失败中更常驻留，但高变化状态同样很粘；因此“低变化盆地”是把旧变化率阈值化后得到的宏状态，",
        "不是从完整路由状态中发现的独立吸引子。已有 20 组 PCA/k-means Markov 检查中，15/20 的瞬态图只有一个完整强连通类。",
        "",
        "5-step hitting probability 的组内 AUC：",
        "",
        "| corpus | t20 | t25 | t30 | 连续 W8 基线 t30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in ("A", "B"):
        ma = markov_auc[(markov_auc.corpus == tag) & (markov_auc.horizon == 5)].set_index("t")
        base = dyn(tag, "route_mobility_w8", 30)
        lines.append(f"| {tag} | {ma.loc[20].auc_success:.3f} | {ma.loc[25].auc_success:.3f} | "
                     f"{ma.loc[30].auc_success:.3f} | {base.auc_success:.3f} |")
    lines += [
        "",
        "结论：Markov hitting probability 没有产生早期信号，并因四状态量化而弱于连续基线。",
        "",
        "## 2. Symbolic entropy rate",
        "",
        "每个路由格的 chunk-to-chunk Hellinger 变化按早期无标签四分位点编码成 4 个符号；在 8-step 窗内估计",
        "`H(symbol)`、`H(symbol_t | symbol_{t-1})`、二者比值、符号 novelty 和二周期率。",
        "",
        "| metric @ t30 | A raw / residual | B raw / residual |",
        "|---|---:|---:|",
    ]
    for metric in ("symbol_marginal_H", "symbol_conditional_H", "symbol_entropy_ratio",
                   "symbol_novelty", "symbol_nonperiod2"):
        a, b = dyn("A", metric), dyn("B", metric)
        lines.append(f"| `{metric}` | {a.auc_success:.3f} / {a.auc_resid_w8:.3f} | "
                     f"{b.auc_success:.3f} / {b.auc_resid_w8:.3f} |")
    b_ratio20 = dyn("B", "symbol_entropy_ratio", 20)
    a_ratio20 = dyn("A", "symbol_entropy_ratio", 20)
    b_ratio_groups = group_robustness.query(
        "corpus == 'B' and family == 'symbolic' and "
        "metric == 'symbol_entropy_ratio' and t == 20"
    )
    b_ratio_positive = int((b_ratio_groups.auc_residual > 0.5).sum())
    lines += [
        "",
        f"置换检验使用 {nperm} 次组内标签置换，并对 5 指标 × 3 时点做 maxT。条件熵有原始区分度，"
        "但对 W8 变化率做组内秩残差后接近机会线；二周期指标也没有跨语料同向增量。",
        f"一个未跨语料复现的早期候选是 B 的 `t20` entropy ratio：raw AUC={b_ratio20.auc_success:.3f}，"
        f"W8 残差 AUC={b_ratio20.auc_resid_w8:.3f}，maxT `p={b_ratio20.p_resid_family:.4f}`，"
        f"{b_ratio_positive}/{len(b_ratio_groups)} 个混合初态同向；A 的对应残差 AUC="
        f"{a_ratio20.auc_resid_w8:.3f}（`p={a_ratio20.p_resid_family:.3f}`）。",
        "所以数据支持的是“失败时 route mobility 降低”，尚不支持可跨语料复现的独立 KS-entropy collapse。",
        "",
        "## 3. 压缩复杂度",
        "",
        "把 `sqrt(p)` 固定量化到 uint8，按 `(cell, expert, time)` 排列，用 raw DEFLATE 长度除以未压缩长度；",
        "delta 版本对时间差分做可逆 modulo-256 编码。它是可计算的压缩代理，不是真正的 Kolmogorov complexity。",
        "",
        "| window / metric @ t30 | A raw / matched residual | B raw / matched residual |",
        "|---|---:|---:|",
    ]
    for width in (8, 20):
        for metric in ("raw_ncl", "delta_ncl"):
            aa = compression[(compression.corpus == "A") & (compression.t == 30) &
                             (compression.window == width) & (compression.metric == metric)].iloc[0]
            bb = compression[(compression.corpus == "B") & (compression.t == 30) &
                             (compression.window == width) & (compression.metric == metric)].iloc[0]
            lines.append(f"| W{width} `{metric}` | {aa.auc_success:.3f} / {aa.auc_resid_matched:.3f} | "
                         f"{bb.auc_success:.3f} / {bb.auc_resid_matched:.3f} |")
    a_delta25 = compression.query(
        "corpus == 'A' and window == 8 and metric == 'delta_ncl' and t == 25"
    ).iloc[0]
    b_delta25 = compression.query(
        "corpus == 'B' and window == 8 and metric == 'delta_ncl' and t == 25"
    ).iloc[0]
    a_delta_groups = group_robustness.query(
        "corpus == 'A' and family == 'compression' and "
        "metric == 'delta_ncl' and window == 8 and t == 25"
    )
    a_delta_negative = int((a_delta_groups.auc_residual < 0.5).sum())
    lines += [
        "",
        "压缩长度能读到失败序列更重复，但和同窗口的平均 Hellinger 控制后没有稳定的双语料增量。",
        f"A 的 W8 delta-compression 在 `t25` 有单语料残差效应：direction-free AUC="
        f"{a_delta25.det_resid_matched:.3f}，maxT `p={a_delta25.p_resid_family:.4f}`，"
        f"{a_delta_negative}/{len(a_delta_groups)} 个混合 worker 同向；但其 raw AUC="
        f"{a_delta25.auc_success:.3f} 仍低于 W8，而且 B 的对应残差 AUC="
        f"{b_delta25.auc_resid_matched:.3f}（`p={b_delta25.p_resid_family:.3f}`）。",
        "因此可以把 compression transition 作为可视化分析，不能替代主检测器。",
        "",
        "## 4. ε-entropy / bits per chunk",
        "",
        "先在 Hellinger 坐标 `sqrt(p)` 上做均匀量化，再重归一化回 simplex。",
        "",
        "| corpus | float16 route（20,480 bit） | 6-bit route（7,680 bit） | ρ(full) |",
        "|---|---:|---:|---:|",
    ]
    for tag in ("A", "B"):
        q16 = quant[(quant.corpus == tag) & (quant.bits_per_probability == 16)].iloc[0]
        q6 = quant[(quant.corpus == tag) & (quant.bits_per_probability == 6)].iloc[0]
        lines.append(f"| {tag} | {q16.auc_success:.3f} | {q6.auc_success:.3f} | {q6.rho_full:.4f} |")
    lines += [
        "",
        "按 `|ΔAUC|≤0.01 且 ρ≥0.99`，两批共同的最小保真点是每个概率 6 bit：相对 float16 输入缩小 2.67×。",
        "这仍是输入路由的存储成本，不是最终报警标量。",
        "",
        "使用 6-bit 概率，不做标签选择时，各自然测量预算的中位 AUC：",
        "",
        "| family | bits/chunk | A median [min,max] | B median [min,max] |",
        "|---|---:|---:|---:|",
    ]
    family_order = ("one_cell", "one_token", "one_layer", "two_layers", "all")
    for family in family_order:
        vals = []
        bits = None
        for tag in ("A", "B"):
            d = cells[(cells.corpus == tag) & (cells.family == family)]
            bits = int(d.bits_per_chunk.iloc[0])
            vals.append(f"{d.auc_success.median():.3f} [{d.auc_success.min():.3f},{d.auc_success.max():.3f}]")
        lines.append(f"| `{family}` | {bits} | {vals[0]} | {vals[1]} |")
    telemetry_a3 = telemetry.query("corpus == 'A' and bits_per_chunk == 3").iloc[0]
    telemetry_b3 = telemetry.query("corpus == 'B' and bits_per_chunk == 3").iloc[0]
    lines += [
        "",
        "若路由在前向时临时可用、只保存在线算出的标量变化率，则 3 bit/chunk 已给出 A/B "
        f"AUC {telemetry_a3.auc_success:.3f} / {telemetry_b3.auc_success:.3f}；"
        "4 bit 时排序相关超过 0.997。这个数字只回答遥测/落盘成本。",
        "",
    ]
    if hidden is not None and len(hidden):
        full = hidden[(hidden.metric == "relative_l2") &
                      (hidden.coordinates_per_cell == 1024)].iloc[0]
        h32 = hidden[(hidden.metric == "relative_l2") &
                     (hidden.coordinates_per_cell == 32)].auc_success
        ratio = disk["hidden_zarr_bytes"] / disk["routes_zarr_bytes"] if disk else np.nan
        lines += [
            "### B-only hidden control",
            "",
            f"完整 hidden 相邻变化率 AUC={full.auc_success:.3f}；固定随机坐标每格取 32/1024 维，"
            f"4 次中位 AUC={h32.median():.3f}。磁盘上的完整 hidden.zarr / routes.zarr = {ratio:.1f}×。",
            "这说明 routing 文件确实更小，但 hidden 的 Trap 变化也能被少量随机坐标保留；当前数据不能声称 MoE routing "
            "在信息论上是唯一低维的表示。A 没有 hidden capture，所以该控制不能做双语料确认。",
            "",
        ]
    zero_lag = cascade_lag.groupby(["corpus", "layer_a", "layer_b"]).best_lag.first()
    corr0 = cascade_lag[cascade_lag.lag == 0].groupby("corpus").correlation.agg(["min", "max"])
    lines += [
        "## 5. 多尺度 / cascade 检查",
        "",
        f"四个深层的 W8 变化率在 A 的零延迟相关为 {corr0.loc['A','min']:.3f}–{corr0.loc['A','max']:.3f}，"
        f"B 为 {corr0.loc['B','min']:.3f}–{corr0.loc['B','max']:.3f}；"
        f"两批语料合计 12 组 layer-pair 的最大相关延迟全部是 0"
        f"（{int((zero_lag == 0).sum())}/{len(zero_lag)}）。",
        "没有观察到逐层传播；与旧 denoise sweep 一样，更像所有切片同时投影同一个 route-mobility 因子。",
        "",
        "## 6. 在线报警与物理 onset",
        "",
        "阈值按 leave-one-group-out 的成功轨迹校准，连续 3 次低于阈值触发；每个指标只按成功误报率选择最接近旧基线的工作点。",
        "",
        "| corpus | metric | FA | detection | median alarm t | known-onset median delay |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in online.iterrows():
        delay = fmt(float(row.get("median_delay", np.nan)), 1)
        lines.append(f"| {row.corpus} | `{row.metric}` | {row.achieved_fa:.3f} | "
                     f"{row.detection:.3f} | {row.median_alarm_t:.1f} | {delay} |")
    lines += [
        "",
        "物理 onset 延迟只在 A 且 `loop_onset_query >= 0` 的轨迹上计算。熵率/压缩没有稳定提前于物理 onset，",
        "也没有在两批同时超过 d9+W8。",
        "",
        "## 7. 救回实验的证据边界",
        "",
        "现有触发式 fork：q32 报警点重抽 flow noise 为 3/8 成功；随机较早点 q19 同样 3/8。",
        "现有数据没有多个噪声幅度，也只有一个失败 trunk，不能估计 `P_escape(σ)`，更不能把 KAM 当理论依据。",
        "下一项真正需要新增 rollout 的实验应固定多个失败 trunk，预注册 σ 网格和相同随机数，再比较报警点、随机时点和不干预。",
        "",
        "## 总结",
        "",
        "1. **保留**：`d9 + cross-chunk Hellinger + W8` 仍是最强且最简单的 train-free MoE 基线。",
        "2. **可作机制图**：失败时低 route-mobility 状态更驻留、符号条件熵和压缩长度下降。",
        "3. **未复现**：B-t20 entropy ratio 与 A-t25 delta-compression 各有单语料残差信号，"
        "但都没有跨语料复现。",
        "4. **新正结果**：6-bit 概率可保真复现基线；只落盘报警变化率时 3–4 bit/chunk 足够。",
        "5. **限制**：B-only hidden 控制同样低维，因此“routing 独有的信息压缩优势”目前不能声称。",
        "",
        "![overview](overview.png)",
        "",
        "## 产物",
        "",
        "- `dynamics_auc.csv`：熵率、递归与时间曲线。",
        "- `markov_*.csv`：转移矩阵、驻留和 hitting probability。",
        "- `compression_auc.csv`：DEFLATE 复杂度代理及基线残差。",
        "- `route_quantization.csv` / `cell_budget.csv` / `telemetry_budget.csv`：信息预算。",
        "- `hidden_budget_B.csv`：B-only hidden 随机坐标控制。",
        "- `group_robustness.csv`：候选信号在各 worker / 初态内的方向与强度。",
        "- `cascade_*.csv`：逐层时序与延迟相关。",
        "- `online_operating_points.csv`：校准后的报警率和 onset 延迟。",
    ]
    (HERE / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nperm", type=int, default=2000)
    parser.add_argument("--skip-hidden", action="store_true")
    args = parser.parse_args()
    all_dynamics = []
    all_markov_trans = []
    all_markov_struct = []
    all_markov_auc = []
    all_compression = []
    all_quant = []
    all_cells = []
    all_telemetry = []
    all_group_robustness = []
    all_cascade_auc = []
    all_cascade_lag = []
    all_online = []
    meta_b = None

    for tag in ("A", "B"):
        print(f"[{tag}] loading dense routing", flush=True)
        P, valid, meta = agglib.build_dense(tag)
        P /= np.maximum(P.sum(-1, keepdims=True), 1e-12)
        if tag == "B":
            meta_b = meta.copy()
        success = meta["success"].to_numpy(bool)
        groups = meta["group"].to_numpy()
        dcell = cell_hellinger(P, valid)
        movement = np.full(dcell.shape[:2], np.nan, np.float64)
        for t in range(1, dcell.shape[1]):
            ok = valid[:, t]
            movement[ok, t] = dcell[ok, t].mean(1)
        mobility_w8 = rolling_mean(movement, WINDOW)
        mobility_w20 = rolling_mean(movement, 20)

        print(f"[{tag}] Markov and symbolic dynamics", flush=True)
        mtrans, mstruct, mauc = markov_experiment(tag, mobility_w8, valid, meta)
        all_markov_trans.append(mtrans)
        all_markov_struct.append(mstruct)
        all_markov_auc.append(mauc)
        symbolic, edges = symbolic_features(dcell, valid)

        dynamics_rows = []
        metric_series = {"route_mobility_w8": mobility_w8, **symbolic}
        for name, series in metric_series.items():
            for t in range(T_ONLINE_LO, min(series.shape[1], T_ONLINE_HI)):
                row = dict(corpus=tag, metric=name, t=t)
                if name == "route_mobility_w8":
                    row.update(score_summary(series[:, t], success, groups))
                else:
                    row.update(score_summary(series[:, t], success, groups, mobility_w8[:, t]))
                dynamics_rows.append(row)
        dynamics = pd.DataFrame(dynamics_rows)

        # MaxT is applied to the pre-specified symbolic family at the three
        # fixed evaluation times, once raw and once residualized.
        keys = [(name, t) for name in symbolic for t in TS]
        raw = np.stack([symbolic[name][:, t] for name, t in keys], axis=1)
        resid = np.stack(
            [agglib.rank_residualise(symbolic[name][:, t], mobility_w8[:, t], groups)
             for name, t in keys], axis=1
        )
        pp, pf, q95 = family_permutation(raw, success, groups, args.nperm, SEED + ord(tag))
        rp, rf, rq95 = family_permutation(resid, success, groups, args.nperm,
                                          SEED + 100 + ord(tag))
        for i, (name, t) in enumerate(keys):
            mask = (dynamics.metric == name) & (dynamics.t == t)
            dynamics.loc[mask, "p_point"] = pp[i]
            dynamics.loc[mask, "p_family"] = pf[i]
            dynamics.loc[mask, "family_null_q95"] = q95
            dynamics.loc[mask, "p_resid_point"] = rp[i]
            dynamics.loc[mask, "p_resid_family"] = rf[i]
            dynamics.loc[mask, "resid_family_null_q95"] = rq95

        print(f"[{tag}] normalized compression lengths", flush=True)
        comp = compression_features(P, valid)
        comp_rows = []
        for (width, name), series in comp.items():
            for t in TS:
                if t < width:
                    continue
                bases = [mobility_w8[:, t]]
                if width != 8:
                    bases.append(mobility_w20[:, t])
                row = dict(corpus=tag, metric=name, window=width, t=t)
                row.update(score_summary(series[:, t], success, groups,
                                         mobility_w8[:, t], bases))
                comp_rows.append(row)
        comp_df = pd.DataFrame(comp_rows)
        ckeys = [(w, name, t) for w in (8, 20) for name in ("raw_ncl", "delta_ncl")
                 for t in TS if t >= w]
        cres = []
        for w, name, t in ckeys:
            bases = [mobility_w8[:, t]] + ([] if w == 8 else [mobility_w20[:, t]])
            cres.append(rank_residualise_multi(comp[(w, name)][:, t], bases, groups))
        cres = np.stack(cres, axis=1)
        cp, cf, cq95 = family_permutation(cres, success, groups, args.nperm,
                                          SEED + 200 + ord(tag))
        for i, (w, name, t) in enumerate(ckeys):
            mask = ((comp_df.window == w) & (comp_df.metric == name) & (comp_df.t == t))
            comp_df.loc[mask, "p_resid_point"] = cp[i]
            comp_df.loc[mask, "p_resid_family"] = cf[i]
            comp_df.loc[mask, "resid_family_null_q95"] = cq95
        compression_time_rows = []
        for t in range(T_ONLINE_LO, T_ONLINE_HI):
            series = comp[(8, "delta_ncl")]
            row = dict(corpus=tag, metric="compression_delta_w8", t=t)
            row.update(score_summary(series[:, t], success, groups, mobility_w8[:, t]))
            compression_time_rows.append(row)
        dynamics = pd.concat([dynamics, pd.DataFrame(compression_time_rows)], ignore_index=True)
        all_dynamics.append(dynamics)
        all_compression.append(comp_df)

        robustness_rows = []
        for name, series in symbolic.items():
            for t in TS:
                robustness_rows.extend(
                    group_robustness_rows(
                        tag, "symbolic", name, WINDOW, t, series[:, t],
                        [mobility_w8[:, t]], meta
                    )
                )
        for (width, name), series in comp.items():
            for t in TS:
                if t < width:
                    continue
                bases = [mobility_w8[:, t]]
                if width != WINDOW:
                    bases.append(mobility_w20[:, t])
                robustness_rows.extend(
                    group_robustness_rows(
                        tag, "compression", name, width, t, series[:, t], bases, meta
                    )
                )
        all_group_robustness.append(pd.DataFrame(robustness_rows))

        print(f"[{tag}] route and telemetry budgets", flush=True)
        quant, cells, telemetry = route_budget_experiment(tag, P, valid, dcell, meta)
        all_quant.append(quant)
        all_cells.append(cells)
        all_telemetry.append(telemetry)

        print(f"[{tag}] layer cascade check", flush=True)
        cauc, clag = cascade_experiment(tag, dcell, meta)
        all_cascade_auc.append(cauc)
        all_cascade_lag.append(clag)

        online_metrics = {
            "route_mobility_w8": mobility_w8,
            "symbol_conditional_H": symbolic["symbol_conditional_H"],
            "compression_delta_w8": comp[(8, "delta_ncl")],
        }
        all_online.append(online_experiment(tag, online_metrics, meta))
        del P, valid, dcell, movement, mobility_w8, mobility_w20, symbolic, comp
        gc.collect()

    dynamics = pd.concat(all_dynamics, ignore_index=True)
    markov_trans = pd.concat(all_markov_trans, ignore_index=True)
    markov_struct = pd.concat(all_markov_struct, ignore_index=True)
    markov_auc = pd.concat(all_markov_auc, ignore_index=True)
    compression = pd.concat(all_compression, ignore_index=True)
    quant = pd.concat(all_quant, ignore_index=True)
    cells = pd.concat(all_cells, ignore_index=True)
    telemetry = pd.concat(all_telemetry, ignore_index=True)
    group_robustness = pd.concat(all_group_robustness, ignore_index=True)
    cascade_auc = pd.concat(all_cascade_auc, ignore_index=True)
    cascade_lag = pd.concat(all_cascade_lag, ignore_index=True)
    online = pd.concat(all_online, ignore_index=True)

    hidden_df = None
    disk = None
    if not args.skip_hidden:
        assert meta_b is not None
        print("[B] hidden-state random-coordinate budget", flush=True)
        hidden_df, disk = hidden_budget_experiment(meta_b)

    outputs = {
        "dynamics_auc.csv": dynamics,
        "markov_transition.csv": markov_trans,
        "markov_structure.csv": markov_struct,
        "markov_hitting_auc.csv": markov_auc,
        "compression_auc.csv": compression,
        "route_quantization.csv": quant,
        "cell_budget.csv": cells,
        "telemetry_budget.csv": telemetry,
        "group_robustness.csv": group_robustness,
        "cascade_layer_auc.csv": cascade_auc,
        "cascade_lag.csv": cascade_lag,
        "online_operating_points.csv": online,
    }
    if hidden_df is not None:
        outputs["hidden_budget_B.csv"] = hidden_df
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False)

    summary = {
        "schema": "himoe.kolmogorov_trainfree.v1",
        "seed": SEED,
        "n_permutations": args.nperm,
        "slice": {"layers": list(LAYERS), "tokens": list(TOKENS), "denoise": 9,
                  "experts": EXPERTS, "window": WINDOW},
        "new_feature_training": False,
        "label_use": "evaluation and threshold calibration only",
        "disk": disk,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_plots(dynamics, quant, cells, hidden_df)
    write_report(dynamics, markov_struct, markov_auc, compression, quant, cells,
                 telemetry, group_robustness, cascade_lag, online, hidden_df, disk,
                 args.nperm)
    print(f"wrote results to {HERE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
