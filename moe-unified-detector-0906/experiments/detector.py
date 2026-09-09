#!/usr/bin/env python3
"""The unified task-agnostic early-warning detector: fit on development, seal, score.

Pipeline (every fitted quantity comes from ``development_main`` and nothing else)

  1. gated chunks    -- at most 6 per suite, ranked by the replicated within-task
                        fixed-chunk AUC profile of `results/auc/`.
  2. candidate terms -- (column, normalisation) cells that replicate at the gated
                        chunks, minus controls, minus channels with no
                        within-episode information.
  3. weights         -- balanced L2 logistic regression on the pooled
                        (episode, gated chunk) rows of development, on signed
                        z-scores; pruned to a small term list and refit.
  4. thresholds      -- quantiles of the *development* score distribution.  The
                        whole recall/FPR curve is therefore defined on
                        development; external and legacy are scored once at every
                        frozen point of that curve and never used to choose one.

Alarm rules compared: single gated chunk, K-of-M over the gated chunks, CUSUM on
the score (gated chunks only, and every chunk).  Two reference arms that are not
detectors are carried through the same machinery: the survival rule ("still
running at chunk q"), whose task-matched lift is 1 by construction, and the
length leak, which is a negative control and is excluded from every ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import protocol as P
import arms

N_CANDIDATE_COLUMNS = 48
N_TERMS = 10
LOGIT_C = 0.05
MIN_GATE_CHUNKS = 2
ALPHA_GRID = np.unique(
    np.concatenate(
        [
            np.geomspace(1e-4, 0.02, 40),
            np.linspace(0.02, 0.35, 60),
            np.linspace(0.35, 1.0, 40),
        ]
    )
)


# --------------------------------------------------------------------------- #
# frozen detector


@dataclass
class Detector:
    name: str
    columns: list[str]
    terms: list[tuple[str, str]]          # (column, normalisation)
    sign: np.ndarray                      # +1 / -1 per term
    weight: np.ndarray                    # fitted weight per term
    mu: np.ndarray                        # (n_chunk, n_term) development mean
    sd: np.ndarray                        # (n_chunk, n_term) development sd
    pop_ref: list[list[np.ndarray]]       # [chunk][term] sorted development values
    gated: dict[str, list[int]]
    prior_offset: np.ndarray | None = None   # (n_suite, n_chunk) log-odds, dev-fitted
    prior_weight: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def survival_prior_logodds(frame: dict) -> np.ndarray:
    """log-odds of P(risk | suite cap, still running at chunk q), from development.

    Uses only the suite horizon cap and the chunk index -- both are rollout
    configuration, not task identity and not an outcome of the scored episode.
    """
    n_chunk = frame["alive"].shape[1]
    out = np.full((len(P.SUITES), n_chunk), -np.inf)
    for i, suite in enumerate(P.SUITES):
        rows = frame["suite"] == suite
        for q in range(n_chunk):
            alive = rows & (frame["length"] > q)
            if alive.sum() < 20:
                out[i, q] = out[i, q - 1] if q else 0.0
                continue
            p = float(frame["risk"][alive].mean())
            p = min(max(p, 1e-4), 1 - 1e-4)
            out[i, q] = np.log(p / (1 - p))
    return out


def _prior_column(frame: dict, offset: np.ndarray) -> np.ndarray:
    idx = np.array([P.SUITES.index(s) for s in frame["suite"]])
    return offset[idx, :]


def within_episode_info(frame: dict, values: np.ndarray) -> pd.DataFrame:
    """Distinct values inside an episode, per column, over alive chunks."""
    n, n_chunk, n_col = values.shape
    alive = frame["alive"]
    rng = np.random.default_rng(P.SEED)
    take = rng.choice(n, size=min(n, 4000), replace=False)
    rows = []
    for j in range(n_col):
        block = np.where(alive[take], values[take, :, j], np.nan)
        n_distinct = np.array(
            [len(np.unique(row[np.isfinite(row)])) for row in block]
        )
        finite_rows = n_distinct > 0
        rows.append(
            {
                "column": frame["columns"][j] if len(frame["columns"]) == n_col
                else str(j),
                "median_distinct_per_episode": float(np.median(n_distinct[finite_rows]))
                if finite_rows.any() else 0.0,
                "frac_episodes_constant": float(
                    (n_distinct[finite_rows] <= 1).mean()
                ) if finite_rows.any() else 1.0,
                "n_distinct_global": int(
                    len(np.unique(values[:, :, j][np.isfinite(values[:, :, j])]))
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_gated_chunks(
    table: pd.DataFrame,
    scope_portable: bool,
    max_chunks: int = P.MAX_GATED_CHUNKS,
    use_all: bool = False,
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """Top-`max_chunks` chunks per suite by the 3rd strongest replicated cell.

    Using the 3rd strongest rather than the strongest means a single lucky cell
    cannot open a gate.  `use_all` is the drop-chunk-gating ablation.
    """
    sel = table[~table["is_control"]]
    if scope_portable:
        sel = sel[sel["is_portable"]]
    prof = []
    for suite in P.SUITES:
        chunks = range(P.FIRST_SCORED_CHUNK, P.IN_WINDOW_END[suite] + 1)
        for chunk in chunks:
            cell = sel[(sel["suite"] == suite) & (sel["chunk"] == chunk)]
            rep = cell[cell["replicated"]]
            eff = np.sort(rep["abs_effect"].to_numpy())[::-1]
            prof.append(
                {
                    "suite": suite,
                    "chunk": chunk,
                    "n_replicated": int(len(rep)),
                    "top1": float(eff[0]) if len(eff) else 0.0,
                    "top3": float(eff[2]) if len(eff) >= 3 else 0.0,
                    "top1_any": float(cell["abs_effect"].max()) if len(cell) else 0.0,
                    "n_tasks": float(cell["n_tasks"].max()) if len(cell) else 0.0,
                    "prior": float(cell["prior"].max()) if len(cell) else np.nan,
                }
            )
    profile = pd.DataFrame(prof)
    gated: dict[str, list[int]] = {}
    for suite in P.SUITES:
        sub = profile[profile["suite"] == suite]
        if use_all:
            gated[suite] = sorted(sub["chunk"].tolist())
            continue
        ranked = sub.sort_values(["top3", "top1", "chunk"], ascending=[False, False, True])
        picked = ranked[ranked["top3"] > 0].head(max_chunks)["chunk"].tolist()
        if not picked:  # a suite with no replicated cell still needs a gate
            picked = ranked.head(max_chunks)["chunk"].tolist()
        gated[suite] = sorted(int(c) for c in picked)
    profile["gated"] = [
        c in gated[s] for s, c in zip(profile["suite"], profile["chunk"])
    ]
    return gated, profile


def candidate_terms(
    table: pd.DataFrame,
    gated: dict[str, list[int]],
    info: pd.DataFrame,
    scope_portable: bool,
    norms: tuple[str, ...],
    n_columns: int = N_CANDIDATE_COLUMNS,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Columns that replicate at >= MIN_GATE_CHUNKS gated chunks, ranked by effect."""
    sel = table[~table["is_control"]].copy()
    if scope_portable:
        sel = sel[sel["is_portable"]]
    keep = np.zeros(len(sel), bool)
    for suite, chunks in gated.items():
        keep |= (sel["suite"] == suite).to_numpy() & sel["chunk"].isin(chunks).to_numpy()
    sel = sel[keep]

    degenerate = set(
        info.loc[info["frac_episodes_constant"] >= 0.999, "column"]
    )
    sel = sel[~sel["column"].isin(degenerate)]

    grp = (
        sel.groupby(["column", "rep"])
        .agg(
            n_replicated=("replicated", "sum"),
            mean_abs_effect=("abs_effect", "mean"),
            max_abs_effect=("abs_effect", "max"),
            mean_effect=("effect", "mean"),
            n_cells=("effect", "size"),
        )
        .reset_index()
    )
    ok = grp[grp["n_replicated"] >= MIN_GATE_CHUNKS].copy()
    if ok.empty:
        ok = grp.copy()
    # `raw` and `pop` share one within-task order, hence one AUC; rank columns on
    # the better of the two forms that were actually measured.
    best = (
        ok.sort_values("mean_abs_effect", ascending=False)
        .drop_duplicates("column")
        .head(n_columns)
    )
    chosen_columns = best["column"].tolist()
    terms = [(c, rep) for c in chosen_columns for rep in norms]
    return terms, grp


def _term_block(
    frame: dict,
    values: np.ndarray,
    columns: list[str],
    terms: list[tuple[str, str]],
    pop_ref: list[list[np.ndarray]] | None,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """(n, chunk, n_term) block of the requested normalisations."""
    index = {c: j for j, c in enumerate(columns)}
    raw_cols = sorted({c for c, _ in terms})
    sub = values[:, :, [index[c] for c in raw_cols]]
    sub_index = {c: j for j, c in enumerate(raw_cols)}
    forms = {"raw": sub}
    if any(rep == "self" for _, rep in terms):
        forms["self"] = P.apply_self(sub)
    if any(rep == "pop" for _, rep in terms):
        if pop_ref is None:
            pop_ref = P.pop_reference(sub, frame["alive"])
        forms["pop"] = P.apply_pop(sub, pop_ref)
    out = np.empty((sub.shape[0], sub.shape[1], len(terms)), np.float32)
    for t, (col, rep) in enumerate(terms):
        out[:, :, t] = forms[rep][:, :, sub_index[col]]
    out[~frame["alive"]] = np.nan
    return out, (pop_ref or [])


def _rows_at_gates(frame: dict, gated: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Row (episode, chunk) index of every episode alive at one of its gated chunks."""
    ep, ch = [], []
    for suite, chunks in gated.items():
        in_suite = np.flatnonzero(frame["suite"] == suite)
        for chunk in chunks:
            take = in_suite[frame["length"][in_suite] > chunk]
            ep.append(take)
            ch.append(np.full(len(take), chunk))
    return np.concatenate(ep), np.concatenate(ch)


def fit(
    frame: dict,
    values: np.ndarray,
    columns: list[str],
    table: pd.DataFrame,
    info: pd.DataFrame,
    *,
    name: str,
    scope_portable: bool = True,
    norms: tuple[str, ...] = P.NORMALISATIONS,
    use_all_chunks: bool = False,
    n_terms: int = N_TERMS,
    use_prior: bool = False,
) -> tuple[Detector, dict[str, Any]]:
    gated, profile = choose_gated_chunks(table, scope_portable, use_all=use_all_chunks)
    terms, term_stats = candidate_terms(table, gated, info, scope_portable, norms)

    block, pop_ref = _term_block(frame, values, columns, terms, None)
    mu, sd = P.chunk_moments(block, frame["alive"])
    z = P.standardise(block, mu, sd)

    # sign every term so that "larger" means "riskier", from development only
    stats = term_stats.set_index(["column", "rep"])
    sign = np.array(
        [
            np.sign(
                stats.loc[(col, rep if rep != "pop" else "raw"), "mean_effect"]
            ) or 1.0
            for col, rep in terms
        ],
        np.float32,
    )
    zs = z * sign[None, None, :]

    ep, ch = _rows_at_gates(frame, gated)
    X = np.nan_to_num(zs[ep, ch, :], nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["risk"][ep]
    offset = survival_prior_logodds(frame) if use_prior else None
    prior_col = _prior_column(frame, offset)[ep, ch][:, None] if use_prior else None

    model = LogisticRegression(
        C=LOGIT_C, max_iter=4000, class_weight="balanced", solver="lbfgs"
    )
    model.fit(X, y)
    order = np.argsort(-np.abs(model.coef_[0]))[:n_terms]
    order = np.sort(order)
    kept = [terms[i] for i in order]

    X2 = X[:, order] if prior_col is None else np.hstack([X[:, order], prior_col])
    model2 = LogisticRegression(
        C=LOGIT_C, max_iter=4000, class_weight="balanced", solver="lbfgs"
    )
    model2.fit(X2, y)
    weight = model2.coef_[0][: len(order)].astype(np.float64)
    prior_weight = float(model2.coef_[0][-1]) if prior_col is not None else 0.0

    kept_columns = [c for c, _ in kept]
    kept_unique = list(dict.fromkeys(kept_columns))
    if pop_ref:
        all_raw = sorted({c for c, _ in terms})
        take = [all_raw.index(c) for c in kept_unique]
        kept_ref = [[chunk_ref[j] for j in take] for chunk_ref in pop_ref]
    else:
        kept_ref = []

    det = Detector(
        name=name,
        columns=kept_columns,
        terms=kept,
        sign=sign[order].astype(np.float64),
        weight=weight,
        mu=mu[:, order],
        sd=sd[:, order],
        pop_ref=kept_ref,
        gated=gated,
        prior_offset=offset,
        prior_weight=prior_weight,
        meta={
            "scope_portable": scope_portable,
            "norms": list(norms),
            "use_all_chunks": use_all_chunks,
            "n_candidate_terms": len(terms),
            "n_fit_rows": int(len(y)),
            "n_fit_risk_rows": int(y.sum()),
            "intercept": float(model2.intercept_[0]),
            "logit_C": LOGIT_C,
            "use_prior": use_prior,
            "prior_weight": prior_weight,
        },
    )
    diagnostics = {"gate_profile": profile, "term_stats": term_stats, "terms": terms}
    return det, diagnostics


def score(det: Detector, frame: dict, values: np.ndarray, columns: list[str]) -> np.ndarray:
    """Score every (episode, chunk); NaN where the episode is not running.

    Every fitted quantity -- sign, weight, chunk mean/sd, population reference --
    comes from the frozen `det`, i.e. from development_main.  Nothing about the
    scored cohort is estimated here.
    """
    index = {c: j for j, c in enumerate(columns)}
    raw_cols = list(dict.fromkeys(det.columns))
    sub = values[:, :, [index[c] for c in raw_cols]]
    sub_index = {c: j for j, c in enumerate(raw_cols)}
    forms = {"raw": sub}
    if any(rep == "self" for _, rep in det.terms):
        forms["self"] = P.apply_self(sub)
    if any(rep == "pop" for _, rep in det.terms):
        forms["pop"] = P.apply_pop(sub, det.pop_ref)
    out = np.zeros(sub.shape[:2], np.float64)
    for t, (col, rep) in enumerate(det.terms):
        v = forms[rep][:, :, sub_index[col]]
        z = (v - det.mu[:, t][None, :]) / det.sd[:, t][None, :]
        z = np.nan_to_num(z * det.sign[t], nan=0.0, posinf=0.0, neginf=0.0)
        out += det.weight[t] * z
    if det.prior_offset is not None and det.prior_weight:
        out += det.prior_weight * _prior_column(frame, det.prior_offset)
    out[~frame["alive"]] = np.nan
    return out


# --------------------------------------------------------------------------- #
# alarm rules.  every comparison is `>=`, never `>`


def gate_mask(frame: dict, gated: dict[str, list[int]]) -> np.ndarray:
    mask = np.zeros(frame["alive"].shape, bool)
    for suite, chunks in gated.items():
        rows = frame["suite"] == suite
        for chunk in chunks:
            mask[rows, chunk] = True
    return mask & frame["alive"]


def rule_single(S, frame, gated, tau, chunk_of_suite) -> np.ndarray:
    fire = np.zeros(S.shape, bool)
    for suite, chunk in chunk_of_suite.items():
        rows = (frame["suite"] == suite) & frame["alive"][:, chunk]
        fire[rows, chunk] = S[rows, chunk] >= tau
    return P.first_alarm(fire)


def rule_k_of_m(S, frame, gmask, tau, k) -> np.ndarray:
    hit = (S >= tau) & gmask
    cum = np.cumsum(hit, axis=1)
    fire = hit & (cum >= k)
    return P.first_alarm(fire)


def rule_cusum(S, frame, mask, tau, drift, floor=0.0) -> np.ndarray:
    n, n_chunk = S.shape
    c = np.zeros(n)
    fire = np.zeros(S.shape, bool)
    for q in range(n_chunk):
        s = np.where(mask[:, q], np.nan_to_num(S[:, q], nan=0.0) - drift, 0.0)
        c = np.maximum(floor, c + s)
        fire[:, q] = mask[:, q] & (c >= tau)
    return P.first_alarm(fire)


def rule_survivor(frame, chunk_of_suite) -> np.ndarray:
    fire = np.zeros(frame["alive"].shape, bool)
    for suite, chunk in chunk_of_suite.items():
        rows = frame["suite"] == suite
        fire[rows, chunk] = frame["alive"][rows, chunk]
    return P.first_alarm(fire)


def rule_length_leak(frame, chunk: int = 0) -> np.ndarray:
    """NEGATIVE CONTROL.  Uses the final length, which is an outcome."""
    at_cap = frame["length"] >= frame["cap"]
    first = np.where(at_cap, chunk, -1)
    return first.astype(np.int64)
