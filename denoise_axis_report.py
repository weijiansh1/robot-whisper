#!/usr/bin/env python3
"""Assemble the denoise-axis grid into report tables + permutation FWE."""
from __future__ import annotations

import numpy as np
import pandas as pd

T_GRID = [15, 20, 25, 30, 35]
pd.set_option("display.width", 250)


def load(c):
    df = pd.read_csv(f"denoise_axis_grid_{c}.csv")
    # convention flip: stored auc has failure as positive class.
    # baseline 0.795 is quoted with SUCCESS as positive -> auc_s = 1 - auc.
    df["auc_s"] = 1.0 - df["auc"]
    return df


def pivot(df, feats=None, prefix=None):
    d = df
    if prefix is not None:
        d = d[d.feature.str.startswith(prefix)]
    if feats is not None:
        d = d[d.feature.isin(feats)]
    p = d.pivot(index="feature", columns="t", values="auc_s")
    p["|max-.5|"] = (p[T_GRID] - 0.5).abs().max(axis=1)
    return p


def within_group_auc(scores, groups, pos):
    ok = np.isfinite(scores)
    num = den = 0.0
    for g in np.unique(groups[ok]):
        m = ok & (groups == g)
        sp, sn = scores[m & pos], scores[m & ~pos]
        if sp.size == 0 or sn.size == 0:
            continue
        diff = sp[:, None] - sn[None, :]
        wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        npair = sp.size * sn.size
        num += wins / npair * npair
        den += npair
    return (num / den) if den else np.nan


def fwe(c, draws=200, seed=20260829, feature_filter=None):
    z = np.load(f"denoise_axis_cube_{c}.npz", allow_pickle=True)
    keys = [k.split("@@") for k in z["keys"]]
    scores, groups, pos, Ts = z["scores"], z["groups"], z["pos"], z["Ts"]
    if feature_filter is not None:
        keep = np.array([feature_filter(k[0]) for k in keys])
        scores = scores[keep]
        keys = [k for k, m in zip(keys, keep) if m]
    obs = np.array([abs(within_group_auc(s, groups, pos) - 0.5) for s in scores])
    obs_max = np.nanmax(obs)
    best = keys[int(np.nanargmax(obs))]

    stratum = np.zeros(len(pos), dtype=np.int64)
    for t in T_GRID:
        stratum = stratum * 2 + (Ts > t).astype(np.int64)
    cells = [
        np.flatnonzero((groups == g) & (stratum == s))
        for g in np.unique(groups)
        for s in np.unique(stratum)
    ]
    cells = [c_ for c_ in cells if c_.size]
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for d in range(draws):
        perm = pos.copy()
        for idx in cells:
            perm[idx] = rng.permutation(pos[idx])
        m = 0.0
        for s in scores:
            a = within_group_auc(s, groups, perm)
            if np.isfinite(a):
                m = max(m, abs(a - 0.5))
        null[d] = m
        if (d + 1) % 50 == 0:
            print(f"    perm {d+1}/{draws} running max-null p95={np.quantile(null[:d+1],0.95):.3f}", flush=True)
    p = (1.0 + int((null >= obs_max - 1e-12).sum())) / (draws + 1.0)
    return dict(
        n_cells=len(keys),
        best=best,
        obs_max_absdev=obs_max,
        obs_max_auc=0.5 + obs_max,
        p_fwe=p,
        null_med=float(np.median(null)),
        null_p95=float(np.quantile(null, 0.95)),
        null_max=float(null.max()),
    )


if __name__ == "__main__":
    A, B = load("A"), load("B")
    dslices = [f"A|xchunk_hell_d{d}|level" for d in range(10)] + [
        "A|xchunk_hell_dmean|level",
        "A|xchunk_hell_early02|level",
        "A|xchunk_hell_late79|level",
        "A|xchunk_hell_late_minus_early|level",
        "A|xchunk_hell_late_over_early|level",
    ]
    print("### 1. per-denoise-slice cross-chunk Hellinger (deep 12-15, action 1-10, 8-step window)")
    print("### corpus A  (AUC oriented success=positive, baseline convention)")
    print(pivot(A, feats=dslices).reindex(dslices).round(4).to_string())
    print("\n### corpus B")
    print(pivot(B, feats=dslices).reindex(dslices).round(4).to_string())

    print("\n\n### 2. within-chunk denoise trajectory statistics, deep layers, action tokens")
    wc = sorted(set(A[A.feature.str.startswith("A|wc_")].feature))
    print("--- corpus A ---")
    print(pivot(A, feats=wc).round(4).sort_values("|max-.5|", ascending=False).to_string())
    print("--- corpus B ---")
    print(pivot(B, feats=wc).round(4).sort_values("|max-.5|", ascending=False).to_string())

    print("\n\n### 3. cross-corpus agreement, all deep-layer action features")
    ma = pivot(A, prefix="A|")
    mb = pivot(B, prefix="A|")
    j = ma[T_GRID].join(mb[T_GRID], lsuffix="_A", rsuffix="_B")
    j["minabs"] = np.minimum(
        (j[[f"{t}_A" for t in T_GRID]].values - 0.5),
        (j[[f"{t}_B" for t in T_GRID]].values - 0.5),
    ).__abs__().max(axis=1)
    same = np.sign(j[[f"{t}_A" for t in T_GRID]].values - 0.5) == np.sign(
        j[[f"{t}_B" for t in T_GRID]].values - 0.5
    )
    j["agree_signs"] = same.sum(axis=1)
    j["repl"] = np.nanmin(
        np.stack(
            [
                np.abs(j[[f"{t}_A" for t in T_GRID]].values - 0.5),
                np.abs(j[[f"{t}_B" for t in T_GRID]].values - 0.5),
            ]
        ),
        axis=0,
    ).max(axis=1) * np.where(same.any(axis=1), 1, 1)
    print(j.round(3).sort_values("repl", ascending=False).head(25).to_string())
