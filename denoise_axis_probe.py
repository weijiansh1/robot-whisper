#!/usr/bin/env python3
"""Probe the flow-denoise iteration axis (10 steps inside each control step)
as a rollout-failure signal for HiMoE-VLA MoE routing.

Label-blind feature construction; labels opened only at AUC time.
Protocol: within-group AUC (group = worker in corpus A, init_state_id in
corpus B), pooled by pair count; risk set at query index t = branches with T>t.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import zarr

ROOT = pathlib.Path("/home/jovyan/work/himoe-vla")
RUN_A = ROOT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
ZARR_A = RUN_A / "formal/server/routes.zarr"
LABELS_A = RUN_A / "analysis/candidate_physical_labels.csv"
DIR_B = (
    ROOT
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
    / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
ZARR_B = DIR_B / "server/routes.zarr"
SUMM_B = DIR_B / "client/summaries.json"

DEEP = np.arange(4, 8)          # HB layers 12,13,14,15
FRONT = np.arange(0, 4)         # HB layers 2,3,4,5
WINDOW = 8
T_GRID = [15, 20, 25, 30, 35]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def normalize(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float32), 0.0)
    return p / p.sum(axis=-1, keepdims=True)


def hell_from_sqrt(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Hellinger distance given sqrt(p) arrays; reduces last axis."""
    return np.sqrt(np.maximum(0.0, 0.5 * np.sum(np.square(sa - sb), axis=-1)))


def norm_entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1) / np.log(p.shape[-1])


# --------------------------------------------------------------------------- #
# per-episode feature extraction
# --------------------------------------------------------------------------- #
def token_block_features(prob: np.ndarray, ent: np.ndarray, tag: str) -> dict:
    """prob: (T, L, 10, K, 32) normalized. ent: (T, L, 10, K). K tokens pooled.

    Returns dict name -> (T,) per-row series.  Names carry `tag` prefix.
    """
    T = prob.shape[0]
    sq = np.sqrt(prob)                                   # (T,L,10,K,32)
    out: dict[str, np.ndarray] = {}

    # ---- cross-chunk: per-denoise-slice Hellinger to previous control step
    per_d = np.full((T, 10), np.nan, dtype=np.float64)
    if T > 1:
        # (T-1, L, 10, K)
        h = hell_from_sqrt(sq[1:], sq[:-1])
        per_d[1:] = h.mean(axis=(1, 3))                  # mean over layers, tokens
    for d in range(10):
        out[f"{tag}|xchunk_hell_d{d}"] = per_d[:, d]
    out[f"{tag}|xchunk_hell_dmean"] = per_d.mean(axis=1)
    out[f"{tag}|xchunk_hell_early02"] = per_d[:, 0:3].mean(axis=1)
    out[f"{tag}|xchunk_hell_late79"] = per_d[:, 7:10].mean(axis=1)
    out[f"{tag}|xchunk_hell_late_minus_early"] = (
        per_d[:, 7:10].mean(axis=1) - per_d[:, 0:3].mean(axis=1)
    )
    out[f"{tag}|xchunk_hell_late_over_early"] = per_d[:, 7:10].mean(axis=1) / np.maximum(
        per_d[:, 0:3].mean(axis=1), 1e-9
    )

    # ---- within-chunk denoise trajectory
    step = hell_from_sqrt(sq[:, :, 1:], sq[:, :, :-1])   # (T,L,9,K)
    path = step.sum(axis=2)                              # (T,L,K)
    net = hell_from_sqrt(sq[:, :, 9], sq[:, :, 0])       # (T,L,K)
    eff = net / np.maximum(path, 1e-9)
    didx = np.arange(9, dtype=np.float64) + 0.5          # midpoint of each hop
    centroid = np.einsum("tlds,d->tls", step.astype(np.float64), didx) / np.maximum(
        path, 1e-9
    )
    front3 = step[:, :, 0:3].sum(axis=2) / np.maximum(path, 1e-9)
    back3 = step[:, :, 6:9].sum(axis=2) / np.maximum(path, 1e-9)
    smax = step.max(axis=2)
    sargmax = step.argmax(axis=2).astype(np.float64)

    out[f"{tag}|wc_path"] = path.mean(axis=(1, 2))
    out[f"{tag}|wc_net"] = net.mean(axis=(1, 2))
    out[f"{tag}|wc_eff"] = eff.mean(axis=(1, 2))
    out[f"{tag}|wc_centroid"] = centroid.mean(axis=(1, 2))
    out[f"{tag}|wc_front3frac"] = front3.mean(axis=(1, 2))
    out[f"{tag}|wc_back3frac"] = back3.mean(axis=(1, 2))
    out[f"{tag}|wc_backminusfront"] = (back3 - front3).mean(axis=(1, 2))
    out[f"{tag}|wc_stepmax"] = smax.mean(axis=(1, 2))
    out[f"{tag}|wc_stepargmax"] = sargmax.mean(axis=(1, 2))
    for d in range(9):
        out[f"{tag}|wc_step{d}to{d+1}"] = step[:, :, d].mean(axis=(1, 2))

    # dispersion of the routing distribution across the 10 iterations
    qbar = prob.mean(axis=2)                             # (T,L,K,32)
    sqbar = np.sqrt(qbar)
    disp = hell_from_sqrt(sq, sqbar[:, :, None]).mean(axis=2)   # (T,L,K)
    out[f"{tag}|wc_disp"] = disp.mean(axis=(1, 2))
    var_sum = prob.var(axis=2).sum(axis=-1)              # (T,L,K)
    out[f"{tag}|wc_varsum"] = var_sum.mean(axis=(1, 2))

    # early vs late denoise contrast (mean d0-2 vs mean d7-9)
    qe = prob[:, :, 0:3].mean(axis=2)
    ql = prob[:, :, 7:10].mean(axis=2)
    out[f"{tag}|wc_earlylate_hell"] = hell_from_sqrt(
        np.sqrt(qe), np.sqrt(ql)
    ).mean(axis=(1, 2))

    # entropy along the denoise axis
    e = ent.mean(axis=(1, 3))                            # (T,10)
    out[f"{tag}|wc_ent_d0"] = e[:, 0]
    out[f"{tag}|wc_ent_d9"] = e[:, 9]
    out[f"{tag}|wc_ent_d9_minus_d0"] = e[:, 9] - e[:, 0]
    out[f"{tag}|wc_ent_mean"] = e.mean(axis=1)
    out[f"{tag}|wc_ent_early02"] = e[:, 0:3].mean(axis=1)
    out[f"{tag}|wc_ent_late79"] = e[:, 7:10].mean(axis=1)
    out[f"{tag}|wc_ent_late_minus_early"] = e[:, 7:10].mean(axis=1) - e[:, 0:3].mean(
        axis=1
    )
    out[f"{tag}|wc_ent_range"] = e.max(axis=1) - e.min(axis=1)
    out[f"{tag}|wc_ent_std"] = e.std(axis=1)
    return out


def episode_features(prob_raw: np.ndarray, ent_raw: np.ndarray) -> dict:
    """prob_raw (T,8,10,11,32) float16, ent_raw (T,8,10,11) float16."""
    prob = normalize(prob_raw)
    ent = np.asarray(ent_raw, dtype=np.float32)
    feats: dict[str, np.ndarray] = {}
    # deep layers 12-15, action tokens 1-10  (baseline geometry)
    feats.update(
        token_block_features(prob[:, DEEP][:, :, :, 1:], ent[:, DEEP][:, :, :, 1:], "A")
    )
    # deep layers 12-15, state token only
    feats.update(
        token_block_features(
            prob[:, DEEP][:, :, :, 0:1], ent[:, DEEP][:, :, :, 0:1], "S"
        )
    )
    # front layers 2-5, action tokens (secondary)
    feats.update(
        token_block_features(
            prob[:, FRONT][:, :, :, 1:], ent[:, FRONT][:, :, :, 1:], "F"
        )
    )
    return feats


# --------------------------------------------------------------------------- #
# corpus loaders
# --------------------------------------------------------------------------- #
def load_corpus_a():
    lab = pd.read_csv(LABELS_A)
    z = zarr.open(str(ZARR_A), mode="r")
    ep = z["episode_id"][:]
    cs = z["control_step"][:]
    probs = z["hb_router_probs"]
    ents = z["hb_entropy"]
    order = {}
    for e in lab.episode_id.to_numpy():
        idx = np.flatnonzero(ep == e)
        order[int(e)] = idx[np.argsort(cs[idx])]
    branches = []
    for _, r in lab.iterrows():
        rows = order[int(r.episode_id)]
        branches.append(
            dict(
                episode_id=int(r.episode_id),
                group=int(r.worker),
                success=bool(r.success),
                rows=rows,
                T=len(rows),
            )
        )
    return branches, probs, ents


def load_corpus_b():
    summ = json.loads(SUMM_B.read_text())
    summ = sorted(summ, key=lambda r: int(r["episode_index"]))
    z = zarr.open(str(ZARR_B), mode="r")
    probs = z["hb_router_probs"]
    ents = z["hb_entropy"]
    ep = z["episode_id"][:]
    off = 0
    branches = []
    for r in summ:
        n = int(r["inference_calls"])
        rows = np.arange(off, off + n)
        assert len(np.unique(ep[rows])) == 1, "episode block not contiguous"
        off += n
        branches.append(
            dict(
                episode_id=int(r["episode_index"]),
                group=int(r["init_state_id"]),
                success=bool(r["success"]),
                rows=rows,
                T=n,
            )
        )
    assert off == probs.shape[0], (off, probs.shape[0])
    return branches, probs, ents


# --------------------------------------------------------------------------- #
# windowed summaries + AUC
# --------------------------------------------------------------------------- #
def window_summaries(series: np.ndarray, t: int, cross_chunk: bool):
    """Return (level, delta) summaries over the 8-step window ending at t."""
    lo = max(0, t - WINDOW + 1)
    seg = series[lo : t + 1]
    seg = seg[np.isfinite(seg)]
    level = float(seg.mean()) if seg.size else np.nan
    lo2 = max(1, t - WINDOW + 1)
    if t >= 1:
        dd = np.abs(series[lo2 : t + 1] - series[lo2 - 1 : t])
        dd = dd[np.isfinite(dd)]
        delta = float(dd.mean()) if dd.size else np.nan
    else:
        delta = np.nan
    return level, delta


def auc_pairwise(scores: np.ndarray, pos: np.ndarray):
    """AUC = P(score_pos > score_neg) + 0.5 ties; returns (auc, n_pairs)."""
    s_pos, s_neg = scores[pos], scores[~pos]
    if s_pos.size == 0 or s_neg.size == 0:
        return np.nan, 0
    diff = s_pos[:, None] - s_neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (s_pos.size * s_neg.size)), int(s_pos.size * s_neg.size)


def within_group_auc(scores, groups, pos):
    ok = np.isfinite(scores)
    num = den = 0.0
    for g in np.unique(groups[ok]):
        m = ok & (groups == g)
        a, npair = auc_pairwise(scores[m], pos[m])
        if npair > 0 and np.isfinite(a):
            num += a * npair
            den += npair
    if den == 0:
        return np.nan, 0
    return num / den, int(den)


# --------------------------------------------------------------------------- #
def build(corpus: str):
    if corpus == "A":
        branches, probs, ents = load_corpus_a()
    else:
        branches, probs, ents = load_corpus_b()

    feat_names = None
    per_branch = []
    for i, b in enumerate(branches):
        rows = b["rows"]
        p = probs[rows[0] : rows[-1] + 1] if np.array_equal(
            rows, np.arange(rows[0], rows[-1] + 1)
        ) else probs.oindex[rows]
        e = ents[rows[0] : rows[-1] + 1] if np.array_equal(
            rows, np.arange(rows[0], rows[-1] + 1)
        ) else ents.oindex[rows]
        f = episode_features(np.asarray(p), np.asarray(e))
        if feat_names is None:
            feat_names = sorted(f)
        per_branch.append(f)
        if (i + 1) % 50 == 0:
            print(f"  [{corpus}] {i+1}/{len(branches)}", flush=True)
    return branches, feat_names, per_branch


def grid(branches, feat_names, per_branch):
    groups = np.array([b["group"] for b in branches])
    pos = np.array([not b["success"] for b in branches])  # 1 = FAILURE
    Ts = np.array([b["T"] for b in branches])
    rows = []
    score_cube = {}
    for t in T_GRID:
        alive = Ts > t
        for name in feat_names:
            cross = "xchunk" in name
            lev = np.full(len(branches), np.nan)
            dlt = np.full(len(branches), np.nan)
            for i, f in enumerate(per_branch):
                if not alive[i]:
                    continue
                lev[i], dlt[i] = window_summaries(f[name], t, cross)
            variants = [("level", lev)]
            if not cross:
                variants.append(("delta", dlt))
            for vname, sc in variants:
                sc = np.where(alive, sc, np.nan)
                a, npair = within_group_auc(sc, groups, pos)
                key = (f"{name}|{vname}", t)
                score_cube[key] = sc
                rows.append(
                    dict(
                        feature=f"{name}|{vname}",
                        t=t,
                        auc=a,
                        pairs=npair,
                        n_alive=int(alive.sum()),
                        n_fail=int((alive & pos).sum()),
                    )
                )
    return pd.DataFrame(rows), score_cube, groups, pos, Ts


def permutation_fwe(score_cube, groups, pos, Ts, draws=200, seed=20260829):
    """Family-wise null for max |AUC-0.5|; labels permuted within
    (group x survival-stratum) so every risk set keeps its exact class counts."""
    rng = np.random.default_rng(seed)
    stratum = np.zeros(len(pos), dtype=np.int64)
    for t in T_GRID:
        stratum = stratum * 2 + (Ts > t).astype(np.int64)
    keys = list(score_cube)
    obs = {}
    for k in keys:
        a, _ = within_group_auc(score_cube[k], groups, pos)
        obs[k] = a
    obs_max = np.nanmax([abs(v - 0.5) for v in obs.values()])
    null_max = np.empty(draws)
    cells = np.unique(np.stack([groups, stratum], 1), axis=0)
    idx_cells = [
        np.flatnonzero((groups == g) & (stratum == s)) for g, s in cells
    ]
    for d in range(draws):
        perm = pos.copy()
        for idx in idx_cells:
            perm[idx] = rng.permutation(pos[idx])
        m = 0.0
        for k in keys:
            a, _ = within_group_auc(score_cube[k], groups, perm)
            if np.isfinite(a):
                m = max(m, abs(a - 0.5))
        null_max[d] = m
    p = (1.0 + np.sum(null_max >= obs_max - 1e-12)) / (draws + 1.0)
    return obs_max, p, null_max


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    branches, feat_names, per_branch = build(which)
    df, cube, groups, pos, Ts = grid(branches, feat_names, per_branch)
    out = ROOT / f"denoise_axis_grid_{which}.csv"
    df.to_csv(out, index=False)
    np.savez_compressed(
        ROOT / f"denoise_axis_cube_{which}.npz",
        keys=np.array([f"{k[0]}@@{k[1]}" for k in cube]),
        scores=np.stack([cube[k] for k in cube]),
        groups=groups,
        pos=pos,
        Ts=Ts,
    )
    print(f"wrote {out}  ({len(df)} cells)")
    piv = df.pivot(index="feature", columns="t", values="auc")
    piv["maxabs"] = (piv - 0.5).abs().max(axis=1)
    print(piv.sort_values("maxabs", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
