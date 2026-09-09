"""LOGO within-group paired-AUC evaluation, sentinels, permutation null,
sensitivity grid and phase-shuffle control for the trap-type phase classifier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc_common import aggregate, phase_bins  # noqa: E402

OUT = Path("/home/jovyan/work/himoe-vla/analysis_phasecls")
BLOCKS = ["back", "front"]
DENOISE = [9, 0]
BINS = [3, 5, 8, 10]
CHANNELS = ["state", "action", "both"]
CONTRASTS = [("stagnation", "loop"), ("stagnation", "other"), ("loop", "other")]
STATE_T, ACTION_T = [0], list(range(1, 11))


# ---------------------------------------------------------------- metric
def paired_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, int]:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean()), int(len(pos) * len(neg))


def logo_auc(X, y, g, return_per_group=False):
    """Leave-one-group-out; within held-out group paired AUC; pooled by pairs."""
    per = {}
    for gv in np.unique(g):
        te = g == gv
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X[tr] - mu) / sd, y[tr])
        s = clf.decision_function((X[te] - mu) / sd)
        a, n = paired_auc(y[te], s)
        if n:
            per[int(gv)] = (a, n)
    if not per:
        return (np.nan, {}) if return_per_group else np.nan
    tot = sum(n for _, n in per.values())
    pooled = sum(a * n for a, n in per.values()) / tot
    return (pooled, per) if return_per_group else pooled


# ---------------------------------------------------------------- features
def build_features(hell_path: Path, ids: list[int], rng_shuffle=None):
    """-> dict[(block, denoise, family, nbins)] = (n_branch, nbins) array"""
    z = np.load(hell_path)
    feats = {}
    series_cache = {}
    for blk in BLOCKS:
        for dn in DENOISE:
            for fam, toks in (("state", STATE_T), ("action", ACTION_T)):
                series_cache[(blk, dn, fam)] = [
                    aggregate(z[str(i)], blk, dn, toks) for i in ids]
    for key, sers in series_cache.items():
        for nb in BINS:
            M = np.stack([phase_bins(s, nb) for s in sers])
            if rng_shuffle is not None:
                M = np.stack([rng_shuffle.permutation(r) for r in M])
            feats[key + (nb,)] = M
    return feats


def channel_matrix(feats, blk, dn, ch, nb):
    if ch == "state":
        return feats[(blk, dn, "state", nb)]
    if ch == "action":
        return feats[(blk, dn, "action", nb)]
    return np.hstack([feats[(blk, dn, "state", nb)], feats[(blk, dn, "action", nb)]])


# ---------------------------------------------------------------- corpora
def load_corpus_a():
    lab = pd.read_csv(OUT / "labels_A_rederived.csv")
    lab = lab[~lab["success"]].copy()
    lab["cls"] = np.where(lab["ref_stag"], "stagnation",
                          np.where(lab["ref_loop"], "loop", "other"))
    lab = lab.sort_values("episode_id").reset_index(drop=True)
    return lab, lab["worker"].to_numpy(), lab["episode_id"].tolist()


def load_corpus_b(stag_col="stag"):
    lab = pd.read_csv(OUT / "labels_B_derived.csv")
    lab = lab[~lab["success"]].copy()
    lab["cls"] = np.where(lab[stag_col], "stagnation",
                          np.where(lab["loop"], "loop", "other"))
    lab = lab.sort_values("episode_id").reset_index(drop=True)
    return lab, lab["init_state_id"].to_numpy(), lab["episode_id"].tolist()


def contrast_mask(cls, a, b):
    return np.isin(cls, [a, b]), None


def run_grid(feats, cls, groups, lengths, extra_cells=None):
    rows = []
    for (ca, cb) in CONTRASTS:
        m = np.isin(cls, [ca, cb])
        y = (cls[m] == ca).astype(int)
        g = groups[m]
        rows.append(dict(contrast=f"{ca}_vs_{cb}", block="-", denoise=-1, channel="length",
                         bins=0, auc=logo_auc(lengths[m][:, None].astype(float), y, g),
                         n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
        for blk in BLOCKS:
            for dn in DENOISE:
                for ch in CHANNELS:
                    for nb in BINS:
                        X = channel_matrix(feats, blk, dn, ch, nb)[m]
                        rows.append(dict(contrast=f"{ca}_vs_{cb}", block=blk, denoise=dn,
                                         channel=ch, bins=nb, auc=logo_auc(X, y, g),
                                         n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
    return pd.DataFrame(rows)


def permutation_null(feats, cls, groups, n_draws, seed, family_cells):
    """Shuffle class labels WITHIN group at branch level; recompute whole pipeline."""
    rng = np.random.default_rng(seed)
    null = np.full((n_draws, len(family_cells)), np.nan)
    for d in range(n_draws):
        perm = cls.copy()
        for gv in np.unique(groups):
            idx = np.flatnonzero(groups == gv)
            perm[idx] = rng.permutation(cls[idx])
        for j, (ca, cb, blk, dn, ch, nb) in enumerate(family_cells):
            m = np.isin(perm, [ca, cb])
            y = (perm[m] == ca).astype(int)
            if y.sum() < 2 or (1 - y).sum() < 2:
                continue
            null[d, j] = logo_auc(channel_matrix(feats, blk, dn, ch, nb)[m], y, groups[m])
        if (d + 1) % 20 == 0:
            print(f"    perm draw {d+1}/{n_draws}", flush=True)
    return null
