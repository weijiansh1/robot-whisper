"""Step 2b: the pairwise rank-difference scan, on all three cohorts.

Eight base series - {boundary, within-chunk} x {front L2-L5, back L12-L15} x
{action tokens 1-10, state token 0} - percentile-ranked per chunk, then every
unordered pair subtracted.  28 pairs x 3 chunks = 84 cells per cohort, the same
grid the exploratory pass ran on external_8b alone.

Gain of a pair is

    | AUC(A - B) - 0.5 |  -  max( |AUC(A) - 0.5|, |AUC(B) - 0.5| )

so it is positive only when the contrast beats both of its own ingredients.

What this script adds over the exploratory pass, and why:

  1. **All three cohorts.**  The argmax cell is a 1-of-84 selection; whether it
     is the same cell in development, external and legacy is the whole question.
     The cell is selected on development alone.
  2. **A task-clustered bootstrap CI on the gain itself**, not on the two AUCs
     separately.  The three AUCs are recomputed inside every bootstrap draw from
     precomputed per-task U statistics, so the difference is resampled jointly
     and its CI is not the (much wider) interval you would get by differencing
     two marginal CIs.
  3. **An autocorrelation-preserving null.**  Shuffling values across episodes
     within a chunk destroys within-episode time structure and so sets too low a
     bar.  Two harder surrogates are used instead:
       * `episode_swap` - permute *whole episode series* among episodes of the
         same task and the same length.  Every episode's autocorrelation, its
         marginal, and the per-chunk marginal within the stratum are preserved
         exactly; only which outcome the series belongs to is destroyed.  This
         is the null that matters for a fixed-chunk AUC.
       * `circular_shift` - independently roll each episode's finite segment.
         Preserves each episode's autocorrelation exactly and destroys the
         alignment between chunk index and value.
     `flat_shuffle`, the exploratory pass's null, is run too so the three are
     directly comparable.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import BASE_CELLS, COHORTS, OUT, chunk_rank, load_all

CHUNKS = (9, 16, 20)
DEV_CHUNKS = (5, 7, 9, 12, 16, 20)
BOOT = 4000
NULL_DRAWS = 24
SEED = 20260906
SERIES = tuple(f"{fam}:{cell}" for fam in ("bnd", "wc") for cell in BASE_CELLS)
PAIRS = tuple(itertools.combinations(SERIES, 2))


def series_of(d, name, ranked):
    return ranked[name]


def ranked_cells(d) -> dict[str, np.ndarray]:
    """Per-chunk percentile rank of each of the eight base series.

    Ranked inside each cohort, matching the exploratory protocol: the question
    here is whether the *argmax cell* agrees across cohorts, which needs each
    cohort ranked on its own terms.  The detector in `freeze_and_score.py`
    instead ranks against the frozen development reference.
    """
    out = {}
    for fam in ("bnd", "wc"):
        for cell in BASE_CELLS:
            raw = d[cell] if fam == "bnd" else d[f"wc_{cell}"]
            out[f"{fam}:{cell}"] = chunk_rank(raw, raw)
    return out


def task_uv(value, risk, task, tasks):
    """Per-task Mann-Whitney U and pair counts, aligned to a fixed task list."""
    u = np.zeros(len(tasks))
    pairs = np.zeros(len(tasks))
    for i, t in enumerate(tasks):
        m = task == t
        pos, neg = value[m & risk], value[m & ~risk]
        if pos.size == 0 or neg.size == 0:
            continue
        r = stats.rankdata(np.r_[neg, pos])
        u[i] = r[neg.size:].sum() - pos.size * (pos.size + 1) / 2.0
        pairs[i] = pos.size * neg.size
    return u, pairs


def gain_with_ci(d, ranked, a, b, chunk, rng, boot=BOOT):
    alive = d["length"] > chunk
    va, vb = ranked[a][:, chunk], ranked[b][:, chunk]
    ok = alive & np.isfinite(va) & np.isfinite(vb)
    if ok.sum() < 20:
        return None
    risk, task = d["risk"][ok], d["task"][ok]
    tasks = np.unique(task)
    if len(tasks) < 3:
        return None
    ua, pa = task_uv(va[ok], risk, task, tasks)
    ub, _ = task_uv(vb[ok], risk, task, tasks)
    ud, _ = task_uv(va[ok] - vb[ok], risk, task, tasks)

    def gains(counts):
        den = counts @ pa
        with np.errstate(invalid="ignore", divide="ignore"):
            aa = np.abs(counts @ ua / den - 0.5)
            bb = np.abs(counts @ ub / den - 0.5)
            dd = np.abs(counts @ ud / den - 0.5)
        return dd - np.maximum(aa, bb), aa, bb, dd

    one = np.ones(len(tasks))
    point, absa, absb, absd = gains(one)
    counts = rng.multinomial(len(tasks), one / len(tasks), size=boot).astype(float)
    draws, *_ = gains(counts)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"a": a, "b": b, "chunk": chunk, "gain": float(point),
            "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0),
            "abs_a": float(absa), "abs_b": float(absb), "abs_diff": float(absd),
            "n": int(ok.sum()), "n_task": int(len(tasks))}


# -------------------------------------------------------------------------
# surrogates
# -------------------------------------------------------------------------
def flat_shuffle(series, d, rng):
    """The exploratory pass's null: permute across episodes within each chunk."""
    out = series.copy()
    for q in range(out.shape[1]):
        col = out[:, q]
        ok = np.flatnonzero(np.isfinite(col))
        col[ok] = col[rng.permutation(ok)]
    return out


def circular_shift(series, d, rng):
    """Roll each episode's finite segment by a random amount.

    Preserves that episode's autocorrelation (circularly) and its own marginal.
    """
    out = series.copy()
    for i in range(out.shape[0]):
        ok = np.flatnonzero(np.isfinite(out[i]))
        if ok.size > 1:
            out[i, ok] = np.roll(out[i, ok], int(rng.integers(ok.size)))
    return out


def episode_swap(series, d, rng):
    """Permute whole episode series among episodes of the same task and length.

    The hardest of the three: within a stratum nothing about the time series
    changes at all, only which episode - and therefore which outcome - it is
    attached to.
    """
    out = series.copy()
    key = np.char.add(d["task"].astype(str), np.char.mod("|%d", d["length"]))
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        if idx.size > 1:
            out[idx] = series[rng.permutation(idx)]
    return out


SURROGATES = {"flat_shuffle": flat_shuffle, "circular_shift": circular_shift,
              "episode_swap": episode_swap}


def scan(d, ranked, chunks, rng, boot=BOOT):
    rows = []
    for a, b in PAIRS:
        for chunk in chunks:
            got = gain_with_ci(d, ranked, a, b, chunk, rng, boot)
            if got:
                rows.append(got)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_all()
    ranked = {c: ranked_cells(d) for c, d in data.items()}

    # ---- development first: this is where the cell is selected --------------
    dev = scan(data["development_main"], ranked["development_main"],
               DEV_CHUNKS, rng)
    dev.to_csv(OUT / "pairwise_development.csv", index=False)
    print("===== development：%d 个格子（28 对 x %d 个 chunk）====="
          % (len(dev), len(DEV_CHUNKS)))
    print("均值增益 %.4f   中位 %.4f   显著为正 %d/%d"
          % (dev.gain.mean(), dev.gain.median(), int(dev.sig.sum()), len(dev)))
    cols = ["a", "b", "chunk", "abs_a", "abs_b", "abs_diff", "gain", "lo", "hi", "sig"]
    print("\n-- development 增益前 12 --")
    print(dev.nlargest(12, "gain")[cols].to_string(index=False))

    # ---- the same 84-cell grid on all three, for cross-cohort agreement -----
    scans = {}
    for cohort in COHORTS:
        s = scan(data[cohort], ranked[cohort], CHUNKS, rng)
        s["cohort"] = cohort
        scans[cohort] = s
    every = pd.concat(scans.values(), ignore_index=True)
    every.to_csv(OUT / "pairwise_all_cohorts.csv", index=False)

    print("\n===== 三个 cohort 上同一 84 格网格 =====")
    print("%-18s %8s %8s %10s %-46s %8s"
          % ("cohort", "均值增益", "中位", "显著为正", "argmax 格子", "增益"))
    argmax = {}
    for cohort in COHORTS:
        s = scans[cohort]
        top = s.loc[s.gain.idxmax()]
        argmax[cohort] = (top.a, top.b, int(top.chunk))
        print("%-18s %8.4f %8.4f %6d/%-3d %-46s %8.4f"
              % (cohort, s.gain.mean(), s.gain.median(), int(s.sig.sum()),
                 len(s), "%s - %s @q%d" % (top.a, top.b, top.chunk), top.gain))

    print("\n各 cohort 的 argmax 是否一致: %s"
          % ("是" if len(set(argmax.values())) == 1 else
             "否 —— " + " | ".join("%s: %s-%s@q%d" % (c, *v)
                                   for c, v in argmax.items())))

    # the exploratory argmax, scored everywhere
    # stored in `itertools.combinations` order, so normalise before looking up
    probe = tuple(sorted(("wc:back_action", "bnd:back_action"),
                         key=SERIES.index)) + (16,)
    print("\n===== 探索期的 argmax（%s - %s @ q%d）在三个 cohort 上 ====="
          % probe)
    print("%-18s %8s %8s %8s %8s %-18s %s"
          % ("cohort", "|A-.5|", "|B-.5|", "|差-.5|", "增益", "95% CI", "显著"))
    probe_rows = []
    for cohort in COHORTS:
        s = scans[cohort]
        r = s[(s.a == probe[0]) & (s.b == probe[1]) & (s.chunk == probe[2])]
        assert len(r) == 1, (cohort, probe, len(r))
        r = r.iloc[0]
        probe_rows.append({"cohort": cohort, **r[cols].to_dict()})
        print("%-18s %8.4f %8.4f %8.4f %8.4f [%+.4f,%+.4f]  %s"
              % (cohort, r.abs_a, r.abs_b, r.abs_diff, r.gain, r.lo, r.hi,
                 "是" if r.sig else "否"))

    # rank of that cell inside each cohort's own 84
    print("\n该格子在各 cohort 84 格中的排名（1 = 最好）:")
    for cohort in COHORTS:
        s = scans[cohort].sort_values("gain", ascending=False).reset_index(drop=True)
        hit = s[(s.a == probe[0]) & (s.b == probe[1]) & (s.chunk == probe[2])]
        assert len(hit) == 1, (cohort, probe)
        print("  %-18s %d / %d   增益 %+.4f  CI [%+.4f,%+.4f]"
              % (cohort, int(hit.index[0]) + 1, len(s),
                 float(hit.iloc[0].gain), float(hit.iloc[0].lo),
                 float(hit.iloc[0].hi)))

    # ---- nulls -------------------------------------------------------------
    print("\n===== 零对照：同样 84 格扫描，%d 次抽样 =====" % NULL_DRAWS)
    print("零假设越强，最大增益的分布越高；观测值必须超过它")
    null_rows = []
    dext = data["external_8b"]
    for kind, fn in SURROGATES.items():
        maxima = []
        for draw in range(NULL_DRAWS):
            r2 = np.random.default_rng(SEED + 1000 * draw + hash(kind) % 997)
            shuffled = {}
            for name in SERIES:
                fam, cell = name.split(":")
                raw = dext[cell] if fam == "bnd" else dext[f"wc_{cell}"]
                shuffled[name] = chunk_rank(fn(raw, dext, r2), raw)
            s = scan(dext, shuffled, CHUNKS, r2, boot=1)
            maxima.append(float(s.gain.max()))
        maxima = np.array(maxima)
        obs = float(scans["external_8b"].gain.max())
        null_rows.append({"surrogate": kind, "median_max_gain": float(np.median(maxima)),
                          "worst_max_gain": float(maxima.max()),
                          "observed_max_gain": obs,
                          "ratio_to_worst": obs / maxima.max(),
                          "p_exceed": float((maxima >= obs).mean()),
                          "n_draws": NULL_DRAWS})
        print("  %-15s 最大增益 中位 %.4f  最坏 %.4f  ——  观测 %.4f  = %.1fx 最坏  p=%.3f"
              % (kind, np.median(maxima), maxima.max(), obs,
                 obs / maxima.max(), (maxima >= obs).mean()))
    pd.DataFrame(null_rows).to_csv(OUT / "pairwise_nulls.csv", index=False)

    (OUT / "pairwise_scan.json").write_text(json.dumps({
        "argmax_per_cohort": {c: list(v) for c, v in argmax.items()},
        "argmax_agrees": len(set(argmax.values())) == 1,
        "exploratory_cell": list(probe),
        "exploratory_cell_per_cohort": probe_rows,
        "mean_gain": {c: float(scans[c].gain.mean()) for c in COHORTS},
        "n_sig": {c: int(scans[c].sig.sum()) for c in COHORTS},
        "n_cells": {c: int(len(scans[c])) for c in COHORTS},
        "nulls": null_rows,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
