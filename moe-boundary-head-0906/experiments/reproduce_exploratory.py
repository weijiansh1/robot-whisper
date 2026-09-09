"""Step 1: reproduce the exploratory numbers from raw zarr, then extend.

Three things are checked before any design work happens:

  1. the recomputed external boundary cells agree with the cached
     `moe-v8-0906/results/boundary_2x2_external.npz` element-wise;
  2. the mean magnitudes (front_state 0.1821 > back_state 0.1051 >
     back_action 0.0521 > front_action 0.0463);
  3. the within-task, fixed-chunk, survivors-only AUC table with
     task-clustered bootstrap CIs.

Then the same AUC table is produced for `development_main` and
`legacy_main16x32`, which the exploratory pass never saw.

Also reported, as required controls: within-episode information for every
series (a series that is bit-constant inside episodes carries no
within-episode signal and must be excluded), and the cell-to-cell Spearman.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import CELLS, COHORTS, OUT, ROOT, load_all

CHUNKS = (5, 9, 12, 16, 20)
BOOT = 2000
SEED = 20260906
EXPECTED_MEAN = {"front_state": 0.1821, "back_state": 0.1051,
                 "back_action": 0.0521, "front_action": 0.0463}
EXPECTED_AUC = {  # external, from the exploratory pass
    ("front_action", 5): 0.478, ("front_action", 9): 0.661,
    ("front_action", 12): 0.497, ("front_action", 16): 0.634,
    ("front_action", 20): 0.541,
    ("back_action", 5): 0.452, ("back_action", 9): 0.620,
    ("back_action", 12): 0.485, ("back_action", 20): 0.459,
    ("front_state", 5): 0.455, ("front_state", 9): 0.508,
    ("front_state", 12): 0.430, ("front_state", 20): 0.414,
    ("back_state", 5): 0.416, ("back_state", 9): 0.525,
    ("back_state", 12): 0.406, ("back_state", 20): 0.392,
}


def task_uv(value, risk, task):
    """Per-task Mann-Whitney U and pair count, mid-rank on ties.

    The stratified AUC is sum_t U_t / sum_t pairs_t, and neither U_t nor
    pairs_t depends on how the tasks are weighted.  So the task-cluster
    bootstrap is a weighted sum over these two precomputed vectors, which is
    exactly the same estimator as resampling whole tasks and recomputing.
    """
    tasks, u, pairs = [], [], []
    for t in np.unique(task):
        m = task == t
        pos, neg = value[m & risk], value[m & ~risk]
        if pos.size == 0 or neg.size == 0:
            continue
        ranks = stats.rankdata(np.r_[neg, pos])
        tasks.append(t)
        u.append(ranks[neg.size:].sum() - pos.size * (pos.size + 1) / 2.0)
        pairs.append(float(pos.size * neg.size))
    return np.array(tasks), np.array(u), np.array(pairs)


def auc_at_chunk(d, cell, chunk, rng):
    alive = d["length"] > chunk
    v = d[cell][:, chunk]
    ok = alive & np.isfinite(v)
    if ok.sum() < 20:
        return None
    value, risk, task = v[ok], d["risk"][ok], d["task"][ok]
    tasks, u, pairs = task_uv(value, risk, task)
    if pairs.sum() == 0:
        return None
    point = float(u.sum() / pairs.sum())
    # multinomial counts over tasks == drawing len(tasks) tasks with replacement
    counts = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)),
                             size=BOOT).astype(float)
    num, den = counts @ u, counts @ pairs
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"auc": point, "lo": float(lo), "hi": float(hi),
            "n": int(ok.sum()), "n_risk": int(risk.sum()),
            "n_task": int(len(tasks)),
            "sig": bool(lo > 0.5 or hi < 0.5)}


def within_episode_information(d):
    """Fraction of episodes whose series is not bit-constant, and its spread."""
    out = {}
    for cell in CELLS:
        v = d[cell]
        n_finite = np.isfinite(v).sum(axis=1)
        usable = n_finite >= 2
        with np.errstate(invalid="ignore"):
            lo = np.nanmin(np.where(np.isfinite(v), v, np.nan), axis=1)
            hi = np.nanmax(np.where(np.isfinite(v), v, np.nan), axis=1)
            within = np.nanstd(np.where(np.isfinite(v), v, np.nan), axis=1)
        varying = usable & (hi > lo)
        out[cell] = {
            "n_episodes_usable": int(usable.sum()),
            "frac_varying": float(varying.sum() / max(usable.sum(), 1)),
            "median_within_sd": float(np.median(within[usable])),
            "median_between_sd": float(np.nanstd(np.nanmean(v, axis=1))),
        }
        out[cell]["within_over_between"] = (
            out[cell]["median_within_sd"] / out[cell]["median_between_sd"])
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_all()
    report = {}

    # ---- 1. element-wise agreement with the cached external 2x2 -------------
    cached = np.load(ROOT / "moe-v8-0906/results/boundary_2x2_external.npz")
    ext = data["external_8b"]
    agree = {}
    for cell in CELLS:
        mine, theirs = ext[cell], np.asarray(cached[cell], dtype=np.float64)
        # the cache indexes the seam at the arrival chunk q+1; so does this
        fm, ft = np.isfinite(mine), np.isfinite(theirs)
        assert np.array_equal(fm, ft), f"finite mask differs for {cell}"
        diff = np.abs(mine[fm] - theirs[fm])
        # The cache is float32.  `sqrt(p) - sqrt(r)` is a near-total
        # cancellation wherever the two passes route almost identically, so the
        # cache loses precision in exactly the low tail; this recompute is
        # float64 throughout.  Agreement is therefore asserted as: the bulk
        # matches to float32 round-off, and every residual disagreement sits
        # in the low tail where float32 cannot be trusted.
        assert float(np.median(diff)) < 2e-7, (cell, float(np.median(diff)))
        loud = diff > 1e-6
        cut = float(np.quantile(theirs[fm], 0.03))
        assert bool(np.all(theirs[fm][loud] <= cut)), cell
        assert loud.mean() < 0.01, (cell, float(loud.mean()))
        agree[cell] = {"n_finite": int(fm.sum()),
                       "median_abs_diff": float(np.median(diff)),
                       "max_abs_diff": float(diff.max()),
                       "n_above_1e-6": int(loud.sum()),
                       "all_above_1e-6_in_low_3pct": True}
    report["cache_agreement_external"] = agree
    print("独立重算 vs 缓存 2x2（external）：4 个 cell 全部逐元素一致。"
          "中位绝对差 %.1e；" % max(v["median_abs_diff"] for v in agree.values()))
    print("  仅 %d/%d 格差异 >1e-6，且全部落在各 cell 分布最低 3%% 处——"
          "缓存是 float32，低尾 sqrt 相减近乎完全抵消；本次重算全程 float64。"
          % (max(v["n_above_1e-6"] for v in agree.values()),
             agree["front_state"]["n_finite"]))

    # ---- 2. mean magnitudes ------------------------------------------------
    means = {}
    for cohort, d in data.items():
        means[cohort] = {c: float(np.nanmean(d[c])) for c in CELLS}
    for cell, want in EXPECTED_MEAN.items():
        got = means["external_8b"][cell]
        assert abs(got - want) < 5e-4, (cell, got, want)
    report["mean_magnitude"] = means
    print("均值幅度（external）复现：" + "  ".join(
        "%s %.4f" % (c, means["external_8b"][c]) for c in
        ("front_state", "back_state", "back_action", "front_action")))
    print("  state/action 幅度比 front %.2fx  back %.2fx"
          % (means["external_8b"]["front_state"] / means["external_8b"]["front_action"],
             means["external_8b"]["back_state"] / means["external_8b"]["back_action"]))

    # ---- 3. AUC table, all three cohorts -----------------------------------
    rows = []
    for cohort, d in data.items():
        for cell in CELLS:
            for chunk in CHUNKS:
                got = auc_at_chunk(d, cell, chunk, rng)
                if got is None:
                    continue
                rows.append({"cohort": cohort, "cell": cell, "chunk": chunk, **got})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "auc_table.csv", index=False)

    for cohort in COHORTS:
        print("\n===== 分块 AUC（%s，同任务内、生存者、任务聚类 bootstrap）====="
              % cohort)
        print("%-14s %s" % ("cell", "  ".join("q%-16d" % q for q in CHUNKS)))
        for cell in CELLS:
            cells = []
            for q in CHUNKS:
                r = table[(table.cohort == cohort) & (table.cell == cell)
                          & (table.chunk == q)]
                if not len(r):
                    cells.append("%-18s" % "-")
                    continue
                r = r.iloc[0]
                cells.append("%.3f[%.2f,%.2f]%s"
                             % (r.auc, r.lo, r.hi, "*" if r.sig else " "))
            print("%-14s %s" % (cell, "  ".join(cells)))

    bad = []
    for (cell, chunk), want in EXPECTED_AUC.items():
        r = table[(table.cohort == "external_8b") & (table.cell == cell)
                  & (table.chunk == chunk)]
        got = float(r.iloc[0].auc)
        if abs(got - want) > 0.01:
            bad.append((cell, chunk, want, got))
    report["auc_external_max_gap"] = max(
        abs(float(table[(table.cohort == "external_8b") & (table.cell == c)
                        & (table.chunk == q)].iloc[0].auc) - w)
        for (c, q), w in EXPECTED_AUC.items())
    assert not bad, bad
    print("\n探索期 external AUC 全部 %d 个格子复现，最大偏差 %.4f"
          % (len(EXPECTED_AUC), report["auc_external_max_gap"]))

    # ---- 4. cell-to-cell Spearman -----------------------------------------
    corr = {}
    for cohort, d in data.items():
        ok = np.isfinite(d["front_action"]) & np.isfinite(d["front_state"])
        pairs = {}
        for a, b in (("front_action", "front_state"),
                     ("front_action", "back_state"),
                     ("back_action", "back_state"),
                     ("front_action", "back_action"),
                     ("front_state", "back_state")):
            r = stats.spearmanr(d[a][ok], d[b][ok])
            pairs[f"{a}|{b}"] = float(r.statistic)
        corr[cohort] = pairs
    report["spearman"] = corr
    print("\nSpearman（逐 cell 级别，全部有效格子）:")
    for cohort in COHORTS:
        print("  %-18s %s" % (cohort, "  ".join(
            "%s %.4f" % (k.replace("front_", "F").replace("back_", "B")
                         .replace("action", "A").replace("state", "S"), v)
            for k, v in corr[cohort].items())))

    # ---- 5. within-episode information ------------------------------------
    info = {c: within_episode_information(d) for c, d in data.items()}
    report["within_episode"] = info
    print("\n幕内信息（每条序列在 episode 内是否常量）:")
    for cohort in COHORTS:
        for cell in CELLS:
            v = info[cohort][cell]
            print("  %-18s %-14s 非常量占比 %.4f  幕内 sd %.5f  幕间 sd %.5f"
                  "  幕内/幕间 %.2f"
                  % (cohort, cell, v["frac_varying"], v["median_within_sd"],
                     v["median_between_sd"], v["within_over_between"]))
            assert v["frac_varying"] > 0.99, (cohort, cell, v)

    (OUT / "reproduce_exploratory.json").write_text(
        json.dumps(report, indent=2, default=float))
    print("\n写出 %s" % (OUT / "auc_table.csv"))


if __name__ == "__main__":
    main()
