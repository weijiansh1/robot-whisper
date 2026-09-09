"""v9 = v8.3 + a spectral head on the routing time series.

Every head built so far measures the routing sequence in the *amplitude*
domain: mobility is a magnitude, curvature a second difference, recurrence a
single-lag similarity.  None of them measures **frequency**.  v7's recurrence
head is a crude proxy - it takes the maximum similarity over a few lags minus
the lag-1 similarity - but it never looks at the spectrum.

Measured on development and external, within task, at a fixed chunk, over
survivors only, with task-clustered bootstrap CIs:

| quantity | chunk | development | external |
|---|---|---|---|
| spectral centroid | 12 | 0.601 [0.554, 0.678]* | 0.564 [0.520, 0.649]* |
| spectral centroid | 20 | 0.602 [0.546, 0.664]* | 0.601 [0.534, 0.667]* |
| high-frequency share | 20 | 0.594 [0.527, 0.688]* | 0.599 [0.526, 0.678]* |

A null matched to this sweep's size (45 cells, episodes shuffled within each
chunk) tops out at |AUC-0.5| = 0.0478 over six draws; the centroid at chunk 20
reaches 0.102 / 0.101, 2.1x the floor, and the two independent cohorts agree to
two decimals.

Reading: a failing episode's routing **oscillates faster** - more of its power
sits at high frequency - rather than drifting smoothly.  That is a different
statement from "it moves more" or "it moves more sharply", which is why the
existing five heads miss it.

The centroid is a ratio of spectral moments and therefore already scale-free,
so like the front/back ratio it is **not** self-baselined.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V8DIR = ROOT / "moe-v8-0906"
sys.path.insert(0, str(V8DIR / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V  # noqa: E402
import freeze_v82 as F  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
LEADS = (0, 4, 8, 12, 16, 20)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
# v8.3, chosen on development in the previous step.
V83 = {"baseline": 2, "width": 6, "confirm": 2, "slope": -0.0015}
# Declared grid for the new head.
GRID = {"window": (6, 8, 10), "quantile": (0.95, 0.975, 0.99),
        "confirm": (1, 2), "slope": (0.0, -0.0015, -0.003)}
SEED = 20260906


def spectral_centroid(mobility: np.ndarray, window: int) -> np.ndarray:
    """Causal spectral centroid of the back-layer mobility sequence.

    At chunk q the transform sees only chunks q-window+1..q, so the head is
    strictly online.  The series is mean-removed inside the window, so the
    centroid describes the *shape* of the fluctuation, not its size.
    """
    series = np.nanmean(mobility[:, :, 4:, 9], axis=2)
    out = np.full_like(series, np.nan, dtype=np.float64)
    freqs = None
    for q in range(window - 1, series.shape[1]):
        segment = series[:, q - window + 1:q + 1]
        if np.isnan(segment).all():
            continue
        centred = np.nan_to_num(segment - np.nanmean(segment, axis=1, keepdims=True))
        power = np.abs(np.fft.rfft(centred, axis=1)) ** 2
        if freqs is None:
            freqs = np.arange(power.shape[1])
        total = power.sum(axis=1)
        valid = total > 1e-12
        centroid = np.full(series.shape[0], np.nan)
        centroid[valid] = (power[valid] * freqs).sum(axis=1) / total[valid]
        # only defined where the window is genuinely populated
        centroid[np.isnan(segment).any(axis=1)] = np.nan
        out[:, q] = centroid
    return out


def v83_alarms(data):
    heads = {c: F.build_heads(d["speed"], V83["baseline"], V83["width"])
             for c, d in data.items()}
    thr = F.thresholds_from(heads["development_main"])
    out = {}
    for cohort, d in data.items():
        firsts = [F.moving_first(heads[cohort][h], thr[h], V.DIRECTION[h],
                                 V83["slope"], V83["confirm"], V83["width"])
                  for h in heads[cohort]]
        out[cohort] = V.union(d["v7"], *firsts)
    return out


def head_alarms(centroid, threshold, slope, confirm, window):
    return F.moving_first(centroid, threshold, "high", slope, confirm, window)


def main() -> None:
    data = {c: V.load_cohort(c) for c in COHORTS}
    mob = {c: np.load(OUT / f"{c}_step_mobility.npz")["mobility"] for c in COHORTS}
    base = v83_alarms(data)

    def profile(alarms):
        prof = {lead: [0, 0] for lead in LEADS}
        for cohort, first in alarms.items():
            d = data[cohort]
            for lead in LEADS:
                s = V.score(first, d["risk"], d["length"], lead)
                prof[lead][0] += s["tp"]
                prof[lead][1] += s["fp"]
        return prof

    p83 = profile(base)
    print("v8.3 全量: " + "  ".join("lead>=%d %d/%d" % (b, *p83[b]) for b in LEADS))
    dev = data["development_main"]
    d83_0 = V.score(base["development_main"], dev["risk"], dev["length"], 0)
    d83_12 = V.score(base["development_main"], dev["risk"], dev["length"], 12)
    print("  development: lead>=0 %d/487 %dFP | lead>=12 %d/487"
          % (d83_0["tp"], d83_0["fp"], d83_12["tp"]))

    rows = []
    for window, q, confirm, slope in itertools.product(
            GRID["window"], GRID["quantile"], GRID["confirm"], GRID["slope"]):
        cd = spectral_centroid(mob["development_main"], window)
        pool = cd[np.isfinite(cd)]
        if pool.size < 1000:
            continue
        threshold = float(np.quantile(pool, q, method="lower"))
        head = head_alarms(cd, threshold, slope, confirm, window)
        u = V.union(base["development_main"], head)
        s0 = V.score(u, dev["risk"], dev["length"], 0)
        s12 = V.score(u, dev["risk"], dev["length"], 12)
        rows.append({"window": window, "quantile": q, "confirm": confirm,
                     "slope": slope, "threshold": threshold,
                     "dev0_tp": s0["tp"], "dev0_fp": s0["fp"],
                     "dev12_tp": s12["tp"]})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "spectral_grid.csv", index=False)

    # Rule, declared before scoring the sealed cohorts: on development only,
    # maximise TP at lead >= 12 subject to not losing ground at lead >= 0 and
    # adding at most 8 false alarms there.  The lead >= 0 constraint is what
    # stops a long-lead objective from degenerating into a length filter.
    ok = grid[(grid.dev0_tp >= d83_0["tp"]) & (grid.dev0_fp <= d83_0["fp"] + 8)]
    print("\n网格 %d 个；满足 dev(lead>=0) 不退步且新增FP<=8 的有 %d 个"
          % (len(grid), len(ok)))
    if not len(ok) or ok.dev12_tp.max() <= d83_12["tp"]:
        print("没有配置在 development 上改善 lead>=12 —— 谱头不成立，v8.3 保持不变")
        return
    pick = ok.sort_values(["dev12_tp", "dev0_fp"], ascending=[False, True]).iloc[0]
    print("选中: 窗口 W=%d, 分位 %.3f, 确认 K=%d, slope %+.4f, 阈值 %.4f"
          % (pick.window, pick["quantile"], pick.confirm, pick.slope, pick.threshold))
    print("  development lead>=12: %d/487 (v8.3 是 %d/487)"
          % (pick.dev12_tp, d83_12["tp"]))

    alarms = {}
    for cohort in COHORTS:
        cd = spectral_centroid(mob[cohort], int(pick.window))
        head = head_alarms(cd, float(pick.threshold), float(pick.slope),
                           int(pick.confirm), int(pick.window))
        alarms[cohort] = V.union(base[cohort], head)
    p9 = profile(alarms)
    np.savez_compressed(OUT / "v9_spectral_alarms.npz",
                        **{f"{c}|v9": v for c, v in alarms.items()})

    print("\n===== 全量 32,960 episodes / 1,358 失败 =====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%-2d" % b for b in LEADS)))
    print("%-6s %s" % ("v8.3", "  ".join("%4d/%-4d" % tuple(p83[b]) for b in LEADS)))
    print("%-6s %s" % ("v9", "  ".join("%4d/%-4d" % tuple(p9[b]) for b in LEADS)))

    print("\n===== 分 cohort (lead>=4) =====")
    for cohort in COHORTS:
        d = data[cohort]
        a = V.score(base[cohort], d["risk"], d["length"], 4)
        b = V.score(alarms[cohort], d["risk"], d["length"], 4)
        print("  %-18s v8.3 %3d/%-4d %3dFP -> v9 %3d/%-4d %3dFP  (%+d/%+d)"
              % (cohort, a["tp"], int(d["risk"].sum()), a["fp"], b["tp"],
                 int(d["risk"].sum()), b["fp"], b["tp"] - a["tp"], b["fp"] - a["fp"]))

    print("\n===== 分 suite (全量, lead>=4 与 lead>=12) =====")
    for suite in SUITES:
        cells = []
        n = 0
        for lead in (4, 12):
            agg = {"v8.3": [0, 0], "v9": [0, 0]}
            for cohort, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                if lead == 4:
                    n += int((m & d["risk"]).sum())
                for arm, first in (("v8.3", base[cohort]), ("v9", alarms[cohort])):
                    t = (first >= 0) & ((d["length"] - first) >= lead)
                    agg[arm][0] += int((t & m & d["risk"]).sum())
                    agg[arm][1] += int((t & m & ~d["risk"]).sum())
            cells.append((agg["v8.3"][0], agg["v9"][0], agg["v9"][1]))
        print("  %-16s L4 %3d->%3d (/%d, %dFP)   L12 %3d->%3d"
              % (suite, cells[0][0], cells[0][1], n, cells[0][2],
                 cells[1][0], cells[1][1]))

    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for cohort in COHORTS:
        m = mob[cohort].copy()
        for q in range(m.shape[1]):
            idx = np.flatnonzero(np.isfinite(m[:, q, 0, 0]))
            m[idx, q] = m[rng.permutation(idx), q]
        cd = spectral_centroid(m, int(pick.window))
        head = head_alarms(cd, float(pick.threshold), float(pick.slope),
                           int(pick.confirm), int(pick.window))
        s = V.score(V.union(base[cohort], head), data[cohort]["risk"],
                    data[cohort]["length"], 4)
        ntp += s["tp"]
        nfp += s["fp"]
    print("\n零对照(mobility 逐 chunk 打乱后重算谱头): %d/1358 TP %d FP"
          "  —— v9 是 %d/1358 %d FP" % (ntp, nfp, *p9[4]))

    (OUT / "v9_spectral_summary.json").write_text(json.dumps({
        "head": {k: (float(pick[k]) if k in ("quantile", "slope", "threshold")
                     else int(pick[k]))
                 for k in ("window", "quantile", "confirm", "slope", "threshold")},
        "v83_profile": {str(b): list(p83[b]) for b in LEADS},
        "v9_profile": {str(b): list(p9[b]) for b in LEADS},
        "null": {"tp": ntp, "fp": nfp},
    }, indent=2))


if __name__ == "__main__":
    main()
