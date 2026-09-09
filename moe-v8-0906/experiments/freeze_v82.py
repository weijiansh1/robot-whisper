"""Freeze v8.2 — the knee of the recall / false-alarm frontier.

v8 fixes its two new heads at a q1..q4 baseline, a causal width-6 mean, K=2
confirmation and a constant threshold.  Each of those four is a free choice
that was never swept.  Sweeping them jointly with a linearly relaxing threshold
produces a frontier, and the frontier has a clear knee:

    v8                        932/1358, 126 FP
    v8.2 (this)               959/1358, 131 FP   +27 TP / +5 FP   exchange 5.4
    v8.1 (slope -0.0025)     1008/1358, 149 FP   +76 TP / +23 FP  exchange 3.3

Past the knee the exchange rate drops from 5.4 to 2.7, so v8.2 is the point that
buys recall most cheaply in false alarms; v8.1 remains available when recall
matters more than precision.

Selection rule, declared before scoring the sealed cohorts: on
`development_main` only, maximise TP at lead >= 4 subject to (a) development
false alarms <= 47 and (b) dominating v8 at *every* lead cutoff, so the
configuration cannot buy aggregate recall by trading away early detections.

Nothing is trained.  Thresholds are order statistics of the unlabeled
development scores; the remaining four numbers are integers or one scalar slope
chosen from a declared grid.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V8  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
LEADS = (0, 2, 4, 8, 12, 16, 20)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
GRID = {"baseline": (2, 3, 4), "width": (3, 4, 6), "confirm": (2, 3),
        "slope": (-0.0010, -0.0015, -0.0020, -0.0025)}
MAX_DEV_FP = 47
SEED = 20260906


def trailing_mean(values, width):
    out = np.full_like(values, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(values[:, max(0, q - width + 1):q + 1], axis=1)
    return out


def build_heads(speed, baseline, width):
    """Both heads from the single [n, chunk, 8, 9] flow_speed array.

    The front/back ratio is already normalised across layers, so only the
    curvature head is self-baselined - forcing v7's self-baseline onto the
    ratio measurably destroys it.
    """
    with np.errstate(invalid="ignore"):
        path = np.nansum(speed, axis=3)
        front = np.nanmean(path[:, :, :4], axis=2)
        back = np.nanmean(path[:, :, 4:], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan

    sub = speed[:, :, V8.BACK][:, :, :, list(V8.CURVATURE_IDX)]
    curvature = np.abs(np.diff(sub, n=2, axis=-1)).mean(axis=(2, 3))
    base = np.nanmean(curvature[:, 1:1 + baseline], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(np.maximum(curvature, 1e-12) / np.maximum(base, 1e-12))
    rel[~np.isfinite(rel)] = np.nan
    return {"frontback_flowpath": trailing_mean(ratio, width),
            "curvature_3step": trailing_mean(rel, width)}


def moving_first(series, threshold, direction, slope, confirm, earliest):
    """Threshold relaxes linearly with the chunk index.  `>=` / `<=`, never
    strict: strict comparison drops the whole tie group at the threshold."""
    q = np.arange(series.shape[1])[None, :]
    moving = threshold + slope * q * (1 if direction == "high" else -1)
    hit = (series >= moving) if direction == "high" else (series <= moving)
    hit &= np.isfinite(series)
    hit[:, :earliest] = False
    held = np.zeros_like(hit)
    run = np.zeros(hit.shape[0], dtype=int)
    for step in range(hit.shape[1]):
        run = np.where(hit[:, step], run + 1, 0)
        held[:, step] = run >= confirm
    return np.where(held.any(axis=1), held.argmax(axis=1), -1)


def thresholds_from(heads_dev):
    out = {}
    for name, series in heads_dev.items():
        pool = series[np.isfinite(series)]
        level = V8.QUANTILE if V8.DIRECTION[name] == "high" else 1 - V8.QUANTILE
        out[name] = float(np.quantile(pool, level, method="lower"))
    return out


def evaluate(data, config):
    heads = {c: build_heads(d["speed"], config["baseline"], config["width"])
             for c, d in data.items()}
    thr = thresholds_from(heads["development_main"])
    earliest = config["width"]   # match v8: the causal mean needs `width` chunks, the baseline is read in parallel
    alarms, profile = {}, {lead: [0, 0] for lead in LEADS}
    for cohort, d in data.items():
        firsts = [moving_first(heads[cohort][h], thr[h], V8.DIRECTION[h],
                               config["slope"], config["confirm"], earliest)
                  for h in heads[cohort]]
        alarms[cohort] = V8.union(d["v7"], *firsts)
        for lead in LEADS:
            s = V8.score(alarms[cohort], d["risk"], d["length"], lead)
            profile[lead][0] += s["tp"]
            profile[lead][1] += s["fp"]
    return alarms, profile, thr, earliest


def main() -> None:
    data = {c: V8.load_cohort(c) for c in COHORTS}
    v8 = {c: np.load(ROOT / "moe-v8-0906/results/v8_full_corpus_alarms.npz")
          [f"{c}|v8"].astype(int) for c in COHORTS}
    v8_profile = {}
    for lead in LEADS:
        tp = sum(V8.score(v8[c], data[c]["risk"], data[c]["length"], lead)["tp"]
                 for c in COHORTS)
        fp = sum(V8.score(v8[c], data[c]["risk"], data[c]["length"], lead)["fp"]
                 for c in COHORTS)
        v8_profile[lead] = (tp, fp)
    assert v8_profile[4] == (932, 126), v8_profile[4]
    dev = data["development_main"]

    rows = []
    for baseline, width, confirm, slope in itertools.product(
            GRID["baseline"], GRID["width"], GRID["confirm"], GRID["slope"]):
        cfg = {"baseline": baseline, "width": width, "confirm": confirm,
               "slope": slope}
        alarms, profile, _, _ = evaluate(data, cfg)
        d4 = V8.score(alarms["development_main"], dev["risk"], dev["length"], 4)
        dominates = all(profile[lead][0] >= v8_profile[lead][0] for lead in LEADS)
        rows.append(cfg | {"dev_tp": d4["tp"], "dev_fp": d4["fp"],
                           "dominates_v8": dominates,
                           "total_tp": profile[4][0], "total_fp": profile[4][1]})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "v82_grid.csv", index=False)

    ok = grid[(grid.dev_fp <= MAX_DEV_FP) & grid.dominates_v8]
    print("网格 %d 个配置；满足 dev FP<=%d 且每档都压过 v8 的有 %d 个"
          % (len(grid), MAX_DEV_FP, len(ok)))
    pick = ok.sort_values(["dev_tp", "dev_fp"], ascending=[False, True]).iloc[0]
    cfg = {k: (int(pick[k]) if k != "slope" else float(pick[k]))
           for k in ("baseline", "width", "confirm", "slope")}
    print("选中 v8.2: 基线 q1..q%d, 平滑 W%d, 确认 K%d, slope %+.4f, 最早 q%d"
          % (cfg["baseline"], cfg["width"], cfg["confirm"], cfg["slope"],
             cfg["width"]))
    print("  development: %d/487 TP, %d FP" % (pick.dev_tp, pick.dev_fp))

    alarms, profile, thr, earliest = evaluate(data, cfg)
    np.savez_compressed(OUT / "v82_alarms.npz",
                        **{f"{c}|v8.2": v for c, v in alarms.items()})
    print("  阈值:", json.dumps({k: round(v, 6) for k, v in thr.items()}))

    print("\n===== 全量 32,960 episodes / 1,358 失败 / 31,602 成功 =====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%-2d" % b for b in LEADS)))
    print("%-6s %s" % ("v8", "  ".join("%4d/%-4d" % v8_profile[b] for b in LEADS)))
    print("%-6s %s" % ("v8.2", "  ".join("%4d/%-4d" % tuple(profile[b])
                                         for b in LEADS)))
    tp, fp = profile[4]
    print("\n  v8.2 lead>=4: 召回 %d/1358 = %.3f | 误报 %d (%.1f/千条成功)"
          " | 精度 %.3f" % (tp, tp / 1358, fp, 1000 * fp / 31602, tp / (tp + fp)))
    print("  相对 v8: %+d TP / %+d FP  兑换率 %.1f"
          % (tp - 932, fp - 126, (tp - 932) / max(fp - 126, 1)))

    print("\n===== 分 cohort (lead>=4) =====")
    for cohort in COHORTS:
        d = data[cohort]
        a = V8.score(v8[cohort], d["risk"], d["length"], 4)
        b = V8.score(alarms[cohort], d["risk"], d["length"], 4)
        print("  %-18s v8 %3d/%-4d %3dFP -> v8.2 %3d/%-4d %3dFP  (+%d/%+d)"
              % (cohort, a["tp"], int(d["risk"].sum()), a["fp"], b["tp"],
                 int(d["risk"].sum()), b["fp"], b["tp"] - a["tp"],
                 b["fp"] - a["fp"]))

    print("\n===== 分 suite (全量, lead>=4) =====")
    for suite in SUITES:
        agg = {"v8": [0, 0], "v8.2": [0, 0]}
        n = 0
        for cohort, d in data.items():
            m = d["suite"] == suite
            if not m.any():
                continue
            n += int((m & d["risk"]).sum())
            for arm, first in (("v8", v8[cohort]), ("v8.2", alarms[cohort])):
                t = (first >= 0) & ((d["length"] - first) >= 4)
                agg[arm][0] += int((t & m & d["risk"]).sum())
                agg[arm][1] += int((t & m & ~d["risk"]).sum())
        print("  %-16s v8 %3d/%-4d %3dFP -> **v8.2 %3d/%-4d %3dFP**"
              % (suite, agg["v8"][0], n, agg["v8"][1],
                 agg["v8.2"][0], n, agg["v8.2"][1]))

    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for cohort, d in data.items():
        heads = build_heads(d["speed"], cfg["baseline"], cfg["width"])
        shuffled = []
        for head, series in heads.items():
            s = series.copy()
            for q in range(s.shape[1]):
                col = s[:, q]
                idx = np.flatnonzero(np.isfinite(col))
                col[idx] = col[rng.permutation(idx)]
            shuffled.append(moving_first(s, thr[head], V8.DIRECTION[head],
                                         cfg["slope"], cfg["confirm"], earliest))
        sc = V8.score(V8.union(d["v7"], *shuffled), d["risk"], d["length"], 4)
        ntp += sc["tp"]
        nfp += sc["fp"]
    print("\n零对照(两个新头逐 chunk 打乱, v7 不变): %d/1358 TP %d FP  —— v8.2 是 %d/1358 %d FP"
          % (ntp, nfp, tp, fp))

    (OUT / "v82_summary.json").write_text(json.dumps({
        "config": cfg, "thresholds": thr, "earliest_chunk": earliest,
        "quantile": V8.QUANTILE, "curvature_steps": list(V8.CURVATURE_IDX),
        "episodes": 32960, "risks": 1358,
        "profile": {str(b): list(profile[b]) for b in LEADS},
        "v8_profile": {str(b): list(v8_profile[b]) for b in LEADS},
        "null": {"tp": ntp, "fp": nfp},
    }, indent=2))


if __name__ == "__main__":
    main()
