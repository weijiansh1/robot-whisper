"""v9 = v8 + a chunk-gated per-step mobility head.

v8's weakness is concentrated: pooled over all three cohorts it reaches
libero_long 592/712 and libero_goal 181/250 but only libero_object 41/81 and
libero_spatial 118/315.  On development, within-task fixed-chunk AUC in exactly
those two suites is high and sharply localised - libero_object scores
0.846-0.892 at chunk 9 across all eight layers, libero_spatial peaks near 0.72
at chunk 7 - at denoising steps v7 never reads.

v7 takes mobility at the final step only, medians it over L12..L15,
self-baselines it against q1..q4 and applies a causal width-6 mean.  Every one
of those is a smoother, and a width-6 trailing mean spreads a one-chunk peak
over six chunks.  So this head does the opposite: no smoothing, a single
(layer, step) cell, and it is only allowed to fire at a small set of gated
chunks.

Nothing is trained.  The threshold is an order statistic of the unlabeled
development scores; the cell and the gate are chosen from a small grid by a
declared rule on development.

**Selection is at lead >= 0.** Selecting at lead >= 4 is unsafe here: risk
episodes run to the cap and successes do not, so a lead filter removes false
alarms far faster than true ones and any chunk gate can exploit it.  A previous
attempt produced "+39 TP / +0 FP" at lead >= 4 that was really 82 TP / 73 FP at
lead >= 0, with 600 of 685 alarms on successful episodes.
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
sys.path.insert(0, str(ROOT / "moe-v8-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V8  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
LEADS = (0, 2, 4, 8, 12)
SEED = 20260906

# Small declared grid.  Gates are absolute chunk indices - legal because a
# chunk counter needs no horizon.
GATES = {
    "q7_9": (7, 8, 9),
    "q7_11": (7, 8, 9, 10, 11),
    "q6_10": (6, 7, 8, 9, 10),
    "q9_13": (9, 10, 11, 12, 13),
}
QUANTILES = (0.95, 0.96, 0.97, 0.98, 0.99, 0.995)
CONFIRMS = (1, 2)
# Selection rule, written before scoring the sealed cohorts.
MIN_EXCHANGE = 2.0          # TP gained per FP added, at lead >= 0
MAX_DEV_FP_ADDED = 40


def load_scores(cohort: str) -> np.ndarray:
    return np.load(OUT / f"{cohort}_step_mobility.npz")["mobility"]


def gated_first(series: np.ndarray, threshold: float, gate: tuple[int, ...],
                confirm: int) -> np.ndarray:
    """First gated chunk where the value is at or above the threshold, held for
    `confirm` consecutive gated chunks.  `>=`, never `>`."""
    hit = np.zeros(series.shape[:2], dtype=bool)
    ok = np.isfinite(series)
    for q in gate:
        if q < series.shape[1]:
            hit[:, q] = ok[:, q] & (series[:, q] >= threshold)
    if confirm > 1:
        held = np.zeros_like(hit)
        run = np.zeros(hit.shape[0], dtype=int)
        for q in range(hit.shape[1]):
            inside = q in gate
            run = np.where(hit[:, q], run + 1, np.where(inside, 0, run))
            held[:, q] = hit[:, q] & (run >= confirm)
        hit = held
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {c: V8.load_cohort(c) for c in COHORTS}
    mob = {c: load_scores(c) for c in COHORTS}
    v8 = {c: np.load(ROOT / "moe-v8-0906/results/v8_full_corpus_alarms.npz")
          [f"{c}|v8"].astype(int) for c in COHORTS}

    base = {c: V8.score(v8[c], data[c]["risk"], data[c]["length"], 0)
            for c in COHORTS}
    b4 = {c: V8.score(v8[c], data[c]["risk"], data[c]["length"], 4)
          for c in COHORTS}
    tot4 = (sum(b4[c]["tp"] for c in COHORTS), sum(b4[c]["fp"] for c in COHORTS))
    assert tot4 == (932, 126), tot4
    print("v8 锚点 lead>=4 全量 %d/1358 TP, %d FP" % tot4)

    dev = data["development_main"]
    dmob = mob["development_main"]

    def build(m, spec):
        """Single cell, or an aggregate over layers and/or steps.

        The per-suite AUC signal is present at *every* layer simultaneously
        (libero_object at chunk 9 scores 0.846-0.892 across all eight), so an
        aggregate should be far more stable than any single cell.
        """
        kind, li, step = spec
        if kind == "cell":
            return m[:, :, li, step].astype(float)
        if kind == "layers_med":
            return np.nanmedian(m[:, :, :, step].astype(float), axis=2)
        if kind == "front_med":
            return np.nanmedian(m[:, :, :4, step].astype(float), axis=2)
        if kind == "back_med":
            return np.nanmedian(m[:, :, 4:, step].astype(float), axis=2)
        if kind == "steps_med":
            return np.nanmedian(m[:, :, li, :].astype(float), axis=2)
        if kind == "all_med":
            return np.nanmedian(m.astype(float).reshape(m.shape[0], m.shape[1], -1),
                                axis=2)
        raise ValueError(kind)

    SPECS = ([("cell", li, st) for li in range(8) for st in range(10)]
             + [("layers_med", 0, st) for st in range(10)]
             + [("front_med", 0, st) for st in range(10)]
             + [("back_med", 0, st) for st in range(10)]
             + [("steps_med", li, 0) for li in range(8)]
             + [("all_med", 0, 0)])
    rows = []
    for spec, gname, q, k in itertools.product(SPECS, GATES, QUANTILES, CONFIRMS):
        kind, li, step = spec
        series = build(dmob, spec)
        gate = GATES[gname]
        pool = series[:, list(gate)]
        pool = pool[np.isfinite(pool)]
        if pool.size < 1000:
            continue
        threshold = float(np.quantile(pool, q, method="lower"))
        first = gated_first(series, threshold, gate, k)
        u = V8.union(v8["development_main"], first)
        s = V8.score(u, dev["risk"], dev["length"], 0)
        gain_tp = s["tp"] - base["development_main"]["tp"]
        gain_fp = s["fp"] - base["development_main"]["fp"]
        rows.append({"kind": kind, "layer": LAYERS[li], "li": li,
                     "step": step, "gate": gname,
                     "quantile": q, "confirm": k, "threshold": threshold,
                     "dev_gain_tp": gain_tp, "dev_gain_fp": gain_fp,
                     "exchange": gain_tp / gain_fp if gain_fp > 0 else np.inf})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "v9_development_grid.csv", index=False)

    ok = grid[(grid.exchange >= MIN_EXCHANGE) &
              (grid.dev_gain_fp <= MAX_DEV_FP_ADDED) & (grid.dev_gain_tp > 0)]
    print("development 网格 %d 个配置, 满足 兑换率>=%.1f 且 新增FP<=%d 的有 %d 个"
          % (len(grid), MIN_EXCHANGE, MAX_DEV_FP_ADDED, len(ok)))
    if not len(ok):
        print("没有配置通过声明的规则 —— v9 不成立，v8 保持不变")
        return
    pick = ok.sort_values(["dev_gain_tp", "dev_gain_fp"],
                          ascending=[False, True]).iloc[0]
    print("选中: %s (%s d%d) gate %s q%.3f K%d  阈值 %.6f"
          % (pick.kind, pick.layer, pick.step, pick.gate, pick["quantile"],
             pick.confirm, pick.threshold))
    print("  development 净增 (lead>=0): +%d TP / +%d FP  兑换率 %.2f"
          % (pick.dev_gain_tp, pick.dev_gain_fp, pick.exchange))

    alarms, out_rows = {}, []
    for c in COHORTS:
        series = build(mob[c], (pick.kind, int(pick.li), int(pick.step)))
        head = gated_first(series, float(pick.threshold),
                           GATES[pick.gate], int(pick.confirm))
        v9 = V8.union(v8[c], head)
        alarms[c] = {"v8": v8[c], "head": head, "v9": v9}
        for arm, first in alarms[c].items():
            row = {"cohort": c, "arm": arm, "n_risk": int(data[c]["risk"].sum())}
            for lead in LEADS:
                s = V8.score(first, data[c]["risk"], data[c]["length"], lead)
                row |= {f"tp{lead}": s["tp"], f"fp{lead}": s["fp"]}
            row["median_lead"] = V8.score(first, data[c]["risk"],
                                          data[c]["length"])["median_lead"]
            out_rows.append(row)
    table = pd.DataFrame(out_rows)
    table.to_csv(OUT / "v9_operating_points.csv", index=False)
    np.savez_compressed(OUT / "v9_alarms.npz",
                        **{f"{c}|{k}": v for c, f in alarms.items()
                           for k, v in f.items()})

    print("\n===== 全量 (32,960 episodes, 1,358 失败) =====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%d" % b for b in LEADS)))
    for arm in ("v8", "v9"):
        cells = []
        for lead in LEADS:
            tp = int(table[table.arm == arm][f"tp{lead}"].sum())
            fp = int(table[table.arm == arm][f"fp{lead}"].sum())
            cells.append("%4d/%-4d" % (tp, fp))
        print("%-6s %s" % (arm, "  ".join(cells)))

    print("\n===== 分 cohort (lead>=4) =====")
    for c in COHORTS:
        a = table[(table.cohort == c) & (table.arm == "v8")].iloc[0]
        b = table[(table.cohort == c) & (table.arm == "v9")].iloc[0]
        print("  %-18s %3d/%-4d %3d FP  ->  %3d/%-4d %3d FP   (+%d/+%d)"
              % (c, a.tp4, a.n_risk, a.fp4, b.tp4, b.n_risk, b.fp4,
                 b.tp4 - a.tp4, b.fp4 - a.fp4))

    print("\n===== 分 suite (全量, lead>=4) =====")
    for suite in ("libero_goal", "libero_long", "libero_object", "libero_spatial"):
        agg = {"v8": [0, 0], "v9": [0, 0], "n": 0}
        for c in COHORTS:
            d = data[c]
            m = d["suite"] == suite
            if not m.any():
                continue
            agg["n"] += int((m & d["risk"]).sum())
            for arm in ("v8", "v9"):
                f = alarms[c][arm]
                t = (f >= 0) & ((d["length"] - f) >= 4)
                agg[arm][0] += int((t & m & d["risk"]).sum())
                agg[arm][1] += int((t & m & ~d["risk"]).sum())
        print("  %-16s %3d/%-4d %3d FP  ->  %3d/%-4d %3d FP   (+%d/+%d)"
              % (suite, agg["v8"][0], agg["n"], agg["v8"][1],
                 agg["v9"][0], agg["n"], agg["v9"][1],
                 agg["v9"][0] - agg["v8"][0], agg["v9"][1] - agg["v8"][1]))

    # Null: shuffle the head's series across episodes within each chunk.
    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for c in COHORTS:
        s = build(mob[c], (pick.kind, int(pick.li), int(pick.step))).copy()
        for q in range(s.shape[1]):
            col = s[:, q]
            idx = np.flatnonzero(np.isfinite(col))
            col[idx] = col[rng.permutation(idx)]
        u = V8.union(v8[c], gated_first(s, float(pick.threshold),
                                        GATES[pick.gate], int(pick.confirm)))
        sc = V8.score(u, data[c]["risk"], data[c]["length"], 4)
        ntp += sc["tp"]
        nfp += sc["fp"]
    v9tot = (int(table[table.arm == "v9"].tp4.sum()),
             int(table[table.arm == "v9"].fp4.sum()))
    print("\n零对照(逐 chunk 打乱该头的序列): %d/1358 TP %d FP  —— v9 是 %d/1358 %d FP"
          % (ntp, nfp, *v9tot))

    (OUT / "v9_summary.json").write_text(json.dumps({
        "selected": {k: (int(v) if isinstance(v, (np.integer,)) else
                         float(v) if isinstance(v, (np.floating,)) else v)
                     for k, v in pick.to_dict().items()},
        "v8": {"tp": tot4[0], "fp": tot4[1]},
        "v9": {"tp": v9tot[0], "fp": v9tot[1]},
        "null": {"tp": ntp, "fp": nfp},
    }, indent=2))


if __name__ == "__main__":
    main()
