"""Freeze v8.1: v8 with a threshold that relaxes linearly with the chunk index.

v8 uses one constant threshold per head.  The decision problem is not constant:
among episodes still running, the fraction that will fail rises from about 0.05
to about 0.44 across the window, so identical evidence is worth more later than
earlier.  A threshold that relaxes with the chunk index encodes that, and it is
a deterministic function of a counter - it cannot be dragged along by the
anomaly the way an adaptive threshold can.

Four alternatives were tested and all lost:

  * running median + MAD of the episode's own past: 935/1358 at 963 FP (7.6x)
  * expanding self-baseline replacing the fixed q1..q4: 920/1358 at 117 FP
  * per-chunk z-normalisation against the reference: 933/1358 at 596 FP (4.7x)
  * chunk-gated per-step mobility head: +8 TP / +3 FP over 5,712 configurations

Three richer shapes for the relaxation were also tested at a matched
development budget - `-log S(q)`, `1 - S(q)` and `sqrt(q)`, where S is the
label-free survival fraction - and all landed at 1050/1358 with 177-189 FP
against linear's 1050/1358 with 173 FP.  **The shape does not matter, only the
amount of relaxation**, so the simplest form is kept.

Nothing here is trained: thresholds are order statistics of the unlabeled
development scores, the slope is one scalar chosen on development against a
declared false-alarm budget, and the rule is boolean.
"""

from __future__ import annotations

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
SLOPE = -0.0025          # chosen on development at a budget of 55 development FP
LEADS = (0, 2, 4, 8, 12, 16)
SEED = 20260906
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")


def moving_first(series, threshold, direction, slope,
                 confirm=V8.CONFIRM, earliest=V8.WIDTH):
    """Threshold moves linearly with the chunk index; negative slope relaxes it.

    The sign flip for `low` heads keeps "negative slope = more permissive" true
    for both directions.  `>=` / `<=`, never strict.
    """
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


def main() -> None:
    data = {c: V8.load_cohort(c) for c in COHORTS}
    heads = {c: V8.heads_from_flow_speed(d["speed"]) for c, d in data.items()}

    thresholds = {}
    for name, series in heads["development_main"].items():
        pool = series[np.isfinite(series)]
        level = V8.QUANTILE if V8.DIRECTION[name] == "high" else 1 - V8.QUANTILE
        thresholds[name] = float(np.quantile(pool, level, method="lower"))

    alarms, rows = {}, []
    for cohort, d in data.items():
        arms = {"v7": d["v7"]}
        flat = [V8.confirmed_first(heads[cohort][h], thresholds[h],
                                   V8.DIRECTION[h], V8.CONFIRM) for h in heads[cohort]]
        arms["v8"] = V8.union(d["v7"], *flat)
        moved = [moving_first(heads[cohort][h], thresholds[h],
                              V8.DIRECTION[h], SLOPE) for h in heads[cohort]]
        arms["v8.1"] = V8.union(d["v7"], *moved)
        alarms[cohort] = arms
        for arm, first in arms.items():
            row = {"cohort": cohort, "arm": arm, "n_risk": int(d["risk"].sum()),
                   "n_safe": int((~d["risk"]).sum())}
            for lead in LEADS:
                s = V8.score(first, d["risk"], d["length"], lead)
                row |= {f"tp{lead}": s["tp"], f"fp{lead}": s["fp"]}
            row |= V8.score(first, d["risk"], d["length"])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "v81_operating_points.csv", index=False)
    np.savez_compressed(OUT / "v81_alarms.npz",
                        **{f"{c}|{k}": v for c, a in alarms.items()
                           for k, v in a.items()})

    def total(arm, lead):
        sub = table[table.arm == arm]
        return int(sub[f"tp{lead}"].sum()), int(sub[f"fp{lead}"].sum())

    print("全量 32,960 episodes / 1,358 失败 / 31,602 成功")
    print("\n===== 提前量剖面 =====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%-2d" % b for b in LEADS)))
    for arm in ("v7", "v8", "v8.1"):
        print("%-6s %s" % (arm, "  ".join("%4d/%-4d" % total(arm, b) for b in LEADS)))

    print("\n===== 四条目标的核对 (lead>=4) =====")
    for arm in ("v7", "v8", "v8.1"):
        tp, fp = total(arm, 4)
        med = float(np.average(table[table.arm == arm].median_lead,
                               weights=table[table.arm == arm].n_risk))
        print("  %-5s 召回 %4d/1358 = %.3f | 误报 %3d (%.1f/千条成功) | 精度 %.3f"
              " | 中位提前 %.1f chunk"
              % (arm, tp, tp / 1358, fp, 1000 * fp / 31602, tp / (tp + fp), med))

    print("\n===== 分 suite (全量, lead>=4) =====")
    for suite in SUITES:
        cells = {}
        n = 0
        for arm in ("v7", "v8", "v8.1"):
            t = f = 0
            for cohort, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                first = alarms[cohort][arm]
                timely = (first >= 0) & ((d["length"] - first) >= 4)
                t += int((timely & m & d["risk"]).sum())
                f += int((timely & m & ~d["risk"]).sum())
                if arm == "v7":
                    n += int((m & d["risk"]).sum())
            cells[arm] = (t, f)
        print("  %-16s v7 %3d/%-4d %3dFP | v8 %3d/%-4d %3dFP | **v8.1 %3d/%-4d %3dFP**"
              % (suite, cells["v7"][0], n, cells["v7"][1], cells["v8"][0], n,
                 cells["v8"][1], cells["v8.1"][0], n, cells["v8.1"][1]))

    print("\n===== 分 cohort (lead>=4) =====")
    for cohort in COHORTS:
        a = table[(table.cohort == cohort) & (table.arm == "v8")].iloc[0]
        b = table[(table.cohort == cohort) & (table.arm == "v8.1")].iloc[0]
        print("  %-18s v8 %3d/%-4d %3dFP -> v8.1 %3d/%-4d %3dFP  (+%d/+%d)"
              % (cohort, a.tp4, a.n_risk, a.fp4, b.tp4, b.n_risk, b.fp4,
                 b.tp4 - a.tp4, b.fp4 - a.fp4))

    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for cohort, d in data.items():
        shuffled = []
        for head, series in heads[cohort].items():
            s = series.copy()
            for q in range(s.shape[1]):
                col = s[:, q]
                idx = np.flatnonzero(np.isfinite(col))
                col[idx] = col[rng.permutation(idx)]
            shuffled.append(moving_first(s, thresholds[head],
                                         V8.DIRECTION[head], SLOPE))
        sc = V8.score(V8.union(d["v7"], *shuffled), d["risk"], d["length"], 4)
        ntp += sc["tp"]
        nfp += sc["fp"]
    tp81, fp81 = total("v8.1", 4)
    print("\n零对照(两个新头逐 chunk 打乱, v7 不变): %d/1358 TP %d FP"
          "  —— v8.1 是 %d/1358 %d FP" % (ntp, nfp, tp81, fp81))

    (OUT / "v81_summary.json").write_text(json.dumps({
        "slope": SLOPE, "thresholds": thresholds, "quantile": V8.QUANTILE,
        "confirm": V8.CONFIRM, "curvature_steps": list(V8.CURVATURE_IDX),
        "episodes": 32960, "risks": 1358,
        "v7": dict(zip(("tp", "fp"), total("v7", 4))),
        "v8": dict(zip(("tp", "fp"), total("v8", 4))),
        "v8.1": dict(zip(("tp", "fp"), total("v8.1", 4))),
        "null": {"tp": ntp, "fp": nfp},
    }, indent=2))


if __name__ == "__main__":
    main()
