"""v8 over the whole corpus: 32,960 episodes, 1,358 risks, three cohorts.

Thresholds come from `development_main` only, at the operating point chosen
there (quantile 0.995, confirmation K=2).  `external_8b` and
`legacy_main16x32` are scored once with no recalibration.  legacy is the
strongest of the three tests: nothing in v7 or v8 was ever fitted on it, and it
uses a different sampling design (16 initial states x 32 noise seeds versus
50 x 8).
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
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

from recompute_task_matched_lift import META_KEYS, SOURCES, cohort_frame  # noqa: E402

FRONT, BACK = slice(0, 4), slice(4, 8)
WIDTH = 6
BASELINE = slice(1, 5)
# flow_speed is stored as steps 1..9, so raw indices 0,4,8 are steps 1,5,9.
CURVATURE_IDX = (0, 4, 8)
QUANTILE = 0.995
CONFIRM = 2
LEADS = (0, 2, 4, 8, 12)
HEADLINE_LEAD = 4
SEED = 20260906
SUITE_OF = {"libero_goal": "goal", "libero_long": "long",
            "libero_object": "object", "libero_spatial": "spatial"}


def trailing_mean(values, width):
    out = np.full_like(values, np.nan, dtype=np.float64)
    for q in range(values.shape[1]):
        with np.errstate(invalid="ignore"):
            out[:, q] = np.nanmean(values[:, max(0, q - width + 1):q + 1], axis=1)
    return out


def heads_from_flow_speed(speed: np.ndarray) -> dict[str, np.ndarray]:
    """Both v8 heads, from the single [n, chunk, 8, 9] array."""
    with np.errstate(invalid="ignore"):
        path = np.nansum(speed, axis=3)                       # per layer
        front = np.nanmean(path[:, :, FRONT], axis=2)
        back = np.nanmean(path[:, :, BACK], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.maximum(front, 1e-12) / np.maximum(back, 1e-12))
    ratio[~np.isfinite(ratio)] = np.nan

    sub = speed[:, :, BACK][:, :, :, list(CURVATURE_IDX)]
    curvature = np.abs(np.diff(sub, n=2, axis=-1)).mean(axis=(2, 3))
    base = np.nanmean(curvature[:, BASELINE], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(np.maximum(curvature, 1e-12) / np.maximum(base, 1e-12))
    rel[~np.isfinite(rel)] = np.nan

    # the ratio is already normalised across layers; only the curvature is
    # self-baselined, matching v7's freeze idiom
    return {"frontback_flowpath": trailing_mean(ratio, WIDTH),
            "curvature_3step": trailing_mean(rel, WIDTH)}


DIRECTION = {"frontback_flowpath": "low", "curvature_3step": "high"}


def confirmed_first(score, threshold, direction, confirm, earliest=WIDTH):
    """First chunk where the crossing has held for `confirm` consecutive chunks.

    `>=`, never `>`: strict `>` silently drops the tie group at the threshold.
    """
    hit = (score >= threshold) if direction == "high" else (score <= threshold)
    hit &= np.isfinite(score)
    hit[:, :earliest] = False
    if confirm > 1:
        held = np.zeros_like(hit)
        run = np.zeros(hit.shape[0], dtype=int)
        for q in range(hit.shape[1]):
            run = np.where(hit[:, q], run + 1, 0)
            held[:, q] = run >= confirm
        hit = held
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def union(*firsts):
    stack = np.stack([np.where(f < 0, 1 << 20, f) for f in firsts])
    m = stack.min(axis=0)
    return np.where(m >= (1 << 20), -1, m)


def score(first, risk, length, lead=HEADLINE_LEAD):
    fired = first >= 0
    lag = np.where(fired, length - first, -1)
    timely = fired & (lag >= lead)
    tp, fp = int((timely & risk).sum()), int((timely & ~risk).sum())
    return {"tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "median_lead": float(np.median(lag[timely & risk]))
            if (timely & risk).any() else float("nan")}


def load_cohort(name):
    speed = np.load(OUT / f"{name}_flow_speed.npz")["flow_speed"]
    if name == "legacy_main16x32":
        table = pd.read_csv(
            ROOT / "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv")
        risk = table.failure.to_numpy(bool)
        length = table.length.to_numpy(int)
        suite = np.array([t.split("/")[0] for t in table.task.astype(str)])
        v7 = np.load(ROOT / "moe-v7-legacy16x32-0906/results/legacy16x32"
                     "/legacy16x32_first_alarms.npz")["legacy_guard"].astype(int)
        assert len(risk) == speed.shape[0] == len(v7), name
    else:
        coh = cohort_frame(name)
        risk, length, suite = coh["risk"], coh["length"], coh["suite"]
        v7 = None
        for bundle, files in SOURCES.items():
            rel = files.get(name)
            if rel is None or not (ROOT / bundle / rel).exists():
                continue
            data = np.load(ROOT / bundle / rel, allow_pickle=True)
            for key in data.files:
                if key not in META_KEYS and "v7_guard" in key:
                    v7 = data[key].astype(int)
        assert v7 is not None, name
    return {"speed": speed, "risk": risk, "length": length,
            "suite": suite, "v7": v7}


def main() -> None:
    names = ("development_main", "external_8b", "legacy_main16x32")
    data = {n: load_cohort(n) for n in names}
    heads = {n: heads_from_flow_speed(d["speed"]) for n, d in data.items()}

    thresholds = {}
    for name, series in heads["development_main"].items():
        pool = series[np.isfinite(series)]
        level = QUANTILE if DIRECTION[name] == "high" else 1.0 - QUANTILE
        thresholds[name] = float(np.quantile(pool, level, method="lower"))
    print("阈值（development 未标注分数的顺序统计量，分位 %.3f，确认 K=%d）:"
          % (QUANTILE, CONFIRM))
    print("  " + json.dumps({k: round(v, 6) for k, v in thresholds.items()}))

    rows, alarms = [], {}
    for name, d in data.items():
        firsts = {"v7": d["v7"]}
        for head, series in heads[name].items():
            firsts[head] = confirmed_first(series, thresholds[head],
                                           DIRECTION[head], CONFIRM)
        firsts["v8"] = union(*firsts.values())
        alarms[name] = firsts
        for arm, first in firsts.items():
            row = {"cohort": name, "arm": arm, "n_risk": int(d["risk"].sum()),
                   "n_safe": int((~d["risk"]).sum())}
            for lead in LEADS:
                s = score(first, d["risk"], d["length"], lead)
                row |= {f"tp_lead{lead}": s["tp"], f"fp_lead{lead}": s["fp"]}
            row |= score(first, d["risk"], d["length"])
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "v8_full_corpus.csv", index=False)
    np.savez_compressed(OUT / "v8_full_corpus_alarms.npz",
                        **{f"{c}|{k}": v for c, f in alarms.items()
                           for k, v in f.items()})

    print("\n===== 全量：提前量 >= %d chunk =====" % HEADLINE_LEAD)
    print("%-18s %8s %8s | %-22s | %-22s"
          % ("cohort", "episodes", "失败", "v7", "v8"))
    tot = {"v7": [0, 0], "v8": [0, 0], "risk": 0, "epi": 0}
    for name in names:
        d = data[name]
        n_epi = len(d["risk"])
        line = []
        for arm in ("v7", "v8"):
            r = table[(table.cohort == name) & (table.arm == arm)].iloc[0]
            line.append("%4d/%-4d TP %4d FP" % (r.tp_lead4, r.n_risk, r.fp_lead4))
            tot[arm][0] += int(r.tp_lead4)
            tot[arm][1] += int(r.fp_lead4)
        tot["risk"] += int(d["risk"].sum())
        tot["epi"] += n_epi
        print("%-18s %8d %8d | %-22s | %-22s"
              % (name, n_epi, int(d["risk"].sum()), line[0], line[1]))
    print("%-18s %8d %8d | %4d/%-4d TP %4d FP | %4d/%-4d TP %4d FP"
          % ("合计", tot["epi"], tot["risk"], tot["v7"][0], tot["risk"],
             tot["v7"][1], tot["v8"][0], tot["risk"], tot["v8"][1]))
    print("%-18s %8s %8s | 精度 %.3f%14s| 精度 %.3f"
          % ("", "", "", tot["v7"][0] / (tot["v7"][0] + tot["v7"][1]), "",
             tot["v8"][0] / (tot["v8"][0] + tot["v8"][1])))

    print("\n===== 分 suite（全量合并）=====")
    print("%-10s %8s | %-16s | %-16s" % ("suite", "失败", "v7", "v8"))
    for suite in SUITE_OF:
        agg = {"v7": [0, 0], "v8": [0, 0], "n": 0}
        for name in names:
            d = data[name]
            m = d["suite"] == suite
            if not m.any():
                continue
            agg["n"] += int((m & d["risk"]).sum())
            for arm in ("v7", "v8"):
                f = alarms[name][arm]
                timely = (f >= 0) & ((d["length"] - f) >= HEADLINE_LEAD)
                agg[arm][0] += int((timely & m & d["risk"]).sum())
                agg[arm][1] += int((timely & m & ~d["risk"]).sum())
        print("%-10s %8d | %4d/%-4d %4d FP | %4d/%-4d %4d FP"
              % (SUITE_OF[suite], agg["n"], agg["v7"][0], agg["n"],
                 agg["v7"][1], agg["v8"][0], agg["n"], agg["v8"][1]))

    print("\n===== 提前量的代价（全量合计）=====")
    print("%-6s %s" % ("臂", "  ".join("lead>=%d" % b for b in LEADS)))
    for arm in ("v7", "v8"):
        cells = []
        for lead in LEADS:
            tp = int(table[table.arm == arm][f"tp_lead{lead}"].sum())
            fp = int(table[table.arm == arm][f"fp_lead{lead}"].sum())
            cells.append("%4d/%-5d" % (tp, fp))
        print("%-6s %s" % (arm, "  ".join(cells)))

    # Null: shuffle each head across episodes within each chunk, so the alarm
    # rate per chunk is preserved but the episode identity is destroyed.
    rng = np.random.default_rng(SEED)
    null_tp = null_fp = 0
    for name, d in data.items():
        shuffled = {}
        for head, series in heads[name].items():
            s = series.copy()
            for q in range(s.shape[1]):
                col = s[:, q]
                ok = np.flatnonzero(np.isfinite(col))
                col[ok] = col[rng.permutation(ok)]
            shuffled[head] = confirmed_first(s, thresholds[head],
                                             DIRECTION[head], CONFIRM)
        u = union(d["v7"], *shuffled.values())
        s = score(u, d["risk"], d["length"])
        null_tp += s["tp"]
        null_fp += s["fp"]
    print("\n零对照（两个新头逐 chunk 打乱，v7 不变）: %d/%d TP %d FP"
          "   —— v8 是 %d/%d TP %d FP"
          % (null_tp, tot["risk"], null_fp, tot["v8"][0], tot["risk"],
             tot["v8"][1]))

    (OUT / "v8_full_corpus_summary.json").write_text(json.dumps({
        "thresholds": thresholds, "quantile": QUANTILE, "confirm": CONFIRM,
        "episodes": tot["epi"], "risks": tot["risk"],
        "v7": {"tp": tot["v7"][0], "fp": tot["v7"][1]},
        "v8": {"tp": tot["v8"][0], "fp": tot["v8"][1]},
        "null": {"tp": null_tp, "fp": null_fp},
    }, indent=2))


if __name__ == "__main__":
    main()
