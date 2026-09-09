"""Step 3: apply the frozen spec once to external_8b and legacy_main16x32.

Everything here reads `results/frozen_spec.json` and changes nothing.  The
thresholds are order statistics of the *unlabeled* development scores, and the
per-chunk rank transform is taken against the development reference too, so no
quantity in the detector is a function of a test cohort.

Reported, in this order:

  * the full-corpus net over v8's 932/1358 at 126 FP, at every lead;
  * the exchange rate, against the ~2 TP per FP bar;
  * per suite and per task, including specifically whether the head helps the
    tasks v8 already fails;
  * leave-one-task-out, which is where a "detector" that is really one task
    collapses;
  * three nulls of increasing strength, including two that preserve each
    episode's autocorrelation;
  * the length negative control, which is not a baseline: risk is *defined* as
    running to the cap, so length recalls 100% by construction and needs the cap
    the protocol forbids.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    ALL_CELLS,
    COHORTS,
    HEADLINE_LEAD,
    LEADS,
    OUT,
    SUITE_OF,
    load_all,
    per_suite,
    score,
    score_all_leads,
    trailing_mean,
    union,
)
from heads import (
    constructions,
    fire_from,
    first_true_from,
    held_runs,
    threshold_of,
    window_of,
)
from pairwise_scan import SURROGATES

SEED = 20260906
NULL_DRAWS = 8
DEV = "development_main"


def build_first(spec, series_by_cohort, dev_scores, n_chunk):
    """First-alarm vector per cohort under the frozen spec.

    Thresholds come from development and are reused verbatim.
    """
    name = spec["score"]
    w, k = spec["width"], spec["confirm"]
    q, earliest = spec["quantile"], spec["earliest"]
    gate = None if spec["gate"] == "-" else tuple(
        int(x) for x in spec["gate"].split("-"))
    start, hi = window_of(earliest, gate)
    combo = spec["direction"]

    if combo in ("OR", "AND"):
        _, form, pair = name.split("|")
        family, letters = pair.split(":")
        act = {"f": "front_action", "b": "back_action"}[letters[0]]
        st = {"f": "front_state", "b": "back_state"}[letters.split("-")[1][0]]
        keys = (f"{form}|{family}:{act}", f"{form}|{family}:{st}")
        dirs = ("high", "low")
        thresholds = [threshold_of(trailing_mean(dev_scores[key], w), q, dr)
                      for key, dr in zip(keys, dirs)]
        out = {}
        for cohort, scores in series_by_cohort.items():
            firsts = []
            for key, dr, thr in zip(keys, dirs, thresholds):
                sm = trailing_mean(scores[key], w)
                ft = first_true_from(held_runs(sm, thr, dr, start) >= k)
                firsts.append(fire_from(ft, start, hi, n_chunk[cohort]))
            fa, fs = firsts
            out[cohort] = (union(fa, fs) if combo == "OR"
                           else np.where((fa >= 0) & (fs >= 0),
                                         np.maximum(fa, fs), -1))
        return out, dict(zip(keys, thresholds)), keys, dirs

    dr = combo
    thr = threshold_of(trailing_mean(dev_scores[name], w), q, dr)
    out = {}
    for cohort, scores in series_by_cohort.items():
        sm = trailing_mean(scores[name], w)
        ft = first_true_from(held_runs(sm, thr, dr, start) >= k)
        out[cohort] = fire_from(ft, start, hi, n_chunk[cohort])
    return out, {name: thr}, (name,), (dr,)


def assert_tie_group_fires(series, thr, direction, width):
    """`>=` / `<=`, never strict: the tie group exactly at the threshold fires."""
    sm = trailing_mean(series, width)
    at = sm == thr
    if not at.any():
        return 0
    hit = (sm >= thr) if direction == "high" else (sm <= thr)
    assert bool(hit[at].all()), "tie group at the threshold does not fire"
    return int(at.sum())


def main() -> None:
    spec = json.loads((OUT / "frozen_spec.json").read_text())
    data = load_all()
    dev = data[DEV]
    n_chunk = {c: data[c]["front_state"].shape[1] for c in COHORTS}
    series = {c: constructions(data[c], dev) for c in COHORTS}
    dev_scores = series[DEV]

    firsts, thresholds, keys, dirs = build_first(spec, series, dev_scores, n_chunk)
    n_tie = {}
    for key, dr in zip(keys, dirs):
        n_tie[key] = assert_tie_group_fires(dev_scores[key], thresholds[key], dr,
                                            spec["width"])
    print("冻结规格: %s" % spec["raw_key"])
    print("阈值（development 未标注分数的顺序统计量）: %s"
          % json.dumps({k: round(v, 6) for k, v in thresholds.items()}))
    print("阈值处并列组大小: %s（已断言其全部触发，用 >= / <= 而非严格不等）"
          % json.dumps(n_tie))

    rows, alarms = [], {}
    for cohort in COHORTS:
        d = data[cohort]
        head = firsts[cohort]
        combined = union(d["v8"], head)
        alarms[cohort] = {"v7": d["v7"], "v8": d["v8"], "head": head,
                          "v8+head": combined}
        for arm, first in alarms[cohort].items():
            rows.append({"cohort": cohort, "arm": arm,
                         "n_risk": int(d["risk"].sum()),
                         "n_safe": int((~d["risk"]).sum()),
                         **score_all_leads(first, d["risk"], d["length"])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "frozen_scores.csv", index=False)
    np.savez_compressed(OUT / "boundary_head_alarms.npz",
                        **{f"{c}|{k}": v for c, f in alarms.items()
                           for k, v in f.items()})

    print("\n===== 全量：v8 与 v8+边界头 =====")
    for lead in LEADS:
        line = []
        tot = {"v8": [0, 0], "v8+head": [0, 0], "risk": 0}
        for cohort in COHORTS:
            r8 = table[(table.cohort == cohort) & (table.arm == "v8")].iloc[0]
            rh = table[(table.cohort == cohort) & (table.arm == "v8+head")].iloc[0]
            tot["risk"] += int(r8.n_risk)
            tot["v8"][0] += int(r8[f"tp{lead}"]); tot["v8"][1] += int(r8[f"fp{lead}"])
            tot["v8+head"][0] += int(rh[f"tp{lead}"])
            tot["v8+head"][1] += int(rh[f"fp{lead}"])
            line.append("%s %d/%d %dFP->%d/%d %dFP"
                        % (cohort.split("_")[0][:4], r8[f"tp{lead}"], r8.n_risk,
                           r8[f"fp{lead}"], rh[f"tp{lead}"], rh.n_risk,
                           rh[f"fp{lead}"]))
        dt = tot["v8+head"][0] - tot["v8"][0]
        df = tot["v8+head"][1] - tot["v8"][1]
        rate = dt / df if df > 0 else float("inf") if dt > 0 else 0.0
        star = " <== 头条" if lead == HEADLINE_LEAD else ""
        print("lead>=%-2d  v8 %4d/%d %5d FP  ->  v8+头 %4d/%d %5d FP   净 %+d TP / %+d FP"
              "   兑换率 %s%s"
              % (lead, tot["v8"][0], tot["risk"], tot["v8"][1],
                 tot["v8+head"][0], tot["risk"], tot["v8+head"][1], dt, df,
                 "%.2f TP/FP" % rate if df > 0 else "inf" if dt > 0 else "-", star))
        if lead == HEADLINE_LEAD:
            headline = {"v8_tp": tot["v8"][0], "v8_fp": tot["v8"][1],
                        "new_tp": tot["v8+head"][0], "new_fp": tot["v8+head"][1],
                        "d_tp": dt, "d_fp": df, "risk": tot["risk"],
                        "rate": rate}

    print("\n各 cohort（lead>=%d）:" % HEADLINE_LEAD)
    for cohort in COHORTS:
        r8 = table[(table.cohort == cohort) & (table.arm == "v8")].iloc[0]
        rh = table[(table.cohort == cohort) & (table.arm == "v8+head")].iloc[0]
        rd = table[(table.cohort == cohort) & (table.arm == "head")].iloc[0]
        print("  %-18s v8 %4d/%-4d %4d FP  ->  %4d/%-4d %4d FP   净 %+d/%+d"
              "   （头单独 %d/%d %d FP）"
              % (cohort, r8.tp4, r8.n_risk, r8.fp4, rh.tp4, rh.n_risk, rh.fp4,
                 rh.tp4 - r8.tp4, rh.fp4 - r8.fp4, rd.tp4, rd.n_risk, rd.fp4))

    # ---- per suite --------------------------------------------------------
    print("\n===== 分 suite（三 cohort 合并，lead>=%d）=====" % HEADLINE_LEAD)
    print("%-8s %8s %-20s %-20s %s" % ("suite", "失败数", "v8", "v8+头", "净"))
    suite_rows = []
    for suite, short in SUITE_OF.items():
        agg = {"v8": [0, 0], "v8+head": [0, 0], "n": 0}
        for cohort in COHORTS:
            d = data[cohort]
            m = d["suite"] == suite
            if not m.any():
                continue
            agg["n"] += int((m & d["risk"]).sum())
            for arm in ("v8", "v8+head"):
                f = alarms[cohort][arm]
                timely = (f >= 0) & ((d["length"] - f) >= HEADLINE_LEAD)
                agg[arm][0] += int((timely & m & d["risk"]).sum())
                agg[arm][1] += int((timely & m & ~d["risk"]).sum())
        suite_rows.append({"suite": short, "n_risk": agg["n"],
                           "v8_tp": agg["v8"][0], "v8_fp": agg["v8"][1],
                           "new_tp": agg["v8+head"][0], "new_fp": agg["v8+head"][1]})
        print("%-8s %8d %4d/%-4d %4d FP     %4d/%-4d %4d FP     %+d TP / %+d FP"
              % (short, agg["n"], agg["v8"][0], agg["n"], agg["v8"][1],
                 agg["v8+head"][0], agg["n"], agg["v8+head"][1],
                 agg["v8+head"][0] - agg["v8"][0],
                 agg["v8+head"][1] - agg["v8"][1]))
    pd.DataFrame(suite_rows).to_csv(OUT / "per_suite.csv", index=False)

    # ---- per task ---------------------------------------------------------
    tt = {}
    for cohort in COHORTS:
        d = data[cohort]
        for task in np.unique(d["task"]):
            m = d["task"] == task
            r = tt.setdefault(task, {"task": task, "n_risk": 0, "n_safe": 0})
            r["n_risk"] += int((m & d["risk"]).sum())
            r["n_safe"] += int((m & ~d["risk"]).sum())
            for arm in ("v8", "v8+head"):
                f = alarms[cohort][arm]
                timely = (f >= 0) & ((d["length"] - f) >= HEADLINE_LEAD)
                r[f"{arm}_tp"] = r.get(f"{arm}_tp", 0) + int((timely & m & d["risk"]).sum())
                r[f"{arm}_fp"] = r.get(f"{arm}_fp", 0) + int((timely & m & ~d["risk"]).sum())
    task_table = pd.DataFrame(list(tt.values()))
    task_table = task_table[task_table.n_risk > 0].copy()
    task_table["v8_recall"] = task_table.v8_tp / task_table.n_risk
    task_table["new_recall"] = task_table["v8+head_tp"] / task_table.n_risk
    task_table["d_tp"] = task_table["v8+head_tp"] - task_table.v8_tp
    task_table["d_fp"] = task_table["v8+head_fp"] - task_table.v8_fp
    task_table = task_table.sort_values("v8_recall").reset_index(drop=True)
    task_table.to_csv(OUT / "per_task.csv", index=False)

    weak = task_table[task_table.v8_recall <= 0.4]
    strong = task_table[task_table.v8_recall >= 0.8]
    print("\n===== 分任务：这个头帮的是不是 v8 本来就做不好的任务 =====")
    print("v8 召回 <= 0.4 的任务 %d 个（共 %d 个失败）：净 %+d TP / %+d FP"
          % (len(weak), int(weak.n_risk.sum()), int(weak.d_tp.sum()),
             int(weak.d_fp.sum())))
    print("v8 召回 >= 0.8 的任务 %d 个（共 %d 个失败）：净 %+d TP / %+d FP"
          % (len(strong), int(strong.n_risk.sum()), int(strong.d_tp.sum()),
             int(strong.d_fp.sum())))
    mid = task_table[(task_table.v8_recall > 0.4) & (task_table.v8_recall < 0.8)]
    print("中间 %d 个任务（共 %d 个失败）：净 %+d TP / %+d FP"
          % (len(mid), int(mid.n_risk.sum()), int(mid.d_tp.sum()),
             int(mid.d_fp.sum())))
    print("\n净增最多的 10 个任务:")
    cols = ["task", "n_risk", "v8_tp", "v8+head_tp", "d_tp", "d_fp", "v8_recall"]
    print(task_table.nlargest(10, "d_tp")[cols].to_string(index=False))
    print("\n新增 FP 最多的 10 个任务:")
    print(task_table.nlargest(10, "d_fp")[cols].to_string(index=False))
    print("\n净增 TP 的任务数: %d / %d   净增 FP 的任务数: %d / %d"
          % (int((task_table.d_tp > 0).sum()), len(task_table),
             int((task_table.d_fp > 0).sum()), len(task_table)))

    # ---- leave one task out ----------------------------------------------
    print("\n===== 留一任务法：去掉贡献最大的任务后还剩多少 =====")
    loto = []
    for task in task_table.task:
        keep = task_table[task_table.task != task]
        loto.append({"dropped": task, "d_tp": int(keep.d_tp.sum()),
                     "d_fp": int(keep.d_fp.sum())})
    loto = pd.DataFrame(loto).sort_values("d_tp")
    loto.to_csv(OUT / "leave_one_task_out.csv", index=False)
    total_dtp = int(task_table.d_tp.sum())
    print("全部任务净增 %+d TP。去掉单个任务后的最小净增 %+d（去掉 %s），"
          "最大 %+d" % (total_dtp, int(loto.d_tp.min()),
                       loto.iloc[0].dropped.split("/")[-1][:44],
                       int(loto.d_tp.max())))
    top = task_table.nlargest(1, "d_tp").iloc[0]
    print("单个任务贡献占比最高: %s 贡献 %d/%d = %.1f%%"
          % (top.task.split("/")[-1][:44], int(top.d_tp), total_dtp,
             100 * top.d_tp / max(total_dtp, 1)))

    # ---- nulls ------------------------------------------------------------
    print("\n===== 零对照：同一条流程，三种代理序列 =====")
    print("flat_shuffle 破坏幕内时间结构（门槛最低）；后两种保留自相关")
    null_rows = []
    for kind, fn in SURROGATES.items():
        tps, fps = [], []
        for draw in range(NULL_DRAWS):
            rng = np.random.default_rng(SEED + 991 * draw + abs(hash(kind)) % 977)
            tp = fp = 0
            for cohort in COHORTS:
                d = data[cohort]
                fake = dict(d)
                for cell in ALL_CELLS:
                    fake[cell] = fn(d[cell], d, rng)
                sc = constructions(fake, dev)
                fst, _, _, _ = build_first(spec, {cohort: sc}, dev_scores,
                                           n_chunk)
                s = score(union(d["v8"], fst[cohort]), d["risk"], d["length"],
                          HEADLINE_LEAD)
                tp += s["tp"]
                fp += s["fp"]
            tps.append(tp)
            fps.append(fp)
        null_rows.append({"surrogate": kind, "median_tp": float(np.median(tps)),
                          "median_fp": float(np.median(fps)),
                          "best_tp": int(max(tps)), "n_draws": NULL_DRAWS})
        print("  %-15s 中位 %d/%d TP  %d FP  （真实 %d/%d TP %d FP，v8 是 %d/%d %d FP）"
              % (kind, np.median(tps), headline["risk"], np.median(fps),
                 headline["new_tp"], headline["risk"], headline["new_fp"],
                 headline["v8_tp"], headline["risk"], headline["v8_fp"]))
    pd.DataFrame(null_rows).to_csv(OUT / "nulls.csv", index=False)

    # ---- length: a negative control, NOT a baseline -----------------------
    print("\n===== 负对照：长度（is_baseline = False）=====")
    print("风险的定义就是跑到上限，所以长度按构造召回 100%，且需要上限——协议禁止")
    len_rows = []
    for chunk in (8, 12, 16, 20):
        tp = fp = risk_n = 0
        for cohort in COHORTS:
            d = data[cohort]
            f = np.where(d["length"] > chunk, chunk, -1)
            s = score(f, d["risk"], d["length"], HEADLINE_LEAD)
            tp += s["tp"]; fp += s["fp"]; risk_n += int(d["risk"].sum())
        len_rows.append({"detector": f"all_running_at_chunk{chunk}", "tp": tp,
                         "fp": fp, "is_baseline": False})
        print("  在 chunk %2d 对所有仍在运行者报警: %d/%d TP  %d FP"
              % (chunk, tp, risk_n, fp))
    pd.DataFrame(len_rows).to_csv(OUT / "negative_control_length.csv", index=False)

    summary = {
        "spec": spec, "thresholds": thresholds, "tie_group_sizes": n_tie,
        "headline_lead": HEADLINE_LEAD, "headline": headline,
        "per_cohort": table.to_dict("records"),
        "per_suite": suite_rows,
        "weak_tasks": {"n": int(len(weak)), "n_risk": int(weak.n_risk.sum()),
                       "d_tp": int(weak.d_tp.sum()), "d_fp": int(weak.d_fp.sum())},
        "strong_tasks": {"n": int(len(strong)), "n_risk": int(strong.n_risk.sum()),
                         "d_tp": int(strong.d_tp.sum()),
                         "d_fp": int(strong.d_fp.sum())},
        "loto_min_d_tp": int(loto.d_tp.min()),
        "top_task_share": float(top.d_tp / max(total_dtp, 1)),
        "nulls": null_rows,
        "negative_control_length": len_rows,
    }
    (OUT / "frozen_summary.json").write_text(json.dumps(summary, indent=2,
                                                        default=float))
    print("\n写出 %s" % (OUT / "frozen_summary.json"))


if __name__ == "__main__":
    main()
