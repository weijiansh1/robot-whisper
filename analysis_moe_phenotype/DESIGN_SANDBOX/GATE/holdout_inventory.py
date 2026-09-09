#!/usr/bin/env python3
"""留出集规模构成清点（GATE 任务 1 的可复现底稿）。

**只统计评估集的构成**：集数、失败数、事件数、组数、集长、onset 位置。
**不计算任何检测器指标**，不读 features/*/rows.npz 的信号列（只读 meta.json 的规模字段）。

用法:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 DESIGN_SANDBOX/GATE/holdout_inventory.py

产物: DESIGN_SANDBOX/GATE/holdout_inventory.json
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_inventory.json")

# 校准/开发集单位（CONSTRAINTS.md）——不属于留出集
CALIB_TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

# loop/trap 通道判无效的任务（AUDIT §7.1 + §8.4：铰接机构语义盲区）。static 通道不受影响。
LOOP_INVALID = {
    ("grid50x8", "libero_long", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
    ("grid50x8", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("main16x32", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
}

SUITES = ["libero_goal", "libero_long", "libero_object", "libero_spatial"]
CORPORA = ["grid50x8", "main16x32"]


def _load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build():
    tasks = []
    for f in sorted(glob.glob(os.path.join(ROOT, "events", "*", "*", "*", "events.csv"))):
        parts = f.split(os.sep)
        corpus, suite, task = parts[-4], parts[-3], parts[-2]
        rows = _load(f)
        li = (corpus, suite, task) in LOOP_INVALID

        meta_p = os.path.join(ROOT, "features", corpus, suite, task, "meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}

        nq = np.array([int(r["n_queries"]) for r in rows])
        scenes = sorted({int(r["scene"]) for r in rows})

        def loop_on(r):
            """loop-invalid 任务的 loop 通道视为缺失（不是证据）。"""
            v = int(r["loop_onset_q"])
            return -1 if (li or v < 0) else v

        def stat_on(r):
            return int(r["static_onset_q"])

        n_fail = sum(1 for r in rows if not int(r["success"]))
        loop_f = sum(1 for r in rows if not int(r["success"]) and loop_on(r) >= 0)
        loop_s = sum(1 for r in rows if int(r["success"]) and loop_on(r) >= 0)
        stat_f = sum(1 for r in rows if not int(r["success"]) and stat_on(r) >= 0)
        stat_s = sum(1 for r in rows if int(r["success"]) and stat_on(r) >= 0)

        # 主端点分母：有可用锚点的致败集（loop∪static，loop 通道已按有效性屏蔽）
        anchored, onsets, dual = 0, [], 0
        for r in rows:
            if int(r["success"]):
                continue
            cand = [(loop_on(r), "loop")] if loop_on(r) >= 0 else []
            if stat_on(r) >= 0:
                cand.append((stat_on(r), "static"))
            if cand:
                anchored += 1
                onsets.append(min(cand)[0])
                dual += len(cand) == 2
        fail_unanchored = n_fail - anchored

        # 误报分母：干净成功集 = 成功 且 无 static 事件 且（无 loop 事件 或 loop 通道无效）
        clean = sum(1 for r in rows if int(r["success"]) and stat_on(r) < 0 and loop_on(r) < 0)
        # loop 通道无效任务上的 loop 标记真值未知，单列
        uncertain = 0
        if li:
            uncertain = sum(1 for r in rows
                            if int(r["success"]) and int(r["loop_onset_q"]) >= 0
                            and int(r["static_onset_q"]) < 0)

        tasks.append(dict(
            corpus=corpus, suite=suite, task=task,
            holdout=(task != CALIB_TASK),
            proxy_grade=rows[0]["proxy_grade"], loop_channel_valid=(not li),
            n_episodes=len(rows), n_success=len(rows) - n_fail, n_fail=n_fail,
            n_groups=len(scenes),
            n_groups_with_fail=len({int(r["scene"]) for r in rows if not int(r["success"])}),
            nq_min=int(nq.min()), nq_p10=int(np.percentile(nq, 10)),
            nq_median=int(np.median(nq)), nq_max=int(nq.max()),
            meta_rows=meta.get("rows"), meta_dropped=len(meta.get("dropped_episodes", [])),
            fatal_loop=loop_f, recovery_loop=loop_s,
            fatal_static=stat_f, benign_static=stat_s,
            fatal_anchored=anchored, fatal_unanchored=fail_unanchored, dual_channel=dual,
            clean_success=clean, uncertain_loop_success=uncertain,
            # FA_primary 分母只用 loop 通道有效的任务（见 HOLDOUT_PROTOCOL §3.2）
            clean_success_loopvalid=(0 if li else clean),
            onset_median=float(np.median(onsets)) if onsets else None,
            onset_p10=float(np.percentile(onsets, 10)) if onsets else None,
        ))

    hold = [t for t in tasks if t["holdout"]]
    keys = ["n_episodes", "n_success", "n_fail", "n_groups", "n_groups_with_fail",
            "fatal_loop", "recovery_loop", "fatal_static", "benign_static",
            "fatal_anchored", "fatal_unanchored", "dual_channel",
            "clean_success", "clean_success_loopvalid", "uncertain_loop_success"]

    def agg(rs):
        return {k: int(sum(r[k] for r in rs)) for k in keys} | {"n_tasks": len(rs)}

    cells = {}
    for c in CORPORA:
        for s in SUITES:
            rs = [t for t in hold if t["corpus"] == c and t["suite"] == s]
            if rs:
                cells[f"{c}/{s}"] = agg(rs)

    out = dict(
        generated_for="DESIGN_SANDBOX/GATE/HOLDOUT_PROTOCOL.md",
        calibration_unit_excluded=CALIB_TASK,
        loop_invalid_tasks=sorted("/".join(k) for k in LOOP_INVALID),
        holdout_total=agg(hold),
        by_corpus={c: agg([t for t in hold if t["corpus"] == c]) for c in CORPORA},
        by_cell=cells,
        per_task=hold,
        calibration_unit_reference=[t for t in tasks if not t["holdout"]],
    )
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    o = build()
    h = o["holdout_total"]
    print(f"holdout tasks={h['n_tasks']}  episodes={h['n_episodes']}  fail={h['n_fail']}")
    print(f"  fatal anchored (primary denominator) = {h['fatal_anchored']}")
    print(f"  clean success  (FA denominator)      = {h['clean_success']}")
    print(f"  recovery loops (S1 negative control) = {h['recovery_loop']}")
    print(f"  unanchored failures (descriptive)    = {h['fatal_unanchored']}")
    print(f"wrote {OUT}")
