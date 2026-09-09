"""E4 补充腿（不改写主管线，只向 summary.json 增补 key="supplementary"）。

补 3 件主 run 缺的东西：
1. PROTOCOL §4 新颖性条款：family1/family2 全部格对 **mob1_w8**（late 窗集均值）
   做组内秩残差后重跑同族 maxT。cycle_gain 由 mob_k 导出、lock-in 判据本身含 mob1_w8，
   不做残差腿就不能主张"新信号"。
2. 阈值稳健性：脉冲 z 的尺度用 raw MAD（不乘 1.4826）重算整层，看结论是否翻转。
3. 混淆项与更高功效的配对层：集长（n_queries）配对 AUC；恢复性 vs 致败 loop 改按
   task 配对（scene 层功效过低）；(c) 的 loop-失败 vs static-失败 switching 对比。

复用主 run 的 load_task（口径逐字一致）。运行：
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_e4_supp.py
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BASE, "phenotype"))

import run_e4 as M                                          # noqa: E402
from stats import joint_maxt, paired_auc, residualise       # noqa: E402

OUT = HERE
NPERM = M.NPERM


def load_all(mad_c):
    """按给定 MAD 尺度重跑一遍装载（mad_c=1.4826 即主 run 口径）。"""
    old = M.MAD_C
    M.MAD_C = mad_c
    recs, zl = [], []
    sub = lambda p: sorted(d for d in os.listdir(p)                      # noqa: E731
                           if os.path.isdir(os.path.join(p, d)))
    for corpus in ("grid50x8", "main16x32"):
        for suite in sub(os.path.join(BASE, "features", corpus)):
            for task in sub(os.path.join(BASE, "features", corpus, suite)):
                r, z, _ = M.load_task(corpus, suite, task)
                recs += r
                zl += z
    M.MAD_C = old
    T = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    gk = [(r["corpus"], r["task"], r["scene"]) for r in recs]
    ug = {k: i for i, k in enumerate(sorted(set(gk)))}
    T["gid"] = np.array([ug[k] for k in gk])
    T["tid"] = np.array([{k: i for i, k in enumerate(sorted(set(
        (r["corpus"], r["task"]) for r in recs)))}[(r["corpus"], r["task"])]
        for r in recs])
    T["redflag"] = np.array([(r["corpus"], r["suite"], r["task"]) in M.REDFLAG
                             for r in recs])
    pop = np.full(len(recs), "other", dtype=object)
    isl = T["o_loop"] >= 0
    nev = (T["o_loop"] < 0) & (T["o_static"] < 0) & (T["o_trap"] < 0)
    pop[isl & (T["succ"] == 1)] = "recovering_loop"
    pop[isl & (T["succ"] == 0)] = "fatal_loop"
    pop[nev & (T["succ"] == 1)] = "clean_success"
    pop[T["redflag"]] = "excluded_redflag"
    T["pop"] = pop
    T["covered"] = (T["n_win"] > 0) & np.isfinite(T["cyc"]) & np.isfinite(T["wV"]) \
        & np.isfinite(T["wA"]) & np.isfinite(T["wM"]) & np.isfinite(T["wW"])
    return T


UNITS = [("grid50x8/libero_goal", "grid50x8", "libero_goal"),
         ("grid50x8/libero_long", "grid50x8", "libero_long"),
         ("grid50x8/libero_object", "grid50x8", "libero_object"),
         ("grid50x8/libero_spatial", "grid50x8", "libero_spatial"),
         ("main16x32/ALL", "main16x32", None)]


def maxt(vals, idx, y, g, tag, resid_base=None):
    """vals: dict name->array(全局)。返回 {name: {auc,n_pairs,p_maxT}}。"""
    cells = []
    for k, v in vals.items():
        x = v[idx].astype(float)
        if resid_base is not None:
            x = residualise(x, resid_base[idx].astype(float), g)
        cells.append((k, x, np.isfinite(x), g))
    o, p = joint_maxt(cells, idx, {int(e): int(t) for e, t in zip(idx, y)},
                      {int(e): int(t) for e, t in zip(idx, g)}, NPERM,
                      M.rng_for(tag))
    return {k: dict(auc=round(float(o[k][0]), 4) if np.isfinite(o[k][0]) else None,
                    n_pairs=int(o[k][1]),
                    p_maxT=round(p[k], 4) if k in p else None) for k in vals}


def main():
    t0 = time.time()
    S = json.load(open(os.path.join(OUT, "summary.json")))
    T = load_all(1.4826)
    print(f"[supp] loaded {len(T['succ'])} eps ({time.time()-t0:.0f}s)", flush=True)
    supp = {"note": "补充腿；主结论与族声明见 units/*。生成脚本 run_e4_supp.py",
            "residual_leg": {}, "length_confound": {}, "family2_task_level": {},
            "mad_scale_robustness": {}, "switching_by_event_class": {}}

    F1 = ("cycle_gain", "lockin_flag", "switch_flag")
    F2 = M.FAM2
    # 主 run 的逐集 lock-in / switch flag 从 pulse_stats.csv 取（与 summary 完全一致）
    import csv as _csv
    lock = np.full(len(T["succ"]), np.nan)
    swi = np.full(len(T["succ"]), np.nan)
    key2i = {(T["corpus"][i], T["task"][i], int(T["epid"][i])): i
             for i in range(len(T["succ"]))}
    for r in _csv.DictReader(open(os.path.join(OUT, "pulse_stats.csv"))):
        i = key2i[(r["corpus"], r["task"], int(r["episode_id"]))]
        if r["lockin_flag"] != "":
            lock[i] = float(r["lockin_flag"])
        if r["switch_flag"] != "":
            swi[i] = float(r["switch_flag"])
    V1 = dict(cycle_gain=T["cyc"], lockin_flag=lock, switch_flag=swi)
    V2 = dict(cycle_gain=T["cyc"], pulse_count=T["n_pulses"].astype(float),
              train_flag=T["train_flag"].astype(float),
              isolated_pulse_flag=T["iso_flag"].astype(float),
              post_pulse_return=T["post_ret"])

    for tag, corpus, suite in UNITS:
        um = (T["corpus"] == corpus) & ((T["suite"] == suite) if suite else True)
        # ---- 1a. family1 残差腿（fail vs succ；flag 固定在观测阈值上）----
        idx = np.flatnonzero(um)
        y = (T["succ"][idx] == 0).astype(int)
        supp["residual_leg"][f"{tag}|family1"] = maxt(
            V1, idx, y, T["gid"][idx], (tag, "f1res"), resid_base=T["wW"])
        supp["residual_leg"][f"{tag}|family1_nonresid_refit"] = maxt(
            V1, idx, y, T["gid"][idx], (tag, "f1raw"))
        # ---- 1b. family2 残差腿（致败 vs 恢复性）----
        pm = um & ~T["redflag"]
        fi = np.flatnonzero(pm & (T["pop"] == "fatal_loop"))
        ri = np.flatnonzero(pm & (T["pop"] == "recovering_loop"))
        if len(fi) and len(ri):
            li = np.concatenate([fi, ri])
            yl = np.r_[np.ones(len(fi), int), np.zeros(len(ri), int)]
            supp["residual_leg"][f"{tag}|family2"] = maxt(
                V2, li, yl, T["gid"][li], (tag, "f2res"), resid_base=T["wW"])
            # ---- 3. task 层配对（scene 层功效过低）----
            supp["family2_task_level"][tag] = maxt(
                V2, li, yl, T["tid"][li], (tag, "f2task"))
            supp["family2_task_level"][tag]["_audit"] = dict(
                n_fatal=len(fi), n_recovering=len(ri),
                n_tasks_with_both=int(sum(
                    1 for g in np.unique(T["tid"][li])
                    if (yl[T["tid"][li] == g] == 1).any()
                    and (yl[T["tid"][li] == g] == 0).any())))
            a, n = paired_auc(T["nq"][li].astype(float), yl, T["gid"][li])
            supp["length_confound"][f"{tag}|fatal_vs_recovering_n_queries"] = dict(
                auc=round(float(a), 4) if np.isfinite(a) else None, n_pairs=int(n),
                median_nq_fatal=float(np.median(T["nq"][fi])),
                median_nq_recovering=float(np.median(T["nq"][ri])))
        a, n = paired_auc(T["nq"][idx].astype(float), y, T["gid"][idx])
        supp["length_confound"][f"{tag}|fail_vs_succ_n_queries"] = dict(
            auc=round(float(a), 4) if np.isfinite(a) else None, n_pairs=int(n))
        # ---- 4. switching：loop 失败 vs static 失败（描述）----
        lf = np.flatnonzero(pm & (T["succ"] == 0) & (T["o_loop"] >= 0) & np.isfinite(swi))
        sf = np.flatnonzero(pm & (T["succ"] == 0) & (T["o_loop"] < 0)
                            & (T["o_static"] >= 0) & np.isfinite(swi))
        ne = np.flatnonzero(pm & (T["succ"] == 0) & (T["o_loop"] < 0)
                            & (T["o_static"] < 0) & np.isfinite(swi))
        sc = np.flatnonzero(pm & (T["succ"] == 1) & np.isfinite(swi))
        supp["switching_by_event_class"][tag] = {
            k: dict(n=int(len(v)),
                    frac_switching=round(float(swi[v].mean()), 4) if len(v) else None,
                    frac_lockin=round(float(lock[v].mean()), 4) if len(v) else None)
            for k, v in (("loop_failure", lf), ("static_only_failure", sf),
                         ("no_event_failure", ne), ("all_success", sc))}
        if len(lf) and len(sf):
            ii = np.concatenate([lf, sf])
            yy = np.r_[np.ones(len(lf), int), np.zeros(len(sf), int)]
            a, n = paired_auc(swi[ii], yy, T["tid"][ii])
            supp["switching_by_event_class"][tag]["loop_vs_static_failure_auc_task"] = \
                dict(auc=round(float(a), 4) if np.isfinite(a) else None, n_pairs=int(n))
        print(f"[supp {tag}] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- grid 合并腿的残差 + task 层 ----
    pm = (T["corpus"] == "grid50x8") & ~T["redflag"]
    fi = np.flatnonzero(pm & (T["pop"] == "fatal_loop"))
    ri = np.flatnonzero(pm & (T["pop"] == "recovering_loop"))
    li = np.concatenate([fi, ri])
    yl = np.r_[np.ones(len(fi), int), np.zeros(len(ri), int)]
    supp["residual_leg"]["grid50x8/ALL_pooled|family2"] = maxt(
        V2, li, yl, T["gid"][li], ("gridpool", "f2res"), resid_base=T["wW"])
    supp["family2_task_level"]["grid50x8/ALL_pooled"] = maxt(
        V2, li, yl, T["tid"][li], ("gridpool", "f2task"))
    supp["family2_task_level"]["grid50x8/ALL_pooled"]["_audit"] = dict(
        n_fatal=len(fi), n_recovering=len(ri),
        n_tasks_with_both=int(sum(1 for g in np.unique(T["tid"][li])
                                  if (yl[T["tid"][li] == g] == 1).any()
                                  and (yl[T["tid"][li] == g] == 0).any())))
    a, n = paired_auc(T["nq"][li].astype(float), yl, T["gid"][li])
    supp["length_confound"]["grid50x8/ALL_pooled|fatal_vs_recovering_n_queries"] = dict(
        auc=round(float(a), 4) if np.isfinite(a) else None, n_pairs=int(n),
        median_nq_fatal=float(np.median(T["nq"][fi])),
        median_nq_recovering=float(np.median(T["nq"][ri])))
    print(f"[supp pooled] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. MAD 尺度稳健性：raw MAD（不乘 1.4826）----
    T2 = load_all(1.0)
    rob = {}
    for cname in ("grid50x8", "main16x32"):
        pm2 = (T2["corpus"] == cname) & ~T2["redflag"]
        for popk in ("clean_success", "recovering_loop", "fatal_loop"):
            v = np.flatnonzero(pm2 & (T2["pop"] == popk))
            if not len(v):
                continue
            rob[f"{cname}|{popk}"] = dict(
                n=int(len(v)),
                frac_any_pulse=round(float((T2["n_pulses"][v] > 0).mean()), 4),
                frac_isolated=round(float(T2["iso_flag"][v].mean()), 4),
                frac_train=round(float(T2["train_flag"][v].mean()), 4),
                pulses_per_100q=round(float(T2["n_pulses"][v].sum()
                                            / T2["nq"][v].sum() * 100), 3))
        fi = np.flatnonzero(pm2 & (T2["pop"] == "fatal_loop"))
        ri = np.flatnonzero(pm2 & (T2["pop"] == "recovering_loop"))
        if len(fi) and len(ri):
            li = np.concatenate([fi, ri])
            yl = np.r_[np.ones(len(fi), int), np.zeros(len(ri), int)]
            V2b = dict(cycle_gain=T2["cyc"], pulse_count=T2["n_pulses"].astype(float),
                       train_flag=T2["train_flag"].astype(float),
                       isolated_pulse_flag=T2["iso_flag"].astype(float),
                       post_pulse_return=T2["post_ret"])
            rob[f"{cname}|family2_scene"] = maxt(V2b, li, yl, T2["gid"][li],
                                                 (cname, "rawmad"))
    supp["mad_scale_robustness"] = {
        "definition": "z=(x-median)/MAD without 1.4826 (主 run 用 1.4826*MAD)", **rob}
    print(f"[supp rawmad] done ({time.time()-t0:.0f}s)", flush=True)

    S["supplementary"] = supp
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(S, f, ensure_ascii=False, indent=1, default=float)
    print(f"SUPP DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
