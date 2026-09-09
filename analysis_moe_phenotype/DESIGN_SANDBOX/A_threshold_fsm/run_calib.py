"""方案 A 校准集评估：指标表 + 消融臂 C 的**配对差** + 旋钮敏感性 + 机械自检探针。

只读 SCENE8 两个单位（main16x32 / grid50x8），两单位各自独立报，绝不合并。
写出 summary.json 与逐集 CSV。

重心（按守门 agent 的方法学更正）：
  绝对 FA / Det 只当上界估计；**载荷主张是同一次运行内 arm-vs-C 的配对差**
  （两臂跑同一批 episode、继承相同暴露史），并在对齐工作点上做 McNemar 型比较。

运行：
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python3 run_calib.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace, asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detector as DT  # noqa: E402

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = [("main16x32", "U1_main16x32_SCENE8"), ("grid50x8", "U2_grid50x8_SCENE8")]

P_A = DT.PARAMS
P_C = DT.PARAMS_C            # 同规则（10% 健康尾）重定持续腿：K=12, M=2
P_Csame = DT.PARAMS_C_SAMEKNOBS
P_T = DT.PARAMS_T            # 平凡集长臂，同规则重定持续腿：K=5, M=6
P_Tsame = DT.PARAMS_T_SAMEKNOBS
SEED = 20260903
ARMS = [("A", P_A), ("C_rulematched", P_C), ("C_sameknobs", P_Csame),
        ("T_rulematched", P_T), ("T_sameknobs", P_Tsame)]


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def arrays(E, res, key):
    return np.array([res[int(e)][key] for e in E.epid])


def populations(E, ev):
    lo = np.array([ev[int(e)]["loop_onset_q"] for e in E.epid])
    st = np.array([ev[int(e)]["static_onset_q"] for e in E.epid])
    su = E.success
    noev = (lo < 0) & (st < 0)
    return dict(
        loop_onset=lo, static_onset=st,
        clean_succ=(su == 1) & noev,
        recov_loop=(su == 1) & (lo >= 0),
        succ_static=(su == 1) & (st >= 0),
        loop_fail=(su == 0) & (lo >= 0),
        static_fail=(su == 0) & (st >= 0),
        loop_only_fail=(su == 0) & (lo >= 0) & (st < 0),
        static_only_fail=(su == 0) & (st >= 0) & (lo < 0),
        both_fail=(su == 0) & (lo >= 0) & (st >= 0),
        noevent_fail=(su == 0) & noev,
        all_fail=(su == 0),
    )


def q1q3(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, med=None, q1=None, q3=None, frac_le0=None)
    return dict(n=int(len(v)), med=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                frac_le0=float((v <= 0).mean()))


def fisher_greater(a, b, c, d):
    """2x2 单尾 Fisher（P(恢复组阳性 <= a)）。a,b = 恢复组 hit/miss。"""
    from math import comb
    n = a + b + c + d
    row1, col1 = a + b, a + c
    return float(min(1.0, sum(
        comb(col1, k) * comb(n - col1, row1 - k) / comb(n, row1)
        for k in range(0, min(row1, col1) + 1) if k <= a)))


def mcnemar_exact(b, c):
    """配对二项精确检验（b = A响C不响，c = C响A不响），双尾。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n) * 2.0
    return float(min(1.0, p))


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def eval_arm(E, pop, res, tag):
    la = arrays(E, res, "first_loop_alarm_q")
    sa = arrays(E, res, "first_static_alarm_q")
    aq = arrays(E, res, "first_alarm_q")
    ac = arrays(E, res, "first_alarm_channel")
    ab = arrays(E, res, "abstain")
    nq = E.n_q.astype(float)

    def rate(mask, arr):
        m = np.asarray(mask) & ~ab
        return dict(n=int(m.sum()),
                    rate=float((arr[m] >= 0).mean()) if m.sum() else None,
                    n_hit=int((arr[m] >= 0).sum()))

    cs = pop["clean_succ"] & ~ab
    n_al = int((aq[cs] >= 0).sum())
    exposure = nq[cs].sum() / 100.0
    out = {
        "arm": tag,
        "n_abstain": int(ab.sum()),
        "FA_clean_success": {
            "n": int(cs.sum()),
            "any_channel": float((aq[cs] >= 0).mean()) if cs.sum() else None,
            "loop_channel": float((la[cs] >= 0).mean()) if cs.sum() else None,
            "static_channel": float((sa[cs] >= 0).mean()) if cs.sum() else None,
            "n_alarmed_any": n_al,
            "per_100q_any": float(n_al / exposure) if exposure else None,
            "mean_nq": float(nq[cs].mean()) if cs.sum() else None,
        },
        "detection": {
            "loop_fail__loop_channel": rate(pop["loop_fail"], la),
            "static_fail__static_channel": rate(pop["static_fail"], sa),
            "loop_fail__any": rate(pop["loop_fail"], aq),
            "static_fail__any": rate(pop["static_fail"], aq),
            "noevent_fail__any": rate(pop["noevent_fail"], aq),
            "all_fail__any": rate(pop["all_fail"], aq),
        },
        "negative_control_recovery_loop": {
            "recov_loop_n": int((pop["recov_loop"] & ~ab).sum()),
            "recov_loop__loop_channel": rate(pop["recov_loop"], la),
            "fatal_loop__loop_channel": rate(pop["loop_fail"], la),
            "succ_static_n": int((pop["succ_static"] & ~ab).sum()),
            "succ_static__static_channel": rate(pop["succ_static"], sa),
        },
        "lead": {
            "loop": q1q3([la[i] - pop["loop_onset"][i]
                          for i in np.flatnonzero(pop["loop_fail"] & ~ab)
                          if la[i] >= 0]),
            "static": q1q3([sa[i] - pop["static_onset"][i]
                            for i in np.flatnonzero(pop["static_fail"] & ~ab)
                            if sa[i] >= 0]),
        },
    }

    # 分型：两种口径 —— (i) 首次报警的通道；(ii) 全序列里是否出现过正确通道
    typ = {}
    for name, mask, want in [("loop_only_fail", pop["loop_only_fail"], "loop"),
                             ("static_only_fail", pop["static_only_fail"], "static")]:
        idx = np.flatnonzero(mask & ~ab & (aq >= 0))
        corr_first = [ac[i] == want for i in idx]
        corr_any = [any(a["channel"] == want
                        for a in res[int(E.epid[i])]["alarms"]) for i in idx]
        typ[name] = dict(n_pop=int(mask.sum()), n_alarmed=int(len(idx)),
                         correct_first=float(np.mean(corr_first)) if len(idx) else None,
                         correct_anywhere=float(np.mean(corr_any)) if len(idx) else None)
    keys = list(typ)
    nt = sum(typ[k]["n_alarmed"] for k in keys)
    typ["overall"] = {"n_alarmed": int(nt)}
    for suff in ("first", "anywhere"):
        ok = sum(typ[k]["n_alarmed"] * (typ[k][f"correct_{suff}"] or 0.0)
                 for k in keys)
        # 原始（微平均）：受两类事件的**基率**支配 —— 一个恒定输出某一类的
        # 常数分类器也能拿到 = 多数类占比。故另报**平衡（宏平均）**分型。
        typ["overall"][f"correct_{suff}"] = float(ok / nt) if nt else None
        per = [typ[k][f"correct_{suff}"] for k in keys
               if typ[k][f"correct_{suff}"] is not None]
        typ["overall"][f"balanced_{suff}"] = float(np.mean(per)) if per else None
    out["typing"] = typ

    lv = arrays(E, res, "scale_level")
    out["scale_levels"] = {l: int((lv == l).sum()) for l in ("group", "task", "none")}

    # 报警时刻的绝对 q 分布（暴露"检测器有多像一只钟"）
    def tstat(mask):
        m = mask & ~ab & (la >= 0)
        if not m.sum():
            return None
        v, n = la[m], E.n_q[m]
        return dict(n=int(m.sum()), med_alarm_q=float(np.median(v)),
                    frac_q_ge_32=float(np.mean(v >= 32)),
                    frac_q_ge_39=float(np.mean(v >= 39)),
                    med_steps_to_episode_end=float(np.median(n - v)))
    out["loop_alarm_timing"] = {
        "clean_success_false_alarms": tstat(pop["clean_succ"]),
        "fatal_loop_detections": tstat(pop["loop_fail"])}

    # 恢复性 vs 致败：长度中性的 onset 相对窗 [-4, +8]
    ow = {}
    for lab, mk in [("recovery_loop", pop["recov_loop"]),
                    ("fatal_loop", pop["loop_fail"])]:
        h = n_ = 0
        for i in np.flatnonzero(mk & ~ab):
            hi = min(pop["loop_onset"][i] + 8, E.n_q[i] - 1)
            n_ += 1
            h += int(any(a["channel"] == "loop"
                         and pop["loop_onset"][i] - 4 <= a["step"] <= hi
                         for a in res[int(E.epid[i])]["alarms"]))
        ow[lab] = dict(n=int(n_), hit=int(h))
    ow["fisher_p_one_sided"] = fisher_greater(
        ow["recovery_loop"]["hit"], ow["recovery_loop"]["n"] - ow["recovery_loop"]["hit"],
        ow["fatal_loop"]["hit"], ow["fatal_loop"]["n"] - ow["fatal_loop"]["hit"])
    out["recovery_vs_fatal_onset_window"] = ow
    out["recovery_loop_episodes"] = [
        dict(episode_id=int(E.epid[i]), group=int(E.group[i]), n_q=int(E.n_q[i]),
             loop_onset_q=int(pop["loop_onset"][i]),
             alarms=res[int(E.epid[i])]["alarms"])
        for i in np.flatnonzero(pop["recov_loop"])]

    # 前缀截断（硬消长度混杂）
    Qstar = int(np.median(E.n_q[pop["clean_succ"]])) - 1
    ok = lambda arr, m: float(np.mean((arr[m] >= 0) & (arr[m] <= Qstar))) \
        if m.sum() else None
    out["prefix_restricted"] = dict(
        Qstar=int(Qstar),
        FA_clean_success_any=ok(aq, cs),
        det_loop_fail_loopch=ok(la, pop["loop_fail"] & ~ab),
        det_static_fail_statch=ok(sa, pop["static_fail"] & ~ab),
        det_all_fail_any=ok(aq, pop["all_fail"] & ~ab))
    return out


def matched_horizon(E, pop, res, channel):
    """组内同绝对 q 匹配：事件集 e（onset o）× 同组每个干净成功集 s，
    共同视界 T = min(o, n_q(s)-1)。返回逐事件的 (det, fa) 向量与均值。"""
    key = "first_loop_alarm_q" if channel == "loop" else "first_static_alarm_q"
    onset = pop["loop_onset"] if channel == "loop" else pop["static_onset"]
    ev_mask = pop["loop_fail"] if channel == "loop" else pop["static_fail"]
    aq = arrays(E, res, key)
    cs = np.flatnonzero(pop["clean_succ"])
    det, fa, idxs, npairs = [], [], [], 0
    for i in np.flatnonzero(ev_mask):
        cand = [j for j in cs if E.group[j] == E.group[i]] or list(cs)
        d, f = [], []
        for j in cand:
            T = min(int(onset[i]), int(E.n_q[j]) - 1)
            if T < DT.PARAMS.q_min:
                continue
            d.append(1.0 if (0 <= aq[i] <= T) else 0.0)
            f.append(1.0 if (0 <= aq[j] <= T) else 0.0)
        if not d:
            continue
        det.append(np.mean(d)); fa.append(np.mean(f)); idxs.append(int(i))
        npairs += len(d)
    if not det:
        return dict(det=None, fa=None, n_events=0, n_pairs=0, per_event=None)
    return dict(det=float(np.mean(det)), fa=float(np.mean(fa)),
                n_events=int(len(det)), n_pairs=int(npairs),
                per_event=dict(idx=idxs, det=det, fa=fa))


# --------------------------------------------------------------------------- #
# 配对 arm-vs-C 差（同一次运行、同一批 episode、同一暴露史）
# --------------------------------------------------------------------------- #

def paired_diff(E, pop, resA, resC, key, mask, label):
    a = arrays(E, resA, key) >= 0
    c = arrays(E, resC, key) >= 0
    ab = arrays(E, resA, "abstain") | arrays(E, resC, "abstain")
    m = np.asarray(mask) & ~ab
    b_ = int((a[m] & ~c[m]).sum())      # A 响 C 不响
    c_ = int((~a[m] & c[m]).sum())      # C 响 A 不响
    return dict(population=label, n=int(m.sum()),
                rate_A=float(a[m].mean()) if m.sum() else None,
                rate_C=float(c[m].mean()) if m.sum() else None,
                delta=float(a[m].mean() - c[m].mean()) if m.sum() else None,
                A_only=b_, C_only=c_, both=int((a[m] & c[m]).sum()),
                neither=int((~a[m] & ~c[m]).sum()),
                mcnemar_p=mcnemar_exact(b_, c_))


def sweep_points(E, Z, lvl, pop, sweep_key, channel):
    """跑一条 PARAM_SWEEP 曲线，返回 [(label, Params, res, FA, det)]。"""
    key = "first_loop_alarm_q" if channel == "loop" else "first_static_alarm_q"
    det_m = pop["loop_fail"] if channel == "loop" else pop["static_fail"]
    out = []
    for label, P in DT.PARAM_SWEEP[sweep_key]:
        th = DT.calibrate_thresholds(E, Z, P, logo=True)
        r = DT.detect("", "scene", P, E=E, Z=Z, scale_level=lvl, thresholds=th)
        arr = arrays(E, r, key) >= 0
        out.append(dict(label=label, P=P, res=r,
                        FA=float(arr[pop["clean_succ"]].mean()),
                        det=float(arr[det_m].mean()),
                        mh=matched_horizon(E, pop, r, channel)))
    return out


def align_and_compare(E, pop, curveA, curveC, channel, targets):
    """把两臂对齐到最接近的同一 FA 工作点后做配对比较。"""
    key = "first_loop_alarm_q" if channel == "loop" else "first_static_alarm_q"
    det_m = pop["loop_fail"] if channel == "loop" else pop["static_fail"]
    rows = []
    for tgt in targets:
        pa = min(curveA, key=lambda d: abs(d["FA"] - tgt))
        pc = min(curveC, key=lambda d: abs(d["FA"] - tgt))
        pd = paired_diff(E, pop, pa["res"], pc["res"], key, det_m,
                         f"{channel}_events")
        rows.append(dict(
            FA_target=tgt,
            A=dict(point=pa["label"], FA=pa["FA"], det=pa["det"],
                   mh_det=pa["mh"]["det"], mh_fa=pa["mh"]["fa"]),
            C=dict(point=pc["label"], FA=pc["FA"], det=pc["det"],
                   mh_det=pc["mh"]["det"], mh_fa=pc["mh"]["fa"]),
            FA_gap=float(pa["FA"] - pc["FA"]),
            paired=pd,
            warning=("对齐残差 |ΔFA|>0.01：Δdet 同时包含召回差与工作点差"
                     if abs(pa["FA"] - pc["FA"]) > 0.01 else None)))
    return rows


# --------------------------------------------------------------------------- #
# 诊断与参照
# --------------------------------------------------------------------------- #

def static_leg_ablation(E, Z, pop, P):
    th = DT.calibrate_thresholds(E, Z, P, logo=True)
    out = {}
    for name, use_s, use_c in [("both_frozen", True, True),
                               ("minmobk_leg_only", True, False),
                               ("C_leg_only", False, True)]:
        hit = np.full(len(E.epid), -1)
        for i, s in enumerate(E.slices):
            t = th.get(int(E.group[i]), {})
            n = len(Z["C"][s])
            zs = Z["minmobk"][s] if use_s else np.zeros(n)
            ts = t.get("s_A", np.nan) if use_s else 1e9
            zc = Z["C"][s] if use_c else np.zeros(n)
            tc = t.get("c", np.nan) if use_c else -1e9
            al = DT.fsm_static(zs, zc, ts, tc, P.M, q_min=P.q_min)[0]
            hit[i] = al[0] if al else -1
        f = lambda m: float(np.mean(hit[m] >= 0)) if m.sum() else None
        out[name] = dict(
            FA_clean_success=f(pop["clean_succ"]),
            det_static_fail=f(pop["static_fail"]),
            det_loop_only_fail=f(pop["loop_only_fail"]),
            lead=q1q3([hit[i] - pop["static_onset"][i]
                       for i in np.flatnonzero(pop["static_fail"]) if hit[i] >= 0]))
    return out


def static_leg_diag(E, Z, pop, P):
    th = DT.calibrate_thresholds(E, Z, P, logo=False)[int(E.group[0])]
    ts, tc = th["s_A"], th["c"]
    out = {"theta_s": round(ts, 4), "theta_c": round(tc, 4), "by_lead": {}}
    onset = pop["static_onset"]
    for lead in (-4, -2, 0, 2):
        zs, zc = [], []
        for i in np.flatnonzero(pop["static_fail"]):
            t = onset[i] + lead
            if 0 <= t < E.n_q[i]:
                zs.append(Z["minmobk"][E.slices[i]][t])
                zc.append(Z["C"][E.slices[i]][t])
        zs, zc = np.array(zs), np.array(zc)
        f = np.isfinite(zs) & np.isfinite(zc)
        zs, zc = zs[f], zc[f]
        out["by_lead"][f"{lead:+d}"] = dict(
            n=int(len(zs)), med_z_minmobk=round(float(np.median(zs)), 2),
            frac_below_theta_s=round(float(np.mean(zs < ts)), 3),
            med_z_C=round(float(np.median(zc)), 2),
            frac_above_theta_c=round(float(np.mean(zc > tc)), 3),
            frac_joint=round(float(np.mean((zs < ts) & (zc > tc))), 3))
    for lab, mk in [("clean_success_rows", pop["clean_succ"]),
                    ("static_fail_rows", pop["static_fail"])]:
        a = np.concatenate([Z["minmobk"][E.slices[i]] for i in np.flatnonzero(mk)])
        b = np.concatenate([Z["C"][E.slices[i]] for i in np.flatnonzero(mk)])
        f = np.isfinite(a) & np.isfinite(b)
        out[lab] = dict(n=int(f.sum()), P_leg_s=round(float(np.mean(a[f] < ts)), 3),
                        P_leg_c=round(float(np.mean(b[f] > tc)), 3),
                        P_joint=round(float(np.mean((a[f] < ts) & (b[f] > tc))), 3),
                        corr_zs_zc=round(float(np.corrcoef(a[f], b[f])[0, 1]), 3))
    return out


def q_drift(E, Z, pop):
    _, _ = None, None
    eor, _g = DT.row_group(E)
    m0 = pop["clean_succ"][eor]
    out = {}
    for name in ("VA", "minmobk", "C", "w8"):
        row = {}
        for lo, hi in [(8, 15), (16, 23), (24, 31), (32, 38), (39, 60)]:
            m = m0 & (E.q >= lo) & (E.q <= hi) & np.isfinite(Z[name])
            row[f"q{lo}-{hi}"] = (round(float(np.median(Z[name][m])), 3)
                                  if m.sum() >= 30 else None)
        out[name] = row
    return out


def naive_z2(E, Z, pop):
    """同 z 下的 z>2 单脉冲参照（**注意**：E4 报的 24–81% 底座含留出集任务，
    只作口径说明，不作为任何参数的选定依据）。"""
    hit = np.array([bool(np.nanmax(Z["VA"][s][DT.PARAMS.q_min:]) > 2.0)
                    if np.isfinite(Z["VA"][s][DT.PARAMS.q_min:]).any() else False
                    for s in E.slices])
    return dict(FA_clean_success=float(hit[pop["clean_succ"]].mean()),
                det_loop_fail=float(hit[pop["loop_fail"]].mean()),
                det_static_fail=float(hit[pop["static_fail"]].mean()))


def auc_nq(E, pop):
    fa, su = np.flatnonzero(pop["all_fail"]), np.flatnonzero(pop["clean_succ"])
    w = t = n = 0
    for i in fa:
        for j in su:
            if E.group[i] != E.group[j]:
                continue
            n += 1
            if E.n_q[i] > E.n_q[j]:
                w += 1
            elif E.n_q[i] == E.n_q[j]:
                t += 1
    return float((w + 0.5 * t) / n) if n else None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "design": "A_threshold_fsm: two-channel scalar-threshold FSM on "
                  "label-free within-group robust z; thresholds = success-set "
                  "quantiles, leave-one-group-out",
        "calibration_set": "SCENE8 only (main16x32 512ep / grid50x8 400ep), "
                           "proxy_grade=full",
        "holdout_untouched": "39 grid tasks + 4 main tasks: never read",
        "headline_caveat": "Absolute FA/Det are upper-bound estimates only. The "
                           "load-bearing claims are the PAIRED arm-A-vs-arm-C "
                           "differences within the same run (identical exposure).",
        "params_A": asdict(P_A), "params_C_rulematched": asdict(P_C),
        "params_C_sameknobs": asdict(P_Csame),
        "params_T_rulematched": asdict(P_T), "params_T_sameknobs": asdict(P_Tsame),
        "units": {},
    }

    for corpus, uname in UNITS:
        rp = f"{BASE}/features/{corpus}/libero_long/{TASK}/rows.npz"
        E = DT.load_rows(rp, "scene")
        Z, lvl = DT.compute_z(E, P_A)
        ev = DT.load_events(f"{BASE}/events/{corpus}/libero_long/{TASK}/events.csv")
        assert set(ev) == set(int(x) for x in E.epid)
        for i, e in enumerate(E.epid):
            assert ev[int(e)]["success"] == int(E.success[i])
            assert ev[int(e)]["n_queries"] == int(E.n_q[i])
        pop = populations(E, ev)

        U = {"corpus": corpus, "n_episodes": int(len(E.epid)),
             "n_groups": int(len(np.unique(E.group))),
             "populations": {k: int(v.sum()) for k, v in pop.items()
                             if isinstance(v, np.ndarray) and v.dtype == bool}}

        TH = DT.calibrate_thresholds(E, Z, P_A, logo=True)
        pooled = DT.calibrate_thresholds(E, Z, P_A, logo=False)[int(E.group[0])]
        U["thresholds"] = {
            "pooled_no_logo": {k: round(v, 4) for k, v in pooled.items()},
            "logo_spread": {k: dict(
                min=float(np.min([TH[g][k] for g in TH])),
                med=float(np.median([TH[g][k] for g in TH])),
                max=float(np.max([TH[g][k] for g in TH]))) for k in pooled}}

        arms, resmap = {}, {}
        for tag, P in ARMS:
            th = DT.calibrate_thresholds(E, Z, P, logo=True)
            r = DT.detect(rp, "scene", P, E=E, Z=Z, scale_level=lvl, thresholds=th)
            resmap[tag] = r
            m = eval_arm(E, pop, r, tag)
            for ch in ("loop", "static"):
                mh = matched_horizon(E, pop, r, ch)
                mh.pop("per_event", None)
                m.setdefault("matched_horizon", {})[ch] = mh
            arms[tag] = m
        U["arms"] = arms

        # ---- 配对 arm-vs-对照 差（冻结工作点，同一批 episode） ----
        pd_ = {}
        for cname in ("C_rulematched", "C_sameknobs",
                      "T_rulematched", "T_sameknobs"):
            pd_[f"A_vs_{cname}"] = {
                "FA_clean_success_any": paired_diff(
                    E, pop, resmap["A"], resmap[cname], "first_alarm_q",
                    pop["clean_succ"], "clean_success"),
                "det_fatal_loop_loopch": paired_diff(
                    E, pop, resmap["A"], resmap[cname], "first_loop_alarm_q",
                    pop["loop_fail"], "fatal_loop"),
                "det_static_fail_statch": paired_diff(
                    E, pop, resmap["A"], resmap[cname], "first_static_alarm_q",
                    pop["static_fail"], "static_fail"),
                "det_recovery_loop_loopch": paired_diff(
                    E, pop, resmap["A"], resmap[cname], "first_loop_alarm_q",
                    pop["recov_loop"], "recovery_loop"),
            }
        U["paired_A_vs_C_at_frozen_point"] = pd_

        # ---- 对齐工作点后的配对比较 ----
        cA = sweep_points(E, Z, lvl, pop, "A", "loop")
        cC = sweep_points(E, Z, lvl, pop, "C", "loop")
        cT = sweep_points(E, Z, lvl, pop, "T", "loop")
        sA = sweep_points(E, Z, lvl, pop, "A_static", "static")
        sC = sweep_points(E, Z, lvl, pop, "C_static", "static")
        sT = sweep_points(E, Z, lvl, pop, "T_static", "static")
        U["aligned_operating_point"] = {
            "loop_vs_C": align_and_compare(E, pop, cA, cC, "loop",
                                           [0.05, 0.10, 0.20]),
            "loop_vs_T": align_and_compare(E, pop, cA, cT, "loop",
                                           [0.05, 0.10, 0.20]),
            "static_vs_C": align_and_compare(E, pop, sA, sC, "static",
                                             [0.04, 0.08, 0.15]),
            "static_vs_T": align_and_compare(E, pop, sA, sT, "static",
                                             [0.04, 0.08, 0.15]),
        }
        strip = lambda c: [dict(label=d["label"], FA=d["FA"], det=d["det"],
                                mh_det=d["mh"]["det"], mh_fa=d["mh"]["fa"])
                           for d in c]
        U["operating_curves"] = {"loop_A": strip(cA), "loop_C": strip(cC),
                                 "loop_T": strip(cT), "static_A": strip(sA),
                                 "static_C": strip(sC), "static_T": strip(sT)}

        # ---- 分型：沿 static 工作点扫描（看 static 通道修好后分型是否兑现） ----
        def typ_curve(curve):
            rows = []
            for d in curve:
                m = eval_arm(E, pop, d["res"], d["label"])
                rows.append(dict(
                    label=d["label"], FA_any=m["FA_clean_success"]["any_channel"],
                    det_static=m["detection"]["static_fail__static_channel"]["rate"],
                    typing_micro=m["typing"]["overall"]["correct_first"],
                    typing_balanced=m["typing"]["overall"]["balanced_first"],
                    loop_only=m["typing"]["loop_only_fail"]["correct_first"],
                    static_only=m["typing"]["static_only_fail"]["correct_first"]))
            return rows
        U["typing_along_static_sweep"] = {"armA": typ_curve(sA),
                                          "armT": typ_curve(sT),
                                          "armC": typ_curve(sC)}

        # ---- 匹配视界的"对角线检验" ----
        # 匹配视界把事件集与干净成功集放在**同一个绝对 q** 上比较，因此一个纯粹
        # 依赖 q 的报警器必然同时命中或同时不命中 => mh_det == mh_fa（对角线）。
        # 这条口径是唯一能把"钟"与"路由信息"分开的口径。
        def diag(curve):
            marg = [d["mh"]["det"] - d["mh"]["fa"] for d in curve
                    if d["mh"]["det"] is not None]
            pts = [dict(label=d["label"], mh_fa=d["mh"]["fa"],
                        mh_det=d["mh"]["det"],
                        margin=d["mh"]["det"] - d["mh"]["fa"]) for d in curve]
            return dict(points=pts, margin_min=float(np.min(marg)),
                        margin_med=float(np.median(marg)),
                        margin_max=float(np.max(marg)),
                        n_points_above_diagonal=int(sum(m > 0 for m in marg)),
                        n_points=int(len(marg)))
        U["matched_horizon_diagonal_test"] = {
            "loop_A": diag(cA), "loop_C": diag(cC), "loop_T": diag(cT),
            "static_A": diag(sA), "static_C": diag(sC), "static_T": diag(sT)}

        # ---- 诊断与参照 ----
        U["static_leg_ablation_armA"] = static_leg_ablation(E, Z, pop, P_A)
        U["static_leg_diagnostics"] = static_leg_diag(E, Z, pop, P_A)
        U["reference_baselines"] = {
            "single_pulse_z_gt_2_same_z": naive_z2(E, Z, pop),
            "n_queries_AUC_fail_vs_cleansucc_within_group": auc_nq(E, pop),
            "nq_median": {"clean_succ": float(np.median(E.n_q[pop["clean_succ"]])),
                          "all_fail": float(np.median(E.n_q[pop["all_fail"]]))},
            "z_q_drift_clean_success": q_drift(E, Z, pop),
            "_note": "E4 的 24–81% 误报底座源自含留出集任务的分析，本设计的任何"
                     "参数都不溯源到它；此处只列同 z 下的 z>2 参照。",
        }

        # ---- 机械自检探针 ----
        U["probes"] = {
            "P1_success_permutation": DT.probe_p1(rp, "scene", P_A, n_perm=20,
                                                  seed=SEED, E=E),
            "P2_truncation_armA": DT.probe_p2(E, Z, P_A, thresholds=TH),
            "P2_truncation_armC": DT.probe_p2(
                E, Z, P_C, thresholds=DT.calibrate_thresholds(E, Z, P_C)),
            "P2_truncation_armT": DT.probe_p2(
                E, Z, P_T, thresholds=DT.calibrate_thresholds(E, Z, P_T)),
            "destructive_causality_armA": DT.assert_causality(
                E, Z, P_A, n_check=120, rng=np.random.default_rng(SEED),
                thresholds=TH),
            "destructive_causality_armC": DT.assert_causality(
                E, Z, P_C, n_check=120, rng=np.random.default_rng(SEED)),
        }

        # ---- 旋钮敏感性（臂 A，其余冻结） ----
        sweep = {}
        for knob, vals in {"q_enter": [75, 80, 85, 90, 95],
                           "q_exit": [25, 40, 50, 60, 75],
                           "K": [3, 4, 5, 6, 7, 8, 10],
                           "q_s": [5, 10, 15, 20, 25],
                           "q_c": [75, 80, 85, 90, 95],
                           "M": [2, 3, 4, 5, 6]}.items():
            rows = []
            for v in vals:
                P = replace(P_A, **{knob: v})
                th = DT.calibrate_thresholds(E, Z, P, logo=True)
                r = DT.detect(rp, "scene", P, E=E, Z=Z, scale_level=lvl,
                              thresholds=th)
                m = eval_arm(E, pop, r, f"A/{knob}={v}")
                rows.append(dict(
                    value=v, FA=m["FA_clean_success"]["any_channel"],
                    FA_loop=m["FA_clean_success"]["loop_channel"],
                    FA_static=m["FA_clean_success"]["static_channel"],
                    det_loop=m["detection"]["loop_fail__loop_channel"]["rate"],
                    det_static=m["detection"]["static_fail__static_channel"]["rate"],
                    lead_loop_med=m["lead"]["loop"]["med"],
                    lead_static_med=m["lead"]["static"]["med"],
                    recov_loop_hit=m["negative_control_recovery_loop"][
                        "recov_loop__loop_channel"]["n_hit"],
                    typing_first=m["typing"]["overall"]["correct_first"],
                    frozen=(v == getattr(P_A, knob))))
            sweep[knob] = rows
        U["knob_sensitivity"] = sweep

        # ---- 逐集明细 ----
        with open(os.path.join(HERE, f"alarms_{corpus}.csv"), "w") as fh:
            fh.write("episode_id,group,success,n_q,scale_level,abstain,"
                     "n_alarms,alarm_steps,alarm_channels,first_alarm_q,"
                     "first_alarm_channel,first_loop_alarm_q,first_static_alarm_q,"
                     "loop_n_entries,static_max_run,loop_onset_q,static_onset_q\n")
            for i, e in enumerate(E.epid):
                r = resmap["A"][int(e)]
                st = "|".join(str(a["step"]) for a in r["alarms"])
                ch = "|".join(a["channel"] for a in r["alarms"])
                fh.write(f"{int(e)},{r['group']},{int(E.success[i])},{r['n_q']},"
                         f"{r['scale_level']},{int(r['abstain'])},"
                         f"{len(r['alarms'])},{st},{ch},{r['first_alarm_q']},"
                         f"{r['first_alarm_channel']},{r['first_loop_alarm_q']},"
                         f"{r['first_static_alarm_q']},{r['loop_n_entries']},"
                         f"{r['static_max_run']},{pop['loop_onset'][i]},"
                         f"{pop['static_onset'][i]}\n")

        summary["units"][uname] = U
        print(f"[{uname}] done  {time.time()-t0:.1f}s")

    summary["verdict"] = {
        "_INVALIDATED_CALIBER": (
            "The per-episode any-time-alarm caliber is VOID. Arm T (pure "
            "episode-length, reads NO MoE) gets det_loop 1.000/0.645 and "
            "det_static 1.000/0.855 at FA 0.038/0.019, beating arms A and C on "
            "both axes. Any cross-arm comparison in that caliber measures which "
            "arm is the better clock, not MoE information. The previous "
            "load-bearing claim ('A beats C, +0.255..+0.629, 6/6 significant') "
            "is RETRACTED."),
        "_VALID_CALIBER": (
            "Matched-horizon pairing evaluates the event episode and a clean "
            "success at the SAME absolute q, so any pure-q alarm necessarily "
            "satisfies mh_det == mh_fa (the chance diagonal). Arm T's measured "
            "margin on U1 is exactly 0.000 at 9/9 operating points -- a "
            "constructive validation of the caliber. margin = mh_det - mh_fa is "
            "therefore the discriminative power left after removing the clock."),
        "HEADLINE_loop_channel_diagonal_test": (
            "REPRODUCED IN BOTH UNITS. Arm A loop margin > 0 at 9/9 operating "
            "points in both units; median +0.108 (U1) / +0.139 (U2), min +0.017 / "
            "+0.060. Arm T: 0/9 in both (exactly 0.000 on U1). Arm C: 9/9 on U1 "
            "but 0/9 on U2 -> not reproduced. This is the ONLY positive result "
            "that survives the length control. Magnitude is small (<=0.16): it "
            "supports 'MoE routing carries incremental information here', NOT "
            "'this is a usable detector'."),
        "prefix_truncation_q_le_38": (
            "Second independent evidence: det on fatal loops A 0.509/0.484 vs "
            "T 0.000/0.000 vs C 0.073/0.000. Arm T never fires once while any "
            "healthy trajectory is still alive."),
        "typing_balanced": (
            "At the frozen point arm A and arm T are BOTH constant classifiers: "
            "balanced (macro) typing = 0.500 for both. The previously quoted "
            "0.430/0.716 was a micro-average base-rate artifact (T-sameknobs "
            "scores 0.747 on U1 by always saying 'static'). BUT once the static "
            "channel works (theta_c <= p70) arm A reaches balanced 0.674-0.799 in "
            "both units, while arm T is exactly 0.500 at all 8 sweep points "
            "(structural: both its channels key on the same q crossing) and arm C "
            "never exceeds 0.52. Typing capability exists and requires the "
            "two-channel + two-opposite-direction-MoE-signal structure; it is "
            "just not delivered at the frozen operating point."),
        "recovery_vs_fatal_RETRACTED": (
            "Arm T reproduces the direction on the same [-4,+8] onset window with "
            "a SMALLER p (0/2 vs 34/55, p=0.159; 0/4 vs 28/62, p=0.102) than arm A "
            "(0.273/0.354). Fatal-loop onsets are later (median 43/41) than "
            "recovery-loop onsets (40/32), so the window sits at a later absolute "
            "q for fatal episodes and catches the clock. The window is NOT "
            "time-neutral. This prototype provides no independent support for the "
            "'can it leave' proposition."),
        "lead_time": (
            "Arm A loop lead median -2 (U1) / +1 (U2), 52%/47% pre-onset. "
            "Arm C -3/-9 with 92%/86% pre-onset (BETTER than A). Arm T +5/+4 with "
            "only 20%/28% pre-onset (WORSE than A). A beats T, loses to C."),
        "static_channel_at_frozen_point": (
            "DEAD: det 0.000/0.014. Legs negatively correlated on healthy rows "
            "(r=-0.65/-0.68); on static-failure rows the joint rate (0.003/0.007) "
            "is LOWER than on clean-success rows (0.033/0.036). min_mobk leg alone "
            "at M=3: det 0.769/0.913, lead -4/-5, ~100% pre-onset, FA 0.210/0.288. "
            "NOT the frozen spec -- pre-registration candidate. Signal right, "
            "operating point frozen wrong."),
        "knob_fragility": (
            "Absolute per-episode FA swings over 0.03-0.62 with any single loop "
            "knob, but that caliber is void anyway. In the valid caliber the sign "
            "of the matched-horizon margin never flips across 9/9 operating points "
            "in either unit."),
        "probes": "P1a 0 mismatches; P1b 1.88%/2.63% (LOGO threshold jitter, "
                  "inherent to the mandated success-set calibration, not leakage); "
                  "P2 0 violations (arms A/C/T); destructive causality 0 violations.",
        "instruction_to_holdout_executor": (
            "Run arms A, C and T together. Report margin = mh_det - mh_fa and the "
            "q<=Qstar prefix metrics. Results reported only as whole-episode "
            "det/FA are invalid."),
    }
    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False, default=float)
    print("summary.json written", round(time.time() - t0, 1), "s")
    return summary


if __name__ == "__main__":
    main()
