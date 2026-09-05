"""E4 复发脉冲列 + lock-in 亚型 + 恢复性 loop 负对照（PROTOCOL §4/§5-E4/§8，AUDIT §7/§8）。

子实验：
(a) 周期复发谱：late 窗逐行 g = mob_1 − min_{k=2..8} mob_k，集级=窗内中位；
    失败 vs 成功组内配对 AUC（进 family1）；best-k 分布（描述）。
(b) 脉冲：score = (V+A) 的组内无标签 robust z（组中位/1.4826·MAD，全组含成败全 query 行）。
    脉冲 = z>2 连续 ≥1 行；孤立 = 结束后 3 行内出现 z<1；train = ≥3 脉冲且相邻
    (next_start−prev_end) ≤ 8；post_pulse_return = 各脉冲结束后 ≤5 行 z 均值的集级均值。
    人群（排除红旗任务）：恢复性 loop = loop_onset≥0 ∧ success=1；致败 loop = loop_onset≥0
    ∧ success=0；干净成功 = success=1 ∧ 无任何事件。
    family2（每单位 5 格 joint maxT，episode 级组内置换，y=1=致败）：
    {cycle_gain, pulse_count, train_flag, isolated_pulse_flag, post_pulse_return}。
    误报审计：干净成功集的脉冲/孤立脉冲/train 集率。
(c) lock-in 象限：late 窗集级均值 (V,A,margin,mob1_w8)；组内成功中位定阈（组内成功且
    有窗覆盖 ≥3 集才定义）；lock-in = V<∧A<∧margin>∧mob1_w8<；switching = V>∧A>。
    family1（每单位 3 格 joint maxT，episode 级组内置换，y=1=失败，置换下阈值随标签重算）：
    {cycle_gain, lockin_flag, switch_flag}。交叉表 + lock-in 失败清单（描述）。

单位：grid50x8 四套件分开 + main16x32 整体；family2 另加 grid50x8/ALL_pooled 补充腿
（组仍 (task,scene)，只是跨套件合并配对）。组：grid=(task,scene)，main=(task,init_state
即 scene 列)。窗：libero_long（含 SCENE8）q∈[25,35]，其余 q∈[12,18]。
红旗（loop/trap 通道无效，AUDIT §7.1/§8.4）：grid long KITCHEN_SCENE3、两语料 goal
open_the_middle_drawer —— 从 (b) 人群与 (c) 交叉表剔除；(a)/(c) 的 fail-vs-succ 不涉事件，保留。

运行：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_e4.py
"""

import csv
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
OUT = os.path.join(BASE, "E4_recurrence_lockin")
sys.path.insert(0, os.path.join(BASE, "phenotype"))
from stats import joint_maxt, paired_auc  # noqa: E402

SEED = 20260903
NPERM = 2000
Z_PULSE, Z_RECOVER = 2.0, 1.0
ISO_WIN, RET_WIN = 3, 5          # 孤立回落窗 / post_pulse_return 窗（行）
TRAIN_GAP, TRAIN_MIN = 8, 3      # train：相邻 next_start−prev_end ≤8，≥3 个脉冲
NEAR_LO, NEAR_HI = -2, 5         # "onset 附近有脉冲" 描述窗（lead）
LEADS = np.arange(-4, 9)         # onset 对齐曲线
MIN_SUCC_THR = 3                 # (c) 组内定阈所需最少成功覆盖集
MAD_C = 1.4826

REDFLAG = {
    ("grid50x8", "libero_long",
     "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
    ("grid50x8", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("main16x32", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
}
FAM1 = ("cycle_gain", "lockin_flag", "switch_flag")
FAM2 = ("cycle_gain", "pulse_count", "train_flag", "isolated_pulse_flag",
        "post_pulse_return")


def rng_for(*names):
    return np.random.default_rng(
        [SEED] + [zlib.crc32(str(n).encode()) for n in names])


def window_of(suite):
    return (25, 35) if suite == "libero_long" else (12, 18)


# ------------------------------------------------------------------ pulse layer

def detect_pulses(z):
    """z>2 的极大连续段列表 [(start,end)]（行下标，含端点）。NaN 视为不越阈。"""
    above = np.zeros(len(z), bool)
    fin = np.isfinite(z)
    above[fin] = z[fin] > Z_PULSE
    if not above.any():
        return []
    d = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(z) - 1]
    return list(zip(starts, ends))


def pulse_stats_ep(z):
    P = detect_pulses(z)
    n_iso, rets = 0, []
    for s, e in P:
        seg = z[e + 1:e + 1 + ISO_WIN]
        seg = seg[np.isfinite(seg)]
        if len(seg) and seg.min() < Z_RECOVER:
            n_iso += 1                       # 集尾无行可验证 → 不算孤立
        seg5 = z[e + 1:e + 1 + RET_WIN]
        seg5 = seg5[np.isfinite(seg5)]
        if len(seg5):
            rets.append(seg5.mean())
    train = 0
    if len(P) >= TRAIN_MIN:
        chain = 1
        for i in range(1, len(P)):
            chain = chain + 1 if P[i][0] - P[i - 1][1] <= TRAIN_GAP else 1
            if chain >= TRAIN_MIN:
                train = 1
                break
    return dict(n_pulses=len(P), n_iso=n_iso, iso_flag=int(n_iso > 0),
                train_flag=train,
                post_ret=float(np.mean(rets)) if rets else np.nan, pulses=P)


# ------------------------------------------------------------------- data load

def load_task(corpus, suite, task):
    d = os.path.join(BASE, "features", corpus, suite, task)
    z = np.load(os.path.join(d, "rows.npz"))
    ep = z["episode_id"].astype(np.int64)
    cs = z["control_step"].astype(np.int64)
    assert np.all(np.diff(ep) >= 0), (corpus, suite, task, "rows 未按 episode 分块")
    uniq, first, cnt = np.unique(ep, return_index=True, return_counts=True)
    start_of = np.zeros(uniq.max() + 1, np.int64)
    start_of[uniq] = cs[first]
    q = cs - start_of[ep]
    assert np.all(q == np.concatenate([np.arange(c) for c in cnt])), "control_step 不连续"

    ev_p = os.path.join(BASE, "events", corpus, suite, task, "events.csv")
    ev = {int(e["episode_id"]): e for e in csv.DictReader(open(ev_p))}
    assert set(ev) == set(int(u) for u in uniq), "events/features episode 集不一致"

    V, A = z["late_flow_volatility"].astype(np.float64), z["route_acceleration"].astype(np.float64)
    M, W = z["top12_margin"].astype(np.float64), z["mob1_w8"].astype(np.float64)
    mk = z["mob_k"].astype(np.float64)
    x = V + A
    scene_row = z["scene"].astype(np.int64)

    # 组内（scene）robust z：全组全行、无标签
    zs = np.full(len(x), np.nan)
    bad_scale = []
    for sc in np.unique(scene_row):
        m = scene_row == sc
        med = np.median(x[m])
        mad = np.median(np.abs(x[m] - med)) * MAD_C
        if mad <= 1e-12:
            bad_scale.append(int(sc))
            continue
        zs[m] = (x[m] - med) / mad

    lo, hi = window_of(suite)
    g_row = mk[:, 0] - np.min(mk[:, 1:], axis=1)   # 窗内 q≥12>8 → 全有限
    recs, zlist = [], []
    for i, e in enumerate(uniq):
        s = slice(first[i], first[i] + cnt[i])
        qe = q[s]
        wm = (qe >= lo) & (qe <= hi)
        n_win = int(wm.sum())
        if n_win:
            cyc = float(np.nanmedian(g_row[s][wm]))
            med_k = np.nanmedian(mk[s][wm], axis=0)
            bestk = int(np.argmin(med_k)) + 1 if np.isfinite(med_k).all() else -1
            wV, wA = float(np.nanmean(V[s][wm])), float(np.nanmean(A[s][wm]))
            wM, wW = float(np.nanmean(M[s][wm])), float(np.nanmean(W[s][wm]))
        else:
            cyc, bestk, wV, wA, wM, wW = np.nan, -1, np.nan, np.nan, np.nan, np.nan
        ze = zs[s].copy()
        ps = pulse_stats_ep(ze)
        e_ev = ev[int(e)]
        assert int(e_ev["success"]) == int(z["success"][first[i]])
        assert int(e_ev["n_queries"]) == cnt[i]
        o_loop = int(e_ev["loop_onset_q"])
        near, nearest_rel, first_pq = 0, np.nan, -1
        if ps["pulses"]:
            first_pq = int(ps["pulses"][0][0])
        if o_loop >= 0 and ps["pulses"]:
            starts = np.array([p[0] for p in ps["pulses"]])
            ends = np.array([p[1] for p in ps["pulses"]])
            near = int(np.any((starts <= o_loop + NEAR_HI) & (ends >= o_loop + NEAR_LO)))
            nearest_rel = float(starts[np.argmin(np.abs(starts - o_loop))] - o_loop)
        recs.append(dict(
            corpus=corpus, suite=suite, task=task,
            scene=int(z["scene"][first[i]]), repeat=int(z["repeat"][first[i]]),
            epid=int(e), succ=int(e_ev["success"]), nq=int(cnt[i]),
            o_loop=o_loop, o_static=int(e_ev["static_onset_q"]),
            o_trap=int(e_ev["trap_onset_q"]),
            n_win=n_win, cyc=cyc, bestk=bestk, wV=wV, wA=wA, wM=wM, wW=wW,
            n_pulses=ps["n_pulses"], n_iso=ps["n_iso"], iso_flag=ps["iso_flag"],
            train_flag=ps["train_flag"], post_ret=ps["post_ret"],
            near_onset=near, nearest_pulse_rel_onset=nearest_rel,
            first_pulse_q=first_pq))
        zlist.append(ze)
    return recs, zlist, bad_scale


# --------------------------------------------------- family1 (fail vs succ) ---

def build_padded(g, S, Cov, cyc, X4):
    ug, inv = np.unique(g, return_inverse=True)
    G = len(ug)
    cnt = np.bincount(inv, minlength=G)
    m = int(cnt.max())
    slot = np.zeros(len(g), np.int64)
    seen = np.zeros(G, np.int64)
    for i in np.argsort(inv, kind="stable"):
        slot[i] = seen[inv[i]]
        seen[inv[i]] += 1
    P = np.zeros((G, m), bool)
    Sp = np.zeros((G, m), bool)
    Cp = np.zeros((G, m), bool)
    CY = np.full((G, m), np.nan)
    X = np.full((G, m, 4), np.nan)
    P[inv, slot] = True
    Sp[inv, slot] = S
    Cp[inv, slot] = Cov
    CY[inv, slot] = cyc
    X[inv, slot] = X4
    return P, Sp, Cp, CY, X, inv, slot


def _fam1_stat(P, C0, CY, X, CMP, fin, Sb, want_flags=False):
    fail, succ = P & ~Sb, P & Sb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thr = np.nanmedian(np.where((Sb & C0)[:, :, None], X, np.nan), axis=1)
    thr[(Sb & C0).sum(1) < MIN_SUCC_THR] = np.nan
    valid = C0 & np.isfinite(thr).all(1)[:, None]
    V, A, Mg, W = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    t0, t1, t2, t3 = (thr[:, k][:, None] for k in range(4))
    lock = valid & (V < t0) & (A < t1) & (Mg > t2) & (W < t3)
    swi = valid & (V > t0) & (A > t1)
    out = {}
    for name, fl in (("lockin_flag", lock), ("switch_flag", swi)):
        f1 = (fail & fl).sum(1)
        f0 = (fail & valid & ~fl).sum(1)
        s1 = (succ & fl).sum(1)
        s0 = (succ & valid & ~fl).sum(1)
        num = (f1 * s0 + 0.5 * (f1 * s1 + f0 * s0)).sum()
        den = ((f1 + f0) * (s1 + s0)).sum()
        out[name] = (num / den if den else np.nan, int(den),
                     int(f1.sum()), int(f0.sum()), int(s1.sum()), int(s0.sum()))
    Fi = (fail & fin).astype(np.float64)
    Sj = (succ & fin).astype(np.float64)
    num = float(np.einsum("gi,gj,gij->", Fi, Sj, CMP))
    den = float((Fi.sum(1) * Sj.sum(1)).sum())
    out["cycle_gain"] = (num / den if den else np.nan, int(den), -1, -1, -1, -1)
    if want_flags:
        return out, lock, swi, valid
    return out


def family1_unit(tag, g, S, Cov, cyc, X4, rng):
    P, S0, C0, CY, X, inv, slot = build_padded(g, S, Cov, cyc, X4)
    G, m = P.shape
    fin = np.isfinite(CY)
    CMP = ((CY[:, :, None] > CY[:, None, :]).astype(np.float64)
           + 0.5 * (CY[:, :, None] == CY[:, None, :]))
    CMP[~(fin[:, :, None] & fin[:, None, :])] = 0.0
    n_s = (S0 & P).sum(1)
    obs, lock, swi, valid = _fam1_stat(P, C0, CY, X, CMP, fin, S0, want_flags=True)
    dev = {k: abs(v[0] - 0.5) for k, v in obs.items() if np.isfinite(v[0])}
    mx = np.zeros(NPERM)
    for it in range(NPERM):
        R = rng.random((G, m))
        R[~P] = 2.0
        rk = np.argsort(np.argsort(R, 1), 1)
        Sb = rk < n_s[:, None]
        o = _fam1_stat(P, C0, CY, X, CMP, fin, Sb)
        mx[it] = max((abs(v[0] - 0.5) for v in o.values() if np.isfinite(v[0])),
                     default=0.0)
    p = {k: float((mx >= d).sum() + 1) / (NPERM + 1) for k, d in dev.items()}
    # 阈值可定义组数审计 + 逐集观测 flag（映射回输入顺序）
    n_thr_ok = int(((S0 & C0).sum(1) >= MIN_SUCC_THR).sum())
    ep_lock = np.where(valid[inv, slot], lock[inv, slot].astype(float), np.nan)
    ep_swi = np.where(valid[inv, slot], swi[inv, slot].astype(float), np.nan)
    return obs, p, dict(n_groups=G, n_groups_thr_ok=n_thr_ok), ep_lock, ep_swi


# ------------------------------------------------------------------------ main

def main():
    t0 = time.time()
    print("[load] 45 tasks", flush=True)
    recs, zlist, bad = [], [], []
    def subdirs(p):
        return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))

    for corpus in ("grid50x8", "main16x32"):
        for suite in subdirs(os.path.join(BASE, "features", corpus)):
            for task in subdirs(os.path.join(BASE, "features", corpus, suite)):
                r, zl, bs = load_task(corpus, suite, task)
                recs += r
                zlist += zl
                if bs:
                    bad.append((corpus, suite, task, bs))
    n = len(recs)
    print(f"[load] episodes={n} bad_mad_groups={bad} ({time.time()-t0:.0f}s)", flush=True)

    T = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    gkey = [(r["corpus"], r["task"], r["scene"]) for r in recs]
    uniq_g = {k: i for i, k in enumerate(sorted(set(gkey)))}
    T["gid"] = np.array([uniq_g[k] for k in gkey])
    T["redflag"] = np.array([(r["corpus"], r["suite"], r["task"]) in REDFLAG for r in recs])
    pop = np.full(n, "other", dtype=object)
    is_loop = T["o_loop"] >= 0
    no_ev = (T["o_loop"] < 0) & (T["o_static"] < 0) & (T["o_trap"] < 0)
    pop[is_loop & (T["succ"] == 1)] = "recovering_loop"
    pop[is_loop & (T["succ"] == 0)] = "fatal_loop"
    pop[no_ev & (T["succ"] == 1)] = "clean_success"
    pop[T["redflag"]] = "excluded_redflag"
    T["pop"] = pop
    covered = (T["n_win"] > 0) & np.isfinite(T["cyc"]) & np.isfinite(T["wV"]) \
        & np.isfinite(T["wA"]) & np.isfinite(T["wM"]) & np.isfinite(T["wW"])
    T["covered"] = covered

    units = [("grid50x8/libero_goal", (T["corpus"] == "grid50x8") & (T["suite"] == "libero_goal")),
             ("grid50x8/libero_long", (T["corpus"] == "grid50x8") & (T["suite"] == "libero_long")),
             ("grid50x8/libero_object", (T["corpus"] == "grid50x8") & (T["suite"] == "libero_object")),
             ("grid50x8/libero_spatial", (T["corpus"] == "grid50x8") & (T["suite"] == "libero_spatial")),
             ("main16x32/ALL", T["corpus"] == "main16x32")]

    summary = dict(
        protocol="E4 v1", seed=SEED, nperm=NPERM,
        windows={"libero_long(含SCENE8)": [25, 35], "goal/object/spatial": [12, 18]},
        pulse_def=dict(z="robust z of (V+A), per-group median/1.4826*MAD, all rows label-free",
                       pulse="z>2 run>=1", isolated=f"z<1 within {ISO_WIN} rows after end",
                       train=f">={TRAIN_MIN} pulses, adjacent start-end gap<={TRAIN_GAP}",
                       post_pulse_return=f"mean z over <={RET_WIN} rows after each pulse end, ep=mean over pulses",
                       near_onset_window=[NEAR_LO, NEAR_HI]),
        families=dict(family1_fail_vs_succ=list(FAM1), family2_fatal_vs_recovering=list(FAM2)),
        lockin_rule="V<succMed & A<succMed & margin>succMed & mob1w8<succMed (per-group, "
                    f"succ covered >={MIN_SUCC_THR}); switching: V>succMed & A>succMed",
        redflag_excluded=[list(k) for k in sorted(REDFLAG)],
        auc_orientation="family1: y=1=fail; family2: y=1=fatal_loop",
        bad_mad_groups=[list(map(str, b[:3])) + [b[3]] for b in bad],
        units={})

    ep_lock_all = np.full(n, np.nan)
    ep_swi_all = np.full(n, np.nan)
    csv_rows_pulse = []
    crosstab_rows = []
    lockin_fail_list = []

    for tag, um in units:
        tu = time.time()
        u = {}
        idx = np.flatnonzero(um)
        y_fail = (T["succ"][idx] == 0).astype(int)
        cov = T["covered"][idx]
        # ---- 存活审计 ----
        u["survival"] = dict(
            n_eps=int(len(idx)), n_fail=int(y_fail.sum()),
            covered_succ=int((cov & (y_fail == 0)).sum()),
            covered_fail=int((cov & (y_fail == 1)).sum()))
        gboth = 0
        for g in np.unique(T["gid"][idx]):
            m = (T["gid"][idx] == g) & cov
            if (y_fail[m] == 1).any() and (y_fail[m] == 0).any():
                gboth += 1
        u["survival"]["groups_with_both_covered"] = gboth

        # ---- family1：cycle_gain + lockin + switching ----
        obs, p, aud, ep_lock, ep_swi = family1_unit(
            tag, T["gid"][idx], T["succ"][idx] == 1, cov,
            T["cyc"][idx], np.c_[T["wV"][idx], T["wA"][idx], T["wM"][idx], T["wW"][idx]],
            rng_for(tag, "family1"))
        ep_lock_all[idx] = ep_lock
        ep_swi_all[idx] = ep_swi
        u["family1"] = {}
        for k in FAM1:
            a, den, f1, f0, s1, s0 = obs[k]
            cell = dict(auc_fail_high=round(float(a), 4) if np.isfinite(a) else None,
                        n_pairs=den, p_maxT=round(p[k], 4) if k in p else None)
            if k != "cycle_gain":
                cell.update(frac_fail=round(f1 / max(f1 + f0, 1), 4), n_fail_flagged=f1,
                            n_fail_scored=f1 + f0,
                            frac_succ=round(s1 / max(s1 + s0, 1), 4), n_succ_flagged=s1,
                            n_succ_scored=s1 + s0)
            u["family1"][k] = cell
        u["family1_audit"] = aud

        # ---- (a) best-k 分布（覆盖集，描述）----
        bk = T["bestk"][idx]
        u["best_k"] = {}
        for lab, m in (("fail", cov & (y_fail == 1)), ("succ", cov & (y_fail == 0))):
            cnts = [int((bk[m] == k).sum()) for k in range(1, 9)]
            tot = max(sum(cnts), 1)
            u["best_k"][lab] = dict(counts_k1_8=cnts, frac_k1=round(cnts[0] / tot, 4),
                                    frac_k_ge2=round(sum(cnts[1:]) / tot, 4),
                                    n=sum(cnts),
                                    median_cycle_gain=round(float(np.nanmedian(T["cyc"][idx][m])), 5)
                                    if m.any() else None,
                                    frac_gain_pos=round(float((T["cyc"][idx][m] > 0).mean()), 4)
                                    if m.any() else None)

        # ---- (b) 人群 + family2 ----
        pm = um & ~T["redflag"]
        pops = {k: np.flatnonzero(pm & (T["pop"] == k))
                for k in ("recovering_loop", "fatal_loop", "clean_success")}
        u["populations"] = {k: int(len(v)) for k, v in pops.items()}
        u["pulse_desc"] = {}
        for k, pidx in pops.items():
            if not len(pidx):
                u["pulse_desc"][k] = None
                continue
            pr = T["post_ret"][pidx]
            u["pulse_desc"][k] = dict(
                n=int(len(pidx)),
                pulse_count_med=float(np.median(T["n_pulses"][pidx])),
                pulse_count_mean=round(float(T["n_pulses"][pidx].mean()), 3),
                frac_any_pulse=round(float((T["n_pulses"][pidx] > 0).mean()), 4),
                frac_train=round(float(T["train_flag"][pidx].mean()), 4),
                frac_isolated=round(float(T["iso_flag"][pidx].mean()), 4),
                post_return_med=round(float(np.nanmedian(pr)), 3)
                if np.isfinite(pr).any() else None,
                n_post_return=int(np.isfinite(pr).sum()),
                cycle_gain_med=round(float(np.nanmedian(T["cyc"][pidx])), 5)
                if np.isfinite(T["cyc"][pidx]).any() else None,
                near_onset_pulse_rate=round(float(T["near_onset"][pidx].mean()), 4)
                if k != "clean_success" else None,
                pulses_per_100q=round(float(T["n_pulses"][pidx].sum()
                                            / T["nq"][pidx].sum() * 100), 3))
        # 误报底座（干净成功）
        ci = pops["clean_success"]
        u["false_alarm_base"] = dict(
            n_clean_succ=int(len(ci)),
            ep_rate_any_pulse=round(float((T["n_pulses"][ci] > 0).mean()), 4),
            ep_rate_isolated_pulse=round(float(T["iso_flag"][ci].mean()), 4),
            ep_rate_train=round(float(T["train_flag"][ci].mean()), 4),
            pulses_per_100q=round(float(T["n_pulses"][ci].sum() / T["nq"][ci].sum() * 100), 3)) \
            if len(ci) else None

        # family2：致败(1) vs 恢复性(0)
        lidx = np.concatenate([pops["fatal_loop"], pops["recovering_loop"]])
        u["family2"] = {}
        if len(pops["fatal_loop"]) and len(pops["recovering_loop"]):
            yl = np.array([1] * len(pops["fatal_loop"]) + [0] * len(pops["recovering_loop"]))
            gl = T["gid"][lidx]
            both = 0
            for g in np.unique(gl):
                m = gl == g
                if (yl[m] == 1).any() and (yl[m] == 0).any():
                    both += 1
            vals = dict(cycle_gain=T["cyc"][lidx],
                        pulse_count=T["n_pulses"][lidx].astype(float),
                        train_flag=T["train_flag"][lidx].astype(float),
                        isolated_pulse_flag=T["iso_flag"][lidx].astype(float),
                        post_pulse_return=T["post_ret"][lidx])
            cells = [(k, v, np.isfinite(v), gl) for k, v in vals.items()]
            ep_label = {int(e): int(v) for e, v in zip(lidx, yl)}
            ep_group = {int(e): int(g) for e, g in zip(lidx, gl)}
            o2, p2 = joint_maxt(cells, lidx, ep_label, ep_group, NPERM,
                                rng_for(tag, "family2"))
            for k in FAM2:
                a, den = o2[k]
                u["family2"][k] = dict(
                    auc_fatal_high=round(float(a), 4) if np.isfinite(a) else None,
                    n_pairs=int(den),
                    p_maxT=round(p2[k], 4) if k in p2 else None)
            u["family2_audit"] = dict(n_fatal=int(len(pops["fatal_loop"])),
                                      n_recovering=int(len(pops["recovering_loop"])),
                                      n_groups_with_both=both)
        else:
            u["family2_audit"] = dict(n_fatal=int(len(pops["fatal_loop"])),
                                      n_recovering=int(len(pops["recovering_loop"])),
                                      n_groups_with_both=0, note="人群不足，跳过")
        # 描述性：两 loop 人群 vs 干净成功（不进族，不给 p）
        u["desc_auc_vs_clean"] = {}
        for k in ("pulse_count", "train_flag", "post_pulse_return"):
            arr = dict(pulse_count=T["n_pulses"].astype(float),
                       train_flag=T["train_flag"].astype(float),
                       post_pulse_return=T["post_ret"])[k]
            row = {}
            for lab in ("fatal_loop", "recovering_loop"):
                sel = np.concatenate([pops[lab], pops["clean_success"]])
                if not len(pops[lab]) or not len(pops["clean_success"]):
                    row[lab] = None
                    continue
                yy = np.array([1] * len(pops[lab]) + [0] * len(pops["clean_success"]))
                a, den = paired_auc(arr[sel], yy, T["gid"][sel])
                row[lab] = dict(auc=round(float(a), 4) if np.isfinite(a) else None,
                                n_pairs=int(den))
            u["desc_auc_vs_clean"][k] = row

        # ---- (c) 交叉表（排除红旗）+ lock-in 失败清单 ----
        ct = {}
        for i in np.flatnonzero(um & ~T["redflag"]):
            if not T["covered"][i] or not np.isfinite(ep_lock_all[i]):
                quad = "uncovered_or_nothr"
            elif ep_lock_all[i] == 1:
                quad = "lockin"
            elif ep_swi_all[i] == 1:
                quad = "switching"
            else:
                quad = "other"
            evc = ("loop_and_static" if T["o_loop"][i] >= 0 and T["o_static"][i] >= 0
                   else "loop_only" if T["o_loop"][i] >= 0
                   else "static_only" if T["o_static"][i] >= 0 else "no_event")
            out = "fail" if T["succ"][i] == 0 else "succ"
            ct[(out, evc, quad)] = ct.get((out, evc, quad), 0) + 1
        for (out, evc, quad), c in sorted(ct.items()):
            crosstab_rows.append(dict(unit=tag, outcome=out, event_class=evc,
                                      quadrant=quad, n=c))
        for i in np.flatnonzero(um & (T["succ"] == 0)):
            if ep_lock_all[i] == 1:
                lockin_fail_list.append(dict(
                    corpus=T["corpus"][i], suite=T["suite"][i], task=T["task"][i],
                    scene=int(T["scene"][i]), repeat=int(T["repeat"][i]),
                    episode_id=int(T["epid"][i]), n_queries=int(T["nq"][i]),
                    loop_onset=int(T["o_loop"][i]), static_onset=int(T["o_static"][i]),
                    redflag_loop_channel=bool(T["redflag"][i]),
                    n_pulses=int(T["n_pulses"][i]),
                    winV=round(float(T["wV"][i]), 5), winA=round(float(T["wA"][i]), 5),
                    winMargin=round(float(T["wM"][i]), 5),
                    winMob1w8=round(float(T["wW"][i]), 5)))
        summary["units"][tag] = u
        print(f"[{tag}] done ({time.time()-tu:.0f}s)", flush=True)

    # ---- family2 补充腿：grid 全套件合并（组不变）----
    tag = "grid50x8/ALL_pooled(supplementary)"
    pm = (T["corpus"] == "grid50x8") & ~T["redflag"]
    fi = np.flatnonzero(pm & (T["pop"] == "fatal_loop"))
    ri = np.flatnonzero(pm & (T["pop"] == "recovering_loop"))
    lidx = np.concatenate([fi, ri])
    yl = np.array([1] * len(fi) + [0] * len(ri))
    gl = T["gid"][lidx]
    both = sum(1 for g in np.unique(gl)
               if (yl[gl == g] == 1).any() and (yl[gl == g] == 0).any())
    vals = dict(cycle_gain=T["cyc"][lidx], pulse_count=T["n_pulses"][lidx].astype(float),
                train_flag=T["train_flag"][lidx].astype(float),
                isolated_pulse_flag=T["iso_flag"][lidx].astype(float),
                post_pulse_return=T["post_ret"][lidx])
    cells = [(k, v, np.isfinite(v), gl) for k, v in vals.items()]
    o2, p2 = joint_maxt(cells, lidx,
                        {int(e): int(v) for e, v in zip(lidx, yl)},
                        {int(e): int(g) for e, g in zip(lidx, gl)},
                        NPERM, rng_for(tag, "family2"))
    summary["units"][tag] = dict(
        family2={k: dict(auc_fatal_high=round(float(o2[k][0]), 4) if np.isfinite(o2[k][0]) else None,
                         n_pairs=int(o2[k][1]),
                         p_maxT=round(p2[k], 4) if k in p2 else None) for k in FAM2},
        family2_audit=dict(n_fatal=int(len(fi)), n_recovering=int(len(ri)),
                           n_groups_with_both=both,
                           note="补充腿：4 套件 loop 集合并做 joint maxT，组仍 (task,scene)"))
    print(f"[{tag}] done", flush=True)

    # ---- onset 对齐曲线（图数据）----
    rngc = rng_for("curves")
    curves = {}
    for corpus in ("grid50x8", "main16x32"):
        cm = (T["corpus"] == corpus) & ~T["redflag"]
        cur = {}
        for popk in ("recovering_loop", "fatal_loop"):
            pidx = np.flatnonzero(cm & (T["pop"] == popk))
            Mz = np.full((len(pidx), len(LEADS)), np.nan)
            for r, i in enumerate(pidx):
                o, z = T["o_loop"][i], zlist[i]
                for c, L in enumerate(LEADS):
                    if 0 <= o + L < len(z):
                        Mz[r, c] = z[o + L]
            cur[popk] = Mz
        clean_by_g = {}
        for i in np.flatnonzero(cm & (T["pop"] == "clean_success")):
            clean_by_g.setdefault(int(T["gid"][i]), []).append(i)
        rows = []
        for i in np.flatnonzero(cm & ((T["pop"] == "recovering_loop")
                                      | (T["pop"] == "fatal_loop"))):
            o = T["o_loop"][i]
            cand = [j for j in clean_by_g.get(int(T["gid"][i]), []) if T["nq"][j] > o]
            if not cand:
                continue
            j = cand[rngc.integers(len(cand))]
            z = zlist[j]
            row = np.full(len(LEADS), np.nan)
            for c, L in enumerate(LEADS):
                if 0 <= o + L < len(z):
                    row[c] = z[o + L]
            rows.append(row)
        cur["clean_success_matched"] = (np.array(rows) if rows
                                        else np.zeros((0, len(LEADS))))
        curves[corpus] = cur
        summary.setdefault("curve_n", {})[corpus] = {
            k: int(len(v)) for k, v in cur.items()}
    np.savez_compressed(
        os.path.join(OUT, "curves_onset_aligned.npz"), leads=LEADS,
        **{f"{c}__{k}": v for c, cur in curves.items() for k, v in cur.items()})

    # ---- lock-in 失败清单与存在性 ----
    n_lockin_fail = len(lockin_fail_list)
    n_lockin_fail_noev = sum(1 for r in lockin_fail_list
                             if r["loop_onset"] < 0 and r["static_onset"] < 0
                             and not r["redflag_loop_channel"])
    summary["lockin_failures"] = dict(
        n_total=n_lockin_fail, n_no_event=n_lockin_fail_noev,
        note="no_event 计数排除红旗任务（loop 通道无效无法判无事件）",
        list=lockin_fail_list)

    # ---- CSV：逐集 pulse_stats ----
    cols = ["corpus", "suite", "task", "scene", "repeat", "episode_id", "success",
            "n_queries", "loop_onset_q", "static_onset_q", "trap_onset_q", "redflag",
            "population", "covered", "n_win_rows", "cycle_gain", "best_k",
            "winV", "winA", "winMargin", "winMob1w8", "n_pulses", "n_isolated_pulses",
            "isolated_pulse_flag", "train_flag", "post_pulse_return",
            "has_pulse_near_onset", "nearest_pulse_rel_onset", "first_pulse_q",
            "lockin_flag", "switch_flag"]
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and not np.isfinite(v))
                     else (round(v, 6) if isinstance(v, float) else v))
    with open(os.path.join(OUT, "pulse_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([fmt(v) for v in (
                T["corpus"][i], T["suite"][i], T["task"][i], int(T["scene"][i]),
                int(T["repeat"][i]), int(T["epid"][i]), int(T["succ"][i]),
                int(T["nq"][i]), int(T["o_loop"][i]), int(T["o_static"][i]),
                int(T["o_trap"][i]), int(T["redflag"][i]), T["pop"][i],
                int(T["covered"][i]), int(T["n_win"][i]), float(T["cyc"][i]),
                int(T["bestk"][i]), float(T["wV"][i]), float(T["wA"][i]),
                float(T["wM"][i]), float(T["wW"][i]), int(T["n_pulses"][i]),
                int(T["n_iso"][i]), int(T["iso_flag"][i]), int(T["train_flag"][i]),
                float(T["post_ret"][i]), int(T["near_onset"][i]),
                float(T["nearest_pulse_rel_onset"][i]), int(T["first_pulse_q"][i]),
                float(ep_lock_all[i]), float(ep_swi_all[i]))])

    with open(os.path.join(OUT, "lockin_crosstab.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unit", "outcome", "event_class",
                                          "quadrant", "n"])
        w.writeheader()
        w.writerows(crosstab_rows)

    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
