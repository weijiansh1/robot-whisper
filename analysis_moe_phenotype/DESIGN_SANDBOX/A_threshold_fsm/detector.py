"""方案 A：标量阈值状态机（two-channel online MoE-routing trap detector）。

只读 MoE routing。两个通道各是一个在线状态机，跑在**组内自校准的 robust z** 上：

    z(x) = (x - median_组(x)) / (1.4826 * MAD_组(x))        # 组 = scene / init_state
                                                            # 定位/尺度**无标签**：用组内全部行

- loop 通道（"能否离开"）：z(V+A) > θ_enter 进入切换带；进入后 K 个有效步内
  从未回落到 z < θ_exit  →  报警(loop)。
- static 通道（"早已建立且持续"）：z(min_{k>=2} mob_k) < θ_s  且  z(C) > θ_c，
  连续 M 个有效步  →  报警(static)。
- 消融臂 C：同结构，一级信号换成 mob1_w8（方向翻转：失败时 mobility 降低，
  所以 loop 通道的进入条件是 z(mob1_w8) < θ_enter^C）。

**标签只进入阈值，不进入归一化**（CONSTRAINTS §1："阈值只能从成功集分布、留一组外
校准"）：θ 是"其它组的成功行"的 z 分位。组内定位/尺度用全部行（E4/E8 的组内无标签
robust z 口径）。因此：

  P1（组内 success 随机置换 -> 组内报警逐位不变）**按构造成立**：
      组内 z 不看标签；该组的 θ 只由**其它组**的成功行决定。
  P2（截断到 s 仍报 s、截断到 s-1 不报 s）**按构造成立**：状态机是纯前向扫描。

参数一次冻死，见 PARAMS / SPEC.md。

接口：
    detect(rows_npz_path, group_col="scene") -> {episode_id: record}
    record["alarms"] = [{"step": int, "channel": "loop"|"static"}, ...]   # 完整序列

运行前：export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, replace

import numpy as np

MAD_C = 1.4826

# ============================================================================
# 冻结参数表（语义与选定依据见 SPEC.md §3；此处只放值）
# ============================================================================


@dataclass(frozen=True)
class Params:
    # --- 6 个旋钮 ---
    q_enter: float = 90.0   # θ_enter = 成功集 z(V+A) 的 p90
    q_exit: float = 50.0    # θ_exit  = 成功集 z(V+A) 的 p50
    K: int = 6              # 进入后允许的最长"未回落"步数
    q_s: float = 10.0       # θ_s = 成功集 z(min_{k>=2}mob_k) 的 p10
    q_c: float = 90.0       # θ_c = 成功集 z(token_consensus) 的 p90
    M: int = 3              # static 双条件需连续满足的步数

    # --- 结构常量（非旋钮，随信号定义域/卫生而定；两臂共用以保证同口径） ---
    q_min: int = 8          # 预热：q<8 时 mob_k(k>=2) / mob1_w8 无定义
    min_scale_rows: int = 60   # 组内定位/尺度所需最少有限行数（否则退到任务级）
    min_thr_rows: int = 30     # LOGO 分位所需最少成功行数（否则该组弃权）
    arm: str = "A"          # "A" = 表型臂；"C" = mob1_w8 消融臂


PARAMS = Params()                                   # 臂 A（冻结）
PARAMS_C = replace(PARAMS, arm="C", K=12, M=2)      # 臂 C：同规则重定持续腿
PARAMS_C_SAMEKNOBS = replace(PARAMS, arm="C")       # 臂 C'：旋钮值与 A 完全相同
PARAMS_T = replace(PARAMS, arm="T", K=5, M=6)       # 臂 T：同规则重定持续腿
PARAMS_T_SAMEKNOBS = replace(PARAMS, arm="T")       # 臂 T'：旋钮值与 A 完全相同

# 执行器用：把各臂对齐到同一工作点的扫描表（只动"进入分位"，其余冻结）。
# 每个元素 = (label, Params)。执行器可在同一次运行里为各臂各挑一个 FA 最接近的点。
_ENTER_GRID = (60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 93.0, 95.0, 97.0)
_QC_GRID = (50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
_QS_GRID = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
PARAM_SWEEP = {
    "A": [(f"A/q_enter={v:g}", replace(PARAMS, q_enter=v)) for v in _ENTER_GRID],
    "C": [(f"C/q_enter={v:g}", replace(PARAMS_C, q_enter=v)) for v in _ENTER_GRID],
    "T": [(f"T/q_enter={v:g}", replace(PARAMS_T, q_enter=v)) for v in _ENTER_GRID],
    # static 通道：臂 A/C 的工作点由 θ_c 主导；臂 T 没有二级腿，只能扫 θ_s
    "A_static": [(f"A/q_c={v:g}", replace(PARAMS, q_c=v)) for v in _QC_GRID],
    "C_static": [(f"C/q_c={v:g}", replace(PARAMS_C, q_c=v)) for v in _QC_GRID],
    "T_static": [(f"T/q_s={v:g}", replace(PARAMS_T, q_s=v)) for v in _QS_GRID],
}


# ============================================================================
# 输入契约 / 载入
# ============================================================================

REQUIRED_COLS = (
    "episode_id", "control_step", "success",          # 结构（success 只用于阈值分位）
    "late_flow_volatility", "route_acceleration",     # V, A  -> 臂 A loop 一级信号
    "mob_k",                                          # min_{k>=2}mob_k -> 臂 A static 一级
    "token_consensus",                                # C -> 两臂共用的 static 二级信号
    "mob1_w8",                                        # 臂 C 一级信号
)


@dataclass
class Episodes:
    """一个任务的逐集视图（行按 episode 连续分块，q = 集内 0..n-1）。"""
    epid: np.ndarray            # (E,)
    group: np.ndarray           # (E,)
    success: np.ndarray         # (E,)
    n_q: np.ndarray             # (E,)
    slices: list                # (E,) 行 slice
    q: np.ndarray               # (R,) 集内 query 序号
    sig: dict                   # name -> (R,) 原始信号
    path: str = ""
    extra: dict = field(default_factory=dict)


def load_rows(rows_npz_path, group_col="scene"):
    d = np.load(rows_npz_path)
    missing = [c for c in REQUIRED_COLS if c not in d]
    if missing:
        raise KeyError(f"rows.npz 缺列: {missing}  ({rows_npz_path})")
    if group_col not in d:
        raise KeyError(f"rows.npz 缺 group_col={group_col}")

    ep = d["episode_id"].astype(np.int64)
    cs = d["control_step"].astype(np.int64)
    assert np.all(np.diff(ep) >= 0), "rows 未按 episode 分块"
    uniq, first, cnt = np.unique(ep, return_index=True, return_counts=True)
    start = np.zeros(int(uniq.max()) + 1, np.int64)
    start[uniq] = cs[first]
    q = cs - start[ep]
    # control_step 是任务级累计计数：断言归零后逐集连续 0..n-1
    assert np.array_equal(q, np.concatenate([np.arange(c) for c in cnt])), \
        "control_step 集内不连续"

    mk = d["mob_k"].astype(np.float64)
    sig = {
        "VA": (d["late_flow_volatility"].astype(np.float64)
               + d["route_acceleration"].astype(np.float64)),
        "minmobk": np.min(mk[:, 1:], axis=1),      # min_{k=2..8} mob_k
        "C": d["token_consensus"].astype(np.float64),
        "w8": d["mob1_w8"].astype(np.float64),
        # --- 臂 T（平凡集长臂）专用：完全不读 MoE，只用集内 query 序号 ---
        "q": q.astype(np.float64),
        "negq": -q.astype(np.float64),
    }
    slices = [slice(int(f), int(f + c)) for f, c in zip(first, cnt)]
    return Episodes(
        epid=uniq.astype(np.int64),
        group=d[group_col][first].astype(np.int64),
        success=d["success"][first].astype(np.int64),
        n_q=cnt.astype(np.int64),
        slices=slices,
        q=q,
        sig=sig,
        path=rows_npz_path,
    )


def load_events(events_csv_path):
    """events.csv -> {episode_id: dict}。**检测器本体从不调用它**；
    只供离线评估/报表使用（执行器独占事件表）。"""
    out = {}
    with open(events_csv_path) as fh:
        for r in csv.DictReader(fh):
            out[int(r["episode_id"])] = {
                "success": int(r["success"]),
                "n_queries": int(r["n_queries"]),
                "loop_onset_q": int(r["loop_onset_q"]),
                "static_onset_q": int(r["static_onset_q"]),
                "trap_onset_q": int(r["trap_onset_q"]),
                "proxy_grade": r["proxy_grade"],
            }
    return out


# ============================================================================
# 组内自校准 z（**无标签**：定位/尺度用组内全部行）
# ============================================================================

def row_group(E: Episodes):
    R = len(E.q)
    eor = np.empty(R, np.int64)
    for i, s in enumerate(E.slices):
        eor[s] = i
    return eor, E.group[eor]


def compute_z(E: Episodes, p: Params = PARAMS):
    """返回 (Z, scale_level)。scale_level[i] ∈ {group, task, none}。

    定位/尺度 = 组内**全部行**（不看 success）的 median / 1.4826·MAD。
    行数不足 min_scale_rows 或 MAD≈0 -> 退到任务级全部行；仍不行 -> NaN。
    """
    R = len(E.q)
    Z = {k: np.full(R, np.nan) for k in E.sig}
    eor, grp_row = row_group(E)
    level = np.array(["group"] * len(E.epid), dtype=object)

    for name, x in E.sig.items():
        fin = np.isfinite(x)
        tv = x[fin]
        t_med = float(np.median(tv)) if len(tv) else np.nan
        t_mad = float(np.median(np.abs(tv - t_med))) * MAD_C if len(tv) else np.nan
        for g in np.unique(E.group):
            m = (grp_row == g) & fin
            if int(m.sum()) >= p.min_scale_rows:
                med = float(np.median(x[m]))
                mad = float(np.median(np.abs(x[m] - med))) * MAD_C
                lv = "group"
            else:
                med, mad, lv = t_med, t_mad, "task"
            if not np.isfinite(mad) or mad <= 1e-12:
                med, mad, lv = t_med, t_mad, "task"
            if not np.isfinite(mad) or mad <= 1e-12:
                lv = "none"
                continue
            gm = grp_row == g
            Z[name][gm] = (x[gm] - med) / mad
            if name == "VA":
                level[E.group == g] = lv
    return Z, level


# ============================================================================
# 阈值：**成功集**分位，留一组外（LOGO）——标签只在这里出现
# ============================================================================

def calibrate_thresholds(E: Episodes, Z, p: Params = PARAMS, logo=True):
    """返回 {group: {knob: value}}。分位在 success==1 且 q>=q_min 的行上算；
    logo=True（默认，且是唯一合规选项）时排除被评估组自身的行。

    ==> 组 g 的阈值只依赖**其它组**的标签 ==> P1 成立。
    """
    R = len(E.q)
    eor, grp_row = row_group(E)
    succ_row = (E.success[eor] == 1) & (E.q >= p.q_min)

    # 臂 C 的分位是臂 A 的**低侧镜像**（失败时 mobility 降低 -> 尾在低侧）
    # 臂 T：一级信号 = 集内 query 序号 q（完全不读 MoE）。
    #   loop 腿与臂 A 同侧（"集跑得久" = z(q) 高）；static 腿用 negq 走低侧，
    #   与臂 A/C 的 θ_s 完全同分位规则（"低 mobility" 的集长类比 = "集已经很长"）。
    spec = [("enter_A", "VA", p.q_enter), ("exit_A", "VA", p.q_exit),
            ("s_A", "minmobk", p.q_s), ("c", "C", p.q_c),
            ("enter_C", "w8", 100.0 - p.q_enter),
            ("exit_C", "w8", 100.0 - p.q_exit),
            ("s_C", "w8", p.q_s),
            ("enter_T", "q", p.q_enter), ("exit_T", "q", p.q_exit),
            ("s_T", "negq", p.q_s)]

    out = {}
    for g in np.unique(E.group):
        m = succ_row & ((grp_row != g) if logo else np.ones(R, bool))
        d = {}
        for knob, name, qq in spec:
            v = Z[name][m]
            v = v[np.isfinite(v)]
            d[knob] = float(np.percentile(v, qq)) if len(v) >= p.min_thr_rows \
                else np.nan
        out[int(g)] = d
    return out


# ============================================================================
# 状态机（在线，因果，输出完整报警序列）
# ============================================================================

def fsm_loop(z, theta_enter, theta_exit, K, q_min=0, side="hi"):
    """进入切换带后 K 个有效步内未回落 -> 报警。**每次进出带最多一次报警**，
    回落后重新武装。返回 (alarm_steps, n_entries)。

    side="hi"：进入 = z > θ_enter，回落 = z < θ_exit（臂 A，V+A 升高）
    side="lo"：进入 = z < θ_enter，回落 = z > θ_exit（臂 C，mobility 降低）
    """
    alarms = []
    if not (np.isfinite(theta_enter) and np.isfinite(theta_exit)):
        return alarms, 0
    inband, clock, armed, n_entries = False, 0, True, 0
    for t in range(q_min, len(z)):
        zt = z[t]
        if not np.isfinite(zt):          # 无效行：不进入、不回落、不推进时钟
            continue
        enter_hit = (zt > theta_enter) if side == "hi" else (zt < theta_enter)
        exit_hit = (zt < theta_exit) if side == "hi" else (zt > theta_exit)
        if not inband:
            if enter_hit:
                inband, clock, armed = True, 0, True
                n_entries += 1
        else:
            if exit_hit:
                inband, clock, armed = False, 0, True
                continue
            clock += 1
            if armed and clock >= K:
                alarms.append(t)
                armed = False           # 本次驻留只报一次，回落后才重新武装
    return alarms, n_entries


def fsm_static(z_s, z_c, theta_s, theta_c, M, q_min=0):
    """两条件同时成立连续 M 个有效步 -> 报警。**每个满足段最多一次报警**。
    返回 (alarm_steps, max_run)。"""
    alarms = []
    if not (np.isfinite(theta_s) and np.isfinite(theta_c)):
        return alarms, 0
    run, best, armed = 0, 0, True
    for t in range(q_min, len(z_s)):
        a, b = z_s[t], z_c[t]
        if not (np.isfinite(a) and np.isfinite(b)):   # 无效行：跳过，不重置
            continue
        if (a < theta_s) and (b > theta_c):
            run += 1
            best = max(best, run)
            if armed and run >= M:
                alarms.append(t)
                armed = False
        else:
            run, armed = 0, True
    return alarms, best


# ============================================================================
# 主接口
# ============================================================================

_ARM_WIRING = {
    "A": dict(loop_sig="VA", loop_side="hi", k_enter="enter_A", k_exit="exit_A",
              stat_sig="minmobk", k_s="s_A", stat_leg2="C"),
    "C": dict(loop_sig="w8", loop_side="lo", k_enter="enter_C", k_exit="exit_C",
              stat_sig="w8", k_s="s_C", stat_leg2="C"),
    # 臂 T 读不到任何 MoE，static 通道的二级腿（token_consensus）无从替代
    # -> 只剩一条腿（恒真哨兵）。单腿比合取**更容易**响，不构成对 T 的削弱。
    "T": dict(loop_sig="q", loop_side="hi", k_enter="enter_T", k_exit="exit_T",
              stat_sig="negq", k_s="s_T", stat_leg2=None),
}


def detect(rows_npz_path, group_col="scene", params: Params = PARAMS,
           E=None, Z=None, scale_level=None, thresholds=None):
    """在线检测器。**不接受 events 表**（结局标签与物理 onset 由执行器独占）。

    返回 {episode_id: {
        "group", "n_q", "scale_level",
        "alarms": [{"step": int, "channel": "loop"|"static"}, ...],   # 时间升序
        "first_loop_alarm_q", "first_static_alarm_q",
        "first_alarm_q", "first_alarm_channel",
        "loop_n_entries", "static_max_run",
        "theta_enter", "theta_exit", "theta_s", "theta_c",
        "abstain": bool                     # 阈值不可定义 -> 弃权，不得计入分母
    }}
    """
    p = params
    if E is None:
        E = load_rows(rows_npz_path, group_col)
    if Z is None:
        Z, scale_level = compute_z(E, p)
    if thresholds is None:
        thresholds = calibrate_thresholds(E, Z, p, logo=True)
    w = _ARM_WIRING[p.arm]

    out = {}
    for i, s in enumerate(E.slices):
        g = int(E.group[i])
        th = thresholds.get(g, {})
        te, tx = th.get(w["k_enter"], np.nan), th.get(w["k_exit"], np.nan)
        ts, tc = th.get(w["k_s"], np.nan), th.get("c", np.nan)
        la, nent = fsm_loop(Z[w["loop_sig"]][s], te, tx, p.K,
                            q_min=p.q_min, side=w["loop_side"])
        if w["stat_leg2"] is None:                 # 臂 T：无二级腿 -> 恒真哨兵
            z2, tc = np.zeros(int(E.n_q[i])), -1e9
        else:
            z2 = Z[w["stat_leg2"]][s]
        sa, srun = fsm_static(Z[w["stat_sig"]][s], z2, ts, tc, p.M,
                              q_min=p.q_min)
        alarms = sorted([{"step": int(t), "channel": "loop"} for t in la]
                        + [{"step": int(t), "channel": "static"} for t in sa],
                        key=lambda a: (a["step"], a["channel"] != "loop"))
        lvl = str(scale_level[i]) if scale_level is not None else "?"
        out[int(E.epid[i])] = dict(
            group=g, n_q=int(E.n_q[i]), scale_level=lvl,
            alarms=alarms,
            first_loop_alarm_q=int(la[0]) if la else -1,
            first_static_alarm_q=int(sa[0]) if sa else -1,
            first_alarm_q=int(alarms[0]["step"]) if alarms else -1,
            first_alarm_channel=alarms[0]["channel"] if alarms else "none",
            loop_n_entries=int(nent), static_max_run=int(srun),
            theta_enter=te, theta_exit=tx, theta_s=ts, theta_c=tc,
            abstain=bool(not np.isfinite(te) or not np.isfinite(tx)
                         or not np.isfinite(ts) or not np.isfinite(tc)
                         or lvl == "none"),
        )
    return out


# ============================================================================
# 机械自检探针
# ============================================================================

def probe_p1(rows_npz_path, group_col="scene", params: Params = PARAMS,
             n_perm=20, seed=20260903, E=None):
    """P1：组内随机置换 `success` 后，**该组内**的报警必须逐位不变。

    另报 P1b（同时置换所有组）作为诚实披露：此时其它组的 LOGO 分位会动，
    组 g 的阈值随之改变——这是 CONSTRAINTS §1 强制的成功集校准本身带来的，
    不是泄漏。
    """
    if E is None:
        E = load_rows(rows_npz_path, group_col)
    base_succ = E.success.copy()
    Z, lvl = compute_z(E, params)
    ref = detect("", group_col, params, E=E, Z=Z, scale_level=lvl)
    rng = np.random.default_rng(seed)
    groups = np.unique(E.group)
    n_bad_a = n_bad_b = 0
    checked_a = checked_b = 0

    for _ in range(n_perm):
        # --- P1a：只置换一个组，只查该组 ---
        g = int(rng.choice(groups))
        E.success[:] = base_succ
        idx = np.flatnonzero(E.group == g)
        E.success[idx] = rng.permutation(E.success[idx])
        Z2, lvl2 = compute_z(E, params)          # 无标签 -> 应完全相同
        r2 = detect("", group_col, params, E=E, Z=Z2, scale_level=lvl2)
        for i, e in enumerate(E.epid):
            if E.group[i] != g:
                continue
            checked_a += 1
            if r2[int(e)]["alarms"] != ref[int(e)]["alarms"]:
                n_bad_a += 1
        # --- P1b：所有组同时置换，查全体 ---
        E.success[:] = base_succ
        for gg in groups:
            ii = np.flatnonzero(E.group == gg)
            E.success[ii] = rng.permutation(E.success[ii])
        Z3, lvl3 = compute_z(E, params)
        r3 = detect("", group_col, params, E=E, Z=Z3, scale_level=lvl3)
        for e in E.epid:
            checked_b += 1
            if r3[int(e)]["alarms"] != ref[int(e)]["alarms"]:
                n_bad_b += 1

    E.success[:] = base_succ
    return dict(n_perm=int(n_perm),
                P1a_within_group_episodes_checked=int(checked_a),
                P1a_mismatches=int(n_bad_a),
                P1b_all_groups_episodes_checked=int(checked_b),
                P1b_mismatches=int(n_bad_b),
                P1b_mismatch_rate=float(n_bad_b / max(1, checked_b)))


def probe_p2(E, Z, params: Params = PARAMS, thresholds=None, scale_level=None):
    """P2：对每个报警步 s —— 截断到 s 仍报 s；截断到 s-1 完全不报 s。
    对每集另抽一个非报警步做对称检查（截断到该步不应凭空出现报警）。"""
    p = params
    w = _ARM_WIRING[p.arm]
    if thresholds is None:
        thresholds = calibrate_thresholds(E, Z, p, logo=True)
    n_alarm = n_bad_keep = n_bad_drop = 0
    for i, s in enumerate(E.slices):
        th = thresholds.get(int(E.group[i]), {})
        te, tx = th.get(w["k_enter"], np.nan), th.get(w["k_exit"], np.nan)
        ts, tc = th.get(w["k_s"], np.nan), th.get("c", np.nan)
        zl, zs = Z[w["loop_sig"]][s], Z[w["stat_sig"]][s]
        if w["stat_leg2"] is None:
            zc, tc = np.zeros(len(zs)), -1e9
        else:
            zc = Z[w["stat_leg2"]][s]
        full = ([("loop", t) for t in fsm_loop(zl, te, tx, p.K, p.q_min,
                                               w["loop_side"])[0]]
                + [("static", t) for t in fsm_static(zs, zc, ts, tc, p.M,
                                                     p.q_min)[0]])
        for ch, t0 in full:
            n_alarm += 1
            for cut, want in ((t0 + 1, True), (t0, False)):   # 截断到 s / s-1
                kl = fsm_loop(zl[:cut], te, tx, p.K, p.q_min, w["loop_side"])[0]
                ks = fsm_static(zs[:cut], zc[:cut], ts, tc, p.M, p.q_min)[0]
                got = (t0 in kl) if ch == "loop" else (t0 in ks)
                if got != want:
                    if want:
                        n_bad_keep += 1
                    else:
                        n_bad_drop += 1
    assert n_bad_keep == 0 and n_bad_drop == 0, \
        f"P2 失败：截断保留 {n_bad_keep} 处、截断消除 {n_bad_drop} 处"
    return dict(n_alarms_checked=int(n_alarm), n_violations=0)


def assert_causality(E, Z, p: Params = PARAMS, n_check=None, rng=None,
                     thresholds=None):
    """破坏性因果检验：把 t0 之后的行换成 NaN / ±1e6 重跑，
    要求 <=t0 的报警序列逐位不变。"""
    rng = np.random.default_rng(20260903) if rng is None else rng
    w = _ARM_WIRING[p.arm]
    if thresholds is None:
        thresholds = calibrate_thresholds(E, Z, p, logo=True)
    idx = np.arange(len(E.epid))
    if n_check is not None and n_check < len(idx):
        idx = rng.choice(idx, n_check, replace=False)
    n_bad = 0
    for i in idx:
        s = E.slices[i]
        th = thresholds.get(int(E.group[i]), {})
        te, tx = th.get(w["k_enter"], np.nan), th.get(w["k_exit"], np.nan)
        ts, tc = th.get(w["k_s"], np.nan), th.get("c", np.nan)
        zl0, zs0 = Z[w["loop_sig"]][s], Z[w["stat_sig"]][s]
        if w["stat_leg2"] is None:
            zc0, tc = np.zeros(len(zs0)), -1e9
        else:
            zc0 = Z[w["stat_leg2"]][s]
        f_l = fsm_loop(zl0, te, tx, p.K, p.q_min, w["loop_side"])[0]
        f_s = fsm_static(zs0, zc0, ts, tc, p.M, p.q_min)[0]
        n = len(zl0)
        for t0 in rng.choice(np.arange(n), size=min(6, n), replace=False):
            t0 = int(t0)
            for fill in (np.nan, 1e6, -1e6):
                zl, zs, zc = zl0.copy(), zs0.copy(), zc0.copy()
                zl[t0 + 1:] = fill
                zs[t0 + 1:] = fill
                zc[t0 + 1:] = fill
                gl = fsm_loop(zl, te, tx, p.K, p.q_min, w["loop_side"])[0]
                gs = fsm_static(zs, zc, ts, tc, p.M, p.q_min)[0]
                if ([a for a in gl if a <= t0] != [a for a in f_l if a <= t0]
                        or [a for a in gs if a <= t0]
                        != [a for a in f_s if a <= t0]):
                    n_bad += 1
    assert n_bad == 0, f"因果性自检失败：{n_bad} 处 t<=t0 的判定被未来行改变"
    return dict(n_episodes_checked=int(len(idx)), n_violations=0)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/jovyan/work/himoe-vla/analysis_moe_phenotype/features/"
        "main16x32/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/rows.npz")
    res = detect(path)
    n_al = sum(1 for r in res.values() if r["alarms"])
    n_ev = sum(len(r["alarms"]) for r in res.values())
    print(f"{os.path.basename(os.path.dirname(path))}: {len(res)} episodes, "
          f"{n_al} alarmed, {n_ev} alarm events, "
          f"{sum(r['abstain'] for r in res.values())} abstained")
