"""
D_freeform -- 极性匹配滤波检测器 (Polarity Matched Filter, PMF)
===============================================================

方案 D：不是"单信号越阈"，也不是"宏状态驻留计数"，而是
**把 E2/E3 给出的 lead 剖面当成模板，在线做（信号 x 时间）二维匹配滤波**，
类型由模板极性的符号直接给出。

三层结构（全部 train-free：只有中位数 / MAD / 分位点 / 计数）：

  L1  健康参照标准化（只用**成功集**，留一组外 LOGO）
        z_i(t) = (x_i(t) - mu_i^(g)(q)) / sigma_i^(g)(q) - delta_i(episode)
      mu/sigma : 参照池 R_g = **除本组以外**所有 success 集，在同一绝对 query q (+-B) 上的中位/MAD
      delta    : 本组**无标签**（不看 success）早窗中位偏移，留一集外(LOEO) -> 组心化

  L2  信号空间的模板内积（匹配滤波的"空间"部分）
        g_c(t) = mean_{i in T_c} ( s_i^c * axis_i(t) )
      T_c = 冻结的**符号**模板（+1/-1，权重一律 1；不用效应量当权重，那会把事件标签写进参数）
      极性 pol(t) = 共享轴在 loop 方向上的内积。
      极性门：loop 通道要求 pol>0，static 通道要求 pol<0。门的原点 0 = 参照集中位，不是旋钮。

  L3  时间形状匹配（匹配滤波的"时间"部分，因果窗）
        G_loop(t)   = mean(g_loop[t-3..t])     W=4  = E2 显著 lead 跨度 (-4..-1)
        G_static(t) = mean(g_static[t-4..t])   W=5  = E3 因果侧显著 lead 跨度 (-4..0)

  报警：G_c(t) >= theta_c^(g)；theta_c^(g) = R_g 中 success 集"每集最大分"的 (1-alpha/2) 分位。
  类型：越阈的通道。返回**完整报警序列**（每个越阈步一条），不只首次。
  次级输出：persist = 首次报警后 exit_delta 拍时该通道是否仍在阈上（"能否离开"，不影响报警时刻）。

标签使用（对照 CONSTRAINTS）：
  - 只读 rows.npz 的 `success` 列，且**只用于构造参照池 R_g（永远排除本组）**。
  - 评估某组的集时，本组的 success 标签完全不参与（探针 P1）。
  - t 时刻的判定只用本集 <=t 的行（探针 P2）；组心化对本集留一。
  - 从不读 events.csv，不知道 onset，也不知道被评集的结局。

输入契约（rows.npz 必需列）：
  episode_id, control_step, success, <group_col>,
  late_flow_volatility, route_acceleration, gate_entropy, top12_margin,
  token_consensus, token_dispersion, mob_k (N,8), mob1_w8
（layer_disagreement / flow_com / state_action_gap / flow_profile 不使用。）

接口： detect(rows_npz_path, group_col) -> {episode_id: record}
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# 冻结参数表（语义与选定依据见 SPEC.md §4）
# ----------------------------------------------------------------------------
CONFIG = dict(
    alpha=0.10,        # 每集**总**误报预算（在参照成功集上）；两通道按 Bonferroni 各分 alpha/2
    B_ref=2,           # 参照 q 窗半宽(+-2)
    n_ref_rows=60,     # 一个 q 可评所需的最少参照行数
    n_ref_eps=25,      # 一个 q 可评所需的最少存活参照集数（决定 q_hi）
    q_lo=8,            # 首个全信号有定义的 query（mob_k k=8 与 mob1_w8 都要 8 步拖尾）
    q_cen_hi=30,       # 组心化 delta 的估计窗上界
    center=True,       # 组心化开关
    n_cen_rows=40,     # 组心化所需最少行数；不足则 delta=0
    W_static=5,        # static 平台滤波窗长
    W_loop_sig=4,      # loop 平台滤波窗长
    W_loop_base=6,     # (仅 filt_loop="CS" 用) 环绕基线窗
    exit_delta=6,      # "能否离开"观察步数
    polarity_gate=True,
    filt_loop="DC",    # "DC"=平台匹配滤波 / "CS"=零均值中心-环绕
    filt_static="DC",
    agg="mean",        # 信号空间聚合："mean"=模板内积（冻结）/ "median"=巧合门（变体）
)

# 执行器对齐工作点用：唯一的工作点旋钮是 alpha
PARAM_SWEEP = [dict(alpha=a) for a in (0.02, 0.05, 0.10, 0.20, 0.35)]

SIG_COLS = ["late_flow_volatility", "route_acceleration", "gate_entropy",
            "top12_margin", "token_consensus", "token_dispersion", "mob1_w8"]
ALL_SIGS = SIG_COLS + ["min_mobk"]

# 轴 = 冗余合并后的方向（成功集行上 |Spearman|>0.90 的信号对合并；见 SPEC §3.2）
AXES = ["inst", "desync", "flat", "sharp", "period"]
T_SHARED = {"inst": +1, "desync": +1, "flat": -1}   # loop/static 方向相反的共享轴 -> 极性门
T_LOOP = {"inst": +1, "desync": +1, "flat": -1}                                   # E2 §7
T_STATIC = {"inst": -1, "desync": -1, "flat": +1, "sharp": -1, "period": -1}      # E3 §6
# 臂 P：loop 模板加一条 manifold 轴 = z(d_healthy)（E5 的"偏离健康流形"，方向 +1）
P_T_LOOP = {"inst": +1, "desync": +1, "flat": -1, "manifold": +1}
P_T_STATIC = dict(T_STATIC)

C_T_LOOP = {"mob1": +1}      # 消融臂 C：高 mobility = churn
C_T_STATIC = {"mob1": -1}    # 低 mobility = sticky
CS_T_LOOP = {"mob1": -1}     # 描述性上界臂 C_sticky：两通道同向（只测检出，不分型）
CS_T_STATIC = {"mob1": -1}
# 平凡集长臂 T：一级信号 = 集内 query 序号 q 本身，**不读任何 MoE 列**。
# 单腿（只有一个通道）且用全额 alpha（不做 Bonferroni 二分），使 T 尽可能好响，不削弱它。
T_T_LOOP = {"clock": +1}


# ----------------------------------------------------------------------------
def _load(rows_npz_path, group_col):
    d = np.load(rows_npz_path, allow_pickle=True)
    ep = d["episode_id"].astype(np.int64)
    cs = d["control_step"].astype(np.int64)
    order = np.lexsort((cs, ep))
    if not np.array_equal(order, np.arange(len(ep))):
        ep, cs = ep[order], cs[order]
    q = np.empty_like(cs)
    starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
    for a, b in zip(starts[:-1], starts[1:]):
        q[a:b] = cs[a:b] - cs[a]
        assert np.array_equal(q[a:b], np.arange(b - a)), "episode rows not contiguous"
    X = {c: d[c].astype(np.float64)[order] for c in SIG_COLS}
    mk = d["mob_k"].astype(np.float64)[order]
    with np.errstate(invalid="ignore"):
        X["min_mobk"] = np.where(np.all(np.isnan(mk[:, 1:]), 1), np.nan, np.nanmin(mk[:, 1:], axis=1))
    eps = []
    for a, b in zip(starts[:-1], starts[1:]):
        eps.append(dict(episode_id=int(ep[a]), group=int(d[group_col].astype(np.int64)[order][a]),
                        success=int(d["success"].astype(np.int64)[order][a]),
                        n_q=int(b - a), a=int(a), b=int(b)))
    return dict(q=q, X=X, eps=eps, n=len(q))


def _trailing_mean(x, a, b, W):
    """因果 trailing mean，NaN 传播；返回长度 b-a 的数组，前 W-1 拍为 NaN。"""
    v = x[a:b]
    n = len(v)
    out = np.full(n, np.nan)
    if n < W:
        return out
    ok = np.isfinite(v)
    vv = np.where(ok, v, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(vv)])
    ck = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    idx = np.arange(W - 1, n)
    s = cs[idx + 1] - cs[idx + 1 - W]
    k = ck[idx + 1] - ck[idx + 1 - W]
    good = k == W
    out[idx[good]] = s[good] / W
    return out


def _cs_filter(x, a, b, S, Bw):
    c = _trailing_mean(x, a, b, S)
    base = _trailing_mean(x, a, b, Bw)
    out = np.full(b - a, np.nan)
    out[S:] = c[S:] - base[:-S] if S > 0 else np.nan   # base 窗结束于 t-S
    out[:S + Bw - 1] = np.nan
    return out


def _axes(Z, arm):
    if arm == "T":
        return {"clock": Z["clock"]}
    if arm in ("C", "C_sticky"):
        return {"mob1": Z["mob1_w8"]}
    A = {"inst": 0.5 * (Z["late_flow_volatility"] + Z["route_acceleration"]),
         "desync": 0.5 * (Z["token_dispersion"] - Z["token_consensus"]),
         "flat": Z["gate_entropy"],
         "sharp": Z["top12_margin"],
         "period": Z["min_mobk"]}
    if "d_healthy" in Z:
        A["manifold"] = Z["d_healthy"]
    return A


def _proj(A, tmpl, agg):
    M = np.column_stack([tmpl[k] * A[k] for k in tmpl])
    with np.errstate(invalid="ignore"):
        return np.nanmean(M, 1) if agg == "mean" else np.nanmedian(M, 1)


def _templates(arm, cfg):
    if arm in ("main", "P"):
        tS = dict(P_T_STATIC if arm == "P" else T_STATIC)
        if cfg.get("drop_period"):
            tS.pop("period", None)
        return dict(P_T_LOOP if arm == "P" else T_LOOP), tS
    if arm == "C":
        return dict(C_T_LOOP), dict(C_T_STATIC)
    if arm == "C_sticky":
        return dict(CS_T_LOOP), dict(CS_T_STATIC)
    if arm == "T":
        return dict(T_T_LOOP), None          # 单腿：static 通道禁用
    raise ValueError(arm)


def _score_pass(D, cfg, arm, ref_idx, ref_eps, q_hi, dh=None):
    """用参照池 ref_idx（行掩码）标准化**全部**行，返回逐集 (GL, GS) 轨迹与 op 掩码。"""
    q, X, eps = D["q"], D["X"], D["eps"]
    if arm == "T":
        # 纯集长臂：z 就是 q 本身（对 q 做同 q 标准化在定义上退化 —— MAD 恒为 0，
        # 这正是本结构对"钟"的天然免疫；这里绕过标准化，让 T 以最有利的形态出场）
        A = {"clock": q.astype(np.float64)}
        gL = _proj(A, T_T_LOOP, cfg["agg"])
        out = {}
        for r in eps:
            a, b = r["a"], r["b"]
            GL = _trailing_mean(gL, a, b, cfg["W_loop_sig"])
            GS = np.full(b - a, np.nan)
            qs = np.arange(b - a)
            out[r["episode_id"]] = (GL, GS, (qs >= cfg["q_lo"]) & (qs <= q_hi))
        return out
    sigs = list(ALL_SIGS)
    if dh is not None:
        X = dict(X); X["d_healthy"] = dh; sigs = sigs + ["d_healthy"]
    Z = {s: np.full(D["n"], np.nan) for s in sigs}
    ALL = sigs
    qmax = int(q.max())
    for qq in range(cfg["q_lo"], qmax + 1):
        ref = ref_idx & (np.abs(q - qq) <= cfg["B_ref"])
        if int(ref.sum()) < cfg["n_ref_rows"]:
            continue
        tgt = q == qq
        if not tgt.any():
            continue
        for s in ALL:
            v = X[s][ref]
            v = v[np.isfinite(v)]
            if v.size < cfg["n_ref_rows"]:
                continue
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            if not np.isfinite(mad) or mad <= 0:
                continue
            Z[s][tgt] = (X[s][tgt] - med) / mad
    # 组心化：本组**无标签**早窗中位，留一集外
    if cfg["center"]:
        win = (q >= cfg["q_lo"]) & (q <= cfg["q_cen_hi"])
        by_grp = {}
        for r in eps:
            by_grp.setdefault(r["group"], []).append(r)
        for g, rs in by_grp.items():
            gm = np.zeros(D["n"], bool)
            for r in rs:
                gm[r["a"]:r["b"]] = True
            gm &= win
            for r in rs:
                own = np.zeros(D["n"], bool)
                own[r["a"]:r["b"]] = True
                m = gm & ~own                       # LOEO：排除本集
                for s in ALL:
                    v = Z[s][m]
                    v = v[np.isfinite(v)]
                    if v.size >= cfg["n_cen_rows"]:
                        Z[s][r["a"]:r["b"]] -= np.median(v)
    tL, tS = _templates(arm, cfg)
    A = _axes(Z, arm)
    gL = _proj(A, tL, cfg["agg"])
    gS = _proj(A, tS, cfg["agg"])
    pol = _proj(A, T_SHARED, cfg["agg"]) if arm in ("main", "P") else _proj(A, tL, cfg["agg"])
    out = {}
    for r in eps:
        a, b = r["a"], r["b"]
        GL = (_trailing_mean(gL, a, b, cfg["W_loop_sig"]) if cfg["filt_loop"] == "DC"
              else _cs_filter(gL, a, b, cfg["W_loop_sig"], cfg["W_loop_base"]))
        GS = (_trailing_mean(gS, a, b, cfg["W_static"]) if cfg["filt_static"] == "DC"
              else _cs_filter(gS, a, b, cfg["W_static"], cfg["W_loop_base"]))
        if cfg["polarity_gate"]:
            PL = _trailing_mean(pol, a, b, cfg["W_loop_sig"])
            PS = _trailing_mean(pol, a, b, cfg["W_static"])
            GL = np.where(np.isfinite(PL) & (PL > 0), GL, np.nan)
            GS = np.where(np.isfinite(PS) & (PS < 0), GS, np.nan)
        qs = np.arange(b - a)
        op = (qs >= cfg["q_lo"]) & (qs <= q_hi)
        out[r["episode_id"]] = (GL, GS, op)
    return out


def _q_hi_for(ref_eps, cfg):
    qmax = max((r["n_q"] for r in ref_eps), default=0) - 1
    best = cfg["q_lo"]
    for qq in range(cfg["q_lo"], qmax + 1):
        if sum(1 for r in ref_eps if r["n_q"] > qq) >= cfg["n_ref_eps"]:
            best = qq
        else:
            break
    return best


def _v2_hell(Ae, Ar):
    """V2 聚合的 Hellinger：逐 (layer,token) 格先算距离再对 40 格取均值。"""
    acc = np.zeros((Ae.shape[0], Ar.shape[0]), np.float32)
    for c in range(Ae.shape[1]):
        bc = Ae[:, c, :] @ Ar[:, c, :].T
        np.clip(bc, None, 1.0, out=bc)
        acc += np.sqrt(np.maximum(1.0 - bc, 0.0))
    return acc / Ae.shape[1]


def _d_healthy(reps_path, D, groups, cfg, ep_of_row, grp_of_row, suc_of_row, K=5):
    """DH[g][row] = 到参照池 R_g（除本组外的 success 行）在同 q(+-B) 的 K-NN 距离中位。
    候选集额外剔除该行**自己所在的 episode**（避免自匹配把尺度压塌）。
    P1：对组 g 的行，候选 = R_g，与组 g 的 success 标签无关。
    P2：只用该行自身的 rep 与参照行，天然因果。"""
    R = np.load(reps_path, mmap_mode="r")
    n, dim = R.shape
    R = np.asarray(R, dtype=np.float32).reshape(n, 40, dim // 40)
    q = D["q"]
    gi_of = {g: i for i, g in enumerate(groups)}
    DH = np.full((len(groups), n), np.nan, np.float32)
    M = K + 8   # 一个 episode 在 +-B 窗内最多贡献 2B+1 行；取足够大的候选前缀
    for qq in range(0, int(q.max()) + 1):
        tgt = np.flatnonzero(q == qq)
        ref = np.flatnonzero(suc_of_row & (np.abs(q - qq) <= cfg["B_ref"]))
        if tgt.size == 0 or ref.size < cfg["n_ref_rows"]:
            continue
        Dm = _v2_hell(R[tgt], R[ref])
        rg, re_ = grp_of_row[ref], ep_of_row[ref]
        te = ep_of_row[tgt]
        for g in groups:
            cols = np.flatnonzero(rg != g)
            if cols.size < M + 1:
                continue
            sub = Dm[:, cols]
            pre = np.argpartition(sub, M, axis=1)[:, :M]
            pv = np.take_along_axis(sub, pre, 1)
            o = np.argsort(pv, axis=1)
            pre, pv = np.take_along_axis(pre, o, 1), np.take_along_axis(pv, o, 1)
            cep = re_[cols][pre]                        # (n_tgt, M) 候选所属 episode
            for i in range(len(tgt)):
                keep = pv[i][cep[i] != te[i]][:K]
                if keep.size == K:
                    DH[gi_of[g], tgt[i]] = np.median(keep)
    return DH


def score(rows_npz_path, group_col="scene", config=None, arm="main"):
    """返回逐集的匹配滤波轨迹、可评范围与 LOGO 阈值原料（供 detect 与多 alpha 复用）。"""
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    if arm == "C_sticky":
        cfg["polarity_gate"] = False       # 两通道同向，极性门无定义
    D = _load(rows_npz_path, group_col)
    eps = D["eps"]
    groups = sorted({r["group"] for r in eps})
    DH = None
    if arm == "P":
        import os
        rp = os.path.join(os.path.dirname(rows_npz_path), "reps.npy")
        if not os.path.exists(rp):
            raise FileNotFoundError(f"arm P requires reps.npy next to rows.npz: {rp}")
        ep_of = np.empty(D["n"], np.int64); gr_of = np.empty(D["n"], np.int64)
        su_of = np.zeros(D["n"], bool)
        for r in eps:
            ep_of[r["a"]:r["b"]] = r["episode_id"]; gr_of[r["a"]:r["b"]] = r["group"]
            su_of[r["a"]:r["b"]] = (r["success"] == 1)
        DH = _d_healthy(rp, D, groups, cfg, ep_of, gr_of, su_of)
    res = dict(cfg=cfg, arm=arm, groups=groups, eps={r["episode_id"]: r for r in eps},
               per_group={}, meta=dict(n_groups=len(groups)))
    for gi, g in enumerate(groups):
        ref_eps = [r for r in eps if r["success"] == 1 and r["group"] != g]   # R_g
        ref_idx = np.zeros(D["n"], bool)
        for r in ref_eps:
            ref_idx[r["a"]:r["b"]] = True
        q_hi = _q_hi_for(ref_eps, cfg)
        tr = _score_pass(D, cfg, arm, ref_idx, ref_eps, q_hi,
                         dh=(None if DH is None else DH[gi].astype(np.float64)))
        mx = {}
        for r in ref_eps:
            GL, GS, op = tr[r["episode_id"]]
            mx[r["episode_id"]] = (_safemax(GL, op), _safemax(GS, op))
        own = {r["episode_id"]: tr[r["episode_id"]] for r in eps if r["group"] == g}
        res["per_group"][g] = dict(q_hi=q_hi, ref_max=mx, own=own, n_ref=len(ref_eps))
    return res


def _safemax(G, op):
    m = op & np.isfinite(G)
    return float(np.max(G[m])) if m.any() else -np.inf


def alarms_from_score(S, alpha=None):
    """把 score() 的结果在给定 alpha 下转成逐集报警记录。"""
    cfg = dict(S["cfg"])
    if alpha is not None:
        cfg["alpha"] = alpha
    a_ch = cfg["alpha"] if S.get("arm") == "T" else cfg["alpha"] / 2.0
    out = {}
    for g, pg in S["per_group"].items():
        vL = np.array([v[0] for v in pg["ref_max"].values()], float)
        vS = np.array([v[1] for v in pg["ref_max"].values()], float)
        thr = {"loop": float(np.quantile(vL, 1 - a_ch)) if vL.size else np.inf,
               "static": float(np.quantile(vS, 1 - a_ch)) if vS.size else np.inf}
        for eid, (GL, GS, op) in pg["own"].items():
            r = S["eps"][eid]
            al = []
            for ch, G in (("loop", GL), ("static", GS)):
                hit = np.where(op & np.isfinite(G) & (G >= thr[ch]))[0]
                for t in hit:
                    al.append(dict(step=int(t), channel=ch, score=float(G[t]),
                                   threshold=thr[ch], margin=float(G[t] - thr[ch])))
            al.sort(key=lambda x: (x["step"], -x["margin"]))
            first = {ch: next((x["step"] for x in al if x["channel"] == ch), None)
                     for ch in ("loop", "static")}
            fa = al[0] if al else None
            persist = None
            if fa is not None:
                G = GL if fa["channel"] == "loop" else GS
                te = fa["step"] + cfg["exit_delta"]
                if te < r["n_q"] and te <= pg["q_hi"] and np.isfinite(G[te]):
                    persist = bool(G[te] >= thr[fa["channel"]])
            out[eid] = dict(episode_id=eid, group=g, n_q=r["n_q"],
                            evaluable_range=[cfg["q_lo"], pg["q_hi"]],
                            alarms=al, first_by_channel=first,
                            first_alarm=(None if fa is None else dict(step=fa["step"], channel=fa["channel"])),
                            alarm=bool(al), alarm_q=(None if fa is None else fa["step"]),
                            alarm_type=(None if fa is None else fa["channel"]),
                            loop_q=first["loop"], static_q=first["static"],
                            persist=persist, thr_loop=thr["loop"], thr_static=thr["static"],
                            maxGL=_safemax(GL, op), maxGS=_safemax(GS, op), q_hi=pg["q_hi"],
                            n_ref_episodes=pg["n_ref"])
    return out


def detect(rows_npz_path, group_col="scene", config=None, arm="main"):
    """CONSTRAINTS 规定的接口。返回 {episode_id: record}；record["alarms"] 是完整序列。"""
    return alarms_from_score(score(rows_npz_path, group_col, config, arm))
