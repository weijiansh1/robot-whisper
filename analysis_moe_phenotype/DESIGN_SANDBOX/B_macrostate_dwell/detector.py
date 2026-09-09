#!/usr/bin/env python3
"""方案 B：宏状态带驻留时间检测器（冻结实现）。

规格见 SPEC.md。公开接口：

    from detector import detect, PARAMS, PARAM_SWEEP
    alarms = detect("features/<corpus>/<suite>/<task>/rows.npz", "scene")
    # -> {episode_id: {"alarms": [{"step": q, "channel": "loop"|"static"}, ...], ...}}

只读 MoE routing 列。**不读 events.csv、不读结局标签、不读任何物理量。**
`rows.npz` 里的 `success` 列在检测路径上一次都没有被访问（探针 P1 的直接保证）。

参数一次冻死于 PARAMS。`PARAM_SWEEP` 供执行器把本臂与消融臂 C 对齐到同一工作点。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- 冻结参数
PARAMS = {
    # 分析窗（继承 E8：stickiness 需 8 步拖尾；本方案的带只用 inst/cons，但窗保持同源）
    "QMIN": 8,
    # 组内参考池准入（**无标签**：组内除本集外的全部行）
    "MIN_REF_EPS": 3,     # 组内（去掉自己后）集数
    "MIN_REF_ROWS": 30,   # 组内（去掉自己后）落在 q>=QMIN 的行数
    # 通道旋钮（唯一由「成功集」标定的量，留一组外，见 SPEC.md §6）
    "K_LOOP": 11,     # switching/desync 带连续驻留步数 -> loop 报警
    "M_STATIC": 9,    # flatten/shared 带连续驻留步数 -> static 报警
    "K_C": 14,        # 消融臂 C：sticky 带（mob1_w8 单轴）连续驻留步数
    # 旋钮选取规则的唯一自由量（先验运维容忍度，见 SPEC.md §6）
    "FA_TARGET": 0.10,
}

# 一维单调族：让执行器把 B 与 C 对齐到同一 FA 工作点。冻结点是其中一个元素。
PARAM_SWEEP = {
    "B": [{"K_LOOP": i + 2, "M_STATIC": i} for i in range(4, 26)],
    "B_loop": [{"K_LOOP": k} for k in range(4, 28)],
    "B_static": [{"M_STATIC": m} for m in range(4, 28)],
    "C": [{"K_C": k} for k in range(4, 34)],
}

AXES = ("inst", "comm", "cons", "stick", "flow")
ARM_CHANNELS = {"B": ("loop", "static"), "C": ("sticky",)}
NEEDED_COLS = ("episode_id", "control_step", "scene",
               "late_flow_volatility", "route_acceleration", "top12_margin",
               "token_consensus", "token_dispersion", "mob1_w8", "flow_com")


# ----------------------------------------------------------------- 载入
def _raw_axes(d):
    """五轴原始量（E8 §1 口径，未 z 化）。"""
    f = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
    return {
        "inst": f("late_flow_volatility") + f("route_acceleration"),
        "comm": f("top12_margin"),
        "cons": f("token_consensus") - f("token_dispersion"),
        "stick": -f("mob1_w8"),
        "flow": f("flow_com"),
    }


def load_unit(rows_npz_path, group_col="scene"):
    """返回 (df, raw_axes)；df 含 episode_id/q/group，行序 = (episode_id, q) 升序。

    **不读取 success**。上游若需要结局标签请自行从 events.csv 取，检测器不碰。
    """
    d = np.load(rows_npz_path)
    missing = [c for c in NEEDED_COLS if c not in d.files]
    if missing:
        raise KeyError(f"rows.npz 缺列: {missing}")
    ep = d["episode_id"].astype(np.int64)
    cs = d["control_step"].astype(np.int64)
    n = len(ep)
    assert (np.lexsort((cs, ep)) == np.arange(n)).all(), "rows.npz 未按 (episode, step) 排序"
    starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
    lens = np.diff(np.r_[starts, n])
    q = cs - np.repeat(cs[starts], lens)          # 集内 query 序号
    assert (q == np.concatenate([np.arange(c) for c in lens])).all(), "集内 q 非 0..n-1 连续"
    df = pd.DataFrame({"episode_id": ep, "q": q,
                       "group": np.asarray(d[group_col]).astype(np.int64)})
    return df, _raw_axes(d)


# ----------------------------------------------------------------- 参考池
def build_reference(df, raw, params=PARAMS):
    """逐集的**组内无标签**中位（leave-one-episode-out）。

    分层（冻结）：
      level='scene' —— 同 group 内除本集外的全部行（q>=QMIN 且五轴有限），
                       需 >=MIN_REF_EPS 集且 >=MIN_REF_ROWS 行；
      level='task'  —— 退化：全 task 除本集外的全部行；同样需 >=MIN_REF_ROWS 行；
      level='none'  —— 都不满足，该集不可评。
    **不使用任何结局标签。** 组内 success 随机置换不改变本函数的任何输出（探针 P1）。
    去掉本集自身 ⇒ 参考池与 t 无关 ⇒ 不破坏在线因果性（探针 P2）。
    """
    qmin = params["QMIN"]
    win = (df["q"].to_numpy() >= qmin)
    for a in AXES:
        win &= np.isfinite(raw[a])
    ep = df["episode_id"].to_numpy()
    grp = df["group"].to_numpy()
    eps = np.unique(ep)
    ep_group = {int(e): int(grp[ep == e][0]) for e in eps}

    idx_all = np.flatnonzero(win)
    by_group = {int(g): idx_all[grp[idx_all] == g] for g in np.unique(grp)}

    out = {}
    for e in eps:
        e = int(e)
        cand = by_group[ep_group[e]]
        cand = cand[ep[cand] != e]
        lvl = "scene"
        if (len(np.unique(ep[cand])) < params["MIN_REF_EPS"]
                or len(cand) < params["MIN_REF_ROWS"]):
            cand = idx_all[ep[idx_all] != e]
            lvl = "task"
            if len(cand) < params["MIN_REF_ROWS"]:
                out[e] = dict(level="none", med={}, mad={}, n_ref_rows=0, n_ref_eps=0)
                continue
        med, mad = {}, {}
        for a in AXES:
            x = raw[a][cand]
            m = float(np.median(x))
            med[a] = m
            mad[a] = float(np.median(np.abs(x - m)))
        out[e] = dict(level=lvl, med=med, mad=mad, n_ref_rows=int(len(cand)),
                      n_ref_eps=int(len(np.unique(ep[cand]))))
    return out


# ----------------------------------------------------------------- 带与驻留
def _episode_slices(df):
    ep = df["episode_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
    ends = np.r_[starts[1:], len(ep)]
    return [(int(ep[s]), s, e) for s, e in zip(starts, ends)]


def _first_cross(band, q):
    """band: bool（按 q 升序、q 连续）。cross[k-1] = 连续驻留首次达到 k 的那一步的 q。"""
    cross, best, r = [], 0, 0
    for i in range(len(band)):
        r = r + 1 if band[i] else 0
        while best < r:
            best += 1
            cross.append(int(q[i]))
    return np.asarray(cross, dtype=np.int64)


def _run_alarms(band, q, K):
    """每一段**连续**在带内的游程达到 K 时报一次警（游程断掉后可重新武装）。"""
    out, r = [], 0
    for i in range(len(band)):
        r = r + 1 if band[i] else 0
        if r == K:
            out.append(int(q[i]))
    return out


def episode_bands(df, raw, ref, params=PARAMS):
    """逐集算各带的布尔序列 + 「连续驻留首达 k」表。

    带定义（继承 E8 §3，未改）：
      switch  = inst=1 & cons=0   （switching/desync 带，loop 通道）
      flatten = inst=0 & cons=1   （flatten/shared 带，static 通道）
      sticky  = stick=1           （消融臂 C，mob1_w8 单轴）
      state5  = 五轴宏状态与上一步完全相同（诊断臂 = E8 P_self 的集级对应量）
    """
    qmin = params["QMIN"]
    qmax = params.get("QMAX")          # 仅诊断用的曝光截断（None = 不截断）
    q_all = df["q"].to_numpy()
    grp_all = df["group"].to_numpy()
    keys = ("switch", "flatten", "sticky", "state5")
    out = {}
    for e, s, t in _episode_slices(df):
        r = ref[e]
        q = q_all[s:t]
        base = dict(level=r["level"], n_q=int(len(q)), group=int(grp_all[s]))
        if r["level"] == "none":
            out[e] = dict(base, n_win=0, band={k: np.zeros(0, bool) for k in keys},
                          q=q[:0], **{f"cross_{k}": np.zeros(0, np.int64) for k in keys},
                          **{f"maxrun_{k}": 0 for k in keys},
                          **{f"occ_{k}": np.nan for k in keys})
            continue
        bits, ok = {}, np.ones(t - s, bool)
        for a in AXES:
            x = raw[a][s:t]
            ok &= np.isfinite(x)
            bits[a] = x > r["med"][a]          # 组内无标签中位二分
        ok &= (q >= qmin)
        if qmax is not None:
            ok &= (q <= qmax)
        code = np.zeros(t - s, np.int64)
        for b, a in enumerate(AXES):
            code |= bits[a].astype(np.int64) << b
        same = np.zeros(t - s, bool)
        same[1:] = ok[1:] & ok[:-1] & (code[1:] == code[:-1])
        band = {"switch": bits["inst"] & (~bits["cons"]) & ok,
                "flatten": (~bits["inst"]) & bits["cons"] & ok,
                "sticky": bits["stick"] & ok,
                "state5": same}
        rec = dict(base, n_win=int(ok.sum()), band=band, q=q,
                   n_ref_rows=r["n_ref_rows"], n_ref_eps=r["n_ref_eps"])
        for k, bd in band.items():
            c = _first_cross(bd, q)
            rec[f"cross_{k}"] = c
            rec[f"maxrun_{k}"] = int(len(c))
            rec[f"occ_{k}"] = float(bd.sum() / ok.sum()) if ok.sum() else np.nan
        out[e] = rec
    return out


# ----------------------------------------------------------------- 公开接口
def detect(rows_npz_path, group_col="scene", params=None, arm="B"):
    """在线检测器（冻结）。

    参数
      rows_npz_path : features/<corpus>/<suite>/<task>/rows.npz
      group_col     : 组列名，默认 "scene"（= init_state）
      params        : 可选覆盖（用于 PARAM_SWEEP 工作点对齐），merge 进 PARAMS
      arm           : "B"（交付，双通道 loop/static）或 "C"（消融臂，单通道 sticky）

    返回 dict：
      {episode_id: {"alarms": [{"step": int q, "channel": "loop"/"static"/"sticky"}, ...],
                    "ref_level": "scene"/"task"/"none", "group": int, "n_q": int}}
    `alarms` 是**完整序列**（按 step 升序），同一通道的带游程每次达到阈值都报一条。
    在线因果性：step=s 的判定只用该集 q<=s 的行 + 与 t 无关的离线参考池。
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    if arm not in ARM_CHANNELS:
        raise ValueError(f"arm 必须是 {list(ARM_CHANNELS)}")
    df, raw = load_unit(rows_npz_path, group_col)
    ref = build_reference(df, raw, p)
    bands = episode_bands(df, raw, ref, p)
    chan_spec = ({"loop": ("switch", p["K_LOOP"]), "static": ("flatten", p["M_STATIC"])}
                 if arm == "B" else {"sticky": ("sticky", p["K_C"])})
    out = {}
    for e, b in bands.items():
        al = []
        for ch, (bname, K) in chan_spec.items():
            for st in _run_alarms(b["band"][bname], b["q"], K):
                al.append({"step": st, "channel": ch})
        al.sort(key=lambda a: (a["step"], a["channel"]))
        out[e] = {"alarms": al, "ref_level": b["level"], "group": b["group"],
                  "n_q": b["n_q"]}
    return out


def alarms_to_frame(alarms):
    """便利函数：dict -> 逐集一行的 DataFrame（首次报警 / 分型 / 报警条数）。"""
    rec = []
    for e, v in alarms.items():
        first = {}
        for a in v["alarms"]:
            first.setdefault(a["channel"], a["step"])
        rec.append(dict(episode_id=e, group=v["group"], n_q=v["n_q"],
                        ref_level=v["ref_level"], n_alarms=len(v["alarms"]),
                        first_step=v["alarms"][0]["step"] if v["alarms"] else -1,
                        first_channel=v["alarms"][0]["channel"] if v["alarms"] else "",
                        **{f"first_{c}": first.get(c, -1)
                           for c in ("loop", "static", "sticky")}))
    return pd.DataFrame(rec).sort_values("episode_id").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    a = detect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "scene")
    f = alarms_to_frame(a)
    print(f.head(20).to_string())
    print(json.dumps({"n_eps": int(len(f)),
                      "episodes_with_alarm": float((f.n_alarms > 0).mean()),
                      "alarms_per_episode_med": float(f.n_alarms.median()),
                      "ref_level": f.ref_level.value_counts().to_dict()}, indent=1))
