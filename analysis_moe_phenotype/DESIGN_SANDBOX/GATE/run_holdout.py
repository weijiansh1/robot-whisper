#!/usr/bin/env python3
"""留出集一次性验证执行器（骨架）。

契约见 GATE/RUNNER.md，判据见 GATE/HOLDOUT_PROTOCOL.md。三阶段：

    --preflight   在校准集 SCENE8 上跑探针 P0-P7，不碰留出集
    --run         在留出集上跑四臂，**只落 alarm 明细，绝不计算/打印任何指标**
    --score       一次性读全部明细，写指标表 + 判定书

设计要点（不是形式主义）：
  * `detect()` 永远看不到 events.csv —— 事件表由本执行器独占，评分也由它独做。
  * `--run` 分支不 import 评分代码；留出集只允许跑一次，顺手打印指标就等于给"重跑"开口子。
  * 探针 P1/P2 用机械方式验证 LOGO 与在线因果，不靠原型自述。

用法:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 DESIGN_SANDBOX/GATE/run_holdout.py --preflight --arms A_threshold_fsm ...
"""

from __future__ import annotations

import argparse
import builtins
import csv
import glob
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------- 冻结常量
# 以下全部来自 HOLDOUT_PROTOCOL.md §7.1，冻结后不得修改（改动须走 AMENDMENTS.md）。

SEED = 20260904
NPERM = 2000
LEAD_L = 10          # 检出窗口左界：onset - L
LEAD_T = 5           # 检出窗口右界：onset + T
FA_GATE_X = 0.05     # G1 主口径
FA_GATE_CELL = 0.10  # G1 逐格上限 = 2X
LEAD_GATE = 2        # G2: Lead50 <= +2
DELTA_MIN = 0.10     # W1
P_ALPHA = 0.05       # W2 (family-wise, maxT over F1)
R2_MIN_POSITIVE = 4  # W5/R2: >=4/6 格 Δ>0
DISC_UNDERPOWERED = 0.60  # §4.3 情形 6
S1_DELTA_MIN = 0.20  # §5.1 次端点判据（描述性，不改判）

# --- AMD-2：匹配视界（matched-horizon）主端点（§3A）---
MH_DECILES = tuple(range(10, 100, 10))   # 每任务 9 个视界 = 成功集 n_queries 的 p10..p90
MH_MARGIN_MIN = 0.10                     # M1 / M4：功效 0.84（108 层）；0.05 只有 0.24
MH_SIGN_MIN = 7                          # M3：>=7/9 视界 margin>0（零假设下 P=46/512=0.0898）
MH_T_TOL = 0.02                          # V 闸：|MH(T)| 容差（理论值恒等 0）
MH_T_MAX_VIOL = 2                        # V 闸：允许超容差的视界数上限
PREFIX_DELTA_MIN = 0.20                  # P1：n=97，Δ=0.20 功效 0.82；Δ=0.10 只有 0.18-0.33

# --- AMD-1：集长平凡对照臂 T（§3.5）---
CLOCK_COL = "__clock__"      # = 集内 query 序号 q，唯一的合成列，不读任何 MoE 列
QCOL = "__q__"               # 同上；供所有臂参考的规范集内时间坐标
DELTA_T_MIN = 0.10           # T1，与 W1 同规格
S7_QUANTILE = 90             # S7 健康长度包络 Q_g = 其他组成功集 n_queries 的 p90（LOGO）

CALIB_TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
GROUP_COL = "scene"

# loop/trap 通道无效（AUDIT §7.1 / §8.4）；static 通道不受影响
LOOP_INVALID = {
    ("grid50x8", "libero_long", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
    ("grid50x8", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("main16x32", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
}

# S1 主口径排除（恢复性 loop 真值可疑，AUDIT §8.4）
S1_EXCLUDE_CELLS = {("grid50x8", "libero_object")}

CHANNELS = ("loop", "static")

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(HERE)
ROOT = os.path.dirname(SANDBOX)
RESULTS = os.path.join(HERE, "results")

# 源码静态扫描的禁字（P4b）
FORBIDDEN_TOKENS = [r"events", r"onset", r"trap_", r"summaries",
                    r"sklearn", r"torch", r"\.fit\(", r"lstsq", r"KMeans", r"PCA"]


# ================================================================ 数据装载


def holdout_tasks():
    """(corpus, suite, task) 三元组列表，43 个。"""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "events", "*", "*", "*", "events.csv"))):
        p = f.split(os.sep)
        corpus, suite, task = p[-4], p[-3], p[-2]
        if task == CALIB_TASK:
            continue
        out.append((corpus, suite, task))
    assert len(out) == 43, f"expected 43 holdout tasks, got {len(out)}"
    return out


def calib_units():
    return [("grid50x8", "libero_long", CALIB_TASK), ("main16x32", "libero_long", CALIB_TASK)]


def rows_path(c, s, t):
    return os.path.join(ROOT, "features", c, s, t, "rows.npz")


def events_path(c, s, t):
    return os.path.join(ROOT, "events", c, s, t, "events.csv")


def load_events(c, s, t):
    """返回 per-episode dict，loop 通道已按有效性屏蔽（无效 -> -1，视为缺失不是证据）。

    另附 AMD-1 的 S7 字段 `q_env` = 该集所属组的健康长度包络 Q_g
    （= 该任务**其他组**成功集 n_queries 的 p90，留一组外，与 §2 参照库同口径）。
    """
    li = (c, s, t) in LOOP_INVALID
    rows = list(csv.DictReader(open(events_path(c, s, t), newline="")))
    succ_by_g = {}
    for r in rows:
        if int(r["success"]):
            succ_by_g.setdefault(int(r["scene"]), []).append(int(r["n_queries"]))
    allsucc = [v for vs in succ_by_g.values() for v in vs]
    env = {}
    for g in {int(r["scene"]) for r in rows}:
        ref = [v for gg, vs in succ_by_g.items() if gg != g for v in vs]
        if len(ref) < 10:                     # 参照贫乏时退回全任务成功集（并记在 S7 说明里）
            ref = allsucc
        env[g] = float(np.percentile(ref, S7_QUANTILE)) if ref else float("inf")
    out = {}
    for r in rows:
        lo, g = int(r["loop_onset_q"]), int(r["scene"])
        out[int(r["episode_id"])] = dict(
            scene=g, success=int(r["success"]), n_queries=int(r["n_queries"]),
            loop_onset=(-1 if li else lo), static_onset=int(r["static_onset_q"]),
            raw_loop_onset=lo, loop_valid=(not li), q_env=env[g],
        )
    return out


# ================================================================ 原型加载


def load_arm(arm_dir):
    path = os.path.join(SANDBOX, arm_dir, "detector.py")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"detector_{arm_dir}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__source_path__ = path
    return mod


def call_detect(mod, npz, params):
    """按签名反射调用；兼容误带 events_csv_path 的实现（传 None 并记 WARN）。"""
    sig = inspect.signature(mod.detect)
    kw = {}
    warn = None
    if "events_csv_path" in sig.parameters:
        kw["events_csv_path"] = None
        warn = "WARN_SIGNATURE: detect() 声明了 events_csv_path，已强制传 None"
    if "group_col" in sig.parameters:
        kw["group_col"] = GROUP_COL
    if "params" in sig.parameters:
        kw["params"] = params
    return mod.detect(npz, **kw), warn


def derive_params_T(mod):
    """AMD-1 §3.5：PARAMS_T = PARAMS | {k: "__clock__" for k in ABLATION_KEYS}。

    优先用模块自带的 PARAMS_T；否则机械派生（要求 ABLATION_KEYS 里的值确实是列名，
    即与 PARAMS_C 一样是信号选择键）。派生失败返回 None ⇒ 该臂无 T 对照，
    其 MoE 载荷主张直接作废（HOLDOUT_PROTOCOL §7.1 第 5 条 / §4.3 情形 7）。
    """
    if getattr(mod, "PARAMS_T", None):
        return dict(mod.PARAMS_T), "module"
    pa, ak = mod.PARAMS, list(getattr(mod, "ABLATION_KEYS", []))
    if not ak:
        return None, "no ABLATION_KEYS"
    # 只把「值是字符串列名」的消融键指向时钟列；其余键原样保留
    cols = [k for k in ak if isinstance(pa.get(k), str)]
    if not cols:
        return None, "ABLATION_KEYS 中没有字符串型信号选择键，须自行导出 PARAMS_T"
    return dict(pa) | {k: CLOCK_COL for k in cols}, f"derived({','.join(cols)})"


def episode_q(episode_id, control_step):
    """集内 query 序号 q ∈ [0, n_queries-1]。

    **重要（2026-09-04 发现）**：`rows.npz` 的 `control_step` 是**整任务的全局行号**
    （0..N-1），**不是集内 query 序号**——只有 episode 0 两者巧合相等。
    已在全部 45 任务上验证：集内 `control_step` 严格 +1 连续、逐集行数 == `n_queries`，
    故 `q = control_step - min(control_step | episode)`。
    事件表的 `*_onset_q` 与 `n_queries` 都是**集内**坐标，所以一切时间比较必须用 q。
    """
    q = np.empty(len(control_step), np.int64)
    for e in np.unique(episode_id):
        m = episode_id == e
        q[m] = control_step[m] - control_step[m].min()
    return q


def materialise_clock_npz(src_npz, out_path):
    """写一份带 `__clock__` / `__q__` = 集内 query 序号 的 rows.npz（只给 T 用）。

    只有 T 读这个文件；A/B/D/C 一律读原始 rows.npz，
    以免多出的列意外进入「遍历全部列」型实现的聚合（有意选择的低风险方案）。
    T 的一级信号必须是**集内已用步数**（在线可得），不是全局行号（那等于集的身份证）。
    """
    d = dict(np.load(src_npz))
    q = episode_q(d["episode_id"], d["control_step"]).astype(np.float32)
    d[CLOCK_COL] = q
    d[QCOL] = q
    np.savez(out_path, **d)
    return out_path


def _norm(res, ep_ids):
    """把 detect 返回值规范化为 {ep: {"alarm_step": int|None, "channel": str|None,
    "score": float|None, "alarms": [(step, channel, score), ...]}}。"""
    out = {}
    for e in ep_ids:
        v = res.get(int(e))
        if v is None:
            out[int(e)] = dict(alarm_step=None, channel=None, score=None, alarms=[])
            continue
        al = [(int(a["step"]), a["channel"], a.get("score")) for a in v.get("alarms") or []]
        st = v.get("alarm_step")
        st = None if st is None or st < 0 else int(st)
        if st is not None and not al:
            al = [(st, v.get("channel"), v.get("score"))]
        out[int(e)] = dict(alarm_step=st, channel=v.get("channel") if st is not None else None,
                           score=v.get("score"), alarms=sorted(al))
    return out


# ================================================================ 探针 P0-P7


class _IOGuard:
    """P4：只允许打开白名单路径。命中违规记录后抛异常。"""

    def __init__(self, whitelist):
        self.wl = {os.path.realpath(p) for p in whitelist}
        self.violations = []

    def _check(self, path):
        try:
            rp = os.path.realpath(str(path))
        except Exception:
            return
        if rp in self.wl or rp.endswith(".py") or "/lib/python" in rp or "site-packages" in rp:
            return
        self.violations.append(rp)
        raise PermissionError(f"P4 I/O 白名单违规: {rp}")

    def __enter__(self):
        self._open, self._load = builtins.open, np.load
        guard = self

        def op(file, *a, **k):
            guard._check(file)
            return guard._open(file, *a, **k)

        def ld(file, *a, **k):
            guard._check(file)
            return guard._load(file, *a, **k)

        builtins.open, np.load = op, ld
        return self

    def __exit__(self, *exc):
        builtins.open, np.load = self._open, self._load
        return False


def _write_npz(src_npz, out_path, mutate=None):
    d = dict(np.load(src_npz))
    if mutate:
        mutate(d)
    np.savez(out_path, **d)
    return out_path


def probe_arm(mod, arm_dir, verbose=True):
    """在校准集单位上跑 P0-P5、P4b。返回 (ok, report dict)。"""
    rep = {"arm": arm_dir, "checks": {}, "warnings": []}

    def rec(name, ok, detail=""):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        if verbose:
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    ok = True
    # ---- P0 符号与 SPEC 一致性
    need = ["SPEC_ID", "ARM", "WARMUP", "REQUIRED_COLUMNS", "PARAMS", "PARAMS_C",
            "ABLATION_KEYS", "detect"]
    miss = [n for n in need if not hasattr(mod, n)]
    ok &= rec("P0.symbols", not miss, f"missing={miss}")
    spec_md = os.path.join(SANDBOX, arm_dir, "SPEC.md")
    sid = None
    if os.path.exists(spec_md):
        m = re.search(r"^spec_id:\s*(\S+)", open(spec_md).read(), re.M)
        sid = m.group(1) if m else None
    ok &= rec("P0.spec_id", sid is not None and sid == getattr(mod, "SPEC_ID", None),
              f"SPEC.md={sid} detector={getattr(mod, 'SPEC_ID', None)}")

    # ---- P5 消融臂同口径
    pa, pc = getattr(mod, "PARAMS", {}), getattr(mod, "PARAMS_C", {})
    ak = set(getattr(mod, "ABLATION_KEYS", []))
    diff = {k for k in set(pa) | set(pc) if pa.get(k) != pc.get(k)}
    ok &= rec("P5.same_keys", set(pa) == set(pc), f"sym_diff={set(pa) ^ set(pc)}")
    ok &= rec("P5.ablation_keys", bool(ak) and diff <= ak, f"diff={diff} declared={ak}")

    # ---- P4b 源码静态扫描
    src = open(mod.__source_path__).read()
    hits = [t for t in FORBIDDEN_TOKENS if re.search(t, src)]
    if hits:
        rep["warnings"].append(f"P4b 命中禁字 {hits} -> 转人工 SPEC 审查")
    rec("P4b.source_scan", True, f"hits={hits} (WARN only)")

    if not ok:
        return False, rep

    c, s, t = calib_units()[0]
    npz = rows_path(c, s, t)
    d = np.load(npz)
    ep = d["episode_id"]
    ep_ids = np.unique(ep)

    # ---- P4 I/O 白名单
    try:
        with _IOGuard([npz]):
            base, warn = call_detect(mod, npz, mod.PARAMS)
        ok &= rec("P4.io_whitelist", True, "只打开了 rows.npz")
    except PermissionError as e:
        return False, (rep | {"checks": rep["checks"] | {"P4.io_whitelist":
                       {"ok": False, "detail": str(e)}}})
    if warn:
        rep["warnings"].append(warn)
    base = _norm(base, ep_ids)

    # ---- P7 覆盖与一致性
    bad = [e for e, v in base.items()
           if (v["alarm_step"] is None) != (v["channel"] is None)
           or (v["channel"] is not None and v["channel"] not in CHANNELS)]
    ok &= rec("P7.coverage", len(base) == len(ep_ids) and not bad,
              f"n={len(base)}/{len(ep_ids)} bad={bad[:5]}")

    # ---- P6 warm-up + P8 时间坐标（alarm_step 必须是**集内** q，不是全局 control_step）
    nq = {int(e): int((ep == e).sum()) for e in ep_ids}         # = n_queries（已全语料验证）
    viol = [(e, v["alarm_step"]) for e, v in base.items() if v["alarm_step"] is not None
            and not (mod.WARMUP <= v["alarm_step"] <= nq[e] - 1)]
    ok &= rec("P6.warmup", not viol, f"violations={viol[:5]} WARMUP={mod.WARMUP}")
    over = [(e, v["alarm_step"], nq[e]) for e, v in base.items()
            if v["alarm_step"] is not None and v["alarm_step"] > nq[e] - 1]
    ok &= rec("P8.time_coord", not over,
              (f"{len(over)} 个集的 alarm_step 超出集长，样例={over[:3]} —— "
               "极可能把 rows.npz 的 `control_step`（**整任务全局行号**）当成了集内 query 序号。"
               "正确做法：q = control_step - min(control_step | episode)（见 RUNNER.md §2.1）")
              if over else "alarm_step 全部落在 [WARMUP, n_queries-1]，坐标系正确")

    # ---- P3 确定性
    again, _ = call_detect(mod, npz, mod.PARAMS)
    again = _norm(again, ep_ids)
    ok &= rec("P3.determinism", again == base, "两次调用逐位比较")

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(SEED)
        # ---- P1 LOGO 标签探针
        scenes = d[GROUP_COL]
        ep_scene = {int(e): int(scenes[ep == e][0]) for e in ep_ids}
        ep_succ = {int(e): int(d["success"][ep == e][0]) for e in ep_ids}
        mixed = [g for g in np.unique(scenes)
                 if len({ep_succ[e] for e in ep_ids if ep_scene[int(e)] == g}) > 1]
        p1_ok, p1_det = True, "no mixed group"
        for g in list(mixed)[:2]:
            gm = [int(e) for e in ep_ids if ep_scene[int(e)] == g]
            perm = rng.permutation([ep_succ[e] for e in gm])
            newlab = dict(zip(gm, perm))

            def mut(dd, gm=gm, newlab=newlab):
                sc = dd["success"].copy()
                for e in gm:
                    sc[dd["episode_id"] == e] = newlab[e]
                dd["success"] = sc

            p = _write_npz(npz, os.path.join(td, f"p1_{g}.npz"), mut)
            r2, _ = call_detect(mod, p, mod.PARAMS)
            r2 = _norm(r2, ep_ids)
            ch = [e for e in gm if r2[e] != base[e]]
            if ch:
                p1_ok, p1_det = False, f"组 {g} 内 {len(ch)} 集报警随组内标签置换而变: {ch[:5]}"
                break
            p1_det = f"检查 {len(mixed[:2])} 个混合标签组，组内报警不变"
        ok &= rec("P1.logo_labels", p1_ok, p1_det)

        # ---- P2 前缀因果探针
        fired = [e for e, v in base.items() if v["alarm_step"] is not None][:5]
        p2_ok, p2_det = True, f"检查 {len(fired)} 个已报警集"
        for e in fired:
            sstep = base[e]["alarm_step"]
            for cut, expect_fire in ((sstep, True), (sstep - 1, False)):
                if cut < 0:
                    continue

                def mut(dd, e=e, cut=cut):
                    # cut 是**集内** q；转成该集的全局 control_step 阈值
                    m = dd["episode_id"] == e
                    base_cs = int(dd["control_step"][m].min())
                    keep = ~(m & (dd["control_step"] > base_cs + cut))
                    for k in list(dd):
                        if isinstance(dd[k], np.ndarray) and dd[k].shape[:1] == keep.shape:
                            dd[k] = dd[k][keep]

                p = _write_npz(npz, os.path.join(td, f"p2_{e}_{cut}.npz"), mut)
                r2, _ = call_detect(mod, p, mod.PARAMS)
                r2 = _norm(r2, ep_ids)
                got = r2.get(e, {}).get("alarm_step")
                good = (got == sstep and r2[e]["channel"] == base[e]["channel"]) \
                    if expect_fire else (got is None)
                if not good:
                    p2_ok = False
                    p2_det = (f"集 {e}: 首报 s={sstep}，截断到 <= {cut} 后得 {got}"
                              f"（期望 {'s' if expect_fire else 'None'}）=> 非因果")
                    break
            if not p2_ok:
                break
        ok &= rec("P2.causality", p2_ok, p2_det)

    rep["ok"] = bool(ok)
    return bool(ok), rep


# ================================================================ 阶段 1: --run


def _write_alarms(fn, res):
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "alarm_step", "channel", "score", "n_alarms", "all_alarms"])
        for e in sorted(res):
            v = res[e]
            w.writerow([e, -1 if v["alarm_step"] is None else v["alarm_step"],
                        v["channel"] or "", "" if v["score"] is None else v["score"],
                        len(v["alarms"]), ";".join(f"{a}:{b}" for a, b, _ in v["alarms"])])


def run_arms(arm_dirs):
    """只落 alarm 明细。**本函数不计算、不打印任何率值。**

    每个原型跑三条曲线：主臂 / 消融臂 C（mob1_w8）/ 对照臂 T（__clock__，AMD-1）。
    """
    os.makedirs(RESULTS, exist_ok=True)
    tasks = holdout_tasks()
    mods = {a: load_arm(a) for a in arm_dirs}
    plans, t_status = [], {}
    for a in arm_dirs:
        m = mods[a]
        plans += [(a, m, m.PARAMS, False), (a + "__C", m, m.PARAMS_C, False)]
        pt, how = derive_params_T(m)
        t_status[a] = how
        if pt is None:
            print(f"[WARN] {a}: 无法构造对照臂 T（{how}）"
                  " -> 该臂 MoE 载荷主张作废（HOLDOUT_PROTOCOL §4.3 情形 7）")
        else:
            plans.append((a + "__T", m, pt, True))
    for label, _, _, _ in plans:
        os.makedirs(os.path.join(RESULTS, label, "alarms"), exist_ok=True)

    manifest = []
    with tempfile.TemporaryDirectory() as td:
        for i, (c, s, t) in enumerate(tasks, 1):
            npz = rows_path(c, s, t)
            ep_ids = np.unique(np.load(npz)["episode_id"])
            clock_npz = None
            for label, mod, params, needs_clock in plans:
                if needs_clock and clock_npz is None:
                    clock_npz = materialise_clock_npz(npz, os.path.join(td, "clock.npz"))
                res, _ = call_detect(mod, clock_npz if needs_clock else npz, params)
                fn = os.path.join(RESULTS, label, "alarms", f"{c}__{s}__{t}.csv")
                _write_alarms(fn, _norm(res, ep_ids))
                manifest.append((fn, hashlib.sha256(open(fn, "rb").read()).hexdigest()))
            if clock_npz:
                os.remove(clock_npz)
            print(f"[--run] {i:2d}/43 {c}/{s}/{t[:40]}  ({len(plans)} 条曲线)")

    with open(os.path.join(RESULTS, "MANIFEST.sha256"), "w") as fh:
        for fn, h in sorted(manifest):
            fh.write(f"{h}  {os.path.relpath(fn, RESULTS)}\n")
    json.dump(t_status, open(os.path.join(RESULTS, "arm_T_construction.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n[--run] 完成。{len(manifest)} 个明细文件已落盘并哈希存档。"
          f"\n[--run] 对照臂 T 构造方式: {t_status}"
          f"\n[--run] 按 RUNNER.md §5，本阶段不输出任何指标。请单独运行 --score。")


# ================================================================ 阶段 2: --score


def _read_alarms(label, c, s, t):
    fn = os.path.join(RESULTS, label, "alarms", f"{c}__{s}__{t}.csv")
    out = {}
    with open(fn, newline="") as fh:
        for r in csv.DictReader(fh):
            st = int(r["alarm_step"])
            al = []
            for chunk in filter(None, (r.get("all_alarms") or "").split(";")):
                a, b = chunk.split(":")
                al.append((int(a), b))
            if st >= 0 and not al:
                al = [(st, r["channel"])]
            out[int(r["episode_id"])] = dict(
                alarm_step=None if st < 0 else st,
                channel=r["channel"] or None, alarms=sorted(al))
    return out


def _anchors(ev, only=None):
    """该集的可用锚点 [(onset, channel)]。loop 通道无效的任务已在 load_events 屏蔽。"""
    a = []
    if ev["loop_valid"] and ev["loop_onset"] >= 0:
        a.append((ev["loop_onset"], "loop"))
    if ev["static_onset"] >= 0:
        a.append((ev["static_onset"], "static"))
    return [x for x in a if only is None or x[1] == only]


def _detected(ev, al, only=None, agnostic=False):
    """返回 (detected: bool|None, lead: int|None, mistype: bool)。窗口规则见 PROTOCOL §3.1。

    detected=None 表示该集没有可用锚点（不入分母）。`only` 限定通道（S3/S1 用）。
    `agnostic=True`（**仅用于对照臂 T**，AMD-1 §3.5 第 5 条）：免除通道匹配要求，
    报警落进任一通道窗口即算命中——这是**故意给地板对照的优待**，必须在报告中声明。
    """
    anchors = _anchors(ev, only)
    if not anchors:
        return None, None, False
    best, mistype = None, False
    for onset, ch in anchors:
        lo, hi = onset - LEAD_L, min(onset + LEAD_T, ev["n_queries"] - 1)
        for step, ach in al["alarms"]:
            if lo <= step <= hi:
                if agnostic or ach == ch:
                    lead = step - onset
                    best = lead if best is None else min(best, lead)
                else:
                    mistype = True
    return (best is not None), best, (best is None and mistype)


def task_horizons(ev):
    """AMD-2 §3A.1：视界网格 = 该任务成功集 n_queries 的 p10..p90，**恰好 9 个、不去重**。

    只用成功集集长定义——不碰失败集、不碰任何信号。
    **不得去重**：短任务（集长中位 8–13）的多个分位会落到同一个整数 q，
    去重会让列表长度随任务变化，位置索引就不再等于分位名次，
    跨任务汇总时视界会错位（实测会把 p20 的层数从 83 打成 39）。
    重复的 q 只是按集长分布的密度加权，口径仍然良定义。
    """
    succ = [v["n_queries"] for v in ev.values() if v["success"] == 1]
    if not succ:
        return []
    return [int(np.floor(np.percentile(succ, d))) for d in MH_DECILES]


def matched_horizon(recs, channel=None):
    """AMD-2 主端点：匹配视界 margin。

    recs: 该臂的逐集记录（须含 corpus/task/group/nq/onsets/alarm_steps/anchored/clean）。
    返回 dict：
      by_h   : {horizon_rank: margin(q)}   逐视界的层均值（视界按任务内的 decile 名次对齐）
      MH     : 9 个视界 margin 的中位
      sign   : margin>0 的视界数 k（/9）
      strata : {stratum_key: 该层在各可用视界上 margin 的均值}  -> 供组级 sign-flip
      sizes  : 逐视界 (可用层数, 事件风险集, 干净成功风险集)
    `channel` 限定通道（None = 任一通道；对照臂 T 传 None，因为 T 通道无关计分）。
    """
    per_rank = {d: [] for d in range(len(MH_DECILES))}
    sizes = {d: [0, 0, 0] for d in range(len(MH_DECILES))}
    strata_vals = {}
    by_task = {}
    for r in recs:
        by_task.setdefault((r["corpus"], r["suite"], r["task"]), []).append(r)
    for tk, rs in by_task.items():
        hz = rs[0]["horizons"]
        for d, q in enumerate(hz):
            bg = {}
            for r in rs:
                if r["nq"] <= q:                       # 该视界上已经结束，不在风险集
                    continue
                fired_by_q = any(st <= q and (channel is None or ch == channel)
                                 for st, ch in r["alarm_steps"])
                if r["anchored"]:
                    ons = [o for o, ch in r["onsets"] if channel is None or ch == channel]
                    if not ons or q < min(ons) - LEAD_L:
                        continue                       # 还没进入计入窗口左界
                    bg.setdefault(r["group"], [[], []])[0].append(fired_by_q)
                elif r["clean"]:
                    bg.setdefault(r["group"], [[], []])[1].append(fired_by_q)
            for g, (ev_, cl_) in bg.items():
                if not ev_ or not cl_:                 # 层在该视界不可用（单边）
                    continue
                m = float(np.mean(ev_)) - float(np.mean(cl_))
                per_rank[d].append(m)
                strata_vals.setdefault(g, []).append(m)
                sizes[d][0] += 1
                sizes[d][1] += len(ev_)
                sizes[d][2] += len(cl_)
    by_h = {d: (float(np.mean(v)) if v else float("nan")) for d, v in per_rank.items()}
    vals = [v for v in by_h.values() if v == v]
    return dict(
        by_h=by_h, MH=(float(np.median(vals)) if vals else float("nan")),
        sign=int(sum(v > 0 for v in vals)), n_horizons=len(vals),
        strata={g: float(np.mean(v)) for g, v in strata_vals.items()},
        sizes={d: tuple(s) for d, s in sizes.items()},
    )


def macro_type_acc(recs):
    """AMD-2 §5A.2：分型准确率 = 两通道召回的**宏平均**。微平均一律禁止。"""
    out = {}
    for ch in CHANNELS:
        v = [r[f"fatal_{ch}_hit"] for r in recs if r[f"fatal_{ch}"]]
        out[ch] = (float(np.mean(v)) if v else float("nan"), len(v))
    rc = [out[ch][0] for ch in CHANNELS]
    return dict(recall_loop=out["loop"][0], n_loop=out["loop"][1],
                recall_static=out["static"][0], n_static=out["static"][1],
                TypeAcc_macro=float(np.nanmean(rc)),
                TypeAcc_const_baseline=0.5,
                note="禁止微平均：本留出集上『恒判 loop』微平均 0.819，宏平均 0.500")


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, ctr - hw, ctr + hw


def _verify_manifest():
    """alarm 明细在 --run 之后不得被改动（§6 R-A：留出集只跑一次，结果不可事后编辑）。"""
    mf = os.path.join(RESULTS, "MANIFEST.sha256")
    if not os.path.exists(mf):
        sys.exit("缺少 results/MANIFEST.sha256：必须先 --run。")
    bad = []
    for line in open(mf):
        h, rel = line.strip().split("  ", 1)
        p = os.path.join(RESULTS, rel)
        if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            bad.append(rel)
    if bad:
        sys.exit(f"MANIFEST 校验失败（{len(bad)} 个文件被改动或缺失）：{bad[:5]}")
    print(f"[--score] MANIFEST 校验通过（{sum(1 for _ in open(mf))} 个明细文件未被改动）。")


def score(arm_dirs):
    sys.path.insert(0, ROOT)
    from phenotype.stats import groupwise_signflip_maxt  # noqa: E402

    labels, has_T = [], {}
    for a in arm_dirs:
        labels += [a, a + "__C"]
        has_T[a] = os.path.isdir(os.path.join(RESULTS, a + "__T", "alarms"))
        if has_T[a]:
            labels.append(a + "__T")
        else:
            print(f"[--score] {a}: 缺对照臂 T -> 该臂 MoE 载荷主张作废（§4.3 情形 7）")
    tasks = holdout_tasks()
    _verify_manifest()

    # ---- 逐集装配（事件表只在这里出现，检测器从未见过）
    per = {lab: [] for lab in labels}          # list of dict rows
    for c, s, t in tasks:
        ev = load_events(c, s, t)
        hz = task_horizons(ev)                     # AMD-2：每任务 9 个视界
        for lab in labels:
            al = _read_alarms(lab, c, s, t)
            ag = lab.endswith("__T")   # AMD-1：只有 T 免除通道匹配
            for e, v in ev.items():
                a = al.get(e, dict(alarm_step=None, channel=None, alarms=[]))
                det, lead, mis = _detected(v, a, agnostic=ag)
                fatal = (v["success"] == 0)
                # 主端点分母：**致败** 且有锚（成功集里的事件绝不进主端点，它们是 S1 的负对照）
                anchored = fatal and det is not None
                det_loop, _, _ = _detected(v, a, only="loop", agnostic=ag)
                det_static, _, _ = _detected(v, a, only="static", agnostic=ag)
                # S7 长度匹配子集：最早锚点早于该组健康长度包络 Q_g
                onsets = [o for o, _ in _anchors(v)]
                in_env = bool(onsets) and (min(onsets) <= v["q_env"])
                # AMD-2 §5A.1 前缀端点：丢弃 q > Q_g 的报警后重判
                a_pref = dict(alarms=[x for x in a["alarms"] if x[0] <= v["q_env"]])
                det_pref, _, _ = _detected(v, a_pref, agnostic=ag)
                clean = (v["success"] == 1 and v["static_onset"] < 0 and v["loop_onset"] < 0)
                recov = (v["success"] == 1 and v["loop_valid"] and v["loop_onset"] >= 0)
                per[lab].append(dict(
                    corpus=c, suite=s, task=t, ep=e, group=(c, t, v["scene"]),
                    # --- AMD-2 匹配视界所需的原始量 ---
                    horizons=hz, nq=v["n_queries"], onsets=_anchors(v),
                    alarm_steps=list(a["alarms"]), q_env=v["q_env"],
                    prefix_reachable=(anchored and in_env),
                    prefix_hit=(bool(det_pref) if (anchored and in_env) else None),
                    success=v["success"], anchored=anchored,
                    s7=(anchored and in_env),          # AMD-1 §5 S7 长度匹配子集
                    detected=bool(det) if anchored else None,
                    lead=lead if anchored else None, mistype=mis and anchored,
                    clean=clean, loop_valid=v["loop_valid"],
                    # S1 负对照：成功集里的瞬态 loop（loop 通道有效任务）
                    recovery_loop=recov, recovery_hit=(bool(det_loop) if recov else None),
                    fatal_loop=(fatal and det_loop is not None),
                    fatal_loop_hit=(bool(det_loop) if (fatal and det_loop is not None) else None),
                    fatal_static=(fatal and det_static is not None),
                    fatal_static_hit=(bool(det_static) if (fatal and det_static is not None) else None),
                    unanchored_fail=(fatal and det is None),
                    fired=(a["alarm_step"] is not None),
                    fired_loop=any(ch == "loop" for _, ch in a["alarms"]),
                    fired_static=any(ch == "static" for _, ch in a["alarms"]),
                    uncertain_loop_alarm=((not v["loop_valid"]) and v["success"] == 1
                                          and any(ch == "loop" for _, ch in a["alarms"])),
                ))

    def cell(r):
        return f"{r['corpus']}/{r['suite']}"

    rows_out, det_ind, det_ch, det_s7, det_pfx, MH = {}, {}, {}, {}, {}, {}
    for lab in labels:
        R = per[lab]
        # ---- AMD-2 主端点：匹配视界 margin（loop 判决 / static 只报 / 全通道给 T 用）
        MH[lab] = {"loop": matched_horizon(R, "loop"),
                   "static": matched_horizon(R, "static"),
                   "any": matched_horizon(R, None)}
        anch = [r for r in R if r["anchored"]]
        ekey = lambda r: (r["corpus"], r["task"], r["ep"])  # noqa: E731
        det_ind[lab] = {ekey(r): bool(r["detected"]) for r in anch}
        det_s7[lab] = {ekey(r): bool(r["detected"]) for r in anch if r["s7"]}
        det_pfx[lab] = {ekey(r): bool(r["prefix_hit"]) for r in R if r["prefix_reachable"]}
        det_ch[lab] = {
            "loop": {ekey(r): bool(r["fatal_loop_hit"]) for r in R if r["fatal_loop"]},
            "static": {ekey(r): bool(r["fatal_static_hit"]) for r in R if r["fatal_static"]},
        }
        det_k = sum(r["detected"] for r in anch)
        fa_pop = [r for r in R if r["clean"] and r["loop_valid"]]
        fa_k = sum(r["fired"] for r in fa_pop)
        fa_p, fa_lo, fa_hi = _wilson(fa_k, len(fa_pop))
        cells_fa = {}
        for cn in sorted({cell(r) for r in fa_pop}):
            sub = [r for r in fa_pop if cell(r) == cn]
            cells_fa[cn] = sum(r["fired"] for r in sub) / len(sub)
        worst = max(cells_fa, key=cells_fa.get) if cells_fa else ""
        leads = np.array([r["lead"] for r in anch if r["detected"]], float)
        l50 = float(np.median(leads)) if len(leads) else float("nan")
        rows_out[lab] = dict(
            arm=lab, det_n=len(anch), det_k=det_k, det_rate=det_k / max(len(anch), 1),
            fa_n=len(fa_pop), fa_k=fa_k, fa_rate=fa_p, fa_ci_lo=fa_lo, fa_ci_hi=fa_hi,
            fa_worst_cell=worst, fa_worst_cell_rate=cells_fa.get(worst, float("nan")),
            lead50=l50,
            lead_q25=float(np.percentile(leads, 25)) if len(leads) else float("nan"),
            lead_q75=float(np.percentile(leads, 75)) if len(leads) else float("nan"),
            mistype_rate=sum(r["mistype"] for r in anch) / max(len(anch), 1),
            gate_G1=bool(fa_p <= FA_GATE_X and cells_fa.get(worst, 1) <= FA_GATE_CELL),
            gate_G2=bool(l50 <= LEAD_GATE),
        )
        rows_out[lab]["admissible"] = rows_out[lab]["gate_G1"] and rows_out[lab]["gate_G2"]

        # ---- 次端点 S1-S4（描述性；不参与选臂与 §4.3 判定）
        def rate(pred, key):
            v = [r[key] for r in R if pred(r)]
            return (float(np.mean(v)) if v else float("nan")), len(v)

        s1_ok = lambda r: (r["corpus"], r["suite"]) not in S1_EXCLUDE_CELLS  # noqa: E731
        fl_m, fl_n = rate(lambda r: r["fatal_loop"] and s1_ok(r), "fatal_loop_hit")
        rl_m, rl_n = rate(lambda r: r["recovery_loop"] and s1_ok(r), "recovery_hit")
        fl_a, fl_an = rate(lambda r: r["fatal_loop"], "fatal_loop_hit")
        rl_a, rl_an = rate(lambda r: r["recovery_loop"], "recovery_hit")
        s2_m, s2_n = rate(lambda r: r["unanchored_fail"], "fired")
        st_m, st_n = rate(lambda r: r["fatal_static"], "fatal_static_hit")
        unc_pop = [r for r in R if not r["loop_valid"] and r["success"] == 1]
        rows_out[lab] |= dict(
            S1_fatal_loop_rate=fl_m, S1_fatal_loop_n=fl_n,
            S1_recovery_loop_rate=rl_m, S1_recovery_loop_n=rl_n,
            S1_delta=fl_m - rl_m,
            S1_delta_all_cells=fl_a - rl_a, S1_n_fatal_all=fl_an, S1_n_recovery_all=rl_an,
            S2_unanchored_fail_rate=s2_m, S2_n=s2_n,
            S3_loop_recall=fl_a, S3_static_recall=st_m, S3_static_n=st_n,
            S4_uncertain_loop_alarm_rate=(
                float(np.mean([r["uncertain_loop_alarm"] for r in unc_pop])) if unc_pop
                else float("nan")),
            S7_n=len(det_s7[lab]),
            S7_det_rate=(float(np.mean(list(det_s7[lab].values())))
                         if det_s7[lab] else float("nan")),
            # ---- AMD-2 主端点与第二证据线
            MH_loop=MH[lab]["loop"]["MH"], MH_loop_sign=MH[lab]["loop"]["sign"],
            MH_loop_strata=len(MH[lab]["loop"]["strata"]),
            MH_static=MH[lab]["static"]["MH"], MH_static_sign=MH[lab]["static"]["sign"],
            MH_any=MH[lab]["any"]["MH"], MH_any_sign=MH[lab]["any"]["sign"],
            MH_macro=float(np.nanmean([MH[lab]["loop"]["MH"], MH[lab]["static"]["MH"]])),
            prefix_n=len(det_pfx[lab]),
            prefix_det_rate=(float(np.mean(list(det_pfx[lab].values())))
                             if det_pfx[lab] else float("nan")),
        ) | {f"Type_{k}": v for k, v in macro_type_acc(R).items()}

    # ---- 族 F1：3 个臂内配对对比，组级 sign-flip maxT
    groups = sorted({r["group"] for r in per[labels[0]] if r["anchored"]})
    gi = {g: i for i, g in enumerate(groups)}
    got = (rows_out[labels[0]]["det_n"], rows_out[labels[0]]["fa_n"], len(groups))
    print(f"[--score] 分母自检: 有锚致败集={got[0]} (预期 332)  "
          f"干净成功集={got[1]} (预期 15591)  含有锚致败集的组={got[2]} (预期 180)")
    if got != (332, 15591, 180):
        sys.exit("[--score] 分母自检失败：评估集构成与 HOLDOUT_PROTOCOL §1.1 不符，中止。"
                 "events/ 被改动过？还是留出集名单被动过？须走 AMENDMENTS.md 程序。")
    def group_effects(pairs):
        """pairs: list[(ref_label_of_arm_j, det_dict_A, det_dict_B)] -> (n_groups, n_arms)。"""
        e = np.full((len(groups), len(pairs)), np.nan)
        for j, (a, A, B) in enumerate(pairs):
            b = {}
            for r in per[a]:
                k = (r["corpus"], r["task"], r["ep"])
                if r["anchored"] and k in A and k in B:
                    b.setdefault(r["group"], []).append(int(A[k]) - int(B[k]))
            for g, vs in b.items():
                e[gi[g], j] = float(np.mean(vs))
        return e

    def pooled(A, B):
        ks = sorted(set(A) & set(B))
        if not ks:
            return float("nan"), float("nan")
        return (float(np.mean([int(A[k]) - int(B[k]) for k in ks])),
                float(np.mean([A[k] != B[k] for k in ks])))

    # ================= AMD-2：V 闸（口径有效性）+ 主端点族 F1_M / F1_MC / F_P =========
    def strata_matrix(dicts):
        """把若干 {stratum: value} 对齐成 (n_strata, n_cells)，缺失填 NaN。"""
        keys = sorted(set().union(*[set(d) for d in dicts])) if dicts else []
        M = np.full((len(keys), len(dicts)), np.nan)
        for j, d in enumerate(dicts):
            for i, k in enumerate(keys):
                if k in d:
                    M[i, j] = d[k]
        return M

    validity = {}
    for a in arm_dirs:
        if not has_T[a]:
            validity[a] = dict(ok=False, reason="缺对照臂 T，无法校验匹配视界口径")
            continue
        mt = MH[a + "__T"]["any"]
        viol = [d for d, v in mt["by_h"].items() if v == v and abs(v) > MH_T_TOL]
        validity[a] = dict(
            ok=bool(abs(mt["MH"]) <= MH_T_TOL and len(viol) <= MH_T_MAX_VIOL),
            MH_T=mt["MH"], sign=mt["sign"], n_horizons=mt["n_horizons"],
            horizons_over_tol=viol, by_h=mt["by_h"], sizes=mt["sizes"],
            tol=MH_T_TOL,
            note="理论值恒等 0（§3A.2 构造性论证）；超容差 ⇒ 匹配视界实现有误 ⇒ 整次运行作废")
    json.dump(validity, open(os.path.join(RESULTS, "mh_validity.json"), "w"),
              indent=1, ensure_ascii=False, default=str)
    v_failed = [a for a, v in validity.items() if not v["ok"]]
    if v_failed:
        print(f"[--score] **V 闸不过**（§4.3 情形 0）: {v_failed} -> "
              "匹配视界口径实现有误，本次运行作废，修复后全部重跑。"
              " 明细见 results/mh_validity.json")

    rng = np.random.default_rng(SEED)
    # F1_M：各臂自身的 MH_loop > 0（= §3A.2 意义下的"打赢 T"）
    obs_m, p_m = groupwise_signflip_maxt(
        strata_matrix([MH[a]["loop"]["strata"] for a in arm_dirs]), NPERM,
        np.random.default_rng(SEED))
    # F1_MC：MH_loop(arm) - MH_loop(C)，按层配对
    mc_strata = []
    for a in arm_dirs:
        sa, sc = MH[a]["loop"]["strata"], MH[a + "__C"]["loop"]["strata"]
        mc_strata.append({g: sa[g] - sc[g] for g in set(sa) & set(sc)})
    obs_mc, p_mc = groupwise_signflip_maxt(strata_matrix(mc_strata), NPERM,
                                           np.random.default_rng(SEED))

    # ---- 族 F1_C（3 格）与族 F1_T（3 格），**各自族内** maxT；合取按 IUT 不跨族校正
    pc = [(a, det_ind[a], det_ind[a + "__C"]) for a in arm_dirs]
    obs_c, p_c = groupwise_signflip_maxt(group_effects(pc), NPERM, rng)
    armsT = [a for a in arm_dirs if has_T[a]]
    if armsT:
        pt = [(a, det_ind[a], det_ind[a + "__T"]) for a in armsT]
        obs_t, p_t = groupwise_signflip_maxt(group_effects(pt), NPERM,
                                             np.random.default_rng(SEED))
    tix = {a: i for i, a in enumerate(armsT)}

    contrasts = []
    for j, a in enumerate(arm_dirs):
        A, C = det_ind[a], det_ind[a + "__C"]
        dC, disc = pooled(A, C)
        cd = dict(arm=a, delta_pooled=dC, disc_rate=disc,
                  delta_group=float(obs_c[j]), p_maxT_F1_C=float(p_c[j]))
        # ---- R1 / R2（vs C）
        by = {}
        for r in per[a]:
            k = (r["corpus"], r["task"], r["ep"])
            if r["anchored"]:
                by.setdefault(r["corpus"], []).append(int(A[k]) - int(C[k]))
                by.setdefault(cell(r), []).append(int(A[k]) - int(C[k]))
        cd["R1_grid"] = float(np.mean(by.get("grid50x8", [np.nan]))) > 0
        cd["R1_main"] = float(np.mean(by.get("main16x32", [np.nan]))) > 0
        cd["R2_cells_positive"] = sum(
            1 for k, v in by.items() if "/" in k and float(np.mean(v)) > 0)
        cd["W1"] = cd["delta_pooled"] >= DELTA_MIN
        cd["W2"] = cd["p_maxT_F1_C"] < P_ALPHA
        cd["W3"] = cd["delta_group"] > 0
        cd["W5"] = (cd["R1_grid"] and cd["R1_main"]
                    and cd["R2_cells_positive"] >= R2_MIN_POSITIVE)

        # ---- AMD-1：T 组（族 F1_T + 通道级 F4 + F5）
        if has_T[a]:
            T = det_ind[a + "__T"]
            dT, discT = pooled(A, T)
            cd |= dict(delta_T_pooled=dT, disc_rate_T=discT,
                       delta_T_group=float(obs_t[tix[a]]),
                       p_maxT_F1_T=float(p_t[tix[a]]))
            cd["T1"] = dT >= DELTA_T_MIN
            cd["T2"] = cd["p_maxT_F1_T"] < P_ALPHA
            cd["T3"] = cd["delta_T_group"] > 0
            # F4：通道级作废判定，**逐格 alpha=0.05，不做 maxT**（liberal voiding）
            for ch in CHANNELS:
                dch, _ = pooled(det_ch[a][ch], det_ch[a + "__T"][ch])
                e1 = group_effects([(a, det_ch[a][ch], det_ch[a + "__T"][ch])])
                _, pch = groupwise_signflip_maxt(e1, NPERM, np.random.default_rng(SEED))
                cd[f"delta_T_{ch}"] = dch
                cd[f"p_T_{ch}"] = float(pch[0])
            # loop 有功效（n=272）-> 可判显著；static 无功效（n=105）-> 只看符号
            cd["T4_loop_load"] = bool(cd["delta_T_loop"] > 0 and cd["p_T_loop"] < P_ALPHA)
            cd["T5_static_sign"] = bool(cd["delta_T_static"] > 0)
            cd["T5_note"] = "static 通道 n=105 功效 0.36，明令不作显著性判定"
            # F5：既有基线自己打不打得过时钟
            cd["delta_C_minus_T"], _ = pooled(C, T)
            # S7 长度匹配子集（描述性，无显著性判据）
            cd["S7_delta_T"], _ = pooled(det_s7[a], det_s7[a + "__T"])
            cd["S7_delta_C"], _ = pooled(det_s7[a], det_s7[a + "__C"])
            cd["beats_T"] = all(cd[k] for k in ("T1", "T2", "T3"))
        else:
            cd |= {k: None for k in ("delta_T_pooled", "delta_T_group", "p_maxT_F1_T",
                                     "delta_T_loop", "delta_T_static", "delta_C_minus_T",
                                     "S7_delta_T", "S7_delta_C")}
            cd |= dict(T1=False, T2=False, T3=False, T4_loop_load=False,
                       T5_static_sign=False, beats_T=False,
                       T_missing="无对照臂 T -> MoE 载荷主张作废（§4.3 情形 7）")
        # ---- AMD-1 的 G3 只留在附录 A 表里（§3A.2 已结构性取代）
        rows_out[a]["gate_G3_appendixA"] = bool(cd["beats_T"])
        cd["W4"] = bool(rows_out[a]["gate_G1"] and rows_out[a]["gate_G2"])
        cd["beats_C_appendixA"] = all(cd[k] for k in ("W1", "W2", "W3", "W4", "W5"))

        # ================= AMD-2：主判决 =================
        cd["MH_loop"] = MH[a]["loop"]["MH"]
        cd["MH_loop_sign"] = MH[a]["loop"]["sign"]
        cd["MH_loop_group"] = float(obs_m[j])
        cd["p_maxT_F1_M"] = float(p_m[j])
        cd["MH_loop_minus_C"] = float(np.nanmean(list(mc_strata[j].values()))
                                      if mc_strata[j] else np.nan)
        cd["MH_loop_minus_C_group"] = float(obs_mc[j])
        cd["p_maxT_F1_MC"] = float(p_mc[j])
        cd["MH_static"] = MH[a]["static"]["MH"]
        cd["MH_static_sign"] = MH[a]["static"]["sign"]
        cd["MH_static_note"] = "视界上 static 事件只剩 42->21，明令只报不测"
        cd["M1"] = bool(cd["MH_loop"] >= MH_MARGIN_MIN)
        cd["M2"] = bool(cd["p_maxT_F1_M"] < P_ALPHA)
        cd["M3"] = bool(cd["MH_loop_sign"] >= MH_SIGN_MIN)
        cd["M4"] = bool(cd["MH_loop_minus_C"] >= MH_MARGIN_MIN
                        and cd["p_maxT_F1_MC"] < P_ALPHA)
        cd["M5"] = cd["W4"]
        cd["M6"] = cd["W5"]
        cd["V_ok"] = bool(validity[a]["ok"])
        cd["MH_admissible"] = bool(cd["M1"] and cd["M2"] and cd["M3"] and cd["M5"])
        cd["verdict_pass"] = bool(cd["V_ok"] and cd["M1"] and cd["M2"] and cd["M3"]
                                  and cd["M4"] and cd["M5"] and cd["M6"])
        rows_out[a]["MH_admissible"] = cd["MH_admissible"]
        rows_out[a]["admissible"] = cd["MH_admissible"]

        # ---- AMD-2 §5A.1 前缀截断端点（确证性次端点，族 F_P）
        cd["prefix_delta_C"], _ = pooled(det_pfx[a], det_pfx[a + "__C"])
        if has_T[a]:
            cd["prefix_delta_T"], _ = pooled(det_pfx[a], det_pfx[a + "__T"])
        cd["P1"] = bool(cd["prefix_delta_C"] >= PREFIX_DELTA_MIN)
        cd["P3"] = bool(cd.get("prefix_delta_T", float("nan")) > 0)
        contrasts.append(cd)

    # F_P：前缀端点的 vs-C 配对差，组级 sign-flip maxT（3 格）
    pfx_pairs = [(a, det_pfx[a], det_pfx[a + "__C"]) for a in arm_dirs]
    _, p_pfx = groupwise_signflip_maxt(group_effects(pfx_pairs), NPERM,
                                       np.random.default_rng(SEED))
    for j, cd in enumerate(contrasts):
        cd["p_maxT_F_P"] = float(p_pfx[j])
        cd["P2"] = bool(cd["p_maxT_F_P"] < P_ALPHA)

    for lab in labels:                       # C / T 行没有闸，标 None 以免误读
        rows_out[lab].setdefault("gate_G3_appendixA", None)
        rows_out[lab].setdefault("MH_admissible", None)
        rows_out[lab].setdefault("admissible", None)

    winner = select_winner(rows_out, contrasts, arm_dirs)
    verdict = declare_verdict(rows_out, contrasts, arm_dirs, winner, validity)

    os.makedirs(RESULTS, exist_ok=True)
    prim = ["arm", "det_n", "det_k", "det_rate", "fa_n", "fa_k", "fa_rate", "fa_ci_lo",
            "fa_ci_hi", "fa_worst_cell", "fa_worst_cell_rate", "lead50", "lead_q25",
            "lead_q75", "mistype_rate",
            "MH_loop", "MH_loop_sign", "MH_loop_strata", "MH_static", "MH_static_sign",
            "MH_macro", "prefix_n", "prefix_det_rate", "Type_TypeAcc_macro",
            "S7_n", "S7_det_rate",
            "gate_G1", "gate_G2", "gate_G3_appendixA", "MH_admissible", "admissible"]
    sec = [k for k in sorted(rows_out[labels[0]]) if k not in prim]
    for fn, cols in (("metrics_primary.csv", prim), ("metrics_secondary.csv", ["arm"] + sec)):
        with open(os.path.join(RESULTS, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for lab in labels:
                w.writerow(rows_out[lab])
    json.dump(dict(contrasts=contrasts, winner=winner, verdict=verdict,
                   seed=SEED, nperm=NPERM,
                   families=dict(
                       F1_M=len(arm_dirs), F1_MC=len(arm_dirs), F_P=len(arm_dirs),
                       F_V=len(armsT),
                       F1_C=len(arm_dirs), F1_T=len(armsT), F4=2 * len(armsT),
                       F5=len(armsT),
                       decision="F1_M + F1_MC（AMD-2 主判决）；F_P 确证性；其余附录/描述",
                       note="各族族内 maxT；合取按 IUT 不跨族校正（z_c≈2.39 不变）"),
                   amd1="集长平凡对照臂 T 已启用（HOLDOUT_PROTOCOL §3.5）",
                   amd2="主端点 = 匹配视界 margin（§3A）；全集 Det 降级为附录 A（§3A.6）"),
              open(os.path.join(RESULTS, "arbitration.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(winner=winner, verdict=verdict, contrasts=contrasts),
                     indent=1, ensure_ascii=False))
    print("\n[--score] 已写 metrics_primary.csv / arbitration.json。"
          "\n[--score] TODO(人工)：按 RUNNER.md §6 补全 RESULTS.md 的七项必含内容"
          "（六条负结果勾选表、全员披露、第二名与偏倚声明、FA 工作点警示、"
          "同步型措辞纪律、S1 声明、事后功效）。")
    return rows_out, contrasts, winner, verdict


def select_winner(rows_out, contrasts, arm_dirs):
    """HOLDOUT_PROTOCOL.md §6 R-B：唯一的、预先指定的确定性选择函数。

    AMD-2：排序量 = `MH_loop`（匹配视界 margin）；
    可采性 = G1 ∧ G2 ∧ M1 ∧ M2 ∧ M3。V 闸不过时不产生 winner（情形 0）。
    """
    by = {c["arm"]: c for c in contrasts}
    if any(not by[a]["V_ok"] for a in arm_dirs):
        return None                                   # §4.3 情形 0：整次运行作废
    adm = [a for a in arm_dirs if by[a]["MH_admissible"]]
    if not adm:
        return None
    return sorted(adm, key=lambda a: (-by[a]["MH_loop"], -by[a]["MH_loop_minus_C"],
                                      rows_out[a]["fa_rate"], a))[0]


def declare_verdict(rows_out, contrasts, arm_dirs, winner, validity=None):
    """HOLDOUT_PROTOCOL.md §4.3 的六条出口，逐条勾选。"""
    by = {c["arm"]: c for c in contrasts}
    always = [
        "留出集曾参与 E4/E7/E8/E9 的聚合统计（HOLDOUT_PROTOCOL §0.1），"
        "绝对 Det/FA/lead 为上界估计，不是干净的泛化估计。",
        "留出集 43 个任务全部为 proxy_grade=objaware，评估等级低于校准集（full）。",
        "grid/object（31 个有锚事件）与 main（57 个）为一致性格，"
        "其上的任何 p 值都不得用于支持或反驳主张。",
    ]
    # 情形 8/9 与 winner 无关，先逐臂扫一遍（AMD-1）
    c8 = [c["arm"] for c in contrasts
          if c.get("delta_C_minus_T") is not None and c["delta_C_minus_T"] <= 0]
    c9 = [c["arm"] for c in contrasts if not c.get("T4_loop_load")]
    v = {"case1_no_admissible": winner is None,
         "case8_baseline_C_loses_to_clock": c8,
         "case9_loop_channel_load_void": c9,
         "case0_matched_horizon_invalid": [a for a, x in (validity or {}).items()
                                           if not x["ok"]],
         "case10_no_above_diagonal_margin": [c["arm"] for c in contrasts
                                             if not (c["M1"] and c["M2"] and c["M3"])]}
    # 情形 0 压倒一切：口径坏了，不产生任何结论
    if v["case0_matched_horizon_invalid"]:
        v["headline"] = (
            "匹配视界口径实现有误，本次运行作废（臂 T 的 margin 偏离构造性零值："
            f"{v['case0_matched_horizon_invalid']}）——不出任何结论，修复后全部重跑")
        v["mandatory_notes"] = _amd2_notes(v, contrasts, rows_out) + always
        return v
    if winner is None:
        allvoid = len(v["case10_no_above_diagonal_margin"]) == len(contrasts)
        v["headline"] = (
            "本轮未能证明 MoE 载荷：在匹配视界（同一绝对 q）上，检测器对事件集的报警率"
            "未显著高于对干净成功集的报警率" if allvoid else
            "MoE 表型检测器未通过可采性（G1/G2）与匹配视界判据的合取")
        v["mandatory_notes"] = _amd2_notes(v, contrasts, rows_out) + always
        return v
    w = by[winner]
    v |= {
        "case2_MH_vs_C_below_threshold": not w["M4"],
        "case4_replication_failed": w["M1"] and not w["M6"],
        "case6_underpowered": (w["disc_rate"] > DISC_UNDERPOWERED and not w["M2"]),
        "case7_beaten_by_clock_appendixA": not w["beats_T"],
    }
    if not (w["M1"] and w["M2"] and w["M3"]):     # 情形 10 优先
        v["headline"] = ("本轮未能证明 MoE 载荷：匹配视界 margin 未达门槛"
                         f"（MH_loop={w['MH_loop']:.3f}, p={w['p_maxT_F1_M']:.4f}, "
                         f"符号 {w['MH_loop_sign']}/9）")
    elif v["case2_MH_vs_C_below_threshold"]:
        v["headline"] = ("MoE 表型检测器未超越既有 mob1_w8 基线（匹配视界口径："
                         f"ΔMH={w['MH_loop_minus_C']:.3f}, p={w['p_maxT_F1_MC']:.4f}）")
    elif v["case4_replication_failed"]:
        v["headline"] = "匹配视界 margin 达标但未跨语料/跨套件复现（R1 或 R2 不过）"
    elif v["case6_underpowered"]:
        v["headline"] = ("功效不足，不作结论（不一致率 "
                         f"{w['disc_rate']:.2f} > {DISC_UNDERPOWERED}）")
    else:
        v["headline"] = (f"{winner} 在留出集的匹配视界口径上立住："
                         f"MH_loop={w['MH_loop']:.3f}（{w['MH_loop_sign']}/9），"
                         f"较 mob1_w8 消融臂 +{w['MH_loop_minus_C']:.3f}；"
                         "纯 q 对照臂 T 的 margin 已构造性归零")
    v["mandatory_notes"] = (_mandatory_notes(rows_out, w, winner)
                            + _amd2_notes(v, contrasts, rows_out) + always)
    return v


def _amd2_notes(v, contrasts, rows_out):
    """AMD-2 的强制措辞（§3A.5 / §3A.6 / §5A）。"""
    n = ["全集 Det 口径已降级为附录 A：该口径受集长混杂支配"
         "（纯 q 对照臂在其上 det loop 可达 1.000/0.645；留出集 70.8% 的致败 onset "
         "发生在该集已长于 90% 成功集之后），**不可作 MoE 载荷主张**，仅供运营参考。",
         "主端点为匹配视界 margin（同一绝对 q 上事件集与干净成功集的报警率之差）；"
         "任何纯 q 报警器的 margin 构造性恒为 0，臂 T 的实测值是该口径的有效性证明。",
         "符号计数 M3（≥7/9）不是检验：零假设下 P(≥7/9 为正)=46/512=0.0898。",
         "分型端点用宏平均：本留出集上『恒判 loop』的微平均是 0.819、宏平均 0.500，"
         "微平均结论一律作废。"]
    for c in contrasts:
        if c.get("MH_static") is not None:
            n.append(f"{c['arm']} static 通道 MH={c['MH_static']}"
                     f"（{c.get('MH_static_sign')}/9）：视界上事件只剩 42→21，"
                     "按协议只报不测。")
        if c.get("P1") is not None and not c["P1"]:
            n.append(f"{c['arm']} 未过前缀截断端点（Δ={c.get('prefix_delta_C')}"
                     f" < {PREFIX_DELTA_MIN}，n=97）：第二条独立证据线未支持主端点。")
        if c.get("P3") is False:
            n.append(f"{c['arm']} 在前缀内未胜过纯 q 对照臂 T——外部实测 T 在前缀内 det 恒为 0，"
                     "此条不成立说明前缀实现或 T 构造有问题，须复核。")
    return n + _amd1_notes(v, contrasts, rows_out)


def _amd1_notes(v, contrasts, rows_out):
    """AMD-1 的强制措辞（§3.3b / §5.2 / §4.3 情形 8-9）。"""
    n = ["留出集中 70.8% 的致败事件（235/332），其物理 onset 发生在该集已长于 90% 成功集之后；"
         "本评估集在结构上无法把这部分检出与『看钟』分开。",
         "对照臂 T 的报警按『通道无关』计分（对任一通道窗口都算命中），"
         "这是有意给地板对照的优待，Det(T) 因此是 T 能力的上界。"]
    for a in v.get("case9_loop_channel_load_void", []):
        n.append(f"{a} 的 loop 通道未证明 MoE 载荷：集长对照臂 T 在本通道上不劣于本检测器，"
                 "本通道的检出可由『失败集更长』单独解释。")
    for a in v.get("case8_baseline_C_loses_to_clock", []):
        n.append(f"{a} 的既有基线 mob1_w8 本身也未超越集长对照（Det(C)−Det(T) ≤ 0），"
                 "本轮『打赢 C』的门槛不构成 MoE 载荷证据。")
    for c in contrasts:
        s7, dt = c.get("S7_delta_T"), c.get("delta_T_pooled")
        if s7 is not None and dt is not None and s7 <= 0 < dt:
            n.append(f"{c['arm']} 相对集长对照的优势集中在『该集已明显超长』之后的区段；"
                     f"在集长信息尚不可得的 97 个集上未观察到相对时钟的优势"
                     f"（Δ_T,S7={s7:.3f}）。")
        if c.get("T5_static_sign") is not None:
            n.append(f"{c['arm']} 的 static 通道 Δ_T={c.get('delta_T_static')}："
                     "n=105、功效 0.36，按协议只报符号，不作显著性判定。")
    return n


def _mandatory_notes(rows_out, w, winner):
    notes = []
    fa_a, fa_c = rows_out[winner]["fa_rate"], rows_out[winner + "__C"]["fa_rate"]
    if abs(fa_a - fa_c) > 0.01:
        notes.append(f"本次 Δ 同时包含召回差与工作点差；两臂工作点不同（FA_arm={fa_a:.4f}, "
                     f"FA_C={fa_c:.4f}），主张只在各自冻结工作点上成立，"
                     "不等价于同 FA 下的召回优势。")
    l50 = rows_out[winner]["lead50"]
    if 0 <= l50 <= LEAD_GATE:
        notes.append("本臂为「同步型 (concurrent)」，全文禁用「提前 / 预警 / early warning」。")
    notes.append("winner 由 argmax over 3 得到，效应量存在向上偏倚；maxT 处理了推断，"
                 "未处理效应量偏倚（见 S5 交叉拟合）。")
    if w["disc_rate"] > 0.40:
        notes.append(f"实际不一致率 {w['disc_rate']:.2f} > 0.40，"
                     "本次对比的事后功效低于预注册假设。")
    s1 = rows_out[winner]["S1_delta"]
    if w["beats_C"] and not (s1 >= S1_DELTA_MIN):
        notes.append("本检测器检出的是『进入异常区』而非『不可逆』；"
                     "REPORT.zh.md §三的『能否离开』结论未在检测器层面兑现"
                     f"（S1_delta={s1:.3f} < {S1_DELTA_MIN}）。")
    if (not w["beats_C"]) and s1 >= S1_DELTA_MIN:
        notes.append(f"主端点为负结果，但次端点 S1 为正（S1_delta={s1:.3f}）；"
                     "两句必须同时写出，不得只写其一（HOLDOUT_PROTOCOL §5.1）。")
    return notes


# ================================================================ CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true", help="校准集探针 P0-P7")
    ap.add_argument("--run", action="store_true", help="留出集执行，只落 alarm 明细")
    ap.add_argument("--score", action="store_true", help="评分并写指标表")
    ap.add_argument("--arms", nargs="+",
                    default=["A_threshold_fsm", "B_macrostate_dwell", "D_freeform"])
    ap.add_argument("--confirm-all-specs-frozen", action="store_true")
    ap.add_argument("--all-arms-present", action="store_true")
    ap.add_argument("--inventory", action="store_true", help="只打印留出集规模构成")
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v) != "1":
            sys.exit(f"请先 export {v}=1（240 核机，不设会慢 100 倍）")

    if a.inventory:
        inv = json.load(open(os.path.join(HERE, "holdout_inventory.json")))
        print(json.dumps(inv["holdout_total"], indent=1, ensure_ascii=False))
        return

    if a.preflight:
        allok, reps = True, []
        for arm in a.arms:
            print(f"\n=== preflight {arm} ===")
            try:
                ok, rep = probe_arm(load_arm(arm), arm)
            except Exception as e:                       # noqa: BLE001
                ok, rep = False, {"arm": arm, "ok": False, "error": repr(e)}
                print(f"  [FAIL] 加载/执行异常: {e!r}")
            allok &= ok
            reps.append(rep)
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(reps, open(os.path.join(RESULTS, "probes.json"), "w"),
                  indent=1, ensure_ascii=False)
        print(f"\npreflight {'PASS' if allok else 'FAIL'} -> results/probes.json")
        sys.exit(0 if allok else 1)

    if a.run:
        if not a.confirm_all_specs_frozen:
            sys.exit("留出集只允许跑一次：需 --confirm-all-specs-frozen，"
                     "且三套 SPEC.md 必须已冻结（HOLDOUT_PROTOCOL §6 R-A）。")
        if os.path.exists(os.path.join(RESULTS, "MANIFEST.sha256")):
            sys.exit("已存在 results/MANIFEST.sha256：重跑须先走 AMENDMENTS.md 程序，"
                     "并把旧目录改名为 results_amd<n>/（§7.2）。")
        run_arms(a.arms)
        return

    if a.score:
        if not a.all_arms_present:
            sys.exit("需 --all-arms-present：四臂明细齐全才允许一次性评分（§6 R-A）。")
        score(a.arms)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
