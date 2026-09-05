"""C4 — 进了 Trap 之后是"待着"还是"走开"？—— 驻留率、逃逸风险、以及 routing 能否预告逃逸。

事件表只记 onset，驻留要重算。这里把冻结的 onset 规则改写成**逐 query 指示函数**
（阈值逐字未动，见 events/build_events.py）：

  in_loop(r)   : ∃ l<r−2 同时满足 eef/obj/grip 接近 + 路径长 ≥0.120 —— "此刻回到了走过的位形"
  in_static(w) : 宽 2 窗的 eef/obj/grip 步长同时低于阈值 —— 映射到 query w+2

第一版用"首次连续 E 个 0"当逃逸，中位驻留恒为 1 —— 因为 in_loop 是**瞬时事件**指示
而不是"当前卡住"的状态。改用驻留率：

  occ_W = mean(in_loop[onset+1 : onset+1+W])    前向**等窗**占用率
  in_trap_W(q) = 中心窗占用 ≥ 0.5               再由它给出真正的驻留段与逃逸时刻

三处必须控制的混杂（不控制就没有结论）：
1. **等窗**：致败集天然更长，用"onset 到集末"的占用率会直接读出集长。所有比较限定
   onset+W < n_queries 的集，双方给同样的 W 个 query 机会。
2. **onset 时刻**：自恢复 loop 的中位 onset=8，致败=31，不是同一批东西。按 onset 分箱
   分层，并另给同 cell（同初始状态）同胞的配对比较。
3. **指示函数自身的上漂**：in_loop 要求累计路径 ≥0.120，候选 l 随 q 单调增多。故给出
   背景率 = 无 onset 的成功集在同一绝对 q 上的命中率。

观测性限制（结论旁必须同时写）：hazard 随 age 下降可以纯由 frailty（易逃的先逃、
留下粘的）造成，不等于"陷得越久越难出"；那是介入问题，只能由在线实验回答。
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
RUNS = {
    "main16x32": (HUB / "cache/HiMoE-VLA", "right-16x32"),
    "grid50x8": (HUB / "cache_new/HiMoE-VLA", "right-50x8-20260903"),
}
LOOP_EEF, LOOP_OBJ, LOOP_GRIP, LOOP_PATH = 0.045, 0.030, 0.012, 0.120
STATIC_EEF, STATIC_OBJ, STATIC_GRIP, STATIC_WIDTH = 0.020, 0.005, 0.001, 2
SCENE8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
WINDOWS = (3, 5, 10)
TRAP_W = 5  # in_trap 的中心窗宽


def indicators(eef, objects, gripper):
    n = len(eef)
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(step)]
    in_loop = np.zeros(n, bool)
    for right in range(3, n):
        lo = slice(0, right - 2)
        ok = ((np.linalg.norm(eef[right] - eef[lo], axis=1) <= LOOP_EEF)
              & (np.linalg.norm(objects[right] - objects[lo], axis=2).max(axis=1) <= LOOP_OBJ)
              & (np.abs(gripper[right] - gripper[lo]) <= LOOP_GRIP)
              & (cum[right] - cum[lo] >= LOOP_PATH))
        in_loop[right] = bool(ok.any())
    obj_step = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    grip_step = np.abs(np.diff(gripper))
    in_static = np.zeros(n, bool)
    if len(step) >= STATIC_WIDTH:
        k = np.ones(STATIC_WIDTH)
        w = ((np.convolve(step, k, "valid") <= STATIC_EEF)
             & (np.convolve(obj_step, k, "valid") <= STATIC_OBJ)
             & (np.convolve(grip_step, k, "valid") <= STATIC_GRIP))
        idx = np.arange(len(w)) + STATIC_WIDTH
        m = idx < n
        in_static[idx[m]] = w[m]
    return in_loop, in_static


def trap_state(ind, w=TRAP_W):
    """中心窗占用 ≥0.5 的状态序列。"""
    k = np.ones(w)
    pad = np.r_[np.zeros(w // 2, bool), ind, np.zeros(w // 2, bool)]
    return (np.convolve(pad.astype(float), k, "valid")[: len(ind)] / w) >= 0.5


def exit_time(state, onset):
    """onset 起首次离开 in_trap 状态；未离开返回 None（右删失）。"""
    n = len(state)
    q = onset
    while q < n and not state[q]:
        q += 1  # onset 处窗口可能还没成型，先找到真正进入
    if q >= n:
        return None, 0
    entered = q
    while q < n and state[q]:
        q += 1
    return (None, n - 1 - entered) if q >= n else (q - entered, q - entered)


def episode_arrays(run, e_id, slots):
    with np.load(run / "client" / f"episode_{e_id:02d}.npz") as d:
        state = np.asarray(d["state"], np.float64)
        sim = np.asarray(d["sim_state"], np.float64)
    return state[:, :3], np.stack([sim[:, lo:lo + 3] for lo in slots], axis=1), state[:, 6:8].mean(1)


def object_slots(run, task):
    if task == SCENE8:
        return [10, 17]
    lay = json.load(open(run / "client" / "sim_layout.json"))
    return [j["state_lo"] for j in lay["joints"]
            if not j["is_robot"] and j["state_hi"] - j["state_lo"] == 7]


def collect(corpus):
    root, run_id = RUNS[corpus]
    recs = []
    bg_hit = np.zeros(120)
    bg_n = np.zeros(120)
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=False)
        run = root / suite / task / run_id
        slots = object_slots(run, task)
        eps, Y, nq = C.outcomes(t)
        _, sc, _ = C.episode_cells(t)
        # onset 处的 routing 特征，用于"逃逸能否被 MoE 预告"
        featmap = {}
        for f in ("late_flow_volatility", "route_acceleration", "gate_entropy",
                  "token_consensus", "token_dispersion", "mob1_w8"):
            if f in t.feats:
                featmap[f] = {}
                for e, q, v in zip(t.ep, t.q, t.feats[f]):
                    featmap[f][(int(e), int(q))] = float(v)
        for i, e in enumerate(eps):
            ev = t.ev[e]
            try:
                eef, objs, grip = episode_arrays(run, e, slots)
            except FileNotFoundError:
                continue
            il, ist = indicators(eef, objs, grip)
            n = len(il)
            # 背景率：**全部成功集**在绝对 q 上的 in_loop 命中率。不能限定"无 onset"——
            # onset 就定义为首次命中，那样背景恒等于 0。
            if ev["success"]:
                m = min(n, 120)
                bg_hit[:m] += il[:m]
                bg_n[:m] += 1
            for kind, ind, onset in (("loop", il, ev["loop_onset_q"]),
                                     ("static", ist, ev["static_onset_q"])):
                if onset < 0 or (kind == "loop" and not t.loop_valid):
                    continue
                st = trap_state(ind)
                dur, dur_obs = exit_time(st, onset)
                r = dict(corpus=corpus, suite=suite, task=task, scene=int(sc[i]),
                         episode=int(e), kind=kind, onset=int(onset), n_queries=int(n),
                         success=int(ev["success"]), remaining=int(n - 1 - onset),
                         dwell=int(dur_obs), escaped=int(dur is not None),
                         occ_to_end=float(ind[onset + 1:].mean()) if onset + 1 < n else np.nan)
                for W in WINDOWS:
                    r[f"occ{W}"] = float(ind[onset + 1: onset + 1 + W].mean()) if onset + W < n else np.nan
                for f, mp in featmap.items():
                    r[f"sig_{f}"] = mp.get((int(e), int(onset)), np.nan)
                recs.append(r)
    return recs, bg_hit / np.maximum(bg_n, 1), bg_n


def matched_within_cell(recs, kind, W):
    """同 cell（同 task+scene）内，自恢复 vs 致败的等窗占用配对比较。"""
    key = f"occ{W}"
    by = {}
    for r in recs:
        if r["kind"] != kind or not np.isfinite(r.get(key, np.nan)):
            continue
        by.setdefault((r["task"], r["scene"]), []).append(r)
    diffs, pairs, cells = [], 0, 0
    for cell, rs in by.items():
        s = [r[key] for r in rs if r["success"]]
        f = [r[key] for r in rs if not r["success"]]
        if not s or not f:
            continue
        cells += 1
        pairs += len(s) * len(f)
        diffs.append(np.mean(f) - np.mean(s))
    if not diffs:
        return None
    d = np.array(diffs)
    # cell 级符号检验
    from scipy import stats
    pos = int((d > 0).sum())
    p = stats.binomtest(max(pos, len(d) - pos), len(d), 0.5).pvalue
    return dict(n_cells=cells, n_pairs=pairs, mean_diff=float(d.mean()),
                median_diff=float(np.median(d)), n_pos=pos, p_sign=float(p))


def by_onset_bin(recs, kind, W, edges=(0, 10, 20, 30, 40, 100)):
    key = f"occ{W}"
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in recs if r["kind"] == kind and a <= r["onset"] < b
               and np.isfinite(r.get(key, np.nan))]
        s = [r[key] for r in sel if r["success"]]
        f = [r[key] for r in sel if not r["success"]]
        if not s or not f:
            continue
        out.append(dict(onset_bin=f"[{a},{b})", n_success=len(s), n_fail=len(f),
                        occ_success=float(np.mean(s)), occ_fail=float(np.mean(f)),
                        diff=float(np.mean(f) - np.mean(s))))
    return out


def escape_auc(recs, kind):
    """在**进了 trap 的同胞**里，onset 处的 routing 能否预告"这一条会逃出去"？

    组内单位 = cell；正类 = 未逃逸/致败。
    """
    sub = [r for r in recs if r["kind"] == kind]
    cells = np.array([f"{r['task']}#{r['scene']}" for r in sub])
    y = np.array([1 - r["success"] for r in sub], np.int8)
    res = []
    for f in ("late_flow_volatility", "route_acceleration", "gate_entropy",
              "token_consensus", "token_dispersion", "mob1_w8"):
        v = np.array([r.get(f"sig_{f}", np.nan) for r in sub], float)
        m = np.isfinite(v)
        if m.sum() < 20:
            continue
        auc, npair, per = C.stratified_auc(v[m], y[m], cells[m])
        if npair == 0:
            continue
        rng = np.random.default_rng(20260904)
        null = C.perm_null_auc(v[m], y[m], cells[m], 2000, rng)
        p = (np.sum(np.abs(null - .5) >= abs(auc - .5)) + 1) / (len(null) + 1)
        res.append(dict(feature=f, auc=float(auc), n_pairs=npair,
                        n_cells=len(per), p_perm=float(p)))
    res.sort(key=lambda r: -abs(r["auc"] - .5))
    return res


def km(durations, events):
    ks = np.arange(0, int(max(durations, default=0)) + 1)
    S, H, R = [], [], []
    s = 1.0
    for k in ks:
        at_risk = int(np.sum(durations >= k))
        d = int(np.sum((durations == k) & events))
        h = d / at_risk if at_risk else np.nan
        s *= (1 - h) if at_risk else 1.0
        S.append(s); H.append(h); R.append(at_risk)
    return ks, np.array(S), np.array(H), np.array(R)


def main(corpus="main16x32"):
    recs, bg, bgn = collect(corpus)
    out = dict(corpus=corpus, n_records=len(recs),
               background_in_loop_rate=[float(v) for v in bg[:60]],
               background_n=[int(v) for v in bgn[:60]])
    print(f"### {corpus}  记录 {len(recs)} 条（loop+static onset）\n")
    for kind in ("loop", "static"):
        sub = [r for r in recs if r["kind"] == kind]
        if not sub:
            continue
        blk = dict(n=len(sub))
        print(f"--- {kind}  n={len(sub)}  "
              f"(成功 {sum(r['success'] for r in sub)} / 失败 {sum(1-r['success'] for r in sub)})")
        print(f"  {'W':>3s} {'n_succ':>6s} {'n_fail':>6s} {'occ_succ':>9s} {'occ_fail':>9s} {'diff':>7s}")
        blk["equal_window"] = {}
        for W in WINDOWS:
            s = [r[f"occ{W}"] for r in sub if r["success"] and np.isfinite(r[f"occ{W}"])]
            f = [r[f"occ{W}"] for r in sub if not r["success"] and np.isfinite(r[f"occ{W}"])]
            if not s or not f:
                continue
            blk["equal_window"][W] = dict(n_success=len(s), n_fail=len(f),
                                          occ_success=float(np.mean(s)),
                                          occ_fail=float(np.mean(f)),
                                          diff=float(np.mean(f) - np.mean(s)))
            print(f"  {W:3d} {len(s):6d} {len(f):6d} {np.mean(s):9.3f} {np.mean(f):9.3f} "
                  f"{np.mean(f)-np.mean(s):7.3f}")
        blk["by_onset_bin"] = {W: by_onset_bin(recs, kind, W) for W in WINDOWS}
        print("  按 onset 分箱（W=5）:")
        for r in blk["by_onset_bin"][5]:
            print(f"    onset{r['onset_bin']:9s} n={r['n_success']:3d}/{r['n_fail']:3d} "
                  f"occ {r['occ_success']:.3f} vs {r['occ_fail']:.3f}  diff={r['diff']:+.3f}")
        blk["matched_within_cell"] = {W: matched_within_cell(recs, kind, W) for W in WINDOWS}
        m = blk["matched_within_cell"][5]
        if m:
            print(f"  同 cell 配对（W=5）: {m['n_cells']} 格, diff={m['mean_diff']:+.3f}, "
                  f"{m['n_pos']}/{m['n_cells']} 同向, p_sign={m['p_sign']:.4f}")
        d = np.array([r["dwell"] for r in sub])
        ev = np.array([r["escaped"] for r in sub], bool)
        ks, S, H, R = km(d, ev)
        blk["km"] = dict(k=[int(x) for x in ks[:25]], S=[float(x) for x in S[:25]],
                         hazard=[float(x) for x in H[:25]], at_risk=[int(x) for x in R[:25]])
        print(f"  in_trap(W={TRAP_W}) 驻留：中位 {np.median(d):.1f} query，逃逸观测率 {ev.mean():.3f}")
        print(f"    hazard(k=0..7) = " + " ".join(f"{h:.3f}" for h in H[:8]))
        print(f"    at_risk        = " + " ".join(f"{r:5d}" for r in R[:8]))
        blk["escape_auc"] = escape_auc(recs, kind)
        print("  onset 处 routing → 能否预告逃逸（组内 AUC，正类=致败）:")
        for r in blk["escape_auc"][:4]:
            print(f"    {r['feature']:22s} AUC={r['auc']:.4f} p={r['p_perm']:.4f} "
                  f"(cells={r['n_cells']}, pairs={r['n_pairs']})")
        out[kind] = blk
        print()
    print("背景 in_loop 命中率（无 onset 的成功集，按绝对 q）:")
    print("  q= " + " ".join(f"{q:5d}" for q in range(0, 40, 5)))
    print("  r= " + " ".join(f"{bg[q]:5.3f}" for q in range(0, 40, 5)))
    OUT.mkdir(exist_ok=True)
    out["records"] = recs
    C.jdump(out, OUT / f"c4_dwell_{corpus}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main16x32")
