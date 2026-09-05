"""C7 — 单轨迹口径：条件首达概率 c^(h)(H_q) 的估计、校准、与固定误报率下的提前量。

前面 C1–C5 回答的是框架 §7 的 β_B/β_W（同 snapshot 变噪声）。研究对象若是**单条轨迹的
在线判断**，那问题应当写成：

    c^(h)(H_q^R) = P( τ_trap ≤ q+h | 这条轨迹到 q 为止的 routing 历史 )

也就是"在风险集上的条件首达概率"。它不需要 fork —— 部署时你本来就只有一条轨迹。

与既有工作的分工（都查过，不重复）：
  · `analysis_signal_matrix_trainfree`：八个 train-free 信号的固定时点 AUC + onset 对齐，
    **显式声明不拟合任何预测器**，所以没有概率、没有校准、没有提前量。
  · `analysis_future`：预测的是物理未来（哪个壶动），不是首达事件。
  · `analysis_cusum`：CUSUM/SPRT 序贯检测，是本脚本报警臂的近亲，但用的是单信号。

本脚本补三件：

1. **概率而非排序**：给出可靠性曲线（预测 c 对实际频率），Brier、ECE。框架的中心主张
   就是 AUC 只保证排序、committor 要求校准。
2. **elapsed-time 由分层承担，不由模型扣**：routing 强烈编码绝对 query 序号
   （analysis_mdp: NMI(state,t) 达 0.54），集长本身 oracle AUC≈0.99。本工作区既有口径是
   **在同一绝对 query 上取无事件轨迹作对照**（signal_matrix README；control 红线同）。
   匹配比把时钟当协变量回归掉更干净，所以这里**按绝对 q 分层**：每个 q 内部
   只用 routing 打分、只在该 q 内评估。模型不含任何时钟特征。
3. **固定误报率下的提前量**：阈值在**干净成功集**上按集级误报率 5%/10% 校准，
   再读被困集的 lead = onset − 首次报警 q。表型报告已指出 z>2 单脉冲在干净成功集
   触发率 24–81%/集不可用；这里给出可用阈值下的实际提前量。

严格因果：所有特征只用 q、q−1、q−2。CV = **留一 task**（routing 100% 编码任务身份，
不留出任务就是在读任务先验）。指标在留出 task 内计算再等权合并。集为聚类单元。
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
SUITE_CAP = {"libero_spatial": 22, "libero_object": 28, "libero_goal": 30, "libero_long": 52}
Z8 = ["late_flow_volatility", "route_acceleration", "gate_entropy", "top12_margin",
      "token_consensus", "token_dispersion", "layer_disagreement", "state_action_gap"]


def build_rows(corpus, horizon):
    """风险集上的 (episode,q) 行 + 严格因果特征 + 竞争风险标签。"""
    X_r, Y_t, Y_d, EP, TASK, QQ, ONS, SUC, SCN = [], [], [], [], [], [], [], [], []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=False)
        eps, _, _ = C.outcomes(t)
        _, scn_of, _ = C.episode_cells(t)
        scn_map = {int(e): int(scn_of[i]) for i, e in enumerate(eps)}
        # 逐 episode 组装序列
        for e in eps:
            ev = t.ev[e]
            onset = ev["trap_onset_q"]
            if onset >= 0 and ev["loop_onset_q"] == onset and not t.loop_valid:
                onset = ev["static_onset_q"]  # 铰接盲区：loop 通道无效，退回 static
            m = t.ep == e
            q = t.q[m]
            if len(q) < 3:
                continue
            Z = np.column_stack([t.feats[f][m] for f in Z8])
            mob = t.feats["mob1_w8"][m] if "mob1_w8" in t.feats else np.zeros(len(q))
            n = len(q)
            done_q = n - 1 if ev["success"] else None
            for i in range(2, n):
                qi = int(q[i])
                # 风险集：尚未发生 trap，且还没结束
                if onset >= 0 and qi >= onset:
                    break
                z, z1, z2 = Z[i], Z[i - 1], Z[i - 2]
                w = Z[max(0, i - 3):i + 1]
                feat = np.concatenate([
                    z, z - z1, z - 2 * z1 + z2, w.mean(0), w.std(0),
                    [mob[i], mob[i] - mob[i - 1]],
                ])
                X_r.append(feat)
                Y_t.append(int(onset >= 0 and 1 <= onset - qi <= horizon))
                Y_d.append(int(done_q is not None and 1 <= done_q - qi <= horizon))
                EP.append(f"{task}#{int(e)}")
                TASK.append(f"{suite}/{task}")
                QQ.append(qi)
                ONS.append(onset)
                SUC.append(int(ev["success"]))
                SCN.append(scn_map[int(e)])
    return (np.asarray(X_r, np.float64),
            np.asarray(Y_t, np.int8), np.asarray(Y_d, np.int8),
            np.asarray(EP), np.asarray(TASK), np.asarray(QQ, np.int32),
            np.asarray(ONS, np.int32), np.asarray(SUC, np.int8),
            np.asarray(SCN, np.int32))


def loto_scores(X, y, task):
    """留一 task 的 L2 logistic，返回逐行的留出概率。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.full(len(y), np.nan)
    for tk in np.unique(task):
        te, tr = task == tk, task != tk
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight=None)
        clf.fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def auc_by_task(p, y, task):
    """在留出 task 内算 AUC，再等权合并（各 task 基率差异极大，直接 pool 无效）。"""
    from sklearn.metrics import roc_auc_score
    vals, ns = [], []
    for tk in np.unique(task):
        m = (task == tk) & np.isfinite(p)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        vals.append(roc_auc_score(y[m], p[m]))
        ns.append(int(m.sum()))
    if not vals:
        return np.nan, 0, []
    return float(np.mean(vals)), len(vals), vals


def calibration(p, y, nbins=10):
    m = np.isfinite(p)
    p, y = p[m], y[m]
    edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p >= a) & (p < b)
        if s.sum() < 20:
            continue
        rows.append(dict(pred=float(p[s].mean()), obs=float(y[s].mean()), n=int(s.sum())))
        ece += s.sum() / len(p) * abs(p[s].mean() - y[s].mean())
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    return dict(bins=rows, ece=float(ece), brier=brier, base_rate=base,
                brier_skill=float(1 - brier / (base * (1 - base) + 1e-12)))


def alarm_lead(p, ep, qq, ons, suc, task, fpr_target):
    """阈值在**干净成功集**（成功且无 trap 事件）上按集级误报率校准，再读被困集的提前量。"""
    order = np.lexsort((qq, ep))
    p, ep, qq, ons, suc, task = (a[order] for a in (p, ep, qq, ons, suc, task))
    clean, trapped = {}, {}
    for e in np.unique(ep):
        m = (ep == e) & np.isfinite(p)
        if m.sum() == 0:
            continue
        o, s = int(ons[m][0]), int(suc[m][0])
        (clean if (o < 0 and s == 1) else (trapped if o >= 0 else {}))[e] = (p[m], qq[m], o)
    if not clean or not trapped:
        return None
    peaks = np.array([v[0].max() for v in clean.values()])
    thr = float(np.quantile(peaks, 1 - fpr_target))
    leads, fired = [], 0
    for e, (pv, qv, o) in trapped.items():
        hit = np.where(pv >= thr)[0]
        if len(hit):
            fired += 1
            leads.append(int(o - qv[hit[0]]))
    leads = np.array(leads)
    return dict(fpr_target=fpr_target, threshold=thr,
                n_clean=len(clean), n_trapped=len(trapped),
                recall=fired / len(trapped),
                median_lead=float(np.median(leads)) if len(leads) else np.nan,
                frac_lead_ge1=float(np.mean(leads >= 1)) if len(leads) else np.nan,
                frac_lead_ge2=float(np.mean(leads >= 2)) if len(leads) else np.nan,
                lead_quartiles=[float(np.quantile(leads, q)) for q in (.25, .5, .75)]
                if len(leads) else None)


def cond_committor_table(p, y, qq, qbins, nq=4):
    """条件 committor 表：行 = 绝对 q 分箱，列 = **箱内** routing 分数分位，格 = 实测 P(h 内进 trap)。

    这是框架要求的"概率而非排序"，且时钟由行固定住 —— 同一行内比较，elapsed time 相同。
    """
    rows = []
    for a, b in zip(qbins[:-1], qbins[1:]):
        m = (qq >= a) & (qq < b) & np.isfinite(p)
        if m.sum() < 200 or y[m].sum() < 10:
            continue
        pv, yv = p[m], y[m]
        edges = np.quantile(pv, np.linspace(0, 1, nq + 1))
        edges[-1] += 1e-12
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s_ = (pv >= lo) & (pv < hi)
            if s_.sum() < 20:
                cells.append(None)
                continue
            k, n = int(yv[s_].sum()), int(s_.sum())
            cells.append(dict(n=n, obs=k / n))
        rows.append(dict(qbin=f"[{a},{b})", n=int(m.sum()), base=float(yv.mean()), cells=cells))
    return rows


def main(corpus="grid50x8", horizons=(1, 2, 3, 5)):
    rng = np.random.default_rng(20260904)
    report = {}
    for h in horizons:
        Xr, yt, yd, ep, task, qq, ons, suc, scn = build_rows(corpus, h)
        p = loto_scores(Xr, yt, task)      # 只吃 routing，没有任何时钟特征
        pdone = loto_scores(Xr, yd, task)
        entry = dict(n_rows=int(len(yt)), n_episodes=int(len(np.unique(ep))),
                     n_tasks=int(len(np.unique(task))),
                     trap_rate=float(yt.mean()), done_rate=float(yd.mean()))
        print(f"\n===== {corpus}  h={h}  风险集行数={len(yt)} 集={len(np.unique(ep))} "
              f"任务={len(np.unique(task))}  P(h 内进 trap)={yt.mean():.4f}", flush=True)

        # 分层 AUC：时钟与任务由匹配控制，不进模型
        strata = {
            "match(task,q)": np.char.add(task, np.char.add("@", qq.astype(str))),
            "match(task,scene,q)": np.char.add(
                np.char.add(task, np.char.add("#", scn.astype(str))),
                np.char.add("@", qq.astype(str))),
        }
        m = np.isfinite(p)
        for name, cell in strata.items():
            auc, npair, per = C.stratified_auc(p[m], yt[m], cell[m])
            null = C.perm_null_auc(p[m], yt[m], cell[m], 400, rng)
            pp = (np.sum(np.abs(null - .5) >= abs(auc - .5)) + 1) / (len(null) + 1)
            aucd = C.stratified_auc(pdone[m], yd[m], cell[m])[0]
            entry[name] = dict(auc_trap=float(auc), n_pairs=npair, n_strata=len(per),
                               p_perm=float(pp), auc_complete=float(aucd))
            print(f"  {name:22s} AUC(trap≤h)={auc:.4f} (p_perm={pp:.4f}, "
                  f"{len(per)} 层, {npair} 配对)   AUC(完成≤h)={aucd:.4f}", flush=True)

        qb = [0, 5, 10, 15, 20, 25, 30, 40, 60]
        tab = cond_committor_table(p, yt, qq, qb)
        entry["cond_committor"] = tab
        print("  条件 committor（行=绝对 q，列=箱内 routing 分数四分位，格=实测 P(h内进trap)）:")
        print(f"    {'qbin':>10s} {'n':>6s} {'基率':>7s}  " +
              "  ".join(f"Q{i+1:d}" for i in range(4)))
        for r in tab:
            cs = "  ".join("  —  " if c is None else f"{c['obs']:.3f}" for c in r["cells"])
            print(f"    {r['qbin']:>10s} {r['n']:6d} {r['base']:7.4f}  {cs}")

        for fpr in (0.05, 0.10):
            al = alarm_lead(p, ep, qq, ons, suc, task, fpr)
            if al:
                entry.setdefault("alarm", []).append(al)
                print(f"  报警@集级FPR={fpr:.0%}: 阈值={al['threshold']:.4f} "
                      f"召回={al['recall']:.3f} 中位提前={al['median_lead']:.1f} query  "
                      f"提前≥1 {al['frac_lead_ge1']:.2f} / ≥2 {al['frac_lead_ge2']:.2f}", flush=True)
        report[f"h{h}"] = entry
    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / f"c7_single_traj_{corpus}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid50x8")
