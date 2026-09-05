"""C2 — 在**逐位相同的观测**上，MoE routing 能不能挑出会成功的那个噪声？

§7 的 β_B / β_W 分解，做在唯一无混杂的那个点上。

为什么 q=0 是唯一干净的点：一个 cell 内的 K 个同胞共享 init_state_id，settle 确定性，
第一次推理之前机器人没有执行过任何动作 —— **q=0 的观测逐位相同**（C3 实测
D_phys(0)=0.00000 m），routing 的全部差异只来自 flow noise ξ_0。从 q=1 起
（replan=10）物理状态已分叉，组内对比就被状态差异污染。

  β_W（候选选择器）: 组内 AUC —— 同 cell 内失败同胞的信号是否高于成功同胞。
  β_B（状态传感器）: 同 task 内 Spearman(cell 均值信号, ĉ_i)。必须限定 task 内，
                     因为 routing 100% 编码任务身份（E7）。

**动作对照臂（必须有）**：q=0 的 routing 只看见 ξ_0，而结局由整条噪声序列
ξ_0..ξ_Q 决定，所以 β_W≈0 有两种读法。动作 chunk a_0(ξ_0, obs) 与 routing 看见
完全相同的信息、是同一次前向的输出，因此：
  · 动作也 ≈0.5 ⇒ 第一块 chunk 本来就不决定命运，该 null 不能归咎于 MoE；
  · 动作 >0.5 而 routing ≈0.5 ⇒ routing 对自己网络输出里已有的、与命运相关的
    差异是盲的 —— 这才是对 MoE 的强指控。

统计：独立单位 = cell；组内标签置换给精确零分布；cell 级 bootstrap 给 CI。
多元探针 = 核岭回归 + leave-one-cell-out，每折的 Cholesky 与 y 无关故只分解一次，
置换只做三角回代。
"""
import os
import pathlib
import sys

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
RUNS = {
    "main16x32": (HUB / "cache/HiMoE-VLA", "right-16x32"),
    "grid50x8": (HUB / "cache_new/HiMoE-VLA", "right-50x8-20260903"),
}
N_PERM = 5000
N_BOOT = 5000
N_PERM_MV = 2000


def collect(corpus, ykey, with_reps=True, with_actions=True):
    """所有 task 的 q=0 行，只保留敏感格（组内既有成功也有失败）。"""
    root, run_id = RUNS[corpus]
    fx = {f: [] for f in C.FEATS}
    R, A, Y, CELL = [], [], [], []
    cells = []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=with_reps)
        if ykey in ("loop", "anyloop") and not t.loop_valid:
            continue
        eps, Yd, _ = C.outcomes(t)
        _, sc, _ = C.episode_cells(t)
        y_all = Yd[ykey]
        epidx = {int(e): i for i, e in enumerate(eps)}
        m0 = t.q == 0
        ep0, sc0 = t.ep[m0], t.scene[m0]
        y0 = np.array([y_all[epidx[int(e)]] for e in ep0], dtype=np.int8)
        r0 = np.asarray(t.reps[m0], dtype=np.float32) if with_reps else None
        run = root / suite / task / run_id
        for s in np.unique(sc0):
            sm = sc0 == s
            k, ns = int(sm.sum()), int(y0[sm].sum())
            if not (0 < ns < k):
                continue
            for f in C.FEATS:
                fx[f].append(t.feats[f][m0][sm] if f in t.feats else np.full(k, np.nan))
            if with_reps:
                R.append(r0[sm])
            if with_actions:
                acts = []
                for e in ep0[sm]:
                    with np.load(run / "client" / f"episode_{int(e):02d}.npz") as d:
                        acts.append(np.asarray(d["actions"][0], np.float32).ravel())  # (10,7)->70
                A.append(np.stack(acts))
            Y.append(y0[sm])
            CELL.append(np.full(k, len(cells)))
            cells.append(f"{suite}/{task}#{s}")
    if not cells:
        return None
    return dict(
        feats={f: np.concatenate(v) for f, v in fx.items()},
        reps=np.concatenate(R) if with_reps else None,
        acts=np.concatenate(A) if with_actions else None,
        y=np.concatenate(Y),
        cell=np.concatenate(CELL),
        cell_names=cells,
    )


def beta_w_univariate(D, rng):
    res = []
    for f in C.FEATS:
        v = D["feats"][f]
        if not np.all(np.isfinite(v)) or v.std() == 0:
            continue
        auc, npair, per = C.stratified_auc(v, D["y"], D["cell"])
        null = C.perm_null_auc(v, D["y"], D["cell"], N_PERM, rng)
        boot = C.bootstrap_cells(v, D["y"], D["cell"], N_BOOT, rng)
        p = (np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (len(null) + 1)
        res.append(dict(feature=f, auc=float(auc), n_pairs=npair, n_cells=len(per),
                        p_perm=float(p),
                        ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                        null_sd=float(null.std())))
    res.sort(key=lambda r: -abs(r["auc"] - 0.5))
    return res


def kridge_loco(X, y, cell, lam=1.0):
    """核岭 + leave-one-cell-out 的组内探针分数。

    X 先做 **cell 内去均值**：把 cell 级（=状态级）信息完全剔除，剩下的只有噪声引起的
    那一部分，所以得分严格是 β_W，借不到 β_B 的力。
    返回 scorer(y_variant) -> scores，Cholesky 只分解一次。
    """
    Xc = np.asarray(X, np.float64).copy()
    for c in np.unique(cell):
        m = cell == c
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    sd = Xc.std(0)
    Xc /= np.where(sd > 1e-9, sd, 1.0)
    K = Xc @ Xc.T
    K /= max(np.trace(K) / len(K), 1e-12)  # 尺度归一，使 lam 可跨特征空间比较
    folds = []
    for c in np.unique(cell):
        te = np.where(cell == c)[0]
        tr = np.where(cell != c)[0]
        cf = cho_factor(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), lower=True)
        folds.append((tr, te, cf, K[np.ix_(te, tr)]))

    def score(yv):
        s = np.empty(len(yv))
        for tr, te, cf, Kte in folds:
            yt = yv[tr].astype(np.float64)
            yt = yt - yt.mean()
            s[te] = Kte @ cho_solve(cf, yt)
        return s

    return score


def beta_w_multivariate(X, y, cell, rng, label, n_perm=N_PERM_MV, lam=1.0):
    if X is None:
        return None
    score = kridge_loco(X, y, cell, lam=lam)
    auc = C.stratified_auc(score(y), y, cell)[0]
    idxs = [np.where(cell == c)[0] for c in np.unique(cell)]
    null = np.empty(n_perm)
    yp = y.copy()
    for b in range(n_perm):
        for i in idxs:
            yp[i] = rng.permutation(y[i])
        null[b] = C.stratified_auc(score(yp), yp, cell)[0]
    p = (np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (n_perm + 1)
    # 以零分布的 95 分位给出"这个设计在本样本量下能检出的最小效应"
    return dict(label=label, dim=int(X.shape[1]), auc=float(auc), p_perm=float(p),
                n_perm=n_perm, null_mean=float(null.mean()), null_sd=float(null.std()),
                detectable_auc_95=float(np.quantile(null, 0.95)))


def beta_b(corpus, ykey):
    rows = []
    for suite, task in C.iter_tasks(corpus):
        t = C.load_task(corpus, suite, task, with_reps=False)
        if ykey in ("loop", "anyloop") and not t.loop_valid:
            continue
        eps, Yd, _ = C.outcomes(t)
        _, sc, _ = C.episode_cells(t)
        y_all = Yd[ykey]
        epidx = {int(e): i for i, e in enumerate(eps)}
        m0 = t.q == 0
        ep0, sc0 = t.ep[m0], t.scene[m0]
        y0 = np.array([y_all[epidx[int(e)]] for e in ep0], float)
        for s in np.unique(sc0):
            sm = sc0 == s
            r = dict(task=f"{suite}/{task}", chat=float(y0[sm].mean()))
            for f in C.FEATS:
                r[f] = float(t.feats[f][m0][sm].mean()) if f in t.feats else np.nan
            rows.append(r)
    res = []
    for f in C.FEATS:
        rhos = []
        for task in sorted({r["task"] for r in rows}):
            sub = [r for r in rows if r["task"] == task and np.isfinite(r.get(f, np.nan))]
            c = np.array([r["chat"] for r in sub])
            v = np.array([r[f] for r in sub])
            if len(sub) < 6 or c.std() == 0 or v.std() == 0:
                continue
            rhos.append(stats.spearmanr(v, c).statistic)
        if not rhos:
            continue
        rhos = np.array(rhos)
        pos = int((rhos > 0).sum())
        p = stats.binomtest(max(pos, len(rhos) - pos), len(rhos), 0.5).pvalue
        res.append(dict(feature=f, mean_rho=float(rhos.mean()), n_tasks=len(rhos),
                        n_pos=pos, p_sign=float(p)))
    res.sort(key=lambda r: -abs(r["mean_rho"]))
    return res


def main():
    rng = np.random.default_rng(20260904)
    report = {}
    for corpus in ("main16x32", "grid50x8"):
        for ykey in ("fail", "loop", "static"):
            D = collect(corpus, ykey)
            if D is None or len(np.unique(D["cell"])) < 5:
                continue
            key = f"{corpus}:{ykey}"
            print(f"\n===== {key}  cells={len(D['cell_names'])} eps={len(D['y'])} "
                  f"events={int(D['y'].sum())}", flush=True)
            uni = beta_w_univariate(D, rng)
            mv_r = beta_w_multivariate(D["reps"], D["y"], D["cell"], rng, "routing_reps1280")
            mv_a = beta_w_multivariate(D["acts"], D["y"], D["cell"], rng, "action_chunk70")
            bb = beta_b(corpus, ykey)
            report[key] = dict(n_cells=len(D["cell_names"]), n_episodes=int(len(D["y"])),
                               n_events=int(D["y"].sum()), cells=D["cell_names"],
                               beta_W_univariate=uni,
                               beta_W_multivariate=[m for m in (mv_r, mv_a) if m],
                               beta_B_within_task=bb)
            print("  β_W 组内 AUC @q=0（观测逐位相同）")
            for r in uni[:4]:
                print(f"    {r['feature']:22s} AUC={r['auc']:.4f} "
                      f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] p={r['p_perm']:.4f}", flush=True)
            for m in (mv_r, mv_a):
                if m:
                    print(f"    {m['label']:22s} AUC={m['auc']:.4f} p={m['p_perm']:.4f} "
                          f"(可检出下限 AUC≈{m['detectable_auc_95']:.3f})", flush=True)
            print("  β_B  task 内 Spearman(cell 均值信号, ĉ)")
            for r in bb[:4]:
                print(f"    {r['feature']:22s} rho={r['mean_rho']:+.3f} "
                      f"({r['n_pos']}/{r['n_tasks']} 同向) p={r['p_sign']:.4f}", flush=True)
    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / "c2_q0_selectability.json")


if __name__ == "__main__":
    main()
