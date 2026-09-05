"""C2b — 把 q=0 的多元组内探针查干净。

C2 首轮给出 routing_reps1280 组内 AUC=0.390 (p=0.0045)：**低于** 0.5 且显著。
反向显著要么是真信号被我定错了方向，要么是探针本身有偏。两个可疑处：

  (a) 训练时 y 只做了**全局**去均值，而 X 做的是 **cell 内**去均值。cell 的基础失败率
      差异因此进了 y 却进不了 X，模型只能用组内噪声去拟合组间成分 —— 在留出格上
      这会系统性地产生反向预测。正确的估计量要求 y 也做 cell 内去均值。
  (b) 没有阳性对照，无法证明这条管线在有信号时能测出来。

本脚本三个臂同跑：
  y_centering ∈ {global, cell}     × 特征 ∈ {routing_reps1280, action_chunk70, 阳性对照}
阳性对照 = 在 routing 特征上叠加 α·(y − cell 均值) 的合成维，α 调到组内 AUC≈0.65，
用来读出这个设计在本样本量下的**可检出下限**。
"""
import os
import pathlib
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, os.path.dirname(__file__))
import cload as C
from c2_q0_selectability import collect

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
N_PERM = 2000


def make_scorer(X, cell, lam=1.0, y_center="cell"):
    Xc = np.asarray(X, np.float64).copy()
    for c in np.unique(cell):
        m = cell == c
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    sd = Xc.std(0)
    Xc /= np.where(sd > 1e-9, sd, 1.0)
    K = Xc @ Xc.T
    K /= max(np.trace(K) / len(K), 1e-12)
    folds = []
    for c in np.unique(cell):
        te = np.where(cell == c)[0]
        tr = np.where(cell != c)[0]
        cf = cho_factor(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), lower=True)
        folds.append((tr, te, cf, K[np.ix_(te, tr)], cell[tr]))

    def score(yv):
        s = np.empty(len(yv))
        for tr, te, cf, Kte, ctr in folds:
            yt = yv[tr].astype(np.float64)
            if y_center == "cell":
                for c in np.unique(ctr):
                    m = ctr == c
                    yt[m] -= yt[m].mean()
            else:
                yt = yt - yt.mean()
            s[te] = Kte @ cho_solve(cf, yt)
        return s

    return score


def evaluate(X, y, cell, rng, label, y_center, n_perm=N_PERM, lam=1.0):
    score = make_scorer(X, cell, lam=lam, y_center=y_center)
    auc = C.stratified_auc(score(y), y, cell)[0]
    idxs = [np.where(cell == c)[0] for c in np.unique(cell)]
    null = np.empty(n_perm)
    yp = y.copy()
    for b in range(n_perm):
        for i in idxs:
            yp[i] = rng.permutation(y[i])
        null[b] = C.stratified_auc(score(yp), yp, cell)[0]
    p = (np.sum(np.abs(null - .5) >= abs(auc - .5)) + 1) / (n_perm + 1)
    return dict(label=label, y_center=y_center, dim=int(X.shape[1]), lam=lam,
                auc=float(auc), p_perm=float(p), null_mean=float(null.mean()),
                null_sd=float(null.std()),
                detect_floor_auc=float(np.quantile(null, 0.975)))


def main():
    rng = np.random.default_rng(20260904)
    report = {}
    for corpus in ("main16x32",):
        D = collect(corpus, "fail")
        y, cell = D["y"], D["cell"]
        arms = {"routing_reps1280": D["reps"], "action_chunk70": D["acts"]}
        # 阳性对照：在 routing 上加一维 α·(y − cell 均值)，α 由组内可分性反推
        yc = y.astype(float).copy()
        for c in np.unique(cell):
            m = cell == c
            yc[m] -= yc[m].mean()
        for alpha in (0.15, 0.30):
            Z = np.asarray(D["reps"], np.float64)
            planted = alpha * yc * Z.std() + rng.normal(0, Z.std() * 0.5, len(yc))
            arms[f"positive_control_a{alpha}"] = np.column_stack([Z, planted])
        res = []
        for label, X in arms.items():
            for yctr in ("cell", "global"):
                r = evaluate(X, y, cell, rng, label, yctr)
                res.append(r)
                print(f"{corpus:10s} {label:24s} y_center={yctr:6s} "
                      f"AUC={r['auc']:.4f} p={r['p_perm']:.4f} "
                      f"null={r['null_mean']:.3f}±{r['null_sd']:.3f} "
                      f"检出下限={r['detect_floor_auc']:.3f}", flush=True)
        report[corpus] = dict(n_cells=len(D["cell_names"]), n_eps=int(len(y)),
                              n_events=int(y.sum()), arms=res)
    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / "c2b_mvprobe.json")


if __name__ == "__main__":
    main()
