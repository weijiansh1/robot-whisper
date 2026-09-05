"""C2c — 组内多元 β_W 的**无迁移、无方向**检验。

C2/C2b 的 LOCO 核岭探针在 fail 口径上稳定给出 AUC=0.390（null 居中 0.501，p=0.0025）。
低于 0.5 且可复现，意味着"判别方向存在但跨 cell 反号"，而不是"没有信息"。可能的机制：
cell 内 y 去中心后，每个 cell 的**稀有类**权重最大（30/32 失败的格里，2 个成功各拿
−0.94 的权重），于是探针学到的是"格内异常者"方向；在留出格上稀有类换了一边，符号就翻。
不论机制为何，需要迁移的估计量都不适合回答"这个 cell 内、routing 空间里，成败是否分开"。

换成两个不需要跨 cell 迁移、也不假定线性方向的统计量，都在 cell 内算、cell 内置换：

  NN 同标一致率  frac_j 1[y_{NN(j)} = y_j]      —— 任意形状的聚集都能测到
  能量统计量     2·E‖X_f−X_s‖ − E‖X_f−X_f'‖ − E‖X_s−X_s'‖  —— 两样本分布差异

零假设：cell 内标签可交换。置换保持每个 cell 的失败数不变，故 cell 的成分异质性
不进入零分布。同时跑 routing / action / 阳性对照三个臂。
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cload as C
from c2_q0_selectability import collect

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
N_PERM = 20000


def cell_blocks(X, y, cell, standardize=True):
    """按 cell 切块，块内标准化（去掉 cell 级尺度差异）并预算距离矩阵。"""
    blocks = []
    for c in np.unique(cell):
        m = cell == c
        Z = np.asarray(X[m], np.float64)
        Z = Z - Z.mean(0, keepdims=True)
        if standardize:
            s = np.linalg.norm(Z, axis=1).mean()
            if s > 1e-12:
                Z = Z / s
        D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        blocks.append((D, y[m].astype(np.int8)))
    return blocks


def nn_concordance(blocks, ys=None):
    num = den = 0
    for i, (D, y0) in enumerate(blocks):
        y = y0 if ys is None else ys[i]
        nn = np.argmin(D, axis=1)
        num += int((y[nn] == y).sum())
        den += len(y)
    return num / den


def energy_stat(blocks, ys=None):
    """按 cell 的配对数加权合并能量统计量（越大 = 两类分布越分开）。"""
    num = den = 0.0
    for i, (D, y0) in enumerate(blocks):
        y = y0 if ys is None else ys[i]
        f, s = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(f) < 1 or len(s) < 1:
            continue
        Dm = np.where(np.isinf(D), 0.0, D)
        between = Dm[np.ix_(f, s)].mean()
        ff = Dm[np.ix_(f, f)].sum() / max(len(f) * (len(f) - 1), 1)
        ss = Dm[np.ix_(s, s)].sum() / max(len(s) * (len(s) - 1), 1)
        w = len(f) * len(s)
        num += w * (2 * between - ff - ss)
        den += w
    return num / den if den else np.nan


def run_arm(X, y, cell, label, rng, n_perm=N_PERM):
    blocks = cell_blocks(X, y, cell)
    obs_nn = nn_concordance(blocks)
    obs_en = energy_stat(blocks)
    null_nn = np.empty(n_perm)
    null_en = np.empty(n_perm)
    ys0 = [b[1] for b in blocks]
    for b in range(n_perm):
        ys = [rng.permutation(v) for v in ys0]
        null_nn[b] = nn_concordance(blocks, ys)
        null_en[b] = energy_stat(blocks, ys)
    p_nn = (np.sum(np.abs(null_nn - null_nn.mean()) >= abs(obs_nn - null_nn.mean())) + 1) / (n_perm + 1)
    p_en = (np.sum(np.abs(null_en - null_en.mean()) >= abs(obs_en - null_en.mean())) + 1) / (n_perm + 1)
    return dict(label=label, dim=int(X.shape[1]), n_cells=len(blocks),
                nn_concordance=float(obs_nn), nn_null_mean=float(null_nn.mean()),
                nn_null_sd=float(null_nn.std()), nn_z=float((obs_nn - null_nn.mean()) / (null_nn.std() + 1e-12)),
                nn_p=float(p_nn),
                energy=float(obs_en), energy_null_mean=float(null_en.mean()),
                energy_null_sd=float(null_en.std()),
                energy_z=float((obs_en - null_en.mean()) / (null_en.std() + 1e-12)),
                energy_p=float(p_en))


def main():
    rng = np.random.default_rng(20260904)
    report = {}
    for corpus in ("main16x32", "grid50x8"):
        for ykey in ("fail", "loop", "static"):
            D = collect(corpus, ykey, with_reps=True, with_actions=True)
            if D is None or len(np.unique(D["cell"])) < 5:
                continue
            y, cell = D["y"], D["cell"]
            arms = {"routing_reps1280": D["reps"], "action_chunk70": D["acts"],
                    "scalar_feats8": np.column_stack(
                        [D["feats"][f] for f in C.FEATS
                         if np.all(np.isfinite(D["feats"][f]))])}
            # 阳性对照：在 routing 上加一维 α·(y − cell 均值)
            yc = y.astype(float).copy()
            for c in np.unique(cell):
                m = cell == c
                yc[m] -= yc[m].mean()
            Z = np.asarray(D["reps"], np.float64)
            sc = Z.std()
            arms["positive_control_a0.10"] = np.column_stack(
                [Z, 0.10 * yc * sc * np.sqrt(Z.shape[1]) + rng.normal(0, sc, len(yc))])
            key = f"{corpus}:{ykey}"
            print(f"\n===== {key}  cells={len(D['cell_names'])} eps={len(y)} "
                  f"events={int(y.sum())}", flush=True)
            res = []
            for label, X in arms.items():
                r = run_arm(X, y, cell, label, rng)
                res.append(r)
                print(f"  {label:24s} dim={r['dim']:5d}  "
                      f"NN={r['nn_concordance']:.4f} (null {r['nn_null_mean']:.4f}, "
                      f"z={r['nn_z']:+.2f}, p={r['nn_p']:.4f})   "
                      f"energy z={r['energy_z']:+.2f} p={r['energy_p']:.4f}", flush=True)
            report[key] = dict(n_cells=len(D["cell_names"]), n_eps=int(len(y)),
                               n_events=int(y.sum()), arms=res)
    OUT.mkdir(exist_ok=True)
    C.jdump(report, OUT / "c2c_within_cell_multivar.json")


if __name__ == "__main__":
    main()
