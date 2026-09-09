#!/usr/bin/env python3
"""Split the routing into the part that is a pose readout and the part that is not.

The map runs both ways -- pose predicts which experts fire at AUC 0.94-0.999, and
the fired set reconstructs the pose at R^2 0.87-0.97 -- so most of what the state
token's routing does is re-encode the arm.  The interesting quantity is therefore
the remainder: fit the routing from the physical state, subtract, and ask what is
left.

A budget, all leave-one-scene-out so nothing is explained by scene identity:

    routing = f(proprio, 8)  +  g(objects, the rest of sim_state)  +  residual

and then three questions about the residual, which are the only ways it can be
anything other than arithmetic:

    is it structured in time, or is it independent draw to draw?  Tie flips are
      memoryless; a signal is not.
    is it shared between rollouts of one scene, or private to each?  The flow
      noise is private; anything driven by the observation is shared.
    does it carry the outcome?

Run on the state token, which is where the input dependence lives, with the ten
action tokens alongside for contrast -- they carry a fixed positional code, so
their budget should look completely different.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
RNG = np.random.default_rng(0)


def loso_fit(X, Y, scene):
    """Leave-one-scene-out prediction of Y from X."""
    pred = np.zeros_like(Y)
    for h in np.unique(scene):
        tr, te = scene != h, scene == h
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=10.0).fit(sc.transform(X[tr]), Y[tr]).predict(
            sc.transform(X[te]))
    return pred


def frac(Y, pred):
    """Fraction of Y's total variance the prediction accounts for."""
    return 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def sauc(x, y, scene):
    num = den = 0.0
    for s in np.unique(scene):
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(x[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else float("nan")


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y_ep = np.array([s["success"] for s in S], bool)
    sc_ep = np.array([s["init_state_id"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])

    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    d0 = np.load(RUN / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.full((len(S), n_rows.max(), d0.shape[1]), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        prop[i, :n_rows[i]] = d["state"][:n_rows[i]]
        sim[i, :n_rows[i]] = d["sim_state"][:n_rows[i]]

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = z["hb_router_probs"].shape[0]
    rows = np.sort(RNG.choice(n, 9000, replace=False))
    P = np.asarray(z["hb_router_probs"].oindex[rows, :, 0, :, :], np.float32)
    ST = P[:, :, 0].reshape(len(rows), -1)                 # state token, 256
    AC = P[:, :, 1:].reshape(len(rows), -1)                # action tokens, 2560
    X8 = prop[ep[rows], step[rows]]
    X47 = sim[ep[rows], step[rows]]
    SC = sc_ep[ep[rows]]
    EPr, STEPr = ep[rows], step[rows]

    print("t08，%d 个控制步样本，留一场景交叉验证\n" % len(rows))
    print("=== 1. 路由方差的信号预算 ===")
    print("  目标                本体感受(8)   +物体位姿(47)   残差")
    out = {}
    for name, Y in (("状态 token (256)", ST), ("动作 token (2560)", AC)):
        p8 = loso_fit(X8, Y, SC)
        p47 = loso_fit(np.c_[X8, X47], Y, SC)
        f8, f47 = frac(Y, p8), frac(Y, p47)
        out[name] = (Y - p47, f8, f47)
        print("  %-18s  %5.1f%%        %5.1f%%        %5.1f%%"
              % (name, 100 * f8, 100 * f47, 100 * (1 - f47)))
    print("  （物体位姿在本体感受之上只多解释 %.1f 个百分点）"
          % (100 * (out["状态 token (256)"][2] - out["状态 token (256)"][1])))

    res = out["状态 token (256)"][0]
    print("\n=== 2. 残差是信号还是算术噪声？ ===")
    # temporal structure: correlation between a residual and the next step's
    same, shuf = [], []
    for e in np.unique(EPr):
        m = np.flatnonzero(EPr == e)
        if len(m) < 4:
            continue
        o = m[np.argsort(STEPr[m])]
        d = np.diff(STEPr[o])
        adj = o[:-1][d == 1], o[1:][d == 1]
        if len(adj[0]) < 2:
            continue
        a, b = res[adj[0]], res[adj[1]]
        same.append(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1)
                                                * np.linalg.norm(b, axis=1) + 1e-9)))
        q = res[RNG.choice(len(res), len(a))]
        shuf.append(np.mean(np.sum(a * q, 1) / (np.linalg.norm(a, axis=1)
                                                * np.linalg.norm(q, axis=1) + 1e-9)))
    print("  相邻控制步的残差余弦   %.4f      随机配对   %.4f"
          % (np.mean(same), np.mean(shuf)))

    # shared between rollouts of a scene, or private?
    sh, pv = [], []
    for s in np.unique(SC):
        for t in range(4, 30, 5):
            m = np.flatnonzero((SC == s) & (STEPr == t))
            if len(m) < 6:
                continue
            V = res[m] / (np.linalg.norm(res[m], axis=1, keepdims=True) + 1e-9)
            C = V @ V.T
            iu = np.triu_indices(len(m), 1)
            sh.append(C[iu].mean())
        m2 = np.flatnonzero(SC == s)
        q1 = RNG.choice(m2, min(400, len(m2)))
        q2 = RNG.choice(m2, min(400, len(m2)))
        V1 = res[q1] / (np.linalg.norm(res[q1], axis=1, keepdims=True) + 1e-9)
        V2 = res[q2] / (np.linalg.norm(res[q2], axis=1, keepdims=True) + 1e-9)
        pv.append(np.mean(np.sum(V1 * V2, 1)))
    print("  同场景同控制步、不同噪声抽样之间的残差余弦  %.4f" % np.mean(sh))
    print("  同场景但随机配对（不同控制步）              %.4f" % np.mean(pv))
    print("  （前者明显更高 = 残差由观测驱动，是共享的；接近 = 各自的噪声）")

    print("\n=== 3. 残差携带成败吗 ===")
    mixed = [int(s) for s in np.unique(sc_ep)
             if 4 <= y_ep[sc_ep == s].sum() <= 28]
    for t in (13, 20, 26, 34):
        m = np.flatnonzero((STEPr == t) & np.isin(SC, mixed))
        if len(m) < 80:
            continue
        yy = y_ep[EPr[m]]
        # a 1-d summary of the residual: its size
        mag = np.linalg.norm(res[m], axis=1)
        raw = np.linalg.norm(ST[m] - ST[m].mean(0), axis=1)
        print("  步 %-3d  n=%3d   残差大小 AUC %.3f    对照：原始路由大小 %.3f"
              % (t, len(m), sauc(mag, yy, SC[m]), sauc(raw, yy, SC[m])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
