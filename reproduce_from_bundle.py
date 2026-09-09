#!/usr/bin/env python3
"""Re-derive the headline numbers from the bundled npz alone.

The analysis scripts as shipped read the 76 GB capture.  This one reads only
`bundle/data/*.npz` -- 43 MB -- and reruns the same functions on it, so the
findings can be checked without the corpus.  It imports `cooccur` and `score`
from `analyze_rules` rather than reimplementing them, so a divergence here would
be a real divergence and not a transcription slip.

Four blocks, matching logs 01, 02, 06 and 07:
    k=0 commitment over the 5 tasks
    the rule search and its within-scene permutation null, on t08
    honest split -- mine on half the scenes, score on the other half
    the same against a baseline that also knows the emitted action chunk

Usage:  python3 reproduce_from_bundle.py [bundle/data]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_rules import cooccur, lit_name, score  # noqa: E402
from analyze_rules_increment import loso_p  # noqa: E402

T08 = "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz"
N_REP = 20
TOPK = 20


def read(path, win=None):
    d = np.load(path, allow_pickle=True)
    nr = d["n_rows"].astype(int)
    W = int(nr.min()) if win is None else win
    off = np.concatenate([[0], np.cumsum(nr)[:-1]])
    take = (off[:, None] + np.arange(W)[None, :]).ravel()
    n = len(nr)
    ids = d["state_token_top4"][take].astype(np.int64).reshape(n, W, 8, 4)
    M = np.zeros((n, W, 8, 32), np.float32)
    np.put_along_axis(M, ids, 1.0, -1)
    return dict(
        n=n, win=W, n_rows=nr, success=d["success"], scene=d["scene"],
        M=M.reshape(n, W, 256),
        prop=d["proprio"][take].reshape(n, W, 8),
        sim=d["sim_state"][take].reshape(n, W, -1),
        act=d["actions"][take].reshape(n, W, 70),
        probs=d["state_token_probs"][take].astype(np.float32).reshape(n, W, 256))


def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> int:
    if len(sys.argv) > 1:
        root = pathlib.Path(sys.argv[1])
    else:
        # works both in the source tree (bundle/data) and inside the unpacked
        # archive, where this file sits in scripts/ next to a sibling data/
        cand = [HERE / "bundle/data", HERE.parent / "data", HERE / "data"]
        root = next((c for c in cand if c.is_dir()), cand[0])
    files = sorted(root.glob("*.npz"))
    if not files:
        print("没找到派生数据；用法: python3 %s <data 目录>" % pathlib.Path(__file__).name)
        return 1
    print("数据目录: %s\n" % root)
    print("=== 1. k=0 承诺（只用 success/scene）===")
    print("  任务                        局数  成功率  H(Y)   H(Y|场景)  已解决  锁死")
    for f in files:
        d = np.load(f, allow_pickle=True)
        y, sc = d["success"], d["scene"]
        HY = Hb(y.mean())
        rr = np.array([y[sc == s].mean() for s in np.unique(sc)])
        HYX = np.mean([Hb(r) for r in rr])
        print("  %-26s %4d  %5.1f%%  %.3f   %.3f     %5.1f%%  %d/%d"
              % (str(d["task"])[:26], len(y), 100 * y.mean(), HY, HYX,
                 100 * (1 - HYX / HY) if HY > 0 else 0.0,
                 int(((rr == 0) | (rr == 1)).sum()), len(rr)))

    D = read(root / T08)
    y, scene, M, n, W = (D["success"], D["scene"], D["M"], D["n"], D["win"])
    fail = ~y
    p_fail = np.zeros(n, np.float32)
    for s in np.unique(scene):
        p_fail[scene == s] = fail[scene == s].mean()
    fires = cooccur(M)
    sup, obs, exp, z = score(fires, fail, p_fail)
    ok = (sup >= 20) & (sup <= n - 5)
    ok[np.tril_indices(256, -1)] = False
    print("\n=== 2. 规则搜索（t08，窗口 %d 步，%d 条可评估合取）===" % (W, ok.sum()))
    rng = np.random.default_rng(0)
    F = fires.astype(np.float32)
    idx = [np.flatnonzero(scene == s) for s in np.unique(scene)]
    e = np.einsum("i,iab->ab", p_fail, F)
    v = np.einsum("i,iab->ab", p_fail * (1 - p_fail), F)
    mz, mp = [], []
    for _ in range(300):
        fp = fail.copy()
        for m in idx:
            fp[m] = fail[m][rng.permutation(len(m))]
        o = np.einsum("i,iab->ab", fp.astype(np.float32), F)
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = (o - e) / np.sqrt(np.maximum(v, 1e-9))
            pure = ok & (o == sup) & (e / np.maximum(sup, 1) < 0.999)
        mz.append(np.nanmax(np.where(ok, zz, -np.inf)))
        mp.append(int(sup[pure].max()) if pure.any() else 0)
    print("  置换零假设：最大 z 的 95%% 分位 %.2f；纯失败规则最大支持度 95%% 分位 %d"
          % (np.quantile(mz, .95), int(np.quantile(mp, .95))))
    order = np.argsort(np.where(ok, z, -np.inf).ravel())[::-1]
    for f in order[:3]:
        a, b = np.unravel_index(f, z.shape)
        print("  %-22s 支持 %4d  失败 %4d  场景期望 %6.1f  z=%.2f"
              % ("%s & %s" % (lit_name(a), lit_name(b)), sup[a, b], obs[a, b],
                 exp[a, b], z[a, b]))
    with np.errstate(invalid="ignore", divide="ignore"):
        pu = ok & (obs == sup) & (exp / np.maximum(sup, 1) < 0.999)
        ps = ok & (obs == 0) & (exp / np.maximum(sup, 1) > 0.001)
    print("  最纯的全失败规则支持度 %d；纯成功方向 %d"
          % (int(np.where(pu, sup, 0).max()), int(np.where(ps, sup, 0).max())))

    print("\n=== 3-4. 诚实划分：按场景挖一半、验另一半（%d 次）===" % N_REP)
    ep_row = np.repeat(np.arange(n), W)
    step = np.tile(np.arange(W), n).astype(np.float32)[:, None]
    yf = np.repeat(fail, W).astype(int)
    si = np.unique(scene, return_inverse=True)[1]
    oh = np.zeros((n * W, si.max() + 1), np.float32)
    oh[np.arange(n * W), si[ep_row]] = 1
    PH = np.c_[oh, D["sim"].reshape(n * W, -1), D["prop"].reshape(-1, 8), step]
    base = {"场景+物理状态": PH, "场景+物理+动作块": np.c_[PH, D["act"].reshape(-1, 70)]}
    res = {}
    for k, X in base.items():
        p = loso_p(X, yf, ep_row)
        ll = -(yf * np.log(np.clip(p, 1e-6, 1)) +
               (1 - yf) * np.log(np.clip(1 - p, 1e-6, 1))).mean() / np.log(2)
        print("  基线「%s」(%d 维) 对数损失 %.3f bit" % (k, X.shape[1], ll))
        res[k] = (yf - p).reshape(n, W)
    rnd = np.random.default_rng(3).random((n, W)) < 0.10
    acc = {k: {"pool": [], "rnd": []} for k in base}
    for r in range(N_REP):
        g = np.random.default_rng(100 + r)
        scp = g.permutation(np.unique(scene))
        A = np.isin(scene, scp[:len(scp) // 2])
        B = ~A
        pfA = np.zeros(A.sum(), np.float32)
        scA = scene[A]
        for s in np.unique(scA):
            pfA[scA == s] = fail[A][scA == s].mean()
        sA, _, _, zA = score(fires[A], fail[A], pfA)
        okA = (sA >= 10) & (sA <= A.sum() - 3)
        okA[np.tril_indices(256, -1)] = False
        if okA.sum() < TOPK:
            continue
        od = np.argsort(np.where(okA, zA, -np.inf).ravel())[::-1][:TOPK]
        pool = np.zeros((n, W), bool)
        for f in od:
            u, w = np.unravel_index(f, zA.shape)
            pool |= (M[:, :, u] > 0.5) & (M[:, :, w] > 0.5)
        for k in base:
            acc[k]["pool"].append(res[k][B][pool[B]].mean())
            acc[k]["rnd"].append(res[k][B][rnd[B]].mean())
    print("  前 %d 条并集，减去同划分随机对照的配对差：" % TOPK)
    for k in base:
        d = np.array(acc[k]["pool"]) - np.array(acc[k]["rnd"])
        print("    vs %-18s %+.4f ± %.4f  (%d/%d 正)"
              % (k, d.mean(), d.std() / np.sqrt(len(d)), int((d > 0).sum()), len(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
