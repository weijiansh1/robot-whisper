#!/usr/bin/env python3
"""Are there routing configurations that never succeed?

The k=0 commitment number was carried entirely by scene identity, which says
nothing about the routing.  A sharper question, and the one actually asked: is
there a *rule* -- "layer L picks expert e AND layer L' picks expert e'" -- whose
episodes all fail, and which still separates inside a single initial scene?

Literals are the 256 (layer, expert) pairs of the state token's top-4, at
denoise step 0.  An episode satisfies a conjunction if both literals hold at the
same control step, anywhere in the window.  Window is 0..34 because t08's
shortest success is 35 control steps -- every one of the 512 episodes is still
running throughout, so no rule can be reading episode length.

Two things have to be controlled or this finds rules no matter what:

  scene    a rule that fires only in s49 (0/32) is pure by construction and
           means nothing.  Every count is scored against the episode's own
           scene base rate, so a locked scene contributes zero excess and zero
           variance automatically.
  search   32,896 conjunctions x either direction.  The null is a within-scene
           label permutation rerun through the identical search, and what is
           compared is the *maximum* statistic, so the multiplicity is priced in.

Reported separately: the best scene-adjusted rules, the purest rules by support,
and -- for anything that survives -- a held-out check (mine on half the scenes,
score on the other half) and what pose the rule corresponds to, since the state
token's routing is mostly a pose readout.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
WIN = None                     # per task: the shortest episode
N_PERM = 300
RNG = np.random.default_rng(0)
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
DIM = ["eef x", "eef y", "eef z", "轴角1", "轴角2", "轴角3", "指 a", "指 b"]


def lit_name(i):
    return "L%d:e%d" % (HB_LAYER[i // 32], i % 32)


def load(run, win=None):
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    nr = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(nr)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    # the window is the shortest episode, so every episode is still running
    # throughout it and no rule can be reading episode length
    WIN = int(nr.min()) if win is None else win
    if nr.min() < WIN:
        return None
    prop = np.zeros((len(S), WIN, 8), np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["state"]
        prop[i] = d[:WIN]
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    rows = (off[:, None] + np.arange(WIN)[None, :]).ravel()
    ids = np.asarray(z["hb_expert_ids"].oindex[rows, :, 0, 0, :]).astype(np.int64)
    hot = np.zeros((len(rows), 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    return dict(y=y, scene=scene, prop=prop, win=WIN,
                M=hot.reshape(len(S), WIN, 256), n=len(S))


def cooccur(M):
    """fires[i, a, b] = literals a and b hold at a common control step of episode i."""
    n, _, L = M.shape
    out = np.zeros((n, L, L), bool)
    for i in range(n):
        out[i] = (M[i].T @ M[i]) > 0.5
    return out


def score(fires, fail, p_fail):
    """Scene-adjusted excess failures, as a z-score, for every conjunction."""
    sup = np.einsum("iab->ab", fires.astype(np.float32))
    obs = np.einsum("i,iab->ab", fail.astype(np.float32), fires.astype(np.float32))
    exp = np.einsum("i,iab->ab", p_fail, fires.astype(np.float32))
    var = np.einsum("i,iab->ab", p_fail * (1 - p_fail), fires.astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - exp) / np.sqrt(np.maximum(var, 1e-9))
    return sup, obs, exp, z


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    suite = sys.argv[2] if len(sys.argv) > 2 else "libero_long"
    run = HUB / "cache/HiMoE-VLA" / suite / task / "right-16x32"
    D = load(run)
    y, scene, M, n, WIN = D["y"], D["scene"], D["M"], D["n"], D["win"]
    fail = ~y
    p_fail = np.zeros(n, np.float32)
    for s in np.unique(scene):
        p_fail[scene == s] = fail[scene == s].mean()
    print("%s" % task)
    print("%d 局，%d 个场景，成功率 %.1f%%，窗口 0-%d 步（全员在场）"
          % (n, len(np.unique(scene)), 100 * y.mean(), WIN - 1))
    print("场景基准失败率: " + " ".join("%.2f" % fail[scene == s].mean()
                                        for s in np.unique(scene)))

    fires = cooccur(M)
    iu = np.triu_indices(256, 0)
    sup, obs, exp, z = score(fires, fail, p_fail)
    MIN_SUP = 20
    ok = (sup >= MIN_SUP) & (sup <= n - 5)
    ok[np.tril_indices(256, -1)] = False
    print("\n可评估的合取（支持度>=%d）: %d / %d"
          % (MIN_SUP, ok.sum(), len(iu[0])))

    # --- permutation null on the max statistic -------------------------------
    idx = [np.flatnonzero(scene == s) for s in np.unique(scene)]
    F = fires.astype(np.float32)
    maxz, maxpure = np.zeros(N_PERM), np.zeros(N_PERM, int)
    for p in range(N_PERM):
        fp = fail.copy()
        for m in idx:
            fp[m] = fail[m][RNG.permutation(len(m))]
        o = np.einsum("i,iab->ab", fp.astype(np.float32), F)
        e = np.einsum("i,iab->ab", p_fail, F)
        v = np.einsum("i,iab->ab", p_fail * (1 - p_fail), F)
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = (o - e) / np.sqrt(np.maximum(v, 1e-9))
        maxz[p] = np.nanmax(np.where(ok, zz, -np.inf))
        # all-fail, and not simply because every firing episode sits in a
        # scene that never succeeds
        with np.errstate(invalid="ignore", divide="ignore"):
            pure = ok & (o == sup) & (e / np.maximum(sup, 1) < 0.999)
        maxpure[p] = int(sup[pure].max()) if pure.any() else 0
    zc = np.quantile(maxz, 0.95)
    pc = int(np.quantile(maxpure, 0.95))
    print("置换零假设（场景内打乱标签，%d 次，同一套搜索）:" % N_PERM)
    print("  最大 z 的 95%% 分位 = %.2f   （中位 %.2f）" % (zc, np.median(maxz)))
    print("  纯失败规则的最大支持度 95%% 分位 = %d   （中位 %d）"
          % (pc, int(np.median(maxpure))))

    zz = np.where(ok, z, -np.inf)
    order = np.argsort(zz.ravel())[::-1][:15]
    print("\n=== 场景校正后最强的合取 ===")
    print("  规则                      支持  实际失败  场景期望    z     超阈值")
    for f in order:
        a, b = np.unravel_index(f, z.shape)
        nm = lit_name(a) if a == b else "%s & %s" % (lit_name(a), lit_name(b))
        print("  %-24s  %4d   %4d    %6.1f   %6.2f   %s"
              % (nm, sup[a, b], obs[a, b], exp[a, b], z[a, b],
                 "是" if z[a, b] > zc else ""))

    with np.errstate(invalid="ignore", divide="ignore"):
        pure = ok & (obs == sup) & (exp / np.maximum(sup, 1) < 0.999)
    print("\n=== \"一定不成功\" 的规则（触发的局全部失败）===")
    if not pure.any():
        print("  没有任何支持度>=%d 的合取是全失败的" % MIN_SUP)
    else:
        pf = np.where(pure, sup, -1)
        for f in np.argsort(pf.ravel())[::-1][:10]:
            a, b = np.unravel_index(f, z.shape)
            if pf[a, b] < 0:
                break
            m = fires[:, a, b]
            nm = lit_name(a) if a == b else "%s & %s" % (lit_name(a), lit_name(b))
            print("  %-24s  支持 %3d 局，全失败；涉及 %d 个场景（基准失败率 %.2f）  %s"
                  % (nm, sup[a, b], len(np.unique(scene[m])), p_fail[m].mean(),
                     "超阈值" if sup[a, b] > pc else "<= 零假设"))
    print("\n  纯失败规则要超过 %d 局支持度才比随机强；纯成功方向做同样的检查:" % pc)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = ok & (obs == 0) & (exp / np.maximum(sup, 1) > 0.001)
    print("  纯成功规则最大支持度 = %d"
          % (int(np.where(ps, sup, 0).max()) if ps.any() else 0))

    # --- held-out on scenes ---------------------------------------------------
    sc = np.unique(scene)
    half = len(sc) // 2
    mine = np.isin(scene, sc[:half])
    test = ~mine
    s1, o1, e1, z1 = score(fires[mine], fail[mine], p_fail[mine])
    ok1 = (s1 >= 10) & ok
    best = np.unravel_index(np.argmax(np.where(ok1, z1, -np.inf)), z.shape)
    s2, o2, e2, z2 = score(fires[test], fail[test], p_fail[test])
    print("\n=== 留出检验：前 %d 个场景挖，后 %d 个场景验 ===" % (half, len(sc) - half))
    print("  挖出来最强的: %s  (挖集 z=%.2f)"
          % ("%s & %s" % (lit_name(best[0]), lit_name(best[1])), z1[best]))
    print("  在留出场景上: 支持 %d，实际失败 %d，期望 %.1f，z=%.2f"
          % (s2[best], o2[best], e2[best], z2[best]))

    # --- what does the strongest rule mean, in pose? --------------------------
    a, b = np.unravel_index(np.argmax(zz), z.shape)
    hit = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
    print("\n=== 最强规则 %s & %s 对应什么位姿 ==="
          % (lit_name(a), lit_name(b)))
    P = D["prop"]
    inn, out = P[hit], P[~hit]
    print("  触发 %d 个控制步 / 共 %d" % (hit.sum(), hit.size))
    print("  维        触发时均值   未触发均值   差 / 标准差")
    for j in range(8):
        d = (inn[:, j].mean() - out[:, j].mean()) / (P[:, :, j].std() + 1e-9)
        print("  %-9s %9.4f   %9.4f      %+.2f" % (DIM[j], inn[:, j].mean(),
                                                   out[:, j].mean(), d))
    st = np.flatnonzero(hit.any(0))
    if len(st):
        print("  触发的控制步范围: %d - %d（中位 %d）"
              % (st.min(), st.max(), int(np.median(np.flatnonzero(hit)[0] % WIN))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
