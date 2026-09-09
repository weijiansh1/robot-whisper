#!/usr/bin/env python3
"""Audit (e): is the t08 rule riding a flow-noise-seed template?

Every scene in right-16x32 shares the same 32 flow-noise seeds (1000-1031), so a
scene-split validation leaves the seed set intact on both sides.  If some seeds
are globally luckier than others, a rule whose firing correlates with seed can
transfer across scenes without being state-specific.  Two checks, both from the
bundle alone (episode order is scene-major, seed = 1000 + episode_index % 32):

  1. seed main effect on outcome: variance of per-seed success rates across the
     16 scenes, against a within-scene permutation null.  If seeds are
     exchangeable this is flat.
  2. seed-disjoint rule validation: mine the conjunctions on 16 seeds (all
     scenes), score on the other 16 seeds, paired rule-minus-random exactly as
     in the honest loop.  If the increment survives with seeds disjoint, the
     rule is not a seed template.

Writes audit_seed_leak.json next to this script.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NPZ = (HERE / "himoe-routing-rules-20260819/data"
       / "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
N_PERM = 10000
N_REP = 20
TOPK = 20
RNG = np.random.default_rng(0)


def lit_name(i):
    return "L%d:e%d" % (HB_LAYER[i // 32], i % 32)


def load():
    z = np.load(NPZ, allow_pickle=True)
    n_rows = z["n_rows"]
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = z["success"].astype(bool)
    scene = z["scene"]
    n = len(y)
    win = int(n_rows.min())
    ids = z["state_token_top4"].astype(np.int64)          # (total, 8, 4)
    hot_all = np.zeros((ids.shape[0], 8, 32), np.float32)
    np.put_along_axis(hot_all, ids, 1.0, -1)
    M = np.zeros((n, win, 256), np.float32)
    for i in range(n):
        M[i] = hot_all[off[i]:off[i] + win].reshape(win, 256)
    seed = np.arange(n) % 32                              # scene-major order
    return y, scene, seed, M, win


def cooccur(M):
    n = M.shape[0]
    out = np.zeros((n, 256, 256), bool)
    for i in range(n):
        out[i] = (M[i].T @ M[i]) > 0.5
    return out


def score(fires, fail, p_fail):
    F = fires.astype(np.float32)
    sup = np.einsum("iab->ab", F)
    obs = np.einsum("i,iab->ab", fail.astype(np.float32), F)
    exp = np.einsum("i,iab->ab", p_fail, F)
    var = np.einsum("i,iab->ab", p_fail * (1 - p_fail), F)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - exp) / np.sqrt(np.maximum(var, 1e-9))
    return sup, obs, exp, z


def main() -> int:
    y, scene, seed, M, win = load()
    fail = ~y
    n = len(y)
    scenes = np.unique(scene)
    out = {"task": "libero_long/t08", "n": int(n), "win": int(win)}
    print("t08: %d eps, %d scenes, window 0-%d, success %.1f%%"
          % (n, len(scenes), win - 1, 100 * y.mean()))

    # ---- 1. seed main effect ------------------------------------------------
    per_seed = np.array([y[seed == s].mean() for s in range(32)])
    obs_var = per_seed.var()
    obs_range = per_seed.max() - per_seed.min()
    null_var = np.zeros(N_PERM)
    null_rng = np.random.default_rng(1)
    for p in range(N_PERM):
        yp = y.copy()
        for s in scenes:
            m = np.flatnonzero(scene == s)
            yp[m] = y[m[null_rng.permutation(len(m))]]
        ps = np.array([yp[seed == q].mean() for q in range(32)])
        null_var[p] = ps.var()
    p_var = float((1 + (null_var >= obs_var).sum()) / (1 + N_PERM))
    out["seed_main_effect"] = {
        "per_seed_success": per_seed.round(4).tolist(),
        "var": float(obs_var), "range": float(obs_range),
        "null_var_p50": float(np.median(null_var)),
        "null_var_p95": float(np.quantile(null_var, .95)),
        "perm_p": p_var}
    print("\n[1] seed main effect: var=%.5f (null p50 %.5f, p95 %.5f)  p=%.4f"
          % (obs_var, np.median(null_var), np.quantile(null_var, .95), p_var))
    print("    per-seed success range %.3f (%.2f-%.2f)"
          % (obs_range, per_seed.min(), per_seed.max()))

    # ---- 2. seed-disjoint rule validation ----------------------------------
    # residual scoring identical in spirit to the honest loop: residual is
    # fail - scene base rate (computed on the scoring half only), and the
    # statistic is rule-minus-random paired within the split.
    fires = cooccur(M)
    rand_hit = np.random.default_rng(3).random((n, win)) < 0.10

    def residual(mask_ep):
        r = np.zeros(mask_ep.sum(), np.float32)
        sc = scene[mask_ep]
        f = fail[mask_ep].astype(np.float32)
        for s in np.unique(sc):
            r[sc == s] = f[sc == s] - f[sc == s].mean()
        return r

    res_rows = {}   # per split cache

    def run_splits(split_fn, label):
        top_d, pool_d = [], []
        names = []
        for r in range(N_REP):
            g = np.random.default_rng(100 + r)
            A, B = split_fn(g)
            pfA = np.zeros(A.sum(), np.float32)
            scA = scene[A]
            for s in np.unique(scA):
                pfA[scA == s] = fail[A][scA == s].mean()
            sA, oA, eA, zA = score(fires[A], fail[A], pfA)
            ok = (sA >= 10) & (sA <= A.sum() - 3)
            ok[np.tril_indices(256, -1)] = False
            order = np.argsort(np.where(ok, zA, -np.inf).ravel())[::-1]
            a, b = np.unravel_index(order[0], zA.shape)
            names.append("%s & %s" % (lit_name(a), lit_name(b)))
            top = (M[:, :, a] > 0.5) & (M[:, :, b] > 0.5)
            pool = np.zeros((n, win), bool)
            for f in order[:TOPK]:
                u, v = np.unravel_index(f, zA.shape)
                pool |= (M[:, :, u] > 0.5) & (M[:, :, v] > 0.5)
            # episode-level residual vs scene base rate on the scoring half
            rB = residual(B)
            eB = np.flatnonzero(B)
            # step-level hit -> episode-level "fires anywhere" is too coarse;
            # score step-level like the honest loop: residual per episode
            # broadcast over its steps
            r_step = np.repeat(rB, win)
            hitB = top[B].ravel()
            poolB = pool[B].ravel()
            rndB = rand_hit[B].ravel()
            tv = r_step[hitB].mean() if hitB.sum() > 20 else np.nan
            pv = r_step[poolB].mean() if poolB.sum() > 20 else np.nan
            rv = r_step[rndB].mean()
            top_d.append(tv - rv)
            pool_d.append(pv - rv)
        top_d, pool_d = np.array(top_d), np.array(pool_d)
        res = {}
        for nm, d in (("top", top_d), ("pool", pool_d)):
            fin = np.isfinite(d)
            res[nm] = {"mean": float(np.nanmean(d)),
                       "sem": float(np.nanstd(d) / np.sqrt(fin.sum())),
                       "pos": int((d[fin] > 0).sum()), "n": int(fin.sum())}
        cnt = {}
        for nm in names:
            cnt[nm] = cnt.get(nm, 0) + 1
        res["top_rules"] = sorted(cnt.items(), key=lambda t: -t[1])[:3]
        print("  %-18s top %+0.4f±%.4f (%d/%d)   pool %+0.4f±%.4f (%d/%d)   %s"
              % (label, res["top"]["mean"], res["top"]["sem"],
                 res["top"]["pos"], res["top"]["n"],
                 res["pool"]["mean"], res["pool"]["sem"],
                 res["pool"]["pos"], res["pool"]["n"], res["top_rules"]))
        return res

    print("\n[2] rule-minus-random, paired within split "
          "(episode-level scene-centred residual, %d splits):" % N_REP)

    def by_scene(g):
        sc = g.permutation(scenes)
        A = np.isin(scene, sc[:len(sc) // 2])
        return A, ~A

    def by_seed(g):
        sd = g.permutation(32)
        A = np.isin(seed, sd[:16])
        return A, ~A

    out["scene_split"] = run_splits(by_scene, "scene-disjoint")
    out["seed_split"] = run_splits(by_seed, "seed-disjoint")

    (HERE / "audit_seed_leak.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_seed_leak.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
