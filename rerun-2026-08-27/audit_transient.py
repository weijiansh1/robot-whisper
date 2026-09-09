#!/usr/bin/env python3
"""Do transient (event-shaped) MoE readouts work at all, and do they see events?

Every MoE feature used so far is a *level*: a mean over a trailing window. That
is the right shape for stasis, where something stops and stays stopped, and the
wrong shape for a mistake, where one step goes wrong and the rest look normal -
a single bad step inside a ten-step mean is diluted tenfold. That is a
structural reason the level readout scores 0.557 on the misplace/drop failures,
not a feature-selection accident.

So this builds an explicitly transient block - order statistics and jumps rather
than means - and runs it in two stages:

  1 VALIDATE  on stasis vs success (n=197 vs 296), where the answer is known.
    If the transient block cannot reproduce a real increment there, the tool is
    broken and nothing it says about events can be trusted.

  2 APPLY     to the 19 non-stasis failures. n is far too small to train on, so
    this is a per-feature within-state comparison with a permutation-controlled
    max-statistic, and is explicitly exploratory.

Same folds, same readout, same cluster bootstrap as every other contrast.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (20, 27, 34)
WINDOW = core.HISTORY          # 7 control steps, same as every other block
DRAWS = 20000
PERMS = 20000
MOE_LEVEL = ("route_identity", "route_geometry", "route_temporal",
             "hidden_identity", "hidden_structure", "hidden_temporal")
STAT_NAMES = (
    "hard_max", "hard_min", "hard_range", "hard_jolt",
    "soft_max", "soft_min", "soft_range", "soft_jolt",
    "ent_max", "ent_min", "ent_range", "ent_jolt",
    "innov_max", "innov_last",
)


def transient_block(run: pathlib.Path, episodes, horizons, cache: pathlib.Path):
    """[episode, horizon, 8 layers x 14 stats x 2 token-aggregations]."""
    if cache.exists():
        with np.load(cache, allow_pickle=False) as p:
            if tuple(p["horizons"].tolist()) == tuple(horizons):
                return np.asarray(p["transient"], np.float32)

    store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    n_layer = 8
    out = np.empty((len(episodes), len(horizons), n_layer * len(STAT_NAMES) * 2), np.float32)

    for pos, ep in enumerate(episodes, 1):
        lo0, hi0 = ep.offset, ep.offset + max(horizons) + 1
        p = np.asarray(store["hb_router_probs"][lo0:hi0], np.float32)   # [T,L,F,Tok,E]
        p = np.maximum(p, 0.0)
        p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
        p = p[:, :, -1]                                                 # final denoise: [T,L,Tok,E]
        ids = np.asarray(store["hb_expert_ids"][lo0:hi0, :, -1], np.int64)  # [T,L,Tok,4]
        hot = np.zeros(ids.shape[:-1] + (p.shape[-1],), bool)
        np.put_along_axis(hot, ids, True, -1)

        ent = -np.sum(p * np.log(np.maximum(p, 1e-12)), -1)             # [T,L,Tok]
        inter = np.logical_and(hot[1:], hot[:-1]).sum(-1)
        union = np.logical_or(hot[1:], hot[:-1]).sum(-1)
        hard = 1.0 - inter / np.maximum(union, 1)                       # [T-1,L,Tok]
        root = np.sqrt(p)
        soft = np.sqrt(np.maximum(0.5 * np.square(root[1:] - root[:-1]).sum(-1), 0.0))

        # innovation: how far the current distribution sits from an EMA of its past
        ema = np.empty_like(p)
        ema[0] = p[0]
        for t in range(1, len(p)):
            ema[t] = 0.5 * ema[t - 1] + 0.5 * p[t - 1]
        innov = np.sqrt(np.maximum(0.5 * np.square(np.sqrt(p) - np.sqrt(ema)).sum(-1), 0.0))

        for hi, h in enumerate(horizons):
            a, b = h - WINDOW + 1, h + 1            # steps; diffs indexed one lower
            H, S, E, I = hard[a - 1:b - 1], soft[a - 1:b - 1], ent[a:b], innov[a:b]
            feats = []
            for X in (H, S, E):
                jolt = np.abs(np.diff(X, axis=0)).max(0) if len(X) > 1 else np.zeros_like(X[0])
                feats += [X.max(0), X.min(0), X.max(0) - X.min(0), jolt]
            feats += [I.max(0), I[-1]]
            stack = np.stack(feats, 0)              # [14, L, Tok]
            out[ep.index, hi] = np.concatenate(
                [stack.mean(-1).ravel(), stack.max(-1).ravel()]).astype(np.float32)
        if pos % 64 == 0:
            print("    transient %d/%d" % (pos, len(episodes)), flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, transient=out, horizons=np.asarray(horizons, np.int16))
    return out


def extract_action(run, episodes):
    out = np.empty((len(episodes), len(core.HORIZONS), 70 * 5), np.float32)
    for ep in episodes:
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            a = np.asarray(p["actions"][: max(core.HORIZONS) + 1], np.float32).reshape(-1, 70)
        for hi, h in enumerate(core.HORIZONS):
            out[ep.index, hi] = core.physical_history(a[h - core.HISTORY + 1: h + 1])
    return out


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(lab, sc, groups):
    num = den = 0.0
    for i in groups:
        y = lab[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, sc[i]) * w
        den += w
    return num / den if den else np.nan


def paired(lab, L, R, groups):
    a = b = 0.0
    den = 0
    for i in groups:
        y = lab[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        a += auc(y, L[i]) * w
        b += auc(y, R[i]) * w
        den += w
    return (a - b) / den if den else np.nan


def boot(fn, groups, seed=20260827):
    rng = np.random.default_rng(seed)
    v = []
    while len(v) < DRAWS:
        ch = rng.integers(0, len(groups), len(groups))
        d = [rng.choice(groups[a], len(groups[a]), replace=True) for a in ch]
        x = fn(d)
        if np.isfinite(x):
            v.append(x)
    v = np.asarray(v)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975)), float((v <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=core.RUN)
    ap.add_argument("--cache", type=pathlib.Path,
                    default=HERE / "token-dynamics/early_structure_features.npz")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    episodes = core.load_episodes(args.run)
    sims = core.load_sim(args.run, episodes)
    stasis, included, _, _ = core.build_targets(episodes, sims)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    blocks["transient"] = transient_block(
        args.run, episodes, HORIZONS, HERE / "transient_block.npz")
    print("transient block %s" % (blocks["transient"].shape,), flush=True)

    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)
    fail = np.asarray([e.failure for e in episodes], np.int8)
    other = (fail == 1) & (stasis == 0)
    succ = fail == 0
    folds = core.double_holdout_folds(state, seeds, included)
    groups = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    groups = [g for g in groups if len(np.unique(stasis[g])) == 2]

    # ---- stage 1: does the transient readout work where the answer is known ----
    BASE = ("proprio", "action")
    FAMS = {"非特权基线": BASE,
            "＋ 水平量 MoE": BASE + MOE_LEVEL,
            "＋ 瞬时量 MoE": BASE + ("transient",),
            "＋ 两者": BASE + MOE_LEVEL + ("transient",)}
    sc = {}
    for h in HORIZONS:
        # transient has its own horizon axis; the rest use core.HORIZONS
        for name, fam in FAMS.items():
            idx = {n: (HORIZONS.index(h) if n == "transient" else core.HORIZONS.index(h))
                   for n in fam}
            parts_tr, parts_te = [], []
            models = []
            for fold_i, (tr, te, _) in enumerate(folds):
                a, b = [], []
                for bi, n in enumerate(fam):
                    x, y = core.pca_block(blocks[n][tr, idx[n]], blocks[n][te, idx[n]],
                                          args.seed + 101 * fold_i + 7919 * bi)
                    a.append(x)
                    b.append(y)
                X, Y = np.concatenate(a, 1).astype(np.float64), np.concatenate(b, 1).astype(np.float64)
                G = X.T @ X
                inv = np.linalg.solve(G + core.RIDGE_ALPHA * np.eye(G.shape[0]), X.T)
                models.append(core.FoldModel(te, tr, Y @ inv, X @ inv))
            sc[(h, name)] = core.predict(models, stasis)
        print("  scored t%d" % h, flush=True)

    print("\n阶段 1 · 验证：停滞陷入 vs 成功（已知答案）")
    print("%-16s %8s %8s %8s   %s" % ("特征", "t20", "t27", "t34", "合并增量 vs 非特权基线"))
    rows = []
    for name in FAMS:
        aucs = [weighted(stasis, sc[(h, name)], groups) for h in HORIZONS]
        line = "%-16s %8.3f %8.3f %8.3f" % (name, *aucs)
        if name != "非特权基线":
            fn = lambda g, n=name: float(np.mean(
                [paired(stasis, sc[(h, n)], sc[(h, "非特权基线")], g) for h in HORIZONS]))
            pt = fn(groups)
            lo, hi, p = boot(fn, groups)
            line += "   %+.3f [%+.3f, %+.3f] p=%.4f%s" % (
                pt, lo, hi, p, "  ✅" if lo > 0 else "")
            rows.append({"family": name, "auc": aucs, "delta": pt, "ci": [lo, hi], "p": p})
        else:
            rows.append({"family": name, "auc": aucs})
        print(line)

    # ---- stage 2: exploratory look at the 19 event failures ----
    T = blocks["transient"]
    names = ["%s|L%d|%s" % (agg, (2,3,4,5,12,13,14,15)[l], STAT_NAMES[s])
             for agg in ("tokmean", "tokmax") for s in range(len(STAT_NAMES))
             for l in range(8)]
    hi34 = HORIZONS.index(34)
    def per_feature(mask, tag):
        """Stage 2's own procedure, so it can be validated on a known answer."""
        g = [np.flatnonzero((succ | mask) & (state == v)) for v in np.unique(state)]
        g = [x for x in g if len(np.unique(mask[x])) == 2]
        lab = mask.astype(int)
        obs = np.array([weighted(lab, T[:, hi34, j], g) for j in range(T.shape[2])])
        rng = np.random.default_rng(args.seed)
        null = np.empty(PERMS)
        for k in range(PERMS):
            perm = lab.copy()
            for x in g:
                perm[x] = rng.permutation(perm[x])
            # max over ALL features, not one random feature. The previous form
            # produced a per-feature 95th percentile and left the 224-way
            # multiplicity uncontrolled; the corrected family-wise result is in
            # transient_familywise.json (events 0/224, maxT p=0.173).
            null[k] = np.abs(np.array([weighted(perm, T[:, hi34, j], g)
                                       for j in range(T.shape[2])]) - .5).max() + .5
        dev = np.abs(obs - 0.5)
        thr = float(np.quantile(np.abs(null - 0.5), 0.95))
        order = np.argsort(-dev)
        n_pass = int((dev > thr).sum())
        print("\n%s ｜ n=%d ｜ 可用 state %d ｜ 零分布 95%% 分位 |AUC−0.5|=%.3f"
              % (tag, int(mask.sum()), len(g), thr))
        for j in order[:4]:
            print("    %-30s AUC %.3f  |偏离| %.3f%s"
                  % (names[j], obs[j], dev[j], "  超过" if dev[j] > thr else ""))
        print("    %d/%d 特征超过零分布 95%% 分位（随机期望 %.0f）"
              % (n_pass, len(obs), 0.05 * len(obs)))
        return {"tag": tag, "n": int(mask.sum()), "states": len(g), "threshold": thr,
                "n_pass": n_pass, "n_features": len(obs),
                "top": [{"feature": names[j], "auc": float(obs[j])} for j in order[:10]]}

    print("\n阶段 1b · 用阶段 2 的逐特征程序验证（已知答案）")
    v1b = per_feature(stasis.astype(bool), "停滞陷入 vs 成功")
    print("\n阶段 2 · 应用（探索性，n=%d）" % other.sum())

    def per_feature(mask, tag):
        """Stage 2's own procedure, so it can be validated on a known answer."""
        g = [np.flatnonzero((succ | mask) & (state == v)) for v in np.unique(state)]
        g = [x for x in g if len(np.unique(mask[x])) == 2]
        lab = mask.astype(int)
        obs = np.array([weighted(lab, T[:, hi34, j], g) for j in range(T.shape[2])])
        rng = np.random.default_rng(args.seed)
        null = np.empty(PERMS)
        for k in range(PERMS):
            perm = lab.copy()
            for x in g:
                perm[x] = rng.permutation(perm[x])
            # max over ALL features, not one random feature. The previous form
            # produced a per-feature 95th percentile and left the 224-way
            # multiplicity uncontrolled; the corrected family-wise result is in
            # transient_familywise.json (events 0/224, maxT p=0.173).
            null[k] = np.abs(np.array([weighted(perm, T[:, hi34, j], g)
                                       for j in range(T.shape[2])]) - .5).max() + .5
        dev = np.abs(obs - 0.5)
        thr = float(np.quantile(np.abs(null - 0.5), 0.95))
        order = np.argsort(-dev)
        n_pass = int((dev > thr).sum())
        print("\n%s ｜ n=%d ｜ 可用 state %d ｜ 零分布 95%% 分位 |AUC−0.5|=%.3f"
              % (tag, int(mask.sum()), len(g), thr))
        for j in order[:4]:
            print("    %-30s AUC %.3f  |偏离| %.3f%s"
                  % (names[j], obs[j], dev[j], "  超过" if dev[j] > thr else ""))
        print("    %d/%d 特征超过零分布 95%% 分位（随机期望 %.0f）"
              % (n_pass, len(obs), 0.05 * len(obs)))
        return {"tag": tag, "n": int(mask.sum()), "states": len(g), "threshold": thr,
                "n_pass": n_pass, "n_features": len(obs),
                "top": [{"feature": names[j], "auc": float(obs[j])} for j in order[:10]]}

    print("\n阶段 1b · 用阶段 2 的逐特征程序验证（已知答案）")
    v1b = per_feature(stasis.astype(bool), "停滞陷入 vs 成功")
    print("\n阶段 2 · 应用（探索性，n=%d）" % other.sum())
    v2 = per_feature(other, "非停滞失败 vs 成功")

    args.out.write_text(json.dumps({
        "stage1": rows,
        "stage1b_validation": v1b, "stage2_events": v2,
    }, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
