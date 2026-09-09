#!/usr/bin/env python3
"""Does the "robot stopped making progress" detector work outside scene8?

The earlier claim that cross-task testing was blocked treated the control step
as a fixed ruler. It is not: the effect appears at t20 of a 52-step cap in
scene8, i.e. about 38% of the way in. Measured on each task's own timescale the
equivalent horizons sit inside that task's common cohort, where every rollout is
still running, so no survivorship bias is introduced.

    task                       cap   common cohort   38% of cap
    scene8                      52        t34            t20
    goal/open_top_drawer        30        t16            t11
    spatial/stove               22        t10             t8

Everything else is held identical to the scene8 analysis: stasis-trap labels
from future object poses, 4x4 initial-state/seed double holdout, per-block
whitened PCA-12, fixed ridge, pair-weighted within-initial-state AUC and an
initial-state cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


CACHE = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
DRAWS = 20000
MOE = ("route_identity", "route_geometry", "route_temporal",
       "hidden_identity", "hidden_structure", "hidden_temporal")
TASKS = {
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": (10, 12, 14, 16),
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (7, 8, 9, 10),
}


def extract_features(run, episodes, cache_path, seed):
    """core.extract_cache, but with the simulator width taken from the data.

    The original hard-codes `46 * 5` for the sim block, which is scene8's object
    count; the other tasks carry more objects (78 state dims) and overflow it.
    Everything else - blocks, sketch seeds, history window - is identical.
    """
    import zarr
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as p:
            if tuple(p["horizons"].tolist()) == tuple(core.HORIZONS):
                return {n: np.asarray(p[n], np.float32) for n in core.BLOCK_NAMES}

    routes = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    hidden_s = zarr.open(str(run / "server" / "hidden.zarr"), mode="r")
    rid = np.asarray(routes["episode_id"][:], np.int32)
    if not np.array_equal(rid, np.asarray(hidden_s["episode_id"][:], np.int32)):
        raise ValueError("route and hidden episode axes differ")

    with np.load(core.episode_path(run, episodes[0]), allow_pickle=False) as p0:
        n_pro = int(np.asarray(p0["state"]).shape[1])
        n_sim = int(np.asarray(p0["sim_state"]).shape[1]) - 1
    count, hc = len(episodes), len(core.HORIZONS)
    arrays = {n: np.empty((count, hc, core.PROJECTED_DIM), np.float32)
              for n in core.BLOCK_NAMES if n not in ("proprio", "sim")}
    arrays["proprio"] = np.empty((count, hc, n_pro * 5), np.float32)
    arrays["sim"] = np.empty((count, hc, n_sim * 5), np.float32)
    sketches = {}
    sseed = {n: seed + 1009 * (i + 1) for i, n in enumerate(core.BLOCK_NAMES)}

    for pos, ep in enumerate(episodes, 1):
        stop = ep.offset + max(core.HORIZONS) + 1
        if not np.all(rid[ep.offset:stop] == ep.index):
            raise ValueError("episode boundary mismatch for %d" % ep.index)
        route = core.normalize_probability(
            np.asarray(routes["hb_router_probs"][ep.offset:stop], np.float32))
        hid = np.asarray(hidden_s["hb_hidden"][ep.offset:stop], np.float32)
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            pro = np.asarray(p["state"][: max(core.HORIZONS) + 1], np.float32)
            sim = np.asarray(p["sim_state"][: max(core.HORIZONS) + 1, 1:], np.float32)
        for hi, h in enumerate(core.HORIZONS):
            lo = h - core.HISTORY + 1
            cr, ch = route[h], hid[h]
            raw = {
                "route_identity": core.route_identity(cr),
                "route_geometry": core.route_geometry(cr),
                "route_temporal": core.route_temporal(route[lo:h + 1]),
                "hidden_identity": core.hidden_identity(ch),
                "hidden_structure": core.hidden_structure(ch),
                "hidden_temporal": core.hidden_temporal(hid[lo:h + 1]),
                "route_state": np.concatenate([
                    core.route_identity(cr, slice(0, 1)),
                    core.route_temporal(route[lo:h + 1], slice(0, 1))]),
                "route_action_1_3": np.concatenate([
                    core.route_identity(cr, slice(1, 4)),
                    core.route_temporal(route[lo:h + 1], slice(1, 4))]),
                "route_action_4_7": np.concatenate([
                    core.route_identity(cr, slice(4, 8)),
                    core.route_temporal(route[lo:h + 1], slice(4, 8))]),
                "route_action_8_10": np.concatenate([
                    core.route_identity(cr, slice(8, 11)),
                    core.route_temporal(route[lo:h + 1], slice(8, 11))]),
            }
            for name, vec in raw.items():
                arrays[name][ep.index, hi] = core.make_sketch(name, vec, sketches, sseed[name])
            arrays["proprio"][ep.index, hi] = core.physical_history(pro[lo:h + 1])
            arrays["sim"][ep.index, hi] = core.physical_history(sim[lo:h + 1])
        if pos % 64 == 0:
            print("    extracted %d/%d" % (pos, count), flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays,
                        horizons=np.asarray(core.HORIZONS, np.int16))
    return arrays


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
    tries = 0
    while len(v) < DRAWS and tries < DRAWS * 20:
        tries += 1
        ch = rng.integers(0, len(groups), len(groups))
        d = [rng.choice(groups[a], len(groups[a]), replace=True) for a in ch]
        x = fn(d)
        if np.isfinite(x):
            v.append(x)
    v = np.asarray(v)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975)), float((v <= 0).mean())


def run_task(task: str, horizons, seed: int, out_dir: pathlib.Path):
    run = CACHE / task / "right-16x32"
    core.HORIZONS = tuple(horizons)
    core.RUN = run
    print("\n=== %s ===\n    horizons %s" % (task, horizons), flush=True)

    episodes = core.load_episodes(run)
    sims = core.load_sim(run, episodes)
    labels, included, _, meta = core.build_targets(episodes, sims)
    cache = out_dir / ("feat__" + task.replace("/", "__") + ".npz")
    blocks = extract_features(run, episodes, cache, seed)
    blocks["action"] = extract_action(run, episodes)
    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)
    folds = core.double_holdout_folds(state, seeds, included)

    groups = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    groups = [g for g in groups if len(np.unique(labels[g])) == 2]
    print("    停滞陷入 %d / 失败 %d ｜ 可用 initial state %d"
          % (meta["n_stasis_trap"], meta["n_failure"], len(groups)), flush=True)
    if len(groups) < 3:
        print("    可用簇太少，跳过统计")
        return {"task": task, "usable_states": len(groups), "skipped": True}

    sc = {}
    for h in horizons:
        idx = horizons.index(h)
        for name, fam in (("base", ("proprio", "action")),
                          ("moe", ("proprio", "action") + MOE)):
            m = core.build_fold_models(blocks, fam, idx, folds, seed + 613 * idx)
            sc[(h, name)] = core.predict(m, labels)

    rows = []
    for h in horizons:
        b = weighted(labels, sc[(h, "base")], groups)
        m = weighted(labels, sc[(h, "moe")], groups)
        rows.append({"horizon": h, "base_auc": b, "moe_auc": m})
        print("    t%-3d 非特权基线 %.3f   加 MoE %.3f   差 %+.3f" % (h, b, m, m - b), flush=True)

    late = horizons[1:]
    fn = lambda g: float(np.mean([paired(labels, sc[(h, "moe")], sc[(h, "base")], g)
                                  for h in late]))
    pt = fn(groups)
    lo, hi, p = boot(fn, groups)
    print("    合并 t%d–t%d 增量 %+.3f  95%% CI [%+.3f, %+.3f]  p=%.4f%s"
          % (late[0], late[-1], pt, lo, hi, p, "  排除 0" if lo > 0 else ""), flush=True)
    return {"task": task, "horizons": list(horizons), "usable_states": len(groups),
            "n_stasis": meta["n_stasis_trap"], "n_fail": meta["n_failure"],
            "per_horizon": rows, "pooled": {"window": list(late), "delta": pt,
                                            "ci": [lo, hi], "p_one_sided": p}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()
    out_dir = args.out.parent
    res = [run_task(t, h, args.seed, out_dir) for t, h in TASKS.items()]
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
