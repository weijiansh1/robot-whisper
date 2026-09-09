#!/usr/bin/env python3
"""Can MoE tell which KIND of failure a rollout is heading for?

Every confound fought in this session came from comparing failures against
successes: successes end early, so "still running" leaks, the common cohort is
capped, and case periods are always the latest. Comparing failure types against
each other removes all of it at once.

In scene8 all 216 failures run to exactly 52 chunks - no exceptions - so there
is no survivorship at any horizon, and horizons past t34 become usable for the
first time.

    label     event-type failure (drop / misplace, n=18)
              vs stasis failure (n=198)
    control   same initial state, same chunk index
    inputs    query-time only, same blocks and readout as everywhere else

This is the signal the operational goal actually needs: given a rollout already
flagged as failing, which failure mode is it, and how early can that be told.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZONS = (12, 20, 27, 34, 41, 48)
DRAWS = 20000
MOE = ("route_identity", "route_geometry", "route_temporal",
       "hidden_identity", "hidden_structure", "hidden_temporal")


def extract_failures(run, episodes, cache, seed):
    """core.extract_cache restricted to rollouts long enough for these horizons."""
    import zarr
    if cache.exists():
        with np.load(cache, allow_pickle=False) as p:
            if tuple(p["horizons"].tolist()) == tuple(core.HORIZONS):
                return {n: np.asarray(p[n], np.float32) for n in core.BLOCK_NAMES}
    routes = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    hid_s = zarr.open(str(run / "server" / "hidden.zarr"), mode="r")
    need = max(core.HORIZONS) + 1
    with np.load(core.episode_path(run, episodes[0]), allow_pickle=False) as p0:
        n_pro = int(np.asarray(p0["state"]).shape[1])
        n_sim = int(np.asarray(p0["sim_state"]).shape[1]) - 1
    n, hc = len(episodes), len(core.HORIZONS)
    A = {k: np.zeros((n, hc, core.PROJECTED_DIM), np.float32)
         for k in core.BLOCK_NAMES if k not in ("proprio", "sim")}
    A["proprio"] = np.zeros((n, hc, n_pro * 5), np.float32)
    A["sim"] = np.zeros((n, hc, n_sim * 5), np.float32)
    sk, ss = {}, {k: seed + 1009 * (i + 1) for i, k in enumerate(core.BLOCK_NAMES)}
    done = 0
    for ep in episodes:
        if ep.length < need:
            continue                                  # a success; never used below
        r = core.normalize_probability(np.asarray(
            routes["hb_router_probs"][ep.offset:ep.offset + need], np.float32))
        h = np.asarray(hid_s["hb_hidden"][ep.offset:ep.offset + need], np.float32)
        with np.load(core.episode_path(run, ep), allow_pickle=False) as p:
            pro = np.asarray(p["state"][:need], np.float32)
            sim = np.asarray(p["sim_state"][:need, 1:], np.float32)
        for hi, H in enumerate(core.HORIZONS):
            lo = H - core.HISTORY + 1
            cr, ch = r[H], h[H]
            raw = {"route_identity": core.route_identity(cr),
                   "route_geometry": core.route_geometry(cr),
                   "route_temporal": core.route_temporal(r[lo:H + 1]),
                   "hidden_identity": core.hidden_identity(ch),
                   "hidden_structure": core.hidden_structure(ch),
                   "hidden_temporal": core.hidden_temporal(h[lo:H + 1]),
                   "route_state": np.concatenate([core.route_identity(cr, slice(0, 1)),
                       core.route_temporal(r[lo:H + 1], slice(0, 1))]),
                   "route_action_1_3": np.concatenate([core.route_identity(cr, slice(1, 4)),
                       core.route_temporal(r[lo:H + 1], slice(1, 4))]),
                   "route_action_4_7": np.concatenate([core.route_identity(cr, slice(4, 8)),
                       core.route_temporal(r[lo:H + 1], slice(4, 8))]),
                   "route_action_8_10": np.concatenate([core.route_identity(cr, slice(8, 11)),
                       core.route_temporal(r[lo:H + 1], slice(8, 11))])}
            for k, v in raw.items():
                A[k][ep.index, hi] = core.make_sketch(k, v, sk, ss[k])
            A["proprio"][ep.index, hi] = core.physical_history(pro[lo:H + 1])
            A["sim"][ep.index, hi] = core.physical_history(sim[lo:H + 1])
        done += 1
        if done % 32 == 0:
            print("    extracted %d" % done, flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **A, horizons=np.asarray(core.HORIZONS, np.int16))
    return A


def extract_action(run, episodes):
    out = np.zeros((len(episodes), len(core.HORIZONS), 70 * 5), np.float32)
    for ep in episodes:
        if ep.length < max(core.HORIZONS) + 1:
            continue
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
    for g in groups:
        y = lab[g]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, sc[g]) * w
        den += w
    return num / den if den else np.nan


def paired(lab, L, R, groups):
    a = b = 0.0
    den = 0
    for g in groups:
        y = lab[g]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        a += auc(y, L[g]) * w
        b += auc(y, R[g]) * w
        den += w
    return (a - b) / den if den else np.nan


def boot(fn, groups, seed=20260827):
    rng = np.random.default_rng(seed)
    v = []
    tries = 0
    while len(v) < DRAWS and tries < DRAWS * 30:
        tries += 1
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
    ap.add_argument("--modes", type=pathlib.Path, default=HERE / "failure_modes.json")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    core.HORIZONS = HORIZONS
    # core.load_episodes demands the prefix from *every* episode; successes are
    # only 35 chunks long. They are excluded from this cohort anyway, so build
    # the list directly and simply never read a horizon they do not have.
    _rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                   key=lambda r: int(r["episode_index"]))
    _len = np.asarray([r["inference_calls"] for r in _rows], np.int64)
    _off = np.r_[0, np.cumsum(_len)[:-1]]
    episodes = [core.Episode(index=int(r["episode_index"]), state=int(r["init_state_id"]),
                             noise_seed=int(r["flow_noise_seed"]),
                             failure=int(not r["success"]), offset=int(o), length=int(L))
                for r, o, L in zip(_rows, _off, _len)]
    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    ok = np.asarray([bool(r["success"]) for r in rows])
    lens = np.asarray([r["inference_calls"] for r in rows], np.int64)
    md = json.loads(args.modes.read_text())
    key = [k for k in md if "SCENE8" in k][0]
    mode = {d["episode"]: d["mode"] for d in md[key]["detail"]}

    fail = np.flatnonzero(~ok)
    print("失败 %d 条；长度 min %d max %d  → %s"
          % (len(fail), lens[fail].min(), lens[fail].max(),
             "全部等长，无幸存者偏差" if lens[fail].min() == lens[fail].max() else "长度不齐！"),
          flush=True)
    event = np.array([mode.get(e) in ("reached_then_lost", "misplaced") for e in range(len(rows))])
    print("事件型 %d（掉落 %d + 放错 %d） vs 停滞型 %d"
          % (event[fail].sum(),
             sum(mode.get(e) == "reached_then_lost" for e in fail),
             sum(mode.get(e) == "misplaced" for e in fail),
             (~event[fail]).sum()), flush=True)

    cache = HERE / "type_features.npz"
    blocks = extract_failures(args.run, episodes, cache, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([e.state for e in episodes], np.int16)
    seeds = np.asarray([e.noise_seed for e in episodes], np.int16)

    inc = ~ok                                   # cohort: failures only
    folds = core.double_holdout_folds(state, seeds, inc)
    lab = event.astype(int)
    groups = [np.flatnonzero(inc & (state == v)) for v in np.unique(state)]
    groups = [g for g in groups if len(np.unique(lab[g])) == 2]
    print("可用 initial state %d（该状态下两类失败都有）" % len(groups), flush=True)

    BASE = ("proprio", "action")
    sc = {}
    for h in HORIZONS:
        idx = HORIZONS.index(h)
        for name, fam in (("base", BASE), ("moe", BASE + MOE)):
            m = core.build_fold_models(blocks, fam, idx, folds, args.seed + 613 * idx)
            sc[(h, name)] = core.predict(m, lab)
        print("  scored t%d" % h, flush=True)

    print("\n在「已知会失败」的 rollout 之间区分类型（事件型 vs 停滞型）")
    print("%6s %10s %10s %9s" % ("chunk", "非特权基线", "加 MoE", "差"))
    out = []
    for h in HORIZONS:
        b = weighted(lab, sc[(h, "base")], groups)
        m = weighted(lab, sc[(h, "moe")], groups)
        out.append({"horizon": h, "base_auc": b, "moe_auc": m})
        print("%6d %10.3f %10.3f %+9.3f" % (h, b, m, m - b), flush=True)

    for tag, hs in (("t20–t34", (20, 27, 34)), ("t34–t48", (34, 41, 48))):
        fn = lambda g, hs=hs: float(np.mean(
            [paired(lab, sc[(h, "moe")], sc[(h, "base")], g) for h in hs]))
        pt = fn(groups)
        lo, hi, p = boot(fn, groups)
        print("合并 %s  MoE 增量 %+.3f  95%% CI [%+.3f, %+.3f]  p=%.4f%s"
              % (tag, pt, lo, hi, p, "  ✅" if lo > 0 else ""), flush=True)
        out.append({"window": tag, "delta": pt, "ci": [lo, hi], "p": p})

    args.out.write_text(json.dumps(
        {"n_event": int(event[fail].sum()), "n_stasis": int((~event[fail]).sum()),
         "states": len(groups), "results": out}, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
