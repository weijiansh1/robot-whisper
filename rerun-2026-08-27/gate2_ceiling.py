#!/usr/bin/env python3
"""Gate 2: is a drop predictable *before* the chunk that causes it runs?

Causal framing, strictly. sim_state[k] is recorded before chunk k executes
(verified: the commanded displacement of A_k correlates 0.977 with s[k+1]-s[k]).
An object that is inside the goal region at k-1 and outside at k therefore left
during chunk k-1, so the deciding query is k-1 and the only admissible inputs
are o_{k-1}, R_{k-1}, A_{k-1}.

    Y^pre_k = 1[executing A_k causes the drop]

Nested information sources answer which of four worlds we are in:
    state alone near chance          -> not predictable at query time at all
    state+action predicts, route not -> risk is in the plan, router hides it
    route predicts but adds nothing  -> router is a compressed state shadow
    route adds over state+action     -> genuine internal sensor

Controls are matched: successful rollouts from the same initial state evaluated
at the same chunk index, so a "risk" signal cannot just be task phase.

n = 12. This is a ceiling probe, not a powered test; read it against the
minimum detectable effect from gate 1.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
P1, P2 = slice(10, 13), slice(17, 20)
PERMS = 20000


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(lab, col, groups):
    num = den = 0.0
    for g in groups:
        y = lab[g]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, col[g]) * w
        den += w
    return num / den if den else np.nan


def maxstat(F, lab, groups, rng, perms=PERMS):
    obs = np.abs(np.array([weighted(lab, F[:, j], groups)
                           for j in range(F.shape[1])]) - .5)
    peak = float(obs.max())
    hits = 0
    for _ in range(perms):
        p = lab.copy()
        for g in groups:
            p[g] = rng.permutation(p[g])
        m = np.abs(np.array([weighted(p, F[:, j], groups)
                             for j in range(F.shape[1])]) - .5).max()
        hits += int(m >= peak)
    return peak, (1 + hits) / (perms + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=core.RUN)
    ap.add_argument("--modes", type=pathlib.Path, default=HERE / "failure_modes.json")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    episodes = core.load_episodes(args.run)
    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    ok = np.asarray([bool(r["success"]) for r in rows])
    state = np.asarray([e.state for e in episodes], np.int16)
    npz = {e: np.load(core.episode_path(args.run, episodes[e]), allow_pickle=True)
           for e in range(len(rows))}
    goal = [np.mean([npz[e]["sim_state"][-1, P] for e in range(len(rows)) if ok[e]], 0)
            for P in (P1, P2)]

    md = json.loads(args.modes.read_text())
    key = [k for k in md if "SCENE8" in k][0]
    mode = {d["episode"]: d["mode"] for d in md[key]["detail"]}

    def drop_chunk(e):
        for P, g in zip((P1, P2), goal):
            d = np.linalg.norm(npz[e]["sim_state"][:, P] - g, axis=1)
            ins = d <= 0.05
            if ins.any():
                j = int(np.flatnonzero(ins)[0])
                a = np.flatnonzero(~ins[j:])
                if len(a):
                    return j + int(a[0]) - 1        # the chunk whose execution did it
        return None

    ev = {}
    for e in range(len(rows)):
        if not ok[e] and mode.get(e) == "reached_then_lost":
            k = drop_chunk(e)
            if k is not None and k >= 1:
                ev[e] = k
    print("掉落事件 %d 条；致因 chunk 索引 %s" % (len(ev), sorted(ev.values())), flush=True)

    # matched controls: same initial state, same chunk index, successful
    pairs = []
    for e, k in ev.items():
        for c in np.flatnonzero(ok & (state == state[e])):
            if len(npz[c]["state"]) > k + 1:
                pairs.append((c, k, 0))
        pairs.append((e, k, 1))
    print("配对样本 %d（同初始状态、同 chunk 索引的成功对照）" % len(pairs), flush=True)

    store = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    hidden = zarr.open(str(args.run / "server/hidden.zarr"), mode="r")
    lens = np.asarray([r["inference_calls"] for r in rows], np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]

    S, A, R, H, lab, grp = [], [], [], [], [], []
    for e, k, y in pairs:
        s = npz[e]["state"][k]
        sim = npz[e]["sim_state"][k]
        rel = np.concatenate([sim[P1] - s[:3], sim[P2] - s[:3]])
        S.append(np.concatenate([s, sim[1:], rel,
                                 [np.linalg.norm(sim[P1] - s[:3]),
                                  np.linalg.norm(sim[P2] - s[:3])]]))
        A.append(npz[e]["actions"][k].ravel())
        r = np.asarray(store["hb_router_probs"][off[e] + k, :, -1], np.float32)
        r = np.maximum(r, 0); r /= np.maximum(r.sum(-1, keepdims=True), 1e-12)
        ent = -np.sum(r * np.log(np.maximum(r, 1e-12)), -1)
        R.append(np.concatenate([r.reshape(-1)[::7], ent.ravel()]))   # thinned + entropy
        h = np.asarray(hidden["hb_hidden"][off[e] + k, :, -1], np.float32)
        H.append(h.reshape(-1)[::37])
        lab.append(y); grp.append(state[e])
    S, A, R, H = map(lambda x: np.asarray(x, np.float64), (S, A, R, H))
    lab = np.asarray(lab); grp = np.asarray(grp)
    groups = [np.flatnonzero(grp == v) for v in np.unique(grp)]
    groups = [g for g in groups if len(np.unique(lab[g])) == 2]
    print("特征维度  state %d  action %d  route %d  hidden %d ｜ 可用 state %d"
          % (S.shape[1], A.shape[1], R.shape[1], H.shape[1], len(groups)), flush=True)

    FAM = {"state": S, "state+action": np.hstack([S, A]), "route": R,
           "hidden": H, "joint": np.hstack([S, A, R])}
    out = []
    rng = np.random.default_rng(args.seed)
    print("\n%-14s %6s %10s %9s" % ("信息源", "维度", "最强|AUC−.5|", "maxT p"))
    for name, F in FAM.items():
        F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
        peak, p = maxstat(F, lab, groups, np.random.default_rng(args.seed), perms=4000)
        out.append({"source": name, "dim": int(F.shape[1]), "peak": peak, "p": p})
        print("%-14s %6d %10.3f %9.4f%s"
              % (name, F.shape[1], peak, p, "  ✅" if p < .05 else ""), flush=True)

    args.out.write_text(json.dumps({"n_event": len(ev), "n_pairs": len(pairs),
                                    "states": len(groups), "results": out},
                                   indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
