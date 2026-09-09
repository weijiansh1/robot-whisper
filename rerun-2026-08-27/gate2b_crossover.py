#!/usr/bin/env python3
"""Gate 2b: case-crossover for the drop event.

The matched-success design failed: drops happen at chunk 39-50, by which point
only 1-23 successful rollouts are still running, so "predicting the drop" was
really "detecting that this rollout has not ended yet". Episode termination on
success destroys the control group.

Case-crossover removes that entirely. Each rollout is its own control. Within a
rollout, the case period is the chunk whose execution dropped the object; the
control periods are that same rollout's earlier chunks during which the object
was also inside the goal region and nothing happened. Scene, seed, policy,
object identity and "the object is at the target" are all held fixed by design.

    case:     chunk k*, the one that caused the drop
    controls: chunks j in [entry, k*) with the object inside the goal region

Only query-time information is admissible: o_k, A_k, R_k, h_k.

The statistic is the within-rollout rank of the case among that rollout's own
periods, averaged over rollouts, so a rollout contributing 28 control periods
does not outweigh one contributing 1. The null re-designates which period is
the case, inside each rollout, which is the exact exchangeability the design
assumes.
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
    lens = np.asarray([r["inference_calls"] for r in rows], np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    npz = {e: np.load(core.episode_path(args.run, episodes[e]), allow_pickle=True)
           for e in range(len(rows))}
    goal = [np.mean([npz[e]["sim_state"][-1, P] for e in range(len(rows)) if ok[e]], 0)
            for P in (P1, P2)]
    md = json.loads(args.modes.read_text())
    key = [k for k in md if "SCENE8" in k][0]
    mode = {d["episode"]: d["mode"] for d in md[key]["detail"]}

    # case and control periods, per rollout
    design = {}
    for e in range(len(rows)):
        if ok[e] or mode.get(e) != "reached_then_lost":
            continue
        for P, g in zip((P1, P2), goal):
            d = np.linalg.norm(npz[e]["sim_state"][:, P] - g, axis=1)
            ins = d <= 0.05
            if not ins.any():
                continue
            j = int(np.flatnonzero(ins)[0])
            a = np.flatnonzero(~ins[j:])
            if not len(a):
                continue
            exit_k = j + int(a[0])
            case = exit_k - 1                       # the chunk that did it
            ctrl = [c for c in range(j, case)]      # same rollout, object still placed
            if ctrl:
                design[e] = (case, ctrl)
            break
    print("case-crossover：%d 条 rollout，case %d 个，对照期 %d 个"
          % (len(design), len(design), sum(len(c) for _, c in design.values())), flush=True)

    store = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    hidden = zarr.open(str(args.run / "server/hidden.zarr"), mode="r")

    def feats(e, k):
        s = npz[e]["state"][k]
        sim = npz[e]["sim_state"][k]
        rel = np.concatenate([sim[P1] - s[:3], sim[P2] - s[:3]])
        st = np.concatenate([s, sim[1:], rel,
                             [np.linalg.norm(sim[P1] - s[:3]),
                              np.linalg.norm(sim[P2] - s[:3])]])
        act = npz[e]["actions"][k].ravel()
        r = np.asarray(store["hb_router_probs"][off[e] + k, :, -1], np.float32)
        r = np.maximum(r, 0.0)
        r /= np.maximum(r.sum(-1, keepdims=True), 1e-12)
        ent = -np.sum(r * np.log(np.maximum(r, 1e-12)), -1)
        rt = np.concatenate([r.reshape(-1)[::7], ent.ravel()])
        h = np.asarray(hidden["hb_hidden"][off[e] + k, :, -1], np.float32).reshape(-1)[::37]
        return st, act, rt, h

    per = {}
    for e, (case, ctrl) in design.items():
        rowsf = [feats(e, k) for k in [case] + ctrl]
        per[e] = {"case": 0, "n": len(rowsf),
                  "S": np.array([r[0] for r in rowsf], np.float64),
                  "A": np.array([r[1] for r in rowsf], np.float64),
                  "R": np.array([r[2] for r in rowsf], np.float64),
                  "H": np.array([r[3] for r in rowsf], np.float64)}
        print("  ep%-4d case+对照 %d" % (e, len(rowsf)), flush=True)

    def ranks_all(mats, case_idx):
        """[dim] mean over rollouts of the case period's within-rollout rank."""
        acc = None
        for X, ci in zip(mats, case_idx):
            v = X[ci]                                   # [dim]
            # exclude the case itself from the ties, and keep the rank in [0,1]
            r = ((X < v).sum(0) + 0.5 * ((X == v).sum(0) - 1)) / max(len(X) - 1, 1)
            r = np.clip(r, 0.0, 1.0)
            acc = r if acc is None else acc + r
        return acc / len(mats)

    for e, d in per.items():
        case, ctrl = design[e]
        d["T"] = np.array([[k] for k in [case] + ctrl], np.float64)   # sentinel: chunk index
    SRC = {"哨兵·chunk索引": lambda d: d["T"], "state": lambda d: d["S"], "state+action": lambda d: np.hstack([d["S"], d["A"]]),
           "route": lambda d: d["R"], "hidden": lambda d: d["H"],
           "joint": lambda d: np.hstack([d["S"], d["A"], d["R"]])}
    rng = np.random.default_rng(args.seed)
    out = []
    print("\n%-14s %6s %12s %10s" % ("信息源", "维度", "最强|rank−.5|", "maxT p"))
    for name, fn in SRC.items():
        mats = [fn(d) for d in per.values()]            # precomputed once
        ns = [len(X) for X in mats]
        dim = mats[0].shape[1]
        obs = np.abs(ranks_all(mats, [0] * len(mats)) - .5)
        peak = float(obs.max())
        hits = 0
        for _ in range(PERMS // 10):
            ci = [int(rng.integers(0, n)) for n in ns]
            hits += int(np.abs(ranks_all(mats, ci) - .5).max() >= peak)
        p = (1 + hits) / (PERMS // 10 + 1)
        out.append({"source": name, "dim": int(dim), "peak": peak, "p": float(p)})
        print("%-14s %6d %12.3f %10.4f%s" % (name, dim, peak, p, "  ✅" if p < .05 else ""),
              flush=True)

    args.out.write_text(json.dumps(
        {"rollouts": len(design), "controls": sum(len(c) for _, c in design.values()),
         "results": out}, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
