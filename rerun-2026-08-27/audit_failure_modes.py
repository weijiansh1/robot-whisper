#!/usr/bin/env python3
"""What actually goes wrong, per task - is every failure the robot getting stuck?

Target objects are identified empirically: an object counts as a target if
successful rollouts move it more than MOVED_M from where it started. The goal
pose for each target is the mean final pose over successful rollouts.

Each failure is then classified by what happened to the targets, using only
recorded simulator state:

  never_grasped     no target ever rose more than LIFT_M above its start
  lifted_not_placed a target was lifted but never came within GOAL_M of its goal
  reached_then_lost a target got within GOAL_M and then left it again
  partial           some targets placed, others not (multi-target tasks only)
  misplaced         targets moved a long way but ended far from any goal

Cross-checked against the stasis-trap label used everywhere else, so the two
definitions can be compared rather than assumed equal.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


CACHE = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
MOVED_M = 0.03
LIFT_M = 0.01
GOAL_M = 0.05
PROGRESS_EPS_M = 0.01


def load_task(run: pathlib.Path):
    rows = sorted(json.loads((run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    layout = json.loads((run / "client/sim_layout.json").read_text())
    objs = [(j["joint"], j["state_lo"]) for j in layout["joints"]
            if not j["is_robot"] and j["state_hi"] - j["state_lo"] == 7]
    sims = []
    for e in range(len(rows)):
        with np.load(run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
            sims.append(np.asarray(p["sim_state"], np.float32))
    return rows, objs, sims


def classify(run: pathlib.Path):
    rows, objs, sims = load_task(run)
    ok = np.asarray([bool(r["success"]) for r in rows])
    if not ok.any() or ok.all():
        return None

    # which objects must move, and where they end up when things go right
    targets = []
    for name, lo in objs:
        sl = slice(lo, lo + 3)
        moved = np.mean([np.linalg.norm(sims[e][-1, sl] - sims[e][0, sl])
                         for e in range(len(rows)) if ok[e]])
        if moved > MOVED_M:
            goal = np.mean([sims[e][-1, sl] for e in range(len(rows)) if ok[e]], 0)
            targets.append({"name": name, "sl": sl, "goal": goal, "moved": float(moved)})

    counts, detail = {}, []
    for e in range(len(rows)):
        if ok[e]:
            continue
        s = sims[e]
        placed = lifted = reached_lost = 0
        travel = 0.0
        for t in targets:
            p = s[:, t["sl"]]
            lift = float((p[:, 2] - p[0, 2]).max())
            d = np.linalg.norm(p - t["goal"], axis=1)
            travel += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            if lift > LIFT_M:
                lifted += 1
            if d[-1] <= GOAL_M:
                placed += 1
            elif d.min() <= GOAL_M:
                reached_lost += 1
        n = len(targets)
        if lifted == 0:
            mode = "never_grasped"
        elif reached_lost:
            mode = "reached_then_lost"
        elif placed and placed < n:
            mode = "partial"
        elif travel > 0.40:
            mode = "misplaced"
        else:
            mode = "lifted_not_placed"
        counts[mode] = counts.get(mode, 0) + 1
        detail.append({"episode": e, "mode": mode, "lifted": lifted,
                       "placed": placed, "travel": round(travel, 3)})

    # the stasis-trap label used by every other analysis, for comparison
    term = [(int(rows[i]["init_state_id"]), int(rows[i]["flow_noise_seed"]),
             np.concatenate([sims[i][-1, t["sl"]] for t in targets]))
            for i in range(len(rows)) if ok[i]]
    stasis = {}
    for d in detail:
        e = d["episode"]
        ref = np.stack([p for st, sd, p in term
                        if st != int(rows[e]["init_state_id"])
                        and sd != int(rows[e]["flow_noise_seed"])])
        pose = np.concatenate([sims[e][:, t["sl"]] for t in targets], axis=1)
        dist = np.linalg.norm(pose[:, None, :] - ref[None, :, :], axis=2).min(1)
        best = np.minimum.accumulate(dist)
        future = np.minimum.accumulate(dist[::-1])[::-1]
        elig = np.flatnonzero((best - future <= PROGRESS_EPS_M) & (best > GOAL_M))
        d["stasis"] = bool(len(elig))
        stasis[d["mode"]] = stasis.get(d["mode"], 0) + (1 if len(elig) else 0)

    return {"targets": [t["name"] for t in targets], "n_fail": int((~ok).sum()),
            "counts": counts, "stasis": stasis, "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    out = {}
    for p in sorted(CACHE.glob("*/*/right-16x32/client/summaries.json")):
        run = p.parents[1]
        task = str(run.relative_to(CACHE).parent)
        res = classify(run)
        if res is None:
            print("%s: 全成功，跳过" % task)
            continue
        out[task] = res
        print("\n%s" % task)
        print("  目标物体: %s" % ", ".join(res["targets"]))
        print("  失败 %d 条：" % res["n_fail"])
        for mode, n in sorted(res["counts"].items(), key=lambda kv: -kv[1]):
            st = res["stasis"].get(mode, 0)
            print("    %-20s %3d 条   其中被判为停滞陷入 %d (%.0f%%)"
                  % (mode, n, st, 100 * st / n))
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
