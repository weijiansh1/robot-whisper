#!/usr/bin/env python3
"""Can the flat 50x8 corpus carry the trap-onset experiment?

The frozen loop rule and the goal references are lifted verbatim from
``analyze_rolling_star_experiment`` / ``analyze_trap_onset_sweep``; only the
corpus reader changes, because the flat grid has no snapshot/branch structure
and its per-query series come straight out of the client npz (one row per
inference call) rather than rolling-star's query index.

Reports, per task: how many episodes trip the loop rule, how many trip the
static rule, and -- the part that decides whether the experiment is runnable --
how those events distribute over the (task, scene) strata that are the flat
grid's only stand-in for a rolling-star snapshot.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

# frozen constants, copied from analyze_rolling_star_experiment.py:99-108
STATIC_WINDOW = 20
STATIC_EEF_PATH_M = 0.020
STATIC_OBJECT_PATH_M = 0.005
STATIC_GRIPPER_PATH_M = 0.001
LOOP_MIN_QUERY_LAG = 3
LOOP_EEF_RETURN_M = 0.045
LOOP_OBJECT_RETURN_M = 0.030
LOOP_GRIPPER_RETURN_M = 0.012
LOOP_EEF_PATH_M = 0.120
LOOP_MAX_PROGRESS_M = 0.035

HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA")
RUN_ID = "right-50x8-20260903"


def targets_of(layout: dict) -> list[dict]:
    """Free joints of the objects of interest -- same selection rolling-star uses."""
    want = set(layout.get("obj_of_interest") or [])
    out = []
    for j in layout["joints"]:
        if j["is_robot"] or j["state_hi"] - j["state_lo"] < 7:
            continue
        if any(j["joint"].startswith(name) for name in want):
            out.append(j)
    return out


def series(npz, sim, targets):
    """eef / objects / gripper per query, straight from the client npz."""
    eef = npz["state"][:, :3].astype(np.float32)
    gripper = npz["state"][:, 6:8].astype(np.float32).mean(axis=1)
    objects = np.stack([sim[:, int(t["state_lo"]):int(t["state_lo"]) + 3]
                        for t in targets], axis=1).astype(np.float32)
    return eef, objects, gripper


def goal_distance(objects, references):
    """max over objects of the min distance to any pooled success terminal."""
    d = np.linalg.norm(objects[:, :, None, :] - references[None, None, :, :], axis=-1)
    return d.min(axis=2).max(axis=1)


def loop_onset(eef, objects, gripper, goal):
    """First query index satisfying the frozen loop rule; -1 if never."""
    n = len(eef)
    if n <= LOOP_MIN_QUERY_LAG:
        return -1
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(eef, axis=0), axis=1))]
    for right in range(LOOP_MIN_QUERY_LAG, n):
        lo = np.arange(0, right - LOOP_MIN_QUERY_LAG + 1)
        ok = ((np.linalg.norm(eef[right] - eef[lo], axis=1) <= LOOP_EEF_RETURN_M)
              & (np.linalg.norm(objects[right] - objects[lo], axis=2).max(axis=1)
                 <= LOOP_OBJECT_RETURN_M)
              & (np.abs(gripper[right] - gripper[lo]) <= LOOP_GRIPPER_RETURN_M)
              & (cum[right] - cum[lo] >= LOOP_EEF_PATH_M)
              & (goal[lo] - goal[right] <= LOOP_MAX_PROGRESS_M))
        if ok.any():
            return right
    return -1


def static_onset(eef, objects, gripper):
    """First query index ending a STATIC_WINDOW-query frozen stretch; -1 if never."""
    n = len(eef)
    if n <= STATIC_WINDOW:
        return -1
    for right in range(STATIC_WINDOW, n):
        left = right - STATIC_WINDOW
        w_eef = np.linalg.norm(np.diff(eef[left:right + 1], axis=0), axis=1).sum()
        w_obj = np.linalg.norm(np.diff(objects[left:right + 1], axis=0), axis=2).sum(0).max()
        w_grip = np.abs(np.diff(gripper[left:right + 1])).sum()
        if (w_eef <= STATIC_EEF_PATH_M and w_obj <= STATIC_OBJECT_PATH_M
                and w_grip <= STATIC_GRIPPER_PATH_M):
            return right
    return -1


def run_task(run: pathlib.Path):
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    layout = json.loads((run / "client/sim_layout.json").read_text())
    targets = targets_of(layout)
    if not targets:
        return None
    cache = {}
    for s in S:
        p = run / ("client/episode_%02d.npz" % s["episode_index"])
        if not p.is_file():
            continue
        d = np.load(p)
        k = int(s["inference_calls"])
        if k < 1:
            continue
        cache[s["episode_index"]] = (d["state"][:k], d["sim_state"][:k])

    succ = [s for s in S if s["success"] and s["episode_index"] in cache]
    if not succ:
        return None
    refs = np.concatenate([
        np.stack([cache[s["episode_index"]][1][-1, int(t["state_lo"]):int(t["state_lo"]) + 3]
                  for s in succ]) for t in targets]).astype(np.float32)

    rows = []
    for s in S:
        i = s["episode_index"]
        if i not in cache:
            continue
        st, sim = cache[i]
        eef, objects, gripper = series({"state": st}, sim, targets)
        g = goal_distance(objects, refs)
        rows.append(dict(ep=i, scene=s["init_state_id"], seed=s["flow_noise_seed"],
                         success=s["success"], n_query=len(eef),
                         loop=loop_onset(eef, objects, gripper, g),
                         static=static_onset(eef, objects, gripper)))
    return rows


def main() -> int:
    only = sys.argv[1:] or None
    runs = sorted(HUB.glob("libero_*/*/" + RUN_ID))
    print("%-13s %-44s %5s %5s %5s %5s %5s %6s %6s"
          % ("suite", "task", "eps", "fail", "loop", "stat", "any", "strata", "usable"))
    tot = Counter()
    for run in runs:
        suite, task = run.parts[-3], run.parts[-2]
        if only and not any(o in task for o in only):
            continue
        try:
            rows = run_task(run)
        except Exception as exc:                       # noqa: BLE001
            print("%-13s %-44s  ERROR %s" % (suite, task[:44], exc))
            continue
        if not rows:
            continue
        ev = [r for r in rows if r["loop"] >= 0 or r["static"] >= 0]
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            by[r["scene"]][0] += 1
            by[r["scene"]][1] += int(r["loop"] >= 0 or r["static"] >= 0)
        usable = sum(1 for n, k in by.values() if 0 < k < n)   # strata with both classes
        tot["eps"] += len(rows); tot["fail"] += sum(1 for r in rows if not r["success"])
        tot["loop"] += sum(1 for r in rows if r["loop"] >= 0)
        tot["static"] += sum(1 for r in rows if r["static"] >= 0)
        tot["any"] += len(ev); tot["strata"] += len(by); tot["usable"] += usable
        print("%-13s %-44s %5d %5d %5d %5d %5d %6d %6d"
              % (suite, task[:44], len(rows), sum(1 for r in rows if not r["success"]),
                 sum(1 for r in rows if r["loop"] >= 0),
                 sum(1 for r in rows if r["static"] >= 0), len(ev), len(by), usable),
              flush=True)
    print()
    print("total: %d eps, %d fail, loop %d, static %d, any-trap %d, "
          "%d scene strata of which %d carry both classes"
          % (tot["eps"], tot["fail"], tot["loop"], tot["static"], tot["any"],
             tot["strata"], tot["usable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
