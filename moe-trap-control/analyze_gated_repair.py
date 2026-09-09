#!/usr/bin/env python3
"""Internal decision (alarm not cleared and routing frozen at alarm+5) applied to wait-after-alarm fork branches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from adaptive_control import RouteRisk
from collection_storage import atomic_json

FROZEN_M = 0.10


def hell(a, b):
    return float(np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2.0) / np.sqrt(a.shape[0]))


def decision(features, meta, main_id, wait, threshold):
    a = meta[main_id]["knn20"]
    with np.load(features / ("%s.npz" % main_id)) as z:
        knn, sp = z["knn"], z["state_probs"]
    t = a + wait
    if t >= len(knn):
        return None
    cleared = bool(knn[t] < threshold)
    lag = hell(sp[t], sp[t - 3])
    return dict(cleared=cleared, frozen=lag < FROZEN_M, lag=lag, knn=float(knn[t]), repair=(not cleared) and lag < FROZEN_M)


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    wait = int(plan["wait_queries"])
    meta = json.loads((args.features / "meta.json").read_text())
    threshold = RouteRisk().threshold
    rows = []
    for task in plan["tasks"]:
        d = decision(args.features, meta, task["main_id"], wait, threshold)
        result_path = args.run / "tasks" / task["main_id"] / "branches/result.json"
        if d is None or not result_path.exists():
            continue
        for b in json.loads(result_path.read_text())["branches"]:
            rows.append(dict(main_id=task["main_id"], failed=task["failed"], base_task=task["base_task"], arm=b["arm"],
                             replicate=b["replicate"], timing=b["timing"], success_window=bool(b["success"]),
                             success_original=bool(b["success_within_original"]), physical_class=b["physical_class"], **d))
    summary = dict(run=str(args.run), wait=wait, rule="repair iff kNN at alarm+%d >= threshold and lag-3 state-routing change < %.2f" % (wait, FROZEN_M),
                   branches=len(rows))
    table = {}
    for failed in (True, False):
        for flag in (True, False):
            for arm in plan["arms"]:
                sub = [r for r in rows if r["failed"] == failed and r["repair"] == flag and r["arm"] == arm]
                if sub:
                    table["%s|%s|%s" % ("failed" if failed else "success", "repair" if flag else "leave", arm)] = dict(
                        n=len(sub), window=sum(r["success_window"] for r in sub), original=sum(r["success_original"] for r in sub),
                        parents=len({r["main_id"] for r in sub}))
    summary["table"] = table
    # policy value: repair flagged with A4, leave unflagged as new_noise
    def outcome(r):
        return r["success_window"]
    value = {}
    for endpoint in ("success_window", "success_original"):
        v = {}
        for failed in (True, False):
            chosen = [r for r in rows if r["failed"] == failed and r["arm"] == ("retract_above_target" if r["repair"] else "new_noise")]
            ungated = [r for r in rows if r["failed"] == failed and r["arm"] == "retract_above_target"]
            baseline = [r for r in rows if r["failed"] == failed and r["arm"] == "new_noise"]
            v["failed" if failed else "success"] = dict(gated="%d/%d" % (sum(r[endpoint] for r in chosen), len(chosen)),
                                                       always_repair="%d/%d" % (sum(r[endpoint] for r in ungated), len(ungated)),
                                                       never_repair="%d/%d" % (sum(r[endpoint] for r in baseline), len(baseline)))
        value[endpoint] = v
    summary["policy_value"] = value
    atomic_json(args.out, summary)
    print(json.dumps(dict(rule=summary["rule"], branches=len(rows))))
    for k, v in table.items():
        print("  %-42s n=%3d B=%2d A=%2d parents=%2d" % (k, v["n"], v["window"], v["original"], v["parents"]))
    print("  policy value:", json.dumps(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("design/post_alarm_routing_20260909"))
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
