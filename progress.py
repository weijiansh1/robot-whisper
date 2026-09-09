#!/usr/bin/env python3
"""Progress of a capture run, with every label read from meta.json.

Written because a dozen progress reports carried the wrong task id: the plan
listed libero_goal:3,0 so the first Goal task was assumed to be t03, but the
driver runs sorted(tasks) and started with t00.  The directory name was right
and the number was right; only the id in the prose was invented.  Nothing here
is typed by hand -- task_id, task_name and status all come from meta.json, and
the baseline comes from the table below keyed on that id.

Usage: ./progress.py [run-id]
"""

from __future__ import annotations

import collections
import glob
import json
import os
import pathlib
import sys

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"

#: our own 500-episode paper-right results, per task id, for context
BASELINE = {
    "libero_10": [49, 50, 50, 44, 48, 50, 43, 47, 31, 50],
    "libero_goal": [50, 49, 50, 45, 50, 50, 48, 50, 48, 50],
    "libero_spatial": [50, 50, 49, 48, 48, 45, 48, 45, 47, 47],
    "libero_object": [48, 50, 49, 47, 49, 48, 45, 48, 49, 50],
}
#: hub directory name -> benchmark id, to look the baseline up
BENCH = {"libero_long": "libero_10", "libero_goal": "libero_goal",
         "libero_spatial": "libero_spatial", "libero_object": "libero_object"}


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "right-16x32"
    total = designed = 0
    rows = []

    # The full plan, so the denominator is the run's size and not just the size
    # of whatever has started.  Without this the total reads 1536 while three of
    # five tasks are underway, then jumps as each new task begins.
    plan_path = HUB / "_pipeline" / run_id / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    planned_total = (plan["n_tasks"] * plan["episodes_per_task"]) if plan else None
    started = set()

    for path in sorted(glob.glob(str(HUB / "cache/*/*/*" / run_id / "meta.json"))):
        run = pathlib.Path(path).parent
        meta = json.loads(pathlib.Path(path).read_text())
        try:
            summaries = json.loads((run / "client/summaries.json").read_text())
        except (OSError, ValueError):
            summaries = []

        n = len(summaries)
        ok = sum(1 for s in summaries if s.get("success"))
        want = int(meta.get("episodes_requested") or 0)
        total += n
        designed += want

        by_scene = collections.defaultdict(lambda: [0, 0])
        for s in summaries:
            st = s.get("init_state_id")
            by_scene[st][1] += 1
            by_scene[st][0] += bool(s.get("success"))
        draws = int(meta.get("draws_per_scene") or 1)
        full = [(st, a) for st, (a, b) in sorted(by_scene.items()) if b == draws]

        hub_dir = run.parents[1].name
        bench = BENCH.get(hub_dir, hub_dir)
        tid = meta.get("task_id")
        base = (BASELINE.get(bench) or [None] * 10)[tid] if tid is not None else None
        started.add((bench, tid))
        rows.append((bench, tid, meta.get("task_name", ""), meta.get("status"),
                     n, want, ok, base, full, draws))

    for bench, tid, name, status, n, want, ok, base, full, draws in rows:
        rate = "%5.1f%%" % (100.0 * ok / n) if n else "    -"
        mark = " COMPLETE" if status == "complete" else ""
        base_s = "base %d/50" % base if base is not None else ""
        print("%-15s t%02d  %4d/%-4d %s  %-11s %s%s"
              % (bench, tid, n, want, rate, base_s, name[:38], mark))
        if full:
            print("      scenes(%d draws): %s"
                  % (draws, "  ".join("s%d:%d" % (st, a) for st, a in full)))

    if plan:
        pending = [(s, t) for s in plan["suites"] for t in plan["select"][s]
                   if (s, t) not in started]
        for s, t in sorted(pending):
            print("%-15s t%02d     -/%-4d      -   %-11s (not started)"
                  % (s, t, plan["episodes_per_task"], ""))
        designed = planned_total
    pct = 100.0 * total / designed if designed else 0.0
    st = os.statvfs(HUB)
    print("\ntotal %d/%d = %.1f%%   disk %.1f GB free"
          % (total, designed, pct, st.f_bavail * st.f_frsize / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
