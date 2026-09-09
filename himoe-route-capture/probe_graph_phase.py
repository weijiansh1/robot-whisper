"""Does the co-selection graph move *within* a task, or is it just a task label?

`analyze_graph_separability.py` asks whether two tasks draw different graphs. Even
a clear yes leaves the question that decides whether a graph feature could ever be
an online control signal: within one task, does the graph change as the episode
progresses, or is it one fixed graph per task that says nothing about the current
state?

Split each task's control steps by phase within the episode -- first third against
last third, by fraction so that unequal episode lengths do not bias the split --
and compare the early/late graph distance against the split-half floor measured
inside each phase at the same sample size:

    floor    d(early even-episodes, early odd-episodes)  and the same for late
    signal   d(early, late) at matched parity

ratio near 1 means the graph is a task constant; well above 1 means it tracks the
phase of the motion, i.e. it is a state signal that a controller could read.

Reusing the parity split inside each phase keeps init-state variation in the floor,
so a graph that merely differs between init states does not count as phase
tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np
import zarr

from analyze_graph_separability import UPPER, lift_matrix, load_index
from probe_routing_graph import TOP_K


def phase_split(rows, fraction=1.0 / 3.0):
    """Control-step indices for the first and last `fraction` of each episode."""
    path = pathlib.Path(rows[0]["path"])
    group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
    ids = np.asarray(group["hb_expert_ids"][:])
    early, late = [], []
    for row in rows:
        start = int(row["control_step_offset"])
        calls = int(row["inference_calls"])
        episode = int(row["episode_index"])
        cut = max(1, int(round(calls * fraction)))
        if calls < 3:
            continue  # too short to have a distinguishable early and late third
        early += [(start + i, episode) for i in range(cut)]
        late += [(start + i, episode) for i in range(calls - cut, calls)]
    return ids, np.asarray(early), np.asarray(late)


def group_lift(ids, entries, layer, rng, n_sites, max_sites, n_null):
    take = entries[:, 0]
    if len(take) > n_sites:
        take = rng.choice(take, n_sites, replace=False)
    sites = ids[take][:, layer, :, 1:, :].reshape(-1, TOP_K)
    if sites.shape[0] > max_sites:
        sites = sites[rng.choice(sites.shape[0], max_sites, replace=False)]
    return lift_matrix(sites, rng, n_null)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="corpus/libero30-right-v1")
    ap.add_argument("--task", action="append",
                    help="task_dir to include; default is every task in INDEX")
    ap.add_argument("--max-sites", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/graph-phase/summary.json")
    args = ap.parse_args()

    rows_by_task = defaultdict(list)
    for row in load_index(args.root):
        rows_by_task[(row["suite"], row["task_dir"])].append(row)
    keys = sorted(k for k in rows_by_task if not args.task or k[1] in args.task)
    if not keys:
        raise SystemExit("no matching task in INDEX.csv")

    rng = np.random.default_rng(args.seed)
    results = []
    print("%-8s %-42s  floor   early-vs-late   ratio" % ("suite", "task"))
    for suite, task_dir in keys:
        ids, early, late = phase_split(rows_by_task[(suite, task_dir)])
        groups = {}
        # equalise every one of the four cells to the smallest, so the floor and
        # the signal are estimated from the same number of control steps
        sizes = [int(np.sum(entries[:, 1] % 2 == parity))
                 for entries in (early, late) for parity in (0, 1)]
        n_steps = min(sizes)
        n_layers = ids.shape[1]
        for name, entries in (("early", early), ("late", late)):
            for parity in (0, 1):
                cell = entries[entries[:, 1] % 2 == parity]
                groups[(name, parity)] = [
                    group_lift(ids, cell, layer, rng, n_steps, args.max_sites, args.n_null)
                    for layer in range(n_layers)]

        def distance(a, b):
            return float(np.mean([1.0 - np.corrcoef(a[layer][UPPER], b[layer][UPPER])[0, 1]
                                  for layer in range(n_layers)]))

        floor = float(np.mean([distance(groups[("early", 0)], groups[("early", 1)]),
                               distance(groups[("late", 0)], groups[("late", 1)])]))
        signal = float(np.mean([distance(groups[("early", p)], groups[("late", p)])
                                for p in (0, 1)]))
        results.append({"suite": suite, "task": task_dir, "n_steps_per_cell": n_steps,
                        "floor": floor, "early_vs_late": signal,
                        "ratio": signal / floor if floor > 0 else None})
        print("%-8s %-42s  %.4f  %.4f         %.2f"
              % (suite, task_dir[:42], floor, signal, signal / max(floor, 1e-12)),
              flush=True)

    ratios = np.array([r["ratio"] for r in results if r["ratio"]])
    print("\nmean ratio %.2f over %d tasks  (1.0 = the graph is a task constant)"
          % (ratios.mean(), len(ratios)))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tasks": results, "mean_ratio": float(ratios.mean())}, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
