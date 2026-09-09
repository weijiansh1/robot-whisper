"""Task separability through the co-selection graph instead of the expert histogram.

`analyze_corpus_routing.py` separates the 30 tasks using per-layer expert
frequencies and gets between/within = 2.80 (equal-steps). That channel is the one
HB-Reg is explicitly trained to flatten -- the paper minimises a DeepSeekMoE
load-balancing loss and reports it converging near its uniform-utilisation lower
bound -- so it is the channel least likely to carry anything.

This runs the identical protocol on the *residual* channel: which experts share a
top-4, after a curveball null has removed exactly the marginal information the
histogram channel uses. Same corpus, same INDEX, same even/odd episode split, same
equal-steps subsampling, same same-suite / cross-suite grouping. Only the feature
and the metric change:

    histogram channel   per-layer expert frequencies      Jensen-Shannon distance
    graph channel       per-layer log co-selection lift   correlation distance

The two distances are not on a common scale, so compare the *ratios* rather than
the raw numbers: each is a between-task signal divided by its own within-task
noise floor, measured with the same split.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np
import zarr

from probe_routing_graph import TOP_K, co_selection, curveball, incidence

N_EXPERTS = 32
UPPER = np.triu_indices(N_EXPERTS, 1)


def load_index(root):
    with (pathlib.Path(root) / "INDEX.csv").open() as stream:
        return list(csv.DictReader(stream))


def task_sites(rows):
    """Per-control-step routing ids plus the episode each step belongs to."""
    path = pathlib.Path(rows[0]["path"])
    group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
    ids = np.asarray(group["hb_expert_ids"][:])
    episode_of, steps = [], []
    for row in rows:
        start = int(row["control_step_offset"])
        stop = start + int(row["inference_calls"])
        steps.extend(range(start, stop))
        episode_of.extend([int(row["episode_index"])] * (stop - start))
    return ids[np.asarray(steps)], np.asarray(episode_of)


def lift_matrix(sites, rng, n_null, trades_per_site=5):
    matrix = incidence(sites)
    observed = co_selection(matrix)
    trades = trades_per_site * matrix.shape[0]
    nulls = np.stack([co_selection(curveball(matrix, trades, rng)) for _ in range(n_null)])
    return np.log(np.maximum(observed, 0.5) / np.maximum(nulls.mean(0), 0.5))


def task_halves(ids, episode_of, rng, equal_steps, max_sites, n_null):
    """Even/odd EPISODE split, matching the histogram channel's split exactly."""
    n_layers, n_suffix = ids.shape[1], ids.shape[3]
    per_layer = []
    picked = []
    for parity in (0, 1):
        take = np.flatnonzero(episode_of % 2 == parity)
        if equal_steps is not None:
            if len(take) < equal_steps:
                raise ValueError("half has %d steps, need %d" % (len(take), equal_steps))
            take = rng.choice(take, equal_steps, replace=False)
        picked.append(take)
    for layer in range(n_layers):
        halves = []
        for take in picked:
            sites = ids[take][:, layer, :, 1:n_suffix, :].reshape(-1, TOP_K)
            if sites.shape[0] > max_sites:
                sites = sites[rng.choice(sites.shape[0], max_sites, replace=False)]
            halves.append(lift_matrix(sites, rng, n_null))
        per_layer.append(halves)
    return per_layer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="corpus/libero30-right-v1")
    ap.add_argument("--equal-steps", type=int, default=192)
    ap.add_argument("--max-sites", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default="analysis/libero30-right-v1-eq192/summary.json")
    ap.add_argument("--out", default="analysis/graph-separability")
    args = ap.parse_args()

    rows_by_task = defaultdict(list)
    for row in load_index(args.root):
        rows_by_task[(row["suite"], row["task_dir"])].append(row)
    keys = sorted(rows_by_task)
    if not keys:
        raise SystemExit("INDEX.csv is empty")

    rng = np.random.default_rng(args.seed)
    per_task = {}
    for suite, task_dir in keys:
        ids, episode_of = task_sites(rows_by_task[(suite, task_dir)])
        per_task[(suite, task_dir)] = task_halves(
            ids, episode_of, rng, args.equal_steps, args.max_sites, args.n_null)
        print("estimated %-8s %s" % (suite, task_dir[:56]), flush=True)

    n_layers = len(per_task[keys[0]])
    # distance = 1 - r over the off-diagonal expert pairs, averaged over layers
    def distance(a, b):
        return float(np.mean([1.0 - np.corrcoef(a[layer][UPPER], b[layer][UPPER])[0, 1]
                              for layer in range(n_layers)]))

    within = {k: distance([h[0] for h in per_task[k]], [h[1] for h in per_task[k]])
              for k in keys}
    same_suite, cross_suite, matrix = [], [], np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                # matched halves on both sides: same sampling noise as the floor
                value = float(np.mean([
                    distance([h[side] for h in per_task[a]], [h[side] for h in per_task[b]])
                    for side in (0, 1)]))
                matrix[i, j] = matrix[j, i] = value
                (same_suite if a[0] == b[0] else cross_suite).append(value)

    floor = float(np.mean(list(within.values())))
    summary = {
        "n_tasks": len(keys),
        "equal_steps": args.equal_steps,
        "within_task_split_half_distance": {
            "mean": floor,
            "max": float(np.max(list(within.values()))),
            "min": float(np.min(list(within.values())))},
        "between_task_same_suite": {"mean": float(np.mean(same_suite)),
                                    "min": float(np.min(same_suite)),
                                    "n_pairs": len(same_suite)},
        "between_task_cross_suite": {"mean": float(np.mean(cross_suite)),
                                     "min": float(np.min(cross_suite)),
                                     "n_pairs": len(cross_suite)},
        "separability_ratio_same_suite_over_within": float(np.mean(same_suite)) / floor,
    }

    print("\n=== graph channel (co-selection lift, correlation distance) ===")
    print("within-task split-half   mean %.4f   (noise floor)" % floor)
    print("between-task same suite  mean %.4f   min %.4f   (%d pairs)"
          % (summary["between_task_same_suite"]["mean"],
             summary["between_task_same_suite"]["min"], len(same_suite)))
    print("between-task cross suite mean %.4f   min %.4f   (%d pairs)"
          % (summary["between_task_cross_suite"]["mean"],
             summary["between_task_cross_suite"]["min"], len(cross_suite)))
    print("ratio between/within = %.2f"
          % summary["separability_ratio_same_suite_over_within"])

    reference = pathlib.Path(args.reference)
    if reference.exists():
        histogram = json.loads(reference.read_text())["summary"]
        print("\n=== against the histogram channel, same split, same equal-steps ===")
        print("histogram (expert frequencies, JS)  ratio %.2f"
              % histogram["separability_ratio_same_suite_over_within"])
        print("graph     (co-selection lift, 1-r)  ratio %.2f"
              % summary["separability_ratio_same_suite_over_within"])
        summary["histogram_ratio"] = histogram["separability_ratio_same_suite_over_within"]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary,
         "within_task": {"%s/%s" % k: v for k, v in within.items()}}, indent=1))
    np.savez_compressed(out / "distances.npz",
                        tasks=np.array(["%s/%s" % k for k in keys]), between=matrix,
                        within=np.array([within[k] for k in keys]))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
