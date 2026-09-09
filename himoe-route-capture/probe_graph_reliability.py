"""Is the co-selection graph task-specific, or one graph the model always draws?

`probe_routing_graph.py` establishes that experts co-select far more structurally
than their marginals imply.  That alone does not say whether a graph feature would
carry any per-state information: a co-selection graph fixed by the weights is a
constant, and a constant predicts nothing.

Comparing two tasks' graphs directly cannot answer it either, because a noisy
estimate correlates poorly with itself.  So this uses the same split-half design
the corpus JS analysis uses:

    within-task r   two halves of the SAME task            <- the reliability ceiling
    cross-task r    matched halves of two DIFFERENT tasks  <- the quantity of interest

    cross / within  ~1  the graph is shared; no per-task signal
                    ~0  the graph is task-specific

Halves are split by control step, so each half keeps the within-control-step
correlation that inflates naive z-scores.  The reliability ceiling therefore
absorbs the pseudo-replication instead of requiring it to be modelled.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

from probe_routing_graph import TOP_K, co_selection, curveball, incidence

N_EXPERTS = 32
UPPER = np.triu_indices(N_EXPERTS, 1)


def lift_matrix(sites, rng, n_null=8, trades_per_site=5):
    matrix = incidence(sites)
    observed = co_selection(matrix)
    trades = trades_per_site * matrix.shape[0]
    nulls = np.stack([co_selection(curveball(matrix, trades, rng)) for _ in range(n_null)])
    return np.log(np.maximum(observed, 0.5) / np.maximum(nulls.mean(0), 0.5))


def halves(ids, layer, token_slice, rng, max_sites):
    """Even/odd control steps, each flattened over denoise rounds and tokens."""
    n_steps = ids.shape[0]
    out = []
    for parity in (0, 1):
        take = np.arange(parity, n_steps, 2)
        sites = ids[take][:, layer, :, token_slice, :].reshape(-1, TOP_K)
        if sites.shape[0] > max_sites:
            sites = sites[rng.choice(sites.shape[0], max_sites, replace=False)]
        out.append(sites)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--token", choices=("action", "state"), default="action")
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--max-sites", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/routing-graph/reliability.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    names, per_task = [], []
    for task in args.task:
        path = pathlib.Path(task)
        group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
        ids = np.asarray(group["hb_expert_ids"][:])
        n_suffix = ids.shape[3]
        token_slice = slice(1, n_suffix) if args.token == "action" else slice(0, 1)
        layers = []
        for layer in range(ids.shape[1]):
            first, second = halves(ids, layer, token_slice, rng, args.max_sites)
            layers.append((lift_matrix(first, rng, args.n_null),
                           lift_matrix(second, rng, args.n_null)))
        per_task.append(layers)
        names.append(path.name)
        print("estimated %s" % path.name[:60], flush=True)

    n_layers = len(per_task[0])
    report = {"tasks": names, "token": args.token, "layers": []}
    print("\nlayer  within-task r   cross-task r   cross/within")
    for layer in range(n_layers):
        within = [float(np.corrcoef(per_task[t][layer][0][UPPER],
                                    per_task[t][layer][1][UPPER])[0, 1])
                  for t in range(len(names))]
        cross = []
        for i in range(len(names)):
            for j in range(len(names)):
                if i < j:
                    # matched halves on both sides, so the two correlations carry
                    # the same sampling noise as the within-task ceiling
                    cross += [float(np.corrcoef(per_task[i][layer][h][UPPER],
                                                per_task[j][layer][h][UPPER])[0, 1])
                              for h in (0, 1)]
        w, c = float(np.mean(within)), float(np.mean(cross))
        report["layers"].append({"layer": layer, "within_mean": w, "within": within,
                                 "cross_mean": c, "cross": cross,
                                 "share": c / w if w > 0 else None})
        print("  %d      %.3f          %.3f          %.2f"
              % (layer, w, c, c / w if w > 0 else float("nan")))

    print("\nreading: cross/within near 1 means every task draws the same graph; "
          "near 0 means the graph is task-specific and therefore carries signal")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
