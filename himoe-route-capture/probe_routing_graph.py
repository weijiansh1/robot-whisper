"""Does HB routing carry structure beyond its per-layer marginals?

Every routing statistic this project has computed so far -- JS between task
profiles, Jaccard between routing states, PCA of the flattened tensor -- is a
function of *marginal* expert frequencies.  On the action tokens those marginals
are nearly uniform (normalised router entropy 0.998), which is exactly the regime
where a marginal summary is least informative and a relational one might not be.

Any graph model of the routing state is a bet that the relational structure is
real.  This script tests the bet before paying for it, on two kinds of edge:

**Co-selection edges** (which experts share a top-4).  Null: the bipartite
site-expert incidence matrix randomised by curveball trades, which preserves
*both* margins exactly -- every site keeps exactly 4 experts and every expert
keeps its exact frequency.  A weaker null (independent column permutation) would
break the degree-4 constraint and manufacture structure from it.

**Adjacency edges** along the three axes that are not the expert axis: depth
(layer to layer), denoise round, and suffix position.  Measured as mutual
information between top-1 choices, against a site-permutation null which destroys
the pairing while keeping both marginals.  An axis whose MI does not clear its
null is an axis a graph should not put edges along.

Run on a handful of tasks; the answer is per-layer, so pooling tasks would only
blur it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

N_EXPERTS = 32
TOP_K = 4


def incidence(ids: np.ndarray, n_experts: int = N_EXPERTS) -> np.ndarray:
    """[n_sites, K] expert ids -> boolean [n_sites, n_experts]."""
    out = np.zeros((ids.shape[0], n_experts), bool)
    out[np.repeat(np.arange(ids.shape[0]), ids.shape[1]), ids.ravel()] = True
    return out


def curveball(matrix: np.ndarray, trades: int, rng) -> np.ndarray:
    """Randomise a binary matrix keeping both row and column sums exact.

    Curveball (Strona et al. 2014): repeatedly pick two rows and randomly
    redistribute the experts that exactly one of them holds.  The shared experts
    and the row totals are untouched, so both margins are invariant by
    construction rather than in expectation.
    """
    work = matrix.copy()
    n_rows = work.shape[0]
    left = rng.integers(0, n_rows, trades)
    right = rng.integers(0, n_rows, trades)
    for a, b in zip(left, right):
        if a == b:
            continue
        row_a, row_b = work[a], work[b]
        only = row_a ^ row_b
        if not only.any():
            continue
        pool = np.flatnonzero(only)
        take = int(row_a[pool].sum())
        chosen = rng.choice(pool, take, replace=False)
        row_a[pool] = False
        row_b[pool] = True
        row_a[chosen] = True
        row_b[chosen] = False
    return work


def co_selection(matrix: np.ndarray) -> np.ndarray:
    counts = matrix.astype(np.int64)
    return counts.T @ counts


def co_selection_test(ids, rng, n_null=20, max_sites=20000, trades_per_site=5):
    """Observed vs margin-preserving null co-selection, as a z-map over expert pairs."""
    if ids.shape[0] > max_sites:
        ids = ids[rng.choice(ids.shape[0], max_sites, replace=False)]
    matrix = incidence(ids)
    observed = co_selection(matrix)
    trades = trades_per_site * matrix.shape[0]
    nulls = np.stack([co_selection(curveball(matrix, trades, rng)) for _ in range(n_null)])
    mean, std = nulls.mean(0), nulls.std(0, ddof=1)
    upper = np.triu_indices(N_EXPERTS, 1)
    z = np.zeros_like(observed, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - mean) / np.where(std > 0, std, np.nan)
    z_pairs = z[upper]
    finite = z_pairs[np.isfinite(z_pairs)]
    # the null spread itself: how far a null draw sits from the null mean, on the
    # same scale, so "max |z| = 12" can be read against what noise alone reaches
    null_z = ((nulls[0] - mean) / np.where(std > 0, std, np.nan))[upper]
    null_z = null_z[np.isfinite(null_z)]
    lift = observed[upper] / np.maximum(mean[upper], 1e-9)
    return {
        "n_sites": int(matrix.shape[0]),
        # kept so a caller can ask whether the graph is the same graph across
        # tasks: a fixed expert-affinity backbone carries no per-state signal,
        # a task-varying one does
        "_lift_matrix": (observed / np.maximum(mean, 1e-9)),
        "_z_matrix": z,
        "max_abs_z": float(np.max(np.abs(finite))) if finite.size else None,
        "frac_pairs_z_gt_3": float(np.mean(np.abs(finite) > 3)) if finite.size else None,
        "null_max_abs_z": float(np.max(np.abs(null_z))) if null_z.size else None,
        "max_lift": float(np.max(lift)),
        "min_lift": float(np.min(lift)),
        "mean_abs_z": float(np.mean(np.abs(finite))) if finite.size else None,
    }


def mutual_information(a: np.ndarray, b: np.ndarray, n=N_EXPERTS) -> float:
    joint = np.zeros((n, n), np.int64)
    np.add.at(joint, (a, b), 1)
    total = joint.sum()
    if total == 0:
        return 0.0
    p = joint / total
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / (px @ py)[mask])))


def mi_test(a, b, rng, n_null=20):
    """MI of a paired top-1 sequence against a null that breaks only the pairing."""
    observed = mutual_information(a, b)
    nulls = np.array([mutual_information(a, b[rng.permutation(b.shape[0])])
                      for _ in range(n_null)])
    return {
        "mi_bits": observed,
        "null_mi_bits": float(nulls.mean()),
        "excess_bits": observed - float(nulls.mean()),
        # the null MI is the finite-sample bias floor (~(n-1)^2 / 2N ln2), so the
        # ratio says how much of the observed MI is structure rather than counting
        "ratio_over_null": observed / max(float(nulls.mean()), 1e-12),
        "z": (observed - float(nulls.mean())) / max(float(nulls.std(ddof=1)), 1e-12),
    }


def load_task(task_path: pathlib.Path):
    group = zarr.open_group(str(task_path / "server" / "routes.zarr"), mode="r")
    return np.asarray(group["hb_expert_ids"][:])  # [N, L, D, S, K]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", required=True,
                    help="path to a corpus task directory; repeatable")
    ap.add_argument("--token", choices=("action", "state"), default="action")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--max-sites", type=int, default=20000)
    ap.add_argument("--decorrelate", action="store_true",
                    help="one site per control step instead of all denoise x token "
                         "sites.  The curveball null treats sites as exchangeable, "
                         "but sites inside one control step share 2.2 bits along the "
                         "denoise axis, so the pooled null SD is too small and z is "
                         "inflated.  This keeps the effect size and fixes the scale.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/routing-graph")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results, matrices = {}, {}
    for task in args.task:
        path = pathlib.Path(task)
        ids = load_task(path)
        n_steps, n_layers, n_denoise, n_suffix, _ = ids.shape
        token = slice(1, n_suffix) if args.token == "action" else slice(0, 1)
        print("\n=== %s  [%d steps, %d layers, %d denoise, %s tokens] ==="
              % (path.name[:60], n_steps, n_layers, n_denoise, args.token), flush=True)

        entry = {"co_selection": {}, "adjacency": {}}
        lifts, zs = [], []

        print("-- co-selection vs margin-preserving null")
        for layer in range(n_layers):
            if args.decorrelate:
                # cycle the round and the token so no single position is privileged
                offset = np.arange(n_steps)
                rounds = offset % n_denoise
                start = 1 if args.token == "action" else 0
                span = (n_suffix - start)
                tokens = start + (offset // n_denoise) % span
                sites = ids[offset, layer, rounds, tokens, :]
            else:
                sites = ids[:, layer, :, token, :].reshape(-1, TOP_K)
            stat = co_selection_test(sites, rng, args.n_null, args.max_sites)
            lifts.append(stat.pop("_lift_matrix"))
            zs.append(stat.pop("_z_matrix"))
            entry["co_selection"]["layer%d" % layer] = stat
            print("   layer %d  max|z| %6.1f (null %4.1f)  pairs |z|>3 %5.1f%%  lift %.2f-%.2f"
                  % (layer, stat["max_abs_z"], stat["null_max_abs_z"],
                     100 * stat["frac_pairs_z_gt_3"], stat["min_lift"], stat["max_lift"]),
                  flush=True)

        print("-- adjacency MI (top-1), observed vs pairing-broken null")
        top1 = ids[..., 0]  # [N, L, D, S]

        def flat(x):
            return np.asarray(x).ravel().astype(np.int64)

        pairs = {
            "depth": [(flat(top1[:, l, :, token]), flat(top1[:, l + 1, :, token]))
                      for l in range(n_layers - 1)],
            "denoise": [(flat(top1[:, :, d, token]), flat(top1[:, :, d + 1, token]))
                        for d in range(n_denoise - 1)],
        }
        if args.token == "action" and n_suffix > 2:
            pairs["suffix"] = [(flat(top1[:, :, :, s]), flat(top1[:, :, :, s + 1]))
                               for s in range(1, n_suffix - 1)]
        # a same-axis control: two sites that share nothing but the control step.
        # if this scores like the adjacency axes, the MI is measuring the control
        # step rather than the axis, and no edge along that axis is warranted.
        control = [(flat(top1[:, 0, 0, token]), flat(top1[:, n_layers - 1, n_denoise - 1, token]))]
        pairs["control_far"] = control

        for axis, items in pairs.items():
            stats = [mi_test(a, b, rng, args.n_null) for a, b in items]
            entry["adjacency"][axis] = stats
            excess = np.array([s["excess_bits"] for s in stats])
            print("   %-12s excess MI %.4f bits (min %.4f max %.4f over %d pairs)  "
                  "null floor %.4f" % (axis, excess.mean(), excess.min(), excess.max(),
                                       len(stats), stats[0]["null_mi_bits"]), flush=True)

        results[path.name] = entry
        matrices[path.name] = (np.stack(lifts), np.stack(zs))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("%s_tokens.json" % args.token)
    target.write_text(json.dumps(results, indent=1))
    names = sorted(matrices)
    np.savez_compressed(
        out / ("%s_tokens_graphs.npz" % args.token),
        tasks=np.array(names),
        lift=np.stack([matrices[n][0] for n in names]),
        z=np.stack([matrices[n][1] for n in names]),
    )
    print("\nwrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
