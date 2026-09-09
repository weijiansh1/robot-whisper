"""Is a flipping donor inside the routing the policy can reach at that same state?

Reads the clouds from rollout_routing_cloud.py and the flip list from the step-11
transplant, and for every (recipient, donor) pair that flipped asks whether the
donor is distinguishable from a draw of

    R_pi(s_k) = { rho(s_k, eps_k) : eps_k }.

The test is nearest-neighbour depth with a leave-one-out calibration, which needs no
assumption about the shape of the cloud:

    d_min(r)     = min_j d(r, c_j)
    null         = { min_{j' != j} d(c_j, c_j') : j }
    p            = (1 + #{ j : null_j >= d_min(r) }) / (N + 1)

A large p means the donor sits no further out than cloud members sit from each other,
so the flip it caused is also reachable by resampling flow noise alone and the
counterexample lifts from Y_phys to Y_pi.  A small p means the transplant is off the
policy's own manifold at that state and the gap does not close this way -- which is a
result, not a failure: it would say the physical lower bound cannot be upgraded by
transplanting between states at all.

Two distances are reported because they can disagree.  Jaccard sees only which four
experts were picked; total variation also sees the weights, which is where the router
is nearly uniform on the action tokens.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

N_EXPERTS = 32
TOP_K = 4


def _onehot(idx: np.ndarray) -> np.ndarray:
    """[..., K] expert ids -> [..., 32] bool membership."""
    flat = idx.reshape(-1, idx.shape[-1]).astype(np.int64)
    out = np.zeros((flat.shape[0], N_EXPERTS), dtype=bool)
    np.put_along_axis(out, flat, True, axis=1)
    return out.reshape(*idx.shape[:-1], N_EXPERTS)


def _dense(idx: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """[..., K] ids + weights -> [..., 32] distribution."""
    flat_idx = idx.reshape(-1, idx.shape[-1]).astype(np.int64)
    flat_w = weight.reshape(-1, weight.shape[-1]).astype(np.float64)
    out = np.zeros((flat_idx.shape[0], N_EXPERTS), dtype=np.float64)
    np.add.at(out, (np.arange(flat_idx.shape[0])[:, None], flat_idx), flat_w)
    return out.reshape(*idx.shape[:-1], N_EXPERTS)


def jaccard_distance(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
    """Mean over routing sites of 1 - |A n B| / |A u B|; 0.933 if the picks are random."""
    a, b = _onehot(a_idx), _onehot(b_idx)
    inter = np.logical_and(a, b).sum(-1).astype(np.float64)
    union = np.logical_or(a, b).sum(-1).astype(np.float64)
    return float(np.mean(1.0 - inter / union))


def tv_distance(a_idx, a_w, b_idx, b_w) -> float:
    """Mean over routing sites of the total variation between the two top-4 mixtures."""
    a, b = _dense(a_idx, a_w), _dense(b_idx, b_w)
    return float(np.mean(0.5 * np.abs(a - b).sum(-1)))


def _pairwise_min(cloud_idx, cloud_w, metric):
    """Leave-one-out nearest-neighbour distance for every cloud member."""
    n = len(cloud_idx)
    d = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            v = metric(cloud_idx[i], cloud_w[i], cloud_idx[j], cloud_w[j])
            d[i, j] = d[j, i] = v
    return d.min(axis=1), d


def depth_test(query_idx, query_w, cloud_idx, cloud_w, metric):
    d_query = np.array([
        metric(query_idx, query_w, cloud_idx[j], cloud_w[j]) for j in range(len(cloud_idx))
    ])
    d_min = float(d_query.min())
    loo, full = _pairwise_min(cloud_idx, cloud_w, metric)
    p = (1 + int(np.sum(loo >= d_min))) / (len(loo) + 1)
    off = full[np.triu_indices(len(loo), 1)]
    return {
        "d_min": d_min,
        "d_mean_to_cloud": float(d_query.mean()),
        "cloud_loo_median": float(np.median(loo)),
        "cloud_loo_max": float(loo.max()),
        "cloud_pairwise_median": float(np.median(off)),
        "cloud_pairwise_max": float(off.max()),
        "p_inside": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-dir", required=True)
    ap.add_argument("--patch-summaries", required=True,
                    help="summaries.json from rollout_patch_probe.py")
    ap.add_argument("--metric", choices=["jaccard", "tv"], default="jaccard")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cloud_dir = pathlib.Path(args.cloud_dir)
    clouds = {}
    for path in sorted(cloud_dir.glob("cloud_*.npz")):
        data = np.load(path)
        clouds[int(data["flow_noise_seed"])] = {
            "natural_idx": data["natural_idx"],
            "natural_weight": data["natural_weight"],
            "cloud_idx": data["cloud_idx"],
            "cloud_weight": data["cloud_weight"],
        }
    if not clouds:
        raise RuntimeError("no cloud_*.npz in %s" % cloud_dir)

    metric = (
        (lambda ai, aw, bi, bw: jaccard_distance(ai, bi))
        if args.metric == "jaccard"
        else tv_distance
    )

    rows = json.loads(pathlib.Path(args.patch_summaries).read_text())
    flips = [
        r for r in rows
        if r.get("outcome_flipped") and r.get("donor") is not None
        and r["flow_noise_seed"] in clouds and r["donor"] in clouds
    ]

    # The episode's own draw is a cloud member by construction; if it does not test
    # as one, the two runs did not reach the same state and nothing below holds.
    sanity = []
    for seed, c in clouds.items():
        result = depth_test(
            c["natural_idx"], c["natural_weight"], c["cloud_idx"], c["cloud_weight"], metric
        )
        result["flow_noise_seed"] = seed
        sanity.append(result)
    print("natural draw vs its own cloud (should look like a member):")
    print("  p_inside  min %.3f  median %.3f"
          % (min(s["p_inside"] for s in sanity), float(np.median([s["p_inside"] for s in sanity]))))
    print("  cloud pairwise %s distance, median over seeds: %.4f"
          % (args.metric, float(np.median([s["cloud_pairwise_median"] for s in sanity]))))

    verdicts = []
    print("\nflipping donors, %s distance:" % args.metric)
    for row in flips:
        seed, donor = row["flow_noise_seed"], row["donor"]
        c = clouds[seed]
        d = clouds[donor]
        result = depth_test(
            d["natural_idx"], d["natural_weight"], c["cloud_idx"], c["cloud_weight"], metric
        )
        result.update(
            flow_noise_seed=seed, donor=donor, arm=row["arm"],
            branch_action_max_abs_delta=row.get("branch_action_max_abs_delta"),
        )
        verdicts.append(result)
        print("  seed %d <- donor %-5d  %-13s d_min %.4f  cloud LOO median %.4f  p=%.3f  %s"
              % (seed, donor, row["arm"], result["d_min"], result["cloud_loo_median"],
                 result["p_inside"],
                 "INSIDE" if result["p_inside"] > 0.05 else "outside"))

    inside = [v for v in verdicts if v["p_inside"] > 0.05]
    print("\n%d/%d flipping donors are indistinguishable from the policy's own reach"
          % (len(inside), len(verdicts)))
    if inside:
        print("-> for those states |Y_pi(s_k)| = 2: the counterexample lifts from "
              "physical to policy-reachable")
    else:
        print("-> every flipping donor is off-manifold at its recipient's state; the "
              "physical bound does not lift by transplanting between states")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "metric": args.metric,
        "n_clouds": len(clouds),
        "natural_sanity": sanity,
        "flip_verdicts": verdicts,
        "n_inside": len(inside),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
