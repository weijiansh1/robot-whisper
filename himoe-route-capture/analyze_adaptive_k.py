"""Can the state token predict how much a query state will fan out?

The state token cannot rank candidates -- it is candidate-invariant by construction
(it attends only to the prefix and never sees the flow timestep), which the routing
cloud confirms at 99.4% identical routing across 64 independent noise draws.  What
it can still do is vary ACROSS query states, and shallow state routing is genuinely
sharp there (M_4 up to 3.5x uniform).  So the live question is budget allocation,
not selection:

    D_K(s) = median_{i<j} || A_i - A_j ||      how far apart this state's proposals land

Predicting D_K(s) is only half of a usable method -- the other half is a fixed
downstream selector, without which extra samples buy nothing, and the honest target
is then G_K(s) = V_selector,K(s) - V_selector,1(s).  G_K needs counterfactual rollouts
and is deliberately out of scope here; this measures the cheap half and, crucially,
whether the ROUTING adds anything over representations that are already available.

That last point is the whole reason for the baseline ladder.  State routing is a
deterministic function of the state hidden state, g(h) = softmax(Wh), so under an
ideal decoder I(D; g(h)) <= I(D; h): routing cannot contain information the hidden
state lacks.  If routing predicts D_K no better than the raw proprioceptive state
does, the honest name is "state-conditioned adaptive compute", not
"routing-aware adaptive compute".

Every predictor is scored by leave-one-state-out ridge, so nothing is fitted and
evaluated on the same state, and all of them get the same regularisation and the
same dimensionality budget.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

N_EXPERTS = 32


def leave_one_out_r2(features, target, ridge=1.0, n_components=8):
    """LOO ridge R^2 on a common PCA budget, so no predictor wins on width alone."""
    x = np.asarray(features, np.float64)
    x = x - x.mean(0)
    scale = x.std(0)
    x = x / np.maximum(scale, 1e-12)
    if x.shape[1] > n_components:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        x = x @ vt[:n_components].T
    x = np.column_stack([np.ones(len(x)), x])
    y = np.asarray(target, np.float64)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    predictions = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = np.linalg.solve(x[keep].T @ x[keep] + penalty, x[keep].T @ y[keep])
        predictions[i] = x[i] @ beta
    residual = np.sum((y - predictions) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-12)), predictions


def spearman(a, b):
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--out", default="analysis/adaptive-k/summary.json")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    files = sorted(run.glob("flow_traces_part*.npz"))
    parts = [np.load(f) for f in files]
    x_traj = np.concatenate([p["x_traj"] for p in parts])
    query_id = np.concatenate([p["query_id"] for p in parts])
    candidate_id = np.concatenate([p["candidate_id"] for p in parts])
    # the tracer keeps the model's batch axis; every capture here is batch 1
    if x_traj.ndim == 5 and x_traj.shape[2] == 1:
        x_traj = x_traj[:, :, 0]
    records = json.loads((run / "query_records.json").read_text()) \
        if (run / "query_records.json").exists() else []
    by_query = {int(r["query_id"]): r for r in records}

    group = zarr.open_group(str(run / "routes.zarr"), mode="r")
    probs = np.asarray(group["hb_router_probs"][:], dtype=np.float32)
    episode = np.asarray(group["episode_id"][:])
    print("%d rows over %d query states" % (len(query_id), len(np.unique(query_id))), flush=True)

    dispersion, medoid_gap, routing, proprio, queries = [], [], [], [], []
    for query in np.unique(query_id):
        rows = np.flatnonzero(query_id == query)
        if len(rows) < 8:
            continue
        rows = rows[np.argsort(candidate_id[rows])]
        final = x_traj[rows][:, -1]
        flat = final.reshape(len(rows), -1)
        distance = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
        upper = np.triu_indices(len(rows), 1)
        dispersion.append(float(np.median(distance[upper])))
        # the medoid is the cheapest action-only selector; its gap to the worst
        # candidate bounds what any selector could win at this state
        totals = distance.sum(1)
        medoid_gap.append(float(distance[totals.argmin()].mean()))

        # state-token routing, first denoise round only: it is bit-identical across
        # rounds because the state token never sees the timestep
        route_rows = np.flatnonzero(episode == query)
        if not len(route_rows):
            continue
        state_probs = probs[route_rows[0], :, 0, 0, :]        # [L, 32]
        ordered = np.sort(state_probs, axis=-1)[:, ::-1]
        entropy = -(np.clip(state_probs, 1e-12, None)
                    * np.log2(np.clip(state_probs, 1e-12, None))).sum(-1)
        routing.append(np.concatenate([ordered[:, :4].sum(-1), entropy]))
        proprio.append(np.asarray(by_query.get(int(query), {}).get("state", []), np.float64))
        queries.append(int(query))

    dispersion = np.asarray(dispersion[: len(routing)])
    medoid_gap = np.asarray(medoid_gap[: len(routing)])
    routing = np.stack(routing)
    have_proprio = all(len(p) for p in proprio)
    print("states = %d, routing features = %d, proprio available = %s"
          % (len(dispersion), routing.shape[1], have_proprio), flush=True)

    predictors = {"state_routing (M4 + entropy per layer)": routing}
    if have_proprio:
        predictors["proprio state (baseline)"] = np.stack(proprio)
        predictors["proprio + routing"] = np.column_stack([np.stack(proprio), routing])

    summary = {"n_states": len(dispersion),
               "dispersion_median": float(np.median(dispersion))}
    print("\n=== predicting D_K(s), leave-one-state-out ridge at a common PCA budget ===")
    print("  predictor                                 R^2      Spearman")
    for name, features in predictors.items():
        r2, predicted = leave_one_out_r2(features, dispersion, args.ridge, args.components)
        rho = spearman(predicted, dispersion)
        summary[name] = {"r2": r2, "spearman": rho}
        print("  %-40s  %+.3f    %+.3f" % (name, r2, rho), flush=True)

    routing_r2 = summary["state_routing (M4 + entropy per layer)"]["r2"]
    if have_proprio:
        base = summary["proprio state (baseline)"]["r2"]
        print("\n  routing over proprio baseline: %+.3f R^2" % (routing_r2 - base))
        print("  routing is a deterministic function of the state, so a gain here is")
        print("  a probe-convenience effect, not extra information")
    print("\nNOTE: D_K(s) is the cheap half. Without a fixed downstream selector, a")
    print("high-dispersion state does not imply that extra samples are worth taking;")
    print("that needs G_K(s) = V_sel,K - V_sel,1 and counterfactual rollouts.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
