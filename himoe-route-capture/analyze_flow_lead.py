"""Conditional Lead Test: what does MoE routing add once the noise and the
provisional action are already known?

The unconditional question -- "can routing predict the final basin?" -- is the wrong
one, because the initial noise alone already predicts a lot of the final geometry
and the provisional action predicts more.  The question that decides whether MoE
state is worth reading is the increment:

    M0(tau)   predict same-basin from d_eps and d_x(tau)
    M1(tau)   the same, plus the routing distance at step tau
    delta AUC(tau) = AUC(M1) - AUC(M0)

Two statistical points that the pilot got wrong and this does not:

* **Pairs are not independent.** With K candidates there are K(K-1)/2 pairs but only
  K draws; every candidate appears in K-1 of them.  All confidence intervals here
  are bootstrapped over QUERY STATES, which are independent, never over pairs.
* **Padding dimensions.** The model's action space is 24-dim (8 EEF + 16 joint) and
  only some are live for this embodiment.  Distances use a data-driven mask -- the
  dimensions that actually vary across candidates -- rather than all 24, so dead
  padding cannot dilute the geometry.

The state token is carried through as a zero control: its routing is identical for
every candidate at a query state, so its delta AUC must come out at 0.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

N_EXPERTS = 32


def live_dims(x, tol=1e-8):
    """Dimensions that vary across candidates; the rest are padding or frozen."""
    spread = x.reshape(x.shape[0], -1, x.shape[-1]).std(axis=(0, 1))
    return spread > tol


def pairwise_norm(values, mask):
    n = len(values)
    flat = values[:, :, mask].reshape(n, -1)
    out = np.zeros((n, n))
    for i in range(n):
        out[i] = np.linalg.norm(flat - flat[i], axis=-1)
    return out


def weighted_jaccard(idx, weight, i, j):
    shape = idx.shape[1:-1]
    dense_i = np.zeros((*shape, N_EXPERTS))
    dense_j = np.zeros((*shape, N_EXPERTS))
    np.put_along_axis(dense_i, idx[i].astype(np.int64), weight[i].astype(np.float64), axis=-1)
    np.put_along_axis(dense_j, idx[j].astype(np.int64), weight[j].astype(np.float64), axis=-1)
    lower = np.minimum(dense_i, dense_j).sum(-1)
    upper = np.maximum(dense_i, dense_j).sum(-1)
    return float(np.mean(1.0 - lower / np.maximum(upper, 1e-12)))


def pairwise_route(idx, weight):
    n = idx.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = weighted_jaccard(idx, weight, i, j)
    return out


def auc(score, label):
    """Lower score should mean same basin (label 1)."""
    positive, negative = score[label == 1], score[label == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = (positive[:, None] < negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def logistic_auc(features, label, ridge=1e-3, iterations=200):
    """AUC of a logistic fit, so several distances can be combined honestly.

    In-sample: this compares nested models on the same pairs, and the question is
    whether the extra feature can help AT ALL.  An in-sample delta of zero is
    therefore already conclusive in the negative direction, which is the direction
    the evidence has been pointing.
    """
    x = np.column_stack([np.ones(len(label))] + [
        (f - f.mean()) / max(f.std(), 1e-12) for f in features])
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        gradient = x.T @ (label - p) - ridge * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(x.shape[1])
        w += np.linalg.solve(hessian + 1e-9 * np.eye(x.shape[1]), gradient)
    # higher logit => same basin, so negate to keep the "lower is same" convention
    return auc(-(x @ w), label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/flow-lead/summary.json")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    # the server writes parts so a kill costs at most one part; concatenate in order
    files = sorted(run.glob("flow_traces_part*.npz")) or [run / "flow_traces.npz"]
    parts = [np.load(f) for f in files]
    x_traj = np.concatenate([p["x_traj"] for p in parts])
    query_id = np.concatenate([p["query_id"] for p in parts])
    candidate_id = np.concatenate([p["candidate_id"] for p in parts])
    print("loaded %d part file(s)" % len(files), flush=True)
    # the tracer keeps the model's batch axis; every capture here is batch 1
    if x_traj.ndim == 5 and x_traj.shape[2] == 1:
        x_traj = x_traj[:, :, 0]
    group = zarr.open_group(str(run / "routes.zarr"), mode="r")
    idx_all = np.asarray(group["hb_expert_ids"][:])
    weight_all = np.asarray(group["hb_selected_prob"][:], dtype=np.float32)
    episode = np.asarray(group["episode_id"][:])
    print("%d rows, %d query states, x_traj %s, routing %s"
          % (len(query_id), len(np.unique(query_id)), x_traj.shape, idx_all.shape), flush=True)

    n_tau = x_traj.shape[1] - 1
    rng_local = np.random.default_rng(args.seed + 1)
    per_state = []
    for query in np.unique(query_id):
        rows = np.flatnonzero(query_id == query)
        if len(rows) < 6:
            continue
        order = np.argsort(candidate_id[rows])
        rows = rows[order]
        traj = x_traj[rows]                                   # [K, T+1, n_action, 24]
        mask = live_dims(traj[:, -1])
        final = pairwise_norm(traj[:, -1], mask)
        upper = np.triu_indices(len(rows), 1)
        label = (final < np.median(final[upper])).astype(int)[upper]
        noise = pairwise_norm(traj[:, 0], mask)[upper]

        route_rows = np.flatnonzero(np.isin(episode, [query]))
        route_rows = route_rows[np.argsort(route_rows)][: len(rows)]
        entry = {"query": int(query), "n_candidates": int(len(rows)), "tau": []}
        for tau in range(n_tau):
            action = pairwise_norm(traj[:, tau], mask)[upper]
            route = pairwise_route(idx_all[route_rows][:, :, tau, 1:],
                                   weight_all[route_rows][:, :, tau, 1:])[upper]
            state_route = pairwise_route(idx_all[route_rows][:, :, tau, :1],
                                         weight_all[route_rows][:, :, tau, :1])[upper]
            base = logistic_auc([noise, action], label)
            entry["tau"].append({
                "tau": tau,
                "auc_noise": auc(noise, label),
                "auc_action": auc(action, label),
                "auc_route": auc(route, label),
                "auc_M0": base,
                "auc_M1": logistic_auc([noise, action, route], label),
                "auc_M1_state_control": logistic_auc([noise, action, state_route], label),
                # in-sample optimism control: the SAME routing distances, permuted
                # across pairs.  A third feature buys some in-sample AUC by itself,
                # and this measures exactly how much, so dAUC can be read net of it.
                "auc_M1_permuted": logistic_auc(
                    [noise, action, route[rng_local.permutation(len(route))]], label),
            })
        per_state.append(entry)
        if len(per_state) % 10 == 0:
            print("  %d states" % len(per_state), flush=True)

    rng = np.random.default_rng(args.seed)
    print("\n tau  AUC(eps) AUC(act) AUC(rt) |   M0     M1    dAUC [95% CI]        perm      net")
    rows_out = []
    for tau in range(n_tau):
        values = {key: np.array([s["tau"][tau][key] for s in per_state])
                  for key in ("auc_noise", "auc_action", "auc_route", "auc_M0",
                              "auc_M1", "auc_M1_state_control", "auc_M1_permuted")}
        delta = values["auc_M1"] - values["auc_M0"]
        control = values["auc_M1_state_control"] - values["auc_M0"]
        optimism = values["auc_M1_permuted"] - values["auc_M0"]
        # bootstrap over STATES: they are the independent unit, pairs are not
        draws = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                          for _ in range(args.bootstrap)])
        low, high = np.percentile(draws, [2.5, 97.5])
        rows_out.append({"tau": tau, **{k: float(v.mean()) for k, v in values.items()},
                         "delta_auc": float(delta.mean()),
                         "delta_ci": [float(low), float(high)],
                         "state_control_delta": float(control.mean()),
                         "permuted_delta": float(optimism.mean()),
                         "delta_net_of_optimism": float((delta - optimism).mean())})
        print("  %2d   %.3f   %.3f   %.3f | %.3f %.3f  %+.4f [%+.4f,%+.4f]  perm %+.4f  net %+.4f"
              % (tau, values["auc_noise"].mean(), values["auc_action"].mean(),
                 values["auc_route"].mean(), values["auc_M0"].mean(),
                 values["auc_M1"].mean(), delta.mean(), low, high,
                 optimism.mean(), (delta - optimism).mean()))

    significant = [r["tau"] for r in rows_out
                   if r["delta_ci"][0] > 0 and r["delta_auc"] > 2 * r["permuted_delta"]]
    print("\nflow steps where routing adds AUC beyond 2x the in-sample optimism: %s"
          % (significant if significant else "none"))
    print("states = %d (the bootstrap unit); pairs per state = %d"
          % (len(per_state), per_state[0]["n_candidates"] * (per_state[0]["n_candidates"] - 1) // 2))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_tau": rows_out, "n_states": len(per_state)}, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
