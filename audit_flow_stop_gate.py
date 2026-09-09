#!/usr/bin/env python3
"""Step-0 gate for the denoise-early-stop idea (offline, free).

Data: runs/flow-lead-cpu-t0s24 — 44 queries x 16 candidates, full x^(0..10)
flow trajectories + per-round HB routing (704 rows, CPU bit-deterministic).

Q1 (gate): if we stop after r of 10 rounds and output the one-jump
extrapolation  x_hat = x_m - t_m * v_m  (v_m is the round-m forward output,
t_m = 1 - 0.1 m, r = m+1), how far is x_hat from the true final x^(10) on the
live 7 action dims?  Anchors: the 4-6% "harmless band" (random expert swap
moves the action that much with zero rate effect) and the candidate-to-
candidate spread (a different noise seed moves the action that much, and
re-rolls outcomes without moving the rate).

Q2 (the routing-readout idea): predict x^(10) from the first m rounds'
action-token routing vs from the current latent x_m, leave-one-episode-out.
The VQ account predicts routing is dominated by the latent it projects.

Writes audit_flow_stop_gate.json.
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "himoe-route-capture/runs/flow-lead-cpu-t0s24"
LIVE = 7
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def rel_err(a, b):
    """Per-row relative L2 between (rows, 10, LIVE) tensors."""
    d = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    n = np.linalg.norm(b.reshape(len(b), -1), axis=1)
    return d / np.maximum(n, 1e-9)


def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    P = alpha * np.eye(Xs.shape[1])
    P[0, 0] = 0.0
    W = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ Y)
    return mu, sd, W


def ridge_pred(model, X):
    mu, sd, W = model
    return np.c_[np.ones(len(X)), (X - mu) / sd] @ W


def loeo(X, Y, ep):
    """Leave-one-episode-out prediction with inner-episode alpha choice."""
    pred = np.zeros_like(Y)
    for h in np.unique(ep):
        tr = ep != h
        eps_tr = np.unique(ep[tr])
        best, best_mse = None, np.inf
        for a in ALPHAS:
            mse = 0.0
            for h2 in eps_tr:
                itr = tr & (ep != h2)
                iva = ep == h2
                m = ridge_fit(X[itr], Y[itr], a)
                mse += float(((ridge_pred(m, X[iva]) - Y[iva]) ** 2).sum())
            if mse < best_mse:
                best, best_mse = a, mse
        m = ridge_fit(X[tr], Y[tr], best)
        pred[ep == h] = ridge_pred(m, X[ep == h])
    return pred


def main() -> int:
    parts = [np.load(p) for p in sorted(glob.glob(str(RUN / "flow_traces_part*.npz")))]
    X = np.concatenate([p["x_traj"] for p in parts]).squeeze(2)   # (704, 11, 10, 24)
    qid = np.concatenate([p["query_id"] for p in parts])
    cid = np.concatenate([p["candidate_id"] for p in parts])
    assert X.shape == (704, 11, 10, 24)
    assert np.all(np.diff(qid) >= 0)
    for q in np.unique(qid):
        assert sorted(cid[qid == q].tolist()) == list(range(16))

    g = zarr.open_group(str(RUN / "routes.zarr"), mode="r")
    # in this run the recorder's "episode_id" is the query id (0..43) and
    # "control_step" is a global row counter -- row order matches the parts
    ep_z = np.asarray(g["episode_id"][:])
    cs_z = np.asarray(g["control_step"][:])
    blk = np.repeat(np.arange(44), 16)
    assert np.all(qid == blk) and np.all(ep_z == qid)
    assert np.all(cs_z == np.arange(704))
    ep_query = qid // 11                                        # rollout id, 4 folds
    probs = np.asarray(g["hb_router_probs"][:], np.float32)     # (704,8,10,11,32)
    probs /= np.maximum(probs.sum(-1, keepdims=True), 1e-9)

    x10 = X[:, 10, :, :LIVE]
    out = {"run": "flow-lead-cpu-t0s24", "rows": 704, "queries": 44,
           "episodes": int(len(np.unique(ep_query)))}

    # ---- anchors ------------------------------------------------------------
    spread = []
    for q in range(44):
        rows = np.flatnonzero(qid == q)
        F = x10[rows].reshape(16, -1)
        D = np.linalg.norm(F[:, None] - F[None], axis=2)
        N = np.linalg.norm(F, axis=1)
        iu = np.triu_indices(16, 1)
        spread.append((D / np.maximum(0.5 * (N[:, None] + N[None]), 1e-9))[iu])
    spread = np.concatenate(spread)
    out["candidate_spread_rel"] = {"median": float(np.median(spread)),
                                   "p10": float(np.quantile(spread, .1)),
                                   "p90": float(np.quantile(spread, .9))}

    # ---- Q1: truncation ladder ---------------------------------------------
    print("flow-lead t0s24: 44 queries x 16 candidates, 4 episodes")
    print("candidate-to-candidate spread (rel L2, live7): median %.3f [p10 %.3f, p90 %.3f]"
          % (np.median(spread), np.quantile(spread, .1), np.quantile(spread, .9)))
    print("harmless band from expert-swap: 0.04-0.06 rel L2\n")
    print("  r=rounds executed   raw x_m err      one-jump x_hat err")
    q1 = {}
    for m in range(10):
        xm = X[:, m, :, :LIVE]
        v = (X[:, m] - X[:, m + 1]) / 0.1
        xhat = (X[:, m] - (1.0 - 0.1 * m) * v)[:, :, :LIVE]
        e_raw = rel_err(xm, x10)
        e_hat = rel_err(xhat, x10)
        q1[m + 1] = {"raw_median": float(np.median(e_raw)),
                     "hat_median": float(np.median(e_hat)),
                     "hat_p90": float(np.quantile(e_hat, .9))}
        print("  r=%-2d                %.3f            %.3f  (p90 %.3f)"
              % (m + 1, np.median(e_raw), np.median(e_hat), np.quantile(e_hat, .9)))
    out["truncation"] = q1

    # ---- Q2: routing readout vs latent readout ------------------------------
    print("\nreadout of x^(10) (median rel err, leave-one-episode-out):")
    print("  m   x_hat(free)   ridge(x_m)   ridge(route<=m)   ridge(x_m+route)")
    q2 = {}
    Y = x10.reshape(704, -1)
    for m in (1, 3, 5, 7):
        xm_f = X[:, m, :, :LIVE].reshape(704, -1)
        rf = probs[:, :, :m, 1:, :].mean(3).reshape(704, -1)    # action tokens
        xhat = (X[:, m] - (1.0 - 0.1 * m) * ((X[:, m] - X[:, m + 1]) / 0.1))[:, :, :LIVE]
        preds = {
            "xhat_free": xhat.reshape(704, -1),
            "ridge_latent": loeo(xm_f, Y, ep_query),
            "ridge_route": loeo(rf, Y, ep_query),
            "ridge_both": loeo(np.c_[xm_f, rf], Y, ep_query),
        }
        row = {}
        for k, P in preds.items():
            e = rel_err(P.reshape(704, 10, LIVE), x10)
            r2 = 1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
            row[k] = {"median_rel_err": float(np.median(e)), "r2": float(r2)}
        q2[m] = row
        print("  %d   %.3f         %.3f        %.3f             %.3f"
              % (m, row["xhat_free"]["median_rel_err"],
                 row["ridge_latent"]["median_rel_err"],
                 row["ridge_route"]["median_rel_err"],
                 row["ridge_both"]["median_rel_err"]))
    out["readout"] = q2

    (HERE / "audit_flow_stop_gate.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_flow_stop_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
