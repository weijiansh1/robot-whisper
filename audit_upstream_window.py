#!/usr/bin/env python3
"""Does the MoE's upstream position buy anything? Three offline tests.

E1  Where does the "preview of this round's correction" live -- front block
    (layers 2-5, cheap to read mid-forward) or back block (12-15, too late)?
E2  Fair trigger fight for intra-round early exit: momentum baseline
    {x_m stats, v_{m-1}, round} vs +front-route vs +back-route.
E3  Oracle ceiling of round-skipping: skip every round whose true update is
    below theta (first-order sim: later v assumed unchanged -- valid only for
    small skipped updates), rounds saved vs action error.

flow-lead-cpu-t0s24, 704 trajectories.  Writes audit_upstream_window.json.
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import zarr
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "himoe-route-capture/runs/flow-lead-cpu-t0s24"
LIVE = 7
BAND = 0.049


def loeo(X, y, ep, alpha=10.0):
    pred = np.zeros_like(y)
    for h in np.unique(ep):
        tr = ep != h
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        P = alpha * np.eye(Xs.shape[1]); P[0, 0] = 0
        w = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ y[tr])
        te = ep == h
        pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ w
    return pred


def main() -> int:
    parts = [np.load(p) for p in sorted(glob.glob(str(RUN / "flow_traces_part*.npz")))]
    X = np.concatenate([p["x_traj"] for p in parts]).squeeze(2)
    qid = np.concatenate([p["query_id"] for p in parts])
    ep = qid // 11
    n = len(X)
    x10n = np.maximum(np.linalg.norm(X[:, 10, :, :LIVE].reshape(n, -1), axis=1), 1e-9)
    V = (X[:, :-1] - X[:, 1:]) / 0.1
    U = np.stack([np.linalg.norm((0.1 * V[:, m, :, :LIVE]).reshape(n, -1), axis=1)
                  for m in range(10)], 1) / x10n[:, None]      # this round's step

    g = zarr.open_group(str(RUN / "routes.zarr"), mode="r")
    P = np.asarray(g["hb_router_probs"][:], np.float32)
    P /= np.maximum(P.sum(-1, keepdims=True), 1e-9)
    A = P[:, :, :, 1:, :]
    R = np.sqrt(A)

    def block_feats(lo, hi, m):
        """Per-layer stats of layers [lo:hi] at round m (+change vs m-1)."""
        a = A[:, lo:hi, m]                                     # (n,L,10,32)
        srt = np.sort(a, -1)
        ent = -(a * np.log(np.maximum(a, 1e-12))).sum(-1).mean(2) / np.log(32)
        mar = (srt[..., -1] - srt[..., -2]).mean(2)
        ms4 = srt[..., -4:].sum(-1).mean(2)
        chg = np.sqrt((0.5 * ((R[:, lo:hi, m] - R[:, lo:hi, m - 1]) ** 2)
                       .sum(-1)).mean(2))
        return np.c_[ent, mar, ms4, chg]                       # (n, 4L)

    # assemble round-stacked design (rounds 1..9)
    rows_y, rows_ep, rows_base, rows_front, rows_back, rows_id = [], [], [], [], [], []
    for m in range(1, 10):
        xm = X[:, m, :, :LIVE].reshape(n, -1)
        xstat = np.c_[np.linalg.norm(xm, axis=1) / x10n,
                      np.abs(xm).max(1) / x10n]
        vprev = U[:, m - 1]
        base = np.c_[xstat, vprev]
        rows_base.append(base)
        rows_front.append(block_feats(0, 4, m))
        rows_back.append(block_feats(4, 8, m))
        rows_y.append(U[:, m])
        rows_ep.append(ep)
        rows_id.append(np.full(n, m))
    Y = np.concatenate(rows_y)
    EP = np.concatenate(rows_ep)
    RID = np.eye(10)[np.concatenate(rows_id)][:, 1:]
    B = np.c_[np.vstack(rows_base), RID]
    F = np.vstack(rows_front)
    K = np.vstack(rows_back)

    out = {}
    arms = {
        "momentum_baseline": B,
        "front_route_only": np.c_[F, RID],
        "back_route_only": np.c_[K, RID],
        "base_plus_front": np.c_[B, F],
        "base_plus_back": np.c_[B, K],
        "base_plus_both": np.c_[B, F, K],
    }
    print("E1/E2: predict this round's update size (LOEO Spearman)")
    for k, Xd in arms.items():
        r = float(spearmanr(loeo(Xd, Y, EP), Y).statistic)
        out[k] = r
        print("  %-22s %.3f" % (k, r))

    # E3: oracle round-skip ceiling (first-order)
    print("\nE3: oracle skip-rounds ceiling (skip round m in 2..9 if U<theta)")
    print("  theta    mean rounds saved   median err   p90 err")
    e3 = []
    for theta in (0.005, 0.01, 0.02, 0.03, 0.05):
        skip = U[:, 2:10] < theta                              # (n,8) never skip 0-1
        err_vec = np.zeros((n, 10, LIVE))
        for m in range(2, 10):
            err_vec += (0.1 * V[:, m, :, :LIVE]) * skip[:, m - 2][:, None, None]
        err = np.linalg.norm(err_vec.reshape(n, -1), axis=1) / x10n
        e3.append({"theta": theta, "saved": float(skip.mean() * 8) / 10,
                   "med": float(np.median(err)), "p90": float(np.quantile(err, .9))})
        print("  %.3f    %.2f of 10          %.3f        %.3f"
              % (theta, skip.mean() * 8, np.median(err), np.quantile(err, .9)))
    out["oracle_skip"] = e3

    (HERE / "audit_upstream_window.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_upstream_window.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
