"""Does the routing rank same-state candidates by what they actually do?

Input is fork_pilot.py's records plus the routing the server recorded for the same
queries.  Every comparison is made *within* a snapshot: the candidates there share an
observation, a proprio state and a world state exactly, so a within-snapshot
correlation cannot be a scene readout.  Targets and features are therefore centred per
snapshot before anything is fitted.

Three predictors, because "routing predicts the outcome" is only interesting relative
to the action it was produced alongside:

  route     the router distributions for that query
  action    the candidate action chunk itself
  both      concatenated

If route alone ranks candidates but adds nothing over action, it is a re-encoding of
the action and a route verifier buys nothing over an action critic.  The denoise
prefix sweep asks the separate question of how early the ranking is available.

Two fits, answering two questions that must not be conflated.

  leave-one-snapshot-out   can a *deployable selector* be built?  It must score a
                           candidate at a state it never trained on.  This is the
                           headline number.
  within-snapshot          is the information there at all?  Fitting and evaluating
                           inside one snapshot is not deployable -- at test time there
                           are no labels at the current state -- but if even this is
                           null, no transfer machinery will rescue it.  This needs
                           enough candidates per snapshot to fit at all, which is why
                           it appears only from the 32-candidate run onward.

Both nulls permute candidate labels within each snapshot, preserving the snapshot
structure and destroying only the candidate-to-outcome pairing; the within-snapshot
null refits the identical pipeline, so it pays for that fit's optimism exactly.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

from within64_analyze import _hat, _loo_scores

RNG = np.random.default_rng(0)
ACTION_TOKENS = slice(1, None)


def snapshot_centre(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64).copy()
    for g in np.unique(group):
        m = group == g
        out[m] -= out[m].mean(0)
    return out


def loso_operator(x: np.ndarray, group: np.ndarray, lam: float = 1.0) -> list:
    """The leave-one-snapshot-out ridge, precomputed as a linear map from labels.

    A ridge fit is linear in y and the design never involves y, so the whole
    leave-one-snapshot-out prediction is a fixed matrix acting on the labels.  Building it
    once turns the permutation test from "re-solve a ridge for every snapshot for every
    permutation" into one matrix product per snapshot covering all permutations, which is
    the same trick within64_analyze.py uses for its leave-one-out probe.
    """
    ops = []
    for g in np.unique(group):
        te = group == g
        tr = ~te
        xt = x[tr]
        mu, sd = xt.mean(0), xt.std(0) + 1e-8
        a = np.hstack([(xt - mu) / sd, np.ones((tr.sum(), 1))])
        b = np.hstack([(x[te] - mu) / sd, np.ones((te.sum(), 1))])
        pen = lam * np.eye(a.shape[1])
        pen[-1, -1] = 0.0
        ops.append((te, tr, b @ np.linalg.solve(a.T @ a + pen, a.T)))
    return ops


def loso_predict(ops: list, y: np.ndarray) -> np.ndarray:
    """Apply the operator to one label vector, or to many as columns of ``y``."""
    pred = np.empty(y.shape)
    for te, tr, m in ops:
        pred[te] = m @ y[tr]
    return pred


def _spearman(pred: np.ndarray, y: np.ndarray) -> float:
    a = np.argsort(np.argsort(pred)).astype(float)
    b = np.argsort(np.argsort(y)).astype(float)
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def within_spearman(pred: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """Mean Spearman correlation between predicted and actual ranks, per snapshot."""
    rs = [_spearman(pred[group == g], y[group == g])
          for g in np.unique(group) if (group == g).sum() >= 3]
    return float(np.mean(rs)) if rs else 0.0


def evaluate(x: np.ndarray, y: np.ndarray, group: np.ndarray, n_perm: int = 1000) -> dict:
    xc = snapshot_centre(x, group)
    yc = snapshot_centre(y[:, None], group).ravel()
    ops = loso_operator(xc, group)
    observed = within_spearman(loso_predict(ops, yc[:, None])[:, 0], y, group)

    # permute candidate labels inside each snapshot: this keeps the snapshot structure and
    # every snapshot's own spread, and destroys only which candidate got which outcome
    yp = np.tile(yc[:, None], (1, n_perm))
    for g in np.unique(group):
        m = np.flatnonzero(group == g)
        for j in range(n_perm):
            yp[m, j] = RNG.permutation(yp[m, j])
    pred = loso_predict(ops, yp)
    null = np.array([within_spearman(pred[:, j], yp[:, j], group) for j in range(n_perm)])
    return {
        "within_snapshot_spearman": round(observed, 3),
        "permutation_p": round(float((1 + (null >= observed).sum()) / (1 + n_perm)), 4),
        "null_mean": round(float(null.mean()), 3),
        "null_p95": round(float(np.quantile(null, 0.95)), 3),
    }


def within_snapshot_information(x: np.ndarray, y: np.ndarray, group: np.ndarray,
                                n_comp: int = 8, n_perm: int = 2000) -> dict:
    """Does the routing distinguish candidates *at one state*, fitting inside that state?

    This is a different question from the leave-one-snapshot-out fit above, and the two
    must not be conflated.  Leave-one-snapshot-out asks whether a *deployable selector*
    exists: it has to predict a candidate's consequence at a state it never trained on.
    That is the harder question, and it is the one that comes back null for the action
    chunk too -- which in a deterministic simulator literally causes the outcome, so a null
    there is about the map not transferring across states, not about the features.

    Fitting inside a snapshot answers the prior question: is the information *there* at
    all?  It is not a selector -- at deployment there are no labels at the current state --
    but if even this is null then no amount of transfer machinery will help.  With 8
    candidates per snapshot it could not be asked; with 32 it can.

    The optimism of fitting and evaluating on 32 points is paid for exactly, because the
    null permutes candidate labels inside each snapshot and refits the identical pipeline.
    """
    rows = []
    for g in np.unique(group):
        m = np.flatnonzero(group == g)
        if len(m) < 12 or np.ptp(y[m]) <= 0:
            continue                      # degenerate label: every candidate did the same
        z = reduce_dims(x[m], min(n_comp, len(m) - 3))
        hat, h_diag = _hat(z)
        yc = y[m] - y[m].mean()
        obs = _loo_scores(hat, h_diag, yc[:, None])[:, 0]
        perms = np.stack([RNG.permutation(yc) for _ in range(n_perm)], axis=1)
        null = _loo_scores(hat, h_diag, perms)
        rows.append((_spearman(obs, yc),
                     np.array([_spearman(null[:, j], perms[:, j]) for j in range(n_perm)])))
    if not rows:
        return {"n_snapshots": 0}
    obs = float(np.mean([r[0] for r in rows]))
    null = np.mean(np.stack([r[1] for r in rows]), axis=0)
    return {"n_snapshots": len(rows),
            "mean_within_snapshot_spearman": round(obs, 3),
            "permutation_p": round(float((1 + (null >= obs).sum()) / (1 + n_perm)), 4),
            "null_mean": round(float(null.mean()), 3),
            "null_p95": round(float(np.quantile(null, 0.95)), 3)}


def reduce_dims(x: np.ndarray, n_comp: int = 20) -> np.ndarray:
    n_comp = min(n_comp, x.shape[0] - 2, x.shape[1])
    xc = x - x.mean(0)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:n_comp].T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-drift-ratio", type=float, default=0.05,
                    help="drop a snapshot whose repeated-branch drift exceeds this "
                         "fraction of its own between-candidate spread")
    args = ap.parse_args()

    pilot = pathlib.Path(args.pilot_dir)
    records = json.loads((pilot / "fork_records.json").read_text())

    # Per-snapshot fidelity gate.  probe_fork_fidelity.py found that whether a branch
    # reproduces is a property of the *state*, not of the restore mode: with a clean
    # prefix the drift is 5e-15, but at two probed states it reached 2e-2 against a
    # candidate spread of 1.3e-2, and there a ranking is reading solver noise.  One
    # global check is therefore not enough; each snapshot carries its own repeat probe
    # and is kept or dropped on it.  Whether the drop correlates with outcome is
    # reported, because a biased drop would move every number downstream.
    def keep(r):
        s = r.get("candidate_spread", 0.0)
        return s <= 0 or r.get("rerun_drift", 0.0) <= args.max_drift_ratio * s
    dropped = sorted({(r["episode"], r["fork_step"]) for r in records if not keep(r)})
    degenerate = sorted({(r["episode"], r["fork_step"]) for r in records
                         if r.get("candidate_spread", 0.0) <= 0})
    records = [r for r in records if keep(r)]
    if not records:
        raise RuntimeError("the drift gate dropped every snapshot")

    group = np.array([r["episode"] * 1000 + r["fork_step"] for r in records])
    y = np.array([r["drawer_delta"] for r in records], dtype=np.float64)
    rows = np.array([r["trace_row"] for r in records])

    g = zarr.open(str(pathlib.Path(args.server_dir) / "routes.zarr"), mode="r")
    probs = np.asarray(g["hb_router_probs"][:], dtype=np.float32)   # [calls, 8, 10, 11, 32]
    if probs.shape[0] <= rows.max():
        raise RuntimeError("trace has %d rows but the pilot references row %d"
                           % (probs.shape[0], rows.max()))
    route = probs[rows][:, :, :, ACTION_TOKENS, :].mean(axis=3)     # [n, 8, 10, 32]

    chunks = []
    for r in records:
        c = np.load(pilot / ("chunks_ep%02d_t%02d.npy" % (r["episode"], r["fork_step"])))
        chunks.append(c[r["candidate"]].ravel())
    action = np.stack(chunks).astype(np.float64)

    n_snap = len(np.unique(group))
    out = {"n_candidates": len(records), "n_snapshots": n_snap,
           "snapshots_dropped_for_drift": ["ep%d/t%d" % d for d in dropped],
           "snapshots_with_degenerate_label": ["ep%d/t%d" % d for d in degenerate],
           "candidates_per_snapshot": len(records) // max(n_snap, 1),
           "drawer_delta_spread_per_snapshot": round(float(np.mean(
               [y[group == gg].max() - y[group == gg].min() for gg in np.unique(group)])), 5)}

    r_flat = route.reshape(len(route), -1)
    # the within-state information question, which N=8 could not support
    out["within_snapshot_fit"] = {
        "route": within_snapshot_information(r_flat, y, group),
        "action": within_snapshot_information(action, y, group),
    }

    out["predictors"] = {
        "route": evaluate(reduce_dims(r_flat), y, group),
        "action": evaluate(reduce_dims(action), y, group),
        "route_plus_action": evaluate(
            np.hstack([reduce_dims(r_flat), reduce_dims(action)]), y, group),
    }

    # how early is the ranking available -- denoise step 0 is where x_t is still the
    # raw noise draw, so a signal there would mean the certificate costs nothing
    out["denoise_prefix"] = [
        {"through_denoise_step": d,
         **evaluate(reduce_dims(route[:, :, :d].reshape(len(route), -1)), y, group, n_perm=500)}
        for d in range(1, route.shape[2] + 1)
    ]

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "denoise_prefix"}, indent=2))
    print("\n去噪前缀（候选级排序何时可得）:")
    for r in out["denoise_prefix"]:
        print("  1..%-2d  rho=%.3f  p=%.3f  null_p95=%.3f"
              % (r["through_denoise_step"], r["within_snapshot_spearman"],
                 r["permutation_p"], r["null_p95"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
