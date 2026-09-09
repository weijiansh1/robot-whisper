"""Is there anything in the routing that the observation does not already explain?

The leading PCs are ~0.94 predictable from the 8-dim proprio state, so the obvious
follow-up is to remove that first and look at what survives.  Two controls, weak
and strong:

  linear   residual against [1, S, S^2, S_i*S_j] -- 45 proprio terms
  knn      residual against the mean routing state of the 15 nearest proprio
           neighbours drawn from *other episodes*, which also absorbs whatever
           nonlinear dependence the quadratic fit misses

Control step 0 is the reference point: there the observation is byte-identical
across all 64 draws, so raw features are already a pure residual and the decode
there is known to be null.  The question is whether steps 12-16, where the raw
decode reads ~0.9, hold anything once the arm's configuration is taken out.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import action_token_probs, load_run
from within64_analyze import decode, reduce_dims

RUN = "within64-s24"
STEPS = (0, 12, 14, 16)
K_NN = 15


def proprio_basis(s: np.ndarray) -> np.ndarray:
    """[1, S, quadratic] -- 45 columns for the 8 proprio dims."""
    quad = [s[:, i] * s[:, j] for i in range(s.shape[1]) for j in range(i, s.shape[1])]
    return np.column_stack([np.ones(len(s)), s] + quad)


def knn_residual(x: np.ndarray, s: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Subtract the mean routing state of the k nearest proprio neighbours.

    Neighbours are taken from other episodes only.  Within an episode consecutive
    control steps sit on top of each other in proprio space, so allowing them
    would subtract the row with itself and manufacture a null.
    """
    ss = (s - s.mean(0)) / (s.std(0) + 1e-12)
    out = np.empty_like(x)
    for i in range(len(x)):
        d = np.linalg.norm(ss - ss[i], axis=1)
        d[ep == ep[i]] = np.inf
        nn = np.argpartition(d, K_NN)[:K_NN]
        out[i] = x[i] - x[nn].mean(0)
    return out


def main() -> None:
    run = load_run("runs/" + RUN, "runs/" + RUN + "-client")
    client = pathlib.Path("runs/" + RUN + "-client")
    feats, t, y, ep, state = [], [], [], [], []
    for e in run.episodes:
        feats.append(action_token_probs(e).reshape(e.n_control, -1))
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
        ep += [e.index] * e.n_control
        with np.load(client / ("episode_%02d.npz" % e.index)) as zf:
            state.append(zf["state"])
    x = np.concatenate(feats).astype(np.float64)
    t, y, ep = np.array(t), np.array(y), np.array(ep)
    state = np.concatenate(state)

    b = proprio_basis(state)
    beta, *_ = np.linalg.lstsq(b, x, rcond=None)
    res_lin = x - b @ beta
    res_knn = knn_residual(x, state, ep)

    tot = (x - x.mean(0)).var(0).sum()
    out = {
        "variance_surviving": {
            "linear_proprio_removed": round(float(res_lin.var(0).sum() / tot), 3),
            "knn_proprio_removed": round(float(res_knn.var(0).sum() / tot), 3),
        },
        "leading_pc_of_residual": {},
        "decode_by_step": [],
    }

    for name, r in (("linear", res_lin), ("knn", res_knn)):
        rc = r - r.mean(0)
        _u, s, vt = np.linalg.svd(rc, full_matrices=False)
        z = rc @ vt[:2].T
        out["leading_pc_of_residual"][name] = {
            "explained": ((s ** 2) / (s ** 2).sum())[:2].round(3).tolist(),
            "pc1_corr_control_step": round(float(np.corrcoef(z[:, 0], t.astype(float))[0, 1]), 3),
            "pc1_corr_success": round(float(np.corrcoef(z[:, 0], y.astype(float))[0, 1]), 3),
        }

    # the decisive test: does the outcome decode survive the control?
    for s_ in STEPS:
        m = t == s_
        if m.sum() < 64:
            continue
        row = {"step": int(s_), "n": int(m.sum()), "n_success": int(y[m].sum())}
        for name, src in (("raw", x), ("linear_residual", res_lin), ("knn_residual", res_knn)):
            d = decode(reduce_dims(src[m]), y[m])
            row[name] = {"balanced_accuracy": round(d["balanced_accuracy"], 3),
                         "permutation_p": round(d["permutation_p"], 4),
                         "null_p95": round(d["null_p95"], 3)}
        # proprio itself, as the thing the routing is standing in for
        d = decode(state[m], y[m])
        row["proprio_only"] = {"balanced_accuracy": round(d["balanced_accuracy"], 3),
                               "permutation_p": round(d["permutation_p"], 4)}
        out["decode_by_step"].append(row)

    pathlib.Path("analysis/within64-s24/pc_meaning6.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
