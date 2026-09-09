#!/usr/bin/env python3
"""Audit (a): is the t08 path-length advantage anything beyond a nonlinear
reparametrisation of the pose trajectory?

The trajectory-shape result (routing path length AUC 0.808 vs physical 0.743,
surviving a gripper-transition regression at 0.715 vs 0.572) compared the
routing's own 256-dim probability space against the raw 47-dim simulator state.
But the state token's routing is ~96% a nonlinear function of the 8-dim proprio
alone, so the honest control is the path length of the *pose-predicted* routing
trajectory: push proprio through a fixed nonlinear map into the same 256-dim
space (leave-one-scene-out so the map never sees the scored scene) and measure
the same features there.

  If AUC(pred) ~= AUC(true), the advantage is a metric effect of the learned
  pose embedding and carries no routing-specific information.
  If AUC(true) >> AUC(pred), the extra 4% non-pose variance is doing the work.

Also reported: raw proprio path length, the residual trajectory (true - pred),
and every arm after regressing out gripper transitions within scene.
Writes audit_pathlen_control.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
NPZ = (HERE / "himoe-routing-rules-20260819/data"
       / "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz")
WIN = 34
RNG = np.random.default_rng(0)
N_PERM = 2000
N_RFF = 1500


def sauc(x, y, scene):
    num = den = 0.0
    for s in np.unique(scene):
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(x[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else float("nan")


def perm_p(x, y, scene, n=N_PERM):
    obs = abs(sauc(x, y, scene) - .5)
    hit = 0
    for _ in range(n):
        yp = y.copy()
        for s in np.unique(scene):
            i = np.flatnonzero(scene == s)
            yp[i] = y[RNG.permutation(i)]
        hit += abs(sauc(x, yp, scene) - .5) >= obs
    return (1 + hit) / (n + 1)


def std_pooled(X):
    """Standardise each channel on the pooled data (recurrence protocol)."""
    X = X - X.reshape(-1, X.shape[-1]).mean(0)
    X = X / (X.reshape(-1, X.shape[-1]).std(0) + 1e-9)
    return X


def path_len(X):
    return np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1)


def resid_within_scene(feat, ctrl, scene):
    """OLS-residualise feat on ctrl inside each scene."""
    out = np.zeros_like(feat)
    for s in np.unique(scene):
        m = scene == s
        f, c = feat[m] - feat[m].mean(), ctrl[m] - ctrl[m].mean()
        v = (c * c).sum()
        out[m] = f - (f * c).sum() / v * c if v > 1e-12 else f
    return out


def main() -> int:
    z = np.load(NPZ, allow_pickle=True)
    n_rows = z["n_rows"]
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = z["success"].astype(bool)
    scene = z["scene"]
    n = len(y)
    probs = z["state_token_probs"].astype(np.float32)     # (total, 8, 32)
    prop_all = z["proprio"]
    sim_all = z["sim_state"]

    R = np.zeros((n, WIN, 256), np.float32)
    P8 = np.zeros((n, WIN, 8), np.float32)
    P47 = np.zeros((n, WIN, sim_all.shape[1]), np.float32)
    for i in range(n):
        R[i] = probs[off[i]:off[i] + WIN].reshape(WIN, 256)
        P8[i] = prop_all[off[i]:off[i] + WIN]
        P47[i] = sim_all[off[i]:off[i] + WIN]

    # ---- pose -> routing map, leave-one-scene-out, random Fourier features --
    Xf = P8.reshape(-1, 8)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xs = (Xf - mu) / sd
    sub = RNG.choice(len(Xs), 3000, replace=False)
    d2 = ((Xs[sub, None, :] - Xs[sub[::-1], :][None, ::50][0][None]) ** 2)
    med = np.median(np.sqrt(((Xs[sub][:, None, :] - Xs[sub[:60]][None]) ** 2)
                            .sum(-1)))
    gamma = 1.0 / (med ** 2 + 1e-9)
    W = RNG.normal(0, np.sqrt(2 * gamma), (8, N_RFF))
    b = RNG.uniform(0, 2 * np.pi, N_RFF)

    def phi(X):
        return np.sqrt(2.0 / N_RFF) * np.cos(((X - mu) / sd) @ W + b)

    Yf = R.reshape(-1, 256)
    scene_row = np.repeat(scene, WIN)
    pred = np.zeros_like(Yf)
    lam = 1.0
    for h in np.unique(scene):
        tr = scene_row != h
        Ph = phi(Xf[tr])
        A = Ph.T @ Ph + lam * np.eye(N_RFF)
        Bm = Ph.T @ Yf[tr]
        coef = np.linalg.solve(A, Bm)
        te = ~tr
        pred[te] = phi(Xf[te]) @ coef
    ss_res = ((Yf - pred) ** 2).sum()
    ss_tot = ((Yf - Yf.mean(0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    Rp = pred.reshape(n, WIN, 256)
    Rr = (Yf - pred).reshape(n, WIN, 256)

    # gripper transitions (opening = f_a - f_b, closed < 0.04)
    opening = P8[:, :, 6] - P8[:, :, 7]
    closed = opening < 0.04
    trans = (closed[:, 1:] != closed[:, :-1]).sum(1).astype(np.float32)

    arms = {
        "routing_true": std_pooled(R.copy()),
        "routing_pred_from_pose": std_pooled(Rp.copy()),
        "routing_residual": std_pooled(Rr.copy()),
        "sim_state_47": std_pooled(P47.copy()),
        "proprio_8": std_pooled(P8.copy()),
    }
    out = {"task": "libero_long/t08", "n": int(n), "win": WIN,
           "map_loso_R2": float(r2),
           "transitions_auc": float(sauc(trans, y, scene))}
    print("t08 path-length control, %d eps, window %d" % (n, WIN))
    print("pose->routing LOSO R2 (RFF-%d ridge): %.3f" % (N_RFF, r2))
    print("gripper transitions alone: AUC %.3f" % out["transitions_auc"])
    print("\n  arm                        pathlen AUC (p)      after transition-regress AUC (p)   corr(len,trans)")
    for name, X in arms.items():
        pl = path_len(X)
        a1, p1 = sauc(pl, y, scene), perm_p(pl, y, scene)
        rl = resid_within_scene(pl, trans, scene)
        a2, p2 = sauc(rl, y, scene), perm_p(rl, y, scene)
        c = np.corrcoef(pl, trans)[0, 1]
        out[name] = {"auc": float(a1), "p": float(p1),
                     "auc_detrans": float(a2), "p_detrans": float(p2),
                     "corr_transitions": float(c)}
        print("  %-25s  %.3f (p=%.3f)      %.3f (p=%.3f)                  %.3f"
              % (name, a1, p1, a2, p2, c))

    (HERE / "audit_pathlen_control.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_pathlen_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
