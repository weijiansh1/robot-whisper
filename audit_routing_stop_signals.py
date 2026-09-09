#!/usr/bin/env python3
"""Every routing-derived stop signal, scored on one ruler.

For each candidate signal S(routing at rounds <= m) measure how well it tracks
the TRUE remaining error e1(m) = relL2(x_hat_m, x10), per round and pooled,
against the non-routing reference ||v||*t (0.932 pooled).  Also tests the
state-token depth SCHEDULER (predict the per-trajectory minimal safe r before
any latent information exists).  flow-lead-cpu-t0s24, 704 trajectories.
Writes audit_routing_stop_signals.json.
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


def rel(a, b):
    d = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    n = np.linalg.norm(b.reshape(len(b), -1), axis=1)
    return d / np.maximum(n, 1e-9)


def loeo_ridge(X, y, ep, alpha=10.0):
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
    x10 = X[:, 10, :, :LIVE]
    V = (X[:, :-1] - X[:, 1:]) / 0.1
    T = 1.0 - 0.1 * np.arange(10)
    e1 = np.zeros((n, 10))
    vsig = np.zeros((n, 10))
    for m in range(10):
        xh = (X[:, m] - T[m] * V[:, m])[:, :, :LIVE]
        e1[:, m] = rel(xh, x10)
        vsig[:, m] = (T[m] * np.linalg.norm(V[:, m, :, :LIVE].reshape(n, -1), axis=1)
                      / np.maximum(np.linalg.norm(x10.reshape(n, -1), axis=1), 1e-9))
    minr = np.full(n, 11)
    for m in range(9, -1, -1):
        minr[e1[:, m] <= BAND] = m + 1

    g = zarr.open_group(str(RUN / "routes.zarr"), mode="r")
    P = np.asarray(g["hb_router_probs"][:], np.float32)      # (704,8,10,11,32)
    P /= np.maximum(P.sum(-1, keepdims=True), 1e-9)
    A = P[:, :, :, 1:, :]                                     # action tokens
    ST = P[:, :, 0, 0, :].reshape(n, -1)                      # state token, 256
    R = np.sqrt(A)
    ids = np.argsort(A, axis=-1)[..., -4:]                    # top-4 per site

    def hell(a, b):                                           # RMS site Hellinger
        return np.sqrt((0.5 * ((a - b) ** 2).sum(-1)).mean(axis=(1, 2)))

    sig = {}
    # 1) round-to-round change family
    ch = np.full((n, 10), np.nan)
    jc = np.full((n, 10), np.nan)
    fl = np.full((n, 10), np.nan)
    for m in range(1, 10):
        ch[:, m] = hell(R[:, :, m], R[:, :, m - 1])
        inter = np.zeros((n, 8, 10))
        for k in range(4):
            inter += (ids[:, :, m, :, k:k + 1] == ids[:, :, m - 1]).any(-1)
        jc[:, m] = 1 - (inter / (8 - inter)).mean(axis=(1, 2))
        fl[:, m] = (ids[:, :, m, :, -1] != ids[:, :, m - 1, :, -1]).mean(axis=(1, 2))
    sig["chg_hellinger"] = ch
    sig["chg_jaccard"] = jc
    sig["top1_flip"] = fl
    # 2) instantaneous family
    ent = -(A * np.log(np.maximum(A, 1e-12))).sum(-1).mean(axis=(2, 3)) / np.log(32)
    sig["entropy"] = np.transpose(ent, (0, 1))[:, :]           # (n,8?) fix below
    sig["entropy"] = -(A * np.log(np.maximum(A, 1e-12))).sum(-1).mean(axis=(1, 3)) / np.log(32)
    srt = np.sort(A, axis=-1)
    sig["margin"] = (srt[..., -1] - srt[..., -2]).mean(axis=(1, 3))
    sig["mass4"] = srt[..., -4:].sum(-1).mean(axis=(1, 3))
    tok = R.mean(1)                                            # (n,10,10tok,32)
    d_tok = np.sqrt((0.5 * ((tok[:, :, :, None] - tok[:, :, None]) ** 2).sum(-1))
                    .mean(axis=(2, 3)))
    sig["token_coherence"] = d_tok
    T9 = R[:, :, 9].mean(0)                                    # final-round template
    sig["dist_to_final_template"] = np.stack(
        [hell(R[:, :, m], T9[None].repeat(n, 0)) for m in range(10)], 1)
    sig["chg_backblock"] = np.full((n, 10), np.nan)
    for m in range(1, 10):
        sig["chg_backblock"][:, m] = np.sqrt(
            (0.5 * ((R[:, 4:, m] - R[:, 4:, m - 1]) ** 2).sum(-1)).mean(axis=(1, 2)))

    out = {"reference_vnorm_t_pooled": float(
        spearmanr(vsig[:, 1:].ravel(), e1[:, 1:].ravel()).statistic)}
    print("signal                      pooled rho   per-round range     (ref vnorm_t %.3f)"
          % out["reference_vnorm_t_pooled"])
    scores = {}
    for k, S in sig.items():
        rows = []
        for m in range(1, 10):
            s = S[:, m]
            ok = np.isfinite(s)
            rows.append(float(spearmanr(s[ok], e1[ok, m]).statistic))
        pool_ok = np.isfinite(S[:, 1:].ravel())
        pooled = float(spearmanr(S[:, 1:].ravel()[pool_ok],
                                 e1[:, 1:].ravel()[pool_ok]).statistic)
        scores[k] = {"pooled": pooled, "per_round": rows}
        print("  %-24s  %+0.3f      %+0.2f .. %+0.2f"
              % (k, pooled, min(rows), max(rows)))
    out["signals"] = scores

    # 3) learned readout: all routing stats at round m -> e1(m), LOEO
    feats = np.stack([np.nan_to_num(sig[k]) for k in sig], -1)  # (n,10,K)
    Yf, Pf, Ep, Ms = [], [], [], []
    for m in range(1, 10):
        Yf.append(e1[:, m]); Pf.append(feats[:, m]); Ep.append(ep)
        Ms.append(np.full(n, m))
    Yf = np.concatenate(Yf); Pf = np.vstack(Pf)
    Ep = np.concatenate(Ep); Ms = np.concatenate(Ms)
    Xl = np.c_[Pf, np.eye(10)[Ms][:, 1:]]                      # + round one-hot
    pr = loeo_ridge(Xl, Yf, Ep)
    out["learned_route_all"] = float(spearmanr(pr, Yf).statistic)
    vv = np.concatenate([vsig[:, m] for m in range(1, 10)])
    pr2 = loeo_ridge(np.c_[vv], Yf, Ep)
    pr3 = loeo_ridge(np.c_[vv, Xl], Yf, Ep)
    out["learned_vnorm_only"] = float(spearmanr(pr2, Yf).statistic)
    out["learned_vnorm_plus_route"] = float(spearmanr(pr3, Yf).statistic)
    print("\nlearned (LOEO):  route-all %.3f   vnorm %.3f   vnorm+route %.3f"
          % (out["learned_route_all"], out["learned_vnorm_only"],
             out["learned_vnorm_plus_route"]))

    # ablation: is the revival information or calibration?
    pr4 = loeo_ridge(Pf, Yf, Ep)                      # all signals, NO round id
    out["ablate_no_round"] = float(spearmanr(pr4, Yf).statistic)
    k0 = list(sig).index("chg_hellinger")
    Xh = np.c_[Pf[:, k0], np.eye(10)[Ms][:, 1:]]      # one dead signal + round id
    pr5 = loeo_ridge(Xh, Yf, Ep)
    out["ablate_hell_plus_round"] = float(spearmanr(pr5, Yf).statistic)
    pr6 = loeo_ridge(np.eye(10)[Ms][:, 1:], Yf, Ep)   # round id alone
    out["ablate_round_only"] = float(spearmanr(pr6, Yf).statistic)
    print("ablation: all-signals-no-round %.3f | hellinger+round %.3f | round-only %.3f"
          % (out["ablate_no_round"], out["ablate_hell_plus_round"],
             out["ablate_round_only"]))

    # fair fight: give the latent side equal feature richness
    xh_all = np.stack([(X[:, m] - T[m] * V[:, m])[:, :, :LIVE] for m in range(10)], 1)
    lat_rows = []
    for m in range(1, 10):
        vt = np.abs(V[:, m]) * T[m]                       # (n,10,24)
        per_dim = vt.mean(1)                              # 24
        per_pos = np.linalg.norm(vt[:, :, :LIVE], axis=2)  # 10
        stab_m = rel(xh_all[:, m], xh_all[:, m - 1])[:, None]
        lat_rows.append(np.c_[per_dim, per_pos, vsig[:, m][:, None], stab_m])
    Lf = np.vstack(lat_rows)
    Xlat = np.c_[Lf, np.eye(10)[Ms][:, 1:]]
    pr_l = loeo_ridge(Xlat, Yf, Ep)
    pr_lr = loeo_ridge(np.c_[Lf, Pf, np.eye(10)[Ms][:, 1:]], Yf, Ep)
    out["fair_latent_rich"] = float(spearmanr(pr_l, Yf).statistic)
    out["fair_latent_rich_plus_route"] = float(spearmanr(pr_lr, Yf).statistic)
    print("fair fight: latent-rich %.3f | latent-rich+route %.3f"
          % (out["fair_latent_rich"], out["fair_latent_rich_plus_route"]))

    # 4) state-token depth scheduler: predict min_safe_r before any rounds
    for label, F in (("state_token_256", ST),
                     ("control_step_only", np.eye(11)[qid % 11]),
                     ("state+cs", np.c_[ST, np.eye(11)[qid % 11]])):
        p = loeo_ridge(F, minr.astype(float), ep)
        rho = float(spearmanr(p, minr).statistic)
        ss = 1 - ((p - minr) ** 2).sum() / ((minr - minr.mean()) ** 2).sum()
        out["scheduler_" + label] = {"spearman": rho, "r2": float(ss)}
        print("scheduler %-18s  rho %.3f  R2 %.3f" % (label, rho, ss))

    (HERE / "audit_routing_stop_signals.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_routing_stop_signals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
