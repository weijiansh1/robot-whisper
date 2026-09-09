#!/usr/bin/env python3
"""Fill the remaining empirical cells of the denoise-stop design space, offline.

On flow-lead-cpu-t0s24 (704 trajectories):
  A3  second-order extrapolation vs the one-jump x_hat
  B4  signal quality of ||v||*t as a stop signal (corr with true x_hat error)
  B5  signal quality of ||x_hat_m - x_hat_{m-1}|| (extrapolation stability)
  C3  per-query heterogeneity of the minimal safe r (is adaptive worth it?)
  C4  does the safe r depend on the control step within the episode?
Writes audit_stop_design_space.json.
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / "himoe-route-capture/runs/flow-lead-cpu-t0s24"
LIVE = 7
BAND = 0.049          # seed-reroll perturbation scale (rate-neutral)


def rel(a, b):
    d = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    n = np.linalg.norm(b.reshape(len(b), -1), axis=1)
    return d / np.maximum(n, 1e-9)


def main() -> int:
    parts = [np.load(p) for p in sorted(glob.glob(str(RUN / "flow_traces_part*.npz")))]
    X = np.concatenate([p["x_traj"] for p in parts]).squeeze(2)
    qid = np.concatenate([p["query_id"] for p in parts])
    x10 = X[:, 10, :, :LIVE]
    n = len(X)
    cs = qid % 11                                   # control step within episode

    V = (X[:, :-1] - X[:, 1:]) / 0.1                # v_m, m=0..9
    T = 1.0 - 0.1 * np.arange(10)

    e1 = np.zeros((n, 10))                          # one-jump
    e2 = np.full((n, 10), np.nan)                   # second-order (needs m>=1)
    xh_prev = None
    stab = np.full((n, 10), np.nan)                 # ||x_hat_m - x_hat_{m-1}||
    vsig = np.zeros((n, 10))                        # ||v||*t
    for m in range(10):
        xh = (X[:, m] - T[m] * V[:, m])[:, :, :LIVE]
        e1[:, m] = rel(xh, x10)
        vsig[:, m] = (T[m] * np.linalg.norm(V[:, m, :, :LIVE].reshape(n, -1), axis=1)
                      / np.maximum(np.linalg.norm(x10.reshape(n, -1), axis=1), 1e-9))
        if m >= 1:
            vex = V[:, m] + 0.5 * (V[:, m] - V[:, m - 1])
            xh2 = (X[:, m] - T[m] * vex)[:, :, :LIVE]
            e2[:, m] = rel(xh2, x10)
            stab[:, m] = rel(xh, xh_prev)
        xh_prev = xh

    out = {"rows": int(n), "band": BAND}

    print("A2 vs A3: median rel err by rounds executed r=m+1")
    print("  r     one-jump   2nd-order")
    a = {}
    for m in range(1, 10):
        a[m + 1] = {"e1": float(np.median(e1[:, m])), "e2": float(np.median(e2[:, m]))}
        print("  %-4d  %.3f      %.3f" % (m + 1, a[m + 1]["e1"], a[m + 1]["e2"]))
    out["extrapolation"] = a

    # C3: minimal r per trajectory such that e1 <= BAND
    minr = np.full(n, 11)
    for m in range(9, -1, -1):
        minr[e1[:, m] <= BAND] = m + 1
    dist = {str(r): int((minr == r).sum()) for r in range(1, 12)}
    per_q = np.array([minr[qid == q].mean() for q in np.unique(qid)])
    out["min_safe_r"] = {
        "dist": dist,
        "p10": float(np.quantile(minr, .1)), "p50": float(np.quantile(minr, .5)),
        "p90": float(np.quantile(minr, .9)),
        "frac_le6": float((minr <= 6).mean()),
        "per_query_mean_range": [float(per_q.min()), float(per_q.max())],
        "between_query_sd": float(per_q.std()),
        "within_query_sd": float(np.mean([minr[qid == q].std()
                                          for q in np.unique(qid)])),
    }
    print("\nC3 minimal safe r (e1<=%.3f): p10/p50/p90 = %.0f/%.0f/%.0f, "
          "share r<=6: %.2f" % (BAND, out["min_safe_r"]["p10"],
                                out["min_safe_r"]["p50"], out["min_safe_r"]["p90"],
                                out["min_safe_r"]["frac_le6"]))
    print("   between-query sd %.2f vs within-query sd %.2f"
          % (out["min_safe_r"]["between_query_sd"],
             out["min_safe_r"]["within_query_sd"]))

    # B4/B5: pooled + per-round Spearman of signal vs the TRUE remaining error
    b = {}
    for name, S in (("vnorm_t", vsig), ("xhat_stability", stab)):
        rows = []
        for m in range(1, 10):
            s = S[:, m]
            ok = np.isfinite(s)
            rows.append(float(spearmanr(s[ok], e1[ok, m]).statistic))
        pooled_s = S[:, 1:10].ravel()
        pooled_e = e1[:, 1:10].ravel()
        ok = np.isfinite(pooled_s)
        b[name] = {"per_round_spearman": rows,
                   "pooled": float(spearmanr(pooled_s[ok], pooled_e[ok]).statistic)}
        print("B  %s: pooled rho=%.3f  per-round %.2f..%.2f"
              % (name, b[name]["pooled"], min(rows), max(rows)))
    out["signals"] = b

    # C4: does safe r depend on the control step?
    c4 = {}
    for lo, hi, lab in ((0, 3, "cs0-2"), (3, 7, "cs3-6"), (7, 11, "cs7-10")):
        m = (cs >= lo) & (cs < hi)
        c4[lab] = {"minr_p50": float(np.median(minr[m])),
                   "e1_r6_median": float(np.median(e1[m, 5]))}
    out["by_control_step"] = c4
    print("C4 by control step:", json.dumps(c4))

    (HERE / "audit_stop_design_space.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_stop_design_space.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
