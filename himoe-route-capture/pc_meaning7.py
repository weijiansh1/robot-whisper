"""Sweep the proprio-residual decode over every control step where all 64 draws
are still alive, so the single significant step found at 12 can be read as either
an isolated blip or one step of a window.

Reported alongside each residual accuracy is the proprio-only accuracy, because
the test has no power where proprio already saturates the decodable signal: at
steps where the arm's configuration alone reads ~0.86, a null residual is a
statement about headroom, not about routing.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import action_token_probs, load_run
from within64_analyze import decode, reduce_dims
from pc_meaning6 import knn_residual, proprio_basis

RUN = "within64-s24"


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

    rows = []
    for s in range(17):
        m = t == s
        if m.sum() < 64:
            continue
        r = {"step": int(s)}
        for name, src in (("raw", x), ("linear", res_lin), ("knn", res_knn)):
            d = decode(reduce_dims(src[m]), y[m])
            r[name] = round(d["balanced_accuracy"], 3)
            r[name + "_p"] = round(d["permutation_p"], 4)
        d = decode(state[m], y[m])
        r["proprio"] = round(d["balanced_accuracy"], 3)
        r["proprio_p"] = round(d["permutation_p"], 4)
        rows.append(r)

    n_sig = sum(1 for r in rows if r["knn_p"] < 0.05)
    out = {"rows": rows, "n_steps_tested": len(rows), "n_knn_significant_p05": n_sig,
           "bonferroni_alpha": round(0.05 / len(rows), 4)}
    pathlib.Path("analysis/within64-s24/pc_meaning7.json").write_text(json.dumps(out, indent=2))
    print("%-5s %-7s %-7s %-7s %-7s %s" % ("step", "raw", "linear", "knn", "proprio", "knn p"))
    for r in rows:
        print("%-5d %-7.3f %-7.3f %-7.3f %-7.3f %.4f"
              % (r["step"], r["raw"], r["linear"], r["knn"], r["proprio"], r["knn_p"]))
    print("\n%d/%d steps significant at p<0.05, Bonferroni alpha = %.4f"
          % (n_sig, len(rows), out["bonferroni_alpha"]))


if __name__ == "__main__":
    main()
