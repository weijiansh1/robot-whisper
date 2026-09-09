"""Cheapest precursor to the RouteCert programme, on data already on disk.

The proposed programme asks two questions that cost 3200 continuation branches to
answer properly.  Two weaker versions can be answered right now, at control step
11 -- the one step where routing decodes the outcome (0.852) while the proprio
state is at chance (0.528):

  where   which HB layer carries it, and does it survive removing proprio
  when    how many flow denoising steps are needed before it is present

Step 14 is run alongside as the negative control: there proprio saturates and the
routing residual is already known to be at chance, so any layer/prefix structure
found at 14 is a readout of the arm, not of the candidate.

If the denoise-prefix curve is flat until step 9-10, the early-exit branch of the
programme is dead before any new capture is run.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, action_token_probs, load_run
from within64_analyze import decode, reduce_dims
from pc_meaning6 import knn_residual, proprio_basis

RUN = "within64-s24"
STEPS = (11, 14)


def main() -> None:
    run = load_run("runs/" + RUN, "runs/" + RUN + "-client")
    client = pathlib.Path("runs/" + RUN + "-client")
    acts, t, y, ep, state = [], [], [], [], []
    for e in run.episodes:
        acts.append(action_token_probs(e))                 # [T, 8, 10, 32]
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
        ep += [e.index] * e.n_control
        with np.load(client / ("episode_%02d.npz" % e.index)) as zf:
            state.append(zf["state"])
    a = np.concatenate(acts).astype(np.float64)            # [N, 8, 10, 32]
    x = a.reshape(len(a), -1)
    t, y, ep = np.array(t), np.array(y), np.array(ep)
    state = np.concatenate(state)

    # the same proprio controls as pc_meaning6, fitted on all rows
    b = proprio_basis(state)
    beta, *_ = np.linalg.lstsq(b, x, rcond=None)
    res_knn = knn_residual(x, state, ep).reshape(a.shape)
    res_lin = (x - b @ beta).reshape(a.shape)

    out = {"steps": {}}
    for s in STEPS:
        m = t == s
        yy = y[m]
        rec = {"n": int(m.sum()), "n_success": int(yy.sum()), "per_layer": [],
               "denoise_prefix": [], "denoise_single": []}

        for li, layer in enumerate(HB_LAYERS):
            row = {"layer": layer}
            for name, src in (("raw", a), ("knn_residual", res_knn)):
                d = decode(reduce_dims(src[m][:, li].reshape(int(m.sum()), -1)), yy)
                row[name] = round(d["balanced_accuracy"], 3)
                row[name + "_p"] = round(d["permutation_p"], 4)
            rec["per_layer"].append(row)

        for dstep in range(1, 11):
            row = {"prefix_through_denoise_step": dstep}
            for name, src in (("raw", a), ("knn_residual", res_knn)):
                d = decode(reduce_dims(src[m][:, :, :dstep].reshape(int(m.sum()), -1)), yy)
                row[name] = round(d["balanced_accuracy"], 3)
                row[name + "_p"] = round(d["permutation_p"], 4)
            rec["denoise_prefix"].append(row)

        for dstep in range(10):
            d = decode(reduce_dims(res_knn[m][:, :, dstep].reshape(int(m.sum()), -1)), yy)
            rec["denoise_single"].append(
                {"denoise_step": dstep, "knn_residual": round(d["balanced_accuracy"], 3),
                 "p": round(d["permutation_p"], 4)})

        # the single loudest cell on its own, as a one-number baseline
        d = decode(res_knn[m][:, 7, :, 8].reshape(int(m.sum()), -1), yy)
        rec["L15E8_only_knn_residual"] = {"balanced_accuracy": round(d["balanced_accuracy"], 3),
                                          "p": round(d["permutation_p"], 4)}
        out["steps"][str(s)] = rec

    pathlib.Path("analysis/within64-s24/precursor_step11.json").write_text(json.dumps(out, indent=2))
    for s in STEPS:
        r = out["steps"][str(s)]
        print("\n===== 控制步 %d  (n=%d, 成功 %d) =====" % (s, r["n"], r["n_success"]))
        print("按层（原始 / 扣 proprio）:")
        for row in r["per_layer"]:
            print("  L%-3d %.3f p=%-6.3f   %.3f p=%.3f"
                  % (row["layer"], row["raw"], row["raw_p"],
                     row["knn_residual"], row["knn_residual_p"]))
        print("按去噪前缀（原始 / 扣 proprio）:")
        for row in r["denoise_prefix"]:
            print("  1..%-2d %.3f p=%-6.3f   %.3f p=%.3f"
                  % (row["prefix_through_denoise_step"], row["raw"], row["raw_p"],
                     row["knn_residual"], row["knn_residual_p"]))
        print("  仅 L15/E8（扣 proprio）: %.3f p=%.3f"
              % (r["L15E8_only_knn_residual"]["balanced_accuracy"],
                 r["L15E8_only_knn_residual"]["p"]))


if __name__ == "__main__":
    main()
