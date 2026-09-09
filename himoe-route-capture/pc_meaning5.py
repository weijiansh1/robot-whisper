"""If PC1 is just the loudest cell, what is underneath it?

The pipeline PCA runs on raw probabilities, so a single (layer, expert) cell that
holds 16-31% of the feature variance wins PC1 almost by construction.  This refits
on z-scored features -- every cell contributes equally -- and asks whether the same
two axes come back or something else does.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

RUNS = ["within64-s24", "within64-t1s19", "within64-t3s0"]


def r2(z: np.ndarray, cols: np.ndarray) -> float:
    d = np.column_stack([np.ones(len(z)), cols - cols.mean(0)])
    beta, *_ = np.linalg.lstsq(d, z, rcond=None)
    return float(1 - (z - d @ beta).var() / z.var())


def main() -> None:
    out = {}
    for name in RUNS:
        run = load_run("runs/" + name, "runs/" + name + "-client")
        client = pathlib.Path("runs/" + name + "-client")
        feats, t, y, state = [], [], [], []
        for e in run.episodes:
            feats.append(action_token_probs(e).reshape(e.n_control, -1))
            t += list(range(e.n_control))
            y += [int(e.success)] * e.n_control
            with np.load(client / ("episode_%02d.npz" % e.index)) as zf:
                state.append(zf["state"])
        x = np.concatenate(feats).astype(np.float64)
        t = np.array(t, float)
        y = np.array(y)
        state = np.concatenate(state)
        cells = x.reshape(len(x), len(HB_LAYERS), 10, N_EXPERTS).mean(2)

        xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
        _u, s, vt = np.linalg.svd(xs, full_matrices=False)
        var = (s ** 2) / (s ** 2).sum()
        z = xs @ vt[:3].T

        rec = {"explained_whitened": var[:3].round(3).tolist(), "components": []}
        for c in range(3):
            w = np.abs(vt[c]).reshape(len(HB_LAYERS), 10, N_EXPERTS).sum(1)
            li, ei = np.unravel_index(np.argmax(w), w.shape)
            e2 = (vt[c].reshape(len(HB_LAYERS), 10, N_EXPERTS) ** 2).sum(axis=(1, 2))
            rec["components"].append({
                "component": c + 1,
                "top_cell": {"layer": HB_LAYERS[li], "expert": int(ei)},
                "energy_share_early_block_L2_5": round(float(e2[:4].sum() / e2.sum()), 3),
                "energy_share_layer15": round(float(e2[7] / e2.sum()), 3),
                "top_cell_energy_share": round(float((vt[c].reshape(
                    len(HB_LAYERS), 10, N_EXPERTS)[li, :, ei] ** 2).sum() / e2.sum()), 3),
                "corr_control_step": round(float(np.corrcoef(z[:, c], t)[0, 1]), 3),
                "corr_L15E8": round(float(np.corrcoef(z[:, c], cells[:, 7, 8])[0, 1]), 3),
                "r2_from_proprio": round(r2(z[:, c], state), 3),
                "corr_success": round(float(np.corrcoef(z[:, c], y.astype(float))[0, 1]), 3),
            })
        out[name] = rec
    pathlib.Path("analysis/within64-s24/pc_meaning5.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
