"""Last check: is (layer 15, expert 8) the loudest wire in every capture, and how
big is its swing in units of 1/32?"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

RUNS = ["within64-s24", "within64-t1s19", "within64-t3s0"]
U = 1.0 / N_EXPERTS


def main() -> None:
    out = {}
    for name in RUNS:
        run = load_run("runs/" + name, "runs/" + name + "-client")
        feats, t = [], []
        for e in run.episodes:
            feats.append(action_token_probs(e).mean(axis=2))   # [T, 8, 32] over denoise
            t += list(range(e.n_control))
        x = np.concatenate(feats).astype(np.float64)           # [N, 8, 32]
        t = np.array(t)
        v = x.var(0)
        share = v / v.sum()
        flat = np.argsort(-share, axis=None)[:5]
        p8 = x[:, 7, 8] / U
        curve = [round(float(p8[t == s].mean()), 3) for s in range(min(24, t.max() + 1))]
        out[name] = {
            "task": run.summaries[0].get("task_name", "?"),
            "init_state": run.episodes[0].init_state_id,
            "n_success": run.n_success,
            "top5_cells_by_variance": [
                {"layer": HB_LAYERS[i // N_EXPERTS], "expert": int(i % N_EXPERTS),
                 "variance_share": round(float(share.ravel()[i]), 3),
                 "mean_x_uniform": round(float(x[:, i // N_EXPERTS, i % N_EXPERTS].mean() / U), 3)}
                for i in flat],
            "layer15_expert8": {
                "variance_share": round(float(share[7, 8]), 3),
                "mean_x_uniform": round(float(p8.mean()), 3),
                "p01_x_uniform": round(float(np.quantile(p8, 0.01)), 3),
                "p99_x_uniform": round(float(np.quantile(p8, 0.99)), 3),
                "per_step_mean_x_uniform": curve,
            },
        }
    pathlib.Path("analysis/within64-s24/pc_meaning4.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
