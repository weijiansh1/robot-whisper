"""Does the PC1 = "one expert spikes at the last layer" story replicate?

Runs the same 2-component fit on all three 64-draw captures (two tasks, three
init states) and reports, per run, which single (layer, expert) cell PC1 is
riding and how much of PC1 that one cell reproduces.  Also aligns PC1 to the
grasp -- detected from the proprio finger dims, not from the commanded gripper --
and reports what physically separates success from failure at the control steps
where the outcome split opens and all 64 episodes are still alive.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

RUNS = [("within64-s24", "s24 / goal-t0"),
        ("within64-t1s19", "t1s19 / bowl-on-stove"),
        ("within64-t3s0", "t3s0 / drawer+bowl")]


def r2_from(z: np.ndarray, c: np.ndarray) -> float:
    d = np.column_stack([np.ones(len(z)), c - c.mean()])
    beta, *_ = np.linalg.lstsq(d, z, rcond=None)
    return float(1 - (z - d @ beta).var() / z.var())


def fit(name: str) -> dict:
    run = load_run("runs/" + name, "runs/" + name + "-client")
    client = pathlib.Path("runs/" + name + "-client")
    feats, t, y, ent, state = [], [], [], [], []
    for e in run.episodes:
        f = action_token_probs(e)
        feats.append(f.reshape(e.n_control, -1))
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
        ent.append(e.entropy[:, :, :, 1:].mean(axis=(1, 2, 3)))
        with np.load(client / ("episode_%02d.npz" % e.index)) as zf:
            state.append(zf["state"])
    x = np.concatenate(feats).astype(np.float64)
    t = np.array(t, float)
    y = np.array(y)
    ent = np.concatenate(ent)
    state = np.concatenate(state)
    xc = x - x.mean(0)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    z = xc @ vt[:2].T

    cell_var = xc.var(0).reshape(len(HB_LAYERS), 10, N_EXPERTS).sum(1)   # [8, 32]
    res = {"run": name, "explained": var[:2].round(3).tolist(),
           "n_rows": int(len(x)), "n_success": int(run.n_success)}
    for c in range(2):
        w = np.abs(vt[c]).reshape(len(HB_LAYERS), 10, N_EXPERTS).sum(1)
        li, ei = np.unravel_index(np.argmax(w), w.shape)
        cell = x.reshape(len(x), len(HB_LAYERS), 10, N_EXPERTS)[:, li, :, ei].mean(1)
        res["pc%d" % (c + 1)] = {
            "top_cell": {"layer": HB_LAYERS[li], "expert": int(ei)},
            "cell_share_of_feature_variance": round(float(cell_var[li, ei] / cell_var.sum()), 3),
            "r2_of_pc_from_that_cell": round(r2_from(z[:, c], cell), 3),
            "layer_energy_share_last_layer": round(
                float((vt[c].reshape(len(HB_LAYERS), 10, N_EXPERTS)[7] ** 2).sum()), 3),
            "corr_control_step": round(float(np.corrcoef(z[:, c], t)[0, 1]), 3),
            "corr_router_entropy": round(float(np.corrcoef(z[:, c], ent)[0, 1]), 3),
            "r2_from_proprio": round(float(1 - np.linalg.lstsq(
                np.column_stack([np.ones(len(z)), state - state.mean(0)]), z[:, c],
                rcond=None)[1][0] / (z[:, c].var() * len(z))), 3),
        }
    return res, run, z, t, y, state


def main() -> None:
    out = {"runs": []}
    for name, _label in RUNS:
        res, run, z, t, y, state = fit(name)
        out["runs"].append(res)
        if name != "within64-s24":
            continue

        # grasp time from the proprio finger dims (6, 7): separation collapses
        sep = state[:, 6] - state[:, 7]
        closed = sep < 0.5 * (sep.max() + sep.min())
        grasp_step, aligned = [], []
        cursor = 0
        for e in run.episodes:
            sl = slice(cursor, cursor + e.n_control)
            c = closed[sl]
            g = int(np.argmax(c)) if c.any() else -1
            grasp_step.append(g)
            if g > 0:
                for k in range(-6, 7):
                    if 0 <= g + k < e.n_control:
                        aligned.append((k, z[cursor + g + k, 0], e.success))
            cursor += e.n_control
        gs = np.array(grasp_step)
        out["grasp"] = {
            "n_with_grasp": int((gs > 0).sum()),
            "median_grasp_step": float(np.median(gs[gs > 0])) if (gs > 0).any() else None,
            "grasp_step_success": float(np.median(gs[(gs > 0) & np.array(
                [e.success for e in run.episodes])])) if (gs > 0).any() else None,
            "pc1_aligned_to_grasp": [
                {"offset": k,
                 "pc1": round(float(np.mean([v for kk, v, _ in aligned if kk == k])), 4),
                 "n": int(sum(1 for kk, _, _ in aligned if kk == k))}
                for k in range(-6, 7)],
        }

        # what physically differs at the steps where the split opens, all 64 alive
        rows = []
        for s in (12, 14, 16):
            m = t == s
            if m.sum() < 60:
                continue
            a, b = state[m & (y == 1)], state[m & (y == 0)]
            rows.append({
                "step": s, "n_ok": int((y[m] == 1).sum()), "n_bad": int((y[m] == 0).sum()),
                "state_mean_success": a.mean(0).round(3).tolist(),
                "state_mean_failure": b.mean(0).round(3).tolist(),
                "finger_sep_success": round(float((a[:, 6] - a[:, 7]).mean()), 4),
                "finger_sep_failure": round(float((b[:, 6] - b[:, 7]).mean()), 4),
                "pc2_d": round(float((z[m & (y == 1), 1].mean() - z[m & (y == 0), 1].mean())
                                     / (np.sqrt((z[m & (y == 1), 1].var()
                                                 + z[m & (y == 0), 1].var()) / 2) + 1e-12)), 3),
            })
        out["split_physics"] = rows

    pathlib.Path("analysis/within64-s24/pc_meaning3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
