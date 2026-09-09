"""Follow-ups: is PC1 just "layer-15 sharpness", and is PC2 just "the clock"?

Three checks the first pass leaves open:

  * the raw variance budget -- if layer 15 / late denoising steps already hold most
    of the feature variance, a component pointing there is a property of the data,
    not a discovery about that component
  * how much of each coordinate a single scalar reproduces (layer-15 entropy for
    PC1, control step for PC2), by R^2
  * what the 8 proprio dims are, so the state correlations can be named
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

SERVER, CLIENT = "runs/within64-s24", "runs/within64-s24-client"


def r2(z: np.ndarray, *cols: np.ndarray) -> float:
    d = np.column_stack([np.ones(len(z))] + [c - c.mean() for c in cols])
    beta, *_ = np.linalg.lstsq(d, z, rcond=None)
    resid = z - d @ beta
    return float(1 - resid.var() / z.var())


def main() -> None:
    run = load_run(SERVER, CLIENT)
    client = pathlib.Path(CLIENT)
    feats, t, y, ent_l, top1_l, state, act = [], [], [], [], [], [], []
    for e in run.episodes:
        f = action_token_probs(e)                              # [T, 8, 10, 32]
        feats.append(f.reshape(e.n_control, -1))
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
        ent_l.append(e.entropy[:, :, :, 1:].mean(axis=(2, 3)))  # [T, 8] per layer
        top1_l.append(f.max(-1).mean(axis=2))                   # [T, 8] per layer
        with np.load(client / ("episode_%02d.npz" % e.index)) as z:
            state.append(z["state"])
            act.append(z["actions"])
    x = np.concatenate(feats).astype(np.float64)
    t = np.array(t, float)
    y = np.array(y)
    ent_l = np.concatenate(ent_l)
    top1_l = np.concatenate(top1_l)
    state = np.concatenate(state)
    act = np.concatenate(act)

    xc = x - x.mean(0)
    _u, s_val, vt = np.linalg.svd(xc, full_matrices=False)
    z = xc @ vt[:3].T

    v = xc.var(0).reshape(len(HB_LAYERS), 10, N_EXPERTS)
    out = {
        "raw_variance_share_per_layer": dict(
            zip(map(str, HB_LAYERS), (v.sum(axis=(1, 2)) / v.sum()).round(4).tolist())),
        "raw_variance_share_per_denoise": (v.sum(axis=(0, 2)) / v.sum()).round(4).tolist(),
        "mean_prob_layer15_top6": None,
        "layer15_expert8": None,
    }

    m = x.mean(0).reshape(len(HB_LAYERS), 10, N_EXPERTS)[7].mean(0)   # layer 15
    order = np.argsort(-m)[:6]
    out["mean_prob_layer15_top6"] = [
        {"expert": int(i), "mean_prob": round(float(m[i]), 4), "x_uniform": round(float(m[i] * 32), 3)}
        for i in order]
    p8 = x.reshape(len(x), len(HB_LAYERS), 10, N_EXPERTS)[:, 7, :, 8].mean(1)
    out["layer15_expert8"] = {
        "mean_prob": round(float(p8.mean()), 4),
        "x_uniform": round(float(p8.mean() * 32), 3),
        "rank_by_mean": int((m > m[8]).sum() + 1),
        "corr_with_pc1": round(float(np.corrcoef(p8, z[:, 0])[0, 1]), 3),
        "share_of_total_feature_variance": round(
            float(xc.reshape(len(x), len(HB_LAYERS), 10, N_EXPERTS)[:, 7, :, 8].var(0).sum()
                  / v.sum()), 4),
    }

    # how much of each PC one scalar reproduces
    out["pc_r2"] = {
        "pc1_from_layer15_entropy": round(r2(z[:, 0], ent_l[:, 7]), 3),
        "pc1_from_layer15_top1": round(r2(z[:, 0], top1_l[:, 7]), 3),
        "pc1_from_layer15_expert8_prob": round(r2(z[:, 0], p8), 3),
        "pc1_from_all_layer_entropies": round(r2(z[:, 0], *ent_l.T), 3),
        "pc1_from_control_step_basis": round(
            r2(z[:, 0], *[np.asarray(t == s, float) for s in np.unique(t)[:-1]]), 3),
        "pc2_from_control_step_basis": round(
            r2(z[:, 1], *[np.asarray(t == s, float) for s in np.unique(t)[:-1]]), 3),
        "pc2_from_control_step_linear": round(r2(z[:, 1], t), 3),
        "pc2_from_all_layer_entropies": round(r2(z[:, 1], *ent_l.T), 3),
        "pc2_from_proprio_state": round(r2(z[:, 1], *state.T), 3),
        "pc1_from_proprio_state": round(r2(z[:, 0], *state.T), 3),
    }

    # proprio dims: identify by range / pairing
    out["state_dims"] = [
        {"dim": i, "min": round(float(state[:, i].min()), 3),
         "max": round(float(state[:, i].max()), 3),
         "corr_with_dim7": round(float(np.corrcoef(state[:, i], state[:, 7])[0, 1]), 3)}
        for i in range(state.shape[1])]
    out["gripper_cmd_range"] = [round(float(act[:, :, 6].min()), 3),
                                round(float(act[:, :, 6].max()), 3)]

    # PC1 conditioned on gripper command sign (open vs closing)
    g = act[:, :, 6].mean(1)
    out["pc1_by_gripper"] = {
        "open_cmd_mean": round(float(z[g < 0, 0].mean()), 4),
        "close_cmd_mean": round(float(z[g > 0, 0].mean()), 4),
        "n_open": int((g < 0).sum()), "n_close": int((g > 0).sum()),
        "layer15_entropy_open": round(float(ent_l[g < 0, 7].mean()), 4),
        "layer15_entropy_close": round(float(ent_l[g > 0, 7].mean()), 4),
    }
    # entropy of layer 15 over control step, to see the same dip
    out["layer15_entropy_by_step"] = [
        {"step": int(s), "entropy": round(float(ent_l[t == s, 7].mean()), 4),
         "top1": round(float(top1_l[t == s, 7].mean() * 32), 3)}
        for s in np.unique(t)[:24]]

    pathlib.Path("analysis/within64-s24/pc_meaning2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
