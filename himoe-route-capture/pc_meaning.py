"""What do PC1 / PC2 of the routing PCA actually encode?

The projection in within64_analyze.pca_trajectories / within64_export is fitted on
[N control states, 8 layers x 10 denoise steps x 32 experts] router probabilities.
A component is therefore a *contrast over experts*, modulated across layers and
denoising steps.  This script asks three things about that contrast:

  1. where it lives   -- energy per layer / per denoise step, and whether the
                         [80, 32] loading matrix is rank-1 (one expert pattern
                         re-used everywhere) or genuinely multi-pattern
  2. what it tracks    -- correlation of the coordinate with control step, router
                         sharpness, the commanded action, the proprio state, and
                         the eventual outcome
  3. how it splits     -- how much of each component's variance is within-episode
                         (a clock) vs between-episode (an episode identity)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

SERVER = "runs/within64-s24"
CLIENT = "runs/within64-s24-client"
N_COMP = 4


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def main() -> None:
    run = load_run(SERVER, CLIENT)
    client = pathlib.Path(CLIENT)

    feats, ep_idx, step_idx, succ, n_ctrl = [], [], [], [], []
    ent, top1, act_g, act_tr, state = [], [], [], [], []
    for e in run.episodes:
        f = action_token_probs(e)                       # [T, 8, 10, 32]
        feats.append(f.reshape(e.n_control, -1))
        ep_idx += [e.index] * e.n_control
        step_idx += list(range(e.n_control))
        succ += [int(e.success)] * e.n_control
        n_ctrl += [e.n_control] * e.n_control
        # router sharpness of the same state (action tokens only, all layers)
        ent.append(e.entropy[:, :, :, 1:].mean(axis=(1, 2, 3)))
        top1.append(f.max(-1).mean(axis=(1, 2)))
        with np.load(client / ("episode_%02d.npz" % e.index)) as z:
            a = z["actions"]                            # [T, replan, 7]
            s = z["state"]                              # [T, 8]
        act_g.append(a[:, :, 6].mean(1))                # gripper command
        act_tr.append(np.linalg.norm(a[:, :, :3], axis=2).mean(1))
        state.append(s)

    x = np.concatenate(feats).astype(np.float64)        # [N, 2560]
    ep_idx = np.array(ep_idx)
    t = np.array(step_idx, dtype=float)
    y = np.array(succ)
    n_ctrl = np.array(n_ctrl, dtype=float)
    ent = np.concatenate(ent)
    top1 = np.concatenate(top1)
    act_g = np.concatenate(act_g)
    act_tr = np.concatenate(act_tr)
    state = np.concatenate(state)

    mu = x.mean(0)
    xc = x - mu
    _u, s_val, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s_val ** 2) / (s_val ** 2).sum()
    z = xc @ vt[:N_COMP].T                              # [N, N_COMP]

    out: dict = {"n_rows": int(len(x)), "n_episodes": len(run.episodes),
                 "explained_variance": [float(v) for v in var[:N_COMP]]}

    # ---- 1. where the contrast lives ------------------------------------
    comps = []
    for c in range(N_COMP):
        w = vt[c].reshape(len(HB_LAYERS), 10, N_EXPERTS)
        per_layer = (w ** 2).sum(axis=(1, 2))
        per_denoise = (w ** 2).sum(axis=(0, 2))
        mat = w.reshape(-1, N_EXPERTS)                  # [80 sites, 32 experts]
        _u2, s2, vt2 = np.linalg.svd(mat, full_matrices=False)
        site_gain = _u2[:, 0] * s2[0]                   # rank-1 modulation
        expert_pat = vt2[0]
        order = np.argsort(-np.abs(expert_pat))[:6]
        comps.append({
            "component": c + 1,
            "explained_variance": float(var[c]),
            "energy_per_layer": dict(zip(map(str, HB_LAYERS),
                                         (per_layer / per_layer.sum()).round(4).tolist())),
            "energy_early_block_L2_5": float(per_layer[:4].sum() / per_layer.sum()),
            "energy_per_denoise_step": (per_denoise / per_denoise.sum()).round(4).tolist(),
            "simplex_sum_abs": float(np.abs(w.sum(-1)).max()),
            "rank1_share_of_loading": float(s2[0] ** 2 / (s2 ** 2).sum()),
            "rank1_site_gain_sign_flip": {
                "per_layer_mean": site_gain.reshape(len(HB_LAYERS), 10).mean(1).round(3).tolist(),
                "per_denoise_mean": site_gain.reshape(len(HB_LAYERS), 10).mean(0).round(3).tolist(),
            },
            "top_experts": [{"expert": int(i), "weight": round(float(expert_pat[i]), 3)}
                            for i in order],
        })

    # cosine between components and the mean temporal drift direction
    early = xc[t <= 2].mean(0)
    late = xc[t >= 15].mean(0)
    drift = late - early
    drift /= np.linalg.norm(drift)
    for c in range(N_COMP):
        comps[c]["cos_with_time_drift"] = round(float(vt[c] @ drift), 3)

    # ---- 2. what each coordinate tracks ---------------------------------
    frac = t / np.maximum(n_ctrl - 1, 1)
    covars = {
        "control_step": t,
        "progress_frac": frac,
        "steps_remaining": n_ctrl - 1 - t,
        "router_entropy": ent,
        "top1_prob": top1,
        "gripper_cmd": act_g,
        "translation_norm": act_tr,
        "success": y.astype(float),
    }
    for i in range(state.shape[1]):
        covars["state_dim_%d" % i] = state[:, i]
    for c in range(N_COMP):
        comps[c]["correlations"] = {k: round(corr(z[:, c], v), 3) for k, v in covars.items()}

    # ---- 3. clock vs identity -------------------------------------------
    for c in range(N_COMP):
        zc = z[:, c]
        tot = zc.var()
        step_mean = np.array([zc[t == s].mean() for s in np.unique(t)])
        step_of = np.searchsorted(np.unique(t), t)
        between_t = step_mean[step_of].var()
        ep_mean = {e: zc[ep_idx == e].mean() for e in np.unique(ep_idx)}
        between_ep = np.array([ep_mean[e] for e in ep_idx]).var()
        # outcome separation among rows where both classes are still alive
        alive = t <= min(e.n_control for e in run.episodes) - 1
        za, ya = zc[alive], y[alive]
        pooled = np.sqrt((za[ya == 1].var() + za[ya == 0].var()) / 2)
        comps[c]["variance_split"] = {
            "explained_by_control_step": round(float(between_t / tot), 3),
            "explained_by_episode_identity": round(float(between_ep / tot), 3),
            "success_minus_failure_d": round(
                float((za[ya == 1].mean() - za[ya == 0].mean()) / (pooled + 1e-12)), 3),
        }

    # per-step success/failure gap on PC1/PC2, to see when the split opens
    gaps = []
    for s in range(int(t.max()) + 1):
        m = t == s
        if (y[m] == 1).sum() < 5 or (y[m] == 0).sum() < 5:
            continue
        row = {"step": s, "n_alive": int(m.sum()), "n_ok": int((y[m] == 1).sum())}
        for c in range(2):
            a, b = z[m & (y == 1), c], z[m & (y == 0), c]
            pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
            row["pc%d_d" % (c + 1)] = round(float((a.mean() - b.mean()) / pooled), 3)
        gaps.append(row)

    # mean PC trajectory over time (the "clock" shape)
    traj = [{"step": int(s),
             "pc1": round(float(z[t == s, 0].mean()), 4),
             "pc2": round(float(z[t == s, 1].mean()), 4),
             "n": int((t == s).sum())}
            for s in np.unique(t)]

    out["components"] = comps
    out["outcome_gap_per_step"] = gaps
    out["mean_trajectory"] = traj
    pathlib.Path("analysis/within64-s24/pc_meaning.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:12000])


if __name__ == "__main__":
    main()
