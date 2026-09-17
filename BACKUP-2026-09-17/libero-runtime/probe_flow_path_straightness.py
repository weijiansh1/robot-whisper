"""Is the 10-step flow-matching denoising path of one HiMoE-VLA inference a straight line?

Replays saved queries (observation + fixed flow noise) from episode traces with an
independent model instance, records x_t and v_t at each Euler step, checks that the
returned action chunk matches the trace, and measures straightness in the model's
normalized 24-dim action space (real dims selected by the model's data_mask).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA", "/home/swj/data"))
CHECKPOINT = DATA / "himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10"
CHECKPOINT_SHA = "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256"
for p in (DATA / "srv/src", DATA / "srv/packages/openpi-client/src", DATA / "srv"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_policy():
    import torch
    from himoe_libero_bridge.policies import HiMoEPolicy
    t0 = time.monotonic()
    policy = HiMoEPolicy(str(CHECKPOINT), suite="long", upstream_root=str(DATA / "srv"),
                         require_cuda=True, libero_wrist_layout="released-left")
    policy._policy.model.eval().requires_grad_(False)
    if policy.metadata["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("unexpected checkpoint")
    cfg = policy._policy.model.config
    info = {"load_seconds": time.monotonic() - t0, "num_steps": int(cfg.num_steps),
            "n_action_steps": int(cfg.n_action_steps), "max_action_dim": int(cfg.max_action_dim),
            "gpu": torch.cuda.get_device_name(0)}
    return policy, info


class DenoiseRecorder:
    """Wraps model.denoise_step to record (t, x_t, v_t) for every Euler step of one call."""

    def __init__(self, model):
        self.model = model
        self.records = []
        self.data_mask = None

    def __enter__(self):
        orig = self.model.denoise_step  # bound class method
        rec = self

        def wrapped(state, ppm, pam, pkv, data_mask, x_t, timestep):
            import torch
            v_t = orig(state, ppm, pam, pkv, data_mask, x_t, timestep)
            # emulate the caller's in-place Euler update with identical dtypes (v_t may be bf16 under autocast)
            dt = torch.tensor(-1.0 / rec.model.config.num_steps, dtype=torch.float32, device=x_t.device)
            x_next = (x_t + dt * v_t).detach().float().clone()
            rec.records.append((float(timestep.reshape(-1)[0].item()),
                                x_t.detach().float().clone(), v_t.detach().float().clone(), x_next, str(v_t.dtype)))
            rec.data_mask = data_mask.detach().cpu().numpy().reshape(-1)
            return v_t

        self.model.denoise_step = wrapped
        return self

    def __exit__(self, *_):
        del self.model.denoise_step  # restore class attribute lookup


def replay_episode(policy, episode_dir, max_queries=None):
    import torch
    episode_dir = Path(episode_dir)
    summary = json.loads((episode_dir / "summary.json").read_text())
    npz_path = episode_dir / "episode-trace.npz"
    arrays = np.load(npz_path)
    n = int(arrays["predicted_actions"].shape[0])
    if max_queries:
        n = min(n, max_queries)
    model = policy._policy.model
    dt = -1.0 / model.config.num_steps
    X, V, T = [], [], []
    action_diffs, seconds = [], []
    data_mask = None
    for q in range(n):
        request = {
            "observation/image": arrays["images"][q],
            "observation/wrist_image": arrays["wrist_images"][q],
            "observation/state": arrays["states"][q],
            "prompt": summary["prompt"],
            "flow/noise": np.ascontiguousarray(arrays["flow_noises"][q], dtype=np.float32),
        }
        t0 = time.monotonic()
        with DenoiseRecorder(model) as rec:
            response = policy.infer(request)
        torch.cuda.synchronize()
        seconds.append(time.monotonic() - t0)
        if len(rec.records) != model.config.num_steps:
            raise RuntimeError("expected %d denoise steps, got %d" % (model.config.num_steps, len(rec.records)))
        if model.__dict__.get("denoise_step") is not None:
            raise RuntimeError("denoise_step wrapper leaked")
        data_mask = rec.data_mask
        ts = np.array([r[0] for r in rec.records])
        xs = np.stack([r[1][0].cpu().numpy() for r in rec.records])  # (10, 10, 24) pre-update states
        vs = np.stack([r[2][0].cpu().numpy() for r in rec.records])  # (10, 10, 24) model velocity (float32 copy)
        xn = np.stack([r[3][0].cpu().numpy() for r in rec.records])  # (10, 10, 24) post-update states, exact dtypes
        v_dtype = rec.records[0][4]
        # the post-update state of step k must be bitwise the pre-update state of step k+1
        if not np.array_equal(xn[:-1], xs[1:]):
            raise RuntimeError("Euler recurrence mismatch at query %d: max |diff| %g" % (q, np.abs(xn[:-1] - xs[1:]).max()))
        path = np.concatenate([xs, xn[-1][None]], axis=0)  # (11, 10, 24): noise ... final normalized action
        if not np.allclose(xs[0], arrays["flow_noises"][q], atol=0):
            raise RuntimeError("x_0 is not the supplied noise at query %d" % q)
        action_diffs.append(float(np.max(np.abs(response["actions"] - arrays["predicted_actions"][q]))))
        X.append(path); V.append(vs); T.append(ts)
    return {
        "episode": str(episode_dir), "success": bool(summary.get("success")), "queries": n,
        "trace_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "X": np.stack(X), "V": np.stack(V), "T": np.stack(T), "dt": dt,
        "data_mask": data_mask, "action_max_abs_diff": action_diffs, "seconds": seconds, "v_dtype": v_dtype,
    }


def straightness(path, vel, dims):
    """path (11, tokens, D), vel (10, tokens, D). Flatten selected dims over all tokens."""
    P = path[:, :, dims].reshape(path.shape[0], -1)
    steps = P[1:] - P[:-1]
    Vf = steps / (-0.1)  # effective velocity actually integrated (dt = -0.1); vel arg kept for reference
    seg = np.linalg.norm(steps, axis=1)
    L = float(seg.sum())
    chord = P[-1] - P[0]
    D = float(np.linalg.norm(chord))
    u = chord / D
    vnorm = np.linalg.norm(Vf, axis=1)
    # dt < 0, so the path moves along -v; compare -v with the chord direction
    cos_chord = (-Vf @ u) / vnorm
    cos_consec = np.sum(Vf[:-1] * Vf[1:], axis=1) / (vnorm[:-1] * vnorm[1:])
    rel = P - P[0]
    perp = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1) / D
    one_step = float(np.linalg.norm((P[0] - Vf[0]) - P[-1]) / D)  # constant-velocity extrapolation from step 0
    return {
        "path_over_chord": L / D,
        "cos_chord": cos_chord, "cos_consec": cos_consec,
        "angle_first_last_deg": float(np.degrees(np.arccos(np.clip(np.dot(Vf[0], Vf[-1]) / (vnorm[0] * vnorm[-1]), -1, 1)))),
        "max_perp_over_chord": float(perp.max()), "perp_over_chord": perp,
        "vnorm": vnorm, "vnorm_max_over_min": float(vnorm.max() / vnorm.min()),
        "one_step_endpoint_error": one_step, "chord": D, "length": L,
    }


def per_token_ratio(path, dims):
    out = []
    for j in range(path.shape[1]):
        P = path[:, j, dims]
        L = np.linalg.norm(P[1:] - P[:-1], axis=1).sum()
        D = np.linalg.norm(P[-1] - P[0])
        out.append(float(L / D))
    return np.array(out)


def summarize(ep):
    real = np.flatnonzero(ep["data_mask"])
    alld = np.arange(ep["X"].shape[-1])
    res = {}
    for name, dims in (("real_dims", real), ("all_24_dims", alld)):
        rows = [straightness(ep["X"][q], ep["V"][q], dims) for q in range(ep["queries"])]
        def col(key):
            return np.array([r[key] for r in rows])
        stats = {}
        for key in ("path_over_chord", "angle_first_last_deg", "max_perp_over_chord",
                    "vnorm_max_over_min", "one_step_endpoint_error", "chord", "length"):
            v = col(key)
            stats[key] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
        cc = col("cos_consec")
        stats["cos_consec_min_over_queries"] = {"median": float(np.median(cc.min(axis=1))), "min": float(cc.min())}
        stats["cos_consec_by_step_mean"] = cc.mean(axis=0).tolist()
        stats["cos_chord_by_step_mean"] = col("cos_chord").mean(axis=0).tolist()
        stats["vnorm_by_step_mean"] = col("vnorm").mean(axis=0).tolist()
        stats["perp_by_step_mean"] = col("perp_over_chord").mean(axis=0).tolist()
        tok = np.stack([per_token_ratio(ep["X"][q], dims) for q in range(ep["queries"])])
        stats["path_over_chord_by_token_median"] = np.median(tok, axis=0).tolist()
        res[name] = stats
    res["queries"] = ep["queries"]
    res["success"] = ep["success"]
    res["real_dim_indices"] = real.tolist()
    res["action_max_abs_diff_vs_trace"] = {"max": float(max(ep["action_max_abs_diff"])),
                                           "median": float(np.median(ep["action_max_abs_diff"]))}
    res["infer_seconds_median"] = float(np.median(ep["seconds"]))
    res["v_dtype"] = ep["v_dtype"]
    # bf16 rounding floor: relative size of (float32 dt*v) - (bf16 dt*v) versus the step itself
    step_f32 = ep["dt"] * ep["V"]
    step_real = ep["X"][:, 1:] - ep["X"][:, :-1]
    res["rounding_floor_rel"] = float(np.linalg.norm(step_f32 - step_real) / np.linalg.norm(step_real))
    res["timesteps"] = ep["T"][0].tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--tag-names", action="store_true", help="name outputs <parent2>__<parent1> instead of the episode dir name")
    ap.add_argument("--memory-cap-mib", type=int, default=0, help="cap the CUDA caching allocator (0 = no cap)")
    args = ap.parse_args()
    if args.memory_cap_mib:
        import torch
        total = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_cap_mib * 1024 ** 2 / total), 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy, info = load_policy()
    print("loaded:", json.dumps(info), flush=True)
    report = {"model": info, "episodes": {}}
    for ep_dir in args.episode:
        ep = replay_episode(policy, ep_dir, args.max_queries)
        name = (Path(ep_dir).parent.parent.name + "__" + Path(ep_dir).parent.name) if args.tag_names else Path(ep_dir).name
        np.savez_compressed(out / (name + ".npz"), X=ep["X"], V=ep["V"], T=ep["T"], data_mask=ep["data_mask"],
                            action_max_abs_diff=np.array(ep["action_max_abs_diff"]), dt=ep["dt"])
        s = summarize(ep)
        s["trace_sha256"] = ep["trace_sha256"]
        report["episodes"][name] = s
        print("episode", name, "success", ep["success"], "queries", ep["queries"],
              "action max|diff|", max(ep["action_max_abs_diff"]), "v dtype", ep["v_dtype"],
              "rounding floor rel", s["rounding_floor_rel"], flush=True)
        for scope in ("real_dims", "all_24_dims"):
            st = s[scope]
            print("  [%s] L/D median %.4f (min %.4f max %.4f); angle(v0,v9) median %.1f deg (max %.1f); "
                  "max perp/D median %.4f (max %.4f); |v| max/min median %.3f; one-step endpoint err median %.4f" % (
                      scope, st["path_over_chord"]["median"], st["path_over_chord"]["min"], st["path_over_chord"]["max"],
                      st["angle_first_last_deg"]["median"], st["angle_first_last_deg"]["max"],
                      st["max_perp_over_chord"]["median"], st["max_perp_over_chord"]["max"],
                      st["vnorm_max_over_min"]["median"], st["one_step_endpoint_error"]["median"]), flush=True)
            print("    cos(v_k,v_k+1) by step:", np.round(st["cos_consec_by_step_mean"], 4).tolist())
            print("    cos(-v_k,chord) by step:", np.round(st["cos_chord_by_step_mean"], 4).tolist())
            print("    |v_k| by step:", np.round(st["vnorm_by_step_mean"], 3).tolist())
            print("    L/D by action token (median):", np.round(st["path_over_chord_by_token_median"], 4).tolist())
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("saved", out / "report.json")


if __name__ == "__main__":
    main()
