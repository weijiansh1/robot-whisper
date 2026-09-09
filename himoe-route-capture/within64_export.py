"""Pack the capture into a single self-contained HTML viewer payload.

Size budget drives the layout.  The full tensor is
[64 episodes, ~30 control steps, 8 layers, 10 denoising steps, 11 tokens, 32
experts] = ~54 M values, which is not going into a web page.  Two tiers instead:

  agg     every episode, averaged over denoising steps and action tokens
          -> [E, Tmax, 8, 32], drives the ensemble and per-episode heatmaps
  detail  a handful of showcase episodes at full denoising-step resolution
          -> [D, Tmax, 8, 10, 32], drives the distribution explorer

Values are stored as the residual against each (layer, expert) global mean and
quantised to uint16 rather than uint8: the whole point of these panels is a
deviation of a few tenths of a percent off 1/32, and uint8 over the observed
range would put the quantisation step within an order of magnitude of the signal.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import pathlib

import numpy as np

from within64_lib import (
    HB_LAYERS,
    N_EXPERTS,
    Run,
    action_token_probs,
    load_run,
    state_token_probs,
)

N_SHOWCASE_PER_CLASS = 4


def _quantise(arr: np.ndarray) -> dict:
    """uint16 with an affine map, plus the gzip+base64 blob."""
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo if hi > lo else 1.0
    q = np.clip(np.round((arr - lo) / span * 65535.0), 0, 65535).astype(np.uint16)
    raw = gzip.compress(q.tobytes(order="C"), 6)
    return {
        "shape": list(arr.shape),
        "dtype": "uint16",
        "offset": lo,
        "scale": span / 65535.0,
        "b64": base64.b64encode(raw).decode("ascii"),
        "bytes": len(raw),
    }


def _quantise_u8(arr: np.ndarray) -> dict:
    raw = gzip.compress(arr.astype(np.uint8).tobytes(order="C"), 6)
    return {
        "shape": list(arr.shape),
        "dtype": "uint8",
        "b64": base64.b64encode(raw).decode("ascii"),
        "bytes": len(raw),
    }


def _pad(arrs: list[np.ndarray], t_max: int) -> np.ndarray:
    out = np.zeros((len(arrs), t_max) + arrs[0].shape[1:], dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = a
        if a.shape[0] < t_max:  # hold the last frame so the viewer never shows a hole
            out[i, a.shape[0]:] = a[-1]
    return out


def build_payload(run: Run) -> dict:
    t_max = max(e.n_control for e in run.episodes)
    n_ep = len(run.episodes)
    success = np.array([e.success for e in run.episodes])

    act = [action_token_probs(e) for e in run.episodes]  # [T, 8, 10, 32]
    agg_raw = _pad([a.mean(axis=2) for a in act], t_max)  # [E, T, 8, 32]
    expert_mean = agg_raw.reshape(-1, len(HB_LAYERS), N_EXPERTS).mean(0)  # [8, 32]
    agg = agg_raw - expert_mean

    # showcase episodes: shortest successes and longest failures are the clearest
    ok_idx = [i for i in range(n_ep) if success[i]]
    bad_idx = [i for i in range(n_ep) if not success[i]]
    ok_idx.sort(key=lambda i: run.episodes[i].n_control)
    bad_idx.sort(key=lambda i: -run.episodes[i].n_control)
    show = sorted(ok_idx[:N_SHOWCASE_PER_CLASS] + bad_idx[:N_SHOWCASE_PER_CLASS])

    detail = _pad([act[i] for i in show], t_max) - expert_mean[None, None, :, None, :]
    state_det = _pad([state_token_probs(run.episodes[i]) for i in show], t_max) \
        - expert_mean[None, None, :, None, :]
    ids = _pad([run.episodes[i].ids.astype(np.float32) for i in show], t_max)

    # PCA over every (episode, control step) routing state
    flat = np.concatenate([a.reshape(a.shape[0], -1) for a in act])
    mu = flat.mean(0)
    _u, s, vt = np.linalg.svd(flat - mu, full_matrices=False)
    explained = ((s**2) / (s**2).sum())[:2]
    coords = [((a.reshape(a.shape[0], -1) - mu) @ vt[:2].T).tolist() for a in act]

    # population-level curves, with the constant-population cut-off marked
    feats = _pad([a.reshape(a.shape[0], -1) for a in act], t_max)
    alive = np.zeros((n_ep, t_max), dtype=bool)
    for i, e in enumerate(run.episodes):
        alive[i, : e.n_control] = True
    spread, sep = [], []
    for t in range(t_max):
        rows = feats[:, t][alive[:, t]]
        if len(rows) < 4:
            spread.append(None)
            sep.append(None)
            continue
        centred = rows - rows.mean(0)
        within = float(np.linalg.norm(centred, axis=1).mean())
        spread.append(within)
        lab = success[alive[:, t]]
        if lab.sum() < 5 or (~lab).sum() < 5:
            sep.append(None)
            continue
        gap = rows[lab].mean(0) - rows[~lab].mean(0)
        sep.append(float(np.linalg.norm(gap) / (within + 1e-12)))

    return {
        "meta": {
            "label": run.label,
            "layout": run.layout,
            "checkpoint_sha256": run.checkpoint_sha256[:16],
            "task": run.summaries[0].get("task_name", "?"),
            "prompt": run.summaries[0].get("prompt", "?"),
            "init_state_id": run.episodes[0].init_state_id,
            "n_episodes": n_ep,
            "n_success": int(success.sum()),
            "t_max": t_max,
            "t_all_alive": int(alive.all(axis=0).sum()),
            "uniform": 1.0 / N_EXPERTS,
        },
        "layers": HB_LAYERS,
        "episodes": [
            {
                "i": e.index,
                "seed": e.flow_noise_seed,
                "ok": bool(e.success),
                "steps": e.action_steps,
                "n": e.n_control,
            }
            for e in run.episodes
        ],
        "showcase": show,
        "expert_mean": expert_mean.tolist(),
        "agg": _quantise(agg),
        "detail": _quantise(detail),
        "state_detail": _quantise(state_det),
        "ids": _quantise_u8(ids),
        "pca": {"explained": explained.tolist(), "coords": coords},
        "curves": {
            "spread": spread,
            "sep": sep,
            "alive": alive.sum(0).tolist(),
            "alive_ok": (alive & success[:, None]).sum(0).tolist(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True, help="path to write data.js")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    run = load_run(args.server_dir, args.client_dir, allow_partial=args.allow_partial)
    payload = build_payload(run)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.HIMOE_DATA = " + json.dumps(payload) + ";\n")

    total = out.stat().st_size
    print("episodes=%d success=%d  t_max=%d" % (
        payload["meta"]["n_episodes"], payload["meta"]["n_success"],
        payload["meta"]["t_max"]))
    for k in ("agg", "detail", "state_detail", "ids"):
        print("  %-13s %-26s %6.2f MB gzipped" % (
            k, payload[k]["shape"], payload[k]["bytes"] / 1e6))
    print("wrote %s (%.2f MB)" % (out, total / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
