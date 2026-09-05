#!/usr/bin/env python
"""E1 methodology diagnostic: is input-noise perturbation a usable probe?

The pilot showed D_f (routing distance) essentially unchanged when the relative
perturbation grew 100x (1e-4 -> 1e-2).  Two competing explanations:

  (a) numeric floor -- the model computes in bf16 (eps ~ 8e-3), so relative
      input perturbations at/below that are rounded away and what we measured
      was nondeterministic accumulation noise, or
  (b) genuine saturation -- routing is insensitive to the flow initial value.

This script separates them on a live server (--return-full-probs, so routing
comes back inline; nothing needs to be flushed):

  1. REPEAT   the identical xi twice -> the noise floor of the whole path.
  2. SWEEP    relative scales spanning bf16 eps -> D vs ||dxi||; the log-log
              slope says whether a linear response regime exists at all.
  3. REFERENCE an independent seed -> the scale of a "fully different" draw.

Run in the LIBERO env (the model env's websockets cannot handshake this server).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-vla_control")

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.protocol import FLOW_NOISE_KEY, FLOW_NOISE_SHAPE
from bestofn_protocol import flow_noise, seed_words
import control_metrics as cm
from e1e2_collect import load_snapshot_observation

PROBS_KEY = "recorder/hb_router_probs"
PRIMARY_LAYERS = slice(4, 8)      # stored layers 4..7 = model layers 12-15
ACTION_TOKENS = slice(1, 11)      # token 0 is the state token


def psi_of(probs: np.ndarray, subset: str = "primary") -> np.ndarray:
    """[L, D, U, E] inline response -> Psi [F, ...] sqrt-embedded."""
    p = np.asarray(probs, np.float32)
    if subset == "primary":
        p = p[PRIMARY_LAYERS, :, ACTION_TOKENS, :]
    p = np.transpose(p, (1, 0, 2, 3))            # [F, L, U, E]
    return cm.flatten_query(p)


def ask(client, obs, noise):
    req = dict(obs)
    req[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, np.float32)
    resp = client.infer(req)
    if PROBS_KEY not in resp:
        raise RuntimeError("server must run --return-full-probs")
    return np.asarray(resp[PROBS_KEY]), np.asarray(resp["actions"], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--snapshots", required=True,
                    help="comma list label=dir, e.g. preloop=/x/query_035")
    ap.add_argument("--scales", default="1e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0")
    ap.add_argument("--n-dirs", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--master-seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    jobs = []
    for tok in args.snapshots.split(","):
        label, d = tok.split("=", 1)
        jobs.append((label, pathlib.Path(d)))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.master_seed)
    records = []

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=300.0) as client:
        for label, snap in jobs:
            obs = load_snapshot_observation(snap)
            base = flow_noise(seed_words(args.master_seed, "diag/base", 0, 0, 0))
            base_norm = float(np.linalg.norm(base))
            p_base, a_base = ask(client, obs, base)
            psi_base = psi_of(p_base)

            # 1. determinism floor: identical request repeated
            for r in range(args.n_repeats):
                p, a = ask(client, obs, base)
                D = cm.route_distance(psi_base, psi_of(p))
                records.append({"snapshot": label, "kind": "repeat", "rep": r,
                                "scale": 0.0, "delta_norm": 0.0,
                                "D_last": float(D[-1]), "D_max": float(D.max()),
                                "D_mean": float(D.mean()),
                                "act_max_abs_diff": float(np.abs(a - a_base).max()),
                                "D_f": [float(x) for x in D]})

            # 2. scale sweep (same directions across scales -> paired)
            dirs = []
            for d in range(args.n_dirs):
                g = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
                dirs.append(g / np.linalg.norm(g))
            for scale in scales:
                for d, u in enumerate(dirs):
                    dxi = np.float32(scale * base_norm) * u
                    p, a = ask(client, obs, (base + dxi).astype(np.float32))
                    D = cm.route_distance(psi_base, psi_of(p))
                    records.append({"snapshot": label, "kind": "pert",
                                    "rep": d, "scale": scale,
                                    "delta_norm": float(np.linalg.norm(dxi)),
                                    "D_last": float(D[-1]), "D_max": float(D.max()),
                                    "D_mean": float(D.mean()),
                                    "act_max_abs_diff": float(np.abs(a - a_base).max()),
                                    "D_f": [float(x) for x in D]})

            # 3. independent draws for reference scale
            for i in range(args.n_dirs):
                xi = flow_noise(seed_words(args.master_seed, "diag/indep", 0, 0, i))
                p, a = ask(client, obs, xi)
                D = cm.route_distance(psi_base, psi_of(p))
                records.append({"snapshot": label, "kind": "independent",
                                "rep": i, "scale": float("nan"),
                                "delta_norm": float(np.linalg.norm(xi - base)),
                                "D_last": float(D[-1]), "D_max": float(D.max()),
                                "D_mean": float(D.mean()),
                                "act_max_abs_diff": float(np.abs(a - a_base).max()),
                                "D_f": [float(x) for x in D]})
            print(f"[diag] {label}: {len(records)} records so far", flush=True)

    (out / "records.json").write_text(json.dumps(records, indent=1))

    # ---- summary -----------------------------------------------------------
    def med(sel, key="D_last"):
        v = [r[key] for r in records if sel(r)]
        return float(np.median(v)) if v else float("nan")

    print("\n%-10s %-12s %10s %12s %12s %12s"
          % ("snapshot", "kind/scale", "n", "med ||dxi||", "med D_last", "med dAct"))
    for label, _ in jobs:
        f = med(lambda r: r["snapshot"] == label and r["kind"] == "repeat")
        fa = med(lambda r: r["snapshot"] == label and r["kind"] == "repeat",
                 "act_max_abs_diff")
        n = sum(1 for r in records if r["snapshot"] == label and r["kind"] == "repeat")
        print("%-10s %-12s %10d %12s %12.6f %12.2e"
              % (label, "repeat", n, "0", f, fa))
        for s in scales:
            sel = (lambda r, s=s, label=label: r["snapshot"] == label
                   and r["kind"] == "pert" and r["scale"] == s)
            n = sum(1 for r in records if sel(r))
            print("%-10s %-12s %10d %12.4f %12.6f %12.2e"
                  % (label, f"pert {s:g}", n, med(sel, "delta_norm"),
                     med(sel), med(sel, "act_max_abs_diff")))
        sel = (lambda r, label=label: r["snapshot"] == label
               and r["kind"] == "independent")
        n = sum(1 for r in records if sel(r))
        print("%-10s %-12s %10d %12.4f %12.6f %12.2e"
              % (label, "independent", n, med(sel, "delta_norm"), med(sel),
                 med(sel, "act_max_abs_diff")))

        # log-log slope of D_last vs delta_norm over the sweep (linear response
        # would give slope ~1; a floor gives slope ~0)
        xs, ys = [], []
        for s in scales:
            dn = med(lambda r, s=s, label=label: r["snapshot"] == label
                     and r["kind"] == "pert" and r["scale"] == s, "delta_norm")
            dd = med(lambda r, s=s, label=label: r["snapshot"] == label
                     and r["kind"] == "pert" and r["scale"] == s)
            if dn > 0 and dd > 0:
                xs.append(np.log10(dn)); ys.append(np.log10(dd))
        if len(xs) >= 3:
            hi = [i for i, x in enumerate(xs) if x >= np.log10(0.05)]
            slope_all = np.polyfit(xs, ys, 1)[0]
            msg = f"  log-log slope (all scales) = {slope_all:.2f}"
            if len(hi) >= 2:
                slope_hi = np.polyfit([xs[i] for i in hi], [ys[i] for i in hi], 1)[0]
                msg += f" | (||dxi||>=0.05) = {slope_hi:.2f}"
            print(msg)
    print(f"\nwrote {out/'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
