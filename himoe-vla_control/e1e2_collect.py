#!/usr/bin/env python
"""E1/E2 collector: replay saved snapshots through a route-recorder server with
controlled flow noise (legacy capture mode; zero server modifications).

E1-A  per snapshot: 1 base request + n_dirs x n_scales perturbed requests,
      xi' = xi + scale * ||xi||_2 * u   (u = unit gaussian direction)
E2    per snapshot: n_seeds independent noise draws.

Run inside the model env (py3.11) with PYTHONPATH to the bridge; no LIBERO
environment is needed.  Alignment contract: one inference = one routes.zarr
row keyed by a unique episode_id; this script owns the id space and writes a
client-side manifest.jsonl mapping ids to conditions with SHA-256 receipts.

Plan file (JSON list): [{"snapshot_dir": ..., "trunk_uid": 30012,
                         "q_offset": -2, "arm": "loop"}, ...]
(seed_words requires ints: trunk_uid and q_offset feed the namespace.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.protocol import (
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
)

EPISODE_ID_KEY = "episode_id"   # serve_with_recorder row-alignment contract
from bestofn_protocol import flow_noise, seed_words


def array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


def load_snapshot_observation(snap_dir: pathlib.Path) -> dict:
    z = np.load(snap_dir / "policy_input.npz")
    # prompt lives beside the snapshot (harvester meta.json / rolling-star
    # manifest.json) or one level up (snapshot_fork_recovery keeps a single
    # run-level manifest and per-fork subdirs).
    prompt = None
    for d in (snap_dir, snap_dir.parent):
        for name in ("meta.json", "manifest.json", "trunk.json",
                     "experiment_config.json"):
            f = d / name
            if f.exists():
                blob = json.loads(f.read_text())
                task = blob.get("task")
                prompt = (blob.get("prompt")
                          or (task.get("prompt") if isinstance(task, dict) else None))
                if prompt:
                    break
        if prompt:
            break
    if not prompt:
        raise FileNotFoundError(f"no prompt for {snap_dir}")
    return {
        "observation/image": z["image"],
        "observation/wrist_image": z["wrist_image"],
        "observation/state": z["state"],
        "prompt": prompt,
    }


def unit_direction(master: int, trunk_uid: int, q_offset: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed_words(master, "e1/dir", trunk_uid,
                                           q_offset + 100, d))
    g = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    return g / np.linalg.norm(g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--plan", required=True, help="JSON list of snapshot jobs")
    ap.add_argument("--out", required=True, help="client manifest dir")
    ap.add_argument("--mode", choices=["e1", "e2", "both"], default="both")
    ap.add_argument("--master-seed", type=int, default=20260904)
    ap.add_argument("--n-dirs", type=int, default=32)
    ap.add_argument("--scales", default="1e-4,1e-3,1e-2")
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--episode-id-base", type=int, default=1_000_000)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    args = ap.parse_args()

    plan = json.loads(pathlib.Path(args.plan).read_text())
    scales = [float(s) for s in args.scales.split(",")]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"
    if man_path.exists():
        done_ids = {json.loads(l)["episode_id"] for l in open(man_path)}
        print(f"[resume] {len(done_ids)} requests already recorded")
    else:
        done_ids = set()
    man = open(man_path, "a")

    (out / "config.json").write_text(json.dumps({
        "mode": args.mode, "master_seed": args.master_seed,
        "n_dirs": args.n_dirs, "scales": scales, "n_seeds": args.n_seeds,
        "episode_id_base": args.episode_id_base, "plan": args.plan,
        "port": args.port, "started": time.strftime("%FT%TZ", time.gmtime()),
    }, indent=1))

    eid = args.episode_id_base
    n_sent = n_skip = 0

    def send(client, obs, noise, record):
        nonlocal n_sent, n_skip
        if record["episode_id"] in done_ids:
            n_skip += 1
            return
        noise = np.ascontiguousarray(noise, np.float32)
        assert np.isfinite(noise).all()
        req = dict(obs)
        req[FLOW_NOISE_KEY] = noise
        req[EPISODE_ID_KEY] = int(record["episode_id"])
        resp = client.infer(req)
        got = resp.get(FLOW_NOISE_SHA256_KEY)
        want = array_sha(noise)
        if got != want:
            raise RuntimeError(f"noise ack mismatch eid={record['episode_id']}")
        record["noise_sha256"] = want
        man.write(json.dumps(record) + "\n")
        man.flush()
        n_sent += 1

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=args.inference_timeout) as client:
        if not bool(client.metadata.get("store_full_probs", True)):
            raise RuntimeError("server must run with --store-full-probs")
        for job in plan:
            snap = pathlib.Path(job["snapshot_dir"])
            obs = load_snapshot_observation(snap)
            trunk_uid = int(job["trunk_uid"])
            q_offset = int(job["q_offset"])
            arm = job.get("arm", "?")
            base = flow_noise(seed_words(args.master_seed, "e1/base",
                                         trunk_uid, q_offset + 100, 0))
            base_norm = float(np.linalg.norm(base))
            common = {"snapshot_dir": str(snap), "trunk_uid": trunk_uid,
                      "q_offset": q_offset, "arm": arm}

            if args.mode in ("e1", "both"):
                send(client, obs, base, {**common, "episode_id": eid,
                                         "kind": "e1_base"})
                eid += 1
                for scale in scales:
                    for d in range(args.n_dirs):
                        u = unit_direction(args.master_seed, trunk_uid,
                                           q_offset, d)
                        xi = (base + np.float32(scale * base_norm) * u).astype(np.float32)
                        send(client, obs, xi, {**common, "episode_id": eid,
                                               "kind": "e1_pert", "scale": scale,
                                               "direction": d,
                                               "delta_norm": scale * base_norm})
                        eid += 1

            if args.mode in ("e2", "both"):
                for i in range(args.n_seeds):
                    xi = flow_noise(seed_words(args.master_seed, "e2/seed",
                                               trunk_uid, q_offset + 100, i))
                    send(client, obs, xi, {**common, "episode_id": eid,
                                           "kind": "e2_seed", "seed_index": i})
                    eid += 1
            print(f"[{time.strftime('%H:%M:%S')}] t{trunk_uid}/q{q_offset:+d}/{arm} done "
                  f"(sent so far {n_sent}, skipped {n_skip})", flush=True)

    man.close()
    print(f"collect complete: {n_sent} sent, {n_skip} resumed-skip, "
          f"last eid {eid - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
