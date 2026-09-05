#!/usr/bin/env python
"""Phase-1 trunk harvester for E1/E2/E5.

Runs closed-loop trunks (no branching) on chosen (task, init_state, stream)
cells and, at EVERY policy query, dumps a restorable snapshot
(full_state.json/.npz + policy_input.npz) plus dense per-action physical
streams.  Onset labelling happens OFFLINE with the canonical rule, so no
online decisions are made here; q-relative snapshot selection is a later,
free choice.

Runs in the LIBERO env (py3.8) against a route-recorder server (legacy mode,
--store-full-probs).  Routing rows are keyed episode_id = trunk_uid*100 + q.

Deterministic noise: flow_noise(seed_words(master, "harvest", task, init,
stream, q)) -- namespace disjoint from every prior campaign.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
)

EPISODE_ID_KEY = "episode_id"   # serve_with_recorder row-alignment contract
from bestofn_protocol import flow_noise, seed_words
from branch_snapshot import save_full_state
from rolling_star_collect import (
    _array_sha256,
    _atomic_json,
    _save_snapshot_state,
)
from rollout_with_routes import sim_joint_layout


def run_trunk(client, environment, observation, prompt, cfg, out_dir,
              trunk_uid, master, task_id, init_state, stream,
              max_steps, replan_steps):
    """One closed-loop trunk; returns summary dict."""
    dense = {"sim_state": [], "eef": [], "gripper": [], "action": [],
             "query_index": [], "success": []}
    per_query = {"success_after_query": [], "noise_sha": [], "state": []}
    action_steps = 0
    query = 0
    success = False
    t0 = time.time()

    while action_steps < max_steps and not success:
        qdir = out_dir / ("query_%03d" % query)
        qdir.mkdir(parents=True, exist_ok=True)
        snapshot = save_full_state(environment)
        policy_observation = build_policy_observation(observation, prompt)
        hashes = _save_snapshot_state(qdir, snapshot, policy_observation)
        _atomic_json(qdir / "meta.json", {
            "trunk_uid": trunk_uid, "query": query, "prompt": prompt,
            "action_steps_before": action_steps, **hashes,
        })

        noise = flow_noise(seed_words(master, "harvest", task_id, init_state,
                                      stream, (query,)))
        request = dict(policy_observation)
        request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, np.float32)
        request[EPISODE_ID_KEY] = int(trunk_uid * 100 + query)
        response = client.infer(request)
        if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
            raise RuntimeError(f"noise ack mismatch trunk={trunk_uid} q={query}")
        actions = np.asarray(response["actions"], np.float32)
        per_query["noise_sha"].append(_array_sha256(noise))
        per_query["state"].append(np.asarray(policy_observation["observation/state"],
                                             np.float32))

        take = min(replan_steps, len(actions), max_steps - action_steps)
        for k in range(take):
            observation, _r, _d, _i = environment.step(actions[k].tolist())
            success = bool(environment.check_success())
            dense["sim_state"].append(np.asarray(environment.get_sim_state(),
                                                 np.float64))
            dense["eef"].append(np.asarray(
                observation.get("robot0_eef_pos", np.full(3, np.nan)),
                np.float64))
            dense["gripper"].append(np.asarray(
                observation.get("robot0_gripper_qpos", np.full(2, np.nan)),
                np.float64))
            dense["action"].append(actions[k])
            dense["query_index"].append(query)
            dense["success"].append(success)
            action_steps += 1
            if success:
                break
        per_query["success_after_query"].append(success)
        query += 1

    arrays = {
        "control_sim_state": np.stack(dense["sim_state"]),
        "control_eef_position": np.stack(dense["eef"]),
        "control_gripper_qpos": np.stack(dense["gripper"]),
        "control_action": np.stack(dense["action"]),
        "control_query_index": np.asarray(dense["query_index"], np.int32),
        "control_success": np.asarray(dense["success"], bool),
        "query_state": np.stack(per_query["state"]),
        "query_success_after": np.asarray(per_query["success_after_query"], bool),
    }
    np.savez_compressed(out_dir / "dense.npz", **arrays)
    summary = {
        "trunk_uid": trunk_uid, "task_id": task_id, "init_state": init_state,
        "stream": stream, "success": success, "queries": query,
        "action_steps": action_steps, "wall_s": round(time.time() - t0, 1),
        "noise_sha_per_query": per_query["noise_sha"],
        "episode_id_base": trunk_uid * 100,
    }
    _atomic_json(out_dir / "summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_10")
    ap.add_argument("--task-id", type=int, default=8)
    ap.add_argument("--cells", required=True,
                    help="comma list init:stream, e.g. 10:0,10:1,15:0")
    ap.add_argument("--master-seed", type=int, default=20260904)
    ap.add_argument("--environment-seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--trunk-uid-base", type=int, required=True,
                    help="unique per worker; episode ids = uid*100+q")
    ap.add_argument("--libero-root",
                    default="/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for tok in args.cells.split(","):
        i, s = tok.split(":")
        cells.append((int(i), int(s)))

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=args.inference_timeout) as client:
        if not bool(client.metadata.get("store_full_probs", True)):
            raise RuntimeError("server must run --store-full-probs")
        _atomic_json(out / "server_metadata.json", dict(client.metadata))

        for k, (init_state, stream) in enumerate(cells):
            uid = args.trunk_uid_base + k
            tdir = out / f"trunk_{uid:05d}_i{init_state:02d}_s{stream}"
            if (tdir / "summary.json").exists():
                print(f"[skip] {tdir.name} complete", flush=True)
                continue
            cfg = EpisodeConfig(
                task_suite=args.benchmark, task_id=args.task_id,
                init_state_id=init_state, seed=args.environment_seed,
                host=args.host, port=args.port, libero_root=args.libero_root,
                output_root=str(tdir), settle_steps=args.settle_steps,
                max_steps=args.max_steps, replan_steps=args.replan_steps,
                inference_timeout=args.inference_timeout,
            )
            environment, observation, task, prompt = _load_task(cfg)
            for _ in range(args.settle_steps):
                observation, _r, _d, _i = environment.step(
                    LIBERO_DUMMY_ACTION.tolist())
            if not (out / "sim_layout.json").exists():
                _atomic_json(out / "sim_layout.json",
                             sim_joint_layout(environment))
            s = run_trunk(client, environment, observation, prompt, cfg, tdir,
                          uid, args.master_seed, args.task_id, init_state,
                          stream, args.max_steps, args.replan_steps)
            print(f"[{time.strftime('%H:%M:%S')}] trunk {uid} "
                  f"i{init_state} s{stream}: success={s['success']} "
                  f"q={s['queries']} steps={s['action_steps']} "
                  f"{s['wall_s']}s", flush=True)
            try:
                environment.close()
            except Exception:
                pass
    print("harvest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
