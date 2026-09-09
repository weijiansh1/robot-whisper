"""Feasibility pilot: is a dense progress score cheap enough to estimate Q(s,a)?

Binary task success is the highest-variance possible score.  With M common
future noise streams per candidate the estimate Q_hat has SE = sqrt(p(1-p)/M),
which is 0.18 at M=8 -- larger than any plausible difference between candidates.
That, not the design, is why the 16x8 commitment grid could not rank candidates
(column-preserving permutation p = 0.921).

LIBERO's own reward is sparse (bddl_base_domain.reward returns 1.0 only on
success), so a dense score must come from the simulator.  For LIBERO-Goal task 0
("open the middle drawer of the cabinet") the natural progress variable is the
drawer's slide joint, qpos[38] = wooden_cabinet_1_middle_level.

This runs a small N x M grid and reports, for both scores, how large the
between-candidate spread is relative to the within-candidate noise.  A dense
score is worth adopting only if that ratio is materially better.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

DRAWER_QPOS = 39  # wooden_cabinet_1_middle_level.
# get_sim_state() is MjSimState.flatten() = [time, qpos, qvel], so qpos[i] lives
# at index 1+i.  Reading index 38 silently returns the TOP drawer, which never
# moves in this task -- the trace was exactly 0.0 even on successful episodes.


def run_cell(config, client, first_seed: int, future_seed: int) -> dict:
    env, obs, _task, prompt = _load_task(config)
    first_noise = np.random.default_rng(first_seed).standard_normal(
        FLOW_NOISE_SHAPE).astype(np.float32)
    future_rng = np.random.default_rng(future_seed)
    drawer, steps, success, idx = [], 0, False, 0
    try:
        for _ in range(config.settle_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        while steps < config.max_steps and not success:
            po = build_policy_observation(obs, prompt)
            noise = first_noise if idx == 0 else future_rng.standard_normal(
                FLOW_NOISE_SHAPE).astype(np.float32)
            req = dict(po)
            req[FLOW_NOISE_KEY] = noise
            actions = np.asarray(
                validate_action_response(client.infer(req))[ACTION_KEY], np.float32)
            for a in actions[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                obs, _r, _d, _i = env.step(a.tolist())
                steps += 1
                success = bool(env.check_success())
                if success:
                    break
            drawer.append(float(np.asarray(env.get_sim_state())[DRAWER_QPOS]))
            idx += 1
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return {
        "first_seed": first_seed, "future_seed": future_seed,
        "success": success, "action_steps": steps,
        "drawer_final": drawer[-1] if drawer else 0.0,
        "drawer_max": max(drawer) if drawer else 0.0,
        "drawer_trace": drawer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--n-first", type=int, default=6)
    ap.add_argument("--n-future", type=int, default=8)
    ap.add_argument("--first-seed-base", type=int, default=2000)
    ap.add_argument("--future-seed-base", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with PolicyClient(host="127.0.0.1", port=args.port, connect_timeout=600.0,
                      inference_timeout=300.0) as client:
        for i in range(args.n_first):
            for m in range(args.n_future):
                cfg = EpisodeConfig(
                    task_suite="libero_goal", task_id=args.task_id,
                    init_state_id=args.init_state_id, seed=7, host="127.0.0.1",
                    port=args.port, libero_root=args.libero_root,
                    output_root=str(out), settle_steps=10,
                    max_steps=args.max_steps, replan_steps=args.replan_steps,
                    inference_timeout=300.0)
                t0 = time.time()
                r = run_cell(cfg, client, args.first_seed_base + i,
                             args.future_seed_base + m)
                r.update(candidate=i, future=m, wall_s=round(time.time() - t0, 1))
                rows.append(r)
                print("cand %d future %d  success=%-5s steps=%3d drawer_max=%.4f  %.0fs"
                      % (i, m, r["success"], r["action_steps"], r["drawer_max"],
                         r["wall_s"]), flush=True)
                (out / "cells.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
