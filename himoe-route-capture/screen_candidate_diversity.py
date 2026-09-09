"""Where in a task do flow-noise candidates actually differ?

Before any best-of-N work is worth running, the candidate pool has to be
non-degenerate: the N candidates must lead somewhere different.  On the
commitment grid at control step 0 they did not -- 16 candidates moved the
gripper toward the drawer handle by 0.0389-0.0406 (1.4% relative spread), and
the between-candidate variance of terminal success came out *below* the pure
noise floor, so the apparent Oracle@16 gain of +0.33 was entirely selection on
noise.

Screening is cheap because it needs no full episodes: branch from a saved
snapshot, run each candidate's chunk for replan_steps environment steps, and
measure how far apart the resulting states are.  ~16 x 10 env steps per query
instead of 16 x 300.

Two controls are built in:

  noise floor   candidate 0 is run twice; restoring a snapshot is not perfectly
                exact (robosuite's OSC keeps goal/interpolator state outside the
                MuJoCo vector), so the candidate spread is only meaningful
                relative to this repeat spread.
  action spread the dispersion of the emitted action chunks themselves, which
                costs forward passes only and is the cheapest possible screen.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from branch_snapshot import restore_full_state, save_full_state  # noqa: E402

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

DRAWER_QPOS = 39  # snapshot index; get_sim_state() is [time, qpos, qvel]


def mean_pairwise(x: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between rows."""
    if len(x) < 2:
        return 0.0
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    iu = np.triu_indices(len(x), k=1)
    return float(d[iu].mean())


def probe_state(env, client, prompt, cfg, seeds, repeat_seed) -> dict:
    """Branch the current state once per seed; return dispersion of the outcomes."""
    snap = save_full_state(env)
    obs = restore_full_state(env, snap)
    po = build_policy_observation(obs, prompt)

    chunks, eefs, sims, drawers = [], [], [], []
    for s in list(seeds) + [repeat_seed]:
        obs = restore_full_state(env, snap)
        po = build_policy_observation(obs, prompt)
        req = dict(po)
        req[FLOW_NOISE_KEY] = np.random.default_rng(s).standard_normal(
            FLOW_NOISE_SHAPE).astype(np.float32)
        actions = np.asarray(
            validate_action_response(client.infer(req))[ACTION_KEY], np.float32)
        chunks.append(actions[: cfg.replan_steps].ravel())
        for a in actions[: cfg.replan_steps]:
            obs, _r, _d, _i = env.step(a.tolist())
        st = np.asarray(env.get_sim_state(), dtype=np.float64)
        eefs.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
        sims.append(st.copy())
        drawers.append(float(st[DRAWER_QPOS]))

    chunks, eefs, sims = np.array(chunks), np.array(eefs), np.array(sims)
    n = len(seeds)
    # last entry repeats seeds[0]: the gap between them is the restore noise floor
    floor_eef = float(np.linalg.norm(eefs[0] - eefs[-1]))
    floor_sim = float(np.linalg.norm(sims[0] - sims[-1]))
    return {
        "n_candidates": n,
        "action_chunk_spread": mean_pairwise(chunks[:n]),
        "eef_spread": mean_pairwise(eefs[:n]),
        "sim_spread": mean_pairwise(sims[:n]),
        "eef_noise_floor": floor_eef,
        "sim_noise_floor": floor_sim,
        "eef_snr": floor_eef and mean_pairwise(eefs[:n]) / floor_eef,
        "drawer_mean": float(np.mean(drawers[:n])),
        "drawer_spread": float(np.std(drawers[:n])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--phases", default="0,2,4,6,8,10,12")
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--prefix-seed", type=int, default=1000)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    phases = sorted(int(p) for p in args.phases.split(","))
    cfg = EpisodeConfig(
        task_suite="libero_goal", task_id=args.task_id,
        init_state_id=args.init_state_id, seed=7, host="127.0.0.1",
        port=args.port, libero_root=args.libero_root, output_root=args.out,
        settle_steps=10, max_steps=1000, replan_steps=args.replan_steps,
        inference_timeout=300.0)
    seeds = [args.seed_base + i for i in range(args.n_candidates)]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    env, obs, _task, prompt = _load_task(cfg)
    prefix_rng = np.random.default_rng(args.prefix_seed)
    try:
        with PolicyClient(host="127.0.0.1", port=args.port, connect_timeout=600.0,
                          inference_timeout=300.0) as client:
            for _ in range(cfg.settle_steps):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
            print("task %d init %d | %s" % (args.task_id, args.init_state_id, prompt),
                  flush=True)
            print("%-6s %-13s %-11s %-11s %-9s %s"
                  % ("阶段", "动作块离散", "落点离散", "噪声底", "信噪比", "抽屉"), flush=True)
            for k in range(max(phases) + 1):
                if k in phases:
                    r = probe_state(env, client, prompt, cfg, seeds,
                                    repeat_seed=seeds[0])
                    r.update(phase=k, task_id=args.task_id,
                             init_state_id=args.init_state_id)
                    rows.append(r)
                    print("%-6d %-13.5f %-11.5f %-11.5f %-9.1f %.5f"
                          % (k, r["action_chunk_spread"], r["eef_spread"],
                             r["eef_noise_floor"], r["eef_snr"] or 0.0,
                             r["drawer_mean"]), flush=True)
                    (out / "diversity.json").write_text(json.dumps(rows, indent=2))
                # advance one control step on the prefix stream
                po = build_policy_observation(obs, prompt)
                req = dict(po)
                req[FLOW_NOISE_KEY] = prefix_rng.standard_normal(
                    FLOW_NOISE_SHAPE).astype(np.float32)
                actions = np.asarray(
                    validate_action_response(client.infer(req))[ACTION_KEY], np.float32)
                for a in actions[: cfg.replan_steps]:
                    obs, _r, _d, _i = env.step(a.tolist())
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
