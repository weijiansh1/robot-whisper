"""Can we branch a LIBERO episode from a saved snapshot and get identical futures?

Everything in the same-state candidate experiment rests on this.  If restoring a
snapshot and replaying the same actions does not reproduce the same trajectory,
then any difference measured between candidates is contaminated by restore noise.

``get_sim_state()`` returns ``sim.get_state().flatten()`` -- MuJoCo's time, qpos,
qvel and act.  It does NOT carry robosuite controller state (OSC goal position /
orientation reference / interpolator), so the restore is only guaranteed to be
faithful if the controller re-derives its state from the simulator each step.
This script measures whether that is true rather than assuming it.

Three checks, in increasing strictness:

  A  restore-determinism   restore the same snapshot twice, run the same action
                           sequence, compare states step by step
  B  snapshot-fidelity     compare the observation returned by the restore with
                           the observation that was live when the snapshot was
                           taken
  C  continue-determinism  restore, run a *long* continuation, and check the
                           final state and success flag match a second replay
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from himoe_libero_bridge.preprocess import build_policy_observation


def obs_signature(obs: dict) -> dict:
    """Numeric parts of the observation that the policy actually consumes."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
            out[k] = np.asarray(v, dtype=np.float64)
    return out


def max_abs_diff(a: dict, b: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for k in sorted(set(a) & set(b)):
        if a[k].shape != b[k].shape:
            return float("inf"), f"{k}: shape {a[k].shape} vs {b[k].shape}"
        d = float(np.abs(a[k] - b[k]).max()) if a[k].size else 0.0
        if d > worst:
            worst, where = d, k
    return worst, where


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--branch-step", type=int, default=60,
                    help="env steps to run before taking the snapshot")
    ap.add_argument("--replay-steps", type=int, default=60)
    ap.add_argument("--out", default="analysis/branch-replay")
    args = ap.parse_args()

    cfg = EpisodeConfig(
        task_suite=args.benchmark, task_id=args.task_id,
        init_state_id=args.init_state_id, seed=args.seed,
        host="127.0.0.1", port=0, libero_root=args.libero_root,
        output_root="/tmp", settle_steps=args.settle_steps,
        max_steps=1000, replan_steps=10, inference_timeout=60.0,
    )
    env, obs, task, prompt = _load_task(cfg)
    rng = np.random.default_rng(0)
    # a fixed, reproducible action tape -- no policy needed to test the simulator
    tape = rng.uniform(-0.3, 0.3, size=(args.replay_steps, 7)).astype(np.float64)
    tape[:, 6] = np.sign(tape[:, 6])

    try:
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        for i in range(args.branch_step):
            obs, _, _, _ = env.step(tape[i % len(tape)].tolist())

        live_obs = obs_signature(obs)
        snap = np.asarray(env.get_sim_state(), dtype=np.float64).copy()

        # ---- B: does restoring reproduce the live observation? -------------
        restored = env.set_init_state(snap)
        d_b, where_b = max_abs_diff(live_obs, obs_signature(restored))

        # ---- A / C: two independent replays from the same snapshot ---------
        runs = []
        for _trial in range(2):
            env.set_init_state(snap)
            states, succ = [], False
            for i in range(args.replay_steps):
                o, _r, _d, _info = env.step(tape[i].tolist())
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                succ = succ or bool(env.check_success())
            runs.append((np.stack(states), succ, obs_signature(o)))

        per_step = np.abs(runs[0][0] - runs[1][0]).max(axis=1)
        d_final, where_final = max_abs_diff(runs[0][2], runs[1][2])

        report = {
            "task_id": args.task_id, "init_state_id": args.init_state_id,
            "branch_step": args.branch_step, "replay_steps": args.replay_steps,
            "snapshot_dim": int(snap.size),
            "B_restore_vs_live_obs_max_abs": d_b,
            "B_worst_key": where_b,
            "A_replay_state_max_abs": float(per_step.max()),
            "A_first_step_diff": float(per_step[0]),
            "A_last_step_diff": float(per_step[-1]),
            "A_bitwise_identical": bool(per_step.max() == 0.0),
            "C_success_flags": [runs[0][1], runs[1][1]],
            "C_final_obs_max_abs": d_final,
            "C_worst_key": where_final,
        }
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"replay-t{args.task_id}-s{args.init_state_id}-b{args.branch_step}.json").write_text(
            json.dumps(report, indent=2))

        print("=== 快照维度 %d ===" % snap.size)
        print("B  恢复观测 vs 快照时的活观测   max|Δ| = %.3e   (%s)" % (d_b, where_b or "-"))
        print("A  同快照两次重放，逐步状态差    max|Δ| = %.3e   逐位相同 = %s"
              % (per_step.max(), per_step.max() == 0.0))
        print("   首步 %.3e   末步 %.3e" % (per_step[0], per_step[-1]))
        print("C  两次重放的成功标志            %s" % (report["C_success_flags"],))
        print("   末观测 max|Δ| = %.3e   (%s)" % (d_final, where_final or "-"))
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
