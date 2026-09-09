"""Why do two bitwise-identical simulator states render different wrist images?

branch_replay_test.py found that replaying the same action tape from the same
snapshot reproduces MuJoCo state exactly (max|dstate| = 0) while the returned
``robot0_eye_in_hand_image`` still differs by ~140 grey levels.  Either the
renderer is nondeterministic, or the image attached to an observation is not the
render of the state it is attached to.

This probe separates those: it holds one state fixed and renders it repeatedly,
then restores and renders again, and reports where the difference lives.
"""

from __future__ import annotations

import argparse

import numpy as np

from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)

KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def stats(a: np.ndarray, b: np.ndarray) -> str:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return ("max %3d  mean %6.3f  差异像素 %5.2f%%"
            % (d.max(), d.mean(), 100.0 * (d > 0).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--branch-step", type=int, default=60)
    args = ap.parse_args()

    cfg = EpisodeConfig(
        task_suite="libero_goal", task_id=args.task_id,
        init_state_id=args.init_state_id, seed=7, host="127.0.0.1", port=0,
        libero_root=args.libero_root, output_root="/tmp", settle_steps=10,
        max_steps=1000, replan_steps=10, inference_timeout=60.0,
    )
    env, obs, _task, _prompt = _load_task(cfg)
    rng = np.random.default_rng(0)
    tape = rng.uniform(-0.3, 0.3, size=(120, 7))
    tape[:, 6] = np.sign(tape[:, 6])
    try:
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        for i in range(args.branch_step):
            obs, _, _, _ = env.step(tape[i].tolist())
        live = {k: np.asarray(obs[k]).copy() for k in KEYS if k in obs}
        snap = np.asarray(env.get_sim_state(), dtype=np.float64).copy()

        print("=== 1) 同一活状态，连续两次 _get_observations（不 step）===")
        o1 = env.env._get_observations()
        o2 = env.env._get_observations()
        for k in live:
            print("  %-26s %s" % (k, stats(np.asarray(o1[k]), np.asarray(o2[k]))))

        print("=== 2) 恢复快照后的观测 vs 快照时的活观测 ===")
        r1 = env.set_init_state(snap)
        for k in live:
            print("  %-26s %s" % (k, stats(live[k], np.asarray(r1[k]))))

        print("=== 3) 连续两次恢复同一快照 ===")
        r2 = env.set_init_state(snap)
        for k in live:
            print("  %-26s %s" % (k, stats(np.asarray(r1[k]), np.asarray(r2[k]))))

        print("=== 4) 恢复后再 step 一次同样的动作，两趟对比 ===")
        env.set_init_state(snap)
        a1, _, _, _ = env.step(tape[args.branch_step].tolist())
        env.set_init_state(snap)
        b1, _, _, _ = env.step(tape[args.branch_step].tolist())
        for k in live:
            print("  %-26s %s" % (k, stats(np.asarray(a1[k]), np.asarray(b1[k]))))

        print("=== 5) 恢复后 sim 状态是否与快照一致 ===")
        env.set_init_state(snap)
        back = np.asarray(env.get_sim_state(), dtype=np.float64)
        print("  max|Δstate| = %.3e" % np.abs(back - snap).max())
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
