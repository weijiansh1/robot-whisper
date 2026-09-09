"""Candidate order contaminates a fork; does zeroing the solver warm start fix it?

``get_sim_state``/``set_state_from_flattened`` round-trip time, qpos, qvel and act,
but not ``qacc_warmstart`` -- the solver's starting guess, which carries over from
whatever ran last.  Back-to-back reruns of one chunk therefore agree to 5e-15, while
the same chunk rerun *after* seven other candidates does not, which makes the error
order-dependent rather than random: exactly the kind that biases a ranking.

Measured here both ways, with and without zeroing the warm start on restore.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from himoe_libero_bridge.libero_runtime import LIBERO_DUMMY_ACTION, EpisodeConfig, _load_task

from rollout_with_routes import sim_joint_layout

PROGRESS_JOINT = "wooden_cabinet_1_middle_level"


def restore(env, snap, zero_warmstart):
    env.regenerate_obs_from_state(snap)
    # robosuite counts steps in Python, outside the MuJoCo state, so the horizon
    # keeps ticking across restores and a long forking session terminates mid-branch
    try:
        env.env.timestep = 0
        env.env.done = False
    except AttributeError:
        pass
    if zero_warmstart:
        env.sim.data.qacc_warmstart[:] = 0.0
        env.sim.forward()


def run(env, chunk):
    for a in chunk:
        env.step(np.asarray(a, np.float64).tolist())
    return np.asarray(env.get_sim_state(), np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--fork-step", type=int, default=12)
    args = ap.parse_args()

    with np.load("%s/episode_%02d.npz" % (args.client_dir, args.episode)) as z:
        acts = z["actions"]
    others = []
    for k in range(1, 8):
        with np.load("%s/episode_%02d.npz" % (args.client_dir, (args.episode + k) % 60)) as z:
            others.append(z["actions"][args.fork_step][:10])

    config = EpisodeConfig(
        task_suite="libero_goal", task_id=0, init_state_id=24, seed=7,
        host="127.0.0.1", port=0, libero_root=args.libero_root, output_root="/tmp",
        settle_steps=10, max_steps=300, replan_steps=10)
    env, observation, _t, _p = _load_task(config)
    layout = sim_joint_layout(env)
    d_lo = [j for j in layout["joints"] if j["joint"] == PROGRESS_JOINT][0]["state_lo"]
    out = {}
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        for t in range(args.fork_step):
            for a in acts[t][:10]:
                observation, _, _, _ = env.step(np.asarray(a, np.float64).tolist())
        snap = np.asarray(env.get_sim_state(), np.float64)
        target = acts[args.fork_step][:10]

        for zero in (False, True):
            restore(env, snap, zero)
            s1 = run(env, target)
            for o in others:                      # seven other candidates in between
                restore(env, snap, zero)
                run(env, o)
            restore(env, snap, zero)
            s2 = run(env, target)
            out["zero_warmstart=%s" % zero] = {
                "state_error_after_interleave": float(np.abs(s1 - s2).max()),
                "drawer": [round(float(s1[d_lo]), 9), round(float(s2[d_lo]), 9)],
                "drawer_error": float(abs(s1[d_lo] - s2[d_lo])),
            }
        spread = []
        for o in others:
            restore(env, snap, True)
            spread.append(float(run(env, o)[d_lo]))
        out["between_candidate_drawer_spread"] = round(max(spread) - min(spread), 9)
        print(json.dumps(out, indent=2))
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
