"""Is a restored snapshot good enough to fork candidates from?

The whole same-state candidate experiment rests on being able to run N different
action chunks from one identical world state.  ``regenerate_obs_from_state`` restores
time/qpos/qvel/act, but NOT the solver's warm-start (``qacc_warmstart``), and this
pipeline has already been shown not to be bit-reproducible on contact-rich steps.  So
the premise has to be measured before any pilot is built on it.

Three things are checked at a post-contact control step:

  restore    does the observation come back identical right after restoring
  rerun      does executing the same chunk from the restored state land in the same
             place as the first time
  branch     do two *different* chunks from the same restored state separate by much
             more than the rerun noise -- if not, the forking signal is below the
             floor and N candidates cannot be told apart

Run in the LIBERO env, no GPU and no policy needed: the chunks are canned.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from himoe_libero_bridge.libero_runtime import LIBERO_DUMMY_ACTION, EpisodeConfig, _load_task
from himoe_libero_bridge.preprocess import build_policy_observation

from rollout_with_routes import sim_joint_layout


def run_chunk(env, chunk):
    for a in chunk:
        env.step(np.asarray(a, np.float64).tolist())
    return np.asarray(env.get_sim_state(), np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--client-dir", required=True,
                    help="capture to take a real trajectory prefix and real action chunks from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fork-step", type=int, default=11, help="control step to fork at")
    args = ap.parse_args()

    with np.load("%s/episode_%02d.npz" % (args.client_dir, args.episode)) as z:
        acts = z["actions"]
    # two genuinely different candidates: this episode's chunk at the fork step, and
    # another episode's chunk at the same step
    with np.load("%s/episode_%02d.npz" % (args.client_dir, args.episode + 1)) as z:
        acts_other = z["actions"]

    config = EpisodeConfig(
        task_suite=args.benchmark, task_id=args.task_id, init_state_id=args.init_state_id,
        seed=args.seed, host="127.0.0.1", port=0, libero_root=args.libero_root,
        output_root="/tmp", settle_steps=10, max_steps=300, replan_steps=10)
    env, observation, _task, prompt = _load_task(config)
    layout = sim_joint_layout(env)
    drawer = [j for j in layout["joints"] if j["joint"] == "wooden_cabinet_1_middle_level"]
    d_lo = drawer[0]["state_lo"] if drawer else None

    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        for t in range(args.fork_step):
            for a in acts[t][:10]:
                observation, _, _, _ = env.step(np.asarray(a, np.float64).tolist())

        snap = np.asarray(env.get_sim_state(), np.float64)
        obs_before = np.asarray(build_policy_observation(observation, prompt)["observation/state"],
                                np.float32)

        # (1) restore fidelity
        observation = env.regenerate_obs_from_state(snap)
        obs_after = np.asarray(build_policy_observation(observation, prompt)["observation/state"],
                               np.float32)
        restore_err = float(np.abs(obs_after - obs_before).max())
        state_err = float(np.abs(np.asarray(env.get_sim_state(), np.float64) - snap).max())

        # (2) same chunk twice from the same snapshot
        env.regenerate_obs_from_state(snap)
        s_a = run_chunk(env, acts[args.fork_step][:10])
        env.regenerate_obs_from_state(snap)
        s_b = run_chunk(env, acts[args.fork_step][:10])
        rerun_err = float(np.abs(s_a - s_b).max())

        # (3) a different chunk from the same snapshot
        env.regenerate_obs_from_state(snap)
        s_c = run_chunk(env, acts_other[args.fork_step][:10])
        branch_gap = float(np.abs(s_a - s_c).max())

        out = {
            "fork_step": args.fork_step,
            "restore_obs_error": restore_err,
            "restore_sim_state_error": state_err,
            "rerun_same_chunk_error": rerun_err,
            "different_chunk_gap": branch_gap,
            "gap_over_noise": (branch_gap / rerun_err) if rerun_err > 0 else float("inf"),
        }
        if d_lo is not None:
            out["drawer_same_chunk"] = [round(float(s_a[d_lo]), 6), round(float(s_b[d_lo]), 6)]
            out["drawer_other_chunk"] = round(float(s_c[d_lo]), 6)
        print(json.dumps(out, indent=2))
        print("\nrestore is usable iff rerun error is ~0 and the different-chunk gap is "
              "orders of magnitude larger")
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
