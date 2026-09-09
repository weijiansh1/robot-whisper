"""Recover the object state for an existing capture by replaying its actions.

The capture already stores the executed action chunks, and LIBERO is deterministic
given (task, init state, seed).  So the episodes can be reproduced in the env with
no policy and no GPU, and ``get_sim_state()`` read off at each control step.  That
retrofits the object DOFs -- the cabinet slide joint above all -- onto captures made
before rollout_with_routes.py started saving them.

The replay is only trustworthy if it lands on the same trajectory, so every episode
is checked against the recorded run on four counts: control-step count, executed
action steps, success flag, and the policy's own ``observation/state`` at every
step.  Any mismatch is a hard error rather than a warning -- a silently divergent
replay would attach the wrong object state to the routing.

Writes a sidecar directory; the original capture is never modified.
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

from rollout_with_routes import sim_joint_layout

# Most episodes replay bit-exactly, but contact-rich ones pick up a sub-millimetre
# offset partway through that then stays bounded -- MuJoCo's constraint solver is
# not bit-reproducible once objects settle.  The counts and the success flag must
# still match exactly; the state tolerance is set where a divergence would start to
# matter physically rather than where float noise begins.  Note the direction of
# the residual risk: a slightly noisy control under-removes, which flatters the
# routing rather than the other way round.
STATE_TOL = 5e-3


def replay_one(config, saved_actions, summary):
    env, observation, _task, prompt = _load_task(config)
    layout = sim_joint_layout(env)
    sim_seq, state_seq = [], []
    steps = 0
    success = False
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())

        t = 0
        while steps < config.max_steps and not success:
            if t >= len(saved_actions):
                break
            policy_observation = build_policy_observation(observation, prompt)
            state_seq.append(np.asarray(policy_observation["observation/state"], np.float32))
            sim_seq.append(np.asarray(env.get_sim_state(), np.float32))
            for action in saved_actions[t][: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _r, _d, _i = env.step(action.tolist())
                steps += 1
                success = bool(env.check_success())
                if success:
                    break
            t += 1
    finally:
        try:
            env.close()
        except BaseException:
            pass

    check = {
        "episode": summary["episode_index"],
        "n_control_recorded": int(summary["inference_calls"]),
        "n_control_replay": len(sim_seq),
        "action_steps_recorded": int(summary["action_steps"]),
        "action_steps_replay": steps,
        "success_recorded": bool(summary["success"]),
        "success_replay": bool(success),
    }
    return np.stack(sim_seq), np.stack(state_seq), layout, check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="replay only the first N episodes")
    args = ap.parse_args()

    client = pathlib.Path(args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = json.loads((client / "summaries.json").read_text())
    if args.limit:
        summaries = summaries[: args.limit]

    checks, layout = [], None
    for summary in summaries:
        with np.load(client / ("episode_%02d.npz" % summary["episode_index"])) as z:
            saved_actions = z["actions"]
            saved_state = z["state"]
        config = EpisodeConfig(
            task_suite=args.benchmark,
            task_id=int(summary["task_id"]),
            init_state_id=int(summary["init_state_id"]),
            seed=int(summary["seed"]),
            host="127.0.0.1",
            port=0,
            libero_root=args.libero_root,
            output_root=str(out),
            settle_steps=args.settle_steps,
            max_steps=args.max_steps,
            replan_steps=args.replan_steps,
        )
        sim, state, layout, check = replay_one(config, saved_actions, summary)

        bad = []
        if check["n_control_replay"] != check["n_control_recorded"]:
            bad.append("control-step count")
        if check["action_steps_replay"] != check["action_steps_recorded"]:
            bad.append("action-step count")
        if check["success_replay"] != check["success_recorded"]:
            bad.append("success flag")
        err = float(np.abs(state - saved_state[: len(state)]).max()) if len(state) else np.inf
        check["max_state_error"] = err
        if err > STATE_TOL:
            bad.append("observation/state (max err %.2e)" % err)
        check["ok"] = not bad
        check["reasons"] = bad

        # written either way: a diverged episode is still worth inspecting, and the
        # consumer gates on replay_check.json rather than on file existence
        np.save(out / ("sim_state_%02d.npy" % summary["episode_index"]), sim)
        checks.append(check)
        print("ep %-3d %-4s n=%-3d steps=%-3d success=%-5s  state err=%.2e %s"
              % (check["episode"], "ok" if check["ok"] else "DIV",
                 check["n_control_replay"], check["action_steps_replay"],
                 check["success_replay"], err, "; ".join(bad)))

    (out / "sim_layout.json").write_text(json.dumps(layout, indent=2))
    (out / "replay_check.json").write_text(json.dumps(checks, indent=2))
    errs = np.array([c["max_state_error"] for c in checks])
    ok = np.array([c["ok"] for c in checks])
    succ = np.array([c["success_recorded"] for c in checks])
    print("\nreplayed %d episodes; %d within tolerance (%.0e), %d diverged"
          % (len(checks), int(ok.sum()), STATE_TOL, int((~ok).sum())))
    print("state error  bit-exact: %d   median %.2e   p95 %.2e   max %.2e"
          % (int((errs == 0).sum()), np.median(errs), np.quantile(errs, 0.95), errs.max()))
    # divergence that correlates with the outcome would bias any analysis run on the
    # surviving subset, so the split is reported rather than left implicit
    print("success rate  kept: %d/%d   dropped: %d/%d"
          % (int(succ[ok].sum()), int(ok.sum()),
             int(succ[~ok].sum()), int((~ok).sum())))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
