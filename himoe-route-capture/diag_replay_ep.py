"""Where does one episode's replay start to diverge?

Error at step 0 means the initial state was not restored identically; error that is
zero for a while and then grows means a tiny perturbation is being amplified by
contact dynamics, which bounds how far a replay can be trusted.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from himoe_libero_bridge.libero_runtime import LIBERO_DUMMY_ACTION, EpisodeConfig, _load_task
from himoe_libero_bridge.preprocess import build_policy_observation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    args = ap.parse_args()

    client = pathlib.Path(args.client_dir)
    summaries = json.loads((client / "summaries.json").read_text())
    summary = [s for s in summaries if s["episode_index"] == args.episode][0]
    with np.load(client / ("episode_%02d.npz" % args.episode)) as z:
        saved_actions, saved_state = z["actions"], z["state"]

    config = EpisodeConfig(
        task_suite=args.benchmark, task_id=int(summary["task_id"]),
        init_state_id=int(summary["init_state_id"]), seed=int(summary["seed"]),
        host="127.0.0.1", port=0, libero_root=args.libero_root, output_root="/tmp",
        settle_steps=10, max_steps=300, replan_steps=10)
    env, observation, _task, prompt = _load_task(config)
    steps, success = 0, False
    print("ep %d  recorded: n=%d steps=%d success=%s"
          % (args.episode, summary["inference_calls"], summary["action_steps"],
             summary["success"]))
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        for t in range(len(saved_actions)):
            if steps >= config.max_steps or success:
                break
            st = np.asarray(build_policy_observation(observation, prompt)["observation/state"],
                            np.float32)
            err = float(np.abs(st - saved_state[t]).max())
            flag = "" if err == 0 else ("  <-- 首次偏离" if err > 0 and t and True else "")
            print("  t=%-3d steps=%-4d err=%.3e%s" % (t, steps, err, flag if err else ""))
            for action in saved_actions[t][: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _r, _d, _i = env.step(action.tolist())
                steps += 1
                success = bool(env.check_success())
                if success:
                    break
    finally:
        try:
            env.close()
        except BaseException:
            pass
    print("replay: steps=%d success=%s" % (steps, success))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
