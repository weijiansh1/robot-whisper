"""Collect sibling candidates at many query states for the Routing Lead Test.

At every control step of a normal rollout, the same observation is queried K times
with independent flow noise.  No environment step happens between those K calls, so
all K candidates answer the identical observation and differ only in noise.  The
episode then advances on candidate 0, which keeps the trajectory a legitimate
on-policy rollout rather than a cherry-picked one.

What each query state yields:

    x_traj      [K, T, n_action_steps, 24]   provisional chunk at every Euler step
    routing     [K, L, T, n_suffix, 32]      full softmax, via the server's zarr
    final       [K, n_action_steps, 7]       the executed-space action chunk

The two built-in negative controls come for free and should be checked before any
positive result is believed:

  * the state token (suffix index 0) is routed identically for all K candidates --
    it attends only to the prefix, which the noise does not touch -- so any method
    that scores candidates from it must come out at chance;
  * AS routing is constant for this whole deployment, same conclusion.

Needs serve_flow_trace.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from flow_query_ids import query_id_at
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

QUERY_KEY = "flow/query_id"
CANDIDATE_KEY = "flow/candidate_id"
def run_episode(config, client, seed, n_candidates, query_base, rng_seed):
    """One rollout; every control step becomes a query state with K siblings."""
    environment, observation, task, prompt = _load_task(config)
    rng = np.random.default_rng(rng_seed)
    records, chunks = [], []
    steps, control, success = 0, 0, False
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        while steps < config.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            query = query_id_at(query_base, control)
            executed = None
            for candidate in range(n_candidates):
                request = dict(policy_observation)
                request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
                request[QUERY_KEY] = query
                request[CANDIDATE_KEY] = candidate
                response = validate_action_response(client.infer(request))
                actions = np.asarray(response[ACTION_KEY], np.float32)
                chunks.append(actions)
                if candidate == 0:
                    executed = actions
            records.append({
                "query_id": query, "flow_noise_seed": seed, "control_step": control,
                "n_candidates": n_candidates,
                "state": np.asarray(policy_observation["observation/state"], np.float32).tolist(),
                "action_steps_before": steps,
            })

            for action in executed[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _reward, _done, _info = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
            control += 1
    finally:
        try:
            environment.close()
        except BaseException:
            pass
    return records, np.stack(chunks), success, steps, control


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--n-episodes", type=int, default=16)
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--noise-seed-base", type=int, default=2000)
    ap.add_argument(
        "--query-base",
        type=int,
        default=0,
        help="first server-global query ID for this client invocation",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", default="flow-lead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=args.inference_timeout) as client:
        metadata = client.metadata
        if metadata.get("flow_tracer") != "serve_flow_trace":
            raise RuntimeError("this driver needs serve_flow_trace.py")
        validate_policy_suite(metadata, args.benchmark)
        (out / "server_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        all_records, all_chunks, episodes = [], [], []
        if args.query_base < 0:
            raise ValueError("--query-base must be non-negative")
        query_base = args.query_base
        for index in range(args.n_episodes):
            seed = args.noise_seed_base + index
            config = EpisodeConfig(
                task_suite=args.benchmark, task_id=args.task_id,
                init_state_id=args.init_state_id, seed=args.seed,
                host=args.host, port=args.port, libero_root=args.libero_root,
                output_root=str(out), settle_steps=args.settle_steps,
                max_steps=args.max_steps, replan_steps=args.replan_steps,
                inference_timeout=args.inference_timeout,
            )
            records, chunks, success, steps, control = run_episode(
                config, client, seed, args.n_candidates, query_base, seed)
            # Outcome was previously printed only; persist it so the flow-trajectory
            # corpus is self-describing and can be split by success/failure offline.
            episodes.append({
                "episode_index": index, "flow_noise_seed": seed,
                "benchmark": args.benchmark, "task_id": args.task_id,
                "init_state_id": args.init_state_id, "success": bool(success),
                "action_steps": int(steps), "control_steps": int(control),
                "query_first": int(records[0]["query_id"]) if records else None,
                "query_last": int(records[-1]["query_id"]) if records else None,
            })
            all_records += records
            all_chunks.append(chunks)
            query_base += control
            print("[%2d] seed %d  %s  %d control steps, %d action steps  (next query ID %d)"
                  % (index, seed, "OK " if success else "FAIL", control, steps, query_base),
                  flush=True)

    np.savez_compressed(out / "candidate_chunks.npz",
                        chunks=np.concatenate(all_chunks).astype(np.float32),
                        query_id=np.asarray([r["query_id"] for r in all_records
                                             for _ in range(r["n_candidates"])], np.int32))
    (out / "query_records.json").write_text(json.dumps(all_records, indent=1))
    (out / "episodes.json").write_text(json.dumps(episodes, indent=1))
    (out / "experiment_config.json").write_text(json.dumps({
        "protocol": "flow-lead-v1", "label": args.label,
        "benchmark": args.benchmark, "task_id": args.task_id,
        "init_state_id": args.init_state_id, "n_episodes": args.n_episodes,
        "n_candidates": args.n_candidates, "noise_seed_base": args.noise_seed_base,
        "query_base": args.query_base,
        "query_end_exclusive": query_base,
        "replan_steps": args.replan_steps, "max_steps": args.max_steps,
    }, indent=2, sort_keys=True))
    print("\n%s: %d query states x %d candidates in %.1f min"
          % (args.label, len(all_records), args.n_candidates, (time.time() - started) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
