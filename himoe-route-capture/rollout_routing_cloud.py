"""What routing can the policy itself reach at one state, and is a donor inside it?

The step-11 transplant proved that some control-step-11 states admit both outcomes
*physically*: overriding the routing there flips the rollout.  It did not prove the
policy could ever get there on its own, because the transplanted routing came from
a different state and the policy at this state would never produce it.  Formally the
transplant explores

    Y_do(s_k) = { G(s_k, r, eps_obs) : r }

while the quantity that matters for a commitment curve is

    Y_pi(s_k) = { F(s_k, eps) : eps }

and neither set contains the other.  The gap closes for any donor that lies inside

    R_pi(s_k) = { rho(s_k, eps_k) : eps_k }

the routing the policy reaches at this very state by resampling only its own flow
noise -- a flip driven by such a donor is a genuine Y_pi counterexample.

This driver measures R_pi.  Each episode is run to control step K with its own noise,
and at that step the same observation is re-queried N times with independent noise
draws.  No environment step happens in between, so every draw sees the identical
state; the episode then advances on its natural action only.  The cloud is not
assumed non-degenerate -- at fixed state the cross-noise top-4 Jaccard was 0.33-0.56,
so it has real extent, but whether donors land inside it is the open question.

Writes one npz per seed; analyse with analyze_routing_cloud.py.
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
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

RECORD_KEY = "patch/record"
RETURN_KEY = "patch/return_routing"
ROUTING_IDX_KEY = "patch/routing_idx"
ROUTING_WEIGHT_KEY = "patch/routing_weight"


def _routing_call(client, request, slot):
    request = dict(request)
    request[RECORD_KEY] = slot
    request[RETURN_KEY] = True
    response = validate_action_response(client.infer(request))
    if ROUTING_IDX_KEY not in response:
        raise RuntimeError("server did not return routing; needs serve_patched_router.py")
    return (
        np.asarray(response[ROUTING_IDX_KEY], np.uint8),
        np.asarray(response[ROUTING_WEIGHT_KEY], np.float32),
        np.asarray(response[ACTION_KEY], np.float32),
    )


def collect_cloud(config, client, flow_noise_seed, branch_control, n_cloud, cloud_seed_base):
    """Run to ``branch_control`` on the episode's own noise, then resample there."""
    environment, observation, task, prompt = _load_task(config)
    rng = np.random.default_rng(flow_noise_seed)
    cloud_rng = np.random.default_rng(cloud_seed_base + flow_noise_seed)
    steps = 0
    control = 0
    success = False
    natural = None
    cloud_idx, cloud_weight = [], []
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        while steps < config.max_steps and not success and control <= branch_control:
            policy_observation = build_policy_observation(observation, prompt)
            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)

            if control == branch_control:
                idx, weight, actions = _routing_call(client, request, "s%d" % flow_noise_seed)
                natural = (idx, weight)
                # The cloud is the policy's own reach at this state: identical
                # observation, independent flow noise, no environment step between.
                for draw in range(n_cloud):
                    alt = dict(policy_observation)
                    alt[FLOW_NOISE_KEY] = cloud_rng.standard_normal(
                        FLOW_NOISE_SHAPE
                    ).astype(np.float32)
                    c_idx, c_weight, _ = _routing_call(client, alt, "cloud")
                    cloud_idx.append(c_idx)
                    cloud_weight.append(c_weight)
            else:
                response = validate_action_response(client.infer(request))
                actions = np.asarray(response[ACTION_KEY], np.float32)

            if control == branch_control:
                break

            for action in actions[: config.replan_steps]:
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

    if natural is None:
        return None
    return {
        "flow_noise_seed": flow_noise_seed,
        "branch_control": branch_control,
        "action_steps_before_branch": steps,
        "natural_idx": natural[0],
        "natural_weight": natural[1],
        "cloud_idx": np.stack(cloud_idx),
        "cloud_weight": np.stack(cloud_weight),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--branch-control", type=int, default=11)
    ap.add_argument("--n-recipients", type=int, default=16)
    ap.add_argument("--n-cloud", type=int, default=64)
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--cloud-seed-base", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", default="routing-cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.noise_seed_base + i for i in range(args.n_recipients)]
    started = time.time()

    with PolicyClient(
        host=args.host, port=args.port, connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        metadata = client.metadata
        if metadata.get("route_patcher") != "serve_patched_router":
            raise RuntimeError("this driver needs serve_patched_router.py")
        validate_policy_suite(metadata, args.benchmark)
        (out / "server_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        collected = []
        for seed in seeds:
            config = EpisodeConfig(
                task_suite=args.benchmark, task_id=args.task_id,
                init_state_id=args.init_state_id, seed=args.seed,
                host=args.host, port=args.port, libero_root=args.libero_root,
                output_root=str(out), settle_steps=args.settle_steps,
                max_steps=args.max_steps, replan_steps=args.replan_steps,
                inference_timeout=args.inference_timeout,
            )
            record = collect_cloud(
                config, client, seed, args.branch_control, args.n_cloud, args.cloud_seed_base
            )
            if record is None:
                print("[cloud] seed %d ended before control step %d -- skipped"
                      % (seed, args.branch_control), flush=True)
                continue
            np.savez_compressed(out / ("cloud_%d.npz" % seed), **record)
            collected.append(seed)
            print("[cloud] seed %d  cloud %s  natural %s"
                  % (seed, record["cloud_idx"].shape, record["natural_idx"].shape), flush=True)

    (out / "experiment_config.json").write_text(json.dumps({
        "protocol": "routing-cloud-v1",
        "label": args.label,
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "branch_control": args.branch_control,
        "n_recipients": args.n_recipients,
        "n_cloud": args.n_cloud,
        "noise_seed_base": args.noise_seed_base,
        "cloud_seed_base": args.cloud_seed_base,
        "environment_seed": args.seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "collected_seeds": collected,
    }, indent=2, sort_keys=True))
    print("%s: %d seeds in %.1f min" % (args.label, len(collected), (time.time() - started) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
