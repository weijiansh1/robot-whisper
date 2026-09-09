"""Run LIBERO episodes and capture HB-MoE routing at every control step.

Follows the audited protocol in libero_runtime.run_episode (settle 10, replan
10, seed 7, fixed per-episode flow noise) but turns on the server's routing
capture and keeps the trace instead of a video.

Runs in the LIBERO env (python 3.8); the model server holds the hooks.
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
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    ROUTING_EXPERT_WEIGHTS_KEY,
    ROUTING_LAYER_INDICES_KEY,
    validate_action_response,
)

# Deliberately not added to the audited bridge protocol: an opt-in extra request
# key that only serve_with_recorder.py consumes (it pops the key before the
# policy sees the observation).  Sent only when the server advertises
# ``episode_id_key`` in its metadata, so pointing this client at any other
# server keeps the old wire format exactly.
EPISODE_ID_KEY = "episode_id"


def should_send_episode_id(metadata):
    """Only stamp episode_id on servers that say they consume it.

    Sending it to a server without a recorder would pass an unexpected key
    through to the policy, so the default on any silence is False.
    """
    return bool(metadata) and metadata.get("episode_id_key") == EPISODE_ID_KEY


def sim_joint_layout(environment):
    """Where each joint sits inside the flattened MuJoCo state.

    ``get_sim_state()`` returns ``[time] + qpos + qvel``, so a joint's qpos slice
    is offset by one.  ``get_joint_qpos_addr`` gives an int for a 1-DOF joint and
    a ``(lo, hi)`` pair for free/ball joints, hence the branch.
    """
    model = environment.sim.model
    joints = []
    for name in model.joint_names:
        addr = model.get_joint_qpos_addr(name)
        lo, hi = (int(addr[0]), int(addr[1])) if isinstance(addr, tuple) else (int(addr), int(addr) + 1)
        joints.append({
            "joint": str(name),
            "state_lo": 1 + lo,
            "state_hi": 1 + hi,
            "is_robot": str(name).startswith(("robot0_", "gripper0_")),
        })
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "state_dim": 1 + int(model.nq) + int(model.nv),
        "layout": "[time] + qpos + qvel",
        "joints": joints,
        "obj_of_interest": [str(o) for o in getattr(environment, "obj_of_interest", [])],
    }


def run_one(config, client, flow_noise_seed, no_capture=False, episode_id=None):
    """One episode.  Returns (summary, routes, weights, states, actions, sim, layers, layout).

    ``sim`` is the full flattened MuJoCo state at every control step.  The policy's
    own ``observation/state`` covers the robot only -- eef pose and the two fingers
    -- so it cannot control for object progress (on the drawer tasks, the cabinet's
    slide joint).  Capturing the sim state makes that controllable after the fact.

    ``episode_id``, when given, is stamped on every request so a server-side
    recorder can label its rows.  The server writes one flat zarr with no episode
    boundaries, so without this the only way back to per-episode slices is
    accumulating ``inference_calls`` from summaries.json and hoping nothing was
    dropped.  Only send it to a server that advertises support: other servers pass
    the observation straight to the policy, which does not expect the extra key.
    """
    environment, observation, task, prompt = _load_task(config)
    rng = np.random.default_rng(flow_noise_seed)
    ids_seq, w_seq, state_seq, action_seq, sim_seq = [], [], [], [], []
    layout = sim_joint_layout(environment)
    layer_indices = None
    steps = 0
    success = False
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        while steps < config.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(
                np.float32
            )
            if not no_capture:
                request[ROUTING_CAPTURE_KEY] = True
            if episode_id is not None:
                request[EPISODE_ID_KEY] = int(episode_id)
            response = validate_action_response(client.infer(request))
            actions = response[ACTION_KEY]

            if not no_capture:
                if ROUTING_EXPERT_IDS_KEY not in response:
                    raise RuntimeError("server returned no routing trace")
                ids_seq.append(np.asarray(response[ROUTING_EXPERT_IDS_KEY], np.uint8))
                w_seq.append(np.asarray(response[ROUTING_EXPERT_WEIGHTS_KEY], np.float16))
                if layer_indices is None:
                    layer_indices = np.asarray(response[ROUTING_LAYER_INDICES_KEY], np.int16)
            else:
                ids_seq.append(np.zeros((1,), np.uint8))
                w_seq.append(np.zeros((1,), np.float16))
            state_seq.append(np.asarray(policy_observation["observation/state"], np.float32))
            action_seq.append(np.asarray(actions[: config.replan_steps], np.float32))
            # same instant as the policy observation, so the two align index-for-index
            sim_seq.append(np.asarray(environment.get_sim_state(), np.float32))

            for action in actions[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _reward, _done, _info = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
    finally:
        try:
            environment.close()
        except BaseException:
            pass

    summary = {
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "seed": config.seed,
        "flow_noise_seed": flow_noise_seed,
        "task_name": str(task.name),
        "prompt": prompt,
        "success": success,
        "action_steps": steps,
        "inference_calls": len(ids_seq),
    }
    return (
        summary,
        np.stack(ids_seq),
        np.stack(w_seq),
        np.stack(state_seq),
        np.stack(action_seq),
        np.stack(sim_seq),
        layer_indices,
        layout,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument(
        "--init-state-ids",
        help="Comma-separated init states to repeat; with --repeats this replaces --episodes. "
             "Holding the init state fixed makes flow noise the only source of variation.",
    )
    ap.add_argument("--repeats", type=int, default=1,
                    help="Flow-noise repeats per init state")
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    ap.add_argument("--no-routing-capture", action="store_true",
                    help="Server captures routes itself (serve_with_recorder.py)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        meta = client.metadata
        validate_policy_suite(meta, args.benchmark)
        if not args.no_routing_capture and not meta.get("routing_capture_supported", False):
            raise RuntimeError("server does not support routing capture")
        send_episode_id = should_send_episode_id(meta)
        print("server layout :", meta.get("libero_wrist_layout"))
        print("checkpoint sha:", meta.get("checkpoint_sha256", "")[:16])
        print("episode_id    :", "sent" if send_episode_id
              else "not advertised by server (boundaries inferred from inference_calls)")
        # Server handshake verbatim, plus the client-side truth: the server
        # advertising routing_capture_supported says nothing about whether this
        # run actually requested per-response routing.
        meta_out = dict(meta)
        meta_out["client_routing_capture"] = not args.no_routing_capture
        if args.no_routing_capture:
            meta_out["client_routing_note"] = (
                "routing was captured server-side only (serve_with_recorder.py); "
                "episode npz files carry no routing arrays"
            )
        (out / "server_metadata.json").write_text(json.dumps(meta_out, indent=2, sort_keys=True))

        if args.init_state_ids:
            states_to_run = [int(v) for v in args.init_state_ids.split(",")]
            plan = [
                (s, r, args.noise_seed_base + r)
                for s in states_to_run
                for r in range(args.repeats)
            ]
        else:
            plan = [(i, 0, i) for i in range(args.episodes)]
        print("plan: %d episodes (%d init states x %d noise repeats)"
              % (len(plan), len(set(p[0] for p in plan)), args.repeats))

        summaries = []
        for episode, (init_state, repeat, noise_seed) in enumerate(plan):
            config = EpisodeConfig(
                task_suite=args.benchmark,
                task_id=args.task_id,
                init_state_id=init_state,
                seed=args.seed,
                host=args.host,
                port=args.port,
                libero_root=args.libero_root,
                output_root=str(out),
                settle_steps=args.settle_steps,
                max_steps=args.max_steps,
                replan_steps=args.replan_steps,
                inference_timeout=args.inference_timeout,
            )
            t0 = time.time()
            summary, ids, weights, states, actions, sim_states, layers, layout = run_one(
                config, client, flow_noise_seed=noise_seed,
                no_capture=args.no_routing_capture,
                episode_id=episode if send_episode_id else None,
            )
            summary["wall_s"] = round(time.time() - t0, 1)
            summary["label"] = args.label
            summary["repeat"] = repeat
            summary["episode_index"] = episode
            # one scene per capture, so the layout must not drift between episodes
            if episode == 0:
                (out / "sim_layout.json").write_text(json.dumps(layout, indent=2))
            elif sim_states.shape[1] != layout["state_dim"]:
                raise RuntimeError(
                    "sim state dim changed mid-capture: %d != %d"
                    % (sim_states.shape[1], layout["state_dim"])
                )
            arrays = dict(state=states, actions=actions, sim_state=sim_states)
            if not args.no_routing_capture:
                arrays.update(expert_ids=ids, expert_weights=weights,
                              layer_indices=layers)
            # No stub routing keys in no-capture mode: an absent key fails
            # loud, an all-zero placeholder reads as a real route.
            np.savez_compressed(out / ("episode_%02d.npz" % episode), **arrays)
            summaries.append(summary)
            print(
                "ep %-3d init=%-2d rep=%-2d success=%-5s steps=%3d infers=%2d  %.0fs  routes%s"
                % (
                    episode,
                    init_state,
                    repeat,
                    summary["success"],
                    summary["action_steps"],
                    summary["inference_calls"],
                    summary["wall_s"],
                    "=server-side" if args.no_routing_capture else str(ids.shape),
                )
            )
            (out / "summaries.json").write_text(json.dumps(summaries, indent=2))

    n_ok = sum(1 for s in summaries if s["success"])
    print("\n%s: %d/%d success" % (args.label, n_ok, len(summaries)))


if __name__ == "__main__":
    main()
