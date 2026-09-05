#!/usr/bin/env python3
"""CALVIN policy server with the MoE route recorder attached.

Wraps, rather than edits, the audited CALVIN server: `create_policy` builds
exactly the object `himoe-calvin-eval` already talks to, and this only attaches
the recorder to the MoEVLA module and brackets `policy.infer` with
begin/end_control_step.  Same shape as `serve_with_recorder.py` for LIBERO.

Why CALVIN is worth the wiring: on LIBERO the state token's routing turned out to
be ~96% a nonlinear function of the robot's own 8 numbers, replicated across
three checkpoints.  CALVIN-D is a different simulator, a different action space
(`calvin_d_joint`, whose data_mask is the *other* AS signature), and its
`robot_obs` is 15-dim carrying **both** the end-effector pose and the seven joint
angles.  So it answers a sharper question than LIBERO could: given a model
trained in joint space, does the routing encode the joints or the end effector?

The proprioception is recorded here rather than reconstructed later, because on
this pipeline the client sends it to the server -- `observation/state` is
`robot_obs` verbatim (evaluator.py) -- so routing and state come from the same
call and cannot drift out of alignment.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--calvin-root", required=True)
    ap.add_argument("--wrist-layout", default="released-left")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-steps", type=int, default=32)
    ap.add_argument(
        "--low-memory-load",
        action="store_true",
        help="mmap checkpoint into a meta-constructed model to bound CPU memory",
    )
    args = ap.parse_args()

    import himoe_calvin_alignment.policy_server as ps

    inner_args = argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir, upstream_root=args.upstream_root,
        calvin_root=args.calvin_root, wrist_layout=args.wrist_layout,
        host=args.host, port=args.port)
    if args.low_memory_load:
        from himoe_calvin_alignment.constants import TRAIN_CONFIG
        from himoe_low_memory_load import low_memory_himoe_load

        with low_memory_himoe_load(
            args.checkpoint_dir,
            args.upstream_root,
            TRAIN_CONFIG,
            target_device="cuda",
        ) as load_audit:
            policy = ps.create_policy(inner_args)
        expected_weights = str(
            (pathlib.Path(policy.metadata["checkpoint_path"]) / "pytorch_model.pth")
            .expanduser()
            .resolve()
        )
        if load_audit.checkpoint_path != expected_weights:
            raise RuntimeError("low-memory loader checkpoint identity mismatch")
        policy.metadata["checkpoint_load_audit"] = load_audit.as_metadata()
        print(
            "low-memory CUDA checkpoint load: %d tensors, %.2f GiB target"
            % (
                load_audit.state_key_count,
                load_audit.target_tensor_bytes / (1024 ** 3),
            ),
            flush=True,
        )
    else:
        policy = ps.create_policy(inner_args)

    from himoe_route_store import ZarrRouteWriter
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates

    core = policy._policy.model
    gates = discover_gates(core)
    print("discovered %d gates (%d HB, %d AS)"
          % (len(gates), sum(g.kind == "HB" for g in gates),
             sum(g.kind == "AS" for g in gates)), flush=True)
    cfg = core.config
    n_denoise = int(cfg.num_steps)
    n_suffix = int(cfg.n_action_steps) + 1
    print("n_denoise=%d n_suffix=%d" % (n_denoise, n_suffix), flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recorder = HiMoERouteRecorder(core, store_full_probs=True).attach()
    writer = ZarrRouteWriter(
        str(out / "routes.zarr"),
        n_hb_layers=sum(g.kind == "HB" for g in gates),
        n_as_layers=sum(g.kind == "AS" for g in gates),
        n_denoise=n_denoise, n_suffix=n_suffix,
        chunk_steps=args.chunk_steps, store_full_probs=True)

    states: list[np.ndarray] = []
    prompts: list[str] = []
    stats = {"calls": 0, "as_collapsed": 0}
    inner_infer = policy.infer

    def infer(observation):
        # a control step is one call; the episode id is not known to this server
        # (CALVIN's unit is a five-subtask chain driven entirely client-side), so
        # the prompt is stored per row and the chain is reconstructed offline
        recorder.begin_control_step(episode_id=0, control_step=stats["calls"])
        try:
            result = inner_infer(observation)
        finally:
            rec = recorder.end_control_step()
        writer.append(rec)
        states.append(np.asarray(observation["observation/state"], np.float32))
        prompts.append(str(observation.get("prompt", "")))
        stats["calls"] += 1
        if getattr(rec, "as_collapsed", True):
            stats["as_collapsed"] += 1
        if stats["calls"] % 200 == 0:
            print("  %d control steps captured" % stats["calls"], flush=True)
        return result

    policy.infer = infer
    policy.metadata["route_recorder"] = "himoe_router_recorder"
    policy.metadata["route_recorder_gates"] = len(gates)

    def flush(*_):
        writer.close()
        if states:
            np.save(out / "state.npy", np.stack(states))
            (out / "prompts.json").write_text(json.dumps(prompts))
        (out / "capture_summary.json").write_text(json.dumps({
            "control_steps": stats["calls"],
            "as_collapsed": stats["as_collapsed"],
            "hook_verified_calls": recorder.verified_calls,
            "hook_verify_failures": recorder.verify_failures,
            "state_dim": int(states[0].shape[0]) if states else None,
            "n_denoise": n_denoise, "n_suffix": n_suffix,
            "dataset_config": "calvin_d_joint",
        }, indent=2))
        print("flushed %d control steps to %s" % (stats["calls"], out), flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, flush)
    signal.signal(signal.SIGINT, flush)

    from moevla.serving.websocket_policy_server import WebsocketPolicyServer
    print("serving on ws://%s:%d" % (args.host, args.port), flush=True)
    try:
        WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                              metadata=policy.metadata).serve_forever()
    finally:
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
