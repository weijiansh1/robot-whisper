"""Serve paired baseline/drop/random interventions and write independent v4 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--suite", default="goal")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--libero-wrist-layout", default="checkpoint-right")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-queries", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0, help="only used by --gpu cpu")
    parser.add_argument(
        "--drop-policy",
        default="categorical_weighted",
        help="v4 smoke is preregistered with categorical_weighted",
    )
    parser.add_argument(
        "--noop-audit-first-baseline",
        action="store_true",
        help="repeat the first baseline with capture disabled and require exact output",
    )
    args = parser.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    if args.gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        from probe_flow_lead_cpu import install_cpu_autocast

        install_cpu_autocast(torch)
        if args.threads:
            torch.set_num_threads(args.threads)
        from himoe_libero_bridge.policies import HiMoEPolicy

        policy = HiMoEPolicy(
            checkpoint_dir=args.checkpoint_dir,
            suite=args.suite,
            upstream_root=args.upstream_root,
            require_cuda=False,
            libero_wrist_layout=args.libero_wrist_layout,
        )
        print("running on CPU with %d threads" % torch.get_num_threads(), flush=True)
    else:
        policy = create_policy(
            "himoe",
            args.checkpoint_dir,
            args.upstream_root,
            args.gpu,
            args.suite,
            args.libero_wrist_layout,
        )

    from himoe_activation_recorder import HBActivationRMSRecorder
    from himoe_hb5_intervention import DROP_POLICIES, HB5D0Intervention
    from himoe_intervention_protocol import (
        OnlinePairAudit,
        pop_intervention_identity,
        request_digests,
    )
    from himoe_intervention_store import (
        FORMAT,
        STORE_NAME,
        ZarrHB5InterventionWriter,
    )
    from serve_flow_trace import FlowTracer

    if args.drop_policy not in DROP_POLICIES:
        raise ValueError("--drop-policy must be one of %s" % (DROP_POLICIES,))
    core = policy._policy.model
    cfg = core.config
    n_denoise = int(cfg.num_steps)
    n_action_steps = int(cfg.n_action_steps)
    max_action_dim = int(cfg.max_action_dim)
    action_std = np.asarray(
        policy.metadata.get("normalization_action_std"), dtype=np.float64
    )
    intervention = HB5D0Intervention(drop_policy=args.drop_policy)
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=n_action_steps,
        expected_denoise=n_denoise,
        probe_layer=5,
        intervention=intervention,
    ).attach()
    tracer = FlowTracer(core)
    pair_audit = OnlinePairAudit()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ZarrHB5InterventionWriter(
        str(out / STORE_NAME),
        n_denoise=n_denoise,
        n_action_steps=n_action_steps,
        max_action_dim=max_action_dim,
        hidden_size=recorder.probe_hidden_size,
        top_k=recorder.top_k,
        n_routed_experts=recorder.n_routed_experts,
        action_std=action_std,
        drop_policy=args.drop_policy,
        chunk_queries=args.chunk_queries,
    )

    inner = policy.infer
    stats: dict[str, object] = {
        "calls": 0,
        "infer_s": 0.0,
        "capture_s": 0.0,
        "arm_calls": {"baseline": 0, "drop": 0, "random": 0},
        "max_euler_residual": 0.0,
        "max_probe_routed_reconstruction_error": 0.0,
        "noop_audit_requested": bool(args.noop_audit_first_baseline),
        "noop_audit_complete": False,
        "noop_action_max_abs_error": None,
        "noop_x_traj_max_abs_error": None,
    }

    def infer(observation):
        observation_digest, noise_digest = request_digests(observation)
        pair_id, draw_id, arm, query_id, candidate_id = pop_intervention_identity(
            observation
        )
        episode_id = int(observation.pop("episode_id", -1))
        pair_audit.prepare(pair_id, draw_id, arm, observation_digest, noise_digest)
        noop_response = None
        noop_x_traj = None
        if (
            args.noop_audit_first_baseline
            and arm == "baseline"
            and not bool(stats["noop_audit_complete"])
        ):
            noop_observation = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
            tracer.begin()
            noop_response = inner(noop_observation)
            noop_x_traj = np.moveaxis(tracer.trajectory(), 1, 0)
        recorder.begin(
            episode_id=episode_id,
            control_step=int(stats["calls"]),
            query_id=query_id,
            candidate_id=candidate_id,
            intervention_pair_id=pair_id,
            intervention_draw_id=draw_id,
            intervention_arm=arm,
        )
        tracer.begin()
        started = time.perf_counter()
        try:
            response = inner(observation)
        except BaseException:
            recorder.cancel()
            tracer.begin()
            raise
        after_infer = time.perf_counter()
        record = recorder.end()
        residual = tracer.euler_residual()
        if residual > 0.05:
            raise RuntimeError(
                "Euler residual %.3e is too large for bf16 velocity rounding" % residual
            )
        x_traj = np.moveaxis(tracer.trajectory(), 1, 0)
        if noop_response is not None:
            action_error = float(
                np.max(
                    np.abs(
                        np.asarray(response["actions"], dtype=np.float32)
                        - np.asarray(noop_response["actions"], dtype=np.float32)
                    )
                )
            )
            trajectory_error = float(
                np.max(
                    np.abs(
                        x_traj.astype(np.float64)
                        - np.asarray(noop_x_traj, dtype=np.float64)
                    )
                )
            )
            if action_error != 0.0 or trajectory_error != 0.0:
                raise RuntimeError(
                    "baseline instrumentation changed output: action %.3e, x_traj %.3e"
                    % (action_error, trajectory_error)
                )
            stats["noop_audit_complete"] = True
            stats["noop_action_max_abs_error"] = action_error
            stats["noop_x_traj_max_abs_error"] = trajectory_error
        pair_audit.commit(
            record,
            x_traj,
            observation_digest=observation_digest,
            noise_digest=noise_digest,
        )
        writer.append(
            record,
            x_traj,
            observation_sha256=observation_digest,
            flow_noise_sha256=noise_digest,
        )
        after_capture = time.perf_counter()

        stats["calls"] = int(stats["calls"]) + 1
        stats["infer_s"] = float(stats["infer_s"]) + after_infer - started
        stats["capture_s"] = float(stats["capture_s"]) + after_capture - after_infer
        arm_calls = stats["arm_calls"]
        arm_calls[arm] += 1
        stats["max_euler_residual"] = max(float(stats["max_euler_residual"]), residual)
        stats["max_probe_routed_reconstruction_error"] = max(
            float(stats["max_probe_routed_reconstruction_error"]),
            float(np.max(record.hb_probe_routed_reconstruction_max_abs_error)),
        )
        print(
            "pair=%d draw=%d arm=%s captured (infer %.0f ms)"
            % (pair_id, draw_id, arm, 1000 * (after_infer - started)),
            flush=True,
        )
        return response

    policy.infer = infer
    policy.metadata.update(
        {
            "hb5_intervention_recorder": FORMAT,
            "hb5_intervention_store": STORE_NAME,
            "hb5_intervention_pair_key": "intervention/pair_id",
            "hb5_intervention_draw_key": "intervention/draw_id",
            "hb5_intervention_arm_key": "intervention/arm",
            "hb5_intervention_arms": ["baseline", "drop", "random"],
            "hb5_intervention_drop_policy": args.drop_policy,
            "hb5_intervention_layer": 5,
            "hb5_intervention_denoise": 0,
            "hb5_intervention_tokens": "10 action tokens only; state token unchanged",
            "hb5_intervention_requires_explicit_flow_noise": True,
            "hb5_intervention_noop_audit": bool(args.noop_audit_first_baseline),
            "flow_tracer": "serve_flow_trace",
            "num_steps": n_denoise,
            "n_action_steps": n_action_steps,
        }
    )

    closed = {"value": False}

    def shutdown() -> None:
        if closed["value"]:
            return
        closed["value"] = True
        writer.close()
        recorder.close()
        tracer.close()
        pair_summary = pair_audit.summary()
        calls = int(stats["calls"])
        summary = {
            "format": FORMAT,
            "queries": calls,
            "drop_policy": args.drop_policy,
            "mean_infer_ms": 1000 * float(stats["infer_s"]) / max(1, calls),
            "mean_capture_ms": 1000 * float(stats["capture_s"]) / max(1, calls),
            "arm_calls": stats["arm_calls"],
            "max_euler_residual": stats["max_euler_residual"],
            "max_probe_routed_reconstruction_error": stats[
                "max_probe_routed_reconstruction_error"
            ],
            "noop_audit_requested": stats["noop_audit_requested"],
            "noop_audit_complete": stats["noop_audit_complete"],
            "noop_action_max_abs_error": stats["noop_action_max_abs_error"],
            "noop_x_traj_max_abs_error": stats["noop_x_traj_max_abs_error"],
            **pair_summary,
            "paired_audit_complete": pair_summary["incomplete_pairs"] == 0,
        }
        (out / "capture_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + json.dumps(summary, indent=2), flush=True)

    def stop(*_args):
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        "serving on ws://%s:%d; paired HB5/d0 policy=%s"
        % (args.host, args.port, args.drop_policy),
        flush=True,
    )
    try:
        PolicyServer(
            policy, args.host, args.port, getattr(policy, "backend_name", "himoe")
        ).serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
