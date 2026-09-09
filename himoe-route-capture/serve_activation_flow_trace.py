"""Serve HiMoE and jointly capture x_traj plus true HB expert amplitudes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")

QUERY_KEY = "flow/query_id"
CANDIDATE_KEY = "flow/candidate_id"


def request_control_step(
    episode_id: int,
    candidate_id: int | None,
    global_call: int,
    per_episode: dict[int, int],
) -> int:
    """Use candidate ids for K pools, otherwise count within each episode."""
    if candidate_id is not None:
        return int(candidate_id)
    if episode_id < 0:
        return int(global_call)
    control_step = per_episode.get(episode_id, 0)
    per_episode[episode_id] = control_step + 1
    return control_step


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
    parser.add_argument("--chunk-queries", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0, help="only used by --gpu cpu")
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
    from himoe_activation_store import FORMAT, ZarrActivationFlowWriter
    from serve_flow_trace import FlowTracer

    core = policy._policy.model
    cfg = core.config
    n_denoise = int(cfg.num_steps)
    n_action_steps = int(cfg.n_action_steps)
    max_action_dim = int(cfg.max_action_dim)
    action_std = np.asarray(
        policy.metadata.get("normalization_action_std"), dtype=np.float64
    )
    if action_std.shape != (7,) or np.any(~np.isfinite(action_std)) or np.any(action_std <= 0):
        raise RuntimeError("policy metadata has invalid normalization_action_std")
    recorder = HBActivationRMSRecorder(
        core,
        n_action_steps=n_action_steps,
        expected_denoise=n_denoise,
    ).attach()
    tracer = FlowTracer(core)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ZarrActivationFlowWriter(
        str(out / "activation_flow.zarr"),
        hb_layers=recorder.hb_layers,
        n_denoise=n_denoise,
        n_action_steps=n_action_steps,
        max_action_dim=max_action_dim,
        action_std=action_std,
        top_k=recorder.top_k,
        chunk_queries=args.chunk_queries,
        probe_hb_layer=recorder.probe_layer,
        probe_hidden_size=recorder.probe_hidden_size,
        n_routed_experts=recorder.n_routed_experts,
    )

    inner = policy.infer
    stats = {
        "calls": 0,
        "infer_s": 0.0,
        "capture_s": 0.0,
        "max_euler_residual": 0.0,
        "max_topk_weight_sum_error": 0.0,
        "max_probe_routed_reconstruction_error": 0.0,
    }
    per_episode_control: dict[int, int] = {}

    def infer(observation):
        # K-candidate drivers supply flow/query_id and flow/candidate_id.  Normal
        # rollout drivers supply episode_id; the append index remains an unambiguous
        # global control-step key in either mode.
        query = observation.pop(QUERY_KEY, None)
        episode = int(observation.pop("episode_id", -1))
        # flow/query_id identifies a same-state candidate pool, not a complete
        # rollout.  Without an explicit episode marker it cannot license LOEO.
        candidate = observation.pop(CANDIDATE_KEY, None)
        control = request_control_step(
            episode, candidate, stats["calls"], per_episode_control
        )
        recorder.begin(
            episode_id=episode,
            control_step=control,
            query_id=-1 if query is None else int(query),
            candidate_id=-1 if candidate is None else int(candidate),
        )
        tracer.begin()
        start = time.perf_counter()
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
        # Existing FlowTracer returns [D+1,B,A,H]; the store is batch-first.
        x_traj = np.moveaxis(tracer.trajectory(), 1, 0)
        writer.append(record, x_traj)
        after_capture = time.perf_counter()

        stats["calls"] += 1
        stats["infer_s"] += after_infer - start
        stats["capture_s"] += after_capture - after_infer
        stats["max_euler_residual"] = max(stats["max_euler_residual"], residual)
        stats["max_topk_weight_sum_error"] = max(
            stats["max_topk_weight_sum_error"],
            float(np.max(record.hb_topk_weight_sum_error)),
        )
        stats["max_probe_routed_reconstruction_error"] = max(
            stats["max_probe_routed_reconstruction_error"],
            float(
                np.max(record.hb_probe_routed_reconstruction_max_abs_error)
            ),
        )
        if stats["calls"] % 50 == 0:
            print(
                "captured %d queries (infer %.0f ms, capture %.1f ms)"
                % (
                    stats["calls"],
                    1000 * stats["infer_s"] / stats["calls"],
                    1000 * stats["capture_s"] / stats["calls"],
                ),
                flush=True,
            )
        return response

    policy.infer = infer
    policy.metadata["activation_flow_recorder"] = FORMAT
    policy.metadata["flow_tracer"] = "serve_flow_trace"
    policy.metadata["activation_flow_query_key"] = QUERY_KEY
    policy.metadata["activation_flow_candidate_key"] = CANDIDATE_KEY
    policy.metadata["activation_flow_store"] = "activation_flow.zarr"
    policy.metadata["episode_id_key"] = "episode_id"
    policy.metadata["activation_loeo_group_key"] = "episode_id"
    policy.metadata["activation_flow_query_id"] = (
        "request flow/query_id when present; stored explicitly, -1 when absent"
    )
    policy.metadata["activation_flow_candidate_id"] = (
        "request flow/candidate_id when present; stored explicitly, -1 when absent"
    )
    policy.metadata["activation_definition"] = (
        "weighted routed RMS plus per-selected-expert pre-gate raw RMS; "
        "HB5/d0 actual pre-gate vector probe; action tokens only"
    )
    policy.metadata["num_steps"] = n_denoise
    policy.metadata["n_action_steps"] = n_action_steps
    policy.metadata["hb_layers"] = recorder.hb_layers
    policy.metadata["hb_top_k"] = recorder.top_k
    policy.metadata["activation_probe_hb_layer"] = recorder.probe_layer
    policy.metadata["activation_probe_denoise"] = recorder.probe_denoise
    policy.metadata["activation_probe_hidden_size"] = recorder.probe_hidden_size
    policy.metadata["activation_probe_n_routed_experts"] = (
        recorder.n_routed_experts
    )

    done = {"value": False}

    def shutdown() -> None:
        if done["value"]:
            return
        done["value"] = True
        writer.close()
        recorder.close()
        tracer.close()
        summary = {
            "format": FORMAT,
            "queries": stats["calls"],
            "mean_infer_ms": 1000 * stats["infer_s"] / max(1, stats["calls"]),
            "mean_capture_ms": 1000 * stats["capture_s"] / max(1, stats["calls"]),
            "max_euler_residual": stats["max_euler_residual"],
            "max_topk_weight_sum_error": stats["max_topk_weight_sum_error"],
            "max_probe_routed_reconstruction_error": stats[
                "max_probe_routed_reconstruction_error"
            ],
            "n_denoise": n_denoise,
            "n_action_steps": n_action_steps,
            "max_action_dim": max_action_dim,
            "normalization_action_std": action_std.tolist(),
            "hb_layers": recorder.hb_layers,
            "probe_hb_layer": recorder.probe_layer,
            "probe_denoise": recorder.probe_denoise,
            "probe_hidden_size": recorder.probe_hidden_size,
            "n_routed_experts": recorder.n_routed_experts,
        }
        (out / "capture_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + json.dumps(summary, indent=2), flush=True)

    import signal

    def stop(*_args):
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(
        "serving on ws://%s:%d; HB layers=%s"
        % (args.host, args.port, recorder.hb_layers),
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
