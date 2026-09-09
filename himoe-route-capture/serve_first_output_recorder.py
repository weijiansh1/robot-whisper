"""Serve HiMoE-VLA while capturing true MoE outputs at control step zero.

This server is wire-compatible with ``rollout_with_routes.py
--no-routing-capture``.  The client already stamps ``episode_id`` when the
server advertises support; the explicit scene/draw plan below converts that id
to the exact task/init-state/flow-seed identity used by right-16x32.

Example plan arguments for that corpus::

    --scene-ids 0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49 \
    --draws 32 --noise-seed-base 1000

The normal client still runs full episodes to obtain success labels, but hooks
and tensor transfers are enabled only for each episode's first policy call.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", required=True, choices=["goal", "spatial", "object", "long"])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="paper-right")
    ap.add_argument("--scene-ids", required=True)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--store-hb-vectors",
        action="store_true",
        help="persist full fp16 HB routed/shared vectors (~1.6 GiB per 16x32 task); "
             "without this they are reduced to exact norms/cosines/convergence",
    )
    args = ap.parse_args()

    from himoe_libero_bridge.protocol import FLOW_NOISE_KEY, FLOW_NOISE_SHAPE
    from himoe_libero_bridge.server import PolicyServer, create_policy
    from himoe_first_output_recorder import (
        FirstControlIdentity,
        FirstControlOutputRecorder,
        flow_noise_digest,
        seeded_flow_noise_digest,
    )
    from himoe_first_output_store import EpisodePlan, ZarrFirstOutputWriter, parse_scene_ids

    plan = EpisodePlan(
        scene_ids=parse_scene_ids(args.scene_ids),
        draws=args.draws,
        noise_seed_base=args.noise_seed_base,
    )
    policy = create_policy(
        "himoe",
        args.checkpoint_dir,
        args.upstream_root,
        args.gpu,
        args.suite,
        args.libero_wrist_layout,
    )
    core = policy._policy.model
    cfg = core.config
    n_action_steps = int(cfg.n_action_steps)
    n_denoise = int(cfg.num_steps)
    recorder = FirstControlOutputRecorder(
        core,
        n_action_steps=n_action_steps,
        expected_denoise=n_denoise,
        store_hb_vectors=args.store_hb_vectors,
    ).attach()
    hidden_size = int(recorder.hb[0].module.gate.weight.shape[1])

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ZarrFirstOutputWriter(
        str(out / "first_control_outputs.zarr"),
        expected_k=plan.draws,
        task_id=args.task_id,
        task_name=args.task_name,
        suite=args.suite,
        benchmark=args.benchmark,
        hb_layers=recorder.hb_layers,
        as_layers=recorder.as_layers,
        n_denoise=n_denoise,
        n_action_steps=n_action_steps,
        hidden_size=hidden_size,
        store_hb_vectors=args.store_hb_vectors,
    )

    stats = {
        "calls": 0,
        "captured": 0,
        "infer_s": 0.0,
        "capture_s": 0.0,
        "started_unix_ns": time.time_ns(),
    }
    episode_control: dict[int, int] = {}
    inner = policy.infer

    def infer(observation):
        if "episode_id" not in observation:
            raise RuntimeError(
                "this server requires episode_id; use rollout_with_routes.py, which "
                "sends it after reading the advertised episode_id_key"
            )
        episode_id = int(observation.pop("episode_id"))
        control_step = episode_control.get(episode_id, 0)
        episode_control[episode_id] = control_step + 1
        init_state_id, flow_seed, _repeat = plan.decode(episode_id)
        capture = control_step == 0
        if capture:
            if FLOW_NOISE_KEY not in observation:
                raise RuntimeError("first control request has no %r" % FLOW_NOISE_KEY)
            actual_digest = flow_noise_digest(observation[FLOW_NOISE_KEY])
            expected_digest = seeded_flow_noise_digest(flow_seed, FLOW_NOISE_SHAPE)
            if actual_digest != expected_digest:
                raise RuntimeError(
                    "episode %d first-row flow noise does not match decoded seed %d; "
                    "the declared scene/draw plan or episode ordering is wrong"
                    % (episode_id, flow_seed)
                )
            recorder.begin(
                FirstControlIdentity(
                    episode_id=episode_id,
                    task_id=args.task_id,
                    init_state_id=init_state_id,
                    flow_seed=flow_seed,
                    control_step=control_step,
                    flow_noise_sha256=actual_digest,
                )
            )
        t0 = time.perf_counter()
        try:
            response = inner(observation)
        except BaseException:
            if capture:
                recorder.cancel()
            raise
        t1 = time.perf_counter()
        if capture:
            writer.append(recorder.end())
        t2 = time.perf_counter()
        stats["calls"] += 1
        stats["captured"] += int(capture)
        stats["infer_s"] += t1 - t0
        stats["capture_s"] += t2 - t1
        if capture and stats["captured"] % plan.draws == 0:
            print(
                "captured %d/%d first rows (%d same-state groups)"
                % (stats["captured"], plan.episodes, writer.groups),
                flush=True,
            )
        return response

    policy.infer = infer
    policy.metadata["episode_id_key"] = "episode_id"
    policy.metadata["first_control_output_capture"] = True
    policy.metadata["first_control_output_format"] = "himoe_first_control_moe_output_v1"
    policy.metadata["first_control_output_task_id"] = args.task_id
    policy.metadata["first_control_output_task_name"] = args.task_name
    policy.metadata["first_control_output_scene_ids"] = list(plan.scene_ids)
    policy.metadata["first_control_output_draws"] = plan.draws
    policy.metadata["first_control_output_noise_seed_base"] = plan.noise_seed_base
    policy.metadata["n_action_steps"] = n_action_steps
    policy.metadata["num_steps"] = n_denoise

    closed = {"value": False}

    def shutdown() -> None:
        if closed["value"]:
            return
        closed["value"] = True
        writer.close()
        recorder.close()
        summary = {
            **stats,
            "candidate_rows": writer.rows,
            "groups": writer.groups,
            "incomplete_groups": writer.incomplete_groups,
            "expected_episodes": plan.episodes,
            "expected_k": plan.draws,
            "scene_ids": list(plan.scene_ids),
            "task_id": args.task_id,
            "task_name": args.task_name,
            "suite": args.suite,
            "benchmark": args.benchmark,
            "store_hb_vectors": bool(args.store_hb_vectors),
            "mean_infer_ms": 1000 * stats["infer_s"] / max(1, stats["calls"]),
            "mean_first_row_capture_ms": (
                1000 * stats["capture_s"] / max(1, stats["captured"])
            ),
        }
        (out / "first_control_output_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    def _bye(*_args):
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)
    print(
        "serving on ws://%s:%d; first-row outputs for %d scenes x K=%d"
        % (args.host, args.port, len(plan.scene_ids), plan.draws),
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
