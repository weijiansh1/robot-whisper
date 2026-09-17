"""Command line interface for bridge setup and acceptance runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

from himoe_libero_bridge.batch import (
    SUITE_MAX_STEPS,
    VIDEO_POLICIES,
    BatchConfig,
    collect_source_identity,
    load_batch_summary,
    parse_id_spec,
    resume_batch,
    run_batch,
)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.doctor import run_doctor
from himoe_libero_bridge.libero_runtime import (
    EpisodeConfig,
    run_determinism_check,
    run_environment_check,
    run_episode,
    run_episode_replay,
    run_routing_probe,
)
from himoe_libero_bridge.protocol import IMAGE_SHAPE, STATE_DIM
from himoe_libero_bridge.rad_batch import (
    RadBatchConfig,
    load_rad_batch_summary,
    resume_rad_batch,
    run_rad_batch,
)
from himoe_libero_bridge.rad_experiment import PAIRED_SELECTORS
from himoe_libero_bridge.rad_runtime import (
    RadBlockConfig,
    RadEpisodeConfig,
    collect_rad_experiment_identity,
    run_rad_block,
    run_rad_episode,
)
from himoe_libero_bridge.policies import LIBERO_WRIST_LAYOUTS
from himoe_libero_bridge.server import PolicyServer, create_policy
from himoe_libero_bridge.suites import SUITE_NAMES, get_suite

DEFAULT_CACHE = "/home/jovyan/.cache/himoe-libero-bridge"
DEFAULT_LIBERO_ROOT = DEFAULT_CACHE + "/upstream/LIBERO"
DEFAULT_HIMOE_ROOT = DEFAULT_CACHE + "/upstream/HiMoE-VLA"


def _add_connection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def _episode_config(args: argparse.Namespace) -> EpisodeConfig:
    return EpisodeConfig(
        libero_root=args.libero_root,
        output_root=args.output_root,
        host=args.host,
        port=args.port,
        task_suite=get_suite(args.suite).benchmark,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        settle_steps=args.settle_steps,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        render_size=args.render_size,
        fps=args.fps,
        inference_timeout=args.inference_timeout,
        flow_noise_seed=getattr(args, "flow_noise_seed", None),
    )


def _rad_episode_config(args: argparse.Namespace) -> RadEpisodeConfig:
    return RadEpisodeConfig(
        libero_root=args.libero_root,
        output_root=args.output_root,
        host=args.host,
        port=args.port,
        task_suite=get_suite(args.suite).benchmark,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        settle_steps=args.settle_steps,
        max_steps=(
            args.max_steps
            if args.max_steps is not None
            else SUITE_MAX_STEPS[args.suite]
        ),
        replan_steps=args.replan_steps,
        render_size=args.render_size,
        fps=args.fps,
        inference_timeout=args.inference_timeout,
        selector=args.selector,
        noise_seed=args.noise_seed,
        random_selector_seed=args.random_selector_seed,
    )


def _add_rad_episode_arguments(
    parser: argparse.ArgumentParser, *, include_selector: bool = True
) -> None:
    _add_connection(parser)
    parser.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    parser.add_argument(
        "--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT)
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Suite default: Goal 300, Spatial 220, Object 280",
    )
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--inference-timeout", type=float, default=180.0)
    if include_selector:
        parser.add_argument("--selector", choices=PAIRED_SELECTORS, required=True)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--random-selector-seed", type=int, default=1042)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run a mock or HiMoE policy server")
    serve.add_argument("--backend", choices=("mock", "himoe"), default="mock")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--gpu", default="0")
    serve.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    serve.add_argument("--checkpoint-dir", default=os.environ.get("HIMOE_CHECKPOINT"))
    serve.add_argument("--upstream-root", default=os.environ.get("HIMOE_ROOT", DEFAULT_HIMOE_ROOT))
    serve.add_argument(
        "--libero-wrist-layout",
        choices=LIBERO_WRIST_LAYOUTS,
        default="released-left",
        help="Controlled LIBERO camera-slot A/B; the released implementation is the default",
    )

    run = subparsers.add_parser("run", help="Run one LIBERO task and one episode")
    _add_connection(run)
    run.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    run.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT))
    run.add_argument("--output-root", default="artifacts")
    run.add_argument("--task-id", type=int, default=0)
    run.add_argument("--init-state-id", type=int, default=0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--settle-steps", type=int, default=10)
    run.add_argument("--max-steps", type=int, default=300)
    run.add_argument("--replan-steps", type=int, default=10)
    run.add_argument("--render-size", type=int, default=512)
    run.add_argument("--fps", type=int, default=20)
    run.add_argument("--inference-timeout", type=float, default=180.0)
    run.add_argument(
        "--flow-noise-seed",
        type=int,
        help="Use explicit deterministic flow noise and save an exact episode trace",
    )

    rad_run = subparsers.add_parser(
        "rad-run", help="Run one predeclared closed-loop RAD selector arm"
    )
    _add_rad_episode_arguments(rad_run)
    rad_run.add_argument("--output-root", default="artifacts")

    rad_block = subparsers.add_parser(
        "rad-block", help="Run or resume one paired four-arm RAD block"
    )
    _add_rad_episode_arguments(rad_block, include_selector=False)
    rad_block.add_argument("--block-dir", required=True)
    rad_block.add_argument("--arm-order-seed", type=int, default=2042)

    rad_batch = subparsers.add_parser(
        "rad-batch", help="Run or resume a paired multi-case four-arm RAD batch"
    )
    rad_batch.add_argument(
        "--resume",
        help="Existing RAD batch directory (all experiment options are immutable)",
    )
    rad_batch.add_argument("--batch-dir")
    rad_batch.add_argument("--host")
    rad_batch.add_argument("--port", type=int)
    rad_batch.add_argument("--suite", choices=SUITE_NAMES)
    rad_batch.add_argument("--libero-root")
    rad_batch.add_argument(
        "--task-ids",
        metavar="IDS",
        help="Task IDs as comma/ranges or 'all'",
    )
    rad_batch.add_argument(
        "--init-state-ids",
        metavar="IDS",
        help="Initial-state IDs as comma/ranges (for example 0-4)",
    )
    rad_batch.add_argument("--env-seed", type=int)
    rad_batch.add_argument("--master-seed", type=int)
    rad_batch.add_argument("--settle-steps", type=int)
    rad_batch.add_argument("--max-steps", type=int)
    rad_batch.add_argument("--replan-steps", type=int)
    rad_batch.add_argument("--render-size", type=int)
    rad_batch.add_argument("--fps", type=int)
    rad_batch.add_argument("--inference-timeout", type=float)
    rad_batch.add_argument("--bootstrap-samples", type=int)
    rad_batch.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, retry infrastructure-failed or fairness-incomplete cases",
    )
    rad_batch.add_argument(
        "--max-new-cases",
        type=int,
        help="Run at most this many pending paired cases in this invocation",
    )

    batch = subparsers.add_parser(
        "batch", help="Run or resume an auditable sequence of LIBERO episodes"
    )
    batch.add_argument(
        "--resume",
        help="Existing batch directory (schedule options are immutable when resuming)",
    )
    batch.add_argument("--host", help="Policy host; may be overridden when resuming")
    batch.add_argument("--port", type=int, help="Policy port; may be overridden when resuming")
    batch.add_argument(
        "--inference-timeout",
        type=float,
        help="Inference timeout; may be overridden when resuming",
    )
    batch.add_argument("--suite", choices=SUITE_NAMES)
    batch.add_argument("--libero-root")
    batch.add_argument("--output-root")
    batch.add_argument(
        "--task-ids",
        metavar="IDS",
        help="Task IDs as comma/ranges (for example 0,2,4-6) or 'all'",
    )
    batch.add_argument(
        "--init-state-ids",
        metavar="IDS",
        help="Initial-state IDs as comma/ranges; cycled in episode order",
    )
    batch.add_argument(
        "--episodes-per-task",
        "--episode-count",
        dest="episodes_per_task",
        type=int,
    )
    batch.add_argument(
        "--start-seed",
        type=int,
        help="Fixed LIBERO environment seed used for every episode",
    )
    batch.add_argument(
        "--flow-noise-seed-start",
        type=int,
        help="Optional task-local explicit flow-noise seed sequence",
    )
    batch.add_argument("--settle-steps", type=int)
    batch.add_argument("--max-steps", type=int)
    batch.add_argument("--replan-steps", type=int)
    batch.add_argument("--render-size", type=int)
    batch.add_argument("--fps", type=int)
    batch.add_argument("--video-policy", choices=VIDEO_POLICIES)
    batch.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, retry infrastructure-failed episodes",
    )
    batch.add_argument(
        "--max-new-episodes",
        type=int,
        help="Run at most this many pending episodes in this invocation",
    )

    replay = subparsers.add_parser(
        "replay-episode", help="Replay and verify a recorded episode trace"
    )
    _add_connection(replay)
    replay.add_argument("trace", help="Episode artifact directory, trace JSON, or trace NPZ")
    replay.add_argument(
        "--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT)
    )
    replay.add_argument("--output-root", default="artifacts")
    replay.add_argument("--inference-timeout", type=float, default=180.0)
    replay.add_argument(
        "--require-server-restart",
        action="store_true",
        help="Reject replay against the same model-server process used for capture",
    )

    env_check = subparsers.add_parser("env-check", help="Create, render and step LIBERO without a model")
    _add_connection(env_check)
    env_check.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    env_check.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT))
    env_check.add_argument("--output-root", default="artifacts")
    env_check.add_argument("--task-id", type=int, default=0)
    env_check.add_argument("--init-state-id", type=int, default=0)
    env_check.add_argument("--seed", type=int, default=7)
    env_check.add_argument("--settle-steps", type=int, default=0)
    env_check.add_argument("--max-steps", type=int, default=10)
    env_check.add_argument("--replan-steps", type=int, default=10)
    env_check.add_argument("--render-size", type=int, default=512)
    env_check.add_argument("--fps", type=int, default=20)
    env_check.add_argument("--inference-timeout", type=float, default=180.0)
    env_check.add_argument("--action-steps", type=int, default=10)

    determinism = subparsers.add_parser(
        "determinism-check", help="Run one fixed LIBERO observation and flow noise twice"
    )
    _add_connection(determinism)
    determinism.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    determinism.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT))
    determinism.add_argument("--output-root", default="artifacts")
    determinism.add_argument("--task-id", type=int, default=0)
    determinism.add_argument("--init-state-id", type=int, default=0)
    determinism.add_argument("--seed", type=int, default=7)
    determinism.add_argument("--noise-seed", type=int, default=42)
    determinism.add_argument("--settle-steps", type=int, default=10)
    determinism.add_argument("--max-steps", type=int, default=300)
    determinism.add_argument("--replan-steps", type=int, default=10)
    determinism.add_argument("--render-size", type=int, default=512)
    determinism.add_argument("--fps", type=int, default=20)
    determinism.add_argument("--inference-timeout", type=float, default=180.0)

    routing_probe = subparsers.add_parser(
        "routing-probe", help="Generate fixed-noise candidates and capture HB-MoE routes"
    )
    _add_connection(routing_probe)
    routing_probe.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    routing_probe.add_argument(
        "--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT)
    )
    routing_probe.add_argument("--output-root", default="artifacts")
    routing_probe.add_argument("--task-id", type=int, default=0)
    routing_probe.add_argument("--init-state-id", type=int, default=0)
    routing_probe.add_argument("--seed", type=int, default=7)
    routing_probe.add_argument("--noise-seed", type=int, default=42)
    routing_probe.add_argument("--candidate-count", type=int, default=8)
    routing_probe.add_argument("--rad-neighbors", type=int, default=2)
    routing_probe.add_argument("--settle-steps", type=int, default=10)
    routing_probe.add_argument("--max-steps", type=int, default=300)
    routing_probe.add_argument("--replan-steps", type=int, default=10)
    routing_probe.add_argument("--render-size", type=int, default=512)
    routing_probe.add_argument("--fps", type=int, default=20)
    routing_probe.add_argument("--inference-timeout", type=float, default=180.0)

    protocol_check = subparsers.add_parser("protocol-check", help="Send one synthetic request to a policy server")
    _add_connection(protocol_check)
    protocol_check.add_argument("--timeout", type=float, default=30.0)
    protocol_check.add_argument("--expected-suite", choices=SUITE_NAMES)

    doctor = subparsers.add_parser("doctor", help="Report runtime prerequisites")
    doctor.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    doctor.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT))
    doctor.add_argument("--upstream-root", default=os.environ.get("HIMOE_ROOT", DEFAULT_HIMOE_ROOT))
    doctor.add_argument("--checkpoint-dir", default=os.environ.get("HIMOE_CHECKPOINT"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if args.command == "serve":
        checkpoint_dir = args.checkpoint_dir or str(get_suite(args.suite).checkpoint_dir(DEFAULT_CACHE))
        policy = create_policy(
            args.backend,
            checkpoint_dir,
            args.upstream_root,
            args.gpu,
            args.suite,
            args.libero_wrist_layout,
        )
        PolicyServer(policy, args.host, args.port, getattr(policy, "backend_name", args.backend)).serve_forever()
        return 0
    if args.command == "run":
        artifact_dir = run_episode(_episode_config(args))
        print(artifact_dir)
        return 0
    if args.command == "rad-run":
        artifact_dir = run_rad_episode(
            _rad_episode_config(args),
            bridge_source_identity=collect_source_identity(),
        )
        print(artifact_dir)
        return 0
    if args.command == "rad-block":
        artifact_dir = run_rad_block(
            RadBlockConfig(
                block_dir=args.block_dir,
                libero_root=args.libero_root,
                host=args.host,
                port=args.port,
                task_suite=get_suite(args.suite).benchmark,
                task_id=args.task_id,
                init_state_id=args.init_state_id,
                seed=args.seed,
                settle_steps=args.settle_steps,
                max_steps=(
                    args.max_steps
                    if args.max_steps is not None
                    else SUITE_MAX_STEPS[args.suite]
                ),
                replan_steps=args.replan_steps,
                render_size=args.render_size,
                fps=args.fps,
                inference_timeout=args.inference_timeout,
                noise_seed=args.noise_seed,
                random_selector_seed=args.random_selector_seed,
                arm_order_seed=args.arm_order_seed,
            ),
            experiment_identity_provider=lambda block_config: (
                collect_rad_experiment_identity(
                    block_config, collect_source_identity()
                )
            ),
        )
        print(artifact_dir)
        return 0
    if args.command == "rad-batch":
        if args.resume:
            immutable_options = {
                "batch_dir": args.batch_dir,
                "host": args.host,
                "port": args.port,
                "suite": args.suite,
                "libero_root": args.libero_root,
                "task_ids": args.task_ids,
                "init_state_ids": args.init_state_ids,
                "env_seed": args.env_seed,
                "master_seed": args.master_seed,
                "settle_steps": args.settle_steps,
                "max_steps": args.max_steps,
                "replan_steps": args.replan_steps,
                "render_size": args.render_size,
                "fps": args.fps,
                "inference_timeout": args.inference_timeout,
                "bootstrap_samples": args.bootstrap_samples,
            }
            supplied = sorted(
                key for key, value in immutable_options.items() if value is not None
            )
            if supplied:
                parser.error(
                    "--resume cannot change immutable options: %s"
                    % ", ".join(supplied)
                )
            artifact_dir = resume_rad_batch(
                args.resume,
                retry_failed=args.retry_failed,
                max_new_cases=args.max_new_cases,
            )
        else:
            if args.retry_failed:
                parser.error("--retry-failed requires --resume")
            if not args.batch_dir:
                parser.error("--batch-dir is required for a new RAD batch")
            suite = args.suite or "goal"
            artifact_dir = run_rad_batch(
                RadBatchConfig(
                    batch_dir=args.batch_dir,
                    libero_root=args.libero_root
                    or os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT),
                    host=args.host or "127.0.0.1",
                    port=8000 if args.port is None else args.port,
                    task_suite=get_suite(suite).benchmark,
                    task_ids=parse_id_spec(args.task_ids or "0", allow_all=True),
                    init_state_ids=parse_id_spec(args.init_state_ids or "0"),
                    env_seed=7 if args.env_seed is None else args.env_seed,
                    master_seed=42 if args.master_seed is None else args.master_seed,
                    settle_steps=(
                        10 if args.settle_steps is None else args.settle_steps
                    ),
                    max_steps=(
                        SUITE_MAX_STEPS[suite]
                        if args.max_steps is None
                        else args.max_steps
                    ),
                    replan_steps=(
                        10 if args.replan_steps is None else args.replan_steps
                    ),
                    render_size=(
                        256 if args.render_size is None else args.render_size
                    ),
                    fps=20 if args.fps is None else args.fps,
                    inference_timeout=(
                        180.0
                        if args.inference_timeout is None
                        else args.inference_timeout
                    ),
                    bootstrap_samples=(
                        10_000
                        if args.bootstrap_samples is None
                        else args.bootstrap_samples
                    ),
                ),
                max_new_cases=args.max_new_cases,
            )
        summary = load_rad_batch_summary(str(artifact_dir))
        print(artifact_dir)
        return 1 if (
            summary.get("infra_failed_cases", 0)
            or summary.get("incomplete_cases", 0)
        ) else 0
    if args.command == "batch":
        if args.resume:
            immutable_options = {
                "suite": args.suite,
                "libero_root": args.libero_root,
                "output_root": args.output_root,
                "task_ids": args.task_ids,
                "init_state_ids": args.init_state_ids,
                "episodes_per_task": args.episodes_per_task,
                "start_seed": args.start_seed,
                "flow_noise_seed_start": args.flow_noise_seed_start,
                "settle_steps": args.settle_steps,
                "max_steps": args.max_steps,
                "replan_steps": args.replan_steps,
                "render_size": args.render_size,
                "fps": args.fps,
                "video_policy": args.video_policy,
            }
            supplied = sorted(key for key, value in immutable_options.items() if value is not None)
            if supplied:
                parser.error(
                    "--resume cannot change immutable options: %s" % ", ".join(supplied)
                )
            artifact_dir = resume_batch(
                args.resume,
                host=args.host,
                port=args.port,
                inference_timeout=args.inference_timeout,
                retry_failed=args.retry_failed,
                max_new_episodes=args.max_new_episodes,
            )
        else:
            if args.retry_failed:
                parser.error("--retry-failed requires --resume")
            suite = args.suite or "goal"
            config = BatchConfig(
                libero_root=args.libero_root
                or os.environ.get("LIBERO_ROOT", DEFAULT_LIBERO_ROOT),
                output_root=args.output_root or "artifacts",
                host=args.host or "127.0.0.1",
                port=8000 if args.port is None else args.port,
                suite=suite,
                task_ids=parse_id_spec(args.task_ids or "0", allow_all=True),
                init_state_ids=parse_id_spec(args.init_state_ids or "0"),
                episodes_per_task=args.episodes_per_task or 1,
                start_seed=7 if args.start_seed is None else args.start_seed,
                flow_noise_seed_start=args.flow_noise_seed_start,
                settle_steps=10 if args.settle_steps is None else args.settle_steps,
                max_steps=(
                    SUITE_MAX_STEPS[suite]
                    if args.max_steps is None
                    else args.max_steps
                ),
                replan_steps=10 if args.replan_steps is None else args.replan_steps,
                render_size=256 if args.render_size is None else args.render_size,
                fps=20 if args.fps is None else args.fps,
                inference_timeout=(
                    180.0 if args.inference_timeout is None else args.inference_timeout
                ),
                video_policy=args.video_policy or "all",
            )
            artifact_dir = run_batch(
                config,
                max_new_episodes=args.max_new_episodes,
            )
        summary = load_batch_summary(str(artifact_dir))
        print(artifact_dir)
        return 1 if summary.get("failed_episodes", 0) else 0
    if args.command == "replay-episode":
        artifact_dir = run_episode_replay(
            args.trace,
            libero_root=args.libero_root,
            output_root=args.output_root,
            host=args.host,
            port=args.port,
            inference_timeout=args.inference_timeout,
            require_server_restart=args.require_server_restart,
        )
        print(artifact_dir)
        return 0
    if args.command == "env-check":
        artifact_dir = run_environment_check(_episode_config(args), action_steps=args.action_steps)
        print(artifact_dir)
        return 0
    if args.command == "determinism-check":
        artifact_dir = run_determinism_check(_episode_config(args), noise_seed=args.noise_seed)
        print(artifact_dir)
        return 0
    if args.command == "routing-probe":
        artifact_dir = run_routing_probe(
            _episode_config(args),
            candidate_count=args.candidate_count,
            noise_seed=args.noise_seed,
            rad_neighbors=args.rad_neighbors,
        )
        print(artifact_dir)
        return 0
    if args.command == "protocol-check":
        observation = {
            "observation/image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            "observation/wrist_image": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            "observation/state": np.zeros((STATE_DIM,), dtype=np.float32),
            "prompt": "move the robot for a protocol check",
        }
        with PolicyClient(args.host, args.port, args.timeout, args.timeout) as client:
            if args.expected_suite and client.metadata.get("suite") != args.expected_suite:
                raise RuntimeError(
                    "Policy suite mismatch: expected %s, server advertises %r"
                    % (args.expected_suite, client.metadata.get("suite"))
                )
            response = client.infer(observation)
            print(json.dumps({"metadata": client.metadata, "actions_shape": list(response["actions"].shape)}, indent=2))
        return 0
    if args.command == "doctor":
        checkpoint_dir = args.checkpoint_dir or str(get_suite(args.suite).checkpoint_dir(DEFAULT_CACHE))
        return run_doctor(args.libero_root, args.upstream_root, checkpoint_dir, args.suite)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
