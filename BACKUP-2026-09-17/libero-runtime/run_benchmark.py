"""Run one LIBERO-Plus or LIBERO-Pro episode against the HiMoE server."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
LONG_TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
PLUS_CAMERA_TASK = LONG_TASK + "_view_1_15_100_0_0_initstate_0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("plus", "pro"), required=True)
    parser.add_argument("--suite", choices=("long", "goal", "object", "spatial"), default="long")
    parser.add_argument("--perturbation", choices=("swap", "object", "lan", "task", "env", "none"), default="swap",
                        help="Pro perturbation; 'none' runs the unperturbed original suite from the Pro source")
    task_args = parser.add_mutually_exclusive_group()
    task_args.add_argument("--task-id", type=int)
    task_args.add_argument("--task-name")
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flow-noise-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    source = ROOT / "upstream" / ("LIBERO-plus" if args.benchmark == "plus" else "LIBERO-PRO")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT.parent / "srv" / "src"))
    sys.path.insert(0, str(source))
    if args.benchmark == "plus":
        sys.path.insert(0, str(ROOT / "dependencies" / "libero-plus"))
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs" / ("libero-" + args.benchmark))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from himoe_libero_bridge.batch import SUITE_MAX_STEPS
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, run_episode, validate_policy_suite
    from himoe_libero_bridge.suites import get_suite
    from libero.libero import benchmark
    if args.benchmark == "plus" and os.environ.get("HIMOE_FAST_PERTURB", "1") != "0":
        import fast_perturbations   # bit-identical numba glass_blur: 0.84 s -> 0.012 s per frame
        fast_perturbations.install()

    base_suite = get_suite(args.suite).benchmark
    task_suite = base_suite if (args.benchmark == "plus" or args.perturbation == "none") else base_suite + "_" + args.perturbation
    suite = benchmark.get_benchmark_dict()[task_suite]()
    task_name = args.task_name
    if args.task_id is None and task_name is None and args.suite == "long":
        task_name = PLUS_CAMERA_TASK if args.benchmark == "plus" else LONG_TASK
    if task_name is None and args.task_id is None and args.perturbation == "none" and args.suite == "long":
        task_name = LONG_TASK
    if task_name is not None:
        matches = [index for index in range(suite.n_tasks) if suite.get_task(index).name == task_name]
        if len(matches) != 1:
            parser.error("Expected one matching task, found %d for %r" % (len(matches), task_name))
        task_id = matches[0]
    else:
        task_id = args.task_id if args.task_id is not None else 0
    if not 0 <= task_id < suite.n_tasks:
        parser.error("task-id must be in [0, %d)" % suite.n_tasks)
    task = suite.get_task(task_id)

    with PolicyClient(args.host, args.port, connect_timeout=10, inference_timeout=60) as client:
        validate_policy_suite(client.metadata, task_suite)

    details = {
        "benchmark_variant": args.benchmark,
        "benchmark_source": str(source),
        "benchmark_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "base_policy_suite": base_suite,
        "selected_task_id": task_id,
        "selected_task_name": task.name,
        "selected_task_prompt": task.language,
    }
    benchmark_diff = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"])
    details["benchmark_tracked_diff_sha256"] = hashlib.sha256(benchmark_diff).hexdigest()
    details["benchmark_tracked_files_modified"] = bool(benchmark_diff)
    if args.benchmark == "plus":
        classification = json.loads((source / "libero/libero/benchmark/task_classification.json").read_text())
        details["perturbation"] = next(
            (item for item in classification.get(base_suite, []) if item["name"] == task.name), None
        )
    else:
        details["perturbation"] = args.perturbation
    print(json.dumps(details, indent=2), flush=True)

    artifact = run_episode(
        EpisodeConfig(
            libero_root=str(source),
            output_root=str(args.output_root or ROOT / "simulations" / args.benchmark),
            host=args.host,
            port=args.port,
            task_suite=task_suite,
            task_id=task_id,
            init_state_id=args.init_state_id,
            seed=args.seed,
            max_steps=args.max_steps if args.max_steps is not None else SUITE_MAX_STEPS[args.suite],
            render_size=args.render_size,
            inference_timeout=60,
            flow_noise_seed=args.flow_noise_seed,
        )
    )
    summary_path = artifact / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(details)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary.get(key) for key in (
        "benchmark_variant", "status", "success", "action_steps", "inference_calls",
        "duration_seconds", "video"
    )}, indent=2), flush=True)
    print(str(artifact), flush=True)


if __name__ == "__main__":
    main()
