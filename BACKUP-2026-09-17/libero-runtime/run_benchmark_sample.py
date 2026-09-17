"""Run a reproducible, task-balanced sample of Plus camera and Pro swap tasks."""

import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))

from himoe_libero_bridge.batch import BatchConfig, build_job_plan, run_batch, resume_batch
from himoe_libero_bridge.client import PolicyClient


def read_task_map(source):
    module = ast.parse((source / "libero/libero/benchmark/libero_suite_task_map.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing task map in %s" % source)


def make_plan(variant, sample_seed, episodes_per_task):
    pro_source = ROOT / "upstream/LIBERO-PRO"
    plus_source = ROOT / "upstream/LIBERO-plus"
    pro_map = read_task_map(pro_source)
    plus_map = read_task_map(plus_source)
    base_tasks = pro_map["libero_10"]
    assert len(base_tasks) == 10
    assert set(base_tasks) == set(pro_map["libero_10_swap"])
    classification = json.loads(
        (plus_source / "libero/libero/benchmark/task_classification.json").read_text()
    )["libero_10"]
    initial_ids = random.Random(sample_seed).sample(range(50), episodes_per_task)
    camera_rng = random.Random(sample_seed + 1)
    planned_tasks = []
    for base_id, base_name in enumerate(base_tasks):
        candidates = sorted(
            (item for item in classification
             if item["category"] == "Camera Viewpoints"
             and item["name"].split("_view_")[0] == base_name),
            key=lambda item: item["name"],
        )
        cameras = camera_rng.sample(candidates, episodes_per_task)
        for episode_index, init_id in enumerate(initial_ids):
            if variant == "plus":
                task_name = cameras[episode_index]["name"]
                task_id = plus_map["libero_10"].index(task_name)
                perturbation = cameras[episode_index]
            else:
                task_name = base_name
                task_id = pro_map["libero_10_swap"].index(task_name)
                perturbation = "swap"
            planned_tasks.append({
                "base_task_id": base_id,
                "base_task_name": base_name,
                "episode_index": episode_index,
                "init_state_id": init_id,
                "seed": 7,
                "flow_noise_seed": sample_seed + 50 * base_id + init_id,
                "task_name": task_name,
                "task_id": task_id,
                "perturbation": perturbation,
            })
    return {
        "schema": "himoe-perturbation-sample-v1",
        "benchmark": variant,
        "sample_seed": sample_seed,
        "episodes_per_task": episodes_per_task,
        "init_state_ids": initial_ids,
        "sampling": "All 10 base tasks equally weighted; initial states sampled without replacement; Plus camera variants sampled without replacement within each base task.",
        "scope": "Plus camera viewpoints only; Pro object-position swap only. Excludes earlier exploratory episodes.",
        "jobs": planned_tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("plus", "pro"), required=True)
    parser.add_argument("--sample-seed", type=int, default=20260914)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--output-root", type=Path, default=ROOT / "samples")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.sample_seed < 0 or not 1 <= args.episodes_per_task <= 20:
        parser.error("sample-seed must be non-negative and episodes-per-task must be in [1, 20]")
    plan = make_plan(args.benchmark, args.sample_seed, args.episodes_per_task)
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    with PolicyClient("127.0.0.1", args.port, connect_timeout=10) as client:
        metadata = client.metadata
        if metadata.get("benchmark") != "libero_10":
            raise RuntimeError("This sample requires LIBERO-10 model weights")
    lookup = {(job["base_task_id"], job["init_state_id"]): job for job in plan["jobs"]}
    source = ROOT / "upstream" / ("LIBERO-plus" if args.benchmark == "plus" else "LIBERO-PRO")
    config = BatchConfig(
        libero_root=str(source), output_root=str(args.output_root / args.benchmark),
        host="127.0.0.1", port=args.port, suite="long", task_ids=tuple(range(10)),
        init_state_ids=tuple(plan["init_state_ids"]), episodes_per_task=args.episodes_per_task,
        start_seed=7, flow_noise_seed_start=args.sample_seed,
        max_steps=520, render_size=256, inference_timeout=60, video_policy="all",
    )
    for job in build_job_plan(config):
        assert job["flow_noise_seed"] == lookup[(job["task_id"], job["init_state_id"])]["flow_noise_seed"]

    def episode_runner(episode):
        job = lookup[(episode.task_id, episode.init_state_id)]
        job_dir = Path(episode.output_root)
        batch_dir = job_dir.parents[1]
        plan_path = batch_dir / "scenario-plan.json"
        if plan_path.exists():
            saved = json.loads(plan_path.read_text())
            if saved["plan_sha256"] != plan_hash:
                raise RuntimeError("Stored scenario plan differs from requested sample")
            if saved["server_metadata"]["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
                raise RuntimeError("Model checkpoint changed since sample creation")
        else:
            plan_path.write_text(json.dumps({
                **plan, "plan_sha256": plan_hash, "server_metadata": metadata,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, indent=2) + "\n")
        print(json.dumps({"event": "sample_episode_started", "benchmark": args.benchmark,
                          "batch_dir": str(batch_dir), **job}), flush=True)
        command = [
            sys.executable, "-u", str(ROOT / "run_benchmark.py"),
            "--benchmark", args.benchmark, "--task-name", job["task_name"],
            "--init-state-id", str(episode.init_state_id),
            "--seed", str(episode.seed), "--flow-noise-seed", str(episode.flow_noise_seed),
            "--max-steps", str(episode.max_steps), "--render-size", str(episode.render_size),
            "--port", str(episode.port), "--output-root", str(job_dir),
        ]
        with (job_dir / "runner.log").open("a") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        if completed.returncode:
            raise RuntimeError("Episode process exited %d; see %s" % (completed.returncode, job_dir / "runner.log"))
        summaries = sorted(job_dir.glob("episode-*/summary.json"))
        if not summaries:
            raise RuntimeError("Episode process produced no summary")
        summary_path = max(summaries, key=lambda path: path.stat().st_mtime_ns)
        summary = json.loads(summary_path.read_text())
        if summary["server_metadata"]["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise RuntimeError("Episode used different checkpoint weights")
        print(json.dumps({"event": "sample_episode_finished", "benchmark": args.benchmark,
                          "base_task_id": episode.task_id, "init_state_id": episode.init_state_id,
                          "success": summary["success"], "action_steps": summary["action_steps"],
                          "artifact_dir": str(summary_path.parent)}), flush=True)
        return summary_path.parent

    if args.resume:
        stored = json.loads((args.resume / "scenario-plan.json").read_text())
        if stored["plan_sha256"] != plan_hash:
            raise RuntimeError("Resume arguments do not match the stored sample")
        result = resume_batch(str(args.resume), port=args.port, episode_runner=episode_runner)
    else:
        result = run_batch(config, episode_runner=episode_runner)
    print(str(result), flush=True)


if __name__ == "__main__":
    main()
