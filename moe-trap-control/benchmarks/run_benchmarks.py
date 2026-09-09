#!/usr/bin/env python3
"""Run isolated Pro/Plus environments against the existing HiMoE model matrix."""

import argparse
import concurrent.futures
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent.parent / "himoe-vla-cache/libero-extensions"
BASE = Path("/home/jovyan/.cache/himoe-libero-bridge")
BRIDGE = HERE.parent.parent / "himoe-libero-wrist-fix/src"
ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)
ROOTS = {"pro": CACHE / "LIBERO-PRO", "plus": CACHE / "LIBERO-plus"}
SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_list(value):
    result = tuple(int(item) for item in value.split(","))
    if not result or len(set(result)) != len(result) or any(g not in ALLOWED_GPUS for g in result):
        raise argparse.ArgumentTypeError("Only physical GPUs 0,1,2,3,4,5,7 are allowed; GPU 6 is excluded")
    return result


def environment(benchmark, gpu, render, initialize=False):
    if gpu not in ALLOWED_GPUS:
        raise ValueError("Physical GPU 6 is excluded")
    root = ROOTS[benchmark]
    config = CACHE / "config" / benchmark
    data = root / "libero/libero"
    if initialize:
        (root / "datasets").mkdir(exist_ok=True)
        write_json(config / "config.yaml", {
            "benchmark_root": str(data), "bddl_files": str(data / "bddl_files"),
            "init_states": str(data / "init_files"), "assets": str(data / "assets"),
            "datasets": str(root / "datasets"),
        })
    env = os.environ.copy()
    system = CACHE / "system-libs"
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": render, "PYOPENGL_PLATFORM": render,
        "LIBERO_CONFIG_PATH": str(config), "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (root, CACHE / "python-extras", BRIDGE))),
        "LD_LIBRARY_PATH": os.pathsep.join(map(str, (
            system / "usr/lib/x86_64-linux-gnu",
            BASE / "system-libs/usr/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        ))),
        "MAGICK_HOME": str(system / "usr"),
        "MAGICK_CONFIGURE_PATH": str(system / "etc/ImageMagick-6"),
        "MAGICK_CODER_MODULE_PATH": str(system / "usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/coders"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    return env


def variants(benchmark):
    inventory = json.loads((HERE.parent / "design/benchmark_inventory.json").read_text())
    return [row for row in inventory["variants"] if row["benchmark"] == benchmark]


def load_suite(registry, cache):
    from libero.libero import benchmark
    if registry not in cache:
        # Plus prints thousands of task IDs when constructing a suite.
        with contextlib.redirect_stdout(io.StringIO()):
            cache[registry] = benchmark.get_benchmark_dict()[registry](task_order_index=0)
    return cache[registry]


def inventory_worker(args):
    import numpy as np
    import torch
    from libero.libero import get_libero_path
    suites, states, records, errors = {}, {}, [], []
    original_load = torch.load

    def cached_load(path, *positional, **keyword):
        key = str(path)
        if key not in states:
            states[key] = original_load(path, *positional, **keyword)
        return states[key]

    torch.load = cached_load
    try:
        for row in variants(args.benchmark):
            try:
                suite = load_suite(row["registry"], suites)
                task = suite.get_task(row["registry_index"])
                if task.name != row["task_name"]:
                    raise ValueError("Registry/task manifest mismatch")
                filename = task.bddl_file
                if args.benchmark == "plus" and "_view_" in filename:
                    filename = filename.split("_view_")[0] + ".bddl"
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / filename
                if not bddl.is_file():
                    raise FileNotFoundError(str(bddl))
                initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
                if initial.ndim != 2 or not len(initial) or not np.isfinite(initial).all():
                    raise ValueError("Invalid initial state array: " + str(initial.shape))
                records.append({
                    "variant_id": row["variant_id"], "registry": row["registry"],
                    "task_index": row["registry_index"], "bddl": str(bddl),
                    "initial_state_shape": list(initial.shape), "initial_state_dtype": str(initial.dtype),
                })
            except Exception as error:
                errors.append({"variant": row, "error": repr(error)})
    finally:
        torch.load = original_load
    report = {
        "benchmark": args.benchmark, "checked_tasks": len(records), "errors": errors,
        "minimum_init_states": min((r["initial_state_shape"][0] for r in records), default=0),
        "distinct_init_files": len(states), "tasks": records,
        "status": "passed" if not errors else "failed", "simulation_executed": False,
    }
    write_json(args.output / "result.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("tasks", "errors")}), flush=True)
    if errors:
        raise RuntimeError("%d task files failed inventory validation" % len(errors))


def generate_pro_env_worker(args):
    import numpy as np
    import torch
    from libero.libero.envs.env_wrapper import ControlEnv
    from perturbation import BDDLCombinedPerturbator, PerturbFlags

    row = next(row for row in variants("pro") if row["variant_id"] == args.variant)
    if row["category"] != "Environment":
        raise ValueError("Only missing Pro environment perturbations may be generated")
    root = ROOTS["pro"]
    data = root / "libero/libero"
    source = data / "bddl_files" / row["suite"] / (row["task_name"] + ".bddl")
    target = data / "bddl_files" / row["registry"] / source.name
    state_file = data / "init_files" / row["registry"] / (row["task_name"] + ".pruned_init")
    source_text = source.read_text()
    pipeline = BDDLCombinedPerturbator({"environment": str(root / "libero_ood/ood_environment.yaml")})
    generated = pipeline.perturb_content(
        content=source_text, task_suite_name=row["suite"], task_name=row["task_name"],
        flags=PerturbFlags(use_environment=True), seed=args.seed,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text() != generated:
        raise FileExistsError("Refusing to replace an existing different BDDL")
    target.write_text(generated)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        raise FileExistsError("Initial states already exist: " + str(state_file))
    env = None
    initial = []
    started = time.monotonic()
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        env = ControlEnv(
            bddl_file_name=str(target), use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=False, hard_reset=False,
        )
        env.seed(args.seed)
        for index in range(50):
            env.reset()
            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            if not np.isfinite(state).all():
                raise ValueError("Non-finite generated initial state")
            initial.append(state)
        initial = np.stack(initial)
        if len(np.unique(initial, axis=0)) != len(initial):
            raise ValueError("Generated initial states are duplicated")
        temporary = state_file.with_name(state_file.name + ".tmp")
        torch.save(initial, temporary)
        temporary.replace(state_file)
    finally:
        if env is not None:
            env.close()
    write_json(args.output / "result.json", {
        "status": "passed", "variant": row, "seed": args.seed,
        "source": "local generation using pinned official BDDLCombinedPerturbator",
        "initial_states": str(state_file), "initial_state_shape": list(initial.shape),
        "initial_states_sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
        "bddl": str(target), "bddl_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "same_bddl_as_base": generated == source_text,
        "generation": "50 seeded ControlEnv resets, no renderer, hard_reset=False",
        "elapsed_seconds": time.monotonic() - started,
    })


def rollout_worker(args):
    import numpy as np
    from PIL import Image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.preprocess import build_policy_observation
    from himoe_libero_bridge.suites import get_suite

    row = next(row for row in variants(args.benchmark) if row["variant_id"] == args.variant)
    suite = load_suite(row["registry"], {})
    task = suite.get_task(row["registry_index"])
    initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
    if args.init_index < 0 or args.init_index >= len(initial):
        raise ValueError("Initial state index is out of range")
    horizon = args.max_steps if args.max_steps is not None else row["horizon_steps"]
    if horizon > row["horizon_steps"]:
        raise ValueError("Requested rollout exceeds the benchmark horizon")
    gpu = args.gpus[0]
    port = 8800 + gpu * 10 + SUITES.index(row["suite"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env, client = None, None
    started = time.monotonic()
    report = {
        "benchmark": args.benchmark, "variant": row, "physical_gpu": gpu,
        "port": port, "init_index": args.init_index, "seed": args.seed,
        "hidden_capture": False, "intervention": False, "action_steps": 0,
        "queries": 0, "success": False, "render_backend": args.render,
        "purpose": "installation_smoke" if args.mode == "smoke" else "baseline_rollout",
        "horizon": horizon, "status": "running",
    }
    write_json(args.output / "result.json", report)
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
            render_gpu_device_id=gpu, horizon=row["horizon_steps"] + 11,
        )
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial[args.init_index])
        for _ in range(10):
            obs, _, _, _ = env.step(np.asarray([0.0] * 6 + [-1.0]))
        # BDDL contains the actual instruction, including language perturbations.
        # Registry filenames can contain camera/noise/texture parameters.
        prompt = env.language_instruction
        request = build_policy_observation(obs, prompt)
        pixels = {}
        for key, name in (("observation/image", "agentview"), ("observation/wrist_image", "wrist")):
            frame = request[key]
            if frame.shape != (224, 224, 3) or float(frame.std()) < 1:
                raise ValueError("Blank or invalid camera: " + name)
            Image.fromarray(frame).save(args.output / (name + ".png"))
            pixels[name] = {"std": float(frame.std()), "min": int(frame.min()), "max": int(frame.max())}
        report["camera_pixels"] = pixels
        report["prompt"] = str(prompt)
        report["sim_state_shape"] = list(np.asarray(env.get_sim_state()).shape)
        client = PolicyClient(port=port, connect_timeout=15, inference_timeout=180)
        metadata = client.metadata
        if metadata.get("bundle_physical_gpu") != gpu or metadata.get("benchmark") != row["suite"]:
            raise ValueError("Policy/GPU/suite identity mismatch")
        model = ("goal", "spatial", "object", "long")[SUITES.index(row["suite"])]
        if metadata.get("checkpoint_sha256") != get_suite(model).weights_sha256:
            raise ValueError("Policy checkpoint identity mismatch")
        report["server_identity"] = {key: metadata[key] for key in (
            "checkpoint_sha256", "normalization_stats_sha256", "bundle_physical_gpu", "bundle_port",
        )}
        with (args.output / "queries.jsonl").open("w") as query_log:
            while report["action_steps"] < horizon and not report["success"]:
                request = build_policy_observation(obs, prompt)
                request["flow/noise"] = rng.standard_normal((10, 24)).astype(np.float32)
                request["routing/capture"] = False
                before = time.monotonic()
                response = client.infer(request)
                actions = np.asarray(response["actions"])
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise ValueError("Invalid policy actions")
                noise_hash = hashlib.sha256(request["flow/noise"].tobytes()).hexdigest()
                if response.get("flow/noise_sha256") != noise_hash:
                    raise ValueError("Policy did not acknowledge the supplied flow noise")
                query_log.write(json.dumps({
                    "query": report["queries"], "action_step": report["action_steps"],
                    "seconds": time.monotonic() - before, "actions": actions.tolist(),
                    "noise_sha256": noise_hash,
                }) + "\n")
                query_log.flush()
                report["queries"] += 1
                for action in actions[:min(10, horizon - report["action_steps"])]:
                    obs, _, done, _ = env.step(action.tolist())
                    report["action_steps"] += 1
                    report["success"] = bool(done or env.check_success())
                    if report["success"]:
                        break
        last = build_policy_observation(obs, prompt)["observation/image"]
        Image.fromarray(last).save(args.output / "agentview-final.png")
        if not np.isfinite(np.asarray(env.get_sim_state())).all():
            raise ValueError("Non-finite simulator state after actions")
        report["status"] = "passed"
        report["truncated"] = not report["success"] and horizon < row["horizon_steps"]
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "result.json", report)
    print(json.dumps({key: report[key] for key in ("benchmark", "physical_gpu", "status", "queries", "action_steps", "success")}), flush=True)


def launch_job(args, benchmark, gpu, variant, output):
    output.mkdir(parents=True, exist_ok=True)
    command = [str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()), args.mode,
               "--worker", "--benchmark", benchmark, "--gpus", str(gpu), "--output", str(output),
               "--seed", str(args.seed), "--init-index", str(args.init_index), "--render", args.render]
    if variant:
        command.extend(["--variant", variant["variant_id"]])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    with (output / "worker.log").open("w") as log:
        try:
            result = subprocess.run(command, env=environment(benchmark, gpu, args.render),
                                    cwd=str(ROOTS[benchmark]), stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = {"benchmark": benchmark, "gpu": gpu, "exit_code": code, "output": str(output)}
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "smoke", "run", "generate-pro-env"))
    parser.add_argument("--benchmark", choices=("pro", "plus", "all"), default="all")
    parser.add_argument("--gpus", type=gpu_list, default=ALLOWED_GPUS)
    parser.add_argument("--suite", choices=SUITES, default="libero_goal")
    parser.add_argument("--category")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--variant")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "generate-pro-env":
        if args.benchmark == "plus":
            parser.error("Environment generation is only needed for Pro")
        args.benchmark = "pro"
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.mode == "smoke" and args.max_steps is None:
        args.max_steps = 20
    if args.output is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = HERE / "runs" / (args.mode + "-" + stamp)
    args.output = args.output.resolve()
    if args.worker:
        import libero.libero
        if ROOTS[args.benchmark] not in Path(libero.libero.__file__).parents:
            raise RuntimeError("Wrong LIBERO package imported")
        if args.mode == "inventory":
            inventory_worker(args)
        elif args.mode == "generate-pro-env":
            generate_pro_env_worker(args)
        else:
            rollout_worker(args)
        return
    benchmarks = ("pro", "plus") if args.benchmark == "all" else (args.benchmark,)
    jobs = []
    for name in benchmarks:
        environment(name, args.gpus[0], args.render, initialize=True)
        if args.mode == "inventory":
            jobs.append((name, args.gpus[0], None, args.output / name))
            continue
        rows = variants(name)
        if args.mode == "generate-pro-env":
            rows = [row for row in rows if row["category"] == "Environment"
                    and not (ROOTS["pro"] / "libero/libero/init_files" / row["registry"]
                             / (row["task_name"] + ".pruned_init")).exists()]
        elif args.mode == "smoke":
            selected = {}
            for row in rows:
                selected.setdefault((row["suite"], row["category"]), row)
            rows = list(selected.values())
        elif args.variant:
            rows = [row for row in rows if row["variant_id"] == args.variant]
        else:
            rows = [row for row in rows if row["suite"] == args.suite
                    and row["registry_index"] == args.task_index
                    and (args.category is None or row["category"].lower() == args.category.lower())]
        if not rows:
            parser.error("No matching benchmark task")
        for index, row in enumerate(rows):
            gpu = args.gpus[index % len(args.gpus)]
            jobs.append((name, gpu, row, args.output / name / row["variant_id"]))
    # One process per GPU at a time, keeping the two LIBERO module namespaces isolated.
    grouped = {gpu: [job for job in jobs if job[1] == gpu] for gpu in args.gpus}
    def group_worker(group):
        return [launch_job(args, *job) for job in group]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = [row for group in pool.map(group_worker, grouped.values()) for row in group]
    report = {"gpus": args.gpus, "excluded_gpu": 6, "mode": args.mode, "jobs": results,
              "passed": sum(row["exit_code"] == 0 for row in results), "total": len(results)}
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "jobs"}), flush=True)
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
