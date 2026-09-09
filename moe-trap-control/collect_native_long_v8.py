#!/usr/bin/env python3
"""Collect original Long mains, exact C0, and the frozen v8 selector branches."""

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

import numpy as np

from adaptive_control import RouteRisk
from collect_preflight_worker import snapshot, verify_egl_device
from collect_v8_closed_loop import ClosedLoopSession
from collection_routes import PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, save_snapshot
from fixed_recovery_control import TriggerMonitor
from native_long_runtime import ROOT, COMMIT
from v8_closed_loop import PROTOCOL, SETTINGS, limits


class NativeLongSession(ClosedLoopSession):
    def __init__(self, args, cache):
        from benchmarks.run_benchmarks import load_suite
        import libero.libero
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from himoe_libero_bridge.client import PolicyClient
        from himoe_libero_bridge.preprocess import build_policy_observation
        from himoe_libero_bridge.suites import get_suite

        if (args.gpu not in (0, 1, 2, 3) or args.render_gpu not in (0, 3) or
                os.environ.get("CUDA_VISIBLE_DEVICES") != "" or
                ROOT not in Path(libero.libero.__file__).resolve().parents or
                args.control["contract"] != SETTINGS):
            raise ValueError("Native environment, GPU, or settings identity")
        self.args, self.cache, self.task = args, cache, args.control["parent"]
        self.row = self.task["variant"]
        if (self.row["benchmark"] != "native_long" or self.row["registry"] != "libero_10" or
                self.row["variant_id"] != args.variant or args.seed != self.task["noise_seed"] or
                args.init_index != self.task["init_index"] or args.main_id != self.task["main_id"]):
            raise ValueError("Native task identity changed")
        suite = load_suite("libero_10", cache.setdefault("suites", {}))
        task = suite.get_task(self.row["registry_index"])
        self.initial = np.asarray(suite.get_task_init_states(self.row["registry_index"]))
        self.bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        initial_sha = hashlib.sha256(np.ascontiguousarray(self.initial[args.init_index]).tobytes()).hexdigest()
        if (task.name != self.row["task_name"] or str(self.bddl) != self.row["bddl_path"] or
                digest(self.bddl) != self.row["bddl_sha256"] or
                digest(self.row["init_file"]) != self.row["init_file_sha256"] or
                initial_sha != self.task["initial_state_sha256"]):
            raise ValueError("Official BDDL or initial state changed")
        self.risk = cache.setdefault("risk", RouteRisk())
        self.env_class, self.build_observation = OffScreenRenderEnv, build_policy_observation
        self.prompt, self.horizon = task.language, 520
        self.rng, self.env = np.random.default_rng(args.seed), None
        self.tape_writer, self.tape = None, None
        self.thresholds, self.margins = limits()
        self.client = cache.get("client")
        if self.client is None:
            self.client = PolicyClient(port=args.port, connect_timeout=30, inference_timeout=180)
            cache["client"] = self.client
        metadata = self.client.metadata
        if (metadata.get("bundle_physical_gpu") != args.gpu or
                metadata.get("checkpoint_sha256") != get_suite("long").weights_sha256):
            raise ValueError("Wrong Long model endpoint")
        is_main = args.control["kind"] == "native_main"
        self.parent = None if is_main else Path(self.task["parent_directory"])
        self.original = None if is_main else json.loads((self.parent / "result.json").read_text())
        if not is_main:
            original = self.original
            if (original["status"] != "completed" or not original["main_complete"] or
                    original["main_intervention"] or original["variant"] != self.row or
                    original["seed"] != args.seed or original["init_index"] != args.init_index or
                    original["main_id"] != args.main_id or original["prompt"] != self.prompt):
                raise ValueError("Invalid original native parent")
            for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
                if digest(self.parent / filename) != self.task[key]:
                    raise ValueError("Native parent commitment changed")
            for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                        "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
                if metadata[key] != original["model_metadata"][key]:
                    raise ValueError("Native parent model identity: "+key)
        args.output.mkdir(parents=True, exist_ok=False)
        self.report = dict(status="running", protocol=PROTOCOL, contract=SETTINGS,
            main_id=args.main_id, job_kind=args.control["kind"], variant=self.row, model="long",
            model_metadata=metadata, gpu=args.gpu, render_gpu=args.render_gpu,
            egl_device_uuid=verify_egl_device(args.render_gpu), seed=args.seed, init_index=args.init_index,
            hidden_capture=False, main_intervention=False, main_complete=not is_main, reused_main=not is_main,
            queries=0, action_steps=0, success=False, first_alarm_query=None,
            first_knn_alarm=self.task.get("first_alarm"), first_alarms=self.task.get("first_alarms"),
            analysis_role="native", c0=None, events=[], branches=[], invalid_pair=False,
            actual_model_queries=0, candidate_queries=0, reused_candidate_queries=0,
            environment_pid=os.getpid(), worker_job_index=cache["jobs"],
            rng_tape_sha256=None, all_native_suffixes_exact=False, prompt=self.prompt,
            original_libero_root=str(ROOT), original_libero_commit=COMMIT, perturbation=False,
            bddl_path=str(self.bddl), bddl_sha256=digest(self.bddl), initial_state_sha256=initial_sha)
        if not is_main:
            self.report.update(parent_directory=str(self.parent),
                parent_commit_sha256=self.task["parent_commit_sha256"],
                parent_manifest_sha256=self.task["parent_manifest_sha256"],
                native_success=self.original["success"], action_steps=self.original["action_steps"],
                success=self.original["success"], first_alarm_query=self.original["first_alarm_query"])
        self.save()

    def new_env(self):
        self.close_env()
        args = self.args
        path = Path(tempfile.gettempdir()) / ("himoe-collection-egl-%d-%d.lock" % (os.getuid(), args.render_gpu))
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            random.seed(args.seed)
            np.random.seed(args.seed)
            self.env = self.env_class(bddl_file_name=str(self.bddl), camera_heights=224,
                camera_widths=224, render_gpu_device_id=args.render_gpu, horizon=531)
            self.env.seed(args.seed)
            self.env.reset()
            return self.env.set_init_state(self.initial[args.init_index])

    def main_rollout(self):
        from PIL import Image
        obs = self.new_env()
        for _ in range(10):
            obs, _, _, _ = self.env.step([0.] * 6 + [-1.])
        if self.prompt != str(self.env.language_instruction):
            raise ValueError("Original language prompt changed")
        request = self.build_observation(obs, self.prompt)
        for key, name in (("observation/image", "agentview"), ("observation/wrist_image", "wrist")):
            if np.asarray(request[key]).std() < 1:
                raise ValueError("Blank original camera")
            Image.fromarray(request[key]).save(self.args.output / (name+".png"))
        save_snapshot(self.args.output / "preflight_q000", snapshot(self.env, obs, self.rng, 0, 0, self.prompt))
        monitor = TriggerMonitor()
        writer = ShardWriter(self.args.output / "main")
        steps, q, success = 0, 0, False
        try:
            while steps < 520 and not success:
                if monitor.first["v8_frozen"] == q-1 and q > 0:
                    save_snapshot(self.args.output / "alarm_snapshot", snapshot(self.env, obs, self.rng, q, steps, self.prompt))
                request, response, inference = self.query(obs, self.rng.standard_normal((10, 24)).astype(np.float32))
                alarm = monitor.update(response[PROBS_KEY], request["observation/state"][:3])
                before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                obs, count, success, duration = self.advance(obs, response["actions"], min(10, 520-steps), steps)
                after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                writer.append(self.record(request, response, q, steps, count, before, after, success, alarm, inference, duration))
                steps += count
                q += 1
                self.report.update(queries=q, action_steps=steps, success=success)
                if q % 8 == 0:
                    self.save()
        finally:
            self.report["main_shards"] = writer.close()
        first_v7 = monitor.first["v7_frozen"]
        self.report.update(main_complete=True, first_alarms=monitor.first,
            first_alarm_query=None if first_v7 < 0 else first_v7)
        atomic_json(self.args.output / "main_complete.json", dict(queries=q, action_steps=steps,
            main_id=self.args.main_id, success=success, manifest_sha256=digest(self.args.output / "main/manifest.json")))
        Image.fromarray(self.build_observation(obs, self.prompt)["observation/image"]).save(self.args.output / "agentview-final.png")


def run(args, cache):
    session = NativeLongSession(args, cache)
    started = time.monotonic()
    try:
        kind = args.control["kind"]
        if kind == "native_main":
            session.main_rollout()
        elif kind == "fixed_replay":
            session.replay()
        elif kind == "fixed_branches":
            session.branches()
        else:
            raise ValueError("Unknown native job")
        session.report["status"] = "completed"
    except BaseException:
        session.report.update(status="failed", invalid_pair=True, error=traceback.format_exc())
        raise
    finally:
        session.close_env()
        session.report["elapsed_seconds"] = time.monotonic()-started
        session.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--render-gpu", type=int, choices=(0, 3), required=True)
    parser.add_argument("--port", type=int, required=True)
    args, cache = parser.parse_args(), {}
    args.model, args.benchmark, args.exact_noise_fastpath = "long", "native_long", False
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            for key in ("variant", "seed", "main_id", "init_index", "control"):
                setattr(task, key, message[key])
            task.output = Path(message["output"])
            cache["jobs"] = cache.get("jobs", 0)+1
            result = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                result["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"] = 1
            print("COLLECTION_RESULT "+json.dumps(result), flush=True)
    finally:
        if cache.get("client"):
            cache["client"].close()


if __name__ == "__main__":
    main()
