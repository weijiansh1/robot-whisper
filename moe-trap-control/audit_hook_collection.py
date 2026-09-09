#!/usr/bin/env python3
"""Bounded hook and snapshot diagnostics; never load models or use GPU 6."""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
ROUTES = HERE.parent / "himoe-route-capture"
sys.path.insert(0, str(ROUTES))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def live_probe(output):
    import numpy as np
    from probe_preloaded_models import check_checkpoint, infer, observation, open_endpoint

    rows = []
    with contextlib.ExitStack() as stack:
        for model in ("goal", "spatial", "object", "long"):
            connection, metadata = open_endpoint(stack, "127.0.0.1", 0, model)
            if metadata["bundle_physical_gpu"] != 0:
                raise RuntimeError("Unexpected policy GPU")
            check_checkpoint(metadata, model)
            responses = []
            for capture in (False, True, True, False):
                response, seconds = infer(connection, observation(model, 91008, capture), model)
                responses.append((response, seconds))
            before, captured, repeated, after = [row[0] for row in responses]
            ids = np.asarray(captured["routing/expert_ids"])
            weights = np.asarray(captured["routing/expert_weights"])
            rows.append({
                "model": model, "physical_gpu": 0, "port": metadata["bundle_port"],
                "request_seconds": [row[1] for row in responses],
                "capture_preserves_actions": np.array_equal(before["actions"], captured["actions"]),
                "repeat_preserves_actions": np.array_equal(captured["actions"], repeated["actions"]),
                "hook_removal_preserves_actions": np.array_equal(before["actions"], after["actions"]),
                "repeat_preserves_ids": np.array_equal(ids, repeated["routing/expert_ids"]),
                "repeat_preserves_weights": np.array_equal(weights, repeated["routing/expert_weights"]),
                "no_route_fields_when_disabled": not any(key.startswith("routing/") for key in after),
                "ids_shape": list(ids.shape), "weights_shape": list(weights.shape),
                "ids_min": int(ids.min()), "ids_max": int(ids.max()),
                "combine_sum_max_error": float(np.abs(weights.sum(-1) - 1).max()),
                "response_keys": sorted(captured),
                "full_hb_probabilities_returned": "recorder/hb_router_probs" in captured,
                "hidden_returned": any("hidden" in key.lower() for key in captured),
            })
    write_json(output, {"physical_gpu": 0, "real_queries": 16, "models": rows})
    print(json.dumps(rows), flush=True)


def weight_probe(output):
    import torch
    from torch import nn
    from himoe_router_recorder import HiMoERouteRecorder, ControlStepRecord, combine_weight_from_raw

    class MoEGate_load_bal(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_routed_experts = 5
            self.top_k = 4
            self.weight = nn.Parameter(torch.tensor([[4.0], [3.0], [2.0], [1.0], [0.0]]))

        def forward(self, hidden):
            scores = torch.nn.functional.linear(hidden.reshape(-1, 1), self.weight).softmax(-1)
            weight, ids = scores.topk(4, sorted=True)
            return ids, weight / weight.sum(-1, keepdim=True), None

    core = nn.Module()
    core.layers = nn.ModuleList([nn.Module() for _ in range(3)])
    core.layers[2].mlp = nn.Module()
    core.layers[2].mlp.gate = gate = MoEGate_load_bal()
    recorder = HiMoERouteRecorder(core, store_full_probs=True, expect_gates=1,
                                 store_hidden=False, verify_steps=0)
    hidden = torch.ones(1, 11, 1)
    recorder.attach()
    recorder.begin_control_step(0, 0)
    _, original_weights, _ = gate(hidden)
    ordinary = combine_weight_from_raw(recorder._buf[2][0]["raw"], 4)
    ordinary_error = float((ordinary.reshape(-1, 4) - original_weights).abs().max())
    recorder.close()

    def swap_last_slot(_module, _inputs, result):
        ids, weights, auxiliary = result
        swapped = ids.clone()
        swapped[:, -1] = 4
        return swapped, weights, auxiliary

    intervention = gate.register_forward_hook(swap_last_slot)
    recorder.attach()
    try:
        recorder.begin_control_step(0, 1)
        ids, effective_weights, _ = gate(hidden)
        captured = recorder._buf[2][0]
        reconstructed = combine_weight_from_raw(captured["raw"], 4).reshape(-1, 4)
        report = {
            "workload": "CPU mock gate; replace weakest native slot with best unselected expert; preserve slot weights",
            "ordinary_weight_max_error": ordinary_error,
            "intervened_weight_max_error": float((reconstructed - effective_weights).abs().max()),
            "effective_ids_captured": torch.equal(ids, captured["idx"].reshape(-1, 4)),
            "effective_ids": ids[0].tolist(), "effective_weights": effective_weights[0].tolist(),
            "reconstructed_weights": reconstructed[0].tolist(),
            "hidden_buffered": "hidden" in captured,
            "record_schema_fields": [field.name for field in dataclasses.fields(ControlStepRecord)],
        }
        if ordinary_error > 1e-7 or report["intervened_weight_max_error"] <= 1e-3:
            raise RuntimeError("The expected recorder discrepancy was not reproduced")
        write_json(output, report)
        print(json.dumps(report), flush=True)
    finally:
        recorder.close()
        intervention.remove()


def snapshot_worker(benchmark, output):
    import copy
    import random
    import numpy as np
    from branch_snapshot import restore_full_state, save_full_state
    from benchmarks.run_benchmarks import load_suite, variants
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    candidates = [row for row in variants(benchmark) if row["suite"] == "libero_goal"]
    if benchmark == "plus":
        row = next(row for row in candidates if row["task_name"].endswith("_noise_40"))
    else:
        row = next(row for row in candidates if row["category"] == "Semantic")
    suite = load_suite(row["registry"], {})
    task = suite.get_task(row["registry_index"])
    initial = np.asarray(suite.get_task_init_states(row["registry_index"]))[0]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    action = np.asarray([0.02, -0.01, 0.01, 0.0, 0.0, 0.0, -1.0])

    def new_env():
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
                                 horizon=311, render_gpu_device_id=0)
        env.seed(7)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(np.asarray([0.0] * 6 + [-1.0]))
        return env, obs

    def advance(env):
        for _ in range(3):
            obs, _, _, _ = env.step(action)
        return np.asarray(env.get_sim_state()).copy(), np.asarray(obs["agentview_image"]).copy()

    def difference(left, right):
        return float(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).max())

    def pixel_errors(left, right):
        delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        return {"max": float(delta.max()), "mean": float(delta.mean()),
                "different_pixel_fraction": float(np.any(delta != 0, axis=-1).mean())}

    observable_fields = ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled")

    def restore_observables(target, saved):
        target.env._obs_cache = copy.deepcopy(saved["cache"])
        for name, values in saved["observables"].items():
            for key, value in values.items():
                setattr(target.env._observables[name], key, copy.deepcopy(value))

    random.seed(7)
    np.random.seed(7)
    env, original_obs = new_env()
    fresh = None
    try:
        snapshot = save_full_state(env)
        original_frame = np.asarray(original_obs["agentview_image"]).copy()
        numpy_state = np.random.get_state()
        observable_state = {
            "cache": copy.deepcopy(env.env._obs_cache),
            "observables": {
                name: {key: copy.deepcopy(getattr(item, key)) for key in observable_fields}
                for name, item in env.env._observables.items()
            },
        }
        reference_state, reference_frame = advance(env)
        restored_obs = restore_full_state(env, snapshot)
        restored_frame = np.asarray(restored_obs["agentview_image"]).copy()
        immediate_sim_error = difference(env.get_sim_state(), snapshot["sim"])
        unpaired_state, unpaired_frame = advance(env)
        restore_full_state(env, snapshot)
        np.random.set_state(numpy_state)
        paired_state, paired_frame = advance(env)
        restore_full_state(env, snapshot)
        np.random.set_state(numpy_state)
        restore_observables(env, observable_state)
        cached_state, cached_frame = advance(env)
        # Recreate the simulator after closing the original one.
        env.close()
        env = None
        fresh, _ = new_env()
        fresh_obs = restore_full_state(fresh, snapshot)
        fresh_initial_frame = np.asarray(fresh_obs["agentview_image"]).copy()
        np.random.set_state(numpy_state)
        fresh_state, fresh_frame = advance(fresh)
        restore_full_state(fresh, snapshot)
        np.random.set_state(numpy_state)
        restore_observables(fresh, observable_state)
        fresh_cached_state, fresh_cached_frame = advance(fresh)
        report = {
            "benchmark": benchmark, "variant": row, "model_queries": 0, "renderer": "osmesa",
            "snapshot_keys": sorted(snapshot), "snapshot_dtype": str(snapshot["sim"].dtype),
            "immediate_restored_sim_max_error": immediate_sim_error,
            "immediate_restored_rgb_max_error": difference(original_frame, restored_frame),
            "unpaired_replay_sim_max_error": difference(reference_state, unpaired_state),
            "unpaired_replay_rgb_max_error": difference(reference_frame, unpaired_frame),
            "rng_rewound_replay_sim_max_error": difference(reference_state, paired_state),
            "rng_rewound_replay_rgb_max_error": difference(reference_frame, paired_frame),
            "rng_and_observables_replay_sim_max_error": difference(reference_state, cached_state),
            "rng_and_observables_replay_rgb_max_error": difference(reference_frame, cached_frame),
            "fresh_simulator_initial_rgb_max_error": difference(original_frame, fresh_initial_frame),
            "fresh_simulator_rng_rewound_sim_max_error": difference(reference_state, fresh_state),
            "fresh_simulator_rng_rewound_rgb_max_error": difference(reference_frame, fresh_frame),
            "fresh_simulator_rng_and_observables_sim_max_error": difference(reference_state, fresh_cached_state),
            "fresh_simulator_rng_and_observables_rgb_max_error": difference(reference_frame, fresh_cached_frame),
            "replay_pixel_errors": {
                "legacy": pixel_errors(reference_frame, unpaired_frame),
                "rng_only": pixel_errors(reference_frame, paired_frame),
                "rng_and_observables": pixel_errors(reference_frame, cached_frame),
                "fresh_rng_and_observables": pixel_errors(reference_frame, fresh_cached_frame),
            },
            "observable_probe_fields": list(observable_fields),
            "production_snapshot_helper_modified": False,
            "scope": "one early snapshot, three fixed actions; not contact/terminal fidelity coverage",
        }
        write_json(output, report)
        print(json.dumps(report), flush=True)
    finally:
        if env is not None:
            env.close()
        if fresh is not None:
            fresh.close()


def snapshots(output):
    from benchmarks.run_benchmarks import BASE, environment
    reports = []
    for name in ("pro", "plus"):
        target = output.with_name(output.stem + "-" + name + ".json")
        env = environment(name, 0, "osmesa", initialize=True)
        env["PYTHONPATH"] += os.pathsep + str(HERE) + os.pathsep + str(ROUTES)
        command = [str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()),
                   "snapshot-worker", "--benchmark", name, "--out", str(target)]
        subprocess.run(command, env=env, check=True, timeout=240)
        reports.append(json.loads(target.read_text()))
    write_json(output, {"snapshots": reports, "model_queries": 0, "gpu_rendering": False})


def sampling_probe(output):
    with (HERE / "design/collection_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    readiness = json.loads((HERE / "benchmarks/READINESS.json").read_text())
    unchanged = set(readiness["pro_environment_generation"]["bddl_unchanged_from_base"])
    folds = collections.defaultdict(set)
    for row in rows:
        folds[row["base_task"]].add(row["base_task_fold"])
    screen = [row for row in rows if row["screen"] == "1"]

    def counts(items):
        return dict(collections.Counter(row["benchmark"] for row in items))

    report = {
        "main_rows": len(rows), "unique_main_ids": len({row["main_id"] for row in rows}),
        "main_by_benchmark": counts(rows), "screen_by_benchmark": counts(screen),
        "status_counts": dict(collections.Counter(row["status"] for row in rows)),
        "base_tasks": len(folds),
        "base_task_fold_conflicts": {task: sorted(values) for task, values in folds.items() if len(values) > 1},
        "plus_screen_suite_category_counts": dict(collections.Counter(
            row["suite"] + "/" + row["category"] for row in screen if row["benchmark"] == "plus")),
        "invalid_screen_probabilities": sum(not 0 < float(row["screen_variant_inclusion_probability"]) <= 1
                                             for row in screen),
        "unchanged_pro_variant_ids": sorted(unchanged),
        "unchanged_pro_main_rows": sum(row["variant_id"] in unchanged for row in rows),
        "unchanged_pro_screen_rows": sum(row["variant_id"] in unchanged for row in screen),
        "ood_main_by_benchmark": counts([row for row in rows if row["variant_id"] not in unchanged]),
        "ood_screen_by_benchmark": counts([row for row in screen if row["variant_id"] not in unchanged]),
        "manifest_has_explicit_ood_or_control_flag": any(key in rows[0] for key in (
            "is_ood", "is_control", "analysis_role", "bddl_unchanged_from_base")),
        "manifest_modified": False,
    }
    write_json(output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("live", "weights", "snapshots", "snapshot-worker", "sampling"))
    parser.add_argument("--benchmark", choices=("pro", "plus"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "live":
        live_probe(args.out)
    elif args.mode == "weights":
        weight_probe(args.out)
    elif args.mode == "snapshots":
        snapshots(args.out)
    elif args.mode == "sampling":
        sampling_probe(args.out)
    else:
        snapshot_worker(args.benchmark, args.out)
