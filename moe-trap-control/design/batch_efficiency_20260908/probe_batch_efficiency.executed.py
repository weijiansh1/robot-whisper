#!/usr/bin/env python3
"""Bounded, temporary Goal batch probe; never replaces a running model service."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

# This helper starts CPU-only. The explicit UUID binding below precedes torch import.
import probe_preloaded_models as probe
import numpy as np
import serve_model_bundle as bundle

HERE = Path(__file__).resolve().parent
FIELDS = ("actions", "routing/expert_ids", "routing/expert_weights", "routing/layer_indices")


class BatchRouteCapture:
    """Capture actual gate outputs as [batch, flow, layer, action, top-k]."""

    def __init__(self, policy, batch_size):
        self.layers = policy._routing_layers
        self.batch_size = batch_size
        self.records = []
        self.handles = []

    def __enter__(self):
        try:
            for layer, gate in self.layers:
                self.handles.append(gate.register_forward_hook(self.hook(layer)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def hook(self, layer):
        def capture(_module, inputs, output):
            batch, sequence = inputs[0].shape[:2]
            if batch != self.batch_size or sequence < 10:
                raise RuntimeError("Unexpected routing batch or token count")
            ids, weights, _ = output
            self.records.append((layer,
                ids.reshape(batch, sequence, 4)[:, -10:].detach().clone(),
                weights.reshape(batch, sequence, 4)[:, -10:].detach().clone()))
        return capture

    def arrays(self):
        import torch

        layers = [layer for layer, _ in self.layers]
        if [r[0] for r in self.records] != layers * 10:
            raise RuntimeError("HB capture is missing a flow step or has incorrect layer order")
        shape = (10, len(layers), self.batch_size, 10, 4)
        ids = torch.stack([r[1] for r in self.records]).reshape(shape).permute(2, 0, 1, 3, 4)
        weights = torch.stack([r[2] for r in self.records]).reshape(shape).permute(2, 0, 1, 3, 4)
        return (ids.to(device="cpu", dtype=torch.int16).numpy(),
                weights.to(device="cpu", dtype=torch.float32).numpy(),
                np.asarray(layers, dtype=np.int16))


def stack_tree(rows):
    import torch

    first = rows[0]
    if isinstance(first, dict):
        if any(row.keys() != first.keys() for row in rows):
            raise ValueError("Batch observation structures differ")
        return {key: stack_tree([row[key] for row in rows]) for key in first}
    if isinstance(first, (list, tuple)):
        if any(len(row) != len(first) for row in rows):
            raise ValueError("Batch observation lengths differ")
        return type(first)(stack_tree([row[i] for row in rows]) for i in range(len(first)))
    return torch.stack([torch.as_tensor(row) for row in rows])


def infer_batch(policy, requests):
    import torch
    from himoe_libero_bridge.protocol import validate_action_response, validate_observation
    from moevla.models.model import from_dict, preprocess_observation_and_to_device
    from moevla.policies.policy import tree_map

    if not requests:
        raise ValueError("An empty batch is not supported")
    core = policy._policy
    observations, states, noises = [], [], []
    for request in requests:
        request = validate_observation(request)
        inputs = tree_map(lambda x: x, request)
        if not inputs.pop("routing/capture", False):
            raise ValueError("This probe requires top-4 capture on every request")
        noises.append(np.array(inputs.pop("flow/noise"), dtype=np.float32, copy=True))
        inputs = core._input_transform(inputs)
        observations.append(from_dict(inputs))
        states.append(np.array(inputs["state"], copy=True))
    observation = preprocess_observation_and_to_device(stack_tree(observations), train=False,
                                                       device=torch.device("cuda:0"))
    noise = torch.tensor(np.stack(noises), dtype=torch.float32, device="cuda:0")
    with BatchRouteCapture(policy, len(requests)) as capture:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = core._sample_actions(observation["images"], observation["image_masks"],
                observation["tokenized_prompt"], observation["tokenized_prompt_mask"],
                observation["state"], observation["data_mask"], noise=noise)
        actions = actions.cpu().numpy()
    ids, weights, layers = capture.arrays()
    responses = []
    for i in range(len(requests)):
        response = core._output_transform({"state": states[i], "actions": actions[i]})
        response.update({"routing/expert_ids": ids[i], "routing/expert_weights": weights[i],
                         "routing/layer_indices": layers,
                         "flow/noise_sha256": hashlib.sha256(noises[i].tobytes()).hexdigest()})
        responses.append(validate_action_response(response))
    return responses


def compare(expected, actual):
    if len(expected) != len(actual):
        raise RuntimeError("Batch response count changed")
    errors, exact = {}, {}
    for key in FIELDS:
        left = np.stack([r[key] for r in expected])
        right = np.stack([r[key] for r in actual])
        if left.shape != right.shape:
            raise RuntimeError("Response shape changed for " + key)
        if not np.isfinite(right).all():
            raise RuntimeError("Non-finite response for " + key)
        exact[key] = bool(np.array_equal(left, right))
        errors[key] = float(np.abs(left.astype(np.float64) - right).max())
    ids_a = np.stack([r["routing/expert_ids"] for r in expected])
    ids_b = np.stack([r["routing/expert_ids"] for r in actual])
    set_changed = np.any(np.sort(ids_a, axis=-1) != np.sort(ids_b, axis=-1), axis=-1)
    action_a = np.stack([r["actions"] for r in expected])
    action_b = np.stack([r["actions"] for r in actual])
    noise_equal = all(a["flow/noise_sha256"] == b["flow/noise_sha256"] for a, b in zip(expected, actual))
    if not noise_equal:
        raise RuntimeError("Flow noise identity changed")
    return dict(samples=len(expected), all_fields_exact=all(exact.values()), exact=exact,
        max_abs_errors=errors, action_mean_abs_error=float(np.abs(action_a - action_b).mean()),
        action_max_abs_per_dimension=np.abs(action_a - action_b).max(axis=(0, 1)).tolist(),
        route_sets_changed=int(set_changed.sum()), route_sets_total=int(set_changed.size),
        route_set_changed_percent=100 * float(set_changed.mean()), noise_hashes_equal=noise_equal)


def measure(policy, requests, gpu, seconds, *, original=False):
    import torch
    from serve_model_matrix import _trim_heap

    def query():
        if original:
            result = [policy.infer(requests[0])]
        else:
            result = infer_batch(policy, requests)
        _trim_heap()
        return result

    for _ in range(2):
        query()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, telemetry_errors, latencies = [], [], []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(probe.telemetry((gpu,)))
            except Exception as error:
                telemetry_errors.append(repr(error))
            stop.wait(.5)

    thread = threading.Thread(target=monitor, name="batch-gpu-telemetry")
    started = time.monotonic()
    thread.start()
    try:
        while time.monotonic() - started < seconds:
            tick = time.monotonic()
            query()
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - tick)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("GPU telemetry did not stop")
    if telemetry_errors or not samples:
        raise RuntimeError("GPU telemetry failed: " + str(telemetry_errors))
    rows = [r for sample in samples for r in sample["gpus"]]
    return dict(mode="original_single" if original else "batch_adapter", batch_size=len(requests),
        queries=len(latencies) * len(requests), batches=len(latencies), seconds=elapsed,
        queries_per_second=len(latencies) * len(requests) / elapsed,
        batch_latency_mean_seconds=statistics.mean(latencies),
        batch_latency_p95_seconds=float(np.percentile(latencies, 95)),
        utilization_mean_percent=statistics.mean(r["utilization_percent"] for r in rows),
        power_mean_w=statistics.mean(r["power_w"] for r in rows),
        total_gpu_memory_peak_mib=max(r["memory_mib"] for r in rows),
        temporary_process_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        temporary_process_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        gpu_samples=samples, batch_latencies_seconds=latencies)


def profile_single(policy, request):
    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("goal_single_with_top4"):
            policy.infer(request)
        torch.cuda.synchronize()
    return [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                 total_cpu_us=e.cpu_time_total, self_device_us=e.self_device_time_total,
                 total_device_us=e.device_time_total) for e in prof.key_averages()]


def run(args):
    if args.gpu not in probe.ALLOWED_GPUS:
        raise ValueError("GPU 6 is excluded")
    if args.seconds <= 0 or not args.batches or any(b not in (1, 2, 4, 8) for b in args.batches):
        raise ValueError("Use a positive duration and batches 1,2,4,8")
    free = int(subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip())
    if free < 35000:
        raise RuntimeError("Need 35000 MiB spare memory for the temporary model and batch probe")
    uuid = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Expected exactly one physical GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    load_args = bundle.build_parser().parse_args(["--gpu", str(args.gpu), "--base-port", "8990"])
    bundle._install_source_paths(load_args)
    import torch

    if torch.cuda.device_count() != 1 or str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("Physical GPU UUID isolation failed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="loading", started_utc=probe.now(), gpu=args.gpu, gpu_uuid=uuid,
        excluded_gpu=6, pid=os.getpid(), original_servers_replaced=False, hidden_capture=False,
        full_hb_capture=False, simulator=False, alarm_parameters_modified=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), phases=[], comparisons=[])
    probe.write_json(args.output / "results.json", report)
    try:
        endpoint = bundle.model_endpoints(load_args.base_port)[0]
        policy = bundle._load_policies(load_args, (endpoint,))["goal"]
        probe.check_checkpoint(policy.metadata, "goal")
        report["policy_metadata"] = policy.metadata
        seeds = list(range(97000, 97008))
        requests = [probe.observation("goal", seed, capture=True) for seed in seeds]
        prompts = ("pick up the object and place it on the table", "open the drawer",
                   "put the red bowl on the plate", "turn on the stove")
        for i, request in enumerate(requests):
            request["prompt"] = prompts[i % len(prompts)]
        report["input_seeds"] = seeds
        report["input_prompts"] = [r["prompt"] for r in requests]
        baseline = [policy.infer(request) for request in requests]
        with contextlib.ExitStack() as stack:
            connection, metadata = probe.open_endpoint(stack, "127.0.0.1", args.gpu, "goal")
            probe._verify_bundle_identity(metadata, probe.port_for(args.gpu, "goal"), args.gpu,
                                           probe.ALLOWED_GPUS.index(args.gpu), "goal")
            for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                        "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
                if metadata[key] != policy.metadata[key]:
                    raise RuntimeError("Original server identity differs: " + key)
            reference = [probe.infer(connection, request, "goal")[0] for request in requests[:2]]
            report["original_service_comparison"] = compare(reference, baseline[:2])
            report["original_pid"] = metadata["bundle_process_pid"]
        repeated = [policy.infer(request) for request in requests]
        report["single_repeat_comparison"] = compare(baseline, repeated)
        adapter_single = [infer_batch(policy, [request])[0] for request in requests]
        report["adapter_single_comparison"] = compare(baseline, adapter_single)
        for key in ("original_service_comparison", "single_repeat_comparison", "adapter_single_comparison"):
            if not report[key]["all_fields_exact"]:
                raise RuntimeError("Batch-1 equivalence failed: " + key)
        print("EXACT: original service, local policy, repeated inputs, batch-1 adapter", flush=True)
        report["status"] = "measuring"
        configs = [(True, 1)] + [(False, b) for b in args.batches]
        for original, batch in configs:
            print(f"MEASURE original={original} batch={batch}", flush=True)
            phase = measure(policy, requests[:batch], args.gpu, args.seconds, original=original)
            report["phases"].append(phase)
            if not original:
                outputs = [row for start in range(0, len(requests), batch)
                           for row in infer_batch(policy, requests[start:start + batch])]
                same_batch_repeat = infer_batch(policy, requests[:batch])
                reversed_outputs = list(reversed(infer_batch(policy, list(reversed(requests[:batch])))))
                changed_companion = [requests[0]] + requests[-(batch - 1):] if batch > 1 else requests[:1]
                comparison = dict(batch_size=batch, vs_single=compare(baseline, outputs),
                    same_batch_repeat=compare(outputs[:batch], same_batch_repeat),
                    order_permutation=compare(outputs[:batch], reversed_outputs),
                    companion_change_first_row=compare(outputs[:1], infer_batch(policy, changed_companion)[:1]))
                report["comparisons"].append(comparison)
                arrays = {f"sample_{i}_{key.replace('/', '_')}": row[key]
                          for i, row in enumerate(outputs) for key in FIELDS}
                np.savez_compressed(args.output / f"batch{batch}_outputs.npz", **arrays)
                print("CHECK " + json.dumps(comparison), flush=True)
            probe.write_json(args.output / "results.json", report)
            print("RESULT " + json.dumps({k: v for k, v in phase.items()
                if k not in ("gpu_samples", "batch_latencies_seconds")}), flush=True)
        report["profile_single"] = profile_single(policy, requests[0])
        report["post_measure_single_comparison"] = compare(baseline,
            [policy.infer(request) for request in requests])
        if not report["post_measure_single_comparison"]["all_fields_exact"]:
            raise RuntimeError("Original single inference changed after batch measurements")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_utc"] = probe.now()
        probe.write_json(args.output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=probe.ALLOWED_GPUS, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=HERE / "design/batch_efficiency_20260908")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parser.parse_args())
