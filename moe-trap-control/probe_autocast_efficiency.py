#!/usr/bin/env python3
"""Measure persistent autocast cache on a temporary, immutable batch-1 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess

import probe_batch_efficiency as batch_probe

probe = batch_probe.probe
bundle = batch_probe.bundle
HERE = Path(__file__).resolve().parent


def run(args):
    if args.gpu not in probe.ALLOWED_GPUS or args.seconds <= 0:
        raise ValueError("An allowed GPU and positive duration are required")
    free = int(subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip())
    if free < 35000:
        raise RuntimeError("Need 35000 MiB spare memory for the temporary model")
    uuid = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Expected exactly one GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    load_args = bundle.build_parser().parse_args(["--gpu", str(args.gpu), "--base-port", "8990"])
    bundle._install_source_paths(load_args)
    import torch

    if torch.cuda.device_count() != 1 or str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("Physical GPU UUID isolation failed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="loading", started_utc=probe.now(), pid=os.getpid(), gpu=args.gpu,
        gpu_uuid=uuid, excluded_gpu=6, original_servers_replaced=False, hidden_capture=False,
        full_hb_capture=False, simulator=False, alarm_parameters_modified=False,
        phases=[], comparisons=[],
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), Path(batch_probe.__file__))})
    probe.write_json(args.output / "results.json", report)
    try:
        endpoint = bundle.model_endpoints(load_args.base_port)[0]
        policy = bundle._load_policies(load_args, (endpoint,))["goal"]
        probe.check_checkpoint(policy.metadata, "goal")
        report["policy_metadata"] = policy.metadata
        params = list(policy._policy.model.parameters())
        report["parameters_requiring_grad"] = sum(p.numel() for p in params if p.requires_grad)
        report["parameters_total"] = sum(p.numel() for p in params)
        seeds = list(range(99000, 99008))
        requests = [probe.observation("goal", seed, capture=True) for seed in seeds]
        report["input_seeds"] = seeds
        references = [policy.infer(request) for request in requests]

        def check(mode):
            result = batch_probe.compare(references, [policy.infer(request) for request in requests])
            report["comparisons"].append(dict(mode=mode, **result))
            probe.write_json(args.output / "results.json", report)
            if not result["all_fields_exact"]:
                raise RuntimeError("Autocast scope changed actions or routing: " + mode)
            print("EXACT " + mode, flush=True)

        def measure(mode):
            print("MEASURE " + mode, flush=True)
            phase = batch_probe.measure(policy, requests[:1], args.gpu, args.seconds, original=True)
            phase["autocast_scope"] = mode
            report["phases"].append(phase)
            probe.write_json(args.output / "results.json", report)
            print("RESULT " + json.dumps({k: v for k, v in phase.items()
                if k not in ("gpu_samples", "batch_latencies_seconds")}), flush=True)

        report["status"] = "measuring"
        measure("per_query_before")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            check("persistent_before")
            measure("persistent_first")
            measure("persistent_second")
            check("persistent_after")
        check("per_query_restored")
        measure("per_query_after")
        # CUPTI instrumentation can affect later timing; all timed phases finish first.
        report["profile_per_query"] = batch_probe.profile_single(policy, requests[0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            check("persistent_profile_warmup")
            report["profile_persistent"] = batch_probe.profile_single(policy, requests[0])
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
    parser.add_argument("--output", type=Path, default=HERE / "design/autocast_efficiency_20260908")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parser.parse_args())
