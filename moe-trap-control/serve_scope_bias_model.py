#!/usr/bin/env python3
"""Isolated replicas serving full-scope HB gate biases (GPUs 4 and 5 only)."""

import asyncio
import concurrent.futures
import gc
import logging
import os
import signal
import subprocess
import threading

from serve_isolated_model import bundle, DeviceBoundPolicy, serve_shared
from scope_bias_control import ScopeBiasPolicy


def serve(args, endpoints, policies, uuid, replica, owner):
    import torch
    actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != uuid.removeprefix("GPU-"):
        raise ValueError("Isolated GPU UUID mismatch")
    if replica:
        buffers = {}
        for policy in policies.values():
            for module in policy._policy.model.modules():
                for name, value in module._buffers.items():
                    if value is not None:
                        buffers.setdefault(id(value), value.clone())
                        module._buffers[name] = buffers[id(value)]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    bound = {}
    for endpoint in endpoints:
        policy = policies[endpoint.name]
        policy.metadata.update(isolated_process=True, isolated_models=[e.name for e in endpoints],
            isolated_gpu_uuid=uuid, cuda_memory_limit_mib=None, bundle_process_pid=os.getpid(),
            bundle_port=endpoint.port, replica_index=replica, weights_owner_pid=owner,
            shared_cuda_parameters=args.replicas > 1,
            cuda_mps_pipe_directory=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
        bound[endpoint.name] = DeviceBoundPolicy(ScopeBiasPolicy(policy), 0, executor)
    try:
        return asyncio.run(serve_shared(args, endpoints, bound))
    finally:
        executor.shutdown(wait=True)
        torch.cuda.synchronize()


def child(args, endpoints, policies, uuid, replica, owner):
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        result = serve(args, endpoints, policies, uuid, replica, owner)
    finally:
        policies.clear()
        gc.collect()
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    raise SystemExit(result)


def main():
    parser = bundle.build_parser()
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    if args.gpu not in (4, 5):
        parser.error("This experiment is restricted to physical GPUs 4,5")
    uuid = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise ValueError("Expected one physical GPU")
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    bundle._install_source_paths(args)
    logging.basicConfig(level=logging.INFO, force=True)
    import torch
    endpoints = tuple(e for e in bundle.model_endpoints(args.base_port) if e.name == "long")
    policies = bundle._load_policies(args, endpoints)
    children, stopping = [], threading.Event()
    try:
        for policy in policies.values():
            policy._policy.model.eval().requires_grad_(False)
        for replica in range(1, args.replicas):
            ports = tuple(bundle.ModelEndpoint(e.name, e.kind, e.port + 5 * replica) for e in endpoints)
            process = torch.multiprocessing.get_context("spawn").Process(target=child,
                args=(args, ports, policies, uuid, replica, os.getpid()))
            process.start()
            children.append(process)
        def watch():
            while not stopping.wait(.5):
                if any(not p.is_alive() for p in children):
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        threading.Thread(target=watch, daemon=True).start()
        return serve(args, endpoints, policies, uuid, 0, os.getpid())
    finally:
        stopping.set()
        for p in children:
            if p.is_alive():
                p.terminate()
        for p in children:
            p.join(timeout=15)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
