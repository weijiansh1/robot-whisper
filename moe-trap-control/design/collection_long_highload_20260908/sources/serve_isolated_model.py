#!/usr/bin/env python3
"""Serve selected existing checkpoints in one process on an allowed GPU."""

from __future__ import annotations

import concurrent.futures
import asyncio
import gc
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "online-servers"))
import serve_model_bundle as bundle
from serve_model_matrix import DeviceBoundPolicy

ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)


async def serve_shared(args, endpoints, policies):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    servers = [asyncio.create_task(bundle._make_server(args, endpoint, policies[endpoint.name]).run())
               for endpoint in endpoints]
    waiter = asyncio.create_task(stop.wait())
    try:
        completed, _ = await asyncio.wait([waiter, *servers], return_when=asyncio.FIRST_COMPLETED)
        for task in completed - {waiter}:
            task.result()
            raise RuntimeError("Shared replica server returned unexpectedly")
        return 0
    finally:
        for task in [waiter, *servers]:
            task.cancel()
        await asyncio.gather(waiter, *servers, return_exceptions=True)


def serve_replica(args, endpoints, policies, uuid, replica, owner_pid):
    import torch

    actual = str(torch.cuda.get_device_properties(0).uuid)
    if torch.cuda.device_count() != 1 or actual.removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("Replica CUDA UUID differs from the allowed physical GPU")
    if replica:
        # Parameters are read-only CUDA IPC tensors; mutable buffers stay process-local.
        buffers = {}
        for policy in policies.values():
            for module in policy._policy.model.modules():
                for name, value in module._buffers.items():
                    if value is not None:
                        if id(value) not in buffers:
                            buffers[id(value)] = value.clone()
                        module._buffers[name] = buffers[id(value)]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="isolated-cuda0")
    bound = {}
    for endpoint in endpoints:
        policy = policies[endpoint.name]
        policy.metadata.update(isolated_process=True, isolated_models=[e.name for e in endpoints],
            isolated_gpu_uuid=uuid, cuda_memory_limit_mib=args.cuda_memory_limit_mib,
            bundle_process_pid=os.getpid(), bundle_port=endpoint.port, replica_index=replica,
            weights_owner_pid=owner_pid, shared_cuda_parameters=args.replicas > 1)
        if args.full_hb_capture:
            from collection_routes import FullHBPolicy
            policy = FullHBPolicy(policy)
        bound[endpoint.name] = DeviceBoundPolicy(policy, 0, executor)
    try:
        if args.replicas > 1:
            return asyncio.run(serve_shared(args, endpoints, bound))
        return bundle._serve(args, endpoints, bound)
    finally:
        executor.shutdown(wait=True)


def child_replica(args, endpoints, policies, uuid, owner_pid):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    try:
        result = serve_replica(args, endpoints, policies, uuid, 1, owner_pid)
    finally:
        policies.clear()
        gc.collect()
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    raise SystemExit(result)


def main():
    parser = bundle.build_parser()
    parser.description = __doc__
    parser.add_argument("--models", nargs="+", choices=bundle.MODEL_ORDER, default=["goal"])
    parser.add_argument("--full-hb-capture", action="store_true")
    parser.add_argument("--cuda-memory-limit-mib", type=int)
    parser.add_argument("--replicas", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.gpu not in ALLOWED_GPUS:
        parser.error("Physical GPU 6 is excluded; allowed GPUs are 0,1,2,3,4,5,7")
    if len(set(args.models)) != len(args.models):
        parser.error("Models must be unique")
    if args.full_hb_capture and "calvin" in args.models:
        parser.error("Full HB collection is currently scoped to LIBERO policies")
    if args.cuda_memory_limit_mib is not None and args.cuda_memory_limit_mib <= 0:
        parser.error("CUDA memory limit must be positive")
    if args.replicas > 1 and (len(args.models) != 1 or "calvin" in args.models or args.cuda_memory_limit_mib):
        parser.error("Shared replicas require one LIBERO model and the default allocator limit")
    uuid = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
        "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Expected exactly one GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    bundle._install_source_paths(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Isolated process must see exactly one CUDA device")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("CUDA UUID does not match the requested physical GPU")
    if args.cuda_memory_limit_mib is not None:
        total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
        if args.cuda_memory_limit_mib > total_mib:
            parser.error("CUDA memory limit exceeds device memory")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_limit_mib / total_mib, 0)
    endpoints = tuple(e for e in bundle.model_endpoints(args.base_port) if e.name in args.models)
    policies = bundle._load_policies(args, endpoints)
    child = None
    stopping = threading.Event()
    unexpected_exit = []
    if args.replicas > 1:
        for policy in policies.values():
            policy._policy.model.eval().requires_grad_(False)
        child_endpoints = tuple(bundle.ModelEndpoint(e.name, e.kind, e.port + 5) for e in endpoints)
        child = torch.multiprocessing.get_context("spawn").Process(target=child_replica,
            args=(args, child_endpoints, policies, uuid, os.getpid()), name="cuda-ipc-replica")
        child.start()

        def watch_child():
            while not stopping.wait(.5):
                if not child.is_alive():
                    unexpected_exit.append(child.exitcode)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return

        threading.Thread(target=watch_child, daemon=True).start()
    try:
        result = serve_replica(args, endpoints, policies, uuid, 0, os.getpid())
        return 1 if unexpected_exit else result
    finally:
        stopping.set()
        if child is not None:
            # Keep the CUDA storage owner alive until the IPC consumer has exited.
            if child.is_alive():
                child.terminate()
            child.join(timeout=25)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
