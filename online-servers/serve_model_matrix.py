#!/usr/bin/env python3
"""Serve one complete HiMoE-VLA task bundle on every selected GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import logging
import os
import pathlib
import queue
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import serve_model_bundle as bundle


try:
    _MALLOC_TRIM = ctypes.CDLL(None).malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (AttributeError, OSError):
    _MALLOC_TRIM = None
_HEAP_TRIM_LOCK = threading.Lock()


@dataclass(frozen=True)
class MatrixEndpoint:
    physical_gpu: int
    logical_gpu: int
    model: bundle.ModelEndpoint

    @property
    def key(self) -> tuple[int, str]:
        return (self.physical_gpu, self.model.name)


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPUs must be comma-separated integers") from error
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU is required")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    if any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise argparse.ArgumentTypeError("managed GPU indices are 0-7")
    return gpus


def matrix_endpoints(
    gpus: tuple[int, ...], port_base: int
) -> tuple[MatrixEndpoint, ...]:
    records = []
    for logical_gpu, physical_gpu in enumerate(gpus):
        base_port = port_base + physical_gpu * 10
        records.extend(
            MatrixEndpoint(physical_gpu, logical_gpu, endpoint)
            for endpoint in bundle.model_endpoints(base_port)
        )
    return tuple(records)


def _trim_heap() -> None:
    if _MALLOC_TRIM is not None:
        with _HEAP_TRIM_LOCK:
            _MALLOC_TRIM(0)


class DeviceBoundPolicy:
    """Run all policies for one GPU through its single CUDA worker thread."""

    def __init__(
        self,
        policy: Any,
        logical_gpu: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self._policy = policy
        self._logical_gpu = logical_gpu
        self._executor = executor
        self.metadata = policy.metadata
        self.backend_name = getattr(policy, "backend_name", "himoe")

    def _infer_on_device(self, observation: dict[str, Any]) -> dict[str, Any]:
        import torch

        try:
            with torch.cuda.device(self._logical_gpu):
                return self._policy.infer(observation)
        finally:
            _trim_heap()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._executor.submit(self._infer_on_device, observation).result()


def _load_matrix(
    args: argparse.Namespace, records: tuple[MatrixEndpoint, ...]
) -> dict[tuple[int, str], DeviceBoundPolicy]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != len(args.gpus):
        raise RuntimeError(
            "matrix requires %d visible CUDA devices, found %d; "
            "CUDA_VISIBLE_DEVICES=%r"
            % (len(args.gpus), torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES"))
        )

    policies: dict[tuple[int, str], DeviceBoundPolicy] = {}
    executors = {
        logical_gpu: concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="himoe-cuda%d" % logical_gpu,
        )
        for logical_gpu in range(len(args.gpus))
    }
    for record in records:
        torch.cuda.set_device(record.logical_gpu)
        load_args = argparse.Namespace(
            **vars(args),
            gpu=record.physical_gpu,
            logical_gpu=record.logical_gpu,
        )
        before = torch.cuda.memory_allocated(record.logical_gpu)
        started = time.monotonic()
        logging.info(
            "loading gpu%d/logical%d %-7s port=%d (%s)",
            record.physical_gpu,
            record.logical_gpu,
            record.model.name,
            record.model.port,
            bundle._cgroup_memory(),
        )
        if record.model.kind == "libero":
            policy = bundle._load_libero(load_args, record.model)
        else:
            policy = bundle._load_calvin(load_args, record.model)
        policies[record.key] = DeviceBoundPolicy(
            policy, record.logical_gpu, executors[record.logical_gpu]
        )
        gc.collect()
        torch.cuda.empty_cache()
        _trim_heap()
        after = torch.cuda.memory_allocated(record.logical_gpu)
        logging.info(
            "loaded gpu%d %-7s in %.1fs; CUDA +%.2f GiB, total %.2f GiB; %s",
            record.physical_gpu,
            record.model.name,
            time.monotonic() - started,
            bundle._gib(after - before),
            bundle._gib(after),
            bundle._cgroup_memory(),
        )
    return policies


def _serve_matrix(
    args: argparse.Namespace,
    records: tuple[MatrixEndpoint, ...],
    policies: dict[tuple[int, str], DeviceBoundPolicy],
) -> int:
    failures: queue.Queue[tuple[str, str]] = queue.Queue()
    stopping = threading.Event()

    def run(record: MatrixEndpoint) -> None:
        label = "gpu%d/%s" % (record.physical_gpu, record.model.name)
        try:
            server = bundle._make_server(args, record.model, policies[record.key])
            logging.info(
                "serving %-13s on ws://%s:%d",
                label,
                args.host,
                record.model.port,
            )
            server.serve_forever()
            failures.put((label, "server returned unexpectedly"))
        except BaseException:
            failures.put((label, traceback.format_exc()))

    def stop(signum: int, _frame: Any) -> None:
        logging.info("received signal %d; stopping model matrix", signum)
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for record in records:
        threading.Thread(
            target=run,
            args=(record,),
            name="himoe-gpu%d-%s" % (record.physical_gpu, record.model.name),
            daemon=True,
        ).start()

    while not stopping.wait(0.5):
        try:
            name, detail = failures.get_nowait()
        except queue.Empty:
            continue
        logging.error("%s endpoint failed:\n%s", name, detail)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = pathlib.Path("/home/jovyan")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=8800)
    parser.add_argument("--libero-cache", default=str(root / ".cache/himoe-libero-bridge"))
    parser.add_argument("--calvin-cache", default=str(root / ".cache/himoe-calvin-alignment"))
    parser.add_argument(
        "--upstream-root",
        default=str(root / ".cache/himoe-libero-bridge/upstream/HiMoE-VLA"),
    )
    parser.add_argument(
        "--calvin-root",
        default=str(root / ".cache/himoe-calvin-alignment/upstream/calvin"),
    )
    parser.add_argument("--bridge-src", default=str(root / "work/himoe-libero-wrist-fix/src"))
    parser.add_argument(
        "--route-capture-src", default=str(root / "work/himoe-vla/himoe-route-capture")
    )
    parser.add_argument(
        "--calvin-src", default=str(root / "work/himoe-calvin-alignment/src")
    )
    parser.add_argument(
        "--openpi-src",
        default=str(
            root
            / ".cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src"
        ),
    )
    parser.add_argument("--libero-wrist-layout", default="checkpoint-right")
    parser.add_argument("--calvin-wrist-layout", default="released-left")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    bundle._install_source_paths(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        force=True,
    )
    records = matrix_endpoints(args.gpus, args.port_base)
    logging.info(
        "starting model matrix pid=%d physical_gpus=%s endpoints=%d",
        os.getpid(),
        args.gpus,
        len(records),
    )
    policies = _load_matrix(args, records)
    return _serve_matrix(args, records, policies)


if __name__ == "__main__":
    raise SystemExit(main())
