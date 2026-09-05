#!/usr/bin/env python3
"""Load every released HiMoE-VLA policy on one GPU and serve five ports."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pathlib
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any


MODEL_ORDER = ("goal", "spatial", "object", "long", "calvin")
LIBERO_CHECKPOINTS = {
    "goal": "HiMoE-VLA-Libero-Goal",
    "spatial": "HiMoE-VLA-Libero-Spatial",
    "object": "HiMoE-VLA-Libero-Object",
    "long": "HiMoE-VLA-Libero-10",
}
CALVIN_CHECKPOINT = "HiMoE-VLA-CALVIN-D"


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    kind: str
    port: int


def model_endpoints(base_port: int) -> tuple[ModelEndpoint, ...]:
    if not 1 <= base_port <= 65531:
        raise ValueError("base port must leave room for five consecutive ports")
    return tuple(
        ModelEndpoint(
            name=name,
            kind="calvin" if name == "calvin" else "libero",
            port=base_port + offset,
        )
        for offset, name in enumerate(MODEL_ORDER)
    )


def _gib(value: int) -> float:
    return value / float(1024**3)


def _cgroup_memory() -> str:
    root = pathlib.Path("/sys/fs/cgroup")
    try:
        maximum = (root / "memory.max").read_text(encoding="ascii").strip()
        current = int((root / "memory.current").read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return "cgroup memory unavailable"
    return "cgroup current=%.2f GiB max=%s" % (
        _gib(current),
        maximum if maximum == "max" else "%.2f GiB" % _gib(int(maximum)),
    )


def _install_source_paths(args: argparse.Namespace) -> None:
    # Keep the audited bridge ahead of the compatibility copy bundled with CALVIN.
    for source in (
        args.openpi_src,
        args.calvin_src,
        args.bridge_src,
        args.route_capture_src,
    ):
        resolved = str(pathlib.Path(source).expanduser().resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)


def _annotate_policy(
    policy: Any,
    audit: Any,
    endpoint: ModelEndpoint,
    physical_gpu: int,
    logical_gpu: int = 0,
) -> None:
    metadata = policy.metadata
    metadata["checkpoint_load_audit"] = audit.as_metadata()
    metadata.update(
        {
            "bundle_schema": "himoe-vla-five-model-gpu-bundle/v1",
            "bundle_model": endpoint.name,
            "bundle_physical_gpu": physical_gpu,
            "bundle_logical_gpu": logical_gpu,
            "bundle_port": endpoint.port,
            "bundle_process_pid": os.getpid(),
        }
    )


def _load_libero(args: argparse.Namespace, endpoint: ModelEndpoint) -> Any:
    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_libero_bridge.suites import get_suite
    from himoe_low_memory_load import low_memory_himoe_load

    checkpoint = pathlib.Path(args.libero_cache) / "checkpoints" / LIBERO_CHECKPOINTS[
        endpoint.name
    ]
    suite = get_suite(endpoint.name)
    logical_gpu = getattr(args, "logical_gpu", 0)
    with low_memory_himoe_load(
        checkpoint,
        args.upstream_root,
        suite.train_config,
        target_device="cuda:%d" % logical_gpu,
    ) as audit:
        policy = HiMoEPolicy(
            checkpoint_dir=str(checkpoint),
            suite=endpoint.name,
            upstream_root=args.upstream_root,
            require_cuda=True,
            libero_wrist_layout=args.libero_wrist_layout,
        )
    expected = str((checkpoint / "pytorch_model.pth").resolve())
    if audit.checkpoint_path != expected:
        raise RuntimeError("low-memory loader checkpoint identity mismatch")
    _annotate_policy(policy, audit, endpoint, args.gpu, logical_gpu)
    return policy


def _load_calvin(args: argparse.Namespace, endpoint: ModelEndpoint) -> Any:
    from himoe_calvin_alignment import policy_server
    from himoe_calvin_alignment.constants import TRAIN_CONFIG
    from himoe_low_memory_load import low_memory_himoe_load

    checkpoint = pathlib.Path(args.calvin_cache) / "checkpoints" / CALVIN_CHECKPOINT
    policy_args = argparse.Namespace(
        checkpoint_dir=str(checkpoint),
        upstream_root=args.upstream_root,
        calvin_root=args.calvin_root,
        wrist_layout=args.calvin_wrist_layout,
        host=args.host,
        port=endpoint.port,
    )
    logical_gpu = getattr(args, "logical_gpu", 0)
    with low_memory_himoe_load(
        checkpoint,
        args.upstream_root,
        TRAIN_CONFIG,
        target_device="cuda:%d" % logical_gpu,
    ) as audit:
        policy = policy_server.create_policy(policy_args)
    expected = str((checkpoint / "pytorch_model.pth").resolve())
    if audit.checkpoint_path != expected:
        raise RuntimeError("low-memory loader checkpoint identity mismatch")
    _annotate_policy(policy, audit, endpoint, args.gpu, logical_gpu)
    policy.metadata.update(
        {
            "server_instance_id": uuid.uuid4().hex,
            "server_pid": os.getpid(),
            "server_started_unix_ns": time.time_ns(),
        }
    )
    return policy


def _load_policies(
    args: argparse.Namespace,
    endpoints: tuple[ModelEndpoint, ...],
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "bundle requires exactly one visible CUDA device; CUDA_VISIBLE_DEVICES=%r"
            % os.environ.get("CUDA_VISIBLE_DEVICES")
        )
    torch.cuda.set_device(0)
    policies: dict[str, Any] = {}
    for endpoint in endpoints:
        before = torch.cuda.memory_allocated(0)
        started = time.monotonic()
        logging.info(
            "loading %s on physical GPU %d for port %d (%s)",
            endpoint.name,
            args.gpu,
            endpoint.port,
            _cgroup_memory(),
        )
        if endpoint.kind == "libero":
            policy = _load_libero(args, endpoint)
        else:
            policy = _load_calvin(args, endpoint)
        policies[endpoint.name] = policy
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(0)
        logging.info(
            "loaded %-7s in %.1fs; CUDA +%.2f GiB, total %.2f GiB; %s",
            endpoint.name,
            time.monotonic() - started,
            _gib(after - before),
            _gib(after),
            _cgroup_memory(),
        )
    return policies


def _make_server(
    args: argparse.Namespace,
    endpoint: ModelEndpoint,
    policy: Any,
) -> Any:
    if endpoint.kind == "libero":
        from himoe_libero_bridge.server import PolicyServer

        return PolicyServer(
            policy,
            args.host,
            endpoint.port,
            getattr(policy, "backend_name", "himoe"),
        )
    from moevla.serving.websocket_policy_server import WebsocketPolicyServer

    return WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=endpoint.port,
        metadata=policy.metadata,
    )


def _serve(
    args: argparse.Namespace,
    endpoints: tuple[ModelEndpoint, ...],
    policies: dict[str, Any],
) -> int:
    failures: queue.Queue[tuple[str, str]] = queue.Queue()
    stopping = threading.Event()

    def run(endpoint: ModelEndpoint) -> None:
        try:
            server = _make_server(args, endpoint, policies[endpoint.name])
            logging.info(
                "serving %-7s on ws://%s:%d", endpoint.name, args.host, endpoint.port
            )
            server.serve_forever()
            failures.put((endpoint.name, "server returned unexpectedly"))
        except BaseException:
            failures.put((endpoint.name, traceback.format_exc()))

    def stop(signum: int, _frame: Any) -> None:
        logging.info("received signal %d; stopping GPU %d bundle", signum, args.gpu)
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for endpoint in endpoints:
        threading.Thread(
            target=run,
            args=(endpoint,),
            name="himoe-%s-server" % endpoint.name,
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
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument(
        "--libero-cache", default=str(root / ".cache/himoe-libero-bridge")
    )
    parser.add_argument(
        "--calvin-cache", default=str(root / ".cache/himoe-calvin-alignment")
    )
    parser.add_argument(
        "--upstream-root",
        default=str(root / ".cache/himoe-libero-bridge/upstream/HiMoE-VLA"),
    )
    parser.add_argument(
        "--calvin-root",
        default=str(root / ".cache/himoe-calvin-alignment/upstream/calvin"),
    )
    parser.add_argument(
        "--bridge-src", default=str(root / "work/himoe-libero-wrist-fix/src")
    )
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
    endpoints = model_endpoints(args.base_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _install_source_paths(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        force=True,
    )
    logging.info(
        "starting five-model bundle pid=%d physical_gpu=%d ports=%d-%d",
        os.getpid(),
        args.gpu,
        endpoints[0].port,
        endpoints[-1].port,
    )
    policies = _load_policies(args, endpoints)
    return _serve(args, endpoints, policies)


if __name__ == "__main__":
    raise SystemExit(main())
