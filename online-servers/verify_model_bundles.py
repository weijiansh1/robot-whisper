#!/usr/bin/env python3
"""Verify metadata and real inference on every HiMoE-VLA bundle endpoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np


MODELS = ("goal", "spatial", "object", "long", "calvin")


def _install_paths() -> None:
    roots = (
        "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src",
        "/home/jovyan/work/himoe-libero-wrist-fix/src",
    )
    for root in reversed(roots):
        if root not in sys.path:
            sys.path.insert(0, root)


def _observation(state_dim: int) -> dict:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((state_dim,), dtype=np.float32),
        "prompt": "move the robot for a bundle health check",
    }


def _verify_libero(
    host: str, port: int, gpu: int, logical_gpu: int, model: str, infer: bool
) -> None:
    from himoe_libero_bridge.client import PolicyClient

    with PolicyClient(
        host=host, port=port, connect_timeout=10.0, inference_timeout=180.0
    ) as client:
        metadata = client.metadata
        if metadata.get("suite") != model:
            raise RuntimeError("port %d returned suite %r" % (port, metadata.get("suite")))
        _verify_bundle_identity(metadata, port, gpu, logical_gpu, model)
        if infer:
            response = client.infer(_observation(8))
            actions = np.asarray(response["actions"])
            if actions.shape != (10, 7) or not np.all(np.isfinite(actions)):
                raise RuntimeError("port %d returned invalid LIBERO actions" % port)


def _verify_calvin(
    host: str, port: int, gpu: int, logical_gpu: int, infer: bool
) -> None:
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    uri = "ws://%s:%d" % (host, port)
    with websockets.sync.client.connect(
        uri, compression=None, max_size=None, open_timeout=10.0
    ) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=10.0))
        _verify_bundle_identity(metadata, port, gpu, logical_gpu, "calvin")
        if infer:
            connection.send(msgpack_numpy.Packer().pack(_observation(15)))
            frame = connection.recv(timeout=180.0)
            if isinstance(frame, str):
                raise RuntimeError("port %d CALVIN inference failed:\n%s" % (port, frame))
            response = msgpack_numpy.unpackb(frame)
            actions = np.asarray(response["actions"])
            if actions.shape != (10, 8) or not np.all(np.isfinite(actions)):
                raise RuntimeError("port %d returned invalid CALVIN actions" % port)


def _verify_bundle_identity(
    metadata: dict, port: int, gpu: int, logical_gpu: int, model: str
) -> None:
    expected = {
        "bundle_schema": "himoe-vla-five-model-gpu-bundle/v1",
        "bundle_model": model,
        "bundle_physical_gpu": gpu,
        "bundle_logical_gpu": logical_gpu,
        "bundle_port": port,
    }
    mismatches = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError("port %d bundle metadata mismatch: %r" % (port, mismatches))
    audit = metadata.get("checkpoint_load_audit", {})
    target_device = "cuda:%d" % metadata["bundle_logical_gpu"]
    if audit.get("mode") != "torch_mmap_meta_assign" or audit.get(
        "target_device"
    ) != target_device:
        raise RuntimeError("port %d has invalid load audit" % port)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--matrix-gpus",
        default=None,
        help="visible physical GPU order; defaults to --gpus",
    )
    parser.add_argument("--port-base", type=int, default=8800)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    _install_paths()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    matrix_gpus = [
        int(value) for value in (args.matrix_gpus or args.gpus).split(",") if value
    ]
    logical_by_physical = {gpu: logical for logical, gpu in enumerate(matrix_gpus)}
    missing = sorted(set(gpus) - set(logical_by_physical))
    if missing:
        raise RuntimeError("requested GPUs are absent from matrix GPU order: %r" % missing)
    checked = 0
    for gpu in gpus:
        logical_gpu = logical_by_physical[gpu]
        for offset, model in enumerate(MODELS):
            port = args.port_base + gpu * 10 + offset
            if model == "calvin":
                _verify_calvin(
                    args.host, port, gpu, logical_gpu, not args.metadata_only
                )
            else:
                _verify_libero(
                    args.host,
                    port,
                    gpu,
                    logical_gpu,
                    model,
                    not args.metadata_only,
                )
            checked += 1
            mode = "metadata" if args.metadata_only else "inference"
            print("ok gpu%d %-7s :%d %s" % (gpu, model, port, mode), flush=True)
    print("verified %d endpoints" % checked, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
