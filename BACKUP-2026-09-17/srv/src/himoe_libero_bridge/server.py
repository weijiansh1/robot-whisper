"""Strict WebSocket inference server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from typing import Any, Optional

from himoe_libero_bridge.policies import HiMoEPolicy, MockPolicy
from himoe_libero_bridge.protocol import Packer, server_metadata, unpackb, validate_action_response, validate_observation


class PolicyServer:
    def __init__(self, policy: Any, host: str, port: int, backend: str) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = server_metadata(backend)
        self._metadata.update(getattr(policy, "metadata", {}))
        self._metadata.update(
            {
                "server_instance_id": uuid.uuid4().hex,
                "server_pid": os.getpid(),
                "server_started_unix_ns": time.time_ns(),
            }
        )

    async def _handler(self, websocket: Any, *unused: Any) -> None:
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        logging.info("Client connected: %s", getattr(websocket, "remote_address", "unknown"))
        while True:
            try:
                request = await websocket.recv()
                if isinstance(request, str):
                    raise ValueError("Inference request must be a binary msgpack frame")
                observation = validate_observation(unpackb(request))
                start = time.perf_counter()
                response = validate_action_response(self._policy.infer(observation))
                response["server/inference_ms"] = np_float32((time.perf_counter() - start) * 1000.0)
                await websocket.send(packer.pack(response))
            except Exception as error:
                if error.__class__.__name__ in ("ConnectionClosed", "ConnectionClosedOK"):
                    logging.info("Client disconnected")
                    return
                logging.exception("Inference request failed")
                payload = {
                    "error": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                try:
                    await websocket.send(json.dumps(payload))
                    await websocket.close(code=1011, reason="Inference failed")
                finally:
                    return

    async def run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            from websockets.server import serve

        async with serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info("Policy server ready at ws://%s:%d", self._host, self._port)
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())


def np_float32(value: float) -> Any:
    import numpy as np

    return np.float32(value)


def create_policy(
    backend: str,
    checkpoint_dir: Optional[str],
    upstream_root: Optional[str],
    gpu: str,
    suite: str = "goal",
    libero_wrist_layout: str = "released-left",
) -> Any:
    if backend == "mock":
        return MockPolicy(suite=suite)
    if backend != "himoe":
        raise ValueError("Unknown backend: %s" % backend)
    if not checkpoint_dir:
        raise ValueError("--checkpoint-dir is required for the himoe backend")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return HiMoEPolicy(
        checkpoint_dir=checkpoint_dir,
        suite=suite,
        upstream_root=upstream_root,
        require_cuda=True,
        libero_wrist_layout=libero_wrist_layout,
    )
