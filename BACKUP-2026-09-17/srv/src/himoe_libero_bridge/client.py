"""Synchronous WebSocket policy client compatible with the HiMoE protocol."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from himoe_libero_bridge.protocol import (
    Packer,
    unpackb,
    validate_action_response,
    validate_metadata,
    validate_observation,
)


class PolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        connect_timeout: float = 120.0,
        inference_timeout: float = 120.0,
    ) -> None:
        import websockets.sync.client

        self._uri = "ws://%s:%d" % (host, port)
        self._packer = Packer()
        self._inference_timeout = inference_timeout
        self._connection = None
        deadline = time.monotonic() + connect_timeout
        last_error = None
        logging.info("Waiting for policy server at %s", self._uri)
        while time.monotonic() < deadline:
            try:
                self._connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(10.0, max(1.0, deadline - time.monotonic())),
                )
                metadata_frame = self._connection.recv(timeout=10.0)
                if isinstance(metadata_frame, str):
                    raise RuntimeError("Server returned an error during handshake: %s" % metadata_frame)
                self.metadata = validate_metadata(unpackb(metadata_frame))
                break
            except (OSError, TimeoutError) as error:
                last_error = error
                time.sleep(0.5)
        else:
            raise TimeoutError("Policy server did not become ready at %s: %s" % (self._uri, last_error))

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Policy client is closed")
        canonical = validate_observation(observation)
        self._connection.send(self._packer.pack(canonical))
        response = self._connection.recv(timeout=self._inference_timeout)
        if isinstance(response, str):
            raise RuntimeError("Inference server failed:\n%s" % response)
        return validate_action_response(unpackb(response))

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Optional[bool]:
        self.close()
        return None
