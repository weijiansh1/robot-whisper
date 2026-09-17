"""Wire format and strict boundary validation for policy inference."""

from __future__ import annotations

import functools
from typing import Any, Dict, Mapping

import msgpack
import numpy as np

PROTOCOL_NAME = "himoe-libero-v1"
PROTOCOL_VERSION = 2
IMAGE_SHAPE = (224, 224, 3)
STATE_DIM = 8
ACTION_CHUNK_STEPS = 10
ACTION_DIM = 7
MODEL_ACTION_DIM = 24
FLOW_NOISE_SHAPE = (ACTION_CHUNK_STEPS, MODEL_ACTION_DIM)
FLOW_STEPS = 10
HB_MOE_LAYERS = 8
HB_MOE_EXPERTS = 32
HB_MOE_TOP_K = 4
ROUTING_TRACE_SHAPE = (FLOW_STEPS, HB_MOE_LAYERS, ACTION_CHUNK_STEPS, HB_MOE_TOP_K)
ROUTING_LAYER_INDICES_SHAPE = (HB_MOE_LAYERS,)

IMAGE_KEY = "observation/image"
WRIST_IMAGE_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"
ACTION_KEY = "actions"
FLOW_NOISE_KEY = "flow/noise"
FLOW_NOISE_SHA256_KEY = "flow/noise_sha256"
FLOW_NOISE_ACK_FORMAT = "sha256-f32-c-v1"
ROUTING_CAPTURE_KEY = "routing/capture"
ROUTING_EXPERT_IDS_KEY = "routing/expert_ids"
ROUTING_EXPERT_WEIGHTS_KEY = "routing/expert_weights"
ROUTING_LAYER_INDICES_KEY = "routing/layer_indices"


class ProtocolError(ValueError):
    """Raised when a message violates the bridge contract."""


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ProtocolError("Unsupported NumPy dtype: %s" % obj.dtype)
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: Dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def server_metadata(backend: str) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "backend": backend,
        "observation": {
            IMAGE_KEY: list(IMAGE_SHAPE),
            WRIST_IMAGE_KEY: list(IMAGE_SHAPE),
            STATE_KEY: [STATE_DIM],
            PROMPT_KEY: "non-empty string",
            FLOW_NOISE_KEY: {"shape": list(FLOW_NOISE_SHAPE), "optional": True},
            ROUTING_CAPTURE_KEY: {"type": "bool", "optional": True},
        },
        "action": {ACTION_KEY: [ACTION_CHUNK_STEPS, ACTION_DIM]},
        "flow_noise_ack": {
            "key": FLOW_NOISE_SHA256_KEY,
            "format": FLOW_NOISE_ACK_FORMAT,
        },
        "routing": {
            "optional": True,
            ROUTING_EXPERT_IDS_KEY: list(ROUTING_TRACE_SHAPE),
            ROUTING_EXPERT_WEIGHTS_KEY: list(ROUTING_TRACE_SHAPE),
            ROUTING_LAYER_INDICES_KEY: list(ROUTING_LAYER_INDICES_SHAPE),
            "routed_experts": HB_MOE_EXPERTS,
            "top_k": HB_MOE_TOP_K,
        },
    }


def _require_numeric_array(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ProtocolError("%s must be numeric, got %s" % (key, array.dtype))
    if not np.all(np.isfinite(array)):
        raise ProtocolError("%s contains NaN or infinity" % key)
    return array


def validate_observation(observation: Any) -> Dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise ProtocolError("Observation must be a mapping")

    missing = [key for key in (IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, PROMPT_KEY) if key not in observation]
    if missing:
        raise ProtocolError("Observation is missing keys: %s" % ", ".join(missing))

    canonical = dict(observation)
    for key in (IMAGE_KEY, WRIST_IMAGE_KEY):
        image = np.asarray(observation[key])
        if image.shape != IMAGE_SHAPE:
            raise ProtocolError("%s shape must be %s, got %s" % (key, IMAGE_SHAPE, image.shape))
        if image.dtype != np.uint8:
            raise ProtocolError("%s dtype must be uint8, got %s" % (key, image.dtype))
        canonical[key] = np.ascontiguousarray(image)

    state = _require_numeric_array(observation[STATE_KEY], STATE_KEY)
    if state.shape != (STATE_DIM,):
        raise ProtocolError("%s shape must be (%d,), got %s" % (STATE_KEY, STATE_DIM, state.shape))
    canonical[STATE_KEY] = np.ascontiguousarray(state, dtype=np.float32)

    prompt = observation[PROMPT_KEY]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("prompt must be a non-empty string")
    canonical[PROMPT_KEY] = prompt.strip()

    if FLOW_NOISE_KEY in observation:
        flow_noise = _require_numeric_array(observation[FLOW_NOISE_KEY], FLOW_NOISE_KEY)
        if flow_noise.shape != FLOW_NOISE_SHAPE:
            raise ProtocolError(
                "%s shape must be %s, got %s" % (FLOW_NOISE_KEY, FLOW_NOISE_SHAPE, flow_noise.shape)
            )
        canonical[FLOW_NOISE_KEY] = np.array(flow_noise, dtype=np.float32, copy=True, order="C")
    if ROUTING_CAPTURE_KEY in observation:
        capture = observation[ROUTING_CAPTURE_KEY]
        if not isinstance(capture, (bool, np.bool_)):
            raise ProtocolError("%s must be bool" % ROUTING_CAPTURE_KEY)
        canonical[ROUTING_CAPTURE_KEY] = bool(capture)
    return canonical


def validate_action_response(response: Any) -> Dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ProtocolError("Policy response must be a mapping")
    if ACTION_KEY not in response:
        raise ProtocolError("Policy response is missing actions")

    actions = _require_numeric_array(response[ACTION_KEY], ACTION_KEY)
    expected = (ACTION_CHUNK_STEPS, ACTION_DIM)
    if actions.shape != expected:
        raise ProtocolError("actions shape must be %s, got %s" % (expected, actions.shape))

    canonical = dict(response)
    canonical[ACTION_KEY] = np.ascontiguousarray(actions, dtype=np.float32)
    if FLOW_NOISE_SHA256_KEY in response:
        noise_digest = response[FLOW_NOISE_SHA256_KEY]
        if (
            not isinstance(noise_digest, str)
            or len(noise_digest) != 64
            or any(character not in "0123456789abcdef" for character in noise_digest)
        ):
            raise ProtocolError(
                "%s must be a lowercase SHA-256 hex digest" % FLOW_NOISE_SHA256_KEY
            )
        canonical[FLOW_NOISE_SHA256_KEY] = noise_digest

    routing_keys = (
        ROUTING_EXPERT_IDS_KEY,
        ROUTING_EXPERT_WEIGHTS_KEY,
        ROUTING_LAYER_INDICES_KEY,
    )
    present_routing_keys = [key for key in routing_keys if key in response]
    if present_routing_keys and len(present_routing_keys) != len(routing_keys):
        missing = [key for key in routing_keys if key not in response]
        raise ProtocolError("Routing response is missing keys: %s" % ", ".join(missing))
    if present_routing_keys:
        expert_ids = np.asarray(response[ROUTING_EXPERT_IDS_KEY])
        if expert_ids.shape != ROUTING_TRACE_SHAPE:
            raise ProtocolError(
                "%s shape must be %s, got %s"
                % (ROUTING_EXPERT_IDS_KEY, ROUTING_TRACE_SHAPE, expert_ids.shape)
            )
        if not np.issubdtype(expert_ids.dtype, np.integer):
            raise ProtocolError("%s must be an integer array" % ROUTING_EXPERT_IDS_KEY)
        if np.any(expert_ids < 0) or np.any(expert_ids >= HB_MOE_EXPERTS):
            raise ProtocolError("%s contains an out-of-range expert id" % ROUTING_EXPERT_IDS_KEY)
        if np.any(np.diff(np.sort(expert_ids, axis=-1), axis=-1) == 0):
            raise ProtocolError("%s contains duplicate top-k expert ids" % ROUTING_EXPERT_IDS_KEY)

        expert_weights = _require_numeric_array(
            response[ROUTING_EXPERT_WEIGHTS_KEY], ROUTING_EXPERT_WEIGHTS_KEY
        )
        if expert_weights.shape != ROUTING_TRACE_SHAPE:
            raise ProtocolError(
                "%s shape must be %s, got %s"
                % (ROUTING_EXPERT_WEIGHTS_KEY, ROUTING_TRACE_SHAPE, expert_weights.shape)
            )
        if np.any(expert_weights < 0.0) or np.any(expert_weights > 1.0):
            raise ProtocolError("%s must be in [0, 1]" % ROUTING_EXPERT_WEIGHTS_KEY)
        if not np.allclose(np.sum(expert_weights, axis=-1), 1.0, atol=1e-4, rtol=1e-4):
            raise ProtocolError("%s top-k weights must sum to one" % ROUTING_EXPERT_WEIGHTS_KEY)

        layer_indices = np.asarray(response[ROUTING_LAYER_INDICES_KEY])
        if layer_indices.shape != ROUTING_LAYER_INDICES_SHAPE:
            raise ProtocolError(
                "%s shape must be %s, got %s"
                % (ROUTING_LAYER_INDICES_KEY, ROUTING_LAYER_INDICES_SHAPE, layer_indices.shape)
            )
        if not np.issubdtype(layer_indices.dtype, np.integer):
            raise ProtocolError("%s must be an integer array" % ROUTING_LAYER_INDICES_KEY)
        if np.any(np.diff(layer_indices) <= 0):
            raise ProtocolError("%s must be strictly increasing" % ROUTING_LAYER_INDICES_KEY)

        canonical[ROUTING_EXPERT_IDS_KEY] = np.ascontiguousarray(expert_ids, dtype=np.int16)
        canonical[ROUTING_EXPERT_WEIGHTS_KEY] = np.ascontiguousarray(
            expert_weights, dtype=np.float32
        )
        canonical[ROUTING_LAYER_INDICES_KEY] = np.ascontiguousarray(
            layer_indices, dtype=np.int16
        )
    return canonical


def validate_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ProtocolError("Server metadata must be a mapping")
    if metadata.get("protocol") != PROTOCOL_NAME:
        raise ProtocolError("Unexpected server protocol: %r" % metadata.get("protocol"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("Unexpected protocol version: %r" % metadata.get("protocol_version"))
    observation = metadata.get("observation", {})
    if list(observation.get(IMAGE_KEY) or []) != list(IMAGE_SHAPE):
        raise ProtocolError("Server advertises an invalid agent image shape")
    if list(observation.get(WRIST_IMAGE_KEY) or []) != list(IMAGE_SHAPE):
        raise ProtocolError("Server advertises an invalid wrist image shape")
    if list(observation.get(STATE_KEY) or []) != [STATE_DIM]:
        raise ProtocolError("Server advertises an invalid state shape")
    noise_contract = observation.get(FLOW_NOISE_KEY, {})
    if (
        list(noise_contract.get("shape") or []) != list(FLOW_NOISE_SHAPE)
        or noise_contract.get("optional") is not True
    ):
        raise ProtocolError("Server advertises an invalid flow-noise contract")
    action_shape = metadata.get("action", {}).get(ACTION_KEY)
    if list(action_shape or []) != [ACTION_CHUNK_STEPS, ACTION_DIM]:
        raise ProtocolError("Server advertises an invalid action shape: %r" % action_shape)
    noise_ack = metadata.get("flow_noise_ack", {})
    if (
        noise_ack.get("key") != FLOW_NOISE_SHA256_KEY
        or noise_ack.get("format") != FLOW_NOISE_ACK_FORMAT
    ):
        raise ProtocolError("Server does not advertise the required flow-noise acknowledgement")
    return dict(metadata)
