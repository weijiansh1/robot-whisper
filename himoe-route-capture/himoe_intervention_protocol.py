"""Request keys, hashing, and online pairing audits for HB5 interventions."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from himoe_hb5_intervention import ARMS


PAIR_KEY = "intervention/pair_id"
DRAW_KEY = "intervention/draw_id"
ARM_KEY = "intervention/arm"
QUERY_KEY = "flow/query_id"
CANDIDATE_KEY = "flow/candidate_id"
FLOW_NOISE_KEY = "flow/noise"
OBSERVATION_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/state",
    "prompt",
)


def _hash_field(digest, key: str, value: Any) -> None:
    key_bytes = key.encode("utf-8")
    digest.update(struct.pack("<I", len(key_bytes)))
    digest.update(key_bytes)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"S" + struct.pack("<Q", len(payload)) + payload)
        return
    array = np.ascontiguousarray(value)
    dtype = array.dtype.str.encode("ascii")
    digest.update(b"A" + struct.pack("<I", len(dtype)) + dtype)
    digest.update(struct.pack("<I", array.ndim))
    digest.update(struct.pack("<" + "q" * array.ndim, *array.shape))
    digest.update(array.tobytes())


def request_digests(observation: dict[str, Any]) -> tuple[bytes, bytes]:
    """Hash the physical observation separately from the explicit flow noise."""
    missing = [
        key for key in (*OBSERVATION_KEYS, FLOW_NOISE_KEY) if key not in observation
    ]
    if missing:
        raise ValueError("paired intervention request is missing %s" % missing)
    observation_digest = hashlib.sha256()
    for key in OBSERVATION_KEYS:
        _hash_field(observation_digest, key, observation[key])
    noise = np.ascontiguousarray(observation[FLOW_NOISE_KEY], dtype=np.float32)
    if noise.shape != (10, 24):
        raise ValueError("paired intervention requires explicit flow/noise [10,24]")
    return observation_digest.digest(), hashlib.sha256(noise.tobytes()).digest()


def pop_intervention_identity(
    observation: dict[str, Any],
) -> tuple[int, int, str, int, int]:
    """Remove server-only keys before the observation reaches the policy."""
    try:
        pair_id = int(observation.pop(PAIR_KEY))
        draw_id = int(observation.pop(DRAW_KEY))
        arm = str(observation.pop(ARM_KEY))
    except KeyError as error:
        raise ValueError(
            "paired intervention request lacks %s" % error.args[0]
        ) from error
    if pair_id < 0 or draw_id < 0:
        raise ValueError("pair_id and draw_id must be non-negative")
    if arm not in ARMS:
        raise ValueError("arm must be one of %s" % (ARMS,))
    query_id = int(observation.pop(QUERY_KEY, pair_id))
    candidate_id = int(observation.pop(CANDIDATE_KEY, draw_id))
    return pair_id, draw_id, arm, query_id, candidate_id


class OnlinePairAudit:
    """Reject a triad if any pre-intervention value or x0 differs by one bit."""

    _ORIGINAL_FIELDS = (
        "hb_probe_input_hidden",
        "hb_probe_selected_expert_raw",
        "hb_probe_original_routed",
        "hb_probe_dropped_slot",
        "hb_probe_dropped_expert_id",
    )

    def __init__(self) -> None:
        self._pairs: dict[tuple[int, int], dict[str, Any]] = {}
        self.complete_pairs = 0
        self.max_original_pair_difference = 0.0
        self.max_x0_pair_difference = 0.0

    def prepare(
        self,
        pair_id: int,
        draw_id: int,
        arm: str,
        observation_digest: bytes,
        noise_digest: bytes,
    ) -> None:
        key = (int(pair_id), int(draw_id))
        state = self._pairs.get(key)
        if state is None:
            return
        if arm in state["seen"]:
            raise ValueError("pair %s repeats arm %s" % (key, arm))
        if state["observation_digest"] != observation_digest:
            raise ValueError("pair %s changed the physical observation" % (key,))
        if state["noise_digest"] != noise_digest:
            raise ValueError("pair %s changed its explicit flow noise" % (key,))

    def commit(
        self,
        record,
        x_traj: np.ndarray,
        observation_digest: bytes,
        noise_digest: bytes,
    ) -> None:
        key = (int(record.intervention_pair_id), int(record.intervention_draw_id))
        arm = str(record.intervention_arm)
        trajectory = np.asarray(x_traj)
        if trajectory.ndim != 4:
            raise ValueError("x_traj must be batch-first")
        layer_position = np.flatnonzero(np.asarray(record.hb_layers) == 5)
        if layer_position.size != 1:
            raise ValueError("paired audit record does not contain exactly one HB5")
        layer = int(layer_position[0])
        originals = {
            name: np.asarray(getattr(record, name)).copy()
            for name in self._ORIGINAL_FIELDS
        }
        originals["hb5_d0_selected_expert_id"] = np.asarray(
            record.hb_selected_expert_id[:, layer, 0]
        ).copy()
        originals["hb5_d0_selected_expert_weight"] = np.asarray(
            record.hb_selected_expert_weight[:, layer, 0]
        ).copy()
        x0 = trajectory[:, 0].copy()
        state = self._pairs.get(key)
        if state is None:
            state = {
                "seen": set(),
                "observation_digest": observation_digest,
                "noise_digest": noise_digest,
                "originals": originals,
                "x0": x0,
            }
            self._pairs[key] = state
        else:
            self.prepare(key[0], key[1], arm, observation_digest, noise_digest)
            for name, value in originals.items():
                reference = state["originals"][name]
                if not np.array_equal(value, reference):
                    if np.issubdtype(value.dtype, np.number):
                        self.max_original_pair_difference = max(
                            self.max_original_pair_difference,
                            float(
                                np.max(
                                    np.abs(
                                        value.astype(np.float64)
                                        - reference.astype(np.float64)
                                    )
                                )
                            ),
                        )
                    raise ValueError("pair %s changed original field %s" % (key, name))
            self.max_x0_pair_difference = max(
                self.max_x0_pair_difference,
                float(np.max(np.abs(x0.astype(np.float64) - state["x0"]))),
            )
            if not np.array_equal(x0, state["x0"]):
                raise ValueError("pair %s did not reuse the exact same x0" % (key,))
        if arm == "baseline":
            if not np.array_equal(
                record.hb_probe_original_routed,
                record.hb_probe_executed_routed,
            ) or np.count_nonzero(record.hb_probe_intervention_delta):
                raise ValueError("baseline instrumentation is not an exact no-op")
        state["seen"].add(arm)
        if state["seen"] == set(ARMS):
            self.complete_pairs += 1
            del self._pairs[key]

    def summary(self) -> dict[str, float | int]:
        return {
            "complete_pairs": self.complete_pairs,
            "incomplete_pairs": len(self._pairs),
            "max_original_pair_difference": self.max_original_pair_difference,
            "max_x0_pair_difference": self.max_x0_pair_difference,
        }

    def require_complete(self) -> None:
        if self._pairs:
            incomplete = {
                key: sorted(value["seen"]) for key, value in self._pairs.items()
            }
            raise RuntimeError("incomplete paired intervention triads: %s" % incomplete)
